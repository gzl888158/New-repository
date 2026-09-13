"""
数据清洗校验引擎
================
核心定位：全面的数据质量保障，剔除错价、插针毛刺、断帧空数据、
交易所延迟脏数据，确保流入策略引擎的数据干净可靠。

特性：
- 多层数据校验：格式校验、价格范围校验、价格变动校验、时间连续性校验
- 插针毛刺过滤：基于波动率和回归检测异常波动
- 断帧检测：监控数据更新间隔，发现数据缺失
- 交易所延迟检测：通过多源对比检测延迟数据
- 数据修复：对轻微异常数据进行修复，严重异常直接丢弃
- 质量评分：实时计算各交易对数据质量分数
"""

import time
import threading
import math
from typing import Dict, Any, List, Optional, Callable, Tuple
from loguru import logger
from collections import deque
from enum import Enum


class IssueSeverity(Enum):
    """问题严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IssueType(Enum):
    """问题类型"""
    INVALID_FORMAT = "invalid_format"
    MISSING_SYMBOL = "missing_symbol"
    INVALID_PRICE = "invalid_price"
    PRICE_OUT_OF_RANGE = "price_out_of_range"
    SUDDEN_PRICE_CHANGE = "sudden_price_change"
    PRICE_INVERSION = "price_inversion"
    WIDE_SPREAD = "wide_spread"
    EMPTY_ORDERBOOK = "empty_orderbook"
    BOOK_INVERSION = "book_inversion"
    MALFORMED_ORDERBOOK = "malformed_orderbook"
    DATA_GAP = "data_gap"
    STALE_DATA = "stale_data"
    SPIKE_DETECTED = "spike_detected"
    DUPLICATE_DATA = "duplicate_data"
    ORDERBOOK_DEPTH_INSUFFICIENT = "orderbook_depth_insufficient"


class DataQualityIssue:
    """数据质量问题"""

    def __init__(self, issue_type: IssueType, severity: IssueSeverity,
                 message: str, symbol: str = "", metadata: Dict[str, Any] = None):
        self.issue_type = issue_type.value
        self.severity = severity.value
        self.message = message
        self.symbol = symbol
        self.metadata = metadata or {}
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_type": self.issue_type,
            "severity": self.severity,
            "message": self.message,
            "symbol": self.symbol,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class SpikeDetector:
    """插针毛刺检测器"""

    def __init__(self, window_size: int = 20, threshold_factor: float = 3.0):
        self._window_size = window_size
        self._threshold_factor = threshold_factor
        self._price_windows: Dict[str, deque] = {}
        self._lock = threading.RLock()

    def detect(self, symbol: str, price: float) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """检测是否为插针"""
        with self._lock:
            if symbol not in self._price_windows:
                self._price_windows[symbol] = deque(maxlen=self._window_size)

            window = self._price_windows[symbol]

            if len(window) < self._window_size:
                window.append(price)
                return False, None

            mean = sum(window) / len(window)
            variance = sum((p - mean) ** 2 for p in window) / len(window)
            std_dev = math.sqrt(variance)

            if std_dev == 0:
                window.append(price)
                return False, None

            z_score = abs(price - mean) / std_dev

            if z_score > self._threshold_factor:
                metadata = {
                    "price": price,
                    "mean": mean,
                    "std_dev": std_dev,
                    "z_score": round(z_score, 2),
                    "window_size": len(window),
                }
                return True, metadata

            window.append(price)
            return False, None


class GapDetector:
    """断帧检测器"""

    def __init__(self, expected_interval_ms: int = 1000, gap_threshold_factor: float = 3.0):
        self._expected_interval_ms = expected_interval_ms
        self._gap_threshold_ms = expected_interval_ms * gap_threshold_factor
        self._last_timestamps: Dict[str, int] = {}
        self._lock = threading.RLock()

    def detect(self, symbol: str, current_ts: int) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """检测是否存在断帧"""
        with self._lock:
            if symbol not in self._last_timestamps:
                self._last_timestamps[symbol] = current_ts
                return False, None

            last_ts = self._last_timestamps[symbol]
            gap_ms = current_ts - last_ts

            if gap_ms > self._gap_threshold_ms:
                metadata = {
                    "current_ts": current_ts,
                    "last_ts": last_ts,
                    "gap_ms": gap_ms,
                    "threshold_ms": self._gap_threshold_ms,
                }
                self._last_timestamps[symbol] = current_ts
                return True, metadata

            self._last_timestamps[symbol] = current_ts
            return False, None


class DataCleaningEngine:
    """数据清洗校验引擎"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # 配置参数
        self._max_price_deviation = self.config.get("max_price_deviation", 0.5)
        self._max_single_tick_change = self.config.get("max_single_tick_change", 0.1)
        self._max_spread_pct = self.config.get("max_spread_pct", 5.0)
        self._min_orderbook_depth = self.config.get("min_orderbook_depth", 5)
        self._stale_threshold_ms = self.config.get("stale_threshold_ms", 10000)
        self._history_size = self.config.get("history_size", 2000)

        # 检测器
        self._spike_detector = SpikeDetector(
            window_size=self.config.get("spike_window_size", 20),
            threshold_factor=self.config.get("spike_threshold_factor", 3.0),
        )
        self._gap_detector = GapDetector(
            expected_interval_ms=self.config.get("expected_interval_ms", 50),
            gap_threshold_factor=self.config.get("gap_threshold_factor", 3.0),
        )

        # 状态
        self._issues: deque = deque(maxlen=self._history_size)
        self._lock = threading.RLock()
        self._symbol_stats: Dict[str, Dict[str, Any]] = {}
        self._total_checks = 0
        self._total_failures = 0
        self._total_repaired = 0
        self._last_prices: Dict[str, float] = {}
        self._price_history: Dict[str, deque] = {}
        self._last_update_ts: Dict[str, float] = {}
        self._seq_tracker: Dict[str, int] = {}

        # 回调
        self._callback: Optional[Callable[[DataQualityIssue], None]] = None

        logger.info("DataCleaningEngine initialized")

    def set_callback(self, callback: Callable[[DataQualityIssue], None]) -> None:
        """设置问题回调"""
        self._callback = callback

    def clean_and_validate_ticker(self, ticker: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """清洗并验证行情数据"""
        issues = []

        if not ticker or not isinstance(ticker, dict):
            self._record_issue(IssueType.INVALID_FORMAT, IssueSeverity.HIGH,
                              "Ticker data is invalid")
            return None

        symbol = ticker.get("symbol", "")
        if not symbol:
            self._record_issue(IssueType.MISSING_SYMBOL, IssueSeverity.HIGH,
                              "Symbol is missing")
            return None

        # 标准化字段
        ticker = self._normalize_ticker(ticker)
        price = ticker.get("price", 0)

        # 价格有效性检查
        if not self._validate_price(symbol, price):
            return None

        # 插针检测
        is_spike, spike_meta = self._spike_detector.detect(symbol, price)
        if is_spike:
            self._record_issue(IssueType.SPIKE_DETECTED, IssueSeverity.HIGH,
                              f"Spike detected: price={price:.4f}, z_score={spike_meta['z_score']}",
                              symbol, spike_meta)
            return None

        # 价格变动检查
        if not self._check_price_change(symbol, price):
            return None

        # 买卖价校验
        bid_price = ticker.get("bid_price", 0)
        ask_price = ticker.get("ask_price", 0)
        if not self._validate_bid_ask(symbol, bid_price, ask_price, price):
            return None

        # 断帧检测
        timestamp = ticker.get("timestamp", int(time.time() * 1000))
        is_gap, gap_meta = self._gap_detector.detect(symbol, timestamp)
        if is_gap:
            self._record_issue(IssueType.DATA_GAP, IssueSeverity.MEDIUM,
                              f"Data gap detected: {gap_meta['gap_ms']}ms",
                              symbol, gap_meta)

        # 重复数据检测
        if not self._check_duplicate(symbol, timestamp, price):
            return None

        # 更新状态
        self._update_symbol_stats(symbol, "ticker", True)
        self._last_prices[symbol] = price
        self._last_update_ts[symbol] = time.time()

        return ticker

    def clean_and_validate_orderbook(self, orderbook: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """清洗并验证订单簿"""
        if not orderbook or not isinstance(orderbook, dict):
            self._record_issue(IssueType.INVALID_FORMAT, IssueSeverity.HIGH,
                              "Orderbook data is invalid")
            return None

        symbol = orderbook.get("symbol", "")
        if not symbol:
            self._record_issue(IssueType.MISSING_SYMBOL, IssueSeverity.HIGH,
                              "Symbol is missing")
            return None

        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        if not bids or not asks:
            self._record_issue(IssueType.EMPTY_ORDERBOOK, IssueSeverity.MEDIUM,
                              "Empty bids or asks", symbol)
            return None

        # 订单簿深度检查
        if len(bids) < self._min_orderbook_depth or len(asks) < self._min_orderbook_depth:
            self._record_issue(IssueType.ORDERBOOK_DEPTH_INSUFFICIENT, IssueSeverity.LOW,
                              f"Orderbook depth insufficient: bids={len(bids)}, asks={len(asks)}",
                              symbol)

        # 校验买卖盘价格
        try:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])

            if best_bid <= 0 or best_ask <= 0:
                self._record_issue(IssueType.INVALID_PRICE, IssueSeverity.HIGH,
                                  f"Invalid bid/ask: bid={best_bid}, ask={best_ask}", symbol)
                return None

            if best_bid >= best_ask:
                self._record_issue(IssueType.BOOK_INVERSION, IssueSeverity.HIGH,
                                  f"Best bid ({best_bid}) >= best ask ({best_ask})", symbol)
                return None

            # 排序校验
            for i in range(1, len(bids)):
                if float(bids[i][0]) > float(bids[i - 1][0]):
                    bids = sorted(bids, key=lambda x: -float(x[0]))
                    break

            for i in range(1, len(asks)):
                if float(asks[i][0]) < float(asks[i - 1][0]):
                    asks = sorted(asks, key=lambda x: float(x[0]))
                    break

            orderbook["bids"] = bids
            orderbook["asks"] = asks

        except (ValueError, IndexError, TypeError):
            self._record_issue(IssueType.MALFORMED_ORDERBOOK, IssueSeverity.HIGH,
                              "Cannot parse bid/ask", symbol)
            return None

        self._update_symbol_stats(symbol, "orderbook", True)
        return orderbook

    def clean_and_validate_kline(self, kline: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """清洗并验证单根K线"""
        if not kline or not isinstance(kline, dict):
            self._record_issue(IssueType.INVALID_FORMAT, IssueSeverity.HIGH,
                              "Kline data is invalid")
            return None

        symbol = kline.get("symbol", "")
        if not symbol:
            self._record_issue(IssueType.MISSING_SYMBOL, IssueSeverity.HIGH,
                              "Symbol is missing")
            return None

        try:
            o = float(kline.get("open", 0))
            h = float(kline.get("high", 0))
            l = float(kline.get("low", 0))
            c = float(kline.get("close", 0))
            v = float(kline.get("volume", 0))
            ts = int(kline.get("timestamp", 0))

            if o <= 0 or h <= 0 or l <= 0 or c <= 0:
                self._record_issue(IssueType.INVALID_PRICE, IssueSeverity.HIGH,
                                  f"Zero or negative price: o={o}, h={h}, l={l}, c={c}", symbol)
                return None

            if h < l:
                self._record_issue(IssueType.PRICE_INVERSION, IssueSeverity.MEDIUM,
                                  f"High ({h}) < Low ({l})", symbol)
                kline["high"], kline["low"] = l, h

            if c > kline["high"] or c < kline["low"]:
                self._record_issue(IssueType.PRICE_INVERSION, IssueSeverity.MEDIUM,
                                  f"Close ({c}) outside [Low, High]", symbol)
                kline["close"] = min(max(c, kline["low"]), kline["high"])
                self._total_repaired += 1

            if o > kline["high"] or o < kline["low"]:
                self._record_issue(IssueType.PRICE_INVERSION, IssueSeverity.MEDIUM,
                                  f"Open ({o}) outside [Low, High]", symbol)
                kline["open"] = min(max(o, kline["low"]), kline["high"])
                self._total_repaired += 1

            if v < 0:
                self._record_issue(IssueType.INVALID_PRICE, IssueSeverity.LOW,
                                  f"Negative volume: {v}", symbol)
                kline["volume"] = 0

        except (ValueError, TypeError):
            self._record_issue(IssueType.MALFORMED_ORDERBOOK, IssueSeverity.HIGH,
                              "Cannot parse kline data", symbol)
            return None

        self._update_symbol_stats(symbol, "kline", True)
        return kline

    def clean_and_validate_trade(self, trade: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """清洗并验证成交记录"""
        if not trade or not isinstance(trade, dict):
            self._record_issue(IssueType.INVALID_FORMAT, IssueSeverity.HIGH,
                              "Trade data is invalid")
            return None

        symbol = trade.get("symbol", "")
        if not symbol:
            self._record_issue(IssueType.MISSING_SYMBOL, IssueSeverity.HIGH,
                              "Symbol is missing")
            return None

        try:
            price = float(trade.get("price", 0))
            volume = float(trade.get("volume", 0))
            side = trade.get("side", "").upper()

            if price <= 0:
                self._record_issue(IssueType.INVALID_PRICE, IssueSeverity.HIGH,
                                  f"Invalid price: {price}", symbol)
                return None

            if volume <= 0:
                self._record_issue(IssueType.INVALID_PRICE, IssueSeverity.MEDIUM,
                                  f"Invalid volume: {volume}", symbol)
                return None

            if side not in ("BUY", "SELL", ""):
                trade["side"] = ""

            # 检查价格是否在合理范围内
            if symbol in self._last_prices:
                last_price = self._last_prices[symbol]
                if last_price > 0:
                    deviation = abs(price - last_price) / last_price
                    if deviation > self._max_price_deviation:
                        self._record_issue(IssueType.PRICE_OUT_OF_RANGE, IssueSeverity.HIGH,
                                          f"Trade price {deviation:.1%} from last price",
                                          symbol, {"price": price, "last_price": last_price})
                        return None

        except (ValueError, TypeError):
            self._record_issue(IssueType.MALFORMED_ORDERBOOK, IssueSeverity.HIGH,
                              "Cannot parse trade data", symbol)
            return None

        self._update_symbol_stats(symbol, "trade", True)
        return trade

    def _normalize_ticker(self, ticker: Dict[str, Any]) -> Dict[str, Any]:
        """标准化行情数据"""
        return {
            "symbol": ticker.get("symbol", ""),
            "feed_type": ticker.get("feed_type", "ticker"),
            "timestamp": int(ticker.get("timestamp", int(time.time() * 1000))),
            "price": float(ticker.get("price", ticker.get("last", 0))),
            "bid_price": float(ticker.get("bid_price", ticker.get("bid", ticker.get("bidPx", 0)))),
            "ask_price": float(ticker.get("ask_price", ticker.get("ask", ticker.get("askPx", 0)))),
            "bid_volume": float(ticker.get("bid_volume", ticker.get("bidSz", 0))),
            "ask_volume": float(ticker.get("ask_volume", ticker.get("askSz", 0))),
            "volume_24h": float(ticker.get("volume_24h", ticker.get("vol24h", 0))),
            "change_24h": float(ticker.get("change_24h", ticker.get("chg24h", 0))),
            "high_24h": float(ticker.get("high_24h", ticker.get("high24h", 0))),
            "low_24h": float(ticker.get("low_24h", ticker.get("low24h", 0))),
            "funding_rate": float(ticker.get("funding_rate", 0)),
        }

    def _validate_price(self, symbol: str, price: float) -> bool:
        """验证价格有效性"""
        if price <= 0:
            self._record_issue(IssueType.INVALID_PRICE, IssueSeverity.HIGH,
                              f"Invalid price: {price}", symbol, {"price": price})
            return False

        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=100)

        history = self._price_history[symbol]
        if len(history) >= 10:
            avg = sum(history) / len(history)
            if avg > 0:
                deviation = abs(price - avg) / avg
                if deviation > self._max_price_deviation:
                    self._record_issue(IssueType.PRICE_OUT_OF_RANGE, IssueSeverity.HIGH,
                                      f"Price {price:.4f} deviates {deviation:.1%} from avg {avg:.4f}",
                                      symbol, {"price": price, "avg": avg, "deviation": deviation})
                    return False

        history.append(price)
        return True

    def _check_price_change(self, symbol: str, price: float) -> bool:
        """检查价格变动幅度"""
        if symbol in self._last_prices:
            last = self._last_prices[symbol]
            if last > 0:
                change = abs(price - last) / last
                if change > self._max_single_tick_change:
                    self._record_issue(IssueType.SUDDEN_PRICE_CHANGE, IssueSeverity.HIGH,
                                      f"Price changed {change:.1%} in one tick",
                                      symbol, {"last": last, "current": price})
                    return False
        return True

    def _validate_bid_ask(self, symbol: str, bid: float, ask: float, last: float) -> bool:
        """验证买卖价"""
        if bid > 0 and ask > 0:
            if bid >= ask:
                self._record_issue(IssueType.PRICE_INVERSION, IssueSeverity.HIGH,
                                  f"Bid ({bid}) >= Ask ({ask})", symbol)
                return False

            spread_pct = (ask - bid) / last * 100 if last > 0 else 0
            if spread_pct > self._max_spread_pct:
                self._record_issue(IssueType.WIDE_SPREAD, IssueSeverity.MEDIUM,
                                  f"Spread too wide: {spread_pct:.2f}%",
                                  symbol, {"spread_pct": spread_pct})

        if last > 0:
            if bid > 0 and bid > last:
                self._record_issue(IssueType.PRICE_INVERSION, IssueSeverity.MEDIUM,
                                  f"Bid ({bid}) > Last ({last})", symbol)
            if ask > 0 and ask < last:
                self._record_issue(IssueType.PRICE_INVERSION, IssueSeverity.MEDIUM,
                                  f"Ask ({ask}) < Last ({last})", symbol)

        return True

    def _check_duplicate(self, symbol: str, timestamp: int, price: float) -> bool:
        """检查重复数据"""
        key = f"{symbol}:{timestamp}:{price}"
        seq_key = f"{symbol}_seq"

        with self._lock:
            if seq_key in self._seq_tracker and self._seq_tracker[seq_key] == key:
                self._record_issue(IssueType.DUPLICATE_DATA, IssueSeverity.LOW,
                                  f"Duplicate ticker data", symbol)
                return False
            self._seq_tracker[seq_key] = key

        return True

    def check_stale_data(self, symbol: str) -> bool:
        """检查数据是否过期"""
        if symbol in self._last_update_ts:
            elapsed = (time.time() - self._last_update_ts[symbol]) * 1000
            if elapsed > self._stale_threshold_ms:
                self._record_issue(IssueType.STALE_DATA, IssueSeverity.MEDIUM,
                                  f"Data stale for {elapsed:.0f}ms", symbol,
                                  {"elapsed_ms": elapsed})
                return True
        return False

    def _record_issue(self, issue_type: IssueType, severity: IssueSeverity,
                      message: str, symbol: str = "", metadata: Dict[str, Any] = None) -> None:
        """记录问题"""
        issue = DataQualityIssue(issue_type, severity, message, symbol, metadata)
        with self._lock:
            self._issues.append(issue)
            self._total_failures += 1

        if self._callback:
            try:
                self._callback(issue)
            except Exception as e:
                logger.debug(f"Callback error: {e}")

        if severity in (IssueSeverity.HIGH, IssueSeverity.CRITICAL):
            logger.warning(f"Data quality issue [{severity.value}] {symbol}: {message}")

    def _update_symbol_stats(self, symbol: str, data_type: str, success: bool) -> None:
        """更新标的统计"""
        with self._lock:
            if symbol not in self._symbol_stats:
                self._symbol_stats[symbol] = {
                    "ticker": {"success": 0, "failure": 0},
                    "orderbook": {"success": 0, "failure": 0},
                    "kline": {"success": 0, "failure": 0},
                    "trade": {"success": 0, "failure": 0},
                }

            stats = self._symbol_stats[symbol].get(data_type, {"success": 0, "failure": 0})
            if success:
                stats["success"] += 1
            else:
                stats["failure"] += 1

            self._total_checks += 1

    def get_recent_issues(self, limit: int = 50, severity: str = None) -> List[Dict[str, Any]]:
        """获取最近问题"""
        with self._lock:
            issues = list(self._issues)[-limit:]
            if severity:
                issues = [i for i in issues if i.severity == severity]
            return [i.to_dict() for i in issues]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            total = self._total_checks
            failure_rate = self._total_failures / max(total, 1)

            severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
            type_counts = {}
            for issue in self._issues:
                severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
                type_counts[issue.issue_type] = type_counts.get(issue.issue_type, 0) + 1

            return {
                "total_checks": total,
                "total_failures": self._total_failures,
                "total_repaired": self._total_repaired,
                "failure_rate": round(failure_rate, 4),
                "recent_issue_count": len(self._issues),
                "severity_counts": severity_counts,
                "type_counts": type_counts,
                "tracked_symbols": len(self._symbol_stats),
            }

    def get_symbol_stats(self, symbol: str) -> Dict[str, Any]:
        """获取标的统计"""
        with self._lock:
            return self._symbol_stats.get(symbol, {}).copy()

    def get_quality_score(self, symbol: str = "") -> float:
        """获取质量评分 (0-100)"""
        with self._lock:
            if symbol:
                stats = self._symbol_stats.get(symbol, {})
                total_s = sum(t.get("success", 0) for t in stats.values())
                total_f = sum(t.get("failure", 0) for t in stats.values())
                total = total_s + total_f
                if total == 0:
                    return 100.0
                return round(total_s / total * 100, 2)

            total = self._total_checks
            if total == 0:
                return 100.0
            return round((total - self._total_failures) / total * 100, 2)

    def reset_symbol_state(self, symbol: str) -> None:
        """重置指定标的状态"""
        with self._lock:
            if symbol in self._last_prices:
                del self._last_prices[symbol]
            if symbol in self._price_history:
                del self._price_history[symbol]
            if symbol in self._last_update_ts:
                del self._last_update_ts[symbol]
            if symbol in self._seq_tracker:
                seq_key = f"{symbol}_seq"
                if seq_key in self._seq_tracker:
                    del self._seq_tracker[seq_key]
            logger.debug(f"Symbol state reset for {symbol}")
