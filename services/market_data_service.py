"""
行情数据服务，订阅市场行情、回调分发并对数据质量进行校验与监控。
"""
import asyncio
import math
import time
from datetime import datetime
from typing import Dict, Any, List, Callable, Optional
from loguru import logger

from core.models import TickData


class DataQualityChecker:
    """数据质量校验器：检测价格异常、时间戳延迟、数据缺失"""

    def __init__(self, config: Dict[str, Any]):
        self._max_price_change_pct = config.get("market_data", {}).get("max_price_change_pct", 0.05)
        self._max_data_delay_seconds = config.get("market_data", {}).get("max_data_delay_seconds", 30)
        self._max_kline_gap_intervals = max(
            1.0,
            float(config.get("market_data", {}).get("max_kline_gap_intervals", 1.5)),
        )
        self._last_prices: Dict[str, float] = {}
        self._last_update_times: Dict[str, float] = {}
        self._violation_counts: Dict[str, int] = {}

    def check_tick(self, tick: TickData) -> Dict[str, Any]:
        """检查单个tick数据质量，返回质量报告"""
        symbol = tick.symbol
        now = time.time()
        issues = []
        quality_score = 1.0

        numeric_fields = {
            "price": tick.price,
            "volume": tick.volume,
            "bid_price": tick.bid_price,
            "bid_volume": tick.bid_volume,
            "ask_price": tick.ask_price,
            "ask_volume": tick.ask_volume,
        }
        invalid_numeric_fields = [
            name for name, value in numeric_fields.items()
            if not self._is_finite_number(value)
        ]
        if invalid_numeric_fields:
            issues.append(f"non_finite:{','.join(invalid_numeric_fields)}")
            quality_score = 0.0

        # 基本字段校验
        if self._is_finite_number(tick.price) and tick.price <= 0:
            issues.append("invalid_price")
            quality_score = 0.0

        if (
            self._is_finite_number(tick.bid_price)
            and self._is_finite_number(tick.ask_price)
            and (tick.bid_price <= 0 or tick.ask_price <= 0)
        ):
            issues.append("invalid_quote")
            quality_score = max(quality_score, 0.3)

        if (
            self._is_finite_number(tick.bid_price)
            and self._is_finite_number(tick.ask_price)
            and tick.bid_price > tick.ask_price
        ):
            issues.append("inverted_quote")
            quality_score = max(quality_score, 0.5)

        # 价格突变检测
        last_price = self._last_prices.get(symbol)
        if (
            last_price and last_price > 0
            and self._is_finite_number(tick.price)
            and tick.price > 0
        ):
            change_pct = abs(tick.price - last_price) / last_price
            if change_pct > self._max_price_change_pct:
                issues.append(f"price_spike:{change_pct:.4f}")
                quality_score = max(quality_score, 0.2)
                self._violation_counts[symbol] = self._violation_counts.get(symbol, 0) + 1

        # 时间戳延迟检测
        try:
            tick_ts = tick.timestamp.timestamp()
            if not math.isfinite(tick_ts):
                raise ValueError("non-finite timestamp")
            delay = now - tick_ts
        except (AttributeError, OverflowError, OSError, TypeError, ValueError):
            delay = float("inf")
            issues.append("invalid_timestamp")
            quality_score = 0.0
        if delay > self._max_data_delay_seconds:
            issues.append(f"delayed:{delay:.1f}s")
            quality_score = max(quality_score, 0.5)

        if invalid_numeric_fields or "invalid_timestamp" in issues:
            quality_score = 0.0

        # 更新跟踪状态
        if quality_score > 0.5:
            self._last_prices[symbol] = tick.price
            self._last_update_times[symbol] = now

        return {
            "symbol": symbol,
            "quality_score": quality_score,
            "issues": issues,
            "delay_seconds": round(delay, 2),
            "timestamp": now,
        }

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return False

    @staticmethod
    def _parse_kline_timestamp(value: Any) -> Optional[float]:
        try:
            timestamp = float(value)
            if not math.isfinite(timestamp):
                return None
            return timestamp / 1000.0 if timestamp > 1e11 else timestamp
        except (TypeError, ValueError, OverflowError):
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
            except (TypeError, ValueError, OverflowError, OSError):
                return None

    @staticmethod
    def _kline_values(kline: Any) -> Optional[tuple]:
        if isinstance(kline, dict):
            timestamp = kline.get("timestamp", kline.get("ts", kline.get("time")))
            values = (
                kline.get("open"), kline.get("high"), kline.get("low"),
                kline.get("close"), kline.get("vol", kline.get("volume")),
            )
            return (timestamp, values) if timestamp is not None else None
        if isinstance(kline, (list, tuple)) and len(kline) >= 6:
            return kline[0], tuple(kline[1:6])
        return None

    def check_kline_series(
        self,
        symbol: str,
        klines: List[Any],
        timeframe: str,
    ) -> Dict[str, Any]:
        """Check finite OHLCV values and timestamp order/gaps without altering bars."""
        issues = []
        invalid_rows = []
        timestamps = []
        interval_seconds = self._timeframe_seconds(timeframe)
        for index, kline in enumerate(klines or []):
            parsed = self._kline_values(kline)
            if parsed is None:
                issues.append(f"invalid_format:{index}")
                invalid_rows.append(index)
                continue
            raw_timestamp, values = parsed
            timestamp = self._parse_kline_timestamp(raw_timestamp)
            finite = all(self._is_finite_number(value) for value in values)
            if timestamp is None:
                issues.append(f"invalid_timestamp:{index}")
                invalid_rows.append(index)
            else:
                timestamps.append((index, timestamp))
            if not finite:
                issues.append(f"non_finite_ohlcv:{index}")
                invalid_rows.append(index)

        missing_bars = 0
        for (left_index, left_ts), (right_index, right_ts) in zip(timestamps, timestamps[1:]):
            delta = right_ts - left_ts
            if delta <= 0:
                issues.append(f"non_monotonic_timestamp:{left_index}:{right_index}")
                continue
            if interval_seconds and delta > interval_seconds * self._max_kline_gap_intervals:
                gap = max(1, round(delta / interval_seconds) - 1)
                missing_bars += gap
                issues.append(f"kline_gap:{left_index}:{right_index}:{gap}")

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "valid": not issues,
            "bar_count": len(klines or []),
            "missing_bars": missing_bars,
            "invalid_rows": sorted(set(invalid_rows)),
            "issues": issues,
            "checked_at": time.time(),
        }

    @staticmethod
    def _timeframe_seconds(timeframe: str) -> Optional[float]:
        normalized = str(timeframe).strip().lower()
        if normalized.endswith("m") and normalized[:-1].isdigit():
            return int(normalized[:-1]) * 60.0
        if normalized.endswith("h") and normalized[:-1].isdigit():
            return int(normalized[:-1]) * 3600.0
        if normalized.endswith("d") and normalized[:-1].isdigit():
            return int(normalized[:-1]) * 86400.0
        return None

    def is_healthy(self, symbol: str) -> bool:
        """判断某币种数据是否健康"""
        last_update = self._last_update_times.get(symbol)
        if not last_update:
            return False
        return time.time() - last_update < self._max_data_delay_seconds

    def get_quality_summary(self) -> Dict[str, Any]:
        """获取整体数据质量摘要"""
        now = time.time()
        stale_symbols = [
            s for s, ts in self._last_update_times.items()
            if now - ts > self._max_data_delay_seconds
        ]
        return {
            "monitored_symbols": len(self._last_update_times),
            "stale_symbols": stale_symbols,
            "violation_counts": self._violation_counts.copy(),
            "overall_healthy": len(stale_symbols) == 0,
        }


class MarketDataService:
    def __init__(self, config: Dict[str, Any], okx_client, redis_cache, alert_manager=None):
        self.config = config
        self._okx_client = okx_client
        self._redis_cache = redis_cache
        self._alert_manager = alert_manager
        self._rest_fallback_max_concurrency = max(
            1,
            int(config.get("market_data", {}).get("rest_fallback_max_concurrency", 5)),
        )
        self._tick_callbacks: List[Callable] = []
        self._subscribed_symbols = []
        self._quality_checker = DataQualityChecker(config)
        self._fallback_task = None
        self._quality_monitor_task = None
        self._running = False

    def register_tick_callback(self, callback: Callable):
        self._tick_callbacks.append(callback)

    async def subscribe_market_data(self, symbols: List[str]):
        self._subscribed_symbols = symbols
        self._running = True

        async def tick_handler(tick):
            quality_report = self._quality_checker.check_tick(tick)
            if quality_report["quality_score"] <= 0:
                logger.warning(f"Tick data quality too low for {tick.symbol}: {quality_report['issues']}")
                return

            self._redis_cache.set_tick(tick)
            for callback in self._tick_callbacks:
                try:
                    result = callback(tick)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Error in tick callback: {e}")

        self._okx_client.tick_callback = tick_handler

        await self._okx_client.subscribe_market_data(symbols)
        await self._okx_client.subscribe_private_data()

        # 启动REST轮询备用和质量监控
        self._fallback_task = asyncio.create_task(self._rest_fallback_loop())
        self._quality_monitor_task = asyncio.create_task(self._quality_monitor_loop())

        logger.info(f"Subscribed to market data for {len(symbols)} symbols")

    async def _rest_fallback_loop(self):
        """REST API轮询备用：当WebSocket长时间无数据时补充tick，使用指数退避"""
        fallback_interval = self.config.get("market_data", {}).get("rest_fallback_interval_seconds", 5)
        # 半死连接判定阈值：WS 连接状态仍 OPEN 但 books 数据断流超过该秒数即触发 REST 兜底。
        # 必须早于 okx_client 的数据断流重连阈值(120s)与 risk_gate 的 network_timeout_threshold(300s)，
        # 确保 WS 半死期间行情靠 REST 维持、不会因数据断流触发 L5 熔断冻结交易。
        data_stale_threshold = self.config.get("market_data", {}).get("ws_data_stale_threshold_seconds", 30)
        _last_disconnected_log = 0  # 避免日志刷屏，每60秒才记录一次
        _disconnected_start = 0.0  # 断开开始时间
        _current_backoff = fallback_interval  # 当前退避间隔

        while self._running:
            try:
                await asyncio.sleep(_current_backoff)

                ws_down = not self._okx_client.is_ws_public_connected()
                ws_status = self._okx_client.get_ws_status()
                public_data_age = ws_status.get("public_last_data_age")
                # 半死连接检测：WS 连接状态仍 OPEN 但 books 数据已断流超过阈值
                data_stale = public_data_age is not None and public_data_age > data_stale_threshold

                if ws_down or data_stale:
                    now_ts = time.time()
                    if _disconnected_start == 0:
                        _disconnected_start = now_ts

                    # 指数退避：断开时间越长，轮询间隔越大（上限60秒）
                    disconnected_duration = now_ts - _disconnected_start
                    if disconnected_duration > 60:
                        _current_backoff = min(60, fallback_interval * (2 ** min(4, int(disconnected_duration / 60))))

                    if now_ts - _last_disconnected_log >= 60:
                        reason = "disconnected" if ws_down else f"data stale ({public_data_age:.0f}s)"
                        logger.info(f"Public WebSocket {reason} for {disconnected_duration:.0f}s, "
                                   f"REST fallback interval={_current_backoff}s")
                        _last_disconnected_log = now_ts
                else:
                    _last_disconnected_log = 0
                    _disconnected_start = 0
                    _current_backoff = fallback_interval  # 重置退避
                    continue

                fallback_tickers = await self._fetch_rest_fallback_tickers()
                for symbol, ticker in fallback_tickers:
                    try:
                        if not ticker:
                            continue

                        tick = TickData(
                            symbol=symbol,
                            price=float(ticker.get("last", 0)),
                            volume=float(ticker.get("vol24h", 0)),
                            bid_price=float(ticker.get("bidPx", 0)),
                            bid_volume=float(ticker.get("bidSz", 0)),
                            ask_price=float(ticker.get("askPx", 0)),
                            ask_volume=float(ticker.get("askSz", 0)),
                            timestamp=datetime.fromtimestamp(int(ticker.get("ts", 0)) / 1000),
                        )

                        quality_report = self._quality_checker.check_tick(tick)
                        if quality_report["quality_score"] <= 0:
                            continue

                        self._redis_cache.set_tick(tick)
                        for callback in self._tick_callbacks:
                            try:
                                result = callback(tick)
                                if asyncio.iscoroutine(result):
                                    await result
                            except Exception as e:
                                logger.error(f"Error in fallback tick callback: {e}")
                    except Exception as e:
                        logger.debug(f"REST fallback error for {symbol}: {e}")
            except Exception as e:
                logger.error(f"REST fallback loop error: {e}")
                await asyncio.sleep(5)

    async def _fetch_rest_fallback_tickers(self):
        """Fetch subscribed symbols concurrently without blocking the event loop."""
        symbols = list(self._subscribed_symbols)
        semaphore = asyncio.Semaphore(self._rest_fallback_max_concurrency)

        async def fetch(symbol):
            async with semaphore:
                return await asyncio.to_thread(self._okx_client.get_ticker, symbol)

        results = await asyncio.gather(
            *(fetch(symbol) for symbol in symbols),
            return_exceptions=True,
        )
        fallback_tickers = []
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.debug(f"REST fallback error for {symbol}: {result}")
                continue
            fallback_tickers.append((symbol, result))
        return fallback_tickers

    async def _quality_monitor_loop(self):
        """定期检查数据质量并记录"""
        monitor_interval = self.config.get("market_data", {}).get("quality_monitor_interval_seconds", 60)

        while self._running:
            try:
                await asyncio.sleep(monitor_interval)
                summary = self._quality_checker.get_quality_summary()
                if not summary["overall_healthy"]:
                    logger.warning(f"Market data quality degraded: {summary}")
                    await self._emit_quality_alert(summary)
                else:
                    logger.debug(f"Market data quality healthy: {summary['monitored_symbols']} symbols monitored")
            except Exception as e:
                logger.error(f"Quality monitor loop error: {e}")

    async def _emit_quality_alert(self, summary: Dict[str, Any]):
        """数据质量降级时通过告警管理器通知（断流/脏 tick）。

        告警管理器内部按 SYSTEM_DATA_QUALITY 类型做 5 分钟级去重，避免每分钟刷屏。
        """
        if not self._alert_manager:
            return
        stale = summary.get("stale_symbols", [])
        violations = summary.get("violation_counts", {})
        message = (
            f"行情数据质量降级：断流品种 {stale if stale else '无'}；"
            f"脏 tick 累计 {violations if violations else '无'}"
        )
        try:
            await self._alert_manager.send_system_alert("DATA_QUALITY", message, metadata=summary)
        except Exception as e:
            logger.debug(f"Failed to emit data quality alert: {e}")

    def get_subscribed_symbols(self) -> List[str]:
        return self._subscribed_symbols

    def get_ws_status(self) -> Dict[str, Any]:
        return self._okx_client.get_ws_status()

    def get_quality_summary(self) -> Dict[str, Any]:
        return self._quality_checker.get_quality_summary()

    async def shutdown(self):
        self._running = False

        if self._fallback_task:
            self._fallback_task.cancel()
            try:
                await self._fallback_task
            except asyncio.CancelledError:
                pass

        if self._quality_monitor_task:
            self._quality_monitor_task.cancel()
            try:
                await self._quality_monitor_task
            except asyncio.CancelledError:
                pass

        try:
            await self._okx_client.close_websocket()
            logger.info("WebSocket connections closed")
        except Exception as e:
            logger.error(f"Error closing WebSocket: {e}")
