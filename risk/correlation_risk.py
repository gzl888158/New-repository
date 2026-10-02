"""相关性风险控制模块

检测账户持仓中同向高相关性仓位的过度集中，防止行情反向时多仓位同时爆仓。
核心逻辑：
1. 定期拉取账户持仓
2. 获取各 symbol 的近期K线收益序列
3. 计算相关系数矩阵
4. 检测同向高相关（|r|>=0.7）的持仓对，计算"同向高相关保证金占比"
5. 占比超过阈值时按比例减仓最重的仓位
6. 提供对冲建议和相关性矩阵可视化数据
"""
import asyncio
import math
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import numpy as np

from core.direction_unifier import DirectionUnifier


def _finite(value: Any, default: float = 0.0) -> float:
    """安全数值转换：None/非法字符串/NaN/Inf 统一回退到 default。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _position_side(pos: Any) -> str:
    """归一化持仓方向：long/short/buy/sell → long/short；net 模式按数量符号推导。"""
    raw = (getattr(pos, "side", "") or "").strip().lower()
    try:
        return DirectionUnifier.normalize(raw)
    except (ValueError, TypeError):
        pass
    qty = _finite(getattr(pos, "quantity", 0.0), 0.0)
    return "long" if qty >= 0 else "short"


@dataclass
class CorrelationPair:
    """相关性对信息"""
    symbol_a: str
    symbol_b: str
    correlation: float
    direction: str  # same / opposite / unrelated
    margin_a: float = 0
    margin_b: float = 0
    strategy_a: str = ""
    strategy_b: str = ""
    risk_level: str = "low"  # low / medium / high / critical


class CorrelationRiskControl:
    def __init__(self, config: Dict[str, Any], okx_client, redis_cache=None):
        self.config = config
        self.okx_client = okx_client
        self.redis_cache = redis_cache
        self.order_executor = None  # 延迟注入
        self.alert_manager = None  # 延迟注入（减仓失败告警）

        corr_cfg = config.get("risk", {}).get("correlation", {})
        self._threshold = corr_cfg.get("threshold", 0.7)
        self._max_concentration = corr_cfg.get("max_concentration", 0.50)
        self._lookback_bars = corr_cfg.get("lookback_bars", 60)
        self._kline_bar = corr_cfg.get("kline_bar", "1H")
        self._check_interval = corr_cfg.get("check_interval_seconds", 180)
        self._reduce_ratio = corr_cfg.get("reduce_ratio", 0.30)

        self._corr_cache: Dict[tuple, float] = {}
        self._corr_cache_time: Optional[datetime] = None
        self._last_returns: Dict[str, np.ndarray] = {}

        self._is_running = False
        self._tasks: List[asyncio.Task] = []
        self._last_check_result: Dict[str, Any] = {"triggered": False, "detail": ""}
        self._last_positions: List[Any] = []
        self._last_total_equity: float = 0.0
        self._hedge_suggestions: List[Dict[str, Any]] = []

    def set_order_executor(self, executor) -> None:
        """注入订单执行器。"""
        self.order_executor = executor

    def set_alert_manager(self, alert_manager) -> None:
        """注入告警管理器（减仓失败告警）。"""
        self.alert_manager = alert_manager

    async def start(self):
        if self._is_running:
            return
        self._is_running = True
        self._tasks.append(asyncio.create_task(self._monitor_loop()))
        logger.info(f"CorrelationRiskControl started: threshold={self._threshold}, "
                    f"max_concentration={self._max_concentration}, lookback={self._lookback_bars}")

    async def stop(self):
        self._is_running = False
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("CorrelationRiskControl stopped")

    async def _monitor_loop(self):
        while self._is_running:
            try:
                await self._check_correlation_risk()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in correlation risk loop: {e}")
            await asyncio.sleep(self._check_interval)

    async def _check_correlation_risk(self):
        try:
            positions_raw = self.okx_client.get_positions()
        except Exception as e:
            logger.error(f"Failed to get positions for correlation check: {e}")
            return

        if not positions_raw:
            self._last_check_result = {"triggered": False, "detail": "no positions"}
            self._last_positions = []
            return

        parsed = []
        for pos_raw in positions_raw:
            try:
                pos = self.okx_client._parse_position(pos_raw)
            except Exception:
                pos = None
            if pos and abs(_finite(pos.quantity, 0.0)) > 0 and _finite(pos.margin, 0.0) > 0:
                parsed.append(pos)

        self._last_positions = parsed

        if len(parsed) < 2:
            self._last_check_result = {"triggered": False, "detail": f"only {len(parsed)} positions"}
            return

        try:
            account_info = self.okx_client.get_account_info()
            total_equity = _finite(account_info.get("totalEq", 0), 0.0) if account_info else 0.0
        except Exception:
            total_equity = sum(_finite(p.margin, 0.0) for p in parsed)

        self._last_total_equity = total_equity

        if total_equity <= 0:
            return

        corr_matrix = await self._compute_correlation_matrix(parsed)
        if not corr_matrix:
            return

        high_corr_margin = 0.0
        high_corr_pairs: List[Dict[str, Any]] = []
        all_pairs: List[CorrelationPair] = []

        for i in range(len(parsed)):
            for j in range(i + 1, len(parsed)):
                p_a = parsed[i]
                p_b = parsed[j]
                key = tuple(sorted([p_a.symbol, p_b.symbol]))
                r = corr_matrix.get(key)
                if r is None:
                    continue

                side_a = _position_side(p_a)
                side_b = _position_side(p_b)
                same_direction = side_a == side_b and side_a in ("long", "short")

                if abs(r) >= self._threshold:
                    risk_level = "high"
                    if abs(r) >= 0.9:
                        risk_level = "critical"
                    elif abs(r) >= 0.8:
                        risk_level = "high"
                    else:
                        risk_level = "medium"
                else:
                    risk_level = "low"

                direction = "same" if same_direction else ("opposite" if r < -0.3 else "unrelated")

                pair = CorrelationPair(
                    symbol_a=p_a.symbol,
                    symbol_b=p_b.symbol,
                    correlation=round(r, 4),
                    direction=direction,
                    margin_a=_finite(p_a.margin, 0.0),
                    margin_b=_finite(p_b.margin, 0.0),
                    risk_level=risk_level,
                )
                all_pairs.append(pair)

                if same_direction and abs(r) >= self._threshold:
                    pair_margin = _finite(p_a.margin, 0.0) + _finite(p_b.margin, 0.0)
                    high_corr_margin += pair_margin
                    high_corr_pairs.append({
                        "symbol_a": p_a.symbol,
                        "symbol_b": p_b.symbol,
                        "side": side_a,
                        "correlation": round(r, 3),
                        "margin_a": round(_finite(p_a.margin, 0.0), 2),
                        "margin_b": round(_finite(p_b.margin, 0.0), 2),
                        "risk_level": risk_level,
                    })

        concentration = high_corr_margin / total_equity if total_equity > 0 else 0

        self._hedge_suggestions = self._generate_hedge_suggestions(all_pairs, parsed, total_equity)

        if concentration > self._max_concentration:
            logger.warning(
                f"Correlation risk triggered: concentration={concentration:.2%} > "
                f"{self._max_concentration:.2%}, pairs={len(high_corr_pairs)}"
            )
            await self._reduce_high_corr_positions(parsed, corr_matrix, total_equity)
            self._last_check_result = {
                "triggered": True,
                "concentration": concentration,
                "pairs": high_corr_pairs[:5],
                "detail": f"concentration={concentration:.2%}"
            }
        else:
            self._last_check_result = {
                "triggered": False,
                "concentration": concentration,
                "pairs_count": len(high_corr_pairs),
                "detail": f"concentration={concentration:.2%}"
            }

    async def _compute_correlation_matrix(self, positions) -> Dict[tuple, float]:
        symbols = list(set(p.symbol for p in positions))
        if len(symbols) < 2:
            return {}

        returns: Dict[str, np.ndarray] = {}
        for symbol in symbols:
            try:
                klines = self.okx_client.get_kline(symbol, interval=self._kline_bar, limit=self._lookback_bars + 1)
                if not klines or len(klines) < 10:
                    continue
                closes = []
                for k in reversed(klines):
                    if not isinstance(k, (list, tuple)) or len(k) < 5:
                        continue
                    c = _finite(k[4], None)
                    if c is None:
                        continue
                    closes.append(c)
                if len(closes) < 10:
                    continue
                arr = np.array(closes, dtype=float)
                # 防止收盘价为 0 导致除零产生 Inf/NaN
                safe_prev = np.where(arr[:-1] == 0, np.nan, arr[:-1])
                rets = np.diff(arr) / safe_prev
                rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
                returns[symbol] = rets
            except Exception as e:
                logger.debug(f"Failed to get kline for {symbol}: {e}")
                continue

        if len(returns) < 2:
            return {}

        min_len = min(len(r) for r in returns.values())
        if min_len < 5:
            return {}

        for s in returns:
            returns[s] = returns[s][-min_len:]

        self._last_returns = returns

        corr_matrix: Dict[tuple, float] = {}
        sym_list = list(returns.keys())
        for i in range(len(sym_list)):
            for j in range(i + 1, len(sym_list)):
                a, b = sym_list[i], sym_list[j]
                r_arr, s_arr = returns[a], returns[b]
                if np.std(r_arr) == 0 or np.std(s_arr) == 0:
                    corr_matrix[tuple(sorted([a, b]))] = 0.0
                    continue
                try:
                    corr = float(np.corrcoef(r_arr, s_arr)[0, 1])
                except Exception:
                    corr = 0.0
                if np.isnan(corr):
                    corr = 0.0
                corr_matrix[tuple(sorted([a, b]))] = corr

        self._corr_cache = corr_matrix
        self._corr_cache_time = datetime.now()
        return corr_matrix

    def _generate_hedge_suggestions(
        self, all_pairs: List[CorrelationPair], positions, total_equity: float
    ) -> List[Dict[str, Any]]:
        """生成对冲建议"""
        suggestions = []

        long_positions = [p for p in positions if _position_side(p) == "long"]
        short_positions = [p for p in positions if _position_side(p) == "short"]

        long_margin = sum(_finite(p.margin, 0.0) for p in long_positions)
        short_margin = sum(_finite(p.margin, 0.0) for p in short_positions)
        net_exposure = long_margin - short_margin
        net_exposure_pct = net_exposure / total_equity if total_equity > 0 else 0

        if abs(net_exposure_pct) > 0.3:
            suggestions.append({
                "type": "net_exposure",
                "severity": "high" if abs(net_exposure_pct) > 0.5 else "medium",
                "title": "净敞口过高",
                "description": f"净敞口 {net_exposure_pct:+.2%}，建议{'增加空头对冲' if net_exposure > 0 else '增加多头对冲'}",
                "current_net_exposure": round(net_exposure_pct, 4),
                "recommended_action": "reduce_net_exposure",
                "target_exposure": "±20%",
            })

        high_corr_same = [p for p in all_pairs if p.direction == "same" and abs(p.correlation) >= self._threshold]
        if high_corr_same:
            critical_pairs = [p for p in high_corr_same if p.risk_level == "critical"]
            suggestions.append({
                "type": "correlation_concentration",
                "severity": "critical" if critical_pairs else "high",
                "title": f"同向高相关仓位集中 ({len(high_corr_same)}对)",
                "description": f"检测到 {len(high_corr_same)} 对同向高相关仓位，建议减仓集中度最高的仓位或添加反向对冲",
                "high_corr_pairs_count": len(high_corr_same),
                "critical_pairs_count": len(critical_pairs),
                "top_pairs": [
                    {"symbol_a": p.symbol_a, "symbol_b": p.symbol_b, "correlation": p.correlation}
                    for p in sorted(high_corr_same, key=lambda x: abs(x.correlation), reverse=True)[:3]
                ],
                "recommended_action": "reduce_concentration_or_hedge",
            })

        if len(positions) >= 2:
            unique_symbols = list(set(p.symbol for p in positions))
            if len(unique_symbols) <= 2:
                suggestions.append({
                    "type": "diversification",
                    "severity": "medium",
                    "title": "品种分散不足",
                    "description": f"当前仅持有 {len(unique_symbols)} 个币种，建议增加低相关性品种分散风险",
                    "current_symbols": unique_symbols,
                    "recommended_action": "add_low_correlation_assets",
                })

        return suggestions

    async def _reduce_high_corr_positions(self, positions, corr_matrix, total_equity):
        try:
            sorted_positions = sorted(positions, key=lambda p: p.margin, reverse=True)
            reduced_count = 0
            for pos in sorted_positions:
                is_high_corr = False
                for other in positions:
                    if other.symbol == pos.symbol:
                        continue
                    if _position_side(other) != _position_side(pos):
                        continue
                    key = tuple(sorted([pos.symbol, other.symbol]))
                    r = corr_matrix.get(key, 0)
                    if abs(r) >= self._threshold:
                        is_high_corr = True
                        break

                if not is_high_corr:
                    continue

                norm_side = _position_side(pos)
                side = "sell" if norm_side == "long" else "buy"
                raw_side = (getattr(pos, "side", "") or "").strip().lower()
                pos_side = raw_side if raw_side in ("long", "short", "net") else norm_side
                reduce_qty = abs(_finite(pos.quantity, 0.0)) * self._reduce_ratio
                if reduce_qty <= 0:
                    continue

                try:
                    # 优先通过 order_executor 下单（经过五层风控），回退到直接 API
                    last_err: Optional[Exception] = None
                    for _attempt in range(3):
                        try:
                            if self.order_executor and hasattr(self.order_executor, 'handle_signal'):
                                signal = {
                                    "symbol": pos.symbol,
                                    "strategy_name": "correlation_risk",
                                    "signal_type": "correlation_reduce",
                                    "direction": "close",
                                    "side": side,
                                    "quantity": reduce_qty,
                                    "price": 0,
                                    "leverage": pos.leverage,
                                    "pos_side": pos_side,
                                    "order_type": "market",
                                    "reduce_only": True,
                                    "priority": 10,
                                    "timestamp": datetime.now().isoformat(),
                                }
                                await self.order_executor.handle_signal(signal)
                            else:
                                self.okx_client.place_order(
                                    symbol=pos.symbol,
                                    side=side,
                                    order_type="market",
                                    quantity=reduce_qty,
                                    leverage=pos.leverage,
                                    reduce_only=True,
                                    pos_side=pos_side,
                                )
                            last_err = None
                            break
                        except Exception as retry_err:
                            last_err = retry_err
                            if _attempt < 2:
                                await asyncio.sleep(0.5 * (_attempt + 1))
                    if last_err is not None:
                        raise last_err
                    logger.warning(
                        f"Correlation risk: reduced {pos.symbol} ({pos.side}) by "
                        f"{self._reduce_ratio*100:.0f}% (qty={reduce_qty:.4f})"
                    )
                    reduced_count += 1
                    await asyncio.sleep(0.2)
                    if reduced_count >= 2:
                        break
                except Exception as e:
                    logger.error(f"Failed to reduce {pos.symbol} for correlation risk after retries: {e}")
                    if self.alert_manager:
                        try:
                            await self.alert_manager.send_alert(
                                "correlation_reduce_failed",
                                f"相关性减仓失败 {pos.symbol}: {e}",
                                severity="WARNING",
                                symbol=pos.symbol,
                            )
                        except Exception:
                            pass

            logger.info(f"Correlation risk reduction done: {reduced_count} positions reduced")
        except Exception as e:
            logger.error(f"Error in correlation risk reduction: {e}")

    def get_correlation_matrix(self) -> Dict[str, Any]:
        """获取完整相关性矩阵数据（用于可视化）"""
        if not self._last_positions or not self._corr_cache:
            return {
                "symbols": [],
                "matrix": [],
                "positions": [],
                "timestamp": None,
            }

        symbols = sorted(list(set(p.symbol for p in self._last_positions)))
        n = len(symbols)
        matrix = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                key = tuple(sorted([symbols[i], symbols[j]]))
                r = self._corr_cache.get(key, 0.0)
                matrix[i][j] = round(r, 4)
                matrix[j][i] = round(r, 4)

        positions_info = []
        for pos in self._last_positions:
            positions_info.append({
                "symbol": pos.symbol,
                "side": _position_side(pos),
                "quantity": _finite(pos.quantity, 0.0),
                "margin": round(_finite(pos.margin, 0.0), 2),
                "unrealized_pnl": round(_finite(getattr(pos, 'unrealized_pnl', 0), 0.0), 2),
            })

        return {
            "symbols": symbols,
            "matrix": matrix,
            "positions": positions_info,
            "total_equity": round(self._last_total_equity, 2),
            "threshold": self._threshold,
            "lookback_bars": self._lookback_bars,
            "kline_bar": self._kline_bar,
            "timestamp": self._corr_cache_time.isoformat() if self._corr_cache_time else None,
        }

    def get_hedge_suggestions(self) -> List[Dict[str, Any]]:
        """获取对冲建议"""
        return self._hedge_suggestions

    def get_status(self) -> Dict[str, Any]:
        return {
            "threshold": self._threshold,
            "max_concentration": self._max_concentration,
            "last_check": self._last_check_result,
            "cache_size": len(self._corr_cache),
            "cache_time": self._corr_cache_time.isoformat() if self._corr_cache_time else None,
            "positions_count": len(self._last_positions),
            "hedge_suggestions_count": len(self._hedge_suggestions),
        }

    def can_open_position(self, symbol: str, side: str, margin: float) -> Tuple[bool, str]:
        """开仓前相关性风控检查：如果新开仓会导致同向高相关集中度过高，拒绝。

        Args:
            symbol: 待开仓品种
            side: "long" / "short"
            margin: 预计保证金（USDT）

        Returns:
            (allowed, reason) — allowed=False 时 reason 说明拒绝原因
        """
        if not self._last_positions or margin <= 0:
            return True, ""

        total_equity = self._last_total_equity
        if total_equity <= 0:
            return True, ""

        # 计算加入新仓位后的同向高相关保证金
        new_high_corr_margin = 0.0
        for pos in self._last_positions:
            pos_side = _position_side(pos)
            if pos_side != side:
                continue
            key = tuple(sorted([symbol, pos.symbol]))
            r = self._corr_cache.get(key)
            if r is None:
                continue
            if abs(r) >= self._threshold:
                new_high_corr_margin += _finite(pos.margin, 0.0)

        new_high_corr_margin += margin
        new_concentration = new_high_corr_margin / total_equity

        if new_concentration > self._max_concentration:
            reason = (
                f"相关性集中度超限: 新开仓后 {new_concentration:.2%} > "
                f"{self._max_concentration:.2%}"
            )
            logger.warning(f"[CORR_RISK] 拒绝开仓 {symbol} {side}: {reason}")
            return False, reason

        return True, ""
