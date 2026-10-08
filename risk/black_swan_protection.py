"""黑天鹅保护机制

极端行情下的快速响应系统：
- 全市场波动率监控（BTC/ETH等主流币异动检测）
- 闪电崩盘检测（短时大幅下跌）
- 全市场熔断机制
- 紧急全平保护
- 异常成交量检测
"""
import asyncio
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from loguru import logger
import numpy as np


def _finite(value: Any, default: float = 0.0) -> float:
    """安全数值转换：None/非法字符串/NaN/Inf 统一回退到 default。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _close_side_and_pos_side(pos_side_raw: str, qty: float):
    """计算平仓方向与 posSide（对齐 okx_client.close_position 的口径）。

    多头卖、空头买；net 模式按数量正负判断。返回 (side, close_pos_side)。
    """
    ps = (pos_side_raw or "").strip().lower()
    if ps == "long":
        return "sell", "long"
    if ps == "short":
        return "buy", "short"
    return ("sell", "net") if qty > 0 else ("buy", "net")


@dataclass
class SwanEvent:
    """黑天鹅事件记录"""
    event_type: str  # flash_crash / volatility_spike / volume_spike / circuit_breaker
    severity: str  # warning / critical / emergency
    symbol: str
    trigger_price: float
    change_percent: float
    timestamp: datetime
    description: str
    action_taken: str = ""
    action_success: bool = False


class BlackSwanProtection:
    """黑天鹅保护系统"""

    def __init__(self, config: Dict[str, Any], okx_client,
                 global_risk=None, order_executor=None):
        self.config = config
        self.okx_client = okx_client
        self.global_risk = global_risk
        self.order_executor = order_executor

        swan_cfg = config.get("risk", {}).get("black_swan", {})
        self._enabled = swan_cfg.get("enabled", True)
        self._check_interval = swan_cfg.get("check_interval_seconds", 10)

        self._btc_flash_crash_pct = swan_cfg.get("btc_flash_crash_pct", 0.03)
        self._btc_volatility_spike_pct = swan_cfg.get("btc_volatility_spike_pct", 0.05)
        self._flash_crash_window_seconds = swan_cfg.get("flash_crash_window_seconds", 300)
        self._volatility_window_minutes = swan_cfg.get("volatility_window_minutes", 60)

        self._market_circuit_breaker_pct = swan_cfg.get("market_circuit_breaker_pct", 0.08)
        self._emergency_full_close_pct = swan_cfg.get("emergency_full_close_pct", 0.12)

        self._volume_spike_ratio = swan_cfg.get("volume_spike_ratio", 3.0)
        self._volume_lookback_bars = swan_cfg.get("volume_lookback_bars", 20)

        self._monitor_symbols = swan_cfg.get("monitor_symbols", [
            "BTC-USDT-SWAP",
            "ETH-USDT-SWAP",
            "SOL-USDT-SWAP",
        ])

        self._is_running = False
        self._tasks: List[asyncio.Task] = []
        self._is_circuit_broken = False
        self._circuit_breaker_end_time: Optional[datetime] = None
        self._circuit_breaker_reason = ""

        self._price_history: Dict[str, List[tuple]] = {}
        self._volume_history: Dict[str, List[float]] = {}
        self._event_history: List[SwanEvent] = []

        self._max_history_bars = 500

        self._callbacks: Dict[str, List[Callable]] = {
            "flash_crash": [],
            "volatility_spike": [],
            "volume_spike": [],
            "circuit_breaker_triggered": [],
            "circuit_breaker_resolved": [],
        }

    def register_callback(self, event_type: str, callback: Callable):
        """注册事件回调"""
        if event_type in self._callbacks:
            self._callbacks[event_type].append(callback)
            logger.info(f"Registered callback for {event_type}")

    async def start(self):
        if not self._enabled:
            logger.info("Black Swan Protection is disabled")
            return
        if self._is_running:
            return
        self._is_running = True
        self._tasks.append(asyncio.create_task(self._monitor_loop()))
        logger.info(
            f"BlackSwanProtection started: monitoring {len(self._monitor_symbols)} symbols, "
            f"flash_crash_threshold={self._btc_flash_crash_pct:.1%}, "
            f"circuit_breaker={self._market_circuit_breaker_pct:.1%}"
        )

    async def stop(self):
        self._is_running = False
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("BlackSwanProtection stopped")

    async def _monitor_loop(self):
        while self._is_running:
            try:
                if self._is_circuit_broken:
                    await self._check_circuit_breaker_recovery()
                else:
                    await self._check_black_swan_events()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in black swan monitor loop: {e}")
            await asyncio.sleep(self._check_interval)

    async def _check_black_swan_events(self):
        """检查黑天鹅事件"""
        for symbol in self._monitor_symbols:
            try:
                ticker = self.okx_client.get_ticker(symbol)
                if not ticker:
                    continue

                last_price = _finite(ticker.get("last", 0), 0.0)
                if last_price <= 0:
                    continue

                now = datetime.now()

                if symbol not in self._price_history:
                    self._price_history[symbol] = []
                self._price_history[symbol].append((now, last_price))

                if len(self._price_history[symbol]) > self._max_history_bars:
                    self._price_history[symbol] = self._price_history[symbol][-self._max_history_bars:]

                await self._check_flash_crash(symbol, last_price, now)
                await self._check_volatility_spike(symbol, last_price, now)

            except Exception as e:
                logger.warning(f"Error checking black swan for {symbol}: {e}")

    async def _check_flash_crash(self, symbol: str, current_price: float, now: datetime):
        """检测闪电崩盘（短时大幅下跌）"""
        history = self._price_history.get(symbol, [])
        if len(history) < 10:
            return

        window_start = now - timedelta(seconds=self._flash_crash_window_seconds)
        window_prices = [p for t, p in history if t >= window_start]

        if len(window_prices) < 5:
            return

        peak_price = max(window_prices)
        if peak_price <= 0:
            return

        drawdown = (peak_price - current_price) / peak_price

        btc_threshold = self._btc_flash_crash_pct if "BTC" in symbol else self._btc_flash_crash_pct * 1.5

        if drawdown >= btc_threshold:
            severity = "warning"
            # 熔断阈值：跌到 market_circuit_breaker_pct 触发全市场熔断（critical）
            if drawdown >= self._market_circuit_breaker_pct:
                severity = "critical"
            if drawdown >= self._emergency_full_close_pct:
                severity = "emergency"

            event = SwanEvent(
                event_type="flash_crash",
                severity=severity,
                symbol=symbol,
                trigger_price=current_price,
                change_percent=-drawdown,
                timestamp=now,
                description=f"{symbol} 闪电崩盘: {drawdown:.2%} 跌幅在 {self._flash_crash_window_seconds}s 内"
            )

            await self._handle_swan_event(event)

    async def _check_volatility_spike(self, symbol: str, current_price: float, now: datetime):
        """检测波动率飙升"""
        history = self._price_history.get(symbol, [])
        if len(history) < 30:
            return

        window_start = now - timedelta(minutes=self._volatility_window_minutes)
        window_prices = [p for t, p in history if t >= window_start]

        if len(window_prices) < 20:
            return

        returns = []
        for i in range(1, len(window_prices)):
            if window_prices[i-1] > 0:
                returns.append((window_prices[i] - window_prices[i-1]) / window_prices[i-1])

        if len(returns) < 10:
            return

        current_vol = np.std(returns) * np.sqrt(len(returns))
        avg_abs_return = np.mean([abs(r) for r in returns])

        btc_vol_threshold = self._btc_volatility_spike_pct

        if current_vol >= btc_vol_threshold and "BTC" in symbol:
            event = SwanEvent(
                event_type="volatility_spike",
                severity="warning",
                symbol=symbol,
                trigger_price=current_price,
                change_percent=current_vol,
                timestamp=now,
                description=f"{symbol} 波动率飙升: {current_vol:.2%} (近{self._volatility_window_minutes}分钟)"
            )
            await self._handle_swan_event(event)

    async def _handle_swan_event(self, event: SwanEvent):
        """处理黑天鹅事件"""
        logger.warning(f"[BLACK_SWAN] {event.event_type.upper()} {event.severity.upper()}: {event.description}")

        self._event_history.append(event)
        if len(self._event_history) > 100:
            self._event_history = self._event_history[-100:]

        for callback in self._callbacks.get(event.event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in {event.event_type} callback: {e}")

        if event.severity == "emergency":
            await self._trigger_emergency_protection(event)
        elif event.severity == "critical":
            await self._trigger_circuit_breaker(event)

    async def _trigger_circuit_breaker(self, event: SwanEvent):
        """触发熔断机制"""
        if self._is_circuit_broken:
            return

        self._is_circuit_broken = True
        self._circuit_breaker_end_time = datetime.now() + timedelta(minutes=15)  # R65: 30→15
        self._circuit_breaker_reason = event.description

        event.action_taken = "circuit_breaker_triggered"
        event.action_success = True

        logger.critical(
            f"[CIRCUIT_BREAKER] 全市场熔断触发: {event.description}\n"
            f"  暂停时间: 30分钟\n"
            f"  原因: {event.event_type}"
        )

        if self.global_risk and hasattr(self.global_risk, 'pause_trading'):
            self.global_risk.pause_trading(f"熔断: {event.description}")

        for callback in self._callbacks.get("circuit_breaker_triggered", []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in circuit_breaker callback: {e}")

    async def _trigger_emergency_protection(self, event: SwanEvent):
        """触发紧急保护（极端情况下考虑全平）"""
        logger.critical(f"[EMERGENCY] 极端行情保护触发: {event.description}")

        if event.change_percent <= -self._emergency_full_close_pct:
            logger.critical("[EMERGENCY] 跌幅超过紧急阈值，触发全平保护...")
            event.action_taken = "emergency_full_close"
            event.action_success = await self._emergency_close_all()

    async def _emergency_close_all(self) -> bool:
        """紧急全平所有仓位"""
        try:
            if self.order_executor and hasattr(self.order_executor, 'close_all_positions'):
                success = await self.order_executor.close_all_positions(reason="black_swan_emergency")
                logger.warning(f"紧急全平结果: {success}")
                return success
            else:
                # 回退：直接调用 OKX API 全平所有持仓
                logger.warning("无order_executor，尝试直接API全平...")
                return await self._direct_close_all()
        except Exception as e:
            logger.error(f"紧急全平失败: {e}")
            return False

    async def _direct_close_all(self) -> bool:
        """直接通过OKX API全平所有持仓（order_executor不可用时的回退方案）
        P1-5: 批量下单优化 — 一次 API 调用平所有仓位，减少延迟
        """
        try:
            positions = self.okx_client.get_positions()
            if not positions:
                logger.info("无需平仓：无持仓")
                return True

            # P1-5: 构建批量平仓订单体
            order_bodies = []
            for pos_data in positions:
                try:
                    symbol = pos_data.get("instId", "")
                    pos_side = pos_data.get("posSide", "net")
                    pos_qty = _finite(pos_data.get("pos", 0), 0.0)
                    if pos_qty == 0 or not symbol:
                        continue

                    # 平仓方向与 posSide：多头卖、空头买；net 模式按数量正负判断
                    side, close_pos_side = _close_side_and_pos_side(pos_side, pos_qty)

                    # 构建订单体（与 place_order 内部逻辑一致）
                    is_spot = "-SWAP" not in symbol
                    qty = abs(pos_qty)
                    if not is_spot:
                        qty = self.okx_client.coin_to_contracts(symbol, qty)
                        qty = self.okx_client.round_quantity_to_lot(symbol, qty, round_up=True)
                    if qty <= 0:
                        continue

                    body = {
                        "instId": symbol,
                        "side": side,
                        "ordType": "market",
                        "sz": str(qty),
                        "reduceOnly": True,
                    }
                    if is_spot:
                        body["tdMode"] = "cash"
                    else:
                        body["tdMode"] = "isolated"
                        body["posSide"] = close_pos_side

                    order_bodies.append((symbol, close_pos_side, qty, body))
                except Exception as e:
                    logger.error(f"[EMERGENCY] 构建平仓订单失败 {symbol}: {e}")

            # 批量发送（最多 20 单/批）
            if order_bodies:
                batch_bodies = [item[3] for item in order_bodies]
                try:
                    batch_results = self.okx_client.place_batch_orders(batch_bodies)
                    closed_count = sum(1 for r in batch_results if not r.get("_failed", False))
                    for idx, (symbol, close_pos_side, qty, _) in enumerate(order_bodies):
                        if idx < len(batch_results) and not batch_results[idx].get("_failed", False):
                            logger.warning(f"[EMERGENCY] 直接API平仓: {symbol} {close_pos_side} qty={qty}")
                        else:
                            msg = batch_results[idx].get("sMsg", "") if idx < len(batch_results) else "No result"
                            logger.error(f"[EMERGENCY] 直接API平仓失败 {symbol}: {msg}")
                except Exception as e:
                    logger.error(f"[EMERGENCY] 批量平仓失败: {e}")
                    closed_count = 0
            else:
                closed_count = 0

            logger.warning(f"[EMERGENCY] 直接API全平完成: {closed_count}个仓位")
            return True
        except Exception as e:
            logger.error(f"[EMERGENCY] 直接API全平失败: {e}")
            return False

    async def _check_circuit_breaker_recovery(self):
        """检查熔断是否可以恢复"""
        if not self._is_circuit_broken or not self._circuit_breaker_end_time:
            return

        if datetime.now() >= self._circuit_breaker_end_time:
            self._is_circuit_broken = False
            reason = self._circuit_breaker_reason
            self._circuit_breaker_reason = ""

            logger.info(f"[CIRCUIT_BREAKER] 熔断结束，恢复正常交易。原因: {reason}")

            if self.global_risk and hasattr(self.global_risk, '_is_paused'):
                if self.global_risk._pause_reason and "熔断" in self.global_risk._pause_reason:
                    self.global_risk._is_paused = False
                    self.global_risk._pause_reason = None

            for callback in self._callbacks.get("circuit_breaker_resolved", []):
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(None)
                    else:
                        callback(None)
                except Exception as e:
                    logger.error(f"Error in circuit_breaker_resolved callback: {e}")

    def is_circuit_broken(self) -> bool:
        """当前是否处于熔断状态"""
        return self._is_circuit_broken

    def get_status(self) -> Dict[str, Any]:
        """获取黑天鹅保护系统状态"""
        return {
            "enabled": self._enabled,
            "is_running": self._is_running,
            "is_circuit_broken": self._is_circuit_broken,
            "circuit_breaker_end_time": self._circuit_breaker_end_time.isoformat() if self._circuit_breaker_end_time else None,
            "circuit_breaker_reason": self._circuit_breaker_reason,
            "monitor_symbols": self._monitor_symbols,
            "flash_crash_threshold_pct": self._btc_flash_crash_pct,
            "volatility_threshold_pct": self._btc_volatility_spike_pct,
            "circuit_breaker_threshold_pct": self._market_circuit_breaker_pct,
            "recent_events_count": len(self._event_history),
            "recent_events": [
                {
                    "event_type": e.event_type,
                    "severity": e.severity,
                    "symbol": e.symbol,
                    "change_percent": e.change_percent,
                    "timestamp": e.timestamp.isoformat(),
                    "description": e.description,
                    "action_taken": e.action_taken,
                }
                for e in self._event_history[-10:]
            ],
        }

    def can_accept_new_signals(self) -> bool:
        """是否可以接受新信号（熔断时不可以）"""
        if not self._enabled:
            return True
        return not self._is_circuit_broken
