"""
TWAP 执行器 (Time-Weighted Average Price)

将父订单拆分为等时等量的子订单，最小化市场冲击：
  - 线性调度：总时间/切片数 = 每切片间隔
  - 自适应调度：根据市场流动性动态调整切片大小和执行速度
  - 紧急/保守模式：偏离基准时加速/减速
  - 未完成量处理：超时时的市价清算策略
  - 反探测：随机化切片时间和大小
"""
import asyncio
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
import numpy as np
from loguru import logger

from execution.algo_orders.algo_execution_engine import (
    AlgoOrderConfig, AlgoExecutionResult, ExecutionSlice,
    AlgoOrderStatus,
)


@dataclass
class TWAPSlice:
    """TWAP 切片"""
    slice_id: str
    sequence: int
    scheduled_time: datetime
    quantity: float
    # 执行结果
    executed: bool = False
    filled_qty: float = 0.0
    avg_price: float = 0.0
    slippage_bps: float = 0.0


class TWAPState:
    """TWAP 执行状态"""
    IDLE = "idle"
    SCHEDULING = "scheduling"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class TWAPConfig:
    """TWAP 配置"""
    duration_seconds: float = 3600.0
    num_slices: int = 20
    randomize_interval: bool = True     # 随机化间隔时间
    interval_jitter_pct: float = 0.2    # 间隔抖动比例(±20%)
    randomize_size: bool = True         # 随机化切片大小
    size_jitter_pct: float = 0.1        # 大小抖动比例(±10%)
    # 自适应
    adaptive_enabled: bool = True
    urgency_threshold: float = 0.3       # 剩余时间<30%时加速
    passive_on_favorable: bool = True    # 有利价格时保持保守
    # 回退
    fallback_to_market_at_pct: float = 0.90  # 剩余时间<10%时市价清算
    # 限价单参数
    limit_offset_bps: float = 5.0        # 限价偏离中间价 (bps)


# ═══════════════════════════════════════════════════════════════
# TWAP 执行器
# ═══════════════════════════════════════════════════════════════

class TWAPExecutor:
    """TWAP 算法执行器"""

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config.get("twap_executor", {}) if config else {}
        self._default_duration = cfg.get("default_duration", 3600)
        self._default_slices = cfg.get("default_slices", 20)
        self._min_slice_interval = cfg.get("min_slice_interval", 5.0)  # 秒
        self._max_slice_interval = cfg.get("max_slice_interval", 300.0)
        self._limit_offset_bps = cfg.get("limit_offset_bps", 5.0)
        # 回退
        self._fallback_pct = cfg.get("fallback_to_market_at_pct", 0.90)

        logger.info(f"TWAPExecutor initialized: default_duration={self._default_duration}s, "
                    f"slices={self._default_slices}")

    # ── 切片计算 ──────────────────────────────────────────────

    def compute_schedule(self, total_quantity: float, start_time: datetime,
                          end_time: datetime, num_slices: int,
                          jitter: bool = True) -> List[TWAPSlice]:
        """计算 TWAP 等时间隔切片计划"""
        if num_slices <= 0:
            return []

        duration = (end_time - start_time).total_seconds()
        if duration <= 0:
            return []

        base_interval = duration / num_slices
        base_qty = total_quantity / num_slices

        slices = []
        remaining_qty = total_quantity

        for i in range(num_slices):
            # 随机化间隔
            if jitter:
                jitter_factor = 1 + random.uniform(-0.2, 0.2)
                interval = base_interval * jitter_factor
            else:
                interval = base_interval

            # 随机化大小
            if jitter:
                size_jitter = 1 + random.uniform(-0.1, 0.1)
                qty = min(base_qty * size_jitter, remaining_qty)
            else:
                qty = base_qty

            # 最后一片用尽余量
            if i == num_slices - 1:
                qty = remaining_qty

            qty = max(qty, 0.0)
            scheduled = start_time + timedelta(seconds=(i + 0.5) * base_interval)

            slices.append(TWAPSlice(
                slice_id=f"twap_{i:03d}",
                sequence=i,
                scheduled_time=scheduled,
                quantity=round(qty, 6),
            ))

            remaining_qty -= qty

        return slices

    def adjust_schedule(self, slices: List[TWAPSlice],
                         filled_so_far: float, total_qty: float,
                         remaining_time: float, total_duration: float,
                         is_favorable_price: bool) -> List[TWAPSlice]:
        """自适应调整未执行的切片"""
        remaining_qty = total_qty - filled_so_far
        pending = [s for s in slices if not s.executed]

        if not pending or remaining_qty <= 0:
            return slices

        time_ratio = remaining_time / max(total_duration, 1.0)

        # 自适应策略
        if time_ratio < 0.2:
            # 时间紧迫：加速执行，增大切片
            multiplier = 1.5
        elif is_favorable_price and time_ratio > 0.5:
            # 有利价格 + 时间充裕：保守执行
            multiplier = 0.7
        else:
            multiplier = 1.0

        # 重新分配未执行切片的量
        new_base = (remaining_qty / len(pending)) * multiplier
        for i, s in enumerate(pending):
            if i == len(pending) - 1:
                # 最后一片用尽
                cumulative = sum((pending[j].quantity if j < i else 0) for j in range(len(pending)))
                s.quantity = remaining_qty - cumulative
            else:
                s.quantity = new_base
            s.quantity = max(s.quantity, 0.0)

        return slices

    # ── 主执行逻辑 ────────────────────────────────────────────

    async def execute(self, config: AlgoOrderConfig,
                       on_slice_filled: Callable = None) -> AlgoExecutionResult:
        """执行 TWAP 算法订单"""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=config.duration_seconds)

        # 创建切片计划
        twap_cfg = TWAPConfig(
            duration_seconds=config.duration_seconds,
            num_slices=config.num_slices,
            randomize_interval=True,
            randomize_size=True,
            adaptive_enabled=config.adaptive,
            limit_offset_bps=self._limit_offset_bps,
        )

        slices = self.compute_schedule(
            total_quantity=config.total_quantity,
            start_time=start_time,
            end_time=end_time,
            num_slices=config.num_slices,
            jitter=True,
        )

        result = AlgoExecutionResult(
            order_id=config.order_id,
            algo_type="TWAP",
            symbol=config.symbol,
            side=config.side,
            total_quantity=config.total_quantity,
            filled_quantity=0.0,
            fill_rate=0.0,
            arrival_price=0.0,
            start_time=start_time,
        )

        # 获取到达价格
        if config.market_data_fn:
            try:
                market = config.market_data_fn(config.symbol)
                if market:
                    result.arrival_price = float(market.get("mid", market.get("last", 0)))
            except Exception:
                pass

        total_filled = 0.0
        total_cost = 0.0
        slice_count = 0

        for slc in slices:
            # 通知监控器：切片已排定
            if config.on_slice_scheduled_fn:
                config.on_slice_scheduled_fn(slc.sequence, slc.quantity, slc.scheduled_time)

            now = datetime.now()
            remaining_time = max(0, (end_time - now).total_seconds())

            # 检查是否超时
            if remaining_time <= 0 and total_filled < config.total_quantity * 0.95:
                logger.info(f"TWAP {config.order_id}: nearing timeout, "
                           f"switching to market for remaining {config.total_quantity - total_filled:.4f}")
                # 回退到市价清算
                break

            # 自适应调整（每5个切片或时间紧迫时）
            if twap_cfg.adaptive_enabled and (slice_count % 5 == 0 or remaining_time / max(config.duration_seconds, 1) < 0.3):
                favorable = self._is_favorable_price(config)
                slices = self.adjust_schedule(
                    slices, total_filled, config.total_quantity,
                    remaining_time, config.duration_seconds, favorable,
                )

            # 等待调度时间
            wait_seconds = max(0, (slc.scheduled_time - datetime.now()).total_seconds())
            if wait_seconds > 0:
                await asyncio.sleep(min(wait_seconds, 60))  # 最多等60秒

            # 执行切片
            if config.executor_fn and slc.quantity > 0:
                try:
                    order_params = {
                        "symbol": config.symbol,
                        "side": config.side,
                        "quantity": slc.quantity,
                        "order_type": "limit",
                        "price": self._calc_limit_price(config),
                    }
                    fill = config.executor_fn(order_params)
                    if hasattr(fill, '__await__'):
                        fill = await fill

                    # 通知监控器：切片已提交到交易所
                    if config.on_slice_submitted_fn:
                        config.on_slice_submitted_fn(
                            slc.sequence, slc.quantity, order_params.get("price"), "limit"
                        )

                    if fill:
                        slc.executed = True
                        slc.filled_qty = float(fill.get("filled", 0))
                        slc.avg_price = float(fill.get("avg_price", 0))
                        total_filled += slc.filled_qty
                        total_cost += slc.filled_qty * slc.avg_price

                        # 滑点
                        if slc.avg_price > 0 and result.arrival_price > 0:
                            slc.slippage_bps = (slc.avg_price - result.arrival_price) / result.arrival_price * 10000
                            if config.side == "sell":
                                slc.slippage_bps *= -1

                        es = ExecutionSlice(
                            slice_id=slc.slice_id,
                            sequence=slc.sequence,
                            quantity=slc.quantity,
                            filled_quantity=slc.filled_qty,
                            avg_fill_price=slc.avg_price,
                            slippage_bps=round(slc.slippage_bps, 2),
                            status="filled",
                            submitted_at=datetime.now(),
                            filled_at=datetime.now(),
                        )
                        if on_slice_filled:
                            await on_slice_filled(config.order_id, es)

                except Exception as e:
                    logger.warning(f"TWAP slice {slc.slice_id} failed: {e}")
                    es = ExecutionSlice(
                        slice_id=slc.slice_id, sequence=slc.sequence,
                        quantity=slc.quantity, status="rejected",
                        error_message=str(e),
                    )
                    if on_slice_filled:
                        await on_slice_filled(config.order_id, es)

            slice_count += 1

        # 如果有剩余，市价清算
        remaining = config.total_quantity - total_filled
        if remaining > 0 and config.executor_fn:
            try:
                order_params = {
                    "symbol": config.symbol,
                    "side": config.side,
                    "quantity": remaining,
                    "order_type": "market",
                }
                fill = config.executor_fn(order_params)
                if hasattr(fill, '__await__'):
                    fill = await fill
                if fill:
                    q = float(fill.get("filled", 0))
                    px = float(fill.get("avg_price", 0))
                    total_filled += q
                    total_cost += q * px
                    logger.info(f"TWAP {config.order_id}: final sweep filled {q}")
            except Exception as e:
                logger.warning(f"TWAP final sweep failed: {e}")

        # 填充结果
        result.filled_quantity = total_filled
        result.fill_rate = total_filled / max(config.total_quantity, 1e-10)
        result.avg_execution_price = total_cost / max(total_filled, 1e-10)
        result.target_price = result.avg_execution_price  # TWAP = 执行均价
        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - start_time).total_seconds()

        if result.arrival_price > 0:
            result.arrival_slippage_bps = (result.avg_execution_price - result.arrival_price) / result.arrival_price * 10000

        logger.info(f"TWAP {config.order_id} complete: filled={total_filled:.4f}/{config.total_quantity:.4f} "
                    f"avg_px={result.avg_execution_price:.4f} "
                    f"slip={result.arrival_slippage_bps:.1f}bps")

        return result

    def _calc_limit_price(self, config: AlgoOrderConfig) -> Optional[float]:
        """计算TWAP限价"""
        if not config.market_data_fn:
            return None
        try:
            market = config.market_data_fn(config.symbol)
            if market:
                mid = float(market.get("mid", market.get("last", 0)))
                offset = mid * self._limit_offset_bps / 10000.0
                if config.side == "buy":
                    return mid + offset
                else:
                    return mid - offset
        except Exception:
            pass
        return None

    @staticmethod
    def _is_favorable_price(config: AlgoOrderConfig) -> bool:
        """判断当前价格是否有利"""
        # 简化版：用VWAP/TWAP比较，这里用最近价格趋势
        try:
            if config.market_data_fn:
                market = config.market_data_fn(config.symbol)
                if market:
                    mid = float(market.get("mid", 0))
                    vwap = float(market.get("vwap", mid))
                    if config.side == "buy":
                        return mid < vwap  # 买方：低于VWAP有利
                    else:
                        return mid > vwap  # 卖方：高于VWAP有利
        except Exception:
            pass
        return False

    def get_status(self) -> Dict[str, Any]:
        return {
            "default_duration": self._default_duration,
            "default_slices": self._default_slices,
            "min_slice_interval": self._min_slice_interval,
            "limit_offset_bps": self._limit_offset_bps,
        }
