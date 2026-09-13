"""
冰山订单执行器 (Iceberg Order Executor)

将大订单隐藏在显示的小订单之后，避免暴露真实意图：
  - 显示量控制：仅展示总订单的冰山一角 (tip)
  - 自动刷新：显示量成交后自动补充
  - 随机化显示量：每次随机变化tip大小避免探测
  - 价格偏移：可选择限价偏离来提高成交率
  - 最大显示量约束：遵循交易所的冰山单限制
"""
import asyncio
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
import numpy as np
from loguru import logger

from execution.algo_orders.algo_execution_engine import (
    AlgoOrderConfig, AlgoExecutionResult, ExecutionSlice,
    AlgoOrderStatus,
)


@dataclass
class IcebergConfig:
    """冰山订单配置"""
    total_quantity: float
    display_quantity: float             # 可见数量 (tip)
    min_display_quantity: float = 0.001
    max_display_quantity: float = 1.0
    # 随机化
    randomize_display: bool = True       # 随机化显示量
    display_jitter_pct: float = 0.15     # 显示量抖动比例(±15%)
    # 刷新延迟
    refresh_delay_seconds: float = 1.0   # 成交后刷新间隔(秒)
    randomize_delay: bool = True
    delay_jitter_seconds: float = 3.0    # 延迟抖动(秒)
    # 价格
    limit_offset_bps: float = 2.0        # 限价偏离(bps)
    price_improvement: bool = True       # 启用价格改善
    # 风控
    max_slippage_bps: float = 10.0
    cancel_if_not_filled_seconds: float = 60.0  # 超过此时未成交则取消当前显示
    auto_switch_to_market_at_pct: float = 0.90   # 进度>90%时切换到市价


class IcebergState:
    IDLE = "idle"
    DISPLAYING = "displaying"      # 显示中（挂单中）
    WAITING_FILL = "waiting_fill"  # 等待成交
    REFRESHING = "refreshing"      # 刷新中
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass
class IcebergSlice:
    """冰山订单刷新切片"""
    slice_id: str
    sequence: int
    display_qty: float         # 本次显示量
    total_displayed: float     # 累计显示量
    price: float
    # 执行结果
    filled_qty: float = 0.0
    avg_price: float = 0.0
    refresh_count: int = 0     # 刷新次数
    display_time: float = 0.0  # 挂单时间(秒)
    status: str = "pending"


# ═══════════════════════════════════════════════════════════════
# 冰山订单执行器
# ═══════════════════════════════════════════════════════════════

class IcebergOrderExecutor:
    """冰山订单算法执行器"""

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config.get("iceberg_executor", {}) if config else {}
        # 显示量
        self._default_display_ratio = cfg.get("default_display_ratio", 0.1)  # 默认显示10%
        self._min_display = cfg.get("min_display_quantity", 0.001)
        self._max_display = cfg.get("max_display_quantity", 1.0)
        # 随机化
        self._randomize_display = cfg.get("randomize_display", True)
        self._display_jitter_pct = cfg.get("display_jitter_pct", 0.15)
        self._randomize_delay = cfg.get("randomize_delay", True)
        self._delay_jitter_seconds = cfg.get("delay_jitter_seconds", 3.0)
        # 价格
        self._limit_offset_bps = cfg.get("limit_offset_bps", 2.0)
        self._price_improvement = cfg.get("price_improvement", True)
        # 超时
        self._cancel_timeout = cfg.get("cancel_if_not_filled_seconds", 60.0)
        self._auto_market_pct = cfg.get("auto_switch_to_market_at_pct", 0.90)
        # 最大刷新
        self._max_refreshes = cfg.get("max_refreshes", 50)

        logger.info(f"IcebergOrderExecutor initialized: display_ratio={self._default_display_ratio}, "
                    f"jitter={self._display_jitter_pct*100:.0f}%")

    # ── 显示量计算 ────────────────────────────────────────────

    def compute_display_quantity(self, total_remaining: float,
                                  ) -> float:
        """计算本次显示量（冰山一角）"""
        # 基础显示量
        base = total_remaining * self._default_display_ratio

        # 随机化
        if self._randomize_display:
            jitter = random.uniform(-self._display_jitter_pct, self._display_jitter_pct)
            display = base * (1 + jitter)
        else:
            display = base

        # 约束在范围
        display = max(self._min_display, min(display, min(self._max_display, total_remaining)))

        # 不能超过剩余量
        display = min(display, total_remaining)

        # 最后一次直接显示全部
        if total_remaining <= self._max_display:
            display = total_remaining

        return round(display, 6)

    def compute_refresh_delay(self) -> float:
        """计算刷新延迟"""
        base = 1.0
        if self._randomize_delay:
            jitter = random.uniform(0, self._delay_jitter_seconds)
            return base + jitter
        return base

    def compute_limit_price(self, mid_price: float, side: str) -> float:
        """计算冰山限价"""
        offset = mid_price * self._limit_offset_bps / 10000.0
        if side == "buy":
            # 买单：限价略高于中间价以提高成交
            return round(mid_price + offset, 2)
        else:
            # 卖单：限价略低于中间价
            return round(mid_price - offset, 2)

    def compute_price_improvement(self, current_price: float, side: str,
                                   is_fast_market: bool = False) -> float:
        """计算价格改进（在快速市场中微调限价）"""
        if not self._price_improvement or not is_fast_market:
            return current_price

        improvement_bps = random.uniform(0.5, 2.0)
        if side == "buy":
            return current_price * (1 + improvement_bps / 10000.0)
        else:
            return current_price * (1 - improvement_bps / 10000.0)

    # ── 主执行逻辑 ────────────────────────────────────────────

    async def execute(self, config: AlgoOrderConfig,
                       on_slice_filled: Callable = None) -> AlgoExecutionResult:
        """执行冰山订单算法"""
        start_time = datetime.now()
        total_qty = config.total_quantity

        result = AlgoExecutionResult(
            order_id=config.order_id,
            algo_type="ICEBERG",
            symbol=config.symbol,
            side=config.side,
            total_quantity=total_qty,
            filled_quantity=0.0,
            fill_rate=0.0,
            arrival_price=0.0,
            start_time=start_time,
        )

        # 获取中间价
        mid_price = 0.0
        if config.market_data_fn:
            try:
                market = config.market_data_fn(config.symbol)
                if market:
                    mid_price = float(market.get("mid", market.get("last", 0)))
                    result.arrival_price = mid_price
            except Exception:
                pass

        total_filled = 0.0
        total_cost = 0.0
        sequence = 0
        slices_log: List[IcebergSlice] = []

        while total_filled < total_qty and sequence < self._max_refreshes:
            remaining = total_qty - total_filled

            # 进度 > auto_market_pct 时切换到市价
            progress = total_filled / max(total_qty, 1e-10)
            if progress >= self._auto_market_pct:
                if config.executor_fn:
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
                    except Exception as e:
                        logger.warning(f"Iceberg final sweep failed: {e}")
                break

            # 计算显示量
            display_qty = self.compute_display_quantity(remaining)
            limit_price = self.compute_limit_price(mid_price, config.side)

            # 通知监控器：切片已排定
            if config.on_slice_scheduled_fn:
                config.on_slice_scheduled_fn(sequence, display_qty, datetime.now())

            # 挂单
            slice_entry = IcebergSlice(
                slice_id=f"iceberg_{sequence:03d}",
                sequence=sequence,
                display_qty=display_qty,
                total_displayed=total_filled + display_qty,
                price=limit_price,
            )
            display_start = time.time()

            # 提交限价单
            if config.executor_fn:
                try:
                    order_params = {
                        "symbol": config.symbol,
                        "side": config.side,
                        "quantity": display_qty,
                        "order_type": "limit",
                        "price": limit_price,
                    }
                    fill = config.executor_fn(order_params)
                    if hasattr(fill, '__await__'):
                        fill = await fill

                    # 通知监控器：切片已提交到交易所
                    if config.on_slice_submitted_fn:
                        config.on_slice_submitted_fn(sequence, display_qty, limit_price, "limit")

                    if fill:
                        filled_qty = float(fill.get("filled", 0))
                        avg_px = float(fill.get("avg_price", 0))
                        slice_entry.filled_qty = filled_qty
                        slice_entry.avg_price = avg_px
                        slice_entry.display_time = time.time() - display_start

                        total_filled += filled_qty
                        total_cost += filled_qty * avg_px
                        slice_entry.status = "filled" if filled_qty >= display_qty * 0.9 else "partial"

                        # 回调
                        es = ExecutionSlice(
                            slice_id=slice_entry.slice_id,
                            sequence=sequence,
                            quantity=display_qty,
                            filled_quantity=filled_qty,
                            avg_fill_price=avg_px,
                            order_type="limit",
                            status=slice_entry.status,
                            submitted_at=datetime.now(),
                            filled_at=datetime.now(),
                        )
                        if on_slice_filled:
                            await on_slice_filled(config.order_id, es)
                    else:
                        slice_entry.status = "rejected"

                except Exception as e:
                    logger.warning(f"Iceberg slice {sequence} failed: {e}")
                    slice_entry.status = "error"
                    es = ExecutionSlice(
                        slice_id=slice_entry.slice_id, sequence=sequence,
                        quantity=display_qty, status="rejected",
                        error_message=str(e),
                    )
                    if on_slice_filled:
                        await on_slice_filled(config.order_id, es)

            slices_log.append(slice_entry)
            sequence += 1

            # 刷新延迟
            delay = self.compute_refresh_delay()
            await asyncio.sleep(delay)

        # 填充结果
        result.filled_quantity = total_filled
        result.fill_rate = total_filled / max(total_qty, 1e-10)
        result.avg_execution_price = total_cost / max(total_filled, 1e-10)
        result.target_price = result.avg_execution_price
        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - start_time).total_seconds()

        if result.arrival_price > 0 and result.avg_execution_price > 0:
            slip = (result.avg_execution_price - result.arrival_price) / result.arrival_price * 10000
            if config.side == "sell":
                slip *= -1
            result.arrival_slippage_bps = slip

        logger.info(f"Iceberg {config.order_id} complete: "
                    f"filled={total_filled:.4f}/{total_qty:.4f} "
                    f"refreshes={sequence} avg_px={result.avg_execution_price:.4f}")

        return result

    def get_status(self) -> Dict[str, Any]:
        return {
            "display_ratio": self._default_display_ratio,
            "randomize_display": self._randomize_display,
            "randomize_delay": self._randomize_delay,
            "limit_offset_bps": self._limit_offset_bps,
            "auto_market_pct": self._auto_market_pct,
        }
