"""
投资组合再平衡器 (Portfolio Rebalancer)

完整再平衡流水线：
  - 漂移检测 (Drift Detection) — 阈值带触发
  - 交易生成 (Trade Generation) — 净额抵消、批量分组、成本估算
  - 执行调度 (Execution Scheduling) — 优先级排序、TWAP分块、市场择时
  - 历史记录 (Rebalance History) — 有效性跟踪、频率分析、成本累计
  - 审计日志 (Audit Log) — 不可篡改的哈希链决策记录

依赖: numpy, loguru, asyncio (Python stdlib)
"""
import asyncio
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from core.direction_unifier import DirectionUnifier


def _safe_float(value: Any, default: float = 0.0) -> float:
    """将任意输入安全转换为有限浮点数；None/NaN/Inf/非法值回退 default。"""
    try:
        if value is None:
            return default
        v = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return v


# ═══════════════════════════════════════════════════════════════
# 枚举
# ═══════════════════════════════════════════════════════════════

class BandType(Enum):
    RELATIVE = "relative"
    ABSOLUTE = "absolute"
    ADAPTIVE = "adaptive"
    COMPOSITE = "composite"


class Direction(Enum):
    OVERWEIGHT = "overweight"
    UNDERWEIGHT = "underweight"
    BALANCED = "balanced"


class ActionType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class RebalanceStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionMode(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

@dataclass
class DriftStatus:
    """单个策略的漂移状态"""
    strategy_name: str
    current_weight: float
    target_weight: float
    drift: float               # current - target
    drift_pct: float           # relative drift: (current - target) / target
    direction: str             # overweight / underweight / balanced
    action: str                # buy / sell / hold
    trade_amount_usdt: float
    priority: int              # 1=critical, 2=high, 3=medium, 4=low
    reason: str


@dataclass
class RebalanceEvent:
    """单次再平衡事件记录"""
    event_id: str
    timestamp: float
    trigger_reason: str
    trigger_type: str           # relative / absolute / adaptive / composite
    pre_weights: Dict[str, float]
    target_weights: Dict[str, float]
    post_weights: Dict[str, float]
    trades: List[Dict[str, Any]]
    total_cost_usdt: float
    total_benefit_usdt: float
    status: str                 # completed / partial / failed / cancelled
    drift_severity: float
    execution_duration_seconds: float
    notes: str = ""


@dataclass
class AuditEntry:
    """审计日志条目"""
    entry_id: str
    timestamp: float
    trigger: str
    pre_weights: Dict[str, float]
    post_weights: Dict[str, float]
    trades: List[Dict[str, Any]]
    total_cost: float
    total_benefit: float
    hash_prev: str
    hash_current: str
    drift_severity: float
    execution_mode: str


# ═══════════════════════════════════════════════════════════════
# 2. RebalanceBand — 阈值带
# ═══════════════════════════════════════════════════════════════

class RebalanceBand:
    """
    阈值带系统：决定何时触发再平衡。

    支持四种模式：
      - relative:   |current - target| / target > threshold
      - absolute:   |current - target| > absolute_threshold
      - adaptive:   波动率越高，带越宽（容忍更大漂移）
      - composite:  max(relative, absolute)
    含迟滞 (hysteresis)：触发后需回落到中点以下才解除。
    """

    def __init__(
        self,
        band_type: BandType = BandType.COMPOSITE,
        relative_threshold: float = 0.20,
        absolute_threshold: float = 0.05,
        adaptive_base: float = 0.15,
        adaptive_sensitivity: float = 2.0,
        hysteresis_enabled: bool = True,
    ):
        self.band_type = band_type
        self.relative_threshold = relative_threshold
        self.absolute_threshold = absolute_threshold
        self.adaptive_base = adaptive_base
        self.adaptive_sensitivity = adaptive_sensitivity
        self.hysteresis_enabled = hysteresis_enabled

        # 每个策略的触发状态：True = 已触发（处于带外）
        self._triggered: Dict[str, bool] = {}

    def is_breached(
        self,
        strategy_name: str,
        current_weight: float,
        target_weight: float,
        market_volatility: float = 0.02,
    ) -> Tuple[bool, float, str]:
        """
        检查单个策略是否突破阈值带。

        Returns:
            (breached, effective_threshold, band_type_used)
        """
        current_weight = _safe_float(current_weight, 0.0)
        target_weight = _safe_float(target_weight, 0.0)
        market_volatility = _safe_float(market_volatility, 0.02)

        effective = self._effective_threshold(strategy_name, target_weight, market_volatility)
        drift = abs(current_weight - target_weight)

        if target_weight > 0:
            drift_rel = drift / target_weight
        else:
            drift_rel = drift if drift > 0 else 0.0

        breached = False
        band_used = self.band_type.value

        if self.band_type == BandType.RELATIVE:
            breached = drift_rel > self.relative_threshold
            band_used = "relative"
        elif self.band_type == BandType.ABSOLUTE:
            breached = drift > self.absolute_threshold
            band_used = "absolute"
        elif self.band_type == BandType.ADAPTIVE:
            breached = drift_rel > effective
            band_used = "adaptive"
        elif self.band_type == BandType.COMPOSITE:
            rel_breach = drift_rel > self.relative_threshold
            abs_breach = drift > self.absolute_threshold
            breached = rel_breach or abs_breach
            band_used = "composite"

        # 迟滞逻辑
        if self.hysteresis_enabled:
            breached = self._apply_hysteresis(strategy_name, breached, current_weight, target_weight)

        return breached, effective, band_used

    def _effective_threshold(
        self,
        strategy_name: str,
        target_weight: float,
        market_volatility: float,
    ) -> float:
        """计算自适应有效阈值"""
        if self.band_type == BandType.ADAPTIVE:
            return self.adaptive_base + self.adaptive_sensitivity * market_volatility
        elif self.band_type == BandType.COMPOSITE:
            adaptive = self.adaptive_base + self.adaptive_sensitivity * market_volatility
            return min(self.relative_threshold, max(adaptive, self.absolute_threshold / max(target_weight, 1e-8)))
        return self.relative_threshold

    def _apply_hysteresis(
        self,
        strategy_name: str,
        breached: bool,
        current_weight: float,
        target_weight: float,
    ) -> bool:
        """
        迟滞：一旦触发，需 drift 回落到阈值 * 0.5 以下才解除。
        """
        previously_triggered = self._triggered.get(strategy_name, False)
        if target_weight <= 0:
            midpoint = 0.0
        else:
            midpoint = target_weight * 0.5

        drift = abs(current_weight - target_weight)

        if not previously_triggered and breached:
            self._triggered[strategy_name] = True
            return True

        if previously_triggered:
            if drift < midpoint:
                self._triggered[strategy_name] = False
                return False
            return True

        return False

    def reset_trigger(self, strategy_name: str) -> None:
        self._triggered.pop(strategy_name, None)

    def reset_all(self) -> None:
        self._triggered.clear()


# ═══════════════════════════════════════════════════════════════
# 3. RebalanceTriggerDetector — 触发检测
# ═══════════════════════════════════════════════════════════════

class RebalanceTriggerDetector:
    """
    检测投资组合漂移：

    - 阈值带突破检测（逐策略 + 聚合）
    - 组合级严重度评分
    - 优先级排序（drift * 策略重要性）
    - 最小交易规模过滤
    - 触发冷却期（cooldown）
    """

    def __init__(
        self,
        band: RebalanceBand,
        config: Dict[str, Any],
    ):
        self.band = band
        self._min_trade_size_usdt = config.get("min_trade_size_usdt", 20.0)
        self._cooldown_seconds = config.get("cooldown_seconds", 300.0)
        self._importance_map: Dict[str, float] = config.get("strategy_importance", {})
        self._last_trigger_time: float = 0.0

    async def detect(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        market_volatility: float = 0.02,
    ) -> Dict[str, Any]:
        """
        全面检测。

        Returns:
            dict with:
              - needs_rebalance: bool
              - drifts: List[DriftStatus]
              - severity: float (0~1)
              - breached_count: int
              - total_strategies: int
              - on_cooldown: bool
              - cooldown_remaining: float
              - priority_queue: List[str] (strategy names sorted)
        """
        now = time.time()
        cooldown_remaining = max(0.0, self._cooldown_seconds - (now - self._last_trigger_time))
        on_cooldown = cooldown_remaining > 0

        all_strategies = set(list(current_weights.keys()) + list(target_weights.keys()))
        if not all_strategies:
            return {
                "needs_rebalance": False,
                "drifts": [],
                "severity": 0.0,
                "breached_count": 0,
                "total_strategies": 0,
                "on_cooldown": on_cooldown,
                "cooldown_remaining": cooldown_remaining,
                "priority_queue": [],
            }

        drifts: List[DriftStatus] = []
        breached_drifts: List[DriftStatus] = []

        for name in all_strategies:
            cur = _safe_float(current_weights.get(name, 0.0), 0.0)
            tgt = _safe_float(target_weights.get(name, 0.0), 0.0)
            drift_val = cur - tgt
            drift_rel = (drift_val / tgt) if abs(tgt) > 1e-10 else 0.0

            breached, effective_threshold, band_used = self.band.is_breached(
                name, cur, tgt, market_volatility
            )

            if breached:
                direction = Direction.OVERWEIGHT.value if drift_val > 0 else Direction.UNDERWEIGHT.value
                action = ActionType.SELL.value if drift_val > 0 else ActionType.BUY.value
            else:
                if abs(drift_rel) < 0.01:
                    direction = Direction.BALANCED.value
                    action = ActionType.HOLD.value
                elif drift_val > 0:
                    direction = Direction.OVERWEIGHT.value
                    action = ActionType.HOLD.value
                else:
                    direction = Direction.UNDERWEIGHT.value
                    action = ActionType.HOLD.value

            # 优先级：critical(1), high(2), medium(3), low(4)
            priority = self._compute_priority(name, drift_val, cur, tgt)

            # 交易金额（需配合总权益才能计算，此处放占位）
            trade_amount = abs(drift_val)  # 比例值，后续乘以 total_equity

            reason = self._build_reason(name, breached, drift_val, drift_rel, effective_threshold, band_used)

            ds = DriftStatus(
                strategy_name=name,
                current_weight=cur,
                target_weight=tgt,
                drift=drift_val,
                drift_pct=drift_rel,
                direction=direction,
                action=action,
                trade_amount_usdt=trade_amount,
                priority=priority,
                reason=reason,
            )
            drifts.append(ds)
            if breached:
                breached_drifts.append(ds)

        severity = self._compute_severity(drifts)
        breached_count = len(breached_drifts)
        needs_rebalance = (not on_cooldown) and breached_count > 0

        # 按优先级 + drift 排序
        breached_drifts.sort(key=lambda d: (d.priority, -abs(d.drift)))
        priority_queue = [d.strategy_name for d in breached_drifts]

        return {
            "needs_rebalance": needs_rebalance,
            "drifts": drifts,
            "severity": severity,
            "breached_count": breached_count,
            "total_strategies": len(all_strategies),
            "on_cooldown": on_cooldown,
            "cooldown_remaining": cooldown_remaining,
            "priority_queue": priority_queue,
        }

    def _compute_priority(
        self,
        name: str,
        drift_val: float,
        current_weight: float,
        target_weight: float,
    ) -> int:
        """计算策略再平衡优先级 1~4。"""
        magnitude = abs(_safe_float(drift_val, 0.0))
        importance = _safe_float(self._importance_map.get(name, 1.0), 1.0)

        score = magnitude * importance

        if score > 0.08:
            return 1
        elif score > 0.04:
            return 2
        elif score > 0.015:
            return 3
        return 4

    def _build_reason(
        self,
        name: str,
        breached: bool,
        drift_val: float,
        drift_rel: float,
        effective_threshold: float,
        band_used: str,
    ) -> str:
        if not breached:
            return f"{name}: drift {drift_val:.4f} within band ({band_used})"
        return (
            f"{name}: drift {drift_val:.4f} ({drift_rel*100:.1f}%) "
            f"exceeds {band_used} threshold {effective_threshold:.4f}"
        )

    def _compute_severity(self, drifts: List[DriftStatus]) -> float:
        """组合级漂移严重度 0~1。"""
        if not drifts:
            return 0.0

        breached = [d for d in drifts if d.action != ActionType.HOLD.value]
        if not breached:
            return 0.0

        total_drift = sum(abs(d.drift) for d in breached)
        n = len(breached)
        severity = (total_drift / max(n, 1)) * math.sqrt(n / max(len(drifts), 1))
        return min(1.0, severity * 5.0)

    def mark_triggered(self) -> None:
        self._last_trigger_time = time.time()

    def reset_cooldown(self) -> None:
        self._last_trigger_time = 0.0

    def filter_minimum_trade_size(
        self,
        trades: List[Dict[str, Any]],
        total_equity: float,
    ) -> List[Dict[str, Any]]:
        """过滤低于最小交易规模的交易。"""
        filtered = []
        for t in trades:
            amount = _safe_float(t.get("amount_usdt", t.get("amount", 0.0)), 0.0)
            amount = abs(amount)
            if amount >= self._min_trade_size_usdt:
                filtered.append(t)
            else:
                logger.debug(f"Skip {t.get('strategy','?')}: amount {amount:.2f} < min {self._min_trade_size_usdt}")
        return filtered


# ═══════════════════════════════════════════════════════════════
# 4. TradeListGenerator — 交易清单生成
# ═══════════════════════════════════════════════════════════════

class TradeListGenerator:
    """
    生成再平衡交易清单：

    - 基于漂移状态生成买卖
    - 同一资产对冲净额抵消
    - 最小交易量取整
    - 成本估算（滑点 + 手续费）
    - 成本收益分析
    - 批量分组执行
    """

    def __init__(
        self,
        config: Dict[str, Any],
    ):
        self._lot_size_map: Dict[str, float] = config.get("lot_sizes", {})
        self._fee_rate = config.get("fee_rate", 0.001)          # 0.1%
        self._slippage_bps = config.get("slippage_bps", 5.0)    # 5 bps
        self._cost_benefit_multiplier = config.get("cost_benefit_multiplier", 1.5)
        self._max_batch_size = config.get("max_batch_size", 10)

    async def generate(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        total_equity: float,
        prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        生成再平衡交易计划。

        Returns:
            {
                "trades": [...],
                "net_trades": [...],  # 净额抵消后
                "batches": [[...], ...],
                "estimated_cost": float,
                "estimated_benefit": float,
                "cost_benefit_ratio": float,
                "should_execute": bool,
                "rejected_trades": [...],
            }
        """
        prices = prices or {}
        total_equity = _safe_float(total_equity, 0.0)
        if total_equity <= 0:
            logger.warning("TradeListGenerator: total_equity <= 0, cannot generate rebalance plan")
            return {
                "trades": [],
                "net_trades": [],
                "batches": [],
                "estimated_cost": 0.0,
                "estimated_benefit": 0.0,
                "cost_benefit_ratio": 0.0,
                "should_execute": False,
                "rejected_trades": [],
            }
        raw_trades = self._generate_raw_trades(current_weights, target_weights, total_equity, prices)
        net_trades = self._net_off_trades(raw_trades)
        rounded_trades = self._round_to_lot_sizes(net_trades)
        cost_est = await self._estimate_cost(rounded_trades, prices)
        benefit_est = self._estimate_benefit(rounded_trades, current_weights, target_weights, total_equity)
        accepted_trades, rejected_trades = self._cost_benefit_filter(
            rounded_trades, cost_est, benefit_est
        )
        batches = self._create_batches(accepted_trades)

        total_cost = sum(t.get("estimated_cost", 0.0) for t in accepted_trades)
        total_benefit = sum(t.get("estimated_benefit", 0.0) for t in accepted_trades)

        should_execute = len(accepted_trades) > 0 and total_benefit > total_cost * self._cost_benefit_multiplier

        return {
            "trades": raw_trades,
            "net_trades": rounded_trades,
            "batches": batches,
            "estimated_cost": total_cost,
            "estimated_benefit": total_benefit,
            "cost_benefit_ratio": total_benefit / max(total_cost, 1e-8),
            "should_execute": should_execute,
            "rejected_trades": rejected_trades,
        }

    def _generate_raw_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        total_equity: float,
        prices: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """生成原始交易（未抵扣）。"""
        all_strategies = set(list(current_weights.keys()) + list(target_weights.keys()))
        trades = []
        # 策略名到资产的映射：简单起见，策略名即资产名；可从 prices keys 推断
        for name in sorted(all_strategies):
            cur = _safe_float(current_weights.get(name, 0.0), 0.0)
            tgt = _safe_float(target_weights.get(name, 0.0), 0.0)
            drift = cur - tgt
            if abs(drift) < 1e-8:
                continue
            amount_usdt = abs(drift) * total_equity
            side = "sell" if drift > 0 else "buy"
            price = _safe_float(prices.get(name), 1.0)
            quantity = amount_usdt / price if price > 0 else 0.0

            trades.append({
                "strategy": name,
                "side": side,
                "amount_usdt": amount_usdt,
                "price": price,
                "quantity": quantity,
                "current_weight": cur,
                "target_weight": tgt,
                "drift": drift,
                "asset": name,
                "reduce_only": side == "sell",
            })
        return trades

    def _net_off_trades(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对冲抵消：同一资产的 buy/sell 净额至单向交易。
        """
        grouped: Dict[str, Dict[str, Any]] = {}

        for t in trades:
            asset = t["asset"]
            if asset not in grouped:
                grouped[asset] = {
                    "buy_amount": 0.0,
                    "sell_amount": 0.0,
                    "buy_quantity": 0.0,
                    "sell_quantity": 0.0,
                    "price": t["price"],
                    "asset": asset,
                    "strategies_buy": [],
                    "strategies_sell": [],
                }
            g = grouped[asset]
            if t["side"] == "buy":
                g["buy_amount"] += t["amount_usdt"]
                g["buy_quantity"] += t["quantity"]
                g["strategies_buy"].append(t["strategy"])
            else:
                g["sell_amount"] += t["amount_usdt"]
                g["sell_quantity"] += t["quantity"]
                g["strategies_sell"].append(t["strategy"])
            g["price"] = max(g["price"], t["price"])  # 保守取高价

        net_trades = []
        for asset, g in grouped.items():
            net_amount = g["buy_amount"] - g["sell_amount"]
            if abs(net_amount) < 1e-8:
                logger.debug(f"Asset {asset}: fully offset, no net trade needed")
                continue
            side = "buy" if net_amount > 0 else "sell"
            net_quantity = abs(net_amount) / g["price"] if g["price"] > 0 else 0.0

            net_trades.append({
                "asset": asset,
                "side": side,
                "amount_usdt": abs(net_amount),
                "quantity": net_quantity,
                "price": g["price"],
                "strategies_buy": g["strategies_buy"],
                "strategies_sell": g["strategies_sell"],
                "net_amount": net_amount,
                "reduce_only": side == "sell",
            })
        return net_trades

    def _round_to_lot_sizes(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按最小交易单位取整。"""
        for t in trades:
            asset = t["asset"]
            lot = self._lot_size_map.get(asset, 0.0)
            if lot > 0:
                qty = t["quantity"]
                rounded = math.floor(qty / lot) * lot
                t["quantity"] = rounded
                t["amount_usdt"] = rounded * t["price"]
                t["rounded"] = True
            else:
                t["rounded"] = False
        return trades

    async def _estimate_cost(
        self,
        trades: List[Dict[str, Any]],
        prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """估算交易成本（手续费 + 滑点）。"""
        total_fee = 0.0
        total_slippage = 0.0
        per_trade = {}

        for t in trades:
            amount = t["amount_usdt"]
            fee = amount * self._fee_rate
            slippage = amount * (self._slippage_bps / 10000.0)
            cost = fee + slippage
            total_fee += fee
            total_slippage += slippage
            t["estimated_fee"] = fee
            t["estimated_slippage"] = slippage
            t["estimated_cost"] = cost
            per_trade[t["asset"]] = cost

        return {
            "total_fee": total_fee,
            "total_slippage": total_slippage,
            "total_cost": total_fee + total_slippage,
            "per_trade": per_trade,
        }

    def _estimate_benefit(
        self,
        trades: List[Dict[str, Any]],
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        total_equity: float,
    ) -> Dict[str, float]:
        """估算再平衡收益（降低跟踪误差的价值）。"""
        pre_error = self._tracking_error(current_weights, target_weights)
        # 模拟执行后权重
        simulated = dict(current_weights)
        for t in trades:
            asset = t["asset"]
            drift = t.get("net_amount", 0.0)
            simulated[asset] = simulated.get(asset, 0.0) - (drift / total_equity if total_equity > 0 else 0.0)
        post_error = self._tracking_error(simulated, target_weights)
        error_reduction = max(0.0, pre_error - post_error)

        # 简化为：tracking_error 降低带来的名义收益
        benefit_total = error_reduction * total_equity * 0.01  # 1% per unit error reduction
        per_trade = {}
        for t in trades:
            drift_val = t.get("drift")
            if drift_val is None:
                drift_val = t.get("net_amount", 0.0) / total_equity if total_equity > 0 else 0.0
            share = abs(_safe_float(drift_val, 0.0))
            per_trade[t["asset"]] = benefit_total * share / max(len(trades), 1)
            t["estimated_benefit"] = per_trade[t["asset"]]

        return {
            "tracking_error_before": pre_error,
            "tracking_error_after": post_error,
            "error_reduction": error_reduction,
            "total_benefit": benefit_total,
            "per_trade": per_trade,
        }

    @staticmethod
    def _tracking_error(current: Dict[str, float], target: Dict[str, float]) -> float:
        """计算平均绝对漂移。"""
        keys = set(list(current.keys()) + list(target.keys()))
        if not keys:
            return 0.0
        total = sum(
            abs(_safe_float(current.get(k, 0.0), 0.0) - _safe_float(target.get(k, 0.0), 0.0))
            for k in keys
        )
        return total / len(keys)

    def _cost_benefit_filter(
        self,
        trades: List[Dict[str, Any]],
        cost_est: Dict[str, float],
        benefit_est: Dict[str, float],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """成本收益过滤：收益 > 成本 * multiplier 才执行。"""
        accepted = []
        rejected = []
        for t in trades:
            cost = t.get("estimated_cost", 0.0)
            benefit = t.get("estimated_benefit", 0.0)
            multiplier = self._cost_benefit_multiplier
            if benefit > cost * multiplier:
                accepted.append(t)
                t["cost_benefit_passed"] = True
            else:
                rejected.append(t)
                t["cost_benefit_passed"] = False
                logger.debug(
                    f"Reject {t['asset']}: benefit {benefit:.4f} <= cost {cost:.4f} * {multiplier}"
                )
        return accepted, rejected

    def _create_batches(self, trades: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """按最大批次大小分组。"""
        batches = []
        for i in range(0, len(trades), self._max_batch_size):
            batch = trades[i:i + self._max_batch_size]
            if batch:
                batches.append(batch)
        return batches


# ═══════════════════════════════════════════════════════════════
# 5. ExecutionScheduler — 执行调度
# ═══════════════════════════════════════════════════════════════

class ExecutionScheduler:
    """
    交易执行调度器：

    - 优先级排序
    - 顺序/并行执行决策
    - 市场择时：高波动期避免再平衡
    - TWAP分块：大额拆小
    - 进度追踪
    - 部分成交处理
    """

    def __init__(
        self,
        config: Dict[str, Any],
        executor: Optional[Callable] = None,
    ):
        self._executor = executor
        self._max_chunk_size_pct = config.get("max_chunk_size_pct", 0.05)
        self._execution_priority = config.get("execution_priority", "drift_descending")
        self._max_parallel = config.get("max_parallel_trades", 3)
        self._volatility_threshold = config.get("execution_volatility_threshold", 0.05)
        self._chunk_delay_seconds = config.get("chunk_delay_seconds", 1.0)

        # 追踪状态
        self._execution_progress: Dict[str, Dict[str, Any]] = {}
        self._is_executing = False

    async def schedule(
        self,
        plan: Dict[str, Any],
        total_equity: float,
        market_volatility: float = 0.02,
    ) -> Dict[str, Any]:
        """调度执行。"""
        if self._is_executing:
            return {"status": "rejected", "reason": "already executing", "results": []}

        if market_volatility > self._volatility_threshold:
            logger.warning(f"Market volatility {market_volatility:.4f} exceeds threshold, deferring rebalance")
            return {"status": "deferred", "reason": "high_volatility", "volatility": market_volatility, "results": []}

        trades = plan.get("net_trades", plan.get("trades", []))
        if not trades:
            return {"status": "skipped", "reason": "no_trades", "results": []}

        sorted_trades = self._sort_by_priority(trades)
        chunks = self._create_chunks(sorted_trades, total_equity)

        self._is_executing = True
        self._init_progress(sorted_trades)

        try:
            results = await self._execute_chunks(chunks)
        finally:
            self._is_executing = False

        final_status = self._aggregate_status(results)
        return {
            "status": final_status,
            "results": results,
            "executed_count": len(results),
            "total_count": len(sorted_trades),
        }

    def _sort_by_priority(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self._execution_priority == "drift_descending":
            return sorted(trades, key=lambda t: -abs(t.get("drift", t.get("amount_usdt", 0.0))))
        if self._execution_priority == "cost_ascending":
            return sorted(trades, key=lambda t: t.get("estimated_cost", float("inf")))
        if self._execution_priority == "value_descending":
            return sorted(trades, key=lambda t: -t.get("amount_usdt", 0.0))
        return trades

    def _create_chunks(
        self,
        trades: List[Dict[str, Any]],
        total_equity: float,
    ) -> List[Dict[str, Any]]:
        """对大额交易拆分为 TWAP 小块。"""
        chunks = []
        max_chunk_usdt = total_equity * self._max_chunk_size_pct

        for t in trades:
            amount = t.get("amount_usdt", 0.0)
            if amount <= max_chunk_usdt or max_chunk_usdt <= 0:
                chunks.append(dict(t, chunk_index=1, total_chunks=1))
            else:
                num_chunks = max(1, math.ceil(amount / max_chunk_usdt))
                chunk_amount = amount / num_chunks
                chunk_qty = t.get("quantity", 0.0) / num_chunks if t.get("quantity") else 0.0
                for i in range(num_chunks):
                    chunks.append({
                        **t,
                        "amount_usdt": chunk_amount,
                        "quantity": chunk_qty,
                        "chunk_index": i + 1,
                        "total_chunks": num_chunks,
                    })
        return chunks

    def _init_progress(self, trades: List[Dict[str, Any]]) -> None:
        """初始化执行进度追踪。"""
        self._execution_progress = {}
        for i, t in enumerate(trades):
            key = t.get("asset", f"trade_{i}")
            self._execution_progress[key] = {
                "status": "pending",
                "filled_amount": 0.0,
                "total_amount": t.get("amount_usdt", 0.0),
                "attempts": 0,
                "error": None,
            }

    async def _execute_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行分块交易。"""
        results = []

        # 决定执行模式
        total_usdt = sum(c.get("amount_usdt", 0.0) for c in chunks)
        total_trades = len(set(c.get("asset", "?") for c in chunks))
        mode = ExecutionMode.SEQUENTIAL if total_trades <= 2 else ExecutionMode.PARALLEL

        logger.info(f"Executing {len(chunks)} chunks across {total_trades} assets, mode={mode.value}")

        if mode == ExecutionMode.SEQUENTIAL:
            for chunk in chunks:
                result = await self._execute_single(chunk)
                results.append(result)
                await asyncio.sleep(self._chunk_delay_seconds)
        else:
            # 并行执行，但限流
            semaphore = asyncio.Semaphore(self._max_parallel)

            async def _bounded_execute(chunk: Dict) -> Dict:
                async with semaphore:
                    return await self._execute_single(chunk)

            tasks = [_bounded_execute(c) for c in chunks]
            results = await asyncio.gather(*tasks)

        return results

    async def _execute_single(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """执行单笔交易。"""
        asset = trade.get("asset", "unknown")
        side = trade.get("side", "buy")
        amount = trade.get("amount_usdt", 0.0)

        result = {
            "asset": asset,
            "side": side,
            "requested_amount": amount,
            "filled_amount": 0.0,
            "status": "failed",
            "error": None,
        }

        # 更新进度
        key = asset
        if key in self._execution_progress:
            self._execution_progress[key]["status"] = "in_progress"
            self._execution_progress[key]["attempts"] += 1

        if self._executor is not None:
            try:
                exec_result = self._executor(trade)
                if asyncio.iscoroutine(exec_result):
                    exec_result = await exec_result
                if not isinstance(exec_result, dict):
                    exec_result = {}
                filled = _safe_float(exec_result.get("filled", 0.0), 0.0)
                result["filled_amount"] = filled
                result["status"] = "filled" if filled > 0 else "failed"
                result["execution_details"] = exec_result
            except Exception as e:
                logger.error(f"Execute {asset} {side} failed: {e}")
                result["error"] = str(e)
                result["status"] = "failed"
        else:
            # 模拟执行
            fill_ratio = np.random.normal(0.98, 0.02)
            fill_ratio = min(1.0, max(0.0, fill_ratio))
            amount = _safe_float(amount, 0.0)
            result["filled_amount"] = amount * fill_ratio
            result["status"] = "filled" if fill_ratio > 0.5 else "partial"

        # 更新进度
        if key in self._execution_progress:
            self._execution_progress[key]["filled_amount"] = result["filled_amount"]
            self._execution_progress[key]["status"] = result["status"]
            self._execution_progress[key]["error"] = result.get("error")

        return result

    def _aggregate_status(self, results: List[Dict[str, Any]]) -> str:
        filled = sum(1 for r in results if r["status"] == "filled")
        total = len(results)
        if total == 0:
            return RebalanceStatus.COMPLETED.value
        if filled == total:
            return RebalanceStatus.COMPLETED.value
        if filled > 0:
            return RebalanceStatus.PARTIAL.value
        return RebalanceStatus.FAILED.value

    def get_progress(self) -> Dict[str, Any]:
        return {
            "is_executing": self._is_executing,
            "progress": dict(self._execution_progress),
        }


# ═══════════════════════════════════════════════════════════════
# 6. RebalanceHistory — 历史记录
# ═══════════════════════════════════════════════════════════════

class RebalanceHistory:
    """
    再平衡历史记录：

    - 事件存储（timestamp, trigger, trades, costs, pre/post weights）
    - 有效性追踪：再平衡后权重是否接近目标
    - 频率分析
    - 累计成本
    - 成功率
    """

    def __init__(self, max_size: int = 10000):
        self._events: List[RebalanceEvent] = []
        self._max_size = max_size
        self._lock = asyncio.Lock()

    async def record(self, event: RebalanceEvent) -> None:
        async with self._lock:
            # 幂等去重：同一 event_id 不重复记录
            if any(e.event_id == event.event_id for e in self._events):
                logger.debug(f"Rebalance event {event.event_id} already recorded, skip")
                return
            self._events.append(event)
            if len(self._events) > self._max_size:
                self._events = self._events[-self._max_size:]

    async def get_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        async with self._lock:
            recent = self._events[-limit:] if len(self._events) > limit else self._events
            return [self._event_to_dict(e) for e in recent]

    async def get_effectiveness(self) -> Dict[str, Any]:
        """追踪再平衡有效性。"""
        async with self._lock:
            completed = [e for e in self._events if e.status in ("completed", "partial")]
            if not completed:
                return {"avg_post_drift": 0.0, "success_rate": 0.0, "mean_reversion_speed": 0.0, "count": 0}

            post_drifts = []
            for e in completed:
                for name, post_w in e.post_weights.items():
                    tgt = e.target_weights.get(name, post_w)
                    if abs(tgt) > 1e-8:
                        post_drifts.append(abs(post_w - tgt))

            avg_post_drift = np.mean(post_drifts) if post_drifts else 0.0
            success_rate = sum(1 for d in post_drifts if d < 0.01) / max(len(post_drifts), 1)

            return {
                "avg_post_drift": float(avg_post_drift),
                "success_rate": float(success_rate),
                "mean_reversion_speed": 0.0,  # 需要更多数据点
                "count": len(completed),
            }

    async def get_frequency_analysis(self, window_days: int = 30) -> Dict[str, Any]:
        """频率分析。"""
        async with self._lock:
            now = time.time()
            cutoff = now - window_days * 86400
            recent = [e for e in self._events if e.timestamp >= cutoff]

            if not recent:
                return {"events_in_window": 0, "avg_interval_hours": 0.0, "events_per_day": 0.0}

            events_in_window = len(recent)
            events_per_day = events_in_window / max(window_days, 1)

            intervals = []
            for i in range(1, len(recent)):
                intervals.append(recent[i].timestamp - recent[i - 1].timestamp)
            avg_interval_hours = np.mean(intervals) / 3600.0 if intervals else 0.0

            return {
                "events_in_window": events_in_window,
                "avg_interval_hours": float(avg_interval_hours),
                "events_per_day": float(events_per_day),
            }

    async def get_cumulative_costs(self) -> Dict[str, float]:
        """累计再平衡成本。"""
        async with self._lock:
            total_cost = sum(e.total_cost_usdt for e in self._events)
            total_benefit = sum(e.total_benefit_usdt for e in self._events)
            return {
                "cumulative_cost": total_cost,
                "cumulative_benefit": total_benefit,
                "net_pnl": total_benefit - total_cost,
                "event_count": len(self._events),
            }

    async def get_statistics(self) -> Dict[str, Any]:
        """综合统计。"""
        effectiveness = await self.get_effectiveness()
        frequency = await self.get_frequency_analysis()
        costs = await self.get_cumulative_costs()
        return {
            **effectiveness,
            **frequency,
            **costs,
        }

    @staticmethod
    def _event_to_dict(event: RebalanceEvent) -> Dict[str, Any]:
        return {
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "trigger_reason": event.trigger_reason,
            "trigger_type": event.trigger_type,
            "pre_weights": event.pre_weights,
            "target_weights": event.target_weights,
            "post_weights": event.post_weights,
            "trades": event.trades,
            "total_cost_usdt": event.total_cost_usdt,
            "total_benefit_usdt": event.total_benefit_usdt,
            "status": event.status,
            "drift_severity": event.drift_severity,
            "execution_duration_seconds": event.execution_duration_seconds,
            "notes": event.notes,
        }


# ═══════════════════════════════════════════════════════════════
# 8. RebalanceAuditLogger — 审计日志（哈希链防篡改）
# ═══════════════════════════════════════════════════════════════

class RebalanceAuditLogger:
    """
    不可篡改的再平衡审计日志。

    哈希链：每个条目包含：
      - 上一跳 hash (hash_prev)
      - 当前内容 hash (hash_current)
      破坏任一节点将导致整链验证失败。
    """

    def __init__(self):
        self._entries: List[AuditEntry] = []
        self._lock = asyncio.Lock()
        self._entry_counter: int = 0

    async def log(
        self,
        trigger: str,
        pre_weights: Dict[str, float],
        post_weights: Dict[str, float],
        trades: List[Dict[str, Any]],
        total_cost: float,
        total_benefit: float,
        drift_severity: float,
        execution_mode: str = "unknown",
    ) -> AuditEntry:
        """追加一条审计条目，返回生成的条目。"""
        async with self._lock:
            self._entry_counter += 1
            entry_id = f"audit_{self._entry_counter:06d}"
            ts = time.time()

            prev_hash = ""
            if self._entries:
                prev_hash = self._entries[-1].hash_current

            # 构建内容哈希
            content_str = json.dumps({
                "entry_id": entry_id,
                "timestamp": ts,
                "trigger": trigger,
                "pre_weights": pre_weights,
                "post_weights": post_weights,
                "trades": trades,
                "cost": total_cost,
                "benefit": total_benefit,
                "severity": drift_severity,
                "hash_prev": prev_hash,
            }, sort_keys=True, ensure_ascii=False)

            hash_current = hashlib.sha256(content_str.encode()).hexdigest()

            entry = AuditEntry(
                entry_id=entry_id,
                timestamp=ts,
                trigger=trigger,
                pre_weights=pre_weights,
                post_weights=post_weights,
                trades=trades,
                total_cost=total_cost,
                total_benefit=total_benefit,
                hash_prev=prev_hash,
                hash_current=hash_current,
                drift_severity=drift_severity,
                execution_mode=execution_mode,
            )
            self._entries.append(entry)
            logger.debug(f"Audit entry {entry_id} logged, hash={hash_current[:12]}...")
            return entry

    async def verify(self) -> Dict[str, Any]:
        """验证整条哈希链的完整性。"""
        async with self._lock:
            results = {
                "valid": True,
                "total_entries": len(self._entries),
                "broken_at": None,
                "invalid_entries": [],
            }

            for i in range(1, len(self._entries)):
                prev_entry = self._entries[i - 1]
                curr_entry = self._entries[i]

                if curr_entry.hash_prev != prev_entry.hash_current:
                    results["valid"] = False
                    results["broken_at"] = i
                    results["invalid_entries"].append(curr_entry.entry_id)

                # 重新计算当前条目 hash
                content_str = json.dumps({
                    "entry_id": curr_entry.entry_id,
                    "timestamp": curr_entry.timestamp,
                    "trigger": curr_entry.trigger,
                    "pre_weights": curr_entry.pre_weights,
                    "post_weights": curr_entry.post_weights,
                    "trades": curr_entry.trades,
                    "cost": curr_entry.total_cost,
                    "benefit": curr_entry.total_benefit,
                    "severity": curr_entry.drift_severity,
                    "hash_prev": curr_entry.hash_prev,
                }, sort_keys=True, ensure_ascii=False)

                recomputed = hashlib.sha256(content_str.encode()).hexdigest()
                if recomputed != curr_entry.hash_current:
                    results["valid"] = False
                    results["invalid_entries"].append(curr_entry.entry_id)
                    if results["broken_at"] is None:
                        results["broken_at"] = i

            return results

    async def export_json(self, filepath: str) -> None:
        """导出审计日志到 JSON 文件。"""
        async with self._lock:
            data = []
            for e in self._entries:
                dt_str = None
                ts = _safe_float(e.timestamp, 0.0)
                try:
                    dt_str = datetime.fromtimestamp(ts).isoformat()
                except (OSError, ValueError, OverflowError):
                    dt_str = None
                data.append({
                    "entry_id": e.entry_id,
                    "timestamp": ts,
                    "datetime": dt_str,
                    "trigger": e.trigger,
                    "pre_weights": e.pre_weights,
                    "post_weights": e.post_weights,
                    "trades": e.trades,
                    "total_cost": e.total_cost,
                    "total_benefit": e.total_benefit,
                    "hash_prev": e.hash_prev,
                    "hash_current": e.hash_current,
                    "drift_severity": e.drift_severity,
                    "execution_mode": e.execution_mode,
                })
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Audit log exported to {filepath}, {len(data)} entries")

    async def get_entries(self) -> List[Dict[str, Any]]:
        """获取所有审计条目。"""
        async with self._lock:
            return [
                {
                    "entry_id": e.entry_id,
                    "timestamp": e.timestamp,
                    "trigger": e.trigger,
                    "hash_current": e.hash_current,
                    "total_cost": e.total_cost,
                    "total_benefit": e.total_benefit,
                }
                for e in self._entries
            ]

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()
            self._entry_counter = 0


# ═══════════════════════════════════════════════════════════════
# 7. PortfolioRebalancer — 主类
# ═══════════════════════════════════════════════════════════════

class PortfolioRebalancer:
    """
    投资组合再平衡器 — 顶层编排器。

    集成:
      - RebalanceBand (阈值带)
      - RebalanceTriggerDetector (触发检测)
      - TradeListGenerator (交易生成)
      - ExecutionScheduler (执行调度)
      - RebalanceHistory (历史记录)
      - RebalanceAuditLogger (审计日志)

    Usage:
        rebalancer = PortfolioRebalancer(config)
        await rebalancer.start()

        detection = await rebalancer.check_rebalance_needed(current_weights, target_weights)
        if detection["needs_rebalance"]:
            plan = await rebalancer.generate_rebalance_plan(current_weights, target_weights, total_equity)
            result = await rebalancer.execute_rebalance(plan)
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        rebalancer_cfg = config.get("portfolio_rebalancer", {})

        # 阈值带配置
        self._band = RebalanceBand(
            band_type=BandType("composite"),
            relative_threshold=rebalancer_cfg.get("relative_band_threshold", 0.20),
            absolute_threshold=rebalancer_cfg.get("absolute_band_threshold", 0.05),
            adaptive_base=rebalancer_cfg.get("adaptive_band_base", 0.15),
            adaptive_sensitivity=rebalancer_cfg.get("adaptive_band_sensitivity", 2.0),
            hysteresis_enabled=rebalancer_cfg.get("hysteresis_enabled", True),
        )

        # 检测器
        detector_cfg = {
            "min_trade_size_usdt": rebalancer_cfg.get("min_trade_size_usdt", 20.0),
            "cooldown_seconds": rebalancer_cfg.get("cooldown_seconds", 300.0),
            "strategy_importance": rebalancer_cfg.get("strategy_importance", {}),
        }
        self._detector = RebalanceTriggerDetector(self._band, detector_cfg)

        # 交易生成器
        generator_cfg = {
            "lot_sizes": rebalancer_cfg.get("lot_sizes", {}),
            "fee_rate": rebalancer_cfg.get("fee_rate", 0.001),
            "slippage_bps": rebalancer_cfg.get("slippage_bps", 5.0),
            "cost_benefit_multiplier": rebalancer_cfg.get("cost_benefit_multiplier", 1.5),
            "max_batch_size": rebalancer_cfg.get("max_batch_size", 10),
        }
        self._generator = TradeListGenerator(generator_cfg)

        # 执行调度器
        scheduler_cfg = {
            "max_chunk_size_pct": rebalancer_cfg.get("max_chunk_size_pct", 0.05),
            "execution_priority": rebalancer_cfg.get("execution_priority", "drift_descending"),
            "max_parallel_trades": rebalancer_cfg.get("max_parallel_trades", 3),
            "execution_volatility_threshold": rebalancer_cfg.get("execution_volatility_threshold", 0.05),
            "chunk_delay_seconds": rebalancer_cfg.get("chunk_delay_seconds", 1.0),
        }
        self._scheduler = ExecutionScheduler(scheduler_cfg)

        # 历史记录 & 审计
        self._history = RebalanceHistory(max_size=rebalancer_cfg.get("history_max_size", 10000))
        self._audit = RebalanceAuditLogger()

        # 状态
        self._lock = asyncio.Lock()
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None
        self._auto_interval = rebalancer_cfg.get("auto_check_interval", 60.0)
        self._target_weights: Dict[str, float] = {}
        self._total_equity: float = 0.0
        self._event_counter: int = 0
        self._data_dir = rebalancer_cfg.get("data_dir", "./data")

        # 重启恢复：加载持久化的再平衡目标与权益
        self._load_state()

        # P1: 组合总敞口硬限制 —— 多策略并发开仓时防止总敞口超限
        # max_total_exposure_pct: 总敞口占权益的最大倍数（默认 3.0 = 300%，即最大 3x 杠杆）
        self._max_total_exposure_pct = _safe_float(
            rebalancer_cfg.get("max_total_exposure_pct", 3.0), 3.0
        )
        self._current_gross_exposure: float = 0.0

    async def start(self) -> None:
        if self._running:
            return
        logger.info("PortfolioRebalancer starting")
        self._running = True
        if self._auto_interval > 0 and (self._loop_task is None or self._loop_task.done()):
            self._loop_task = asyncio.create_task(self._auto_loop())

    async def stop(self) -> None:
        if not self._running and (self._loop_task is None or self._loop_task.done()):
            return
        logger.info("PortfolioRebalancer stopping")
        self._running = False
        if self._loop_task is not None and not self._loop_task.done():
            self._loop_task.cancel()
        if self._loop_task is not None:
            await asyncio.gather(self._loop_task, return_exceptions=True)
        self._loop_task = None

    async def _auto_loop(self) -> None:
        """自动定期检查循环。"""
        while self._running:
            try:
                if self._target_weights and self._total_equity > 0:
                    current = await self._fetch_current_weights()
                    detection = await self.check_rebalance_needed(current, self._target_weights)
                    if detection.get("needs_rebalance"):
                        logger.info(f"Auto-detect: rebalance needed, severity={detection.get('severity', 0):.4f}")
                        plan = await self.generate_rebalance_plan(
                            current, self._target_weights, self._total_equity
                        )
                        if plan.get("should_execute"):
                            await self.execute_rebalance(plan)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Auto loop error: {e}")
            try:
                await asyncio.sleep(self._auto_interval)
            except asyncio.CancelledError:
                raise

    async def _fetch_current_weights(self) -> Dict[str, float]:
        """获取当前权重（可被子类重写以对接真实数据源）。"""
        return {}

    # ── 公共 API ────────────────────────────────────────────

    async def check_rebalance_needed(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        market_volatility: float = 0.02,
    ) -> Dict[str, Any]:
        """检查是否需要再平衡。"""
        async with self._lock:
            result = await self._detector.detect(current_weights, target_weights, market_volatility)
            logger.info(
                f"Rebalance check: needed={result['needs_rebalance']}, "
                f"breached={result['breached_count']}/{result['total_strategies']}, "
                f"severity={result['severity']:.4f}"
            )
            return result

    async def generate_rebalance_plan(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        total_equity: float,
        prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """生成再平衡计划。"""
        async with self._lock:
            plan = await self._generator.generate(current_weights, target_weights, total_equity, prices)
            logger.info(
                f"Rebalance plan: {len(plan.get('net_trades', []))} net trades, "
                f"est_cost={plan['estimated_cost']:.4f}, est_benefit={plan['estimated_benefit']:.4f}, "
                f"should_execute={plan['should_execute']}"
            )
            # 存储供 auto loop 使用
            self._target_weights = {
                k: _safe_float(v, 0.0) for k, v in target_weights.items()
            }
            self._total_equity = _safe_float(total_equity, 0.0)
            self._save_state()
            return plan

    async def execute_rebalance(
        self,
        plan: Dict[str, Any],
        executor: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """执行再平衡。"""
        pre_weights = {}
        if self._target_weights:
            pre_weights = await self._fetch_current_weights() or {}

        exec_start = time.time()

        async with self._lock:
            self._detector.mark_triggered()

            # 用提供的 executor 覆盖默认
            if executor is not None:
                self._scheduler._executor = executor

            result = await self._scheduler.schedule(
                plan,
                self._total_equity,
            )
            exec_duration = time.time() - exec_start

            # 获取执行后的权重（近似）
            post_weights = self._compute_post_weights(
                pre_weights or {},
                result.get("results", []),
            )

            # 记录历史
            trades_list = plan.get("net_trades", plan.get("trades", []))
            self._event_counter += 1
            event = RebalanceEvent(
                event_id=f"evt_{int(time.time()*1000)}_{self._event_counter}",
                timestamp=time.time(),
                trigger_reason=f"breached {len(trades_list)} strategies",
                trigger_type=self._band.band_type.value,
                pre_weights=pre_weights,
                target_weights=self._target_weights,
                post_weights=post_weights,
                trades=trades_list,
                total_cost_usdt=plan.get("estimated_cost", 0.0),
                total_benefit_usdt=plan.get("estimated_benefit", 0.0),
                status=result.get("status", "unknown"),
                drift_severity=0.0,
                execution_duration_seconds=exec_duration,
            )
            await self._history.record(event)

            # 审计日志
            await self._audit.log(
                trigger=f"breached {len(trades_list)} strategies",
                pre_weights=pre_weights,
                post_weights=post_weights,
                trades=trades_list,
                total_cost=plan.get("estimated_cost", 0.0),
                total_benefit=plan.get("estimated_benefit", 0.0),
                drift_severity=0.0,
                execution_mode="twap" if plan.get("batches") else "direct",
            )

            logger.info(f"Rebalance executed: status={result['status']}, duration={exec_duration:.2f}s")
            self._save_state()
            return result

    async def estimate_rebalance_cost(
        self,
        trades: List[Dict[str, Any]],
        market_conditions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """独立估算再平衡成本。"""
        fee_rate = self._config.get("portfolio_rebalancer", {}).get("fee_rate", 0.001)
        slippage_bps = self._config.get("portfolio_rebalancer", {}).get("slippage_bps", 5.0)

        if market_conditions:
            slippage_bps = market_conditions.get("slippage_override", slippage_bps)

        total_fee = 0.0
        total_slippage = 0.0

        for t in trades:
            amount = t.get("amount_usdt", t.get("amount", 0.0))
            total_fee += amount * fee_rate
            total_slippage += amount * (slippage_bps / 10000.0)

        return {
            "estimated_fee": total_fee,
            "estimated_slippage": total_slippage,
            "total_cost": total_fee + total_slippage,
            "fee_rate": fee_rate,
            "slippage_bps": slippage_bps,
        }

    async def get_rebalance_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return await self._history.get_events(limit)

    async def check_tax_efficiency(self, proposed_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        税务效率检查（简化版）。
        实际应用中可根据持有时长判断短期/长期资本利得税率。

        Returns:
            {
                "tax_impact_usdt": 估计税务影响,
                "short_term_trades": int,
                "long_term_trades": int,
                "warnings": [...],
            }
        """
        short_term = 0
        long_term = 0
        tax_impact = 0.0
        warnings = []

        for t in proposed_trades:
            side = t.get("side", "")
            if side != "sell":
                continue
            amount = t.get("amount_usdt", t.get("amount", 0.0))
            holding_hours = t.get("holding_hours", 24)

            if holding_hours < 24:
                short_term += 1
                tax_rate = 0.15  # 假设 15% 短期资本利得
                tax_impact += amount * 0.5 * tax_rate  # 假设 50% 为利润
            else:
                long_term += 1
                tax_rate = 0.10
                tax_impact += amount * 0.5 * tax_rate

        if short_term > 3:
            warnings.append(f"High short-term trade count ({short_term}), consider deferring some sales")

        return {
            "tax_impact_usdt": tax_impact,
            "short_term_trades": short_term,
            "long_term_trades": long_term,
            "warnings": warnings,
        }

    def get_summary(self) -> Dict[str, Any]:
        """获取再平衡器运行摘要。"""
        return {
            "running": self._running,
            "band_type": self._band.band_type.value,
            "relative_threshold": self._band.relative_threshold,
            "absolute_threshold": self._band.absolute_threshold,
            "hysteresis_enabled": self._band.hysteresis_enabled,
            "min_trade_size_usdt": self._detector._min_trade_size_usdt,
            "cooldown_seconds": self._detector._cooldown_seconds,
            "cost_benefit_multiplier": self._generator._cost_benefit_multiplier,
            "max_chunk_size_pct": self._scheduler._max_chunk_size_pct,
            "execution_priority": self._scheduler._execution_priority,
            "auto_interval": self._auto_interval,
            "target_weights_tracked": bool(self._target_weights),
            "total_equity_tracked": self._total_equity,
        }

    @staticmethod
    def _compute_post_weights(
        pre_weights: Dict[str, float],
        results: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """根据执行结果估算事后权重。"""
        post = dict(pre_weights)
        for r in results:
            asset = r.get("asset", "")
            filled = _safe_float(r.get("filled_amount", 0.0), 0.0)
            raw_side = r.get("side", "buy")
            try:
                side = DirectionUnifier.to_side(str(raw_side))
            except (ValueError, TypeError):
                side = "buy"
            if asset not in post:
                post[asset] = 0.0
            # 简化：方向性调整
            if side == "buy":
                post[asset] = _safe_float(post.get(asset, 0.0), 0.0) + (filled / 10000.0)  # 归一化近似
            else:
                post[asset] = max(0.0, _safe_float(post.get(asset, 0.0), 0.0) - (filled / 10000.0))

        # 归一化
        total = sum(post.values())
        if total > 0:
            post = {k: v / total for k, v in post.items()}
        return post

    # ── 状态持久化 ────────────────────────────────────────

    def _save_state(self) -> None:
        """持久化再平衡目标权重与总权益，重启后可恢复。"""
        try:
            os.makedirs(self._data_dir, exist_ok=True)
            state_path = os.path.join(self._data_dir, "portfolio_rebalancer_state.json")
            state = {
                "last_updated": datetime.now().isoformat(),
                "target_weights": {k: _safe_float(v, 0.0) for k, v in self._target_weights.items()},
                "total_equity": _safe_float(self._total_equity, 0.0),
            }
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            logger.debug(f"Portfolio rebalancer state saved to {state_path}")
        except Exception as e:
            logger.warning(f"Failed to save portfolio rebalancer state: {e}")

    def _load_state(self) -> bool:
        """加载持久化的再平衡状态；任何异常都不影响启动。"""
        try:
            state_path = os.path.join(self._data_dir, "portfolio_rebalancer_state.json")
            if not os.path.exists(state_path):
                return False
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            raw_weights = state.get("target_weights", {})
            if isinstance(raw_weights, dict):
                self._target_weights = {
                    str(k): _safe_float(v, 0.0) for k, v in raw_weights.items()
                }
            self._total_equity = _safe_float(state.get("total_equity"), 0.0)
            logger.info(f"Portfolio rebalancer state restored: {len(self._target_weights)} targets, equity={self._total_equity}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load portfolio rebalancer state: {e}")
            return False

    # ── P1: 总敞口硬限制 ────────────────────────────────────

    def update_gross_exposure(self, gross_notional: float) -> None:
        """更新当前总名义敞口（由调度器/持仓管理器定期同步）。"""
        self._current_gross_exposure = _safe_float(gross_notional, 0.0)

    def check_exposure_limit(
        self, additional_notional: float, equity: float = 0.0,
    ) -> Dict[str, Any]:
        """检查新增敞口是否会突破组合总敞口硬限制。

        Returns:
            {"allowed": bool, "current_exposure": float, "max_exposure": float,
             "projected_exposure": float, "headroom": float, "reason": str}
        """
        eq = _safe_float(equity, self._total_equity)
        max_exposure = eq * self._max_total_exposure_pct
        current = self._current_gross_exposure
        additional = _safe_float(additional_notional, 0.0)
        projected = current + additional
        headroom = max_exposure - current

        if max_exposure <= 0:
            return {
                "allowed": True,
                "current_exposure": current,
                "max_exposure": max_exposure,
                "projected_exposure": projected,
                "headroom": headroom,
                "reason": "no_equity_no_limit",
            }

        if projected > max_exposure:
            return {
                "allowed": False,
                "current_exposure": current,
                "max_exposure": max_exposure,
                "projected_exposure": projected,
                "headroom": headroom,
                "reason": f"exposure_limit_exceeded: projected {projected:.2f} > max {max_exposure:.2f}",
            }

        return {
            "allowed": True,
            "current_exposure": current,
            "max_exposure": max_exposure,
            "projected_exposure": projected,
            "headroom": headroom,
            "reason": "within_limit",
        }

    # ── 便捷方法 ──────────────────────────────────────────

    async def update_target_weights(self, target_weights: Dict[str, float]) -> None:
        """更新目标权重。"""
        async with self._lock:
            self._target_weights = {
                k: _safe_float(v, 0.0) for k, v in target_weights.items()
            }
            self._save_state()
            logger.info(f"Target weights updated: {len(self._target_weights)} strategies")

    async def update_total_equity(self, total_equity: float) -> None:
        """更新总权益。"""
        async with self._lock:
            self._total_equity = _safe_float(total_equity, 0.0)
            self._save_state()

    async def reset_cooldown(self) -> None:
        """重置冷却期。"""
        async with self._lock:
            self._detector.reset_cooldown()
            logger.info("Rebalance cooldown reset")

    async def verify_audit_chain(self) -> Dict[str, Any]:
        """验证审计日志哈希链。"""
        return await self._audit.verify()

    async def export_audit_log(self, filepath: str) -> None:
        """导出审计日志。"""
        await self._audit.export_json(filepath)
