"""
多层级风控拦截层系统
====================
核心定位：所有交易信号必须逐层校验，任意一层拦截直接驳回下单，不可逆

五层串行校验：
  L1 事前风控 — 账户余额、可用保证金、单笔上限、单币种仓位上限、全局杠杆上限、单日最大亏损
  L2 事中风控 — API频率限流、滑点阈值、市价/限价价差、网络延迟超时
  L3 持仓实时风控 — 爆仓价格预警、浮动亏损阶梯减仓、资金费率大额亏损减仓
  L4 全局单日风控 — 单日最大亏损硬限制、单日最大开仓次数、连续亏损降杠杆/暂停
  L5 紧急熔断风控 — 交易所异常、网络断开、程序崩溃、极端插针，一键全平+停止策略
"""

import asyncio
import threading
import time
import json
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from loguru import logger

# P0: 统一持仓方向处理 - 杜绝方向转换不一致
from core.direction_unifier import DirectionUnifier
# P0: 持久化全局 Kill Switch（fail-closed，仅禁开仓、平仓穿透）
from core.kill_switch import KillSwitch
from utils.helpers import safe_float, safe_finite, safe_div


# ============================================================================
# 枚举与数据类
# ============================================================================

class RiskLayer(Enum):
    """风控层级"""
    L0_KILL_SWITCH = "L0_kill_switch"
    L1_PRE_TRADE = "L1_pre_trade"
    L2_IN_TRADE = "L2_in_trade"
    L3_POSITION = "L3_position"
    L4_DAILY = "L4_daily"
    L5_EMERGENCY = "L5_emergency"


class RiskAction(Enum):
    """风控动作"""
    PASS = "pass"
    REJECT = "reject"
    REDUCE = "reduce"
    CLOSE_ALL = "close_all"
    PAUSE = "pause"
    FREEZE = "freeze"


@dataclass
class RiskCheckResult:
    """单层风控检查结果"""
    layer: RiskLayer
    passed: bool
    action: RiskAction
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer.value,
            "passed": self.passed,
            "action": self.action.value,
            "reason": self.reason,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class RiskGateResult:
    """五层风控总结果"""
    passed: bool
    action: RiskAction
    blocked_layer: Optional[RiskLayer]
    results: List[RiskCheckResult] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "action": self.action.value,
            "blocked_layer": self.blocked_layer.value if self.blocked_layer else None,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary
        }


# ============================================================================
# L1: 事前风控（下单前校验）
# ============================================================================

class PreTradeRiskChecker:
    """
    事前风控：下单前逐项校验
    
    检查项：
    1. 账户余额是否足够
    2. 可用保证金是否足够
    3. 单笔最大下单金额
    4. 单币种总仓位上限
    5. 全局总杠杆上限
    6. 单日最大亏损阈值
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        trading = self.config.get("trading", {})
        risk = self.config.get("risk", {})
        
        self._max_single_order_usd = risk.get("max_single_order_usd") or trading.get("max_single_order_usd") or 500.0
        self._max_symbol_position_ratio = risk.get("max_symbol_position_ratio") or 0.25
        self._max_total_leverage = trading.get("max_total_leverage") or 20
        self._daily_max_loss = trading.get("daily_max_loss") or 0.04
        self._min_balance = trading.get("min_balance") or 5.0
        
        self._symbol_positions: Dict[str, float] = {}
        self._daily_pnl: float = 0.0
        self._daily_start_equity: float = 0.0
        self._current_equity: float = 0.0
        self._available_margin: float = 0.0
        self._lock = threading.RLock()

    def update_account(self, equity: float, available_margin: float,
                       daily_pnl: float, daily_start_equity: float) -> None:
        """更新账户状态"""
        with self._lock:
            self._current_equity = equity
            self._available_margin = available_margin
            self._daily_pnl = daily_pnl
            self._daily_start_equity = daily_start_equity

    def update_symbol_position(self, symbol: str, position_value: float) -> None:
        """更新币种持仓价值"""
        with self._lock:
            self._symbol_positions[symbol] = position_value

    def remove_symbol_position(self, symbol: str) -> None:
        """移除币种持仓记录（用于平仓后清理）"""
        with self._lock:
            self._symbol_positions.pop(symbol, None)

    def sync_positions_from_exchange(self, positions_map: Dict[str, float]) -> None:
        """全量同步持仓价值（替换而非累积，防止残留数据）"""
        with self._lock:
            self._symbol_positions = dict(positions_map)

    def check(self, signal: Dict[str, Any]) -> RiskCheckResult:
        """执行事前风控检查"""
        details = {}
        
        with self._lock:
            equity = self._current_equity
            available_margin = self._available_margin
            daily_pnl = self._daily_pnl
            daily_start = self._daily_start_equity

        # 识别平仓信号：平仓信号不触发仓位限制检查（仓位在减少，不是增加）
        is_close_signal = signal.get("reduce_only", False)
        if not is_close_signal:
            sig_type = str(signal.get("signal_type", "")).lower()
            close_signal_types = ["close", "exit", "reduce", "stop_loss", "take_profit", "trailing", "tp", "sl", "liquidation"]
            is_close_signal = any(st in sig_type for st in close_signal_types)

        # 1. 账户余额检查
        if equity < self._min_balance:
            return RiskCheckResult(
                RiskLayer.L1_PRE_TRADE, False, RiskAction.REJECT,
                f"账户余额不足: {equity:.2f} < {self._min_balance}",
                {"equity": equity, "min_balance": self._min_balance}
            )
        details["equity"] = round(equity, 2)

        # 2. 可用保证金检查
        # P0 fail-closed：缺价格/数量时 signal.get 默认 0 会让 order_value/margin_required
        # 恒为 0，从而绕过保证金、单笔上限、仓位上限等所有后续校验。非法订单必须显式拒绝。
        try:
            order_qty = safe_float(signal.get("quantity"), 0.0)
            order_price = safe_float(signal.get("price"), 0.0)
        except (TypeError, ValueError):
            order_qty = 0.0
            order_price = 0.0
        if order_qty <= 0 or order_price <= 0:
            return RiskCheckResult(
                RiskLayer.L1_PRE_TRADE, False, RiskAction.REJECT,
                f"订单数量或价格非法: quantity={order_qty}, price={order_price}",
                {"quantity": order_qty, "price": order_price}
            )

        order_value = order_qty * order_price
        leverage = safe_float(signal.get("leverage"), 1.0)
        margin_required = safe_div(order_value, leverage, order_value) if leverage > 0 else order_value
        
        if margin_required > available_margin:
            return RiskCheckResult(
                RiskLayer.L1_PRE_TRADE, False, RiskAction.REJECT,
                f"可用保证金不足: 需要{margin_required:.2f}, 可用{available_margin:.2f}",
                {"margin_required": round(margin_required, 2),
                 "available_margin": round(available_margin, 2)}
            )
        details["margin_required"] = round(margin_required, 2)

        # 3. 单笔最大下单金额
        if order_value > self._max_single_order_usd:
            return RiskCheckResult(
                RiskLayer.L1_PRE_TRADE, False, RiskAction.REJECT,
                f"单笔下单金额超限: {order_value:.2f} > {self._max_single_order_usd}",
                {"order_value": round(order_value, 2),
                 "max_single_order": self._max_single_order_usd}
            )
        details["order_value"] = round(order_value, 2)

        # 4. 单币种总仓位上限（仅对开仓信号检查，平仓信号跳过）
        symbol = signal.get("symbol", "")
        current_pos = self._symbol_positions.get(symbol, 0)
        if not is_close_signal:
            new_total = current_pos + margin_required
            max_symbol_value = equity * self._max_symbol_position_ratio
            
            if new_total > max_symbol_value:
                return RiskCheckResult(
                    RiskLayer.L1_PRE_TRADE, False, RiskAction.REJECT,
                    f"单币种仓位超限: {symbol} 总仓位{new_total:.2f} > 上限{max_symbol_value:.2f}",
                    {"symbol": symbol, "current_pos": round(current_pos, 2),
                     "new_total": round(new_total, 2),
                     "max_symbol": round(max_symbol_value, 2)}
                )
            details["symbol_position"] = round(new_total, 2)
        else:
            details["symbol_position"] = round(current_pos, 2)

        # 5. 全局总杠杆上限（仅对开仓信号检查，平仓信号跳过）
        if not is_close_signal:
            # _symbol_positions存储保证金，乘以杠杆得到名义价值
            existing_margin = sum(self._symbol_positions.values())
            # 统一使用5x杠杆估算（大部分策略使用5x）
            avg_leverage = max(leverage, 3)  # 至少3x
            total_position_value = existing_margin * avg_leverage + order_value
            total_leverage = safe_div(total_position_value, equity, 0.0) if equity > 0 else 0
            
            if total_leverage > self._max_total_leverage:
                return RiskCheckResult(
                    RiskLayer.L1_PRE_TRADE, False, RiskAction.REJECT,
                    f"全局总杠杆超限: {total_leverage:.2f}x > {self._max_total_leverage}x",
                    {"total_leverage": round(total_leverage, 2),
                     "max_leverage": self._max_total_leverage}
                )
            details["total_leverage"] = round(total_leverage, 2)
        else:
            total_position_value = sum(self._symbol_positions.values())
            total_leverage = safe_div(total_position_value, equity, 0.0) if equity > 0 else 0
            details["total_leverage"] = round(total_leverage, 2)

        # 6. 单日最大亏损阈值
        if daily_start > 0:
            daily_loss_pct = safe_div(-daily_pnl, daily_start, 0.0) if daily_pnl < 0 else 0
            if daily_loss_pct >= self._daily_max_loss:
                return RiskCheckResult(
                    RiskLayer.L1_PRE_TRADE, False, RiskAction.REJECT,
                    f"单日亏损超限: {daily_loss_pct*100:.2f}% >= {self._daily_max_loss*100:.2f}%",
                    {"daily_pnl": round(daily_pnl, 2),
                     "daily_loss_pct": round(daily_loss_pct, 4)}
                )
            details["daily_loss_pct"] = round(daily_loss_pct, 4)

        return RiskCheckResult(
            RiskLayer.L1_PRE_TRADE, True, RiskAction.PASS,
            "事前风控全部通过", details
        )


# ============================================================================
# L2: 事中风控（下单传输校验）
# ============================================================================

class InTradeRiskChecker:
    """
    事中风控：下单传输过程校验
    
    检查项：
    1. API请求频率限流
    2. 滑点阈值拦截
    3. 市价/限价价差校验
    4. 网络延迟超时拦截
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        risk = self.config.get("risk", {})
        
        self._max_api_rps = risk.get("max_api_rps") or 10
        self._max_slippage_pct = risk.get("max_slippage_pct") or 0.003
        self._max_spread_pct = risk.get("max_spread_pct") or 0.002
        self._max_latency_ms = risk.get("max_latency_ms") or 3000
        
        self._api_call_times: deque = deque(maxlen=100)
        self._latency_history: deque = deque(maxlen=20)
        self._lock = threading.RLock()

    def record_api_call(self) -> None:
        """记录API调用时间"""
        with self._lock:
            self._api_call_times.append(time.time())

    def record_latency(self, latency_ms: float) -> None:
        """记录网络延迟"""
        with self._lock:
            self._latency_history.append(latency_ms)

    def check(self, signal: Dict[str, Any], market_data: Dict[str, Any] = None) -> RiskCheckResult:
        """执行事中风控检查"""
        details = {}
        now = time.time()

        # 1. API请求频率限流
        with self._lock:
            recent_calls = [t for t in self._api_call_times if now - t < 1.0]
            current_rps = len(recent_calls)
        
        if current_rps >= self._max_api_rps:
            return RiskCheckResult(
                RiskLayer.L2_IN_TRADE, False, RiskAction.REJECT,
                f"API频率超限: {current_rps} rps > {self._max_api_rps} rps",
                {"current_rps": current_rps, "max_rps": self._max_api_rps}
            )
        details["api_rps"] = current_rps

        # 2. 网络延迟超时拦截
        with self._lock:
            if self._latency_history:
                avg_latency = safe_finite(sum(self._latency_history) / len(self._latency_history), 0.0)
            else:
                avg_latency = 0.0
        
        if avg_latency > self._max_latency_ms:
            return RiskCheckResult(
                RiskLayer.L2_IN_TRADE, False, RiskAction.REJECT,
                f"网络延迟超限: {avg_latency:.0f}ms > {self._max_latency_ms}ms",
                {"avg_latency_ms": round(avg_latency, 0),
                 "max_latency_ms": self._max_latency_ms}
            )
        details["avg_latency_ms"] = round(avg_latency, 0)

        # 3. 滑点阈值拦截
        if market_data:
            expected_price = safe_float(signal.get("price"), 0.0)
            best_bid = safe_float(market_data.get("bid_price"), 0.0)
            best_ask = safe_float(market_data.get("ask_price"), 0.0)
            
            if expected_price > 0 and best_ask > 0:
                slippage = safe_div(abs(expected_price - best_ask), best_ask, 0.0)
                if slippage > self._max_slippage_pct:
                    return RiskCheckResult(
                        RiskLayer.L2_IN_TRADE, False, RiskAction.REJECT,
                        f"滑点超限: {slippage*100:.3f}% > {self._max_slippage_pct*100:.3f}%",
                        {"expected_price": expected_price,
                         "best_ask": best_ask,
                         "slippage_pct": round(slippage, 6)}
                    )
                details["slippage_pct"] = round(slippage, 6)

            # 4. 市价/限价价差校验
            if best_bid > 0 and best_ask > 0:
                spread_pct = safe_div(best_ask - best_bid, best_bid, 0.0)
                if spread_pct > self._max_spread_pct:
                    return RiskCheckResult(
                        RiskLayer.L2_IN_TRADE, False, RiskAction.REJECT,
                        f"买卖价差过大: {spread_pct*100:.3f}% > {self._max_spread_pct*100:.3f}%",
                        {"spread_pct": round(spread_pct, 6),
                         "max_spread_pct": self._max_spread_pct}
                    )
                details["spread_pct"] = round(spread_pct, 6)

        return RiskCheckResult(
            RiskLayer.L2_IN_TRADE, True, RiskAction.PASS,
            "事中风控全部通过", details
        )


# ============================================================================
# L3: 持仓实时风控（持仓持续监控）
# ============================================================================

class PositionRiskChecker:
    """
    持仓实时风控：持仓持续监控
    
    检查项：
    1. 实时爆仓价格预警
    2. 浮动亏损阶梯减仓触发
    3. 资金费率大额亏损自动减仓
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        risk = self.config.get("risk", {})
        
        self._liquidation_warning_pct = risk.get("liquidation_warning_pct") or 0.10
        self._liquidation_critical_pct = risk.get("liquidation_critical_pct") or 0.05
        
        self._loss_tier1_pct = risk.get("position_loss_tier1_pct") or 0.05
        self._loss_tier2_pct = risk.get("position_loss_tier2_pct") or 0.10
        self._loss_tier3_pct = risk.get("position_loss_tier3_pct") or 0.15
        
        self._loss_tier1_reduce = risk.get("loss_tier1_reduce_ratio") or 0.30
        self._loss_tier2_reduce = risk.get("loss_tier2_reduce_ratio") or 0.50
        self._loss_tier3_reduce = risk.get("loss_tier3_reduce_ratio") or 1.00
        
        self._max_funding_loss_pct = risk.get("max_funding_loss_pct") or 0.005
        
        self._position_states: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        # 手动开单白名单（symbol 集合）：这些持仓由用户手动管理，L3 不自动减仓/平仓
        self._manual_override_symbols: set = set()

    def set_manual_override_symbols(self, symbols) -> None:
        """注入手动开单白名单（symbol 集合）。L3 检查时跳过这些持仓，避免误自动平仓。"""
        with self._lock:
            self._manual_override_symbols = set(symbols or [])

    def update_position(self, symbol: str, entry_price: float, current_price: float,
                        size: float, leverage: float, side: str,
                        liquidation_price: float = 0, funding_rate: float = 0) -> None:
        """更新持仓状态"""
        with self._lock:
            if size == 0:
                self._position_states.pop(symbol, None)
                return
            
            unrealized_pnl = (current_price - entry_price) * size * (1 if DirectionUnifier.is_long(side) else -1)
            pnl_pct = safe_div(unrealized_pnl, entry_price * size, 0.0) if entry_price > 0 else 0.0
            
            # 计算到强平距离
            if liquidation_price > 0 and current_price > 0:
                liq_distance = safe_div(abs(current_price - liquidation_price), current_price, 0.0)
            else:
                # OKX维持保证金率约0.5-1%，强平发生在保证金率 = 维持保证金率时
                # 精确估算: liq_distance ≈ 1/leverage - 维持保证金率(取保守值1.0%)
                # 原公式 1/(leverage*2) 对10x是5%，过于保守，造成永久误判全平
                maintenance_margin = 0.01
                liq_distance = (1.0 / leverage - maintenance_margin) if leverage > 0 else 1.0
                liq_distance = max(liq_distance, 0.005)  # 兜底最低0.5%
            
            self._position_states[symbol] = {
                "entry_price": entry_price,
                "current_price": current_price,
                "size": size,
                "leverage": leverage,
                "side": side,
                "liquidation_price": liquidation_price,
                "unrealized_pnl": unrealized_pnl,
                "pnl_pct": pnl_pct,
                "liq_distance": liq_distance,
                "funding_rate": funding_rate,
                "updated_at": datetime.now()
            }

    def check(self, signal: Dict[str, Any] = None) -> List[RiskCheckResult]:
        """
        执行持仓风控检查
        
        返回列表：每个持仓的检查结果
        """
        results = []
        
        with self._lock:
            positions = dict(self._position_states)

        for symbol, pos in positions.items():
            # 手动开单白名单：用户手动管理的持仓不纳入 L3 自动减仓/平仓，
            # 避免时序上先于 orphan 登记被误当普通风险持仓自动处理。
            if symbol in self._manual_override_symbols:
                continue

            details = {"symbol": symbol}
            
            # 1. 爆仓价格预警
            liq_dist = pos["liq_distance"]
            if liq_dist < self._liquidation_critical_pct:
                results.append(RiskCheckResult(
                    RiskLayer.L3_POSITION, False, RiskAction.CLOSE_ALL,
                    f"危急: {symbol} 距强平仅{liq_dist*100:.2f}%，立即平仓",
                    {**details, "liq_distance": round(liq_dist, 4),
                     "action": "close_all"}
                ))
                continue
            elif liq_dist < self._liquidation_warning_pct:
                results.append(RiskCheckResult(
                    RiskLayer.L3_POSITION, False, RiskAction.REDUCE,
                    f"警告: {symbol} 距强平{liq_dist*100:.2f}%，减仓50%",
                    {**details, "liq_distance": round(liq_dist, 4),
                     "action": "reduce_50"}
                ))
            
            # 2. 浮动亏损阶梯减仓
            pnl_pct = pos["pnl_pct"]
            if pnl_pct < -self._loss_tier3_pct:
                results.append(RiskCheckResult(
                    RiskLayer.L3_POSITION, False, RiskAction.CLOSE_ALL,
                    f"三级亏损: {symbol} 亏损{pnl_pct*100:.2f}%，全平",
                    {**details, "pnl_pct": round(pnl_pct, 4),
                     "action": f"reduce_{int(self._loss_tier3_reduce*100)}"}
                ))
            elif pnl_pct < -self._loss_tier2_pct:
                results.append(RiskCheckResult(
                    RiskLayer.L3_POSITION, False, RiskAction.REDUCE,
                    f"二级亏损: {symbol} 亏损{pnl_pct*100:.2f}%，减仓{int(self._loss_tier2_reduce*100)}%",
                    {**details, "pnl_pct": round(pnl_pct, 4),
                     "action": f"reduce_{int(self._loss_tier2_reduce*100)}"}
                ))
            elif pnl_pct < -self._loss_tier1_pct:
                results.append(RiskCheckResult(
                    RiskLayer.L3_POSITION, False, RiskAction.REDUCE,
                    f"一级亏损: {symbol} 亏损{pnl_pct*100:.2f}%，减仓{int(self._loss_tier1_reduce*100)}%",
                    {**details, "pnl_pct": round(pnl_pct, 4),
                     "action": f"reduce_{int(self._loss_tier1_reduce*100)}"}
                ))
            
            # 3. 资金费率大额亏损 —— P0: 仅当正在支付（方向不利）时触发
            funding = pos.get("funding_rate", 0)
            side = DirectionUnifier.normalize(pos.get("side", "long"))
            is_paying_funding = (DirectionUnifier.is_long(side) and funding > 0) or (DirectionUnifier.is_short(side) and funding < 0)
            if is_paying_funding and abs(funding) > self._max_funding_loss_pct:
                results.append(RiskCheckResult(
                    RiskLayer.L3_POSITION, False, RiskAction.REDUCE,
                    f"资金费率不利: {symbol} {side}方向支付费率{funding:.4%}，减仓30%",
                    {**details, "funding_rate": round(funding, 6),
                     "action": "reduce_30", "side": side}
                ))

        if not results:
            results.append(RiskCheckResult(
                RiskLayer.L3_POSITION, True, RiskAction.PASS,
                "持仓实时风控全部通过", {"positions_checked": len(positions)}
            ))

        return results

    def get_position_summary(self) -> Dict[str, Any]:
        """获取持仓风控摘要"""
        with self._lock:
            return {
                "total_positions": len(self._position_states),
                "positions": {
                    sym: {
                        "pnl_pct": round(p["pnl_pct"], 4),
                        "liq_distance": round(p["liq_distance"], 4),
                        "leverage": p["leverage"],
                        "side": p["side"]
                    }
                    for sym, p in self._position_states.items()
                }
            }


# ============================================================================
# L4: 全局单日风控（周期风控）
# ============================================================================

class DailyRiskChecker:
    """
    全局单日风控：周期性校验
    
    检查项：
    1. 单日最大亏损硬限制
    2. 单日最大开仓次数限制
    3. 连续亏损自动降杠杆/暂停交易
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        trading = self.config.get("trading", {})
        
        self._daily_max_loss = trading.get("daily_max_loss") or 0.04
        self._daily_max_loss_hard = trading.get("daily_max_loss_hard") or 0.06
        self._daily_max_trades = trading.get("daily_max_trades") or 100
        self._max_consecutive_losses = trading.get("max_consecutive_losses") or 4
        # 极小亏损阈值：|PnL| 小于此值不计入连续亏损（避免网格策略手续费"亏损"误触发暂停）
        self._min_pnl_for_loss = trading.get("min_pnl_for_loss") or 0.001
        # 同秒批量亏损合并：N秒内的连续亏损视为同一笔交易批次
        self._loss_batch_window_sec = trading.get("loss_batch_window_sec") or 3
        # P0: 连亏暂停自动超时恢复（秒），避免"暂停→无法开仓→无法盈利→无法恢复"死锁
        # 默认30分钟：超时后自动解除连亏暂停，重新允许开仓（硬亏损暂停不自动恢复）
        self._consecutive_pause_timeout = trading.get("consecutive_pause_timeout_sec", 1800)
        
        self._daily_trade_count: Dict[str, int] = {}
        self._daily_pnl: Dict[str, float] = {}
        self._daily_start_equity: Dict[str, float] = {}
        self._consecutive_losses: int = 0
        self._consecutive_loss_history: deque = deque(maxlen=50)
        self._last_loss_record_time: float = 0  # 上次记录亏损的时间戳（秒），用于批次合并
        
        self._leverage_reduction_active = False
        self._trading_paused = False
        self._pause_reason = ""
        self._pause_started_at = 0.0  # P5: 暂停开始时间戳，用于自动恢复
        
        self._lock = threading.RLock()

    def record_trade(self) -> None:
        """记录一笔交易"""
        today = date.today().isoformat()
        with self._lock:
            self._daily_trade_count[today] = self._daily_trade_count.get(today, 0) + 1

    def record_trade_result(self, pnl: float) -> None:
        """记录交易结果
        
        智能亏损计数：
        - |PnL| < min_pnl_for_loss（默认0.001 USDT）：视为手续费级别的"零盈亏"，不计入连续亏损
        - 同 N 秒内（默认3秒）的连续亏损合并为同一批次，避免网格策略快速平仓时连亏计数暴涨
        """
        today = date.today().isoformat()
        now_ts = time.time()
        with self._lock:
            self._daily_pnl[today] = self._daily_pnl.get(today, 0) + pnl
            self._consecutive_loss_history.append({"date": today, "pnl": pnl})
            
            # 极小亏损（手续费级别）不计入连续亏损
            if pnl < 0 and abs(pnl) >= self._min_pnl_for_loss:
                # 批次合并：与上次亏损间隔 < loss_batch_window_sec 则不计为新一次连亏
                time_since_last = now_ts - self._last_loss_record_time
                if time_since_last >= self._loss_batch_window_sec:
                    self._consecutive_losses += 1
                self._last_loss_record_time = now_ts
            elif pnl >= 0:
                self._consecutive_losses = 0
                self._last_loss_record_time = 0
                # P5: 盈利后自动恢复交易（仅当暂停原因是连续亏损时）
                if self._trading_paused and "连续亏损" in self._pause_reason:
                    self._trading_paused = False
                    self._pause_reason = ""
                    self._pause_started_at = 0.0  # P5
                    self._leverage_reduction_active = False
                    logger.info("Trading resumed: profitable trade recorded, consecutive losses reset")
                elif self._leverage_reduction_active:
                    self._leverage_reduction_active = False
                    logger.info("Leverage reduction deactivated: profitable trade recorded")
            
            # 连续亏损触发降杠杆
            if self._consecutive_losses >= self._max_consecutive_losses:
                self._leverage_reduction_active = True
                logger.warning(f"连续亏损{self._consecutive_losses}次，激活降杠杆模式")
            
            # 连续亏损超过2倍阈值，暂停交易
            if self._consecutive_losses >= self._max_consecutive_losses * 2:
                self._trading_paused = True
                self._pause_started_at = time.time()  # P5: 记录暂停开始时间
                self._pause_reason = f"连续亏损{self._consecutive_losses}次，暂停交易"
                logger.error(f"交易已暂停: {self._pause_reason}")

    def update_daily_equity(self, equity: float) -> None:
        """更新每日起始权益
        
        P18: 添加最小权益阈值检查，防止系统启动时 equity 过小导致
        daily_start_equity 被设为极小值，使日亏损比例异常放大。
        同时，如果后续 equity 显著增大（>5x），自动修正 daily_start_equity。
        """
        today = date.today().isoformat()
        # P18: 最小权益阈值 - 低于此值不更新日初权益，防止误触发日亏损限制
        MIN_DAILY_START_EQUITY = 5.0  # 最少5 USDT
        with self._lock:
            if today not in self._daily_start_equity:
                if equity < MIN_DAILY_START_EQUITY:
                    logger.warning(
                        f"P18: Skipping daily_start_equity init for {today}: "
                        f"equity={equity:.2f} < min={MIN_DAILY_START_EQUITY}"
                    )
                    return
                self._daily_start_equity[today] = equity
                # 新的一天，重置每日计数器和暂停状态
                self._daily_pnl[today] = 0
                self._daily_trade_count[today] = 0
                # 跨日后连续亏损计数归零（新的交易日，新的开始）
                old_consecutive = self._consecutive_losses
                self._consecutive_losses = 0
                self._last_loss_record_time = 0
                self._trading_paused = False
                self._pause_reason = ""
                self._pause_started_at = 0.0  # P5
                self._leverage_reduction_active = False
                if old_consecutive > 0:
                    logger.info(f"New trading day: reset consecutive losses ({old_consecutive}→0), trading resumed")
            else:
                # P18: 如果当前equity比日初权益大很多（>5x），说明日初权益可能被错误设置
                # 自动修正为当前equity（保留累计PnL不变）
                current_start = self._daily_start_equity.get(today, 0)
                if current_start > 0 and equity > current_start * 5:
                    logger.warning(
                        f"P18: Correcting daily_start_equity for {today}: "
                        f"{current_start:.2f} -> {equity:.2f} (equity grew >5x, likely init error)"
                    )
                    self._daily_start_equity[today] = equity

    def check(self, signal: Dict[str, Any] = None) -> RiskCheckResult:
        """执行单日风控检查
        
        P5修复：连续亏损暂停时允许平仓信号穿透，避免死锁
        - 单日硬亏损限制暂停：阻止一切操作（包括平仓）
        - 连续亏损暂停：只阻止开仓，允许平仓/止损
        """
        today = date.today().isoformat()
        details = {}
        
        with self._lock:
            # 检查交易暂停状态
            if self._trading_paused:
                # P0: 连亏暂停自动超时恢复——避免"暂停→无法开仓→无法盈利→无法恢复"死锁
                # 仅对连续亏损暂停生效；硬亏损暂停需跨日重置或手动 reset
                is_hard_loss_pause = "单日亏损" in self._pause_reason
                if not is_hard_loss_pause and self._pause_started_at > 0:
                    elapsed = time.time() - self._pause_started_at
                    if elapsed >= self._consecutive_pause_timeout:
                        old_reason = self._pause_reason
                        self._trading_paused = False
                        self._pause_reason = ""
                        self._pause_started_at = 0.0
                        self._leverage_reduction_active = False
                        # 超时恢复时连亏计数减半（非归零：保留风险记忆但给恢复机会）
                        self._consecutive_losses = self._consecutive_losses // 2
                        logger.warning(
                            f"L4 consecutive-loss pause auto-resumed after {elapsed:.0f}s "
                            f"(timeout={self._consecutive_pause_timeout}s), "
                            f"consecutive_losses halved: {self._consecutive_losses}, "
                            f"was: {old_reason}"
                        )
                        # 超时恢复后继续执行后续检查（不 return）

                if self._trading_paused:
                    # P5: 检测是否为平仓信号
                    is_close = self._is_close_signal(signal)
                    # P5: 连续亏损暂停 vs 硬亏损暂停
                    is_hard_loss_pause = "单日亏损" in self._pause_reason

                    if is_hard_loss_pause:
                        # 硬亏损限制：绝对阻止一切操作
                        return RiskCheckResult(
                            RiskLayer.L4_DAILY, False, RiskAction.PAUSE,
                            f"交易已暂停: {self._pause_reason}",
                            {"paused": True, "reason": self._pause_reason}
                        )

                    if is_close:
                        # P5: 连续亏损暂停时允许平仓信号穿透，避免仓位锁死
                        logger.info(
                            f"L4 paused but allowing close signal: {self._pause_reason}, "
                            f"signal_type={signal.get('signal_type', 'unknown')}"
                        )
                        # 不返回，继续后续检查
                    else:
                        return RiskCheckResult(
                            RiskLayer.L4_DAILY, False, RiskAction.PAUSE,
                            f"交易已暂停: {self._pause_reason}",
                            {"paused": True, "reason": self._pause_reason}
                        )

            daily_trades = self._daily_trade_count.get(today, 0)
            daily_pnl = self._daily_pnl.get(today, 0)
            daily_start = self._daily_start_equity.get(today, 0)

        # 1. 单日最大开仓次数
        max_trades = self._daily_max_trades or 100
        if daily_trades >= max_trades:
            return RiskCheckResult(
                RiskLayer.L4_DAILY, False, RiskAction.REJECT,
                f"单日开仓次数超限: {daily_trades} >= {max_trades}",
                {"daily_trades": daily_trades, "max_trades": max_trades}
            )
        details["daily_trades"] = daily_trades

        # 2. 单日最大亏损
        # P18: 添加最小绝对亏损阈值，防止日初权益设置过小导致百分比误判
        # P18-2增强: 对小账户（daily_start < 100 USDT）使用动态阈值
        # 小账户的百分比限制过于严格（42 USDT的3%只有1.28 USDT），
        # 使用 max(0.5, daily_start * 5%) 作为最小绝对亏损阈值
        if daily_start > 0:
            daily_loss_pct = safe_div(-daily_pnl, daily_start, 0.0) if daily_pnl < 0 else 0.0
            abs_loss = abs(daily_pnl) if daily_pnl < 0 else 0
            
            # P18-2增强: 动态最小绝对亏损阈值
            # 小账户（<100 USDT）使用 daily_start * 5% 作为阈值
            # 大账户（>=100 USDT）使用固定 0.5 USDT
            if daily_start < 100.0:
                MIN_ABSOLUTE_LOSS = max(0.5, daily_start * 0.05)  # 至少5%日初权益或0.5 USDT
            else:
                MIN_ABSOLUTE_LOSS = 0.5  # 大账户使用固定阈值
            
            if daily_loss_pct >= self._daily_max_loss_hard:
                # P18: 绝对亏损过小不触发硬限制
                if abs_loss < MIN_ABSOLUTE_LOSS:
                    logger.warning(
                        f"P18: Skipping daily hard loss limit - abs_loss={abs_loss:.4f} < {MIN_ABSOLUTE_LOSS:.2f} "
                        f"(daily_loss_pct={daily_loss_pct*100:.2f}%, daily_start={daily_start:.2f})"
                    )
                else:
                    with self._lock:
                        self._trading_paused = True
                        self._pause_started_at = time.time()  # P5: 记录暂停开始时间
                        self._pause_reason = f"单日亏损{daily_loss_pct*100:.2f}%超过硬限制"
                    
                    return RiskCheckResult(
                        RiskLayer.L4_DAILY, False, RiskAction.PAUSE,
                        f"单日亏损超过硬限制: {daily_loss_pct*100:.2f}% >= {self._daily_max_loss_hard*100:.2f}%",
                        {"daily_loss_pct": round(daily_loss_pct, 4),
                         "daily_pnl": round(daily_pnl, 2),
                         "hard_limit": self._daily_max_loss_hard}
                    )
            
            if daily_loss_pct >= self._daily_max_loss:
                # P18: 绝对亏损过小不触发日亏损限制
                if abs_loss < MIN_ABSOLUTE_LOSS:
                    logger.warning(
                        f"P18: Skipping daily loss limit - abs_loss={abs_loss:.4f} < {MIN_ABSOLUTE_LOSS:.2f} "
                        f"(daily_loss_pct={daily_loss_pct*100:.2f}%, daily_start={daily_start:.2f})"
                    )
                else:
                    return RiskCheckResult(
                        RiskLayer.L4_DAILY, False, RiskAction.REJECT,
                        f"单日亏损超限: {daily_loss_pct*100:.2f}% >= {self._daily_max_loss*100:.2f}%",
                        {"daily_loss_pct": round(daily_loss_pct, 4),
                         "daily_pnl": round(daily_pnl, 2)}
                    )
            details["daily_loss_pct"] = round(daily_loss_pct, 4)

        # 3. 连续亏损降杠杆
        with self._lock:
            if self._leverage_reduction_active:
                details["leverage_reduction"] = True
                details["consecutive_losses"] = self._consecutive_losses

        return RiskCheckResult(
            RiskLayer.L4_DAILY, True, RiskAction.PASS,
            "单日风控全部通过", details
        )

    def get_leverage_multiplier(self) -> float:
        """获取当前杠杆乘数（降杠杆模式下返回0.5）"""
        with self._lock:
            if self._leverage_reduction_active:
                return 0.5
            return 1.0

    def is_paused(self) -> bool:
        """是否暂停交易"""
        with self._lock:
            return self._trading_paused

    def reset(self) -> None:
        """重置风控状态（手动恢复）"""
        with self._lock:
            self._trading_paused = False
            self._pause_reason = ""
            self._pause_started_at = 0.0  # P5
            self._leverage_reduction_active = False
            self._consecutive_losses = 0
            self._last_loss_record_time = 0
            logger.info("DailyRiskChecker state reset manually")

    @staticmethod
    def _is_close_signal(signal: Dict[str, Any] = None) -> bool:
        """P5: 检测信号是否为平仓/减仓信号
        
        平仓信号特征：
        - reduce_only=True
        - signal_type包含 close/exit/reduce/stop_loss/take_profit/trailing/tp/sl/liquidation
        """
        if not signal or not isinstance(signal, dict):
            return False
        
        if signal.get("reduce_only", False):
            return True
        
        sig_type = str(signal.get("signal_type", "")).lower()
        close_keywords = ["close", "exit", "reduce", "stop_loss", "take_profit",
                          "trailing", "tp", "sl", "liquidation"]
        return any(kw in sig_type for kw in close_keywords)

    def to_dict(self) -> Dict[str, Any]:
        today = date.today().isoformat()
        with self._lock:
            return {
                "daily_trades": self._daily_trade_count.get(today, 0),
                "daily_pnl": round(self._daily_pnl.get(today, 0), 2),
                "consecutive_losses": self._consecutive_losses,
                "leverage_reduction_active": self._leverage_reduction_active,
                "trading_paused": self._trading_paused,
                "pause_reason": self._pause_reason,
                "pause_started_at": self._pause_started_at  # P5
            }


# ============================================================================
# L5: 紧急熔断风控（终极兜底）
# ============================================================================

class EmergencyCircuitBreaker:
    """
    紧急熔断风控：终极兜底
    
    触发条件：
    1. 交易所行情异常（价格瞬间暴涨暴跌）
    2. 服务器网络断开
    3. 程序崩溃恢复
    4. 极端插针行情
    
    动作：一键全仓平仓 + 停止所有策略
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        risk = self.config.get("risk", {})
        black_swan = risk.get("black_swan", {})

        self._flash_crash_threshold = black_swan.get("emergency_full_close_pct", 0.12)
        self._flash_crash_window = black_swan.get("flash_crash_window_seconds", 300)
        self._network_timeout_threshold = risk.get("network_timeout_threshold", 120)
        self._network_warning_threshold = self._network_timeout_threshold * 0.5  # 60s 警告，120s 触发
        self._network_warning_issued = False  # 避免重复日志刷屏

        self._emergency_triggered = False
        self._emergency_reason = ""
        self._trigger_time: Optional[datetime] = None
        self._cooldown_seconds = risk.get("emergency_cooldown_seconds", 3600)
        # P4: 网络断连熔断冷却时间（远短于全局冷却，因为网络恢复后应快速解除）
        self._network_disconnect_cooldown = risk.get("network_disconnect_cooldown_seconds", 300)
        # P0: 连续触发升级计数器，防止"触发→冷却→解除→瞬间再触发"振铃循环
        self._emergency_trigger_count = 0
        self._max_escalation_multiplier = 8  # 最多冷却时间x8

        self._price_history: Dict[str, deque] = {}
        self._last_data_time: Optional[float] = None
        self._lock = threading.RLock()

        self._close_all_callback: Optional[Callable] = None
        self._stop_strategies_callback: Optional[Callable] = None

        # 币种隔离：per-symbol 熔断（单币种极端行情只冻结该币种，不全盘停盘）
        self._frozen_symbols: Dict[str, Dict] = {}  # symbol -> {reason, trigger_time, cooldown_until}
        self._symbol_freeze_cooldown = risk.get("symbol_freeze_cooldown_seconds", 600)  # 单币种冻结10分钟
        self._global_trigger_symbols = {"BTC-USDT-SWAP", "BTC-USDT"}  # 这些币种触发时升级为全局熔断
        self._multi_symbol_trigger_count = risk.get("multi_symbol_trigger_count", 3)  # 同时冻结>=3个币种时升级为全局
        self._freeze_symbol_callback: Optional[Callable] = None  # 单币种冻结回调（通知策略冻结该symbol）

    def set_close_all_callback(self, callback: Callable) -> None:
        """设置一键全平回调"""
        self._close_all_callback = callback

    def set_stop_strategies_callback(self, callback: Callable) -> None:
        """设置停止策略回调"""
        self._stop_strategies_callback = callback

    def set_freeze_symbol_callback(self, callback: Callable) -> None:
        """设置单币种冻结回调（通知策略层冻结该symbol，不影响其他symbol）"""
        self._freeze_symbol_callback = callback

    def is_symbol_frozen(self, symbol: str) -> bool:
        """检查币种是否被冻结（币种隔离：冻结的币种禁止开仓，但允许平仓）"""
        with self._lock:
            if symbol not in self._frozen_symbols:
                return False
            info = self._frozen_symbols[symbol]
            cooldown_until = info.get("cooldown_until")
            if cooldown_until and datetime.now() < cooldown_until:
                return True
            # 冻结期过，自动解冻
            del self._frozen_symbols[symbol]
            logger.info(f"Symbol {symbol} unfrozen (cooldown expired)")
            return False

    def get_frozen_symbols(self) -> List[str]:
        """获取当前所有被冻结的币种"""
        with self._lock:
            # 先清理过期的
            now = datetime.now()
            expired = [s for s, info in self._frozen_symbols.items()
                       if info.get("cooldown_until") and now >= info["cooldown_until"]]
            for s in expired:
                del self._frozen_symbols[s]
                logger.info(f"Symbol {s} unfrozen (cooldown expired)")
            return list(self._frozen_symbols.keys())

    def freeze_symbol(self, symbol: str, reason: str) -> None:
        """冻结单个币种（币种隔离：只冻结该币种的策略和开仓，不影响其他币种）"""
        with self._lock:
            cooldown_until = datetime.now() + timedelta(seconds=self._symbol_freeze_cooldown)
            self._frozen_symbols[symbol] = {
                "reason": reason,
                "trigger_time": datetime.now(),
                "cooldown_until": cooldown_until,
            }
        logger.warning(f"⛔ Symbol FROZEN: {symbol} (reason: {reason}, cooldown: {self._symbol_freeze_cooldown}s)")

        # 通知策略层冻结该symbol
        if self._freeze_symbol_callback:
            try:
                self._freeze_symbol_callback(symbol, reason)
            except Exception as e:
                logger.error(f"Error in freeze_symbol callback: {e}")

    def unfreeze_symbol(self, symbol: str) -> None:
        """手动解冻币种"""
        with self._lock:
            self._frozen_symbols.pop(symbol, None)
        logger.info(f"Symbol {symbol} manually unfrozen")

    def update_price(self, symbol: str, price: float) -> None:
        """更新价格数据"""
        with self._lock:
            if symbol not in self._price_history:
                self._price_history[symbol] = deque(maxlen=60)
            self._price_history[symbol].append({
                "price": price,
                "time": time.time()
            })
            self._last_data_time = time.time()

    def update_heartbeat(self) -> None:
        """更新心跳时间戳（用于标记系统仍在运行，防止无持仓时误触发网络断开熔断）"""
        with self._lock:
            self._last_data_time = time.time()

    def _has_active_positions(self) -> bool:
        """检查是否有活跃持仓（价格历史中有最近60秒内的数据点）"""
        now = time.time()
        for history in self._price_history.values():
            if history and (now - history[-1]["time"]) < 60:
                return True
        return False

    def check(self, signal: Dict[str, Any] = None) -> RiskCheckResult:
        """执行紧急熔断检查

        币种隔离架构：
        - 单个非关键币种极端行情 → 只冻结该币种（不影响其他币种运行）
        - BTC等关键币种极端行情 → 升级为全局熔断
        - 同时冻结币种数 >= multi_symbol_trigger_count → 升级为全局熔断
        - 网络断开 → 全局熔断（所有币种同时断连）
        """
        with self._lock:
            # 紧急熔断冷却检查（含升级机制）
            # P3修复：平仓信号必须穿透L5冷却期，防止持仓无法平仓
            if self._emergency_triggered and self._trigger_time:
                elapsed = time.time() - self._trigger_time.timestamp()
                # P4: 网络断连使用更短的冷却时间（300s vs 3600s），网络恢复后自动解除
                if "network_disconnect" in self._emergency_reason:
                    escalated_cooldown = self._network_disconnect_cooldown * min(2 ** self._emergency_trigger_count, self._max_escalation_multiplier)
                else:
                    escalated_cooldown = self._cooldown_seconds * min(2 ** self._emergency_trigger_count, self._max_escalation_multiplier)
                if elapsed < escalated_cooldown:
                    # 检查是否为平仓/减仓信号，允许穿透冷却期
                    # P3修复：使用子串匹配替代startswith，兼容scalping_close_*等前缀格式
                    if signal:
                        sig_type = str(signal.get("signal_type", "")).lower()
                        _close_keywords = ("close", "stop_loss", "stop", "reduce", "exit", "liquidation", "tp", "take_profit")
                        is_close = any(kw in sig_type for kw in _close_keywords) if sig_type else False
                        if is_close:
                            logger.debug(f"L5 cooldown bypass for close signal: {signal.get('symbol')} {sig_type}")
                        else:
                            return RiskCheckResult(
                                RiskLayer.L5_EMERGENCY, False, RiskAction.FREEZE,
                                f"紧急熔断已触发: {self._emergency_reason} (冷却中 {elapsed:.0f}/{escalated_cooldown:.0f}s, 级别{self._emergency_trigger_count})",
                                {"triggered": True, "reason": self._emergency_reason,
                                 "elapsed_seconds": round(elapsed, 0),
                                 "cooldown_seconds": escalated_cooldown,
                                 "escalation_level": self._emergency_trigger_count}
                            )
                    else:
                        return RiskCheckResult(
                            RiskLayer.L5_EMERGENCY, False, RiskAction.FREEZE,
                            f"紧急熔断已触发: {self._emergency_reason} (冷却中 {elapsed:.0f}/{escalated_cooldown:.0f}s, 级别{self._emergency_trigger_count})",
                            {"triggered": True, "reason": self._emergency_reason,
                             "elapsed_seconds": round(elapsed, 0),
                             "cooldown_seconds": escalated_cooldown,
                             "escalation_level": self._emergency_trigger_count}
                        )
                else:
                    # 冷却期结束，自动解除并重置升级计数器
                    self._emergency_triggered = False
                    self._emergency_reason = ""
                    self._emergency_trigger_count = 0
                    logger.info("Emergency circuit breaker cooldown expired, resuming")

            # 检查信号对应币种是否被冻结（per-symbol 熔断）
            if signal:
                sig_symbol = signal.get("symbol", "")
                if sig_symbol and self.is_symbol_frozen(sig_symbol):
                    # 判断是开仓还是平仓信号
                    # P3修复：使用子串匹配，兼容scalping_close_*等前缀格式
                    sig_type = str(signal.get("signal_type", "")).lower()
                    _close_keywords = ("close", "stop_loss", "stop", "reduce", "exit", "liquidation", "tp", "take_profit")
                    is_close = any(kw in sig_type for kw in _close_keywords) if sig_type else False
                    if not is_close:
                        # 开仓信号：冻结的币种禁止开仓
                        return RiskCheckResult(
                            RiskLayer.L5_EMERGENCY, False, RiskAction.FREEZE,
                            f"币种{sig_symbol}已被冻结(单币种熔断)，禁止开仓",
                            {"symbol": sig_symbol, "frozen": True,
                             "reason": self._frozen_symbols.get(sig_symbol, {}).get("reason", "")}
                        )
                    # 平仓信号：允许通过（冻结的币种仍可平仓）

            now = time.time()

            # 1. 全局网络断开检测（渐进式降级：警告→紧急熔断，仅在有活跃持仓时触发）
            if self._last_data_time:
                data_age = now - self._last_data_time
                if data_age > self._network_timeout_threshold:
                    # 超过紧急阈值：触发熔断（仅在有活跃持仓时）
                    if self._has_active_positions():
                        self._trigger_emergency("network_disconnect",
                                               f"网络断开 {data_age:.0f}秒")
                        return RiskCheckResult(
                            RiskLayer.L5_EMERGENCY, False, RiskAction.CLOSE_ALL,
                            f"网络断开超过{self._network_timeout_threshold}秒，触发紧急熔断",
                            {"last_data_age": round(data_age, 0)}
                        )
                    else:
                        logger.debug(f"Network timeout ({data_age:.0f}s) but no active positions, skipping emergency")
                elif data_age > self._network_warning_threshold:
                    # 超过警告阈值但未达紧急阈值：仅记录警告，不触发熔断
                    if self._has_active_positions() and not self._network_warning_issued:
                        self._network_warning_issued = True
                        logger.warning(
                            f"Network data age {data_age:.0f}s exceeds warning threshold "
                            f"({self._network_warning_threshold:.0f}s), "
                            f"will trigger emergency at {self._network_timeout_threshold}s"
                        )
                else:
                    # 网络恢复正常，清除警告标记
                    if self._network_warning_issued:
                        self._network_warning_issued = False
                        logger.debug("Network data flow restored, warning cleared")
                    # P4: 网络恢复后自动解除网络断连触发的紧急熔断
                    if self._emergency_triggered and "network_disconnect" in self._emergency_reason:
                        self._emergency_triggered = False
                        self._emergency_reason = ""
                        self._emergency_trigger_count = 0
                        logger.info("Network data flow restored, auto-resetting emergency circuit breaker")

            # 2. 极端插针行情检测（per-symbol 隔离，基于时间窗口而非tick计数）
            now_ts = time.time()
            for symbol, history in self._price_history.items():
                if len(history) < 3:
                    continue

                # 按时间窗口筛选：只取 _flash_crash_window 秒内的有效价格
                window_prices = [p for p in history if now_ts - p["time"] <= self._flash_crash_window]
                if len(window_prices) < 3:
                    continue

                prices = [p["price"] for p in window_prices]
                recent_max = max(prices)
                recent_min = min(prices)
                current = prices[-1]

                if current > 0:
                    # 检测瞬间暴跌
                    drop_pct = safe_div(recent_max - current, recent_max, 0.0)
                    if drop_pct >= self._flash_crash_threshold:
                        # 币种隔离：判断是否需要升级为全局熔断
                        if symbol in self._global_trigger_symbols:
                            # BTC等关键币种 → 全局熔断
                            self._trigger_emergency("flash_crash",
                                                   f"{symbol}暴跌{drop_pct*100:.2f}%(关键币种)")
                            return RiskCheckResult(
                                RiskLayer.L5_EMERGENCY, False, RiskAction.CLOSE_ALL,
                                f"极端插针: 关键币种{symbol}暴跌{drop_pct*100:.2f}%，触发全局紧急全平",
                                {"symbol": symbol, "drop_pct": round(drop_pct, 4),
                                 "recent_max": recent_max, "current": current}
                            )
                        else:
                            # 非关键币种 → 只冻结该币种
                            self.freeze_symbol(symbol, f"暴跌{drop_pct*100:.2f}%")
                            # 检查是否同时冻结过多币种 → 升级为全局
                            if len(self._frozen_symbols) >= self._multi_symbol_trigger_count:
                                self._trigger_emergency("multi_symbol_crash",
                                                       f"同时{len(self._frozen_symbols)}个币种触发熔断")
                                return RiskCheckResult(
                                    RiskLayer.L5_EMERGENCY, False, RiskAction.CLOSE_ALL,
                                    f"多币种同时极端行情({len(self._frozen_symbols)}个)，触发全局紧急全平",
                                    {"frozen_symbols": list(self._frozen_symbols.keys())}
                                )
                            # 只返回该symbol的冻结结果（如果当前信号是该symbol）
                            if signal and signal.get("symbol") == symbol:
                                return RiskCheckResult(
                                    RiskLayer.L5_EMERGENCY, False, RiskAction.FREEZE,
                                    f"极端插针: {symbol}暴跌{drop_pct*100:.2f}%，币种已冻结",
                                    {"symbol": symbol, "drop_pct": round(drop_pct, 4),
                                     "recent_max": recent_max, "current": current}
                                )

                    # 检测瞬间暴涨
                    surge_pct = safe_div(current - recent_min, recent_min, 0.0) if recent_min > 0 else 0.0
                    if surge_pct >= self._flash_crash_threshold:
                        if symbol in self._global_trigger_symbols:
                            self._trigger_emergency("flash_surge",
                                                   f"{symbol}暴涨{surge_pct*100:.2f}%(关键币种)")
                            return RiskCheckResult(
                                RiskLayer.L5_EMERGENCY, False, RiskAction.CLOSE_ALL,
                                f"极端插针: 关键币种{symbol}暴涨{surge_pct*100:.2f}%，触发全局紧急全平",
                                {"symbol": symbol, "surge_pct": round(surge_pct, 4),
                                 "recent_min": recent_min, "current": current}
                            )
                        else:
                            self.freeze_symbol(symbol, f"暴涨{surge_pct*100:.2f}%")
                            if len(self._frozen_symbols) >= self._multi_symbol_trigger_count:
                                self._trigger_emergency("multi_symbol_surge",
                                                       f"同时{len(self._frozen_symbols)}个币种触发暴涨熔断")
                                return RiskCheckResult(
                                    RiskLayer.L5_EMERGENCY, False, RiskAction.CLOSE_ALL,
                                    f"多币种同时极端行情({len(self._frozen_symbols)}个)，触发全局紧急全平",
                                    {"frozen_symbols": list(self._frozen_symbols.keys())}
                                )
                            if signal and signal.get("symbol") == symbol:
                                return RiskCheckResult(
                                    RiskLayer.L5_EMERGENCY, False, RiskAction.FREEZE,
                                    f"极端插针: {symbol}暴涨{surge_pct*100:.2f}%，币种已冻结",
                                    {"symbol": symbol, "surge_pct": round(surge_pct, 4),
                                     "recent_min": recent_min, "current": current}
                                )

        return RiskCheckResult(
            RiskLayer.L5_EMERGENCY, True, RiskAction.PASS,
            "紧急熔断检查通过", {}
        )

    def _trigger_emergency(self, event_type: str, reason: str) -> None:
        """触发紧急熔断（含升级机制：连续触发时冷却时间翻倍）"""
        with self._lock:
            # 如果在冷却期内再次触发，升级计数
            if self._emergency_triggered and self._trigger_time:
                elapsed = (datetime.now() - self._trigger_time).total_seconds()
                current_cooldown = self._cooldown_seconds * min(2 ** self._emergency_trigger_count, self._max_escalation_multiplier)
                if elapsed < current_cooldown:
                    self._emergency_trigger_count += 1
                    escalated_cooldown = self._cooldown_seconds * min(2 ** self._emergency_trigger_count, self._max_escalation_multiplier)
                    logger.warning(f"Emergency re-triggered within cooldown! Escalation level: {self._emergency_trigger_count}, "
                                 f"cooldown: {escalated_cooldown}s")
            
            self._emergency_triggered = True
            self._emergency_reason = f"[{event_type}] {reason}"
            self._trigger_time = datetime.now()
        
        logger.critical(f"!!! 紧急熔断触发 !!! 原因: {self._emergency_reason}")
        
        # 执行一键全平
        if self._close_all_callback:
            try:
                self._close_all_callback(reason)
                logger.critical("Emergency close_all executed")
            except Exception as e:
                logger.error(f"Error in close_all callback: {e}")
        
        # 执行停止策略
        if self._stop_strategies_callback:
            try:
                self._stop_strategies_callback(reason)
                logger.critical("Emergency stop_strategies executed")
            except Exception as e:
                logger.error(f"Error in stop_strategies callback: {e}")

    def manual_trigger(self, reason: str = "manual") -> None:
        """手动触发紧急熔断"""
        self._trigger_emergency("manual", reason)

    def reset(self) -> None:
        """手动解除熔断"""
        with self._lock:
            self._emergency_triggered = False
            self._emergency_reason = ""
            self._trigger_time = None
            logger.info("Emergency circuit breaker reset manually")

    def is_triggered(self) -> bool:
        """是否已触发"""
        with self._lock:
            return self._emergency_triggered

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "emergency_triggered": self._emergency_triggered,
                "emergency_reason": self._emergency_reason,
                "trigger_time": self._trigger_time.isoformat() if self._trigger_time else None,
                "cooldown_seconds": self._cooldown_seconds,
                "last_data_age": round(time.time() - self._last_data_time, 0) if self._last_data_time else None
            }


# ============================================================================
# 五层风控拦截器统一入口
# ============================================================================

class RiskGate:
    """
    多层级风控拦截器
    
    五层串行校验，任意一层拦截直接驳回，不可逆
    
    使用方式：
        result = risk_gate.validate(signal)
        if not result.passed:
            # 驳回下单
            return
        # 继续下单
    """

    def __init__(self, config: Dict[str, Any] = None, okx_client=None):
        self.config = config or {}
        
        self._l1 = PreTradeRiskChecker(config)
        self._l2 = InTradeRiskChecker(config)
        self._l3 = PositionRiskChecker(config)
        self._l4 = DailyRiskChecker(config)
        self._l5 = EmergencyCircuitBreaker(config)

        # P0: 全局 Kill Switch（L0 检查，先于 L5，仅禁开仓、平仓穿透，重启保持）
        self._kill_switch = KillSwitch()
        
        # P5: 注入 OKX 客户端用于仓位容量查询；缺失时 get_active_position_count 会
        # 因两通道均失败而 fail-closed 拒绝所有开仓信号（历史上「一直没开单」根因）
        self._okx_client = okx_client
        self._position_manager = None
        
        self._lock = threading.RLock()
        
        self._total_checks = 0
        self._total_rejections = 0
        self._rejections_by_layer: Dict[str, int] = defaultdict(int)
        
        self._interception_stats: Dict[str, Dict] = {
            "L0_kill_switch": {"total": 0, "blocks": 0, "reduces": 0, "close_alls": 0, "reasons": {}},
            "L1_pre_trade": {"total": 0, "blocks": 0, "reduces": 0, "close_alls": 0, "reasons": {}},
            "L2_in_trade": {"total": 0, "blocks": 0, "reduces": 0, "close_alls": 0, "reasons": {}},
            "L3_position": {"total": 0, "blocks": 0, "reduces": 0, "close_alls": 0, "reasons": {}},
            "L4_daily": {"total": 0, "blocks": 0, "reduces": 0, "close_alls": 0, "reasons": {}},
            "L5_emergency": {"total": 0, "blocks": 0, "reduces": 0, "close_alls": 0, "reasons": {}},
        }
        
        # P1: 风控事件持久化存储（修复 risk_events 表长期 0 行的审计缺口）
        self._sqlite_storage = None

        logger.info("RiskGate initialized: 5-layer serial risk control")

    def set_sqlite_storage(self, storage):
        """注入SQLite存储，用于将风控拦截事件持久化到 risk_events 表"""
        self._sqlite_storage = storage

    def set_okx_client(self, okx_client) -> None:
        """注入 OKX 客户端（用于 get_active_position_count 仓位容量查询）。

        get_risk_gate 在单例已存在但未注入客户端时补注入，避免两通道查询
        均失败导致 fail-closed 拒绝所有开仓。此方法此前缺失导致运行时 AttributeError。
        """
        self._okx_client = okx_client

    def set_position_manager(self, position_manager) -> None:
        """注入仓位管理器，以便同步持续失败时冻结新开仓。"""
        self._position_manager = position_manager

    def _record_interception(self, layer: str, action: RiskAction, reason: str, symbol: str = ""):
        """记录风控拦截到统计，并持久化到 risk_events 表"""
        if layer not in self._interception_stats:
            self._interception_stats[layer] = {"total": 0, "blocks": 0, "reduces": 0, "close_alls": 0, "reasons": {}}
        
        stat = self._interception_stats[layer]
        stat["total"] += 1
        
        if action == RiskAction.REJECT:
            stat["blocks"] += 1
        elif action == RiskAction.REDUCE:
            stat["reduces"] += 1
        elif action in (RiskAction.CLOSE_ALL, RiskAction.FREEZE):
            stat["close_alls"] += 1
        
        stat["reasons"][reason] = stat["reasons"].get(reason, 0) + 1

        # P1: 持久化拦截事件，修复风控审计缺口（此前 risk_events 表长期 0 行）
        if self._sqlite_storage is not None:
            try:
                severity = "CRITICAL" if action in (RiskAction.CLOSE_ALL, RiskAction.FREEZE) else "WARNING"
                self._sqlite_storage.save_risk_event({
                    "id": f"risk_{layer}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                    "event_type": "risk_interception",
                    "severity": severity,
                    "message": f"[{layer}] {reason}",
                    "symbol": symbol or "",
                    "timestamp": datetime.now(),
                })
            except Exception as e:
                logger.debug(f"Failed to persist risk interception event: {e}")

    def validate(self, signal: Dict[str, Any],
                 market_data: Dict[str, Any] = None,
                 is_close: bool = False) -> RiskGateResult:
        """
        执行五层串行风控校验

        风控前置架构核心：所有下单动作（开仓/平仓/减仓/止损/止盈/条件单）
        必须经过此方法校验通过后才能下发至OKX，从源头控制风险。

        Args:
            signal: 交易信号
            market_data: 市场数据（用于L2事中风控）
            is_close: 是否为平仓/减仓信号。True时跳过L1保证金/仓位上限检查
                      （平仓释放保证金而非占用），但保留L4/L5全局风控。

        Returns:
            RiskGateResult: 总结果
        """
        self._total_checks += 1
        results = []
        # P1: 提取 signal 的 symbol 用于风控事件持久化
        _symbol = signal.get("symbol", "") if isinstance(signal, dict) else ""
        signal_type = (
            str(signal.get("signal_type", "")).lower()
            if isinstance(signal, dict)
            else ""
        )
        is_close = bool(is_close)
        if isinstance(signal, dict):
            is_close = is_close or bool(signal.get("reduce_only", False))
        if not is_close:
            close_keywords = (
                "close", "exit", "reduce", "stop_loss", "take_profit",
                "trailing", "tp", "sl", "liquidation", "margin_call",
            )
            is_close = any(keyword in signal_type for keyword in close_keywords)

        # L0 全局 Kill Switch（最高优先级，先于 L5）
        # fail-closed：仅禁止开仓，平仓/减仓/止损/止盈等降风险信号穿透放行
        if self._kill_switch.is_enabled():
            if not is_close:
                l0_result = RiskCheckResult(
                    RiskLayer.L0_KILL_SWITCH, False, RiskAction.FREEZE,
                    f"全局 Kill Switch 已启用，禁止开仓: {self._kill_switch.get_reason()}",
                    {"kill_switch": True}
                )
                results.append(l0_result)
                self._total_rejections += 1
                self._rejections_by_layer[RiskLayer.L0_KILL_SWITCH.value] += 1
                self._record_interception(RiskLayer.L0_KILL_SWITCH.value, l0_result.action, l0_result.reason, _symbol)
                return RiskGateResult(
                    passed=False, action=l0_result.action,
                    blocked_layer=RiskLayer.L0_KILL_SWITCH, results=results,
                    summary=f"L0 KillSwitch 拦截: {l0_result.reason}"
                )

        if (
            not is_close
            and self._position_manager is not None
            and bool(getattr(self._position_manager, "sync_degraded", False))
        ):
            reason = (
                "持仓同步连续失败，持仓状态未知，暂停新开仓 "
                f"(连续失败 {getattr(self._position_manager, 'sync_fail_streak', 'N/A')} 次)"
            )
            sync_result = RiskCheckResult(
                RiskLayer.L1_PRE_TRADE,
                False,
                RiskAction.FREEZE,
                reason,
                {
                    "position_sync_degraded": True,
                    "sync_fail_streak": getattr(
                        self._position_manager, "sync_fail_streak", None
                    ),
                },
            )
            results.append(sync_result)
            self._total_rejections += 1
            self._rejections_by_layer[RiskLayer.L1_PRE_TRADE.value] += 1
            self._record_interception(
                RiskLayer.L1_PRE_TRADE.value,
                sync_result.action,
                sync_result.reason,
                _symbol,
            )
            return RiskGateResult(
                passed=False,
                action=sync_result.action,
                blocked_layer=RiskLayer.L1_PRE_TRADE,
                results=results,
                summary=f"L1持仓同步降级拦截: {sync_result.reason}",
            )

        # 持仓数据过期（解析失败后未成功同步）：fail-closed，拒绝新开仓
        if (
            not is_close
            and self._position_manager is not None
            and callable(getattr(self._position_manager, "is_data_stale", None))
            and self._position_manager.is_data_stale()
        ):
            reason = "持仓数据过期（解析失败），持仓状态不可信，暂停新开仓"
            stale_result = RiskCheckResult(
                RiskLayer.L1_PRE_TRADE,
                False,
                RiskAction.FREEZE,
                reason,
                {"position_data_stale": True},
            )
            results.append(stale_result)
            self._total_rejections += 1
            self._rejections_by_layer[RiskLayer.L1_PRE_TRADE.value] += 1
            self._record_interception(
                RiskLayer.L1_PRE_TRADE.value,
                stale_result.action,
                stale_result.reason,
                _symbol,
            )
            return RiskGateResult(
                passed=False,
                action=stale_result.action,
                blocked_layer=RiskLayer.L1_PRE_TRADE,
                results=results,
                summary=f"L1持仓数据过期拦截: {stale_result.reason}",
            )

        # L5 紧急熔断（最高优先级，先检查）—— 平仓信号也需检查（熔断冷却期内禁止一切操作）
        l5_result = self._l5.check(signal)
        results.append(l5_result)
        if not l5_result.passed:
            self._total_rejections += 1
            self._rejections_by_layer[l5_result.layer.value] += 1
            self._record_interception(l5_result.layer.value, l5_result.action, l5_result.reason, _symbol)
            return RiskGateResult(
                passed=False, action=l5_result.action,
                blocked_layer=l5_result.layer, results=results,
                summary=f"L5紧急熔断拦截: {l5_result.reason}"
            )

        # L4 全局单日风控 —— 平仓信号也需检查（单日亏损硬限制/连续亏损暂停时禁止一切操作）
        l4_result = self._l4.check(signal)
        results.append(l4_result)
        if not l4_result.passed:
            self._total_rejections += 1
            self._rejections_by_layer[l4_result.layer.value] += 1
            self._record_interception(l4_result.layer.value, l4_result.action, l4_result.reason, _symbol)
            return RiskGateResult(
                passed=False, action=l4_result.action,
                blocked_layer=l4_result.layer, results=results,
                summary=f"L4单日风控拦截: {l4_result.reason}"
            )

        # L1/L2/L3 并行校验（三者独立，无数据依赖）
        # L1 事前风控 —— 平仓/减仓信号跳过（平仓释放保证金，不占用）
        # L2 事中风控 —— 平仓信号也需检查（API频率/延迟/滑点对平仓同样适用）
        # L3 持仓实时风控（不拦截开仓，但产生预警）
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            if not is_close:
                futures["L1"] = executor.submit(self._l1.check, signal)
            futures["L2"] = executor.submit(self._l2.check, signal, market_data)
            futures["L3"] = executor.submit(self._l3.check, signal)

            # 收集结果并按优先级处理（L1 > L2 > L3）
            l1_result = futures["L1"].result() if "L1" in futures else None
            l2_result = futures["L2"].result()
            l3_results = futures["L3"].result()

        # L1 结果处理
        if l1_result is not None:
            results.append(l1_result)
            if not l1_result.passed:
                self._total_rejections += 1
                self._rejections_by_layer[l1_result.layer.value] += 1
                self._record_interception(l1_result.layer.value, l1_result.action, l1_result.reason, _symbol)
                return RiskGateResult(
                    passed=False, action=l1_result.action,
                    blocked_layer=l1_result.layer, results=results,
                    summary=f"L1事前风控拦截: {l1_result.reason}"
                )

        # L2 结果处理
        results.append(l2_result)
        if not l2_result.passed:
            self._total_rejections += 1
            self._rejections_by_layer[l2_result.layer.value] += 1
            self._record_interception(l2_result.layer.value, l2_result.action, l2_result.reason, _symbol)
            return RiskGateResult(
                passed=False, action=l2_result.action,
                blocked_layer=l2_result.layer, results=results,
                summary=f"L2事中风控拦截: {l2_result.reason}"
            )

        # L3 结果处理（不拦截开仓，但产生预警）
        # 只对同symbol的风险持仓产生拦截，不对其他symbol的close_all动作阻断当前信号
        signal_symbol = signal.get("symbol", "") if isinstance(signal, dict) else ""
        signal_sig_type = str(signal.get("signal_type", "")).lower() if isinstance(signal, dict) else ""
        is_close_signal = any(kw in signal_sig_type for kw in ["close", "stop_loss", "take_profit", "reduce", "exit", "liquidation"])
        for r in l3_results:
            results.append(r)
            if not r.passed and r.action == RiskAction.CLOSE_ALL:
                # 只拦截同symbol的信号，且平仓/减仓信号不拦截（减仓是降低风险）
                risk_symbol = r.details.get("symbol", "") if r.details else ""
                if risk_symbol and risk_symbol != signal_symbol:
                    # 其他symbol的清算风险，不应当阻止当前symbol的正常操作
                    continue
                if is_close_signal:
                    # 当前symbol的平仓/减仓信号是在降低风险，不拦截
                    continue
                self._total_rejections += 1
                self._rejections_by_layer[r.layer.value] += 1
                self._record_interception(r.layer.value, r.action, r.reason, _symbol)
                return RiskGateResult(
                    passed=False, action=r.action,
                    blocked_layer=r.layer, results=results,
                    summary=f"L3持仓风控拦截: {r.reason}"
                )

        # 全部通过
        summary = "五层风控全部通过(平仓模式)" if is_close else "五层风控全部通过"
        return RiskGateResult(
            passed=True, action=RiskAction.PASS,
            blocked_layer=None, results=results,
            summary=summary
        )

    def set_manual_override_symbols(self, symbols) -> None:
        """注入手动开单白名单（symbol 集合），委托给 L3 在检查时跳过这些持仓。"""
        self._l3.set_manual_override_symbols(symbols)

    def check_positions(self) -> List[RiskCheckResult]:
        """单独执行L3持仓风控检查（用于后台监控循环）"""
        return self._l3.check()

    def update_account_status(self, equity: float, available_margin: float,
                               daily_pnl: float, daily_start_equity: float) -> None:
        """更新账户状态（L1和L4共用）"""
        self._l1.update_account(equity, available_margin, daily_pnl, daily_start_equity)
        self._l4.update_daily_equity(equity)

    def update_position(self, symbol: str, entry_price: float, current_price: float,
                         size: float, leverage: float, side: str,
                         liquidation_price: float = 0, funding_rate: float = 0) -> None:
        """更新持仓状态（L3）"""
        self._l3.update_position(symbol, entry_price, current_price, size,
                                 leverage, side, liquidation_price, funding_rate)

    def update_symbol_position(self, symbol: str, position_value: float) -> None:
        """更新币种持仓价值（L1）"""
        self._l1.update_symbol_position(symbol, position_value)

    def sync_positions_from_exchange(self, positions_map: Dict[str, float]) -> None:
        """全量同步持仓价值（委托L1）- 替换而非累积，防止残留数据"""
        self._l1.sync_positions_from_exchange(positions_map)

    def remove_symbol_position(self, symbol: str) -> None:
        """移除币种持仓记录（委托L1）"""
        self._l1.remove_symbol_position(symbol)

    def record_trade(self) -> None:
        """记录交易（L4）"""
        self._l4.record_trade()

    def record_trade_result(self, pnl: float) -> None:
        """记录交易结果（L4）"""
        self._l4.record_trade_result(pnl)

    def record_api_call(self) -> None:
        """记录API调用（L2）"""
        self._l2.record_api_call()

    def record_latency(self, latency_ms: float) -> None:
        """记录网络延迟（L2）"""
        self._l2.record_latency(latency_ms)

    def update_price(self, symbol: str, price: float) -> None:
        """更新价格数据（L5）"""
        self._l5.update_price(symbol, price)

    def get_leverage_multiplier(self) -> float:
        """获取当前杠杆乘数（L4降杠杆模式）"""
        return self._l4.get_leverage_multiplier()

    def is_trading_paused(self) -> bool:
        """交易是否暂停"""
        return self._l4.is_paused() or self._l5.is_triggered()

    def is_emergency_triggered(self) -> bool:
        """L5 EmergencyCircuitBreaker 是否已触发（企业级 Kill Switch 自适应联动信号源）。"""
        return self._l5.is_triggered()

    def set_emergency_callbacks(self, close_all_cb: Callable, stop_strategies_cb: Callable) -> None:
        """设置紧急熔断回调"""
        self._l5.set_close_all_callback(close_all_cb)
        self._l5.set_stop_strategies_callback(stop_strategies_cb)

    def set_freeze_symbol_callback(self, callback: Callable) -> None:
        """设置单币种冻结回调（币种隔离：通知策略层冻结该symbol）"""
        self._l5.set_freeze_symbol_callback(callback)

    def reset_interception_stats(self) -> None:
        """重置拦截统计（每日复盘后调用）"""
        self._interception_stats = {
            "L0_kill_switch": {"total": 0, "blocks": 0, "reduces": 0, "close_alls": 0, "reasons": {}},
            "L1_pre_trade": {"total": 0, "blocks": 0, "reduces": 0, "close_alls": 0, "reasons": {}},
            "L2_in_trade": {"total": 0, "blocks": 0, "reduces": 0, "close_alls": 0, "reasons": {}},
            "L3_position": {"total": 0, "blocks": 0, "reduces": 0, "close_alls": 0, "reasons": {}},
            "L4_daily": {"total": 0, "blocks": 0, "reduces": 0, "close_alls": 0, "reasons": {}},
            "L5_emergency": {"total": 0, "blocks": 0, "reduces": 0, "close_alls": 0, "reasons": {}},
        }
        self._total_checks = 0
        self._total_rejections = 0
        self._rejections_by_layer = defaultdict(int)

    def enable_kill_switch(self, reason: str = "", by: str = "") -> None:
        """启用全局 Kill Switch（禁止新开仓，平仓放行，重启保持）。"""
        self._kill_switch.enable(reason=reason, by=by)

    def disable_kill_switch(self, reason: str = "", by: str = "") -> None:
        """解除全局 Kill Switch（恢复新开仓）。"""
        self._kill_switch.disable(reason=reason, by=by)

    def is_kill_switch_enabled(self) -> bool:
        """全局 Kill Switch 是否已启用。"""
        return self._kill_switch.is_enabled()

    def set_kill_switch(self, kill_switch: KillSwitch) -> None:
        """注入自定义 KillSwitch 实例（测试/依赖注入用）。"""
        self._kill_switch = kill_switch

    def is_symbol_frozen(self, symbol: str) -> bool:
        """检查币种是否被冻结（币种隔离架构）"""
        return self._l5.is_symbol_frozen(symbol)

    def get_active_position_count(self) -> Optional[int]:
        """P5: 获取当前活跃持仓数量
        
        用于信号处理器的仓位容量预检查，避免在满载时生成无效信号。
        优先使用 L1 层缓存的持仓数据，回退到 OKX 客户端直接查询。
        
        返回 None 表示两个通道均查询失败（容量未知），调用方应按保守满载处理，
        而非当作 0 持仓（避免满载时误判为未满载继续生成信号）。
        """
        try:
            # 优先从 L1 层获取（已缓存）
            if hasattr(self._l1, 'get_cached_positions'):
                positions = self._l1.get_cached_positions()
                if positions is not None:
                    return sum(1 for p in positions
                             if abs(safe_float(p.get("pos") or p.get("position"), 0.0)) > 0)
        except Exception as e:
            logger.warning(f"L1 cached positions query failed: {e}")
        
        # 回退：从 OKX 客户端获取
        try:
            if hasattr(self, '_okx_client') and self._okx_client:
                checked_query = getattr(self._okx_client, "get_positions_checked", None)
                positions = (
                    checked_query()
                    if callable(checked_query)
                    else self._okx_client.get_positions()
                )
                if positions is not None:
                    return sum(1 for p in positions
                             if abs(safe_float(p.get("pos") or p.get("position"), 0.0)) > 0)
        except Exception as e:
            logger.warning(f"OKX positions query failed: {e}")
        
        # fail-closed：两个通道均失败，返回 None（未知容量），由调用方保守处理
        return None

    def get_frozen_symbols(self) -> List[str]:
        """获取所有被冻结的币种"""
        return self._l5.get_frozen_symbols()

    def get_l4_state(self) -> Dict[str, Any]:
        """P5: 获取L4单日风控状态"""
        return self._l4.to_dict()

    def freeze_symbol(self, symbol: str, reason: str) -> None:
        """手动冻结单个币种"""
        self._l5.freeze_symbol(symbol, reason)

    def unfreeze_symbol(self, symbol: str) -> None:
        """手动解冻单个币种"""
        self._l5.unfreeze_symbol(symbol)

    def manual_emergency_trigger(self, reason: str = "manual") -> None:
        """手动触发紧急熔断"""
        self._l5.manual_trigger(reason)

    def reset_emergency(self) -> None:
        """解除紧急熔断"""
        self._l5.reset()

    def reset_daily(self) -> None:
        """重置单日风控状态"""
        self._l4.reset()

    def get_status(self) -> Dict[str, Any]:
        """获取完整风控状态"""
        with self._lock:
            return {
                "total_checks": self._total_checks,
                "total_rejections": self._total_rejections,
                "rejection_rate": round(self._total_rejections / max(self._total_checks, 1), 4),
                "rejections_by_layer": dict(self._rejections_by_layer),
                "L1_pre_trade": {
                    "equity": self._l1._current_equity,
                    "available_margin": self._l1._available_margin,
                    "daily_pnl": self._l1._daily_pnl
                },
                "L0_kill_switch": self._kill_switch.to_dict(),
                "L2_in_trade": {
                    "current_rps": len([t for t in self._l2._api_call_times 
                                       if time.time() - t < 1.0]),
                    "max_rps": self._l2._max_api_rps
                },
                "L3_position": self._l3.get_position_summary(),
                "L4_daily": self._l4.to_dict(),
                "L5_emergency": self._l5.to_dict(),
                "position_sync": {
                    "degraded": bool(
                        self._position_manager
                        and getattr(
                            self._position_manager, "sync_degraded", False
                        )
                    ),
                    "failure_streak": (
                        getattr(self._position_manager, "sync_fail_streak", 0)
                        if self._position_manager
                        else 0
                    ),
                },
                "trading_paused": self.is_trading_paused()
            }


_gate_instance: Optional[RiskGate] = None

def get_risk_gate(config: Dict[str, Any] = None, okx_client=None) -> RiskGate:
    """获取风控拦截器单例"""
    global _gate_instance
    if _gate_instance is None:
        _gate_instance = RiskGate(config, okx_client=okx_client)
    elif okx_client is not None and _gate_instance._okx_client is None:
        # 单例已存在但未注入 OKX 客户端时，补注入，避免 fail-closed 拒绝开仓
        _gate_instance.set_okx_client(okx_client)
    return _gate_instance


__all__ = [
    "RiskGate",
    "PreTradeRiskChecker",
    "InTradeRiskChecker",
    "PositionRiskChecker",
    "DailyRiskChecker",
    "EmergencyCircuitBreaker",
    "RiskLayer",
    "RiskAction",
    "RiskCheckResult",
    "RiskGateResult",
    "get_risk_gate",
]