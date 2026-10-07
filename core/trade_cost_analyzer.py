"""
精准交易成本分析器 (Trade Cost Analyzer)
生产级模块 - 杜绝磨损型交易

功能：
1. 精确计算每笔交易的全链路成本（手续费、滑点、资金费率、点差）
2. 开单前盈亏平衡点预判，确保每笔交易有利可图
3. 最小盈利阈值检查，过滤微利/无效交易
4. 持仓成本实时追踪，防止隐性亏损
5. 交易频率成本分析，防止过度交易

设计原则：
- 所有成本计算必须保守估计（取上限），确保安全边际
- 盈亏平衡点计算使用 taker fee（即时成交成本）
- 资金费率成本按持仓时间预估
"""

import asyncio
import logging
import math
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ======================================================================
# 企业级多档费率表（静态回退）
# ----------------------------------------------------------------------
# OKX 合约（永续/交割）手续费率，按「分组2 所有其它交易对」计价。
# 数据来源：OKX 2026-04-08 调整后的合约手续费率（普通用户 + VIP1~VIP9）。
# 负费率表示挂单返佣（maker rebate），成本端保守按 0 计。
# 本表可被 config 的 trade_cost.fee_schedule 覆盖。
# ======================================================================
DEFAULT_FEE_SCHEDULE: Dict[str, Tuple[float, float]] = {
    # level: (maker, taker)
    "Lv1":      (0.00020, 0.00050),   # 普通用户
    "VIP1":     (0.00016, 0.00045),
    "VIP2":     (0.00015, 0.00036),
    "VIP3":     (0.00010, 0.00028),
    "VIP4":     (0.00008, 0.00027),
    "VIP5":     (0.00005, 0.00026),
    "VIP6":     (0.00000, 0.00025),
    "VIP7":     (-0.00005, 0.00025),
    "VIP8":     (-0.00010, 0.00025),
    "VIP9":     (-0.00010, 0.00020),
}

# OKX 返回的历史等级别名归一化（旧 Lv 命名 → 新命名）
_LEVEL_ALIASES: Dict[str, str] = {
    "lv0": "Lv1", "lv1": "Lv1", "lv2": "Lv1", "lv3": "Lv1",
    "lv4": "Lv1", "lv5": "Lv1", "普通用户": "Lv1", "regular": "Lv1",
}


@dataclass
class TradeCostBreakdown:
    """交易成本明细"""
    symbol: str
    side: str  # buy/sell
    pos_side: str  # long/short
    price: float
    quantity: float
    leverage: float
    notional: float  # 名义价值 = price * quantity * ctVal

    # 手续费（taker fee = 0.05% = 0.0005）
    entry_fee: float = 0.0  # 开仓手续费
    exit_fee: float = 0.0   # 平仓手续费
    total_fee: float = 0.0  # 总手续费

    # 实际生效费率（企业级资费：VIP等级 + maker/taker + 数据来源）
    maker_fee_rate: float = 0.0
    taker_fee_rate: float = 0.0
    fee_level: str = ""       # VIP等级（Lv1 / VIP1~VIP9）
    fee_source: str = ""      # realtime / static

    # 滑点成本（预估）
    estimated_slippage_pct: float = 0.0005  # 默认0.05%
    slippage_cost: float = 0.0

    # 资金费率成本（预估，按持仓时间）
    estimated_funding_rate: float = 0.0001  # 默认0.01%
    funding_cost: float = 0.0
    estimated_hold_hours: float = 24.0  # 预估持仓时间

    # 点差成本
    spread_pct: float = 0.0
    spread_cost: float = 0.0

    # 总成本
    total_cost: float = 0.0
    total_cost_pct: float = 0.0  # 总成本占名义价值的百分比

    # 盈亏平衡
    breakeven_price_long: float = 0.0  # 多头盈亏平衡价
    breakeven_price_short: float = 0.0  # 空头盈亏平衡价
    min_profit_price_long: float = 0.0  # 多头最小盈利价（成本*3）
    min_profit_price_short: float = 0.0  # 空头最小盈利价

    # 判断
    is_profitable: bool = True  # 是否可盈利
    min_required_move_pct: float = 0.0  # 最小需要的价格变动百分比


class TradeCostAnalyzer:
    """精准交易成本分析器"""

    # 默认费率参数
    DEFAULT_TAKER_FEE = 0.0005   # taker fee 0.05%
    DEFAULT_MAKER_FEE = 0.0002   # maker fee 0.02%
    DEFAULT_SLIPPAGE_PCT = 0.0005  # 预估滑点 0.05%
    DEFAULT_FUNDING_RATE = 0.0001  # 预估资金费率 0.01%/8h
    DEFAULT_SPREAD_PCT = 0.0003   # 预估点差 0.03%

    # 最小盈利要求：盈利必须 >= 总成本 * 3
    MIN_PROFIT_MULTIPLIER = 3.0

    def __init__(self, config: Optional[Dict] = None):
        self._config = config or {}
        self.taker_fee = self._config.get("taker_fee", self.DEFAULT_TAKER_FEE)
        self.maker_fee = self._config.get("maker_fee", self.DEFAULT_MAKER_FEE)
        self.default_slippage = self._config.get("default_slippage_pct", self.DEFAULT_SLIPPAGE_PCT)
        self.default_funding_rate = self._config.get("default_funding_rate", self.DEFAULT_FUNDING_RATE)
        self.default_spread = self._config.get("default_spread_pct", self.DEFAULT_SPREAD_PCT)
        self.min_profit_multiplier = self._config.get("min_profit_multiplier", self.MIN_PROFIT_MULTIPLIER)

        # 企业级资费：多档费率表 + 实时API优先
        self._okx_client = None
        self.fee_schedule = dict(DEFAULT_FEE_SCHEDULE)
        override_schedule = self._config.get("fee_schedule")
        if isinstance(override_schedule, dict):
            for lvl, rates in override_schedule.items():
                if isinstance(rates, (list, tuple)) and len(rates) == 2:
                    self.fee_schedule[str(lvl)] = (float(rates[0]), float(rates[1]))
        # 兼容旧配置：显式提供 taker_fee/maker_fee 时，作为 Lv1 静态费率
        if "taker_fee" in self._config or "maker_fee" in self._config:
            self.fee_schedule["Lv1"] = (self.maker_fee, self.taker_fee)
        self.default_level = self._normalize_level(
            self._config.get("default_level", self._config.get("vip_level", "Lv1"))
        )
        self.fetch_realtime_fee = self._config.get("fetch_realtime_fee", True)
        self.fee_cache_ttl = self._config.get("fee_cache_ttl", 3600)
        self._fee_cache: Dict[str, Dict[str, Any]] = {}

        # 统计
        self._cost_stats: Dict[str, Any] = {
            "total_analyzed": 0,
            "total_blocked": 0,
            "total_fee_saved": 0.0,
            "blocked_reasons": {},
        }

    # ==================================================================
    # 企业级资费：费率解析
    # ==================================================================

    def set_okx_client(self, okx_client) -> None:
        """注入 OKX 客户端，用于拉取账户实时费率（trade-fee API）。"""
        self._okx_client = okx_client

    def _normalize_level(self, level: str) -> str:
        """将 OKX 返回的历史等级别名归一化为标准等级 key。"""
        if not level:
            return "Lv1"
        key = str(level).strip()
        aliased = _LEVEL_ALIASES.get(key.lower(), key)
        # 未知等级回退到普通用户（最保守）
        return aliased if aliased in self.fee_schedule else "Lv1"

    def _lookup_schedule(self, level: str) -> Tuple[float, float]:
        """按等级查静态多档费率表，返回 (maker, taker)。"""
        lvl = self._normalize_level(level)
        return self.fee_schedule.get(lvl, DEFAULT_FEE_SCHEDULE["Lv1"])

    def _get_realtime_fee(self, inst_type: str = "SWAP") -> Optional[Dict[str, Any]]:
        """实时拉取账户费率（带 TTL 缓存）。

        为避免在事件循环内同步阻塞，仅当不在运行中的事件循环时才会真正发请求；
        事件循环内缓存未命中则返回 None，由静态表兜底。
        """
        if not self._okx_client or not self.fetch_realtime_fee:
            return None

        now = time.time()
        cached = self._fee_cache.get(inst_type)
        fresh = cached and now - cached.get("ts", 0) < self.fee_cache_ttl

        try:
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False

        # 事件循环内：有缓存（含过期缓存）直接复用，避免同步阻塞；
        # 无缓存时回退静态表。
        if in_loop:
            return cached.get("data") if cached else None

        # 非事件循环：缓存未过期直接返回，过期则刷新
        if fresh:
            return cached.get("data")

        try:
            data = self._okx_client.get_account_trade_fee(inst_type)
            if data and data.get("taker") is not None:
                self._fee_cache[inst_type] = {"ts": now, "data": data}
                return data
        except Exception as e:
            logger.debug(f"realtime fee fetch failed: {e}")
        return cached.get("data") if cached else None

    def prefetch_fee(self, inst_type: str = "SWAP") -> None:
        """启动阶段预取账户费率（在事件循环外调用），填充缓存。"""
        if not self._okx_client or not self.fetch_realtime_fee:
            return
        try:
            data = self._okx_client.get_account_trade_fee(inst_type)
            if data and data.get("taker") is not None:
                self._fee_cache[inst_type] = {"ts": time.time(), "data": data}
                logger.info(
                    f"Prefetched OKX account fee: level={data.get('level')} "
                    f"maker={data.get('maker')} taker={data.get('taker')}"
                )
        except Exception as e:
            logger.warning(f"Fee prefetch failed: {e}")

    def get_effective_fee_rates(
        self, symbol: Optional[str] = None, inst_type: str = "SWAP"
    ) -> Tuple[float, float, str, str]:
        """返回实际生效费率 (taker, maker, level, source)。

        优先级：实时 trade-fee API（缓存）> 静态多档费率表（按等级）> 默认等级。
        """
        realtime = self._get_realtime_fee(inst_type)
        if realtime:
            taker = float(realtime.get("taker", 0.0))
            maker = float(realtime.get("maker", 0.0))
            level = self._normalize_level(realtime.get("level", ""))
            return taker, maker, level, "realtime"

        maker, taker = self._lookup_schedule(self.default_level)
        return taker, maker, self.default_level, "static"

    def get_fee_summary(self, inst_type: str = "SWAP") -> Dict[str, Any]:
        """返回当前账户资费摘要（供报告/仪表盘展示）。"""
        taker, maker, level, source = self.get_effective_fee_rates(inst_type=inst_type)
        return {
            "inst_type": inst_type,
            "level": level,
            "maker_fee_rate": maker,
            "taker_fee_rate": taker,
            "source": source,
            "fetch_realtime_fee": self.fetch_realtime_fee,
            "fee_cache_ttl": self.fee_cache_ttl,
        }

    def get_cost_params(self, inst_type: str = "SWAP") -> Dict[str, float]:
        """返回统一成本参数，供 handle_signal 等上游层复用，消除参数分歧。"""
        taker, _, _, _ = self.get_effective_fee_rates(inst_type=inst_type)
        return {
            "taker_fee": max(taker, 0.0),
            "slippage_pct": self.default_slippage,
        }

    def calculate_full_cost(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        leverage: float = 1.0,
        pos_side: str = "",
        estimated_hold_hours: float = 24.0,
        is_taker: bool = True,
        ct_val: float = 1.0,
    ) -> TradeCostBreakdown:
        """计算完整交易成本

        Args:
            symbol: 交易对
            side: buy/sell
            price: 价格
            quantity: 数量（张数）
            leverage: 杠杆倍数
            pos_side: 持仓方向 long/short
            estimated_hold_hours: 预估持仓时间（小时）
            is_taker: 是否taker单
            ct_val: 合约面值
        """
        notional = price * quantity * ct_val

        # 企业级资费：实时API优先，回退静态多档费率表
        taker_rate, maker_rate, fee_level, fee_source = self.get_effective_fee_rates(symbol)
        fee_rate = taker_rate if is_taker else maker_rate
        # 负费率（VIP返佣）在成本端保守按 0 计
        fee_rate = max(fee_rate, 0.0)

        # 手续费：开仓 + 平仓（按吃单费率保守计算）
        entry_fee = notional * fee_rate
        exit_fee = notional * fee_rate
        total_fee = entry_fee + exit_fee

        # 滑点成本（保守估计）
        slippage_cost = notional * self.default_slippage

        # 资金费率成本（按预估持仓时间）
        funding_intervals = max(1, estimated_hold_hours / 8.0)  # 每8小时结算一次
        funding_cost = notional * self.default_funding_rate * funding_intervals

        # 点差成本
        spread_cost = notional * self.default_spread

        # 总成本
        total_cost = total_fee + slippage_cost + funding_cost + spread_cost
        total_cost_pct = total_cost / notional if notional > 0 else 0

        # 盈亏平衡价格
        if pos_side == "long" or (side == "buy" and not pos_side):
            # 多头：买入价 + 成本/数量 = 卖出价才能不亏
            breakeven_price_long = price + (total_cost / quantity) if quantity > 0 else price
            breakeven_price_short = 0
            min_profit_price_long = price + (total_cost * self.min_profit_multiplier / quantity) if quantity > 0 else price
            min_profit_price_short = 0
            min_required_move_pct = (min_profit_price_long - price) / price if price > 0 else 0
        else:
            # 空头：卖出价 - 成本/数量 = 买入价才能不亏
            breakeven_price_short = price - (total_cost / quantity) if quantity > 0 else price
            breakeven_price_long = 0
            min_profit_price_short = price - (total_cost * self.min_profit_multiplier / quantity) if quantity > 0 else price
            min_profit_price_long = 0
            min_required_move_pct = (price - min_profit_price_short) / price if price > 0 else 0

        # 判断是否有利可图
        # 微利交易判断：名义价值太小，盈利不够覆盖成本
        min_notional = total_cost * self.min_profit_multiplier * 10  # 至少10倍成本覆盖
        is_profitable = notional >= min_notional and min_required_move_pct < 0.05  # 价格变动<5%可达盈利

        return TradeCostBreakdown(
            symbol=symbol,
            side=side,
            pos_side=pos_side,
            price=price,
            quantity=quantity,
            leverage=leverage,
            notional=notional,
            entry_fee=entry_fee,
            exit_fee=exit_fee,
            total_fee=total_fee,
            maker_fee_rate=maker_rate,
            taker_fee_rate=taker_rate,
            fee_level=fee_level,
            fee_source=fee_source,
            estimated_slippage_pct=self.default_slippage,
            slippage_cost=slippage_cost,
            estimated_funding_rate=self.default_funding_rate,
            funding_cost=funding_cost,
            estimated_hold_hours=estimated_hold_hours,
            spread_pct=self.default_spread,
            spread_cost=spread_cost,
            total_cost=total_cost,
            total_cost_pct=total_cost_pct,
            breakeven_price_long=breakeven_price_long,
            breakeven_price_short=breakeven_price_short,
            min_profit_price_long=min_profit_price_long,
            min_profit_price_short=min_profit_price_short,
            is_profitable=is_profitable,
            min_required_move_pct=min_required_move_pct,
        )

    def should_open_position(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        leverage: float = 1.0,
        pos_side: str = "",
        expected_profit_pct: float = 0.01,
        ct_val: float = 1.0,
        market_volatility: Optional[float] = None,
        expected_hold_hours: float = 24.0,
    ) -> Tuple[bool, str, TradeCostBreakdown]:
        """开仓前判断：此交易是否值得执行

        Args:
            expected_profit_pct: 预期盈利百分比（如0.01 = 1%）
            market_volatility: 币种真实 24h 振幅（(high24-low24)/last），
                用于「震荡磨损」事前校验：窄幅震荡中若振幅不足以容纳止盈目标
                （含开仓成本），则开仓必然被手续费磨损，拒绝执行。None 表示跳过校验。

        Returns:
            (是否应该开仓, 原因, 成本明细)
        """
        self._cost_stats["total_analyzed"] += 1

        try:
            horizon_hours = float(expected_hold_hours)
        except (TypeError, ValueError):
            horizon_hours = 24.0
        if not math.isfinite(horizon_hours) or horizon_hours <= 0:
            horizon_hours = 24.0
        horizon_hours = max(0.25, min(horizon_hours, 24.0 * 7))

        cost = self.calculate_full_cost(
            symbol=symbol, side=side, price=price, quantity=quantity,
            leverage=leverage, pos_side=pos_side, ct_val=ct_val,
            estimated_hold_hours=horizon_hours,
        )

        # 检查1: 名义价值是否足够
        min_notional = cost.total_cost * self.min_profit_multiplier * 10
        if cost.notional < min_notional:
            reason = f"名义价值过低 {cost.notional:.2f} < {min_notional:.2f}"
            self._record_blocked(reason)
            return False, reason, cost

        # 检查2: 预期盈利是否能覆盖成本
        expected_profit_usdt = cost.notional * expected_profit_pct
        if expected_profit_usdt < cost.total_cost * self.min_profit_multiplier:
            reason = (
                f"预期盈利不足以覆盖成本: profit={expected_profit_usdt:.4f} < "
                f"cost*{self.min_profit_multiplier}={cost.total_cost * self.min_profit_multiplier:.4f}"
            )
            self._record_blocked(reason)
            return False, reason, cost

        # 检查3: 最小价格变动要求是否合理
        if cost.min_required_move_pct > 0.05:
            reason = f"所需价格变动过大: {cost.min_required_move_pct:.2%} > 5%"
            self._record_blocked(reason)
            return False, reason, cost

        # 检查4（企业级震荡磨损事前防护）: 当前震荡空间是否足以容纳止盈目标
        # 窄幅震荡中价格缺乏足够移动空间，止盈不可达，开仓只能被手续费磨损。
        # 将24h振幅按持仓周期缩放；短周期与多日策略不能直接共用24h阈值。
        if market_volatility is not None and market_volatility > 0:
            horizon_volatility = market_volatility * math.sqrt(horizon_hours / 24.0)
            required_volatility = expected_profit_pct + cost.total_cost_pct
            if horizon_volatility < required_volatility:
                reason = (
                    f"震荡空间不足: {horizon_hours:g}h估算振幅={horizon_volatility:.4%} "
                    f"(24h振幅={market_volatility:.4%}) < "
                    f"预期盈利+成本={required_volatility:.4%} "
                    f"(tp={expected_profit_pct:.4%}, cost={cost.total_cost_pct:.4%})"
                )
                self._record_blocked(reason)
                return False, reason, cost

        return True, "OK", cost

    def should_close_position(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        quantity: float,
        pos_side: str,
        ct_val: float = 1.0,
        allow_loss: bool = True,
    ) -> Tuple[bool, str, float]:
        """平仓前判断：平仓后是否净盈利

        Args:
            allow_loss: 是否允许亏损平仓（止损）

        Returns:
            (是否应该平仓, 原因, 预估净盈亏)
        """
        notional = entry_price * quantity * ct_val
        taker_rate, _, _, _ = self.get_effective_fee_rates(symbol)
        taker_rate = max(taker_rate, 0.0)
        exit_fee = current_price * quantity * ct_val * taker_rate
        total_fee = notional * taker_rate + exit_fee  # 开仓费+平仓费

        if pos_side == "long":
            gross_pnl = (current_price - entry_price) * quantity * ct_val
        else:
            gross_pnl = (entry_price - current_price) * quantity * ct_val

        net_pnl = gross_pnl - total_fee

        if net_pnl <= 0 and not allow_loss:
            return False, f"平仓净亏损: {net_pnl:.4f} USDT (gross={gross_pnl:.4f}, fee={total_fee:.4f})", net_pnl

        return True, "OK", net_pnl

    def calculate_optimal_quantity(
        self,
        symbol: str,
        price: float,
        available_margin: float,
        leverage: float,
        expected_profit_pct: float = 0.01,
        ct_val: float = 1.0,
    ) -> float:
        """计算最优开仓数量（确保覆盖成本后有盈利）

        Args:
            available_margin: 可用保证金
            expected_profit_pct: 预期盈利百分比

        Returns:
            最优数量（已向下取整到整数）
        """
        if price <= 0 or available_margin <= 0:
            return 0

        max_notional = available_margin * leverage
        max_qty = max_notional / (price * ct_val)

        # 找到最小可盈利数量
        # 总成本 = notional * (2 * taker_fee + slippage + funding + spread)
        taker_rate, _, _, _ = self.get_effective_fee_rates(symbol)
        taker_rate = max(taker_rate, 0.0)
        total_cost_rate = 2 * taker_rate + self.default_slippage + self.default_funding_rate + self.default_spread
        min_profit_rate = total_cost_rate * self.min_profit_multiplier

        # 需要的名义价值: profit >= cost * multiplier
        # notional * expected_profit_pct >= notional * total_cost_rate * multiplier
        # 如果 expected_profit_pct <= total_cost_rate * multiplier，则无法盈利
        if expected_profit_pct <= min_profit_rate:
            return 0  # 当前预期盈利无法覆盖成本

        # 最小名义价值（确保至少盈利1 USDT）
        min_notional_for_1usdt = 1.0 / (expected_profit_pct - min_profit_rate)
        min_qty = min_notional_for_1usdt / (price * ct_val)

        if min_qty > max_qty:
            return 0  # 资金不足，无法实现盈利

        # 返回可用范围内的数量
        return max(0, int(min(max_qty, max_qty * 0.8)))  # 保守使用80%可用资金

    def get_breakeven_prices(
        self,
        entry_price: float,
        quantity: float,
        pos_side: str,
        ct_val: float = 1.0,
    ) -> Dict[str, float]:
        """获取盈亏平衡价格和推荐止盈止损"""
        notional = entry_price * quantity * ct_val
        total_fee = notional * self.taker_fee * 2

        if pos_side == "long":
            breakeven = entry_price + (total_fee / quantity)
            return {
                "breakeven": breakeven,
                "min_tp": breakeven * 1.005,  # 最小止盈0.5%
                "recommended_tp": breakeven * 1.01,  # 推荐止盈1%
                "max_sl": entry_price * 0.98,  # 最大止损2%
            }
        else:
            breakeven = entry_price - (total_fee / quantity)
            return {
                "breakeven": breakeven,
                "min_tp": breakeven * 0.995,
                "recommended_tp": breakeven * 0.99,
                "max_sl": entry_price * 1.02,
            }

    def _record_blocked(self, reason: str):
        """记录被拦截的交易"""
        self._cost_stats["total_blocked"] += 1
        key = reason.split(":")[0] if ":" in reason else reason[:20]
        self._cost_stats["blocked_reasons"][key] = self._cost_stats["blocked_reasons"].get(key, 0) + 1

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return dict(self._cost_stats)