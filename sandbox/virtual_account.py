"""
虚拟账户管理
============
负责策略沙盒中的虚拟资金、持仓、盈亏跟踪

功能：
- 虚拟余额管理（初始本金、可用余额、占用保证金）
- 虚拟持仓管理（开仓/加仓/减仓/平仓）
- 实时盈亏计算（未实现盈亏 + 已实现盈亏）
- 费率模拟（Maker/Taker手续费、资金费率）
- 杠杆控制与爆仓模拟
- 账户快照持久化
"""

import threading
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger


@dataclass
class VirtualPosition:
    """虚拟持仓"""
    symbol: str
    side: str  # "long" | "short"
    quantity: float
    avg_entry_price: float
    leverage: int
    margin_used: float
    open_time: datetime = field(default_factory=datetime.now)
    realized_pnl: float = 0.0
    total_fee_paid: float = 0.0
    funding_fee_paid: float = 0.0
    add_positions: List[Dict[str, Any]] = field(default_factory=list)  # 加仓记录

    @property
    def position_value(self) -> float:
        """持仓名义价值"""
        return self.quantity * self.avg_entry_price

    def unrealized_pnl(self, current_price: float) -> float:
        """未实现盈亏"""
        if self.side == "long":
            return (current_price - self.avg_entry_price) * self.quantity
        else:
            return (self.avg_entry_price - current_price) * self.quantity

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """未实现盈亏百分比（相对保证金）"""
        if self.margin_used <= 0:
            return 0.0
        return self.unrealized_pnl(current_price) / self.margin_used

    def add(self, quantity: float, price: float, fee: float) -> None:
        """加仓"""
        old_value = self.quantity * self.avg_entry_price
        new_value = quantity * price
        total_qty = self.quantity + quantity
        if total_qty > 0:
            self.avg_entry_price = (old_value + new_value) / total_qty
        self.quantity = total_qty
        self.total_fee_paid += fee
        self.add_positions.append({
            "time": datetime.now().isoformat(),
            "quantity": quantity,
            "price": price,
            "fee": fee
        })

    def reduce(self, quantity: float, price: float, fee: float) -> float:
        """减仓，返回已实现盈亏"""
        if quantity > self.quantity:
            quantity = self.quantity
        close_ratio = quantity / self.quantity if self.quantity > 0 else 1.0
        if self.side == "long":
            pnl = (price - self.avg_entry_price) * quantity
        else:
            pnl = (self.avg_entry_price - price) * quantity
        pnl -= fee
        self.quantity -= quantity
        self.total_fee_paid += fee
        self.realized_pnl += pnl
        # 同比例释放保证金
        self.margin_used *= (1 - close_ratio)
        return pnl

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": round(self.quantity, 8),
            "avg_entry_price": round(self.avg_entry_price, 6),
            "leverage": self.leverage,
            "margin_used": round(self.margin_used, 4),
            "open_time": self.open_time.isoformat(),
            "realized_pnl": round(self.realized_pnl, 4),
            "total_fee_paid": round(self.total_fee_paid, 4),
            "funding_fee_paid": round(self.funding_fee_paid, 4),
            "add_count": len(self.add_positions),
        }


@dataclass
class VirtualAccountSnapshot:
    """账户快照"""
    timestamp: datetime
    total_equity: float
    available_balance: float
    margin_used: float
    unrealized_pnl: float
    realized_pnl_total: float
    total_fee_paid: float
    position_count: int
    equity_curve_value: float  # 用于资金曲线绘制

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_equity": round(self.total_equity, 4),
            "available_balance": round(self.available_balance, 4),
            "margin_used": round(self.margin_used, 4),
            "unrealized_pnl": round(self.unrealized_pnl, 4),
            "realized_pnl_total": round(self.realized_pnl_total, 4),
            "total_fee_paid": round(self.total_fee_paid, 4),
            "position_count": self.position_count,
        }


class VirtualAccount:
    """
    虚拟账户
    
    管理策略沙盒中的所有资金和持仓操作。
    支持多交易对、多方向持仓，实时计算盈亏。
    """

    def __init__(self, sandbox_id: str, initial_capital: float = 10000.0,
                 config: Dict[str, Any] = None):
        self.sandbox_id = sandbox_id
        self.initial_capital = initial_capital
        self.config = config or {}

        # 费率配置
        self._maker_fee_rate = self.config.get("maker_fee_rate", 0.0002)
        self._taker_fee_rate = self.config.get("taker_fee_rate", 0.0005)
        self._funding_rate = 0.0  # 实时更新

        # 杠杆限制
        self._max_leverage = self.config.get("max_leverage", 15)
        self._max_position_ratio = self.config.get("max_position_ratio", 0.8)

        # 风险限制
        self._max_daily_loss_pct = self.config.get("max_daily_loss_pct", 0.10)
        self._max_drawdown_pct = self.config.get("max_drawdown_pct", 0.25)
        self._max_concurrent_positions = self.config.get("max_concurrent_positions", 6)

        # 资金状态
        self.balance = initial_capital
        self.available_balance = initial_capital
        self.margin_used = 0.0
        self.realized_pnl_total = 0.0
        self.total_fee_paid = 0.0
        self.total_funding_paid = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

        # 持仓管理
        self._positions: Dict[str, VirtualPosition] = {}
        self._position_lock = threading.RLock()

        # 快照历史（资金曲线）
        self._snapshots: List[VirtualAccountSnapshot] = []
        self._max_snapshots = self.config.get("max_snapshots", 5000)
        
        # 每日统计
        self._daily_pnl: Dict[str, float] = defaultdict(float)
        self._daily_start_equity = initial_capital

        # 状态
        self._is_frozen = False
        self._freeze_reason = ""

        # 创建时记录快照
        self._take_snapshot({})

        logger.info(f"VirtualAccount '{sandbox_id}' created: capital={initial_capital}")

    # ── 资金查询 ─────────────────────────────────────────────────

    def total_equity(self, prices: Dict[str, float] = None) -> float:
        """总权益 = 余额 + 未实现盈亏"""
        unrealized = self._calc_total_unrealized_pnl(prices or {})
        return self.balance + unrealized

    def margin_ratio(self) -> float:
        """保证金使用率"""
        equity = self.total_equity()
        if equity <= 0:
            return 1.0
        return self.margin_used / equity

    @property
    def is_frozen(self) -> bool:
        return self._is_frozen

    @property
    def freeze_reason(self) -> str:
        return self._freeze_reason

    @property
    def position_count(self) -> int:
        with self._position_lock:
            return len(self._positions)

    @property
    def peak_equity(self) -> float:
        """历史最高权益"""
        if not self._snapshots:
            return self.initial_capital
        return max(s.total_equity for s in self._snapshots)

    @property
    def max_drawdown(self) -> float:
        """当前最大回撤"""
        peak = self.peak_equity
        current = self.total_equity()
        if peak <= 0:
            return 0.0
        return (peak - current) / peak

    @property
    def daily_pnl(self) -> float:
        """今日盈亏"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self._daily_pnl.get(today, 0.0)

    # ── 持仓查询 ─────────────────────────────────────────────────

    def get_position(self, symbol: str) -> Optional[VirtualPosition]:
        with self._position_lock:
            return self._positions.get(symbol)

    def get_all_positions(self) -> List[VirtualPosition]:
        with self._position_lock:
            return list(self._positions.values())

    def get_positions_summary(self, prices: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """获取持仓摘要（含当前价格盈亏）"""
        prices = prices or {}
        result = []
        with self._position_lock:
            for symbol, pos in self._positions.items():
                current_price = prices.get(symbol, pos.avg_entry_price)
                summary = pos.to_dict()
                summary["current_price"] = round(current_price, 6)
                summary["unrealized_pnl"] = round(pos.unrealized_pnl(current_price), 4)
                summary["unrealized_pnl_pct"] = round(pos.unrealized_pnl_pct(current_price), 4)
                result.append(summary)
        return result

    def _calc_total_unrealized_pnl(self, prices: Dict[str, float]) -> float:
        """计算总未实现盈亏"""
        total = 0.0
        with self._position_lock:
            for symbol, pos in self._positions.items():
                current_price = prices.get(symbol, pos.avg_entry_price)
                total += pos.unrealized_pnl(current_price)
        return total

    # ── 交易操作 ─────────────────────────────────────────────────

    def open_position(self, symbol: str, side: str, quantity: float,
                      price: float, leverage: int,
                      is_maker: bool = False) -> Tuple[bool, str, Optional[VirtualPosition]]:
        """
        开仓

        Returns:
            (success, message, position)
        """
        if self._is_frozen:
            return False, f"账户已冻结: {self._freeze_reason}", None

        # 参数校验
        if quantity <= 0:
            return False, "数量必须大于0", None
        if price <= 0:
            return False, "价格必须大于0", None
        if leverage < 1 or leverage > self._max_leverage:
            return False, f"杠杆必须在1-{self._max_leverage}之间", None

        with self._position_lock:
            # 检查同币种已有反向持仓
            if symbol in self._positions:
                existing = self._positions[symbol]
                if existing.side != side and existing.quantity > 0:
                    return False, f"已有{existing.side}持仓，请先平仓", None

            # 检查最大并发持仓
            if symbol not in self._positions and len(self._positions) >= self._max_concurrent_positions:
                return False, f"已达最大并发持仓数 {self._max_concurrent_positions}", None

            # 计算保证金和手续费
            position_value = quantity * price
            margin = position_value / leverage
            fee_rate = self._maker_fee_rate if is_maker else self._taker_fee_rate
            fee = position_value * fee_rate

            # 检查余额
            total_needed = margin + fee
            if self.available_balance < total_needed:
                return False, f"余额不足 (需要{total_needed:.2f}, 可用{self.available_balance:.2f})", None

            # 检查单币种仓位上限
            current_equity = self.total_equity()
            max_pos_value = current_equity * self._max_position_ratio
            if position_value > max_pos_value:
                return False, f"仓位价值({position_value:.2f})超过上限({max_pos_value:.2f})", None

            # 扣款
            self.available_balance -= total_needed
            self.margin_used += margin
            self.total_fee_paid += fee

            # 创建持仓
            position = VirtualPosition(
                symbol=symbol,
                side=side,
                quantity=quantity,
                avg_entry_price=price,
                leverage=leverage,
                margin_used=margin,
                total_fee_paid=fee,
            )
            self._positions[symbol] = position

        logger.info(f"[{self.sandbox_id}] 开仓: {symbol} {side} qty={quantity} "
                    f"@ {price} lev={leverage}x margin={margin:.2f} fee={fee:.4f}")
        return True, "开仓成功", position

    def add_position(self, symbol: str, quantity: float,
                     price: float, is_maker: bool = False) -> Tuple[bool, str]:
        """加仓"""
        if self._is_frozen:
            return False, f"账户已冻结"

        with self._position_lock:
            if symbol not in self._positions:
                return False, "无该币种持仓"

            pos = self._positions[symbol]
            position_value = quantity * price
            additional_margin = position_value / pos.leverage
            fee_rate = self._maker_fee_rate if is_maker else self._taker_fee_rate
            fee = position_value * fee_rate
            total_needed = additional_margin + fee

            if self.available_balance < total_needed:
                return False, f"余额不足 (需要{total_needed:.2f})"

            pos.add(quantity, price, fee)
            self.available_balance -= total_needed
            self.margin_used += additional_margin
            self.total_fee_paid += fee

        logger.info(f"[{self.sandbox_id}] 加仓: {symbol} +{quantity} @ {price}")
        return True, "加仓成功"

    def reduce_position(self, symbol: str, quantity: float,
                        price: float, is_maker: bool = False) -> Tuple[bool, str, float]:
        """
        减仓

        Returns:
            (success, message, realized_pnl)
        """
        with self._position_lock:
            if symbol not in self._positions:
                return False, "无该币种持仓", 0.0

            pos = self._positions[symbol]
            if quantity > pos.quantity:
                quantity = pos.quantity

            position_value = quantity * price
            fee_rate = self._maker_fee_rate if is_maker else self._taker_fee_rate
            fee = position_value * fee_rate

            pnl = pos.reduce(quantity, price, fee)

            # 释放保证金 + 盈亏回到余额
            released_margin = position_value / pos.leverage
            self.margin_used = max(0, self.margin_used - released_margin)
            self.balance += pnl
            self.available_balance += released_margin + pnl
            self.total_fee_paid += fee
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1
            elif pnl < 0:
                self.losing_trades += 1

            # 完全平仓时清理
            if pos.quantity <= 0:
                self.margin_used = max(0, self.margin_used - pos.margin_used)
                self.available_balance += pos.margin_used
                del self._positions[symbol]

        logger.info(f"[{self.sandbox_id}] 减仓: {symbol} -{quantity} @ {price} pnl={pnl:.4f}")
        return True, "减仓成功", pnl

    def close_position(self, symbol: str, price: float,
                       is_maker: bool = False) -> Tuple[bool, str, float]:
        """全部平仓"""
        pos = self.get_position(symbol)
        if pos is None:
            return False, "无该币种持仓", 0.0
        return self.reduce_position(symbol, pos.quantity, price, is_maker)

    def close_all_positions(self, prices: Dict[str, float]) -> Dict[str, float]:
        """一键平仓所有"""
        results = {}
        with self._position_lock:
            symbols = list(self._positions.keys())
        for symbol in symbols:
            price = prices.get(symbol, 0)
            if price > 0:
                ok, msg, pnl = self.close_position(symbol, price)
                results[symbol] = pnl
        return results

    # ── 风控 ─────────────────────────────────────────────────────

    def check_liquidation(self, prices: Dict[str, float]) -> Dict[str, Any]:
        """
        检查爆仓风险

        Returns:
            {liquidation_risk: bool, warnings: [...], critical: [...]}
        """
        warnings = []
        critical = []
        equity = self.total_equity(prices)

        with self._position_lock:
            for symbol, pos in self._positions.items():
                current_price = prices.get(symbol, pos.avg_entry_price)
                pnl_pct = pos.unrealized_pnl_pct(current_price)

                # 爆仓检查：亏损超过保证金的90%
                if pnl_pct < -0.90:
                    critical.append({
                        "symbol": symbol,
                        "reason": f"接近爆仓 (亏损{pnl_pct*100:.1f}%)",
                        "pnl_pct": round(pnl_pct, 4),
                        "margin": round(pos.margin_used, 2)
                    })
                elif pnl_pct < -0.70:
                    warnings.append({
                        "symbol": symbol,
                        "reason": f"高亏损警告 (亏损{pnl_pct*100:.1f}%)",
                        "pnl_pct": round(pnl_pct, 4),
                    })

        # 总回撤检查
        peak = self.peak_equity
        drawdown = (peak - equity) / peak if peak > 0 else 0
        if drawdown > self._max_drawdown_pct:
            critical.append({
                "reason": f"总回撤{drawdown*100:.1f}%超过{self._max_drawdown_pct*100}%上限",
                "drawdown": round(drawdown, 4),
                "equity": round(equity, 2),
                "peak": round(peak, 2),
            })

        # 日亏损检查
        if self.daily_pnl < -self.initial_capital * self._max_daily_loss_pct:
            critical.append({
                "reason": f"日亏损超过{self._max_daily_loss_pct*100}%上限",
                "daily_pnl": round(self.daily_pnl, 2),
            })

        liquidation_risk = len(critical) > 0
        if liquidation_risk:
            self._is_frozen = True
            self._freeze_reason = "; ".join(c["reason"] for c in critical)

        return {
            "liquidation_risk": liquidation_risk,
            "warnings": warnings,
            "critical": critical,
            "equity": round(equity, 4),
            "drawdown_pct": round(drawdown, 4),
        }

    def _auto_liquidate(self, prices: Dict[str, float]) -> None:
        """自动强平爆仓持仓"""
        for symbol, pos in list(self._positions.items()):
            current_price = prices.get(symbol, pos.avg_entry_price)
            pnl_pct = pos.unrealized_pnl_pct(current_price)
            if pnl_pct < -0.95:
                logger.critical(f"[{self.sandbox_id}] 强平: {symbol} {pos.side} "
                               f"亏损{pnl_pct*100:.1f}%")
                self.close_position(symbol, current_price, is_maker=False)

    # ── 快照 ─────────────────────────────────────────────────────

    def _take_snapshot(self, prices: Dict[str, float]) -> None:
        """记录账户快照"""
        equity = self.total_equity(prices)
        unrealized = self._calc_total_unrealized_pnl(prices)

        snap = VirtualAccountSnapshot(
            timestamp=datetime.now(),
            total_equity=equity,
            available_balance=self.available_balance,
            margin_used=self.margin_used,
            unrealized_pnl=unrealized,
            realized_pnl_total=self.realized_pnl_total,
            total_fee_paid=self.total_fee_paid,
            position_count=len(self._positions),
            equity_curve_value=equity,
        )

        self._snapshots.append(snap)
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots = self._snapshots[-self._max_snapshots:]

    def take_snapshot(self, prices: Dict[str, float]) -> VirtualAccountSnapshot:
        """公开的取快照方法"""
        self._take_snapshot(prices)
        return self._snapshots[-1]

    # ── 统计 ─────────────────────────────────────────────────────

    def get_stats(self, prices: Dict[str, float] = None) -> Dict[str, Any]:
        """获取账户统计"""
        prices = prices or {}
        equity = self.total_equity(prices)
        peak = self.peak_equity
        drawdown = (peak - equity) / peak if peak > 0 else 0

        with self._position_lock:
            pos_count = len(self._positions)

        total_trades = self.total_trades
        win_rate = self.winning_trades / total_trades if total_trades > 0 else 0

        return {
            "sandbox_id": self.sandbox_id,
            "initial_capital": round(self.initial_capital, 2),
            "total_equity": round(equity, 4),
            "available_balance": round(self.available_balance, 4),
            "margin_used": round(self.margin_used, 4),
            "margin_ratio": round(self.margin_ratio(), 4),
            "realized_pnl_total": round(self.realized_pnl_total, 4),
            "unrealized_pnl": round(self._calc_total_unrealized_pnl(prices), 4),
            "total_fee_paid": round(self.total_fee_paid, 4),
            "total_funding_paid": round(self.total_funding_paid, 4),
            "roi_pct": round((equity - self.initial_capital) / self.initial_capital * 100, 2),
            "peak_equity": round(peak, 4),
            "max_drawdown_pct": round(drawdown * 100, 2),
            "total_trades": total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate_pct": round(win_rate * 100, 2),
            "position_count": pos_count,
            "daily_pnl": round(self.daily_pnl, 4),
            "is_frozen": self._is_frozen,
            "freeze_reason": self._freeze_reason,
            "snapshot_count": len(self._snapshots),
        }

    def get_equity_curve(self) -> List[Dict[str, Any]]:
        """获取资金曲线"""
        return [s.to_dict() for s in self._snapshots[-200:]]  # 最近200个

    def update_daily_pnl(self, pnl: float) -> None:
        """更新今日盈亏"""
        today = datetime.now().strftime("%Y-%m-%d")
        self._daily_pnl[today] += pnl

    def reset_daily_if_needed(self) -> None:
        """跨日重置日统计"""
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self._daily_pnl:
            self._daily_pnl[today] = 0.0
        # 清理超过7天的记录
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        for key in list(self._daily_pnl.keys()):
            if key < cutoff:
                del self._daily_pnl[key]


# ── 工厂 ─────────────────────────────────────────────────────────

_sandbox_accounts: Dict[str, VirtualAccount] = {}
_accounts_lock = threading.RLock()


def create_virtual_account(sandbox_id: str, initial_capital: float = 10000.0,
                           config: Dict[str, Any] = None) -> VirtualAccount:
    """创建虚拟账户"""
    with _accounts_lock:
        if sandbox_id in _sandbox_accounts:
            logger.warning(f"VirtualAccount '{sandbox_id}' already exists, returning existing")
            return _sandbox_accounts[sandbox_id]
        account = VirtualAccount(sandbox_id, initial_capital, config)
        _sandbox_accounts[sandbox_id] = account
        return account


def get_virtual_account(sandbox_id: str) -> Optional[VirtualAccount]:
    """获取虚拟账户"""
    with _accounts_lock:
        return _sandbox_accounts.get(sandbox_id)


def remove_virtual_account(sandbox_id: str) -> bool:
    """移除虚拟账户"""
    with _accounts_lock:
        if sandbox_id in _sandbox_accounts:
            del _sandbox_accounts[sandbox_id]
            return True
        return False


def list_virtual_accounts() -> List[str]:
    """列出所有虚拟账户"""
    with _accounts_lock:
        return list(_sandbox_accounts.keys())
