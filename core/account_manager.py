"""账户与保证金管理模块，负责账户信息查询、策略资金分配、杠杆与风险敞口的实时监控和再平衡。"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from loguru import logger

from utils.helpers import safe_float


class AccountManager:
    def __init__(self, config: Dict[str, Any], okx_client, redis_cache, equity_monitor=None):
        self.config = config
        self.okx_client = okx_client
        self.redis_cache = redis_cache
        self.equity_monitor = equity_monitor
        
        self._total_capital = config["trading"]["total_capital"]
        self._trading_capital_ratio = config["trading"]["trading_capital_ratio"]
        self._max_total_leverage = config["trading"]["max_total_leverage"]
        
        self._strategy_allocations = {
            "grid": config["trading"].get("grid_allocation", 0.30),
            "trend": config["trading"].get("trend_allocation", 0.30),
            "scalping": config["trading"].get("scalping_allocation", 0.20),
            "arbitrage": config["trading"].get("arbitrage_allocation", 0.20),
            # 补全现货策略分配（之前缺失导致 can_open_position 对现货策略返回 False）
            "spot_grid": config["trading"].get("spot_grid_allocation", 0.12),
            "spot_martingale": config["trading"].get("spot_martingale_allocation", 0.10),
        }
        
        self._strategy_margin: Dict[str, float] = {}
        self._strategy_equity: Dict[str, float] = {}
        self._available_capital: Dict[str, float] = {}
        
        self._total_margin_used = 0.0
        self._total_unrealized_pnl = 0.0
        self._current_total_leverage = 0.0
        self._open_position_count = 0
        
        self._pending_orders: Dict[str, Dict[str, Any]] = {}
        self._pending_orders_margin = 0.0

        # symbol -> strategy 精确映射（由策略实例注册，优先于 tier 启发式）
        # 用于区分 arbitrage/spot_grid/spot_martingale 等无法靠 symbol 后缀区分的策略
        self._symbol_strategy_map: Dict[str, str] = {}
        
        self._long_exposure = 0.0
        self._short_exposure = 0.0
        self._net_exposure = 0.0

        # 逐币种未实现盈亏（供 AGI 逐币种精细杠杆/间距守卫使用）
        self._symbol_unrealized_pnl: Dict[str, float] = {}

        # 逐币种现货持币余额（供 AGI 现货持有守卫使用）
        self._spot_holdings: Dict[str, float] = {}
        
        self._last_rebalance_time = datetime.now()
        self._rebalance_interval = timedelta(hours=4)
        
        self._monitor_task = None
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """获取账户信息（供 signal_processor 等调用）"""
        try:
            info = self.okx_client.get_account_info()
            if info is None:
                # 源头告警：查询返回空（不抛异常），下游 fail-closed 依赖此可观测性
                logger.warning("Account info query returned None (empty response)")
            return info
        except Exception as e:
            logger.warning(f"Failed to get account info: {e}")
            return None

    async def start(self):
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info("AccountManager started")
    
    async def _monitor_loop(self):
        while True:
            await self._update_account_status()
            await self._update_pending_orders()
            await self._check_leverage_exposure()
            await asyncio.sleep(30)
    
    async def _update_account_status(self):
        try:
            account_info = await self.okx_client.get_account_info_async()
            account = None
            
            if account_info:
                account = self.okx_client._parse_account_info(account_info)
                self._spot_holdings = self._extract_spot_holdings(account_info)
            
            positions = await self.okx_client.get_positions_async()
            if not positions:
                positions = []
            
            # 降级方案：API返回的账户信息不完整或失败时，用持仓数据估算
            if not account or account.used_margin == 0 or account.total_equity == 0:
                account = self._estimate_account_from_positions(positions, account)
            
            self._total_unrealized_pnl = account.unrealized_pnl
            self._total_margin_used = account.used_margin

            # 用真实账户权益同步资金基准（替代失真的 config total_capital）
            if account.total_equity > 0:
                self._total_capital = float(account.total_equity)
            
            self._strategy_margin = {}
            self._strategy_equity = {}
            self._long_exposure = 0.0
            self._short_exposure = 0.0
            self._symbol_unrealized_pnl = {}
            open_count = 0
            
            for pos_data in positions:
                position = self.okx_client._parse_position(pos_data)
                if position:
                    if abs(float(position.quantity)) > 0:
                        open_count += 1
                    strategy = self._infer_strategy_from_symbol(position.symbol)
                    if strategy not in self._strategy_margin:
                        self._strategy_margin[strategy] = 0.0
                        self._strategy_equity[strategy] = 0.0
                    
                    self._strategy_margin[strategy] += position.margin
                    self._strategy_equity[strategy] += position.unrealized_pnl
                    self._symbol_unrealized_pnl[position.symbol] = (
                        self._symbol_unrealized_pnl.get(position.symbol, 0.0)
                        + float(position.unrealized_pnl)
                    )
                    
                    position_value = float(position.quantity) * position.mark_price
                    if position.side == "long":
                        self._long_exposure += position_value
                    else:
                        self._short_exposure += position_value
            
            self._open_position_count = open_count
            self._net_exposure = abs(self._long_exposure - self._short_exposure)
            
            trading_capital = self._total_capital * self._trading_capital_ratio
            for strategy, allocation in self._strategy_allocations.items():
                allocated = trading_capital * allocation
                used = self._strategy_margin.get(strategy, 0)
                self._available_capital[strategy] = max(0, allocated - used)
            
            if account.total_equity > 0:
                self._current_total_leverage = (self._total_margin_used + abs(self._total_unrealized_pnl)) / account.total_equity
            
            self.redis_cache.set_account_info(account)
            self._last_valid_account = account
            
            # ── 喂入 EquityMonitor：企业级资金变动自适应检测 ──
            if self.equity_monitor is not None and account.total_equity > 0:
                event = self.equity_monitor.feed(
                    equity=account.total_equity,
                    available=account.available_balance if hasattr(account, 'available_balance') else 0.0,
                    margin=account.used_margin,
                    upl=account.unrealized_pnl,
                )
                if event:
                    logger.info(f"EquityMonitor event: {event.event_type.value} "
                                f"(equity={account.total_equity:.2f}, change={event.change_pct:.2%})")
            
            logger.debug(f"Account updated: margin={self._total_margin_used:.2f}, PnL={self._total_unrealized_pnl:.2f}, "
                        f"leverage={self._current_total_leverage:.2f}x, net_exposure={self._net_exposure:.2f}")
        except Exception as e:
            logger.error(f"Failed to update account status: {e}")
    
    def _estimate_account_from_positions(self, positions, partial_account=None):
        """降级方案：用持仓数据估算账户信息"""
        from core.models import AccountInfo
        
        total_equity = 0.0
        used_margin = 0.0
        unrealized_pnl = 0.0
        
        if partial_account:
            total_equity = partial_account.total_equity
            unrealized_pnl = partial_account.unrealized_pnl
            if partial_account.used_margin > 0:
                used_margin = partial_account.used_margin
        
        # 从持仓数据计算已用保证金和未实现盈亏
        pos_used_margin = 0.0
        pos_upl = 0.0
        for pos_data in positions:
            position = self.okx_client._parse_position(pos_data)
            if position:
                pos_used_margin += position.margin
                pos_upl += position.unrealized_pnl
        
        if used_margin == 0 and pos_used_margin > 0:
            used_margin = pos_used_margin
        
        if unrealized_pnl == 0 and pos_upl != 0:
            unrealized_pnl = pos_upl
        
        # 如果总权益为0，使用上次有效账户数据
        if total_equity == 0 and hasattr(self, '_last_valid_account') and self._last_valid_account:
            total_equity = self._last_valid_account.total_equity
            if used_margin == 0:
                used_margin = self._last_valid_account.used_margin
        
        # 最终降级：使用配置的总资金
        if total_equity == 0:
            total_equity = self._total_capital
        
        margin_rate = used_margin / total_equity if total_equity > 0 else 0.0
        
        return AccountInfo(
            total_equity=total_equity,
            available_balance=max(0, total_equity - used_margin),  # equity已含unrealized_pnl，不应再减
            used_margin=used_margin,
            unrealized_pnl=unrealized_pnl,
            margin_rate=margin_rate,
            timestamp=datetime.now()
        )
    
    async def _update_pending_orders(self):
        try:
            orders = await asyncio.to_thread(self.okx_client.get_orders)
            if not orders:
                self._pending_orders = {}
                self._pending_orders_margin = 0.0
                return
            
            pending_margin = 0.0
            new_pending_orders = {}
            
            for order_data in orders:
                order_id = order_data.get("ordId", "")
                if not order_id:
                    continue
                
                state = order_data.get("state", "")
                if state not in ("live", "pending"):
                    continue
                
                symbol = order_data.get("instId", "")
                side = order_data.get("side", "")
                order_type = order_data.get("ordType", "")
                quantity = float(order_data.get("sz", "0"))
                price = float(order_data.get("px", "0"))
                leverage = int(order_data.get("lever", "1"))
                
                margin = quantity * price / leverage if leverage > 0 else 0
                pending_margin += margin
                
                new_pending_orders[order_id] = {
                    "symbol": symbol,
                    "side": side,
                    "type": order_type,
                    "quantity": quantity,
                    "price": price,
                    "leverage": leverage,
                    "margin": margin,
                    "state": state,
                    "strategy": self._infer_strategy_from_symbol(symbol)
                }
            
            self._pending_orders = new_pending_orders
            self._pending_orders_margin = pending_margin
            
            logger.debug(f"Pending orders: {len(self._pending_orders)} orders, margin={pending_margin:.2f}")
        except Exception as e:
            logger.error(f"Failed to update pending orders: {e}")
    
    def register_symbol_strategy(self, symbol: str, strategy_name: str) -> None:
        """注册 symbol -> strategy 精确映射，优先于 tier 启发式推断。

        用于区分 arbitrage/spot_grid/spot_martingale 等无法仅靠 symbol 后缀区分的策略。
        """
        self._symbol_strategy_map[symbol] = strategy_name

    def _infer_strategy_from_symbol(self, symbol: str) -> str:
        # 1) 显式映射优先（由策略实例在启动时注册）
        if symbol in self._symbol_strategy_map:
            return self._symbol_strategy_map[symbol]

        # 2) 现货标的（-USDT 且非 -USDT-SWAP）：按启用的现货策略兜底，避免误判为 trend/grid/scalping
        if symbol.endswith("-USDT") and not symbol.endswith("-USDT-SWAP"):
            strategies = self.config.get("strategies", {})
            if strategies.get("spot_grid", {}).get("enabled", False):
                return "spot_grid"
            if strategies.get("spot_martingale", {}).get("enabled", False):
                return "spot_martingale"
            return "spot_grid"

        # 3) 合约标的：保留 tier 启发式
        tier = self._get_tier_from_symbol(symbol)
        if tier == "tier1":
            return "trend"
        elif tier == "tier2":
            return "grid"
        elif tier == "tier3":
            return "scalping"
        return "grid"
    
    def _get_tier_from_symbol(self, symbol: str) -> str:
        base = symbol.replace("-USDT", "")
        for tier in ["tier1", "tier2", "tier3"]:
            if base in self.config["currencies"].get(f"{tier}_symbols", []):
                return tier
        return "tier2"
    
    async def _check_leverage_exposure(self):
        total_leverage_with_pending = (self._total_margin_used + self._pending_orders_margin + abs(self._total_unrealized_pnl)) / max(self._total_capital * self._trading_capital_ratio, 1)
        
        if total_leverage_with_pending >= self._max_total_leverage:
            logger.warning(f"Total leverage (with pending) {total_leverage_with_pending:.2f}x exceeds max {self._max_total_leverage}x")
            await self._reduce_exposure()
    
    async def _reduce_exposure(self):
        positions = await self.okx_client.get_positions_async()
        if not positions:
            return
        
        position_list = []
        for pos_data in positions:
            position = self.okx_client._parse_position(pos_data)
            if position:
                pnl_ratio = position.unrealized_pnl / position.margin if position.margin > 0 else 0
                position_list.append({
                    "position": position,
                    "pnl_ratio": pnl_ratio,
                    "leverage": position.leverage
                })
        
        position_list.sort(key=lambda x: (x["pnl_ratio"], -x["leverage"]))
        
        target_leverage = self._max_total_leverage * 0.9
        while self._current_total_leverage > target_leverage and position_list:
            item = position_list.pop(0)
            position = item["position"]
            
            side = "sell" if position.side == "long" else "buy"
            reduce_quantity = abs(float(position.quantity)) * 0.3
            
            try:
                await asyncio.to_thread(
                    self.okx_client.place_order,
                    symbol=position.symbol,
                    side=side,
                    order_type="market",
                    quantity=reduce_quantity,
                    leverage=position.leverage
                )
                logger.info(f"Reduced exposure: {position.symbol}, reduced {reduce_quantity:.4f}")
            except Exception as e:
                logger.error(f"Failed to reduce exposure: {e}")
            
            await self._update_account_status()
    
    def get_available_capital(self, strategy_name: str) -> float:
        available = self._available_capital.get(strategy_name, 0.0)
        pending_for_strategy = sum(
            order["margin"] for order in self._pending_orders.values()
            if order["strategy"] == strategy_name
        )
        return max(0, available - pending_for_strategy)
    
    def get_strategy_margin(self, strategy_name: str) -> float:
        return self._strategy_margin.get(strategy_name, 0.0)
    
    def get_strategy_equity(self, strategy_name: str) -> float:
        return self._strategy_equity.get(strategy_name, 0.0)
    
    def get_total_margin_used(self) -> float:
        return self._total_margin_used
    
    def get_total_unrealized_pnl(self) -> float:
        return self._total_unrealized_pnl

    def get_symbol_unrealized_pnl(self) -> Dict[str, float]:
        """返回逐币种未实现盈亏 {symbol: upl}，供 AGI 逐币种精细杠杆/间距守卫使用。"""
        return dict(self._symbol_unrealized_pnl)

    @staticmethod
    def _extract_spot_holdings(account_info: Dict[str, Any]) -> Dict[str, float]:
        """从 /account/balance 的 details 中提取逐币种现货持币余额（非 USDT、cashBal>0）。

        - 现货持有 = 非 USDT 币种的现货现金余额（cashBal，回退 availBal）。
        - 逐币种余额为各币种本位数量（BTC/ETH 等），AGI 侧用「持币种类数」做
          过度分散收敛，不依赖估值（避免引入实时价格与换算复杂度）。
        """
        holdings: Dict[str, float] = {}
        details = account_info.get("details", []) if isinstance(account_info, dict) else []
        for d in details:
            if not isinstance(d, dict):
                continue
            ccy = str(d.get("ccy", "") or "")
            if not ccy or ccy == "USDT":
                continue
            cash = safe_float(d.get("cashBal"), safe_float(d.get("availBal"), 0.0))
            if cash > 0:
                holdings[ccy] = cash
        return holdings

    def get_spot_holdings(self) -> Dict[str, float]:
        """返回逐币种现货持币余额 {ccy: 余额}，供 AGI 现货持有守卫使用。"""
        return dict(self._spot_holdings)
    
    def get_current_leverage(self) -> float:
        return self._current_total_leverage
    
    def get_pending_orders(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._pending_orders)
    
    def get_pending_margin(self) -> float:
        return self._pending_orders_margin
    
    def get_net_exposure(self) -> float:
        return self._net_exposure
    
    def get_long_exposure(self) -> float:
        return self._long_exposure
    
    def get_short_exposure(self) -> float:
        return self._short_exposure
    
    def get_total_equity(self) -> float:
        """返回真实账户权益（USDT）。资金自适应分配引擎与下单链路的权益来源。"""
        return float(self._total_capital)
    
    def get_total_capital(self) -> float:
        """别名：真实账户权益（USDT），与 get_total_equity 同源。"""
        return float(self._total_capital)
    
    def update_total_capital(self, equity: float):
        """用真实权益更新资金基准（供外部同步）。"""
        if equity and equity > 0:
            self._total_capital = float(equity)
    
    def get_open_position_count(self) -> int:
        """当前未平仓持仓数量（专一分配「单持仓」门控依据）。"""
        return int(self._open_position_count)
    
    def can_open_position(self, strategy_name: str, margin_required: float) -> bool:
        available = self.get_available_capital(strategy_name)
        total_available = self._total_capital * self._trading_capital_ratio - (self._total_margin_used + self._pending_orders_margin)
        
        if available >= margin_required:
            return True
        
        if total_available >= margin_required * 0.5:
            logger.info(f"Cross-strategy capital transfer possible for {strategy_name}")
            return True
        
        return False
    
    def notify_order_placed(self, strategy_name: str, margin_used: float):
        if strategy_name in self._strategy_margin:
            self._strategy_margin[strategy_name] += margin_used
            self._total_margin_used += margin_used
            
            trading_capital = self._total_capital * self._trading_capital_ratio
            allocated = trading_capital * self._strategy_allocations.get(strategy_name, 0)
            used = self._strategy_margin.get(strategy_name, 0)
            self._available_capital[strategy_name] = max(0, allocated - used)
        
        logger.debug(f"Order placed: {strategy_name}, margin_used={margin_used:.2f}")
    
    def notify_order_filled(self, strategy_name: str, margin_used: float):
        self.notify_order_placed(strategy_name, margin_used)
    
    def notify_order_canceled(self, strategy_name: str, margin_released: float):
        if strategy_name in self._strategy_margin:
            self._strategy_margin[strategy_name] = max(0, self._strategy_margin[strategy_name] - margin_released)
            self._total_margin_used = max(0, self._total_margin_used - margin_released)
            
            trading_capital = self._total_capital * self._trading_capital_ratio
            allocated = trading_capital * self._strategy_allocations.get(strategy_name, 0)
            used = self._strategy_margin.get(strategy_name, 0)
            self._available_capital[strategy_name] = max(0, allocated - used)
        
        logger.debug(f"Order canceled: {strategy_name}, margin_released={margin_released:.2f}")
    
    async def rebalance(self):
        now = datetime.now()
        if (now - self._last_rebalance_time) < self._rebalance_interval:
            return
        
        logger.info("Starting account rebalancing...")
        
        trading_capital = self._total_capital * self._trading_capital_ratio
        
        for strategy, allocation in self._strategy_allocations.items():
            target_margin = trading_capital * allocation
            current_margin = self._strategy_margin.get(strategy, 0)
            diff = target_margin - current_margin
            
            if abs(diff) / target_margin > 0.20:
                logger.info(f"Rebalancing {strategy}: current={current_margin:.2f}, target={target_margin:.2f}, diff={diff:.2f}")
                
                if diff > 0:
                    pass
                else:
                    await self._reduce_strategy_margin(strategy, abs(diff))
        
        self._last_rebalance_time = now
        logger.info("Account rebalancing completed")
    
    async def _reduce_strategy_margin(self, strategy_name: str, amount: float):
        positions = await self.okx_client.get_positions_async()
        strategy_positions = []
        
        for pos_data in positions:
            position = self.okx_client._parse_position(pos_data)
            if position:
                if self._infer_strategy_from_symbol(position.symbol) == strategy_name:
                    strategy_positions.append(position)
        
        strategy_positions.sort(key=lambda p: abs(p.unrealized_pnl) / p.margin if p.margin > 0 else 0)
        
        reduced_amount = 0.0
        for position in strategy_positions:
            if reduced_amount >= amount:
                break
            
            reduce_ratio = min(1.0, (amount - reduced_amount) / position.margin)
            reduce_quantity = abs(float(position.quantity)) * reduce_ratio
            
            side = "sell" if position.side == "long" else "buy"
            
            try:
                await asyncio.to_thread(
                    self.okx_client.place_order,
                    symbol=position.symbol,
                    side=side,
                    order_type="market",
                    quantity=reduce_quantity,
                    leverage=position.leverage
                )
                reduced_amount += position.margin * reduce_ratio
                logger.info(f"Rebalance: reduced {strategy_name} {position.symbol} by {reduce_quantity:.4f}")
            except Exception as e:
                logger.error(f"Failed to reduce {strategy_name} margin: {e}")
    
    def get_account_summary(self) -> Dict[str, Any]:
        trading_capital = self._total_capital * self._trading_capital_ratio
        
        return {
            "total_capital": self._total_capital,
            "trading_capital": trading_capital,
            "total_margin_used": self._total_margin_used,
            "pending_margin": self._pending_orders_margin,
            "total_unrealized_pnl": self._total_unrealized_pnl,
            "current_leverage": self._current_total_leverage,
            "max_leverage": self._max_total_leverage,
            "long_exposure": self._long_exposure,
            "short_exposure": self._short_exposure,
            "net_exposure": self._net_exposure,
            "strategy_allocation": {
                strategy: {
                    "allocation": allocation,
                    "margin_used": self._strategy_margin.get(strategy, 0),
                    "equity": self._strategy_equity.get(strategy, 0),
                    "available": self.get_available_capital(strategy)
                }
                for strategy, allocation in self._strategy_allocations.items()
            },
            "pending_orders_count": len(self._pending_orders)
        }
    
    def update_strategy_allocation(self, strategy_name: str, new_allocation: float):
        if strategy_name in self._strategy_allocations:
            total_allocation = sum(
                v for k, v in self._strategy_allocations.items()
                if k != strategy_name
            )
            
            if total_allocation + new_allocation <= 1.0:
                self._strategy_allocations[strategy_name] = new_allocation
                logger.info(f"Updated {strategy_name} allocation to {new_allocation:.2%}")
            else:
                logger.warning(f"Cannot update allocation: total would exceed 100%")

    def set_strategy_allocations(self, weights: Dict[str, float]) -> None:
        """单一写入口：批量更新策略资金分配百分比。

        企业级修复（P1-④）：AllocationAgent 再平衡后的最终归一化权重统一经由此方法写入，
        同时同步到 config，使 AccountManager / AdaptivePositionSizer / AllocationAgent
        三处读取同一数据源，消除「初始化快照永不更新」导致的分配值分歧。
        """
        trading_cfg = self.config.get("trading", {})
        for strategy, weight in weights.items():
            if strategy in self._strategy_allocations:
                self._strategy_allocations[strategy] = float(weight)
            trading_cfg[f"{strategy}_allocation"] = round(float(weight), 6)
        logger.info(f"Strategy allocations updated via single write entry: "
                    f"{ {k: round(v, 4) for k, v in self._strategy_allocations.items()} }")