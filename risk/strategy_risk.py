"""负责策略级风控：加仓层数、回撤与每日交易次数限制。"""
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from loguru import logger

from configs.settings import get_currency_tier


class StrategyRiskControl:
    def __init__(self, config: Dict[str, Any], account_manager=None):
        self.config = config
        self._account_manager = account_manager
        self._global_risk = None
        self._max_addition_layers = config["strategies"]["grid"].get("martingale_layers", 5)
        self._max_trend_additions = config["strategies"]["trend"].get("max_additions", 4)
        self._grid_max_drawdown = 0.15
        self._trend_max_drawdown = 0.20
        self._scalping_max_drawdown = 0.10
        self._max_daily_trades = config["trading"]["max_daily_trades_per_symbol"]
        
        self._daily_trade_count: Dict[str, int] = {}
        self._strategy_drawdown: Dict[str, float] = {}
        self._consecutive_losses: Dict[str, int] = {}
        self._last_trade_time: Dict[str, datetime] = {}
        self._position_value_tracker: Dict[str, float] = {}
        
        self._strategy_stats: Dict[str, Dict[str, Any]] = {}
        self._compound_factor = 1.0
        self._last_compound_update = datetime.now()

    def set_global_risk(self, global_risk):
        self._global_risk = global_risk

    def check_addition_limit(self, strategy_name: str, symbol: str, current_additions: int) -> bool:
        if strategy_name == "grid":
            return current_additions < self._max_addition_layers
        elif strategy_name == "trend":
            return current_additions < self._max_trend_additions
        return True

    def check_drawdown(self, strategy_name: str, symbol: str, drawdown: float) -> bool:
        key = f"{strategy_name}:{symbol}"
        self._strategy_drawdown[key] = drawdown
        
        max_drawdown = self._get_max_drawdown(strategy_name)
        if drawdown >= max_drawdown:
            logger.warning(f"Strategy {strategy_name} on {symbol} exceeded max drawdown: {drawdown:.2%}")
            return False
        return True

    def _get_max_drawdown(self, strategy_name: str) -> float:
        if strategy_name == "grid":
            return self._grid_max_drawdown
        elif strategy_name == "trend":
            return self._trend_max_drawdown
        elif strategy_name == "scalping":
            return self._scalping_max_drawdown
        return 0.20

    def check_daily_trade_limit(self, symbol: str) -> bool:
        today = datetime.now().date()
        key = f"{symbol}:{today}"

        if key not in self._daily_trade_count:
            self._daily_trade_count[key] = 0

        if self._daily_trade_count[key] >= self._max_daily_trades:
            logger.warning(f"Symbol {symbol} exceeded daily trade limit: {self._max_daily_trades}")
            return False

        # 仅检查不递增，由 commit_trade 在下单成功后调用
        return True

    def commit_trade(self, symbol: str):
        """下单成功后才递增交易计数，避免下单失败但不回滚"""
        today = datetime.now().date()
        key = f"{symbol}:{today}"
        self._daily_trade_count[key] = self._daily_trade_count.get(key, 0) + 1

    def rollback_trade(self, symbol: str):
        """下单失败时回滚交易计数"""
        today = datetime.now().date()
        key = f"{symbol}:{today}"
        if key in self._daily_trade_count and self._daily_trade_count[key] > 0:
            self._daily_trade_count[key] -= 1

    def check_consecutive_losses(self, strategy_name: str, symbol: str, is_profitable: bool) -> bool:
        key = f"{strategy_name}:{symbol}"

        if is_profitable:
            self._consecutive_losses[key] = 0
            return True

        self._consecutive_losses[key] = self._consecutive_losses.get(key, 0) + 1

        max_losses = self.config["trading"]["max_consecutive_losses"]
        if self._consecutive_losses[key] >= max_losses:
            logger.warning(f"Strategy {strategy_name} on {symbol} has {max_losses} consecutive losses")
            return False

        return True

    def _has_excessive_consecutive_losses(self, strategy_name: str, symbol: str) -> bool:
        """仅检查连续亏损次数是否超限，不修改计数（供 validate_signal 使用）"""
        key = f"{strategy_name}:{symbol}"
        max_losses = self.config["trading"]["max_consecutive_losses"]
        current = self._consecutive_losses.get(key, 0)
        if current >= max_losses:
            logger.warning(f"Strategy {strategy_name} on {symbol} has {current} consecutive losses (max {max_losses})")
            return True
        return False

    def check_slippage(self, symbol: str, expected_price: float, actual_price: float, 
                       is_stop_loss: bool = False) -> bool:
        tier = get_currency_tier(symbol, self.config)
        tier_settings = self.config["currencies"][f"{tier}_settings"]
        base_slippage = tier_settings["slippage"]
        
        slippage_tolerance = base_slippage * 2 if is_stop_loss else base_slippage
        
        if expected_price == 0:
            return True
        
        slippage = abs(actual_price - expected_price) / expected_price
        
        if slippage > slippage_tolerance:
            logger.warning(f"Slippage {slippage:.4%} exceeds tolerance {slippage_tolerance:.4%} for {symbol}")
            if is_stop_loss:
                logger.info("Stop loss order allowed to proceed despite slippage")
                return True
            return False
        
        return True

    def validate_signal(self, signal_data: Dict[str, Any]) -> bool:
        if self._global_risk and not self._global_risk.can_trade():
            logger.warning("Signal rejected: global risk control paused")
            return False

        checks = [
            ("daily_trade_limit", self.check_daily_trade_limit(signal_data["symbol"])),
            ("position_limit", self.check_position_limit(signal_data)),
            ("leverage_limit", self.check_leverage_limit(signal_data)),
            ("confidence_threshold", self.check_confidence_threshold(signal_data)),
            ("trading_hours", self.check_trading_hours(signal_data)),
            ("margin_availability", self.check_margin_availability(signal_data)),
            ("consecutive_losses", not self._has_excessive_consecutive_losses(signal_data.get("strategy_name", ""), signal_data.get("symbol", "")))
        ]
        
        for check_name, result in checks:
            if not result:
                logger.warning(f"Signal validation failed: {check_name}")
                return False
        
        return True
    
    def check_margin_availability(self, signal_data: Dict[str, Any]) -> bool:
        if self._account_manager is None:
            return True
        
        strategy_name = signal_data.get("strategy_name", "grid")
        quantity = signal_data["quantity"]
        price = signal_data["price"]
        leverage = signal_data["leverage"]
        
        margin_required = quantity * price / leverage
        
        if not self._account_manager.can_open_position(strategy_name, margin_required):
            available = self._account_manager.get_available_capital(strategy_name)
            logger.warning(f"Insufficient margin for {strategy_name} {signal_data['symbol']}: "
                          f"required={margin_required:.2f}, available={available:.2f}")
            return False
        
        return True

    def check_position_limit(self, signal_data: Dict[str, Any]) -> bool:
        tier = get_currency_tier(signal_data["symbol"], self.config)
        tier_settings = self.config["currencies"][f"{tier}_settings"]
        position_limit = tier_settings["position_limit"]
        
        trading_capital = self.config["trading"]["total_capital"] * self.config["trading"]["trading_capital_ratio"]
        max_position_value = trading_capital * position_limit
        
        position_value = signal_data["quantity"] * signal_data["price"]
        
        if position_value > max_position_value:
            logger.warning(f"Position {position_value:.2f} exceeds limit {max_position_value:.2f} for {signal_data['symbol']}")
            return False
        
        return True

    def check_leverage_limit(self, signal_data: Dict[str, Any]) -> bool:
        strategy_name = signal_data.get("strategy_name", "")
        # 现货策略不使用杠杆，直接放行
        if strategy_name in ("spot_grid", "spot_martingale"):
            return True

        tier = get_currency_tier(signal_data["symbol"], self.config)
        tier_settings = self.config["currencies"][f"{tier}_settings"]
        leverage_max = tier_settings["leverage_max"]
        leverage_min = tier_settings["leverage_min"]
        
        leverage = signal_data["leverage"]
        
        if leverage > leverage_max or leverage < leverage_min:
            logger.warning(f"Leverage {leverage} outside allowed range [{leverage_min}, {leverage_max}] for {signal_data['symbol']}")
            return False
        
        return True

    def check_confidence_threshold(self, signal_data: Dict[str, Any]) -> bool:
        confidence = signal_data.get("confidence", 0.0)
        # 阈值改为可配置（之前硬编码0.3导致低置信度信号被全量拒绝，与 adaptive_controller 的阈值放宽机制冲突）
        threshold = float(self.config.get("trading", {}).get("min_signal_confidence", 0.15))

        if confidence < threshold:
            logger.warning(f"Signal confidence {confidence:.2f} below threshold {threshold}")
            return False

        return True

    def check_trading_hours(self, signal_data: Dict[str, Any]) -> bool:
        strategy_name = signal_data.get("strategy_name", "")
        
        if strategy_name == "scalping":
            start_hour = self.config["strategies"]["scalping"].get("run_hours_start", 8)
            end_hour = self.config["strategies"]["scalping"].get("run_hours_end", 24)
            current_hour = datetime.now().hour
            
            if current_hour < start_hour or current_hour >= end_hour:
                logger.info(f"Scalping trading hours ({start_hour}:00 - {end_hour}:00) not active, rejecting signal")
                return False
        
        return True

    def update_trade_result(self, strategy_name: str, symbol: str, pnl: float):
        key = f"{strategy_name}:{symbol}"
        if pnl >= 0:
            self._consecutive_losses[key] = 0
        else:
            self._consecutive_losses[key] = self._consecutive_losses.get(key, 0) + 1
        
        self._update_strategy_stats(strategy_name, pnl)
        self._update_compound_factor()
        
        logger.info(f"Trade result updated: {strategy_name} {symbol} PnL={pnl:.2f}")
    
    def _update_strategy_stats(self, strategy_name: str, pnl: float):
        if strategy_name not in self._strategy_stats:
            self._strategy_stats[strategy_name] = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "total_pnl": 0.0
            }
        
        stats = self._strategy_stats[strategy_name]
        stats["total_trades"] += 1
        
        if pnl >= 0:
            stats["winning_trades"] += 1
            stats["avg_win"] = (stats["avg_win"] * (stats["winning_trades"] - 1) + pnl) / stats["winning_trades"]
        else:
            stats["losing_trades"] += 1
            stats["avg_loss"] = (stats["avg_loss"] * (stats["losing_trades"] - 1) + abs(pnl)) / stats["losing_trades"]
        
        stats["total_pnl"] += pnl
    
    def _update_compound_factor(self):
        now = datetime.now()
        if (now - self._last_compound_update).total_seconds() < 3600:
            return
        
        total_pnl = sum(stats["total_pnl"] for stats in self._strategy_stats.values())
        if total_pnl > 0:
            self._compound_factor = min(2.0, self._compound_factor + total_pnl * 0.01)
        else:
            self._compound_factor = max(0.5, self._compound_factor * 0.99)
        
        self._last_compound_update = now
        logger.info(f"Compound factor updated: {self._compound_factor:.4f}")
    
    def calculate_kelly_fraction(self, strategy_name: str) -> float:
        if strategy_name not in self._strategy_stats:
            return 0.1
        
        stats = self._strategy_stats[strategy_name]
        if stats["total_trades"] < 10:
            return 0.08
        
        win_rate = stats["winning_trades"] / stats["total_trades"]
        win_loss_ratio = stats["avg_win"] / stats["avg_loss"] if stats["avg_loss"] > 0 else 1.0
        
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio if win_loss_ratio > 0 else 0
        
        return max(0.05, min(0.30, kelly))
    
    def calculate_position_size(self, strategy_name: str, symbol: str, price: float, leverage: int) -> float:
        tier = get_currency_tier(symbol, self.config)
        tier_settings = self.config["currencies"][f"{tier}_settings"]
        
        trading_capital = self.config["trading"]["total_capital"] * self.config["trading"]["trading_capital_ratio"]
        allocation = self.config["trading"].get(f"{strategy_name}_allocation", 0.20)
        
        kelly_fraction = self.calculate_kelly_fraction(strategy_name)
        position_limit = tier_settings["position_limit"]
        
        base_position = trading_capital * min(allocation, position_limit) * kelly_fraction * self._compound_factor
        quantity = base_position / price
        
        min_quantity = 0.0001
        max_quantity = (trading_capital * min(allocation, position_limit)) / price
        
        return max(min_quantity, min(max_quantity, quantity))
    
    def get_compound_factor(self) -> float:
        return self._compound_factor
    
    def get_strategy_stats(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        return self._strategy_stats.get(strategy_name)