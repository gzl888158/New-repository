"""负责持仓保证金、杠杆、开仓数与交易频次等风控限额监控。"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from loguru import logger


class RiskLimits:
    def __init__(self, config: Dict[str, Any], okx_client, alert_manager=None):
        self.config = config
        self._okx_client = okx_client
        self._alert_manager = alert_manager
        
        self._limits = {
            "single_symbol_max_margin": config["risk"].get("single_symbol_max_margin", 0.25),
            "single_strategy_max_margin": config["risk"].get("single_strategy_max_margin", 0.30),
            "max_leverage": config["trading"].get("max_leverage", 20),
            "max_open_positions": config["trading"].get("max_concurrent_positions", 4),
            "daily_max_trades": config["risk"].get("daily_max_trades", 100),
            "hourly_max_trades": config["risk"].get("hourly_max_trades", 20),
        }
        
        self._daily_trade_count = 0
        self._hourly_trade_count = 0
        self._last_daily_reset = datetime.now().date()
        self._last_hourly_reset = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        self._strategy_margins: Dict[str, float] = {}
        self._symbol_margins: Dict[str, float] = {}
        
        self._violations: List[Dict[str, Any]] = []

    async def start(self):
        asyncio.create_task(self._monitor_loop())
        logger.info("Risk limits service started")

    async def shutdown(self):
        logger.info("Risk limits service shutdown")

    async def _monitor_loop(self):
        while True:
            await self._update_margins()
            await self._check_daily_hourly_limits()
            await asyncio.sleep(10)

    async def _update_margins(self):
        try:
            positions = self._okx_client.get_positions()
            self._strategy_margins = {}
            self._symbol_margins = {}
            
            for pos_data in positions:
                position = self._okx_client._parse_position(pos_data)
                if position and position.margin > 0:
                    if position.symbol not in self._symbol_margins:
                        self._symbol_margins[position.symbol] = 0
                    self._symbol_margins[position.symbol] += position.margin
                    
                    strategy_name = pos_data.get("strategy_name", "unknown")
                    if strategy_name not in self._strategy_margins:
                        self._strategy_margins[strategy_name] = 0
                    self._strategy_margins[strategy_name] += position.margin
        except Exception as e:
            logger.error(f"Failed to update margins: {e}")

    async def _check_daily_hourly_limits(self):
        now = datetime.now()
        
        if now.date() != self._last_daily_reset:
            self._daily_trade_count = 0
            self._last_daily_reset = now.date()
            logger.info(f"Daily trade count reset, date: {now.date()}")
        
        if now.replace(minute=0, second=0, microsecond=0) != self._last_hourly_reset:
            self._hourly_trade_count = 0
            self._last_hourly_reset = now.replace(minute=0, second=0, microsecond=0)
            logger.info(f"Hourly trade count reset, hour: {now.hour}")

    def check_signal(self, signal_data: Dict[str, Any]) -> bool:
        symbol = signal_data.get("symbol", "")
        strategy_name = signal_data.get("strategy_name", "")
        leverage = signal_data.get("leverage", 1)
        quantity = signal_data.get("quantity", 0)
        price = signal_data.get("price", 0)
        
        margin_needed = (quantity * price) / leverage
        
        account_info = self._okx_client.get_account_info()
        total_equity = float(account_info.get("totalEq", 0)) if account_info else 0
        
        if total_equity > 0:
            new_symbol_margin = self._symbol_margins.get(symbol, 0) + margin_needed
            if new_symbol_margin / total_equity > self._limits["single_symbol_max_margin"]:
                logger.warning(f"Symbol margin limit exceeded: {symbol} would be {new_symbol_margin/total_equity:.2%} > {self._limits['single_symbol_max_margin']:.2%}")
                self._record_violation("symbol_margin_limit", symbol, f"margin={new_symbol_margin/total_equity:.2%}")
                return False
            
            new_strategy_margin = self._strategy_margins.get(strategy_name, 0) + margin_needed
            if new_strategy_margin / total_equity > self._limits["single_strategy_max_margin"]:
                logger.warning(f"Strategy margin limit exceeded: {strategy_name} would be {new_strategy_margin/total_equity:.2%} > {self._limits['single_strategy_max_margin']:.2%}")
                self._record_violation("strategy_margin_limit", symbol, f"strategy={strategy_name}, margin={new_strategy_margin/total_equity:.2%}")
                return False
        
        if leverage > self._limits["max_leverage"]:
            logger.warning(f"Leverage limit exceeded: {leverage} > {self._limits['max_leverage']}")
            self._record_violation("leverage_limit", symbol, f"leverage={leverage}")
            return False
        
        if self._hourly_trade_count >= self._limits["hourly_max_trades"]:
            logger.warning(f"Hourly trade limit exceeded: {self._hourly_trade_count} >= {self._limits['hourly_max_trades']}")
            self._record_violation("hourly_trade_limit", symbol, f"count={self._hourly_trade_count}")
            return False
        
        if self._daily_trade_count >= self._limits["daily_max_trades"]:
            logger.warning(f"Daily trade limit exceeded: {self._daily_trade_count} >= {self._limits['daily_max_trades']}")
            self._record_violation("daily_trade_limit", symbol, f"count={self._daily_trade_count}")
            return False
        
        return True

    def increment_trade_count(self):
        self._daily_trade_count += 1
        self._hourly_trade_count += 1

    def _record_violation(self, violation_type: str, symbol: str, detail: str):
        violation = {
            "type": violation_type,
            "symbol": symbol,
            "detail": detail,
            "timestamp": datetime.now()
        }
        self._violations.append(violation)
        
        if len(self._violations) > 100:
            self._violations = self._violations[-100:]
        
        if self._alert_manager:
            asyncio.create_task(
                self._alert_manager.send_alert(
                    "RISK_LIMIT_VIOLATION",
                    f"{violation_type}: {detail}",
                    severity="WARNING",
                    symbol=symbol
                )
            )

    def get_limits_status(self) -> Dict[str, Any]:
        account_info = self._okx_client.get_account_info()
        total_equity = float(account_info.get("totalEq", 0)) if account_info else 0
        
        return {
            "limits": self._limits,
            "current": {
                "daily_trades": self._daily_trade_count,
                "hourly_trades": self._hourly_trade_count,
                "symbol_margins": {k: v/total_equity if total_equity > 0 else 0 for k, v in self._symbol_margins.items()},
                "strategy_margins": {k: v/total_equity if total_equity > 0 else 0 for k, v in self._strategy_margins.items()},
            },
            "violations": self._violations[-10:]
        }

    def get_limits(self) -> Dict[str, float]:
        return self._limits