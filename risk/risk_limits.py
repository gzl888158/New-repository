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
        self._monitor_task = None

    async def start(self):
        # 幂等启动：避免重复调用创建多套监控循环
        if self._monitor_task is not None and not self._monitor_task.done():
            return
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Risk limits service started")

    async def shutdown(self):
        task = self._monitor_task
        self._monitor_task = None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        logger.info("Risk limits service shutdown")

    async def _monitor_loop(self):
        while True:
            try:
                await self._update_margins()
                await self._check_daily_hourly_limits()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in risk limits monitor loop: {e}")
            await asyncio.sleep(10)

    async def _update_margins(self):
        try:
            checked_query = getattr(self._okx_client, "get_positions_checked", None)
            positions = (
                checked_query()
                if callable(checked_query)
                else self._okx_client.get_positions()
            )
            if positions is None:
                # 查询失败：保留旧 margins，避免清空后跳过保证金限额检查（fail-closed）
                logger.warning("get_positions returned None, keeping previous margins")
                return
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
        try:
            leverage = float(signal_data.get("leverage", 1) or 1)
            quantity = float(signal_data.get("quantity", 0) or 0)
            price = float(signal_data.get("price", 0) or 0)
        except (TypeError, ValueError):
            logger.error(f"Invalid numeric fields in signal: {signal_data}")
            self._record_violation("invalid_signal_fields", symbol, "non-numeric quantity/price/leverage")
            return False
        
        if leverage <= 0:
            logger.warning(f"Invalid leverage {leverage} in signal, rejecting (fail-closed)")
            self._record_violation("invalid_leverage", symbol, f"leverage={leverage}")
            return False
        
        margin_needed = (quantity * price) / leverage
        
        try:
            account_info = self._okx_client.get_account_info()
        except Exception as e:
            logger.error(f"Failed to fetch account info during signal check (fail-closed): {e}")
            self._record_violation("account_unavailable", symbol, str(e))
            return False
        
        try:
            total_equity = float(account_info.get("totalEq") or 0) if account_info else 0.0
        except (TypeError, ValueError):
            total_equity = 0.0
        
        # fail-closed：无法确认账户权益时拒绝，避免保证金限额检查被静默跳过
        if not account_info or total_equity <= 0:
            logger.warning(f"Account equity unavailable (equity={total_equity}), rejecting signal (fail-closed)")
            self._record_violation("account_equity_unavailable", symbol, f"equity={total_equity}")
            return False
        
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
            try:
                asyncio.create_task(
                    self._alert_manager.send_alert(
                        "RISK_LIMIT_VIOLATION",
                        f"{violation_type}: {detail}",
                        severity="WARNING",
                        symbol=symbol
                    )
                )
            except Exception:
                # 告警失败不得影响风控判定结果（fail-safe 非关键路径）
                pass

    def get_limits_status(self) -> Dict[str, Any]:
        try:
            account_info = self._okx_client.get_account_info()
            total_equity = float(account_info.get("totalEq") or 0) if account_info else 0.0
        except (TypeError, ValueError):
            total_equity = 0.0
        
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