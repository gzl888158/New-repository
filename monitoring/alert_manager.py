"""告警管理器：负责告警的生成、去重、分级与通知分发。"""
import asyncio
import requests
import json
from collections import deque
from datetime import datetime
from typing import Dict, Any, Optional
from loguru import logger


class AlertManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._webhook_url = config["monitoring"].get("alert_webhook", "")
        self._enabled = config["notifications"]["enabled"]
        self._alert_level = config["notifications"]["level"]
        self._providers = config["notifications"].get("providers", ["webhook"])
        
        self._telegram_token = config.get("telegram", {}).get("bot_token", "")
        self._telegram_chat_id = config.get("telegram", {}).get("chat_id", "")
        
        self._alert_history = deque(maxlen=1000)
        self._rate_limit: Dict[str, deque] = {}
        self._rate_limit_window = 60
    
    async def send_alert(self, alert_type: str, message: str, severity: str = "INFO", 
                         symbol: str = "", metadata: Dict[str, Any] = None):
        if not self._enabled:
            return
        
        if self._should_filter(severity):
            return
        
        # 去重 key = alert_type + symbol，确保不同品种的同类告警互不阻塞
        dedup_key = f"{alert_type}:{symbol}" if symbol else alert_type
        if self._is_rate_limited(dedup_key):
            logger.debug(f"Alert rate limited: {dedup_key}")
            return

        # 相同内容去重：5分钟内相同 key+message 不重复发送（用于持续性告警如 LIQUIDATION）
        content_key = f"{dedup_key}:{message[:80]}"
        if self._is_rate_limited(content_key, window=300):
            logger.debug(f"Alert content dedup: {content_key}")
            return
        
        alert_id = f"{alert_type}:{datetime.now().isoformat()}"
        
        alert_data = {
            "alert_id": alert_id,
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self._alert_history.append(alert_data)

        log_severity = "CRITICAL" if severity == "EMERGENCY" else severity
        logger.log(log_severity, f"Alert [{alert_type}]: {message}")
        
        await asyncio.gather(
            self._send_webhook(alert_data),
            self._send_telegram(message, severity)
        )
    
    def _should_filter(self, severity: str) -> bool:
        levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4, "EMERGENCY": 5}
        alert_level_num = levels.get(self._alert_level, 1)
        # 兜底：未知/更高级别（如 EMERGENCY）默认按 CRITICAL 处理，宁可误报不可漏报
        severity_num = levels.get(severity, 4)
        
        return severity_num < alert_level_num
    
    def _is_rate_limited(self, alert_type: str, window: int = None) -> bool:
        w = window if window is not None else self._rate_limit_window
        now = datetime.now().timestamp()
        timestamps = self._rate_limit.setdefault(alert_type, deque(maxlen=100))
        # 清理超过窗口的旧时间戳
        while timestamps and now - timestamps[0] >= w:
            timestamps.popleft()
        if timestamps:
            return True
        timestamps.append(now)
        return False
    
    async def _send_webhook(self, alert_data: Dict[str, Any]):
        if "webhook" not in self._providers or not self._webhook_url:
            return
        
        try:
            response = await asyncio.to_thread(lambda: requests.post(
                self._webhook_url,
                json=alert_data,
                headers={"Content-Type": "application/json"},
                timeout=5
            ))

            if response.status_code == 200:
                logger.info(f"Webhook alert sent: {alert_data['alert_type']}")
            else:
                logger.error(f"Webhook alert failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")
    
    async def _send_telegram(self, message: str, severity: str):
        if "telegram" not in self._providers or not self._telegram_token or not self._telegram_chat_id:
            return
        
        try:
            severity_emoji = {
                "DEBUG": "🔵",
                "INFO": "ℹ️",
                "WARNING": "⚠️",
                "ERROR": "❌",
                "CRITICAL": "🚨",
                "EMERGENCY": "🚨"
            }
            
            formatted_message = f"{severity_emoji.get(severity, '')} {message}"
            
            url = f"https://api.telegram.org/bot{self._telegram_token}/sendMessage"
            response = await asyncio.to_thread(lambda: requests.post(
                url,
                data={
                    "chat_id": self._telegram_chat_id,
                    "text": formatted_message,
                    "parse_mode": "HTML"
                },
                timeout=5
            ))

            if response.status_code == 200:
                logger.info("Telegram alert sent successfully")
            else:
                logger.error(f"Telegram alert failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
    
    async def send_trade_signal_alert(self, signal_data: Dict[str, Any]):
        message = f"📊 **Trade Signal**\n" \
                  f"Direction: {signal_data['direction'].upper()}\n" \
                  f"Symbol: {signal_data['symbol']}\n" \
                  f"Price: ${signal_data['price']:.4f}\n" \
                  f"Quantity: {signal_data['quantity']}\n" \
                  f"Leverage: {signal_data['leverage']}x\n" \
                  f"Strategy: {signal_data['strategy_name']}"
        
        await self.send_alert(
            alert_type="TRADE_SIGNAL",
            message=message,
            severity="INFO",
            symbol=signal_data["symbol"],
            metadata=signal_data
        )
    
    async def send_risk_alert(self, risk_type: str, message: str, symbol: str = "", 
                              metadata: Dict[str, Any] = None):
        await self.send_alert(
            alert_type=f"RISK_{risk_type.upper()}",
            message=message,
            severity="WARNING",
            symbol=symbol,
            metadata=metadata
        )
    
    async def send_system_alert(self, system_type: str, message: str, 
                                metadata: Dict[str, Any] = None):
        # 系统告警（延迟/CPU/内存等持续性指标）类型级 5 分钟去重，
        # 避免 message 中含动态数值绕过内容去重导致每分钟刷屏
        sys_key = f"SYSTEM_{system_type.upper()}"
        if self._is_rate_limited(sys_key, window=300):
            return
        await self.send_alert(
            alert_type=sys_key,
            message=message,
            severity="ERROR",
            metadata=metadata
        )
    
    async def send_critical_alert(self, message: str, symbol: str = "", 
                                  metadata: Dict[str, Any] = None):
        await self.send_alert(
            alert_type="CRITICAL",
            message=message,
            severity="CRITICAL",
            symbol=symbol,
            metadata=metadata
        )
    
    async def send_daily_summary(self, summary_data: Dict[str, Any]):
        message = f"📈 **Daily Trading Summary**\n" \
                  f"Total Trades: {summary_data.get('total_trades', 0)}\n" \
                  f"Win Rate: {summary_data.get('win_rate', 0):.1%}\n" \
                  f"Total PnL: ${summary_data.get('total_pnl', 0):.2f}\n" \
                  f"Max Win: ${summary_data.get('max_win', 0):.2f}\n" \
                  f"Max Loss: ${summary_data.get('max_loss', 0):.2f}"
        
        await self.send_alert(
            alert_type="DAILY_SUMMARY",
            message=message,
            severity="INFO",
            metadata=summary_data
        )
    
    def get_alert_history(self, limit: int = 50) -> list:
        return sorted(
            self._alert_history,
            key=lambda x: x["timestamp"],
            reverse=True
        )[:limit]