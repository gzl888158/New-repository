"""告警管理器：负责告警的生成、去重、分级与通知分发。"""
import asyncio
import os
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
        # 按严重级别配置速率限制窗口（秒）和每窗口最大条数
        notif = config.get("notifications", {})
        alert_limits = notif.get("alert_rate_limits", {})
        self._rate_limit_windows = {
            "DEBUG": alert_limits.get("DEBUG", 120),
            "INFO": alert_limits.get("INFO", 60),
            "WARNING": alert_limits.get("WARNING", 30),
            "ERROR": alert_limits.get("ERROR", 15),
            "CRITICAL": alert_limits.get("CRITICAL", 5),
            "EMERGENCY": alert_limits.get("EMERGENCY", 0),
        }
        self._rate_limit_max_per_window = alert_limits.get("max_per_window", 3)
        self._retry_max = notif.get("alert_retry_max", 2)
        self._retry_delay = notif.get("alert_retry_delay_seconds", 1.0)
        self._fallback_dir = notif.get("alert_fallback_dir", "data/alerts_failed")
    
    async def send_alert(self, alert_type: str, message: str, severity: str = "INFO", 
                         symbol: str = "", metadata: Dict[str, Any] = None):
        if not self._enabled:
            return
        
        if self._should_filter(severity):
            return
        
        # 去重 key = alert_type + symbol，确保不同品种的同类告警互不阻塞
        dedup_key = f"{alert_type}:{symbol}" if symbol else alert_type
        if self._is_rate_limited(dedup_key, severity=severity):
            logger.debug(f"Alert rate limited: {dedup_key}")
            return

        # 相同内容去重：5分钟内相同 key+message 不重复发送（用于持续性告警如 LIQUIDATION）
        content_key = f"{dedup_key}:{message[:80]}"
        if self._is_rate_limited(content_key, severity=severity, window=300):
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

        results = await asyncio.gather(
            self._send_with_retry("webhook", alert_data),
            self._send_with_retry("telegram", alert_data),
            return_exceptions=True,
        )
        # 至少有一个通道已配置但全部失败 → 写入本地文件降级
        has_configured = (
            ("webhook" in self._providers and bool(self._webhook_url))
            or ("telegram" in self._providers and bool(self._telegram_token) and bool(self._telegram_chat_id))
        )
        if has_configured and all(r is not True for r in results):
            self._fallback_to_file(alert_data)
    
    def _should_filter(self, severity: str) -> bool:
        levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4, "EMERGENCY": 5}
        alert_level_num = levels.get(self._alert_level, 1)
        # 兜底：未知/更高级别（如 EMERGENCY）默认按 CRITICAL 处理，宁可误报不可漏报
        severity_num = levels.get(severity, 4)
        
        return severity_num < alert_level_num
    
    def _is_rate_limited(self, alert_type: str, severity: str = "INFO",
                         window: int = None, max_count: int = None) -> bool:
        # EMERGENCY 告警永不限速
        if severity == "EMERGENCY":
            now = datetime.now().timestamp()
            self._rate_limit.setdefault(alert_type, deque(maxlen=100)).append(now)
            return False

        if window is not None:
            w = window
        else:
            w = self._rate_limit_windows.get(severity, 60)

        if w <= 0:
            return False

        max_per = max_count if max_count is not None else self._rate_limit_max_per_window
        now = datetime.now().timestamp()
        timestamps = self._rate_limit.setdefault(alert_type, deque(maxlen=100))
        while timestamps and now - timestamps[0] >= w:
            timestamps.popleft()
        if len(timestamps) >= max_per:
            return True
        timestamps.append(now)
        return False
    
    async def _send_with_retry(self, channel: str, alert_data: Dict[str, Any]) -> bool:
        """带重试的告警投递；返回 True 表示成功。"""
        for attempt in range(1 + self._retry_max):
            ok = False
            if channel == "webhook":
                ok = await self._send_webhook(alert_data)
            elif channel == "telegram":
                ok = await self._send_telegram(alert_data)
            else:
                return False
            if ok:
                return True
            if attempt < self._retry_max:
                await asyncio.sleep(self._retry_delay * (attempt + 1))
        logger.warning(f"Alert delivery exhausted retries: channel={channel} type={alert_data.get('alert_type')}")
        return False

    def _fallback_to_file(self, alert_data: Dict[str, Any]) -> None:
        """所有通道失败 → 写入本地 JSONL 文件，后续可手动补发或审计。"""
        try:
            os.makedirs(self._fallback_dir, exist_ok=True)
            date_str = datetime.now().strftime("%Y-%m-%d")
            path = os.path.join(self._fallback_dir, f"alerts_{date_str}.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert_data, ensure_ascii=False, default=str) + "\n")
            logger.warning(f"Alert fallback written: {path}")
        except Exception as e:
            logger.error(f"Alert fallback write failed: {e}")

    async def _send_webhook(self, alert_data: Dict[str, Any]) -> bool:
        if "webhook" not in self._providers or not self._webhook_url:
            return False

        try:
            response = await asyncio.to_thread(lambda: requests.post(
                self._webhook_url,
                json=alert_data,
                headers={"Content-Type": "application/json"},
                timeout=5
            ))

            if response.status_code == 200:
                logger.info(f"Webhook alert sent: {alert_data['alert_type']}")
                return True
            else:
                logger.error(f"Webhook alert failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")
            return False

    async def _send_telegram(self, alert_data: Dict[str, Any]) -> bool:
        if "telegram" not in self._providers or not self._telegram_token or not self._telegram_chat_id:
            return False

        try:
            severity_emoji = {
                "DEBUG": "🔵",
                "INFO": "ℹ️",
                "WARNING": "⚠️",
                "ERROR": "❌",
                "CRITICAL": "🚨",
                "EMERGENCY": "🚨"
            }

            message = alert_data.get("message", "")
            severity = alert_data.get("severity", "INFO")
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
                return True
            else:
                logger.error(f"Telegram alert failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
            return False
    
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
        if self._is_rate_limited(sys_key, severity="ERROR", window=300):
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