"""
通知渠道管理器
支持多种通知渠道：邮件、钉钉、飞书等
"""
import asyncio
import json
import smtplib
import ssl
import time
from abc import ABC, abstractmethod
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Optional, Any, List

import requests
from loguru import logger


class NotificationChannel(ABC):
    """通知渠道抽象基类（内置熔断器）"""

    CIRCUIT_FAILURE_THRESHOLD = 5
    CIRCUIT_COOLDOWN_SECONDS = 60.0

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.name = self.__class__.__name__
        self._consecutive_failures = 0
        self._circuit_open = False
        self._circuit_opened_at = 0.0

    def _circuit_allows_send(self) -> bool:
        if not self._circuit_open:
            return True
        if time.monotonic() - self._circuit_opened_at > self.CIRCUIT_COOLDOWN_SECONDS:
            logger.info(f"{self.name} circuit breaker half-open, allowing trial send")
            return True
        return False

    def _record_success(self) -> None:
        if self._consecutive_failures > 0:
            logger.info(f"{self.name} circuit breaker closed (was {self._consecutive_failures} failures)")
        self._consecutive_failures = 0
        self._circuit_open = False

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.CIRCUIT_FAILURE_THRESHOLD:
            self._circuit_open = True
            self._circuit_opened_at = time.monotonic()
            logger.warning(
                f"{self.name} circuit breaker OPEN after {self._consecutive_failures} "
                f"consecutive failures (cooldown {self.CIRCUIT_COOLDOWN_SECONDS}s)"
            )

    @abstractmethod
    async def send(self, message: str, subject: str, severity: str = "INFO", **kwargs) -> bool:
        """发送通知"""
        pass

    def format_message(self, message: str, subject: str, severity: str) -> str:
        """格式化消息"""
        return f"[{severity}] {subject}\n\n{message}"


class EmailChannel(NotificationChannel):
    """邮件通知渠道"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.smtp_server = config.get("smtp_server", "smtp.gmail.com")
        self.smtp_port = config.get("smtp_port", 587)
        self.username = config.get("username")
        self.password = config.get("password")
        self.recipients = config.get("recipients", [])
    
    async def send(self, message: str, subject: str, severity: str = "INFO", **kwargs) -> bool:
        if not self.enabled:
            return True
        
        if not self.username or not self.password or not self.recipients:
            logger.warning("Email channel not properly configured")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = ", ".join(self.recipients)
            msg['Subject'] = f"[{severity}] {subject}"
            
            body = self.format_message(message, subject, severity)
            msg.attach(MIMEText(body, 'plain'))
            
            context = ssl.create_default_context()
            
            # P1-5: 将阻塞式 SMTP 操作移至线程池
            def _send_email():
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.starttls(context=context)
                    server.login(self.username, self.password)
                    server.sendmail(self.username, self.recipients, msg.as_string())
            
            await asyncio.to_thread(_send_email)
            
            logger.info(f"Email sent to {self.recipients}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False


class DingTalkChannel(NotificationChannel):
    """钉钉通知渠道"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.webhook_url = config.get("webhook_url")
        self.secret = config.get("secret")
    
    async def send(self, message: str, subject: str, severity: str = "INFO", **kwargs) -> bool:
        if not self.enabled:
            return True
        
        if not self.webhook_url:
            logger.warning("DingTalk webhook URL not configured")
            return False
        
        try:
            headers = {"Content-Type": "application/json"}
            
            severity_color = {
                "CRITICAL": "#FF0000",
                "WARNING": "#FFA500",
                "INFO": "#1890FF",
                "DEBUG": "#8c8c8c"
            }
            
            body = {
                "msgtype": "markdown",
                "markdown": {
                    "title": subject,
                    "text": f"**【{severity}】{subject}**\n\n{message}"
                },
                "at": {
                    "isAtAll": severity == "CRITICAL"
                }
            }
            
            # P1-5: 将阻塞式 requests.post 移至线程池
            response = await asyncio.to_thread(
                requests.post, self.webhook_url, headers=headers, json=body, timeout=10
            )
            response.raise_for_status()
            
            result = response.json()
            if result.get("errcode") == 0:
                logger.info("DingTalk message sent successfully")
                return True
            else:
                logger.error(f"DingTalk API error: {result.get('errmsg', 'unknown')}")
                return False
        except Exception as e:
            logger.error(f"Failed to send DingTalk message: {e}")
            return False


class FeishuChannel(NotificationChannel):
    """飞书通知渠道"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.webhook_url = config.get("webhook_url")
    
    async def send(self, message: str, subject: str, severity: str = "INFO", **kwargs) -> bool:
        if not self.enabled:
            return True
        
        if not self.webhook_url:
            logger.warning("Feishu webhook URL not configured")
            return False
        
        try:
            headers = {"Content-Type": "application/json"}
            
            body = {
                "msg_type": "text",
                "content": {
                    "text": f"[{severity}] {subject}\n\n{message}"
                }
            }
            
            # P1-5: 将阻塞式 requests.post 移至线程池
            response = await asyncio.to_thread(
                requests.post, self.webhook_url, headers=headers, json=body, timeout=10
            )
            response.raise_for_status()
            
            result = response.json()
            if result.get("code") == 0:
                logger.info("Feishu message sent successfully")
                return True
            else:
                logger.error(f"Feishu API error: {result.get('msg', 'unknown')}")
                return False
        except Exception as e:
            logger.error(f"Failed to send Feishu message: {e}")
            return False


class ConsoleChannel(NotificationChannel):
    """控制台通知渠道"""
    
    async def send(self, message: str, subject: str, severity: str = "INFO", **kwargs) -> bool:
        if not self.enabled:
            return True
        
        full_message = self.format_message(message, subject, severity)
        
        severity_colors = {
            "CRITICAL": "\033[91m",
            "WARNING": "\033[93m",
            "INFO": "\033[94m",
            "DEBUG": "\033[90m"
        }
        
        color = severity_colors.get(severity, "\033[0m")
        reset = "\033[0m"
        
        print(f"{color}{full_message}{reset}")
        logger.info(f"Console notification: {subject}")
        return True


class NotificationManager:
    """通知管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._channels: Dict[str, NotificationChannel] = {}
        self._initialize_channels()
    
    def _initialize_channels(self):
        channel_configs = self.config.get("channels", {})
        
        if "email" in channel_configs:
            self._channels["email"] = EmailChannel(channel_configs["email"])
        
        if "dingtalk" in channel_configs:
            self._channels["dingtalk"] = DingTalkChannel(channel_configs["dingtalk"])
        
        if "feishu" in channel_configs:
            self._channels["feishu"] = FeishuChannel(channel_configs["feishu"])
        
        if "console" in channel_configs:
            self._channels["console"] = ConsoleChannel(channel_configs["console"])
        
        if not self._channels:
            self._channels["console"] = ConsoleChannel({"enabled": True})
        
        logger.info(f"Initialized {len(self._channels)} notification channels")
    
    def get_channel(self, name: str) -> Optional[NotificationChannel]:
        """获取指定渠道"""
        return self._channels.get(name)
    
    def list_channels(self) -> List[str]:
        """列出所有渠道"""
        return list(self._channels.keys())
    
    async def send_to_channel(self, channel_name: str, message: str, subject: str,
                              severity: str = "INFO", **kwargs) -> bool:
        """发送到指定渠道（带熔断保护）"""
        channel = self.get_channel(channel_name)
        if not channel:
            logger.warning(f"Channel {channel_name} not found")
            return False

        if not channel._circuit_allows_send():
            logger.debug(f"{channel_name} circuit breaker open, dropping message")
            return False

        result = await channel.send(message, subject, severity, **kwargs)
        if result:
            channel._record_success()
        else:
            channel._record_failure()
        return result
    
    async def broadcast(self, message: str, subject: str, severity: str = "INFO", 
                       channels: Optional[List[str]] = None, **kwargs) -> Dict[str, bool]:
        """广播到多个渠道"""
        results = {}
        
        target_channels = channels if channels else self._channels.keys()
        
        for channel_name in target_channels:
            result = await self.send_to_channel(channel_name, message, subject, severity, **kwargs)
            results[channel_name] = result
        
        return results
    
    async def send_alert(self, alert_data: Dict[str, Any]) -> Dict[str, bool]:
        """发送告警通知"""
        severity = alert_data.get("severity", "INFO")
        subject = alert_data.get("message", "Alert")
        message = json.dumps(alert_data, indent=2, ensure_ascii=False)
        
        return await self.broadcast(message, subject, severity)
    
    def update_channel_config(self, channel_name: str, config: Dict[str, Any]):
        """更新渠道配置"""
        if channel_name == "email":
            self._channels[channel_name] = EmailChannel(config)
        elif channel_name == "dingtalk":
            self._channels[channel_name] = DingTalkChannel(config)
        elif channel_name == "feishu":
            self._channels[channel_name] = FeishuChannel(config)
        elif channel_name == "console":
            self._channels[channel_name] = ConsoleChannel(config)
        
        logger.info(f"Updated config for channel {channel_name}")
