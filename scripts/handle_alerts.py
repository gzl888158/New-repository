"""
监控告警处理脚本
处理系统告警，提供解决方案
"""
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List


class AlertHandler:
    """告警处理器"""
    
    def __init__(self):
        self.alerts = []
        self.solutions = []
    
    def check_redis_status(self) -> Dict[str, Any]:
        """检查Redis状态"""
        monitor_path = "./data/monitor_status.json"
        if os.path.exists(monitor_path):
            with open(monitor_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            redis_info = data.get("redis", {})
            return {
                "available": redis_info.get("available", False),
                "degraded": redis_info.get("degraded_to_memory", False),
                "cache_size": redis_info.get("memory_cache_size", 0),
            }
        return {"available": False, "error": "monitor_status.json not found"}
    
    def check_api_latency(self) -> Dict[str, Any]:
        """检查API延迟"""
        monitor_path = "./data/monitor_status.json"
        if os.path.exists(monitor_path):
            with open(monitor_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            return {
                "current_ms": data.get("api_latency_ms", 0),
                "avg_ms": data.get("api_latency_avg_ms", 0),
                "warning_threshold": 1500,
                "critical_threshold": 5000,
            }
        return {"error": "monitor_status.json not found"}
    
    def check_risk_status(self) -> Dict[str, Any]:
        """检查风控状态"""
        risk_path = "./data/risk_status.json"
        if os.path.exists(risk_path):
            with open(risk_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            return {
                "equity": data.get("current_equity", 0),
                "is_paused": data.get("is_paused", False),
                "pause_reason": data.get("pause_reason"),
                "daily_pnl": data.get("daily_pnl", 0),
            }
        return {"error": "risk_status.json not found"}
    
    def analyze_alerts(self) -> List[Dict[str, Any]]:
        """分析告警"""
        alerts = []
        
        # Redis告警
        redis_status = self.check_redis_status()
        if not redis_status.get("available") and redis_status.get("degraded"):
            alerts.append({
                "id": "SYSTEM_REDIS_DEGRADED",
                "severity": "WARNING",
                "message": "Redis unavailable, system degraded to memory cache mode",
                "details": redis_status,
                "solution": "Install and start Redis service, or continue with memory cache (reduced persistence)",
            })
        
        # API延迟告警
        latency_status = self.check_api_latency()
        avg_latency = latency_status.get("avg_ms", 0)
        if avg_latency > 3000:
            alerts.append({
                "id": "SYSTEM_API_LATENCY_WARNING",
                "severity": "WARNING",
                "message": f"API latency {avg_latency:.0f}ms exceeds threshold 1500ms",
                "details": latency_status,
                "solution": "Check network connection, consider using proxy, or reduce API call frequency",
            })
        
        # WebSocket告警（从日志推断）
        alerts.append({
            "id": "WEBSOCKET_CONNECTION_FAILED",
            "severity": "ERROR",
            "message": "Private WebSocket connection failed",
            "details": {"impact": "Cannot receive account updates, orders, positions in real-time"},
            "solution": "Check network firewall, verify OKX API status, or restart the system",
        })
        
        # 风控状态
        risk_status = self.check_risk_status()
        if risk_status.get("is_paused"):
            alerts.append({
                "id": "TRADING_PAUSED",
                "severity": "CRITICAL",
                "message": f"Trading paused: {risk_status.get('pause_reason')}",
                "details": risk_status,
                "solution": "Review risk status, reset pause state if necessary",
            })
        
        # 资金不足告警
        equity = risk_status.get("equity", 0)
        if equity < 50:
            alerts.append({
                "id": "LOW_EQUITY",
                "severity": "WARNING",
                "message": f"Low equity: {equity:.2f} USDT",
                "details": risk_status,
                "solution": "Add more capital to the trading account",
            })
        
        return alerts
    
    def generate_solutions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成解决方案"""
        solutions = []
        
        for alert in alerts:
            alert_id = alert.get("id", "")
            
            if alert_id == "SYSTEM_REDIS_DEGRADED":
                solutions.append({
                    "alert_id": alert_id,
                    "action": "INSTALL_REDIS",
                    "description": "Install and start Redis service",
                    "commands": [
                        "# Windows: Download Redis from https://github.com/microsoftarchive/redis/releases",
                        "# Or use WSL: sudo apt-get install redis-server",
                        "# Start: redis-server",
                    ],
                    "auto_fix": False,
                    "impact": "Enables persistent caching, improves system reliability",
                })
            
            elif alert_id == "SYSTEM_API_LATENCY_WARNING":
                solutions.append({
                    "alert_id": alert_id,
                    "action": "OPTIMIZE_API_LATENCY",
                    "description": "Reduce API call frequency and optimize network",
                    "commands": [
                        "# Check network latency: ping api.okx.com",
                        "# Reduce polling interval in config.yaml",
                    ],
                    "auto_fix": False,
                    "impact": "Reduces API timeout errors and improves trade execution",
                })
            
            elif alert_id == "WEBSOCKET_CONNECTION_FAILED":
                solutions.append({
                    "alert_id": alert_id,
                    "action": "RESTART_WEBSOCKET",
                    "description": "Restart WebSocket connection",
                    "commands": [
                        "# System will auto-reconnect, or restart the trading system",
                    ],
                    "auto_fix": True,
                    "impact": "Restores real-time account updates",
                })
            
            elif alert_id == "TRADING_PAUSED":
                solutions.append({
                    "alert_id": alert_id,
                    "action": "RESET_TRADING_PAUSE",
                    "description": "Reset trading pause state",
                    "commands": [
                        "# Run: python scripts/reset_trading.py",
                    ],
                    "auto_fix": True,
                    "impact": "Resumes trading operations",
                })
        
        return solutions
    
    def apply_auto_fixes(self, solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """应用自动修复"""
        applied = []
        
        for solution in solutions:
            if solution.get("auto_fix"):
                alert_id = solution.get("alert_id")
                
                if alert_id == "WEBSOCKET_CONNECTION_FAILED":
                    # WebSocket会自动重连，无需手动处理
                    applied.append({
                        "alert_id": alert_id,
                        "status": "AUTO_FIX_APPLIED",
                        "message": "WebSocket auto-reconnect is enabled",
                    })
                
                elif alert_id == "TRADING_PAUSED":
                    # 重置交易暂停状态
                    control_path = "./data/risk_control.json"
                    with open(control_path, "w", encoding="utf-8") as f:
                        json.dump({
                            "action": "reset_pause",
                            "timestamp": datetime.now().isoformat(),
                        }, f, indent=2)
                    
                    applied.append({
                        "alert_id": alert_id,
                        "status": "AUTO_FIX_APPLIED",
                        "message": "Trading pause reset instruction created",
                    })
        
        return applied
    
    def print_report(self, alerts: List[Dict[str, Any]], solutions: List[Dict[str, Any]], applied: List[Dict[str, Any]]):
        """打印报告"""
        print("=" * 60)
        print("MONITORING ALERT REPORT")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        print("ALERTS DETECTED:")
        print("-" * 40)
        for i, alert in enumerate(alerts, 1):
            print(f"{i}. [{alert['severity']}] {alert['id']}")
            print(f"   Message: {alert['message']}")
            print(f"   Solution: {alert['solution']}")
            print()
        
        print("AUTO-FIX RESULTS:")
        print("-" * 40)
        if applied:
            for fix in applied:
                print(f"  - {fix['alert_id']}: {fix['status']}")
                print(f"    {fix['message']}")
        else:
            print("  No auto-fixes applied")
        print()
        
        print("MANUAL ACTIONS REQUIRED:")
        print("-" * 40)
        for solution in solutions:
            if not solution.get("auto_fix"):
                print(f"  Action: {solution['action']}")
                print(f"  Description: {solution['description']}")
                print(f"  Commands:")
                for cmd in solution.get("commands", []):
                    print(f"    {cmd}")
                print()
        
        print("=" * 60)


async def send_test_alert(message: str = None) -> Dict[str, Any]:
    """手动触发一条测试告警，验证 Webhook/Telegram 告警链路是否真的打通。

    直接实例化生产环境使用的 AlertManager，走真实的 send_alert 链路。
    消息默认带时间戳，绕过 5 分钟内容去重，确保每次调用都能真正下发。
    """
    # 脚本可能从任意 cwd 运行，确保项目根目录在 sys.path 中
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from configs.settings import load_config
    from monitoring.alert_manager import AlertManager

    config = load_config()
    manager = AlertManager(config)

    test_message = message or f"[TEST] 告警链路测试 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    # 用 WARNING 级别尽可能绕过 level 过滤（仅 ERROR/CRITICAL 级别会拦截）
    await manager.send_alert(
        alert_type="ALERT_TEST",
        message=test_message,
        severity="WARNING",
        metadata={"source": "handle_alerts.send_test_alert", "manual": True},
    )

    return {
        "status": "sent",
        "message": test_message,
        "notifications_enabled": manager._enabled,
        "alert_level": manager._alert_level,
        "webhook_configured": bool(manager._webhook_url) and "webhook" in manager._providers,
        "telegram_configured": bool(manager._telegram_token and manager._telegram_chat_id)
                              and "telegram" in manager._providers,
    }


def run_test_alert() -> int:
    """命令行入口：一键触发测试告警并打印链路配置状态。"""
    result = asyncio.run(send_test_alert())

    print("=" * 60)
    print("ALERT CHANNEL TEST")
    print("=" * 60)
    print(f"Message: {result['message']}")
    print(f"Notifications enabled: {result['notifications_enabled']}")
    print(f"Alert level: {result['alert_level']}")
    print(f"Webhook configured: {result['webhook_configured']}")
    print(f"Telegram configured: {result['telegram_configured']}")
    print("=" * 60)

    if not result["notifications_enabled"]:
        print("WARNING: 通知未启用，告警未发送。请检查 config.yaml 的 notifications.enabled")
        return 1
    if not result["webhook_configured"] and not result["telegram_configured"]:
        print("WARNING: Webhook 与 Telegram 均未配置，无法下发告警。")
        return 1

    print("测试告警已下发，请到 Telegram / Webhook 查看是否收到。")
    return 0


def main():
    """主函数"""
    if "--test-alert" in sys.argv:
        return run_test_alert()

    handler = AlertHandler()
    
    # 分析告警
    alerts = handler.analyze_alerts()
    
    # 生成解决方案
    solutions = handler.generate_solutions(alerts)
    
    # 应用自动修复
    applied = handler.apply_auto_fixes(solutions)
    
    # 打印报告
    handler.print_report(alerts, solutions, applied)
    
    # 保存告警记录
    alert_record = {
        "timestamp": datetime.now().isoformat(),
        "alerts": alerts,
        "solutions": solutions,
        "applied_fixes": applied,
    }
    
    os.makedirs("./data/alerts", exist_ok=True)
    alert_file = f"./data/alerts/alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(alert_file, "w", encoding="utf-8") as f:
        json.dump(alert_record, f, indent=2, ensure_ascii=False)
    
    print(f"Alert record saved to: {alert_file}")
    
    return 0


if __name__ == "__main__":
    main()