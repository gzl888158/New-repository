"""企业级实盘交易模块。

提供 `LiveTradingRunner`，封装实盘生命周期：
- fail-closed 预检（API 凭证、Kill Switch、单实例、风控参数、API 连通性）
- 优雅启动/关闭
- JSON 安全运行报告

典型用法：
    from live import LiveTradingRunner
    runner = LiveTradingRunner(config)
    report = await runner.run()
"""
from live.runner import LiveTradingRunner, PreflightCheck
from live._base import safe_float, safe_int, safe_div, safe_finite, safe_json_dumps

__all__ = [
    "LiveTradingRunner",
    "PreflightCheck",
    "safe_float",
    "safe_int",
    "safe_div",
    "safe_finite",
    "safe_json_dumps",
]
