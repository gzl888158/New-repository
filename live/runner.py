"""企业级实盘交易运行器：封装实盘生命周期，提供 fail-closed 预检与优雅启停。

核心设计：
- fail-closed：任何预检失败（凭证缺失、API 不通、Kill Switch 已启用、单实例冲突、
  风控参数越界）一律阻止实盘启动，绝不带病上线。
- JSON 安全：所有运行报告不含 NaN/Inf，可直接落盘或上报。
- 不污染调用方：对传入 config 做深拷贝后再操作。
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import signal
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from live._base import (
    safe_float, safe_int, safe_div, safe_finite,
    _sanitize_for_json, safe_json_dumps,
)


# Kill Switch 持久化状态文件（与 core/kill_switch.py 保持一致）
_KILL_SWITCH_STATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "kill_switch_state.json",
)

# 实盘启动预检中，风控参数的合理上限（超限视为配置错误，fail-closed）
_MAX_REASONABLE_CAPITAL = 10_000_000.0  # 1000 万 USDT
_MAX_REASONABLE_DRAWDOWN = 1.0          # 100%
_MAX_REASONABLE_DAILY_LOSS = 1.0        # 100%
_MIN_TRADING_CAPITAL_RATIO = 0.0
_MAX_TRADING_CAPITAL_RATIO = 1.0


class PreflightCheck:
    """单条预检结果。"""
    def __init__(self, name: str, passed: bool, message: str = "", detail: Any = None):
        self.name = name
        self.passed = bool(passed)
        self.message = str(message)
        self.detail = detail

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "detail": _sanitize_for_json(self.detail),
        }


class LiveTradingRunner:
    """实盘交易生命周期管理器。

    使用方式：
        runner = LiveTradingRunner(config)
        report = await runner.run()   # 阻塞直到收到退出信号
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # 深拷贝，不污染调用方 dict
        self._config: Dict[str, Any] = copy.deepcopy(config) if isinstance(config, dict) else {}
        self._scheduler = None
        self._shutdown_event: Optional[asyncio.Event] = None
        self._start_time: Optional[datetime] = None

    # ── 配置安全访问 ────────────────────────────────────────

    def _cfg(self, *keys: str, default: Any = None) -> Any:
        """链式安全读取 config，缺失返回 default。"""
        cur: Any = self._config
        for k in keys:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k)
            if cur is None:
                return default
        return cur

    # ── 预检：API 凭证 ──────────────────────────────────────

    def _check_api_credentials(self) -> PreflightCheck:
        """校验环境变量中的 API 凭证，缺失/占位符一律 fail-closed。"""
        api_key = os.getenv("OKX_API_KEY", "") or ""
        secret_key = os.getenv("OKX_SECRET_KEY", "") or ""
        passphrase = os.getenv("OKX_PASSPHRASE", "") or ""

        placeholders = {
            "your_api_key", "your_secret_key", "your_passphrase",
            "your_real_api_key_here", "your_real_secret_key_here",
            "your_real_passphrase_here", "",
        }
        for val, label in [(api_key, "OKX_API_KEY"), (secret_key, "OKX_SECRET_KEY"), (passphrase, "OKX_PASSPHRASE")]:
            v = val.strip()
            if not v:
                return PreflightCheck("api_credentials", False, f"{label} 未配置")
            if v in placeholders:
                return PreflightCheck("api_credentials", False, f"{label} 仍为占位符，请填写真实凭证")

        # 不记录完整凭证，仅记录长度用于审计
        return PreflightCheck(
            "api_credentials", True,
            f"凭证已配置 (key_len={len(api_key)}, secret_len={len(secret_key)})",
            {"key_len": len(api_key), "secret_len": len(secret_key)},
        )

    # ── 预检：Kill Switch 状态 ──────────────────────────────

    def _check_kill_switch(self) -> PreflightCheck:
        """读取磁盘 Kill Switch 状态，若已启用则 fail-closed 阻止启动。"""
        try:
            if not os.path.exists(_KILL_SWITCH_STATE_PATH):
                return PreflightCheck("kill_switch", True, "Kill Switch 未启用（状态文件不存在）")
            with open(_KILL_SWITCH_STATE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            enabled = bool(data.get("enabled", False))
            reason = str(data.get("reason", "") or "")
            enabled_by = str(data.get("enabled_by", "") or "")
            if enabled:
                return PreflightCheck(
                    "kill_switch", False,
                    f"Kill Switch 已启用，禁止启动实盘（reason={reason!r}, by={enabled_by!r}）",
                    {"enabled": True, "reason": reason, "enabled_by": enabled_by},
                )
            return PreflightCheck(
                "kill_switch", True, "Kill Switch 未启用",
                {"enabled": False},
            )
        except Exception as e:
            # 读取失败时 fail-closed：宁可阻止启动，也不在状态未知时上线
            return PreflightCheck(
                "kill_switch", False,
                f"读取 Kill Switch 状态失败，保守阻止启动: {e}",
            )

    # ── 预检：单实例锁 ──────────────────────────────────────

    def _check_single_instance(self) -> PreflightCheck:
        """通过 SingleInstance 确保同一时刻只有一个实盘进程。"""
        try:
            from utils.single_instance import SingleInstance
            si = SingleInstance("okx_live_trading")
            acquired = si.acquire()
            # 注意：这里只做检测，不持有锁（真正持锁由 run() 内的 context 管理）
            if acquired:
                si.release()
                return PreflightCheck("single_instance", True, "无其他实盘进程运行")
            return PreflightCheck(
                "single_instance", False,
                "检测到另一个实盘进程正在运行，禁止重复启动",
            )
        except Exception as e:
            return PreflightCheck(
                "single_instance", False,
                f"单实例检测异常: {e}",
            )

    # ── 预检：风控参数合理性 ────────────────────────────────

    def _check_risk_config(self) -> PreflightCheck:
        """校验关键风控参数在合理范围内，越界 fail-closed。"""
        total_capital = safe_float(self._cfg("trading", "total_capital"), 0.0)
        trading_ratio = safe_float(self._cfg("trading", "trading_capital_ratio"), 0.0)
        max_drawdown = safe_float(self._cfg("trading", "max_drawdown"), 0.0)
        daily_max_loss = safe_float(self._cfg("trading", "daily_max_loss"), 0.0)

        issues: List[str] = []
        if total_capital <= 0:
            issues.append(f"total_capital={total_capital} 必须 > 0")
        elif total_capital > _MAX_REASONABLE_CAPITAL:
            issues.append(f"total_capital={total_capital} 超过合理上限 {_MAX_REASONABLE_CAPITAL}")
        if not (_MIN_TRADING_CAPITAL_RATIO <= trading_ratio <= _MAX_TRADING_CAPITAL_RATIO):
            issues.append(f"trading_capital_ratio={trading_ratio} 必须在 [0,1]")
        if not (0 < max_drawdown <= _MAX_REASONABLE_DRAWDOWN):
            issues.append(f"max_drawdown={max_drawdown} 必须在 (0,{_MAX_REASONABLE_DRAWDOWN}]")
        if not (0 < daily_max_loss <= _MAX_REASONABLE_DAILY_LOSS):
            issues.append(f"daily_max_loss={daily_max_loss} 必须在 (0,{_MAX_REASONABLE_DAILY_LOSS}]")

        if issues:
            return PreflightCheck(
                "risk_config", False,
                "风控参数越界: " + "; ".join(issues),
                {"total_capital": total_capital, "trading_capital_ratio": trading_ratio,
                 "max_drawdown": max_drawdown, "daily_max_loss": daily_max_loss},
            )

        trading_capital = safe_finite(total_capital * trading_ratio, 0.0)
        return PreflightCheck(
            "risk_config", True,
            f"风控参数合理 (capital={total_capital}, trading={trading_capital:.2f}, "
            f"dd={max_drawdown:.2%}, daily_loss={daily_max_loss:.2%})",
            {"total_capital": total_capital, "trading_capital": trading_capital,
             "max_drawdown": max_drawdown, "daily_max_loss": daily_max_loss},
        )

    # ── 预检：API 连通性 ────────────────────────────────────

    async def _check_api_connection(self) -> PreflightCheck:
        """实际调用 OKX API 验证连通性，失败 fail-closed。"""
        try:
            from core.okx_client import OKXClient
            client = OKXClient(self._config)
            account_info = None
            if hasattr(client, "get_account_info"):
                account_info = client.get_account_info()
            else:
                account_info = client._make_request("GET", "/api/v5/account/balance")

            if not account_info:
                return PreflightCheck("api_connection", False, "API 返回空响应")

            total_eq = "N/A"
            avail_bal = "N/A"
            used_margin = "N/A"
            if isinstance(account_info, dict):
                if account_info.get("totalEq"):
                    total_eq = str(account_info.get("totalEq"))
                    avail_bal = str(account_info.get("availBal", "N/A"))
                    used_margin = str(account_info.get("usedMargin", "N/A"))
                elif account_info.get("data") and len(account_info["data"]) > 0:
                    data = account_info["data"][0]
                    total_eq = str(data.get("totalEq", "N/A"))
                    avail_bal = str(data.get("availBal", "N/A"))
                    used_margin = str(data.get("usedMargin", "N/A"))

            return PreflightCheck(
                "api_connection", True,
                f"API 连通 (equity={total_eq}, avail={avail_bal}, margin={used_margin})",
                {"total_eq": total_eq, "avail_bal": avail_bal, "used_margin": used_margin},
            )
        except Exception as e:
            return PreflightCheck(
                "api_connection", False,
                f"API 连通性检查失败: {e}",
            )

    # ── 汇总预检 ────────────────────────────────────────────

    async def run_preflight(self) -> Dict[str, Any]:
        """执行全部预检，返回 JSON 安全的报告。任一失败 passed=False。"""
        checks: List[PreflightCheck] = [
            self._check_api_credentials(),
            self._check_kill_switch(),
            self._check_single_instance(),
            self._check_risk_config(),
            await self._check_api_connection(),
        ]
        all_passed = all(c.passed for c in checks)
        failed = [c for c in checks if not c.passed]
        report = {
            "all_passed": all_passed,
            "timestamp": datetime.now().isoformat(),
            "checks": [c.to_dict() for c in checks],
            "failed_count": len(failed),
            "failed_names": [c.name for c in failed],
        }
        return _sanitize_for_json(report)

    # ── 主生命周期 ──────────────────────────────────────────

    async def run(self) -> Dict[str, Any]:
        """执行实盘交易完整生命周期，返回 JSON 安全的运行报告。

        fail-closed：预检未通过时直接返回报告，不启动调度器。
        """
        self._start_time = datetime.now()
        self._shutdown_event = asyncio.Event()

        # 安装信号处理器（仅主线程）
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, self._shutdown_event.set)
                except (NotImplementedError, RuntimeError):
                    signal.signal(sig, lambda *_: self._shutdown_event.set())
        except RuntimeError:
            pass

        logger.warning("=" * 70)
        logger.warning("  LIVE TRADING MODE - REAL FUNDS WILL BE USED")
        logger.warning("=" * 70)

        # 1. 预检
        logger.info("Running pre-flight checks...")
        preflight = await self.run_preflight()
        logger.info(f"Pre-flight report: {safe_json_dumps(preflight)}")

        if not preflight.get("all_passed"):
            msg = (
                f"Pre-flight checks FAILED ({preflight.get('failed_count')} failed): "
                f"{preflight.get('failed_names')}. LIVE TRADING BLOCKED."
            )
            logger.critical(msg)
            return self._build_report("preflight_failed", preflight=preflight, error=msg)

        logger.info("[OK] All pre-flight checks passed, starting live trading...")

        # 2. 打印关键风控参数
        self._log_risk_summary()

        # 3. 获取单实例锁（持锁运行）
        from utils.single_instance import SingleInstance
        instance_lock = SingleInstance("okx_live_trading")
        if not instance_lock.acquire():
            msg = "Failed to acquire single-instance lock during startup."
            logger.critical(msg)
            return self._build_report("instance_lock_failed", preflight=preflight, error=msg)

        try:
            # 4. 启动调度器
            from core.scheduler import TradingScheduler
            self._scheduler = TradingScheduler(self._config)
            logger.info("Starting TradingScheduler...")
            await self._scheduler.start()
            logger.info("TradingScheduler started. Live trading is now ACTIVE.")

            # 5. 等待退出信号
            await self._shutdown_event.wait()
            logger.info("Shutdown signal received, initiating graceful shutdown...")

            return self._build_report("shutdown_normal", preflight=preflight)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Live trading runtime error: {e}\n{tb}")
            return self._build_report("runtime_error", preflight=preflight, error=str(e), traceback=tb)
        finally:
            # 6. 优雅关闭调度器
            if self._scheduler is not None:
                try:
                    await self._scheduler.shutdown()
                    logger.info("TradingScheduler shutdown complete.")
                except Exception as e:
                    logger.error(f"Error during scheduler shutdown: {e}")
            instance_lock.release()
            logger.info("Single-instance lock released.")

    # ── 辅助方法 ────────────────────────────────────────────

    def _log_risk_summary(self) -> None:
        total_capital = safe_float(self._cfg("trading", "total_capital"), 0.0)
        trading_ratio = safe_float(self._cfg("trading", "trading_capital_ratio"), 0.0)
        max_dd = safe_float(self._cfg("trading", "max_drawdown"), 0.0)
        daily_loss = safe_float(self._cfg("trading", "daily_max_loss"), 0.0)
        trading_capital = safe_finite(total_capital * trading_ratio, 0.0)
        is_testnet = bool(self._cfg("okx", "is_testnet", False))
        rest_url = str(self._cfg("okx", "rest_url", "N/A"))

        logger.info("=" * 70)
        logger.info("LIVE TRADING RISK SUMMARY")
        logger.info(f"  Mode: {'Testnet' if is_testnet else 'LIVE (REAL FUNDS)'}")
        logger.info(f"  API:  {rest_url}")
        logger.info(f"  Total Capital:      {total_capital:.2f} USDT")
        logger.info(f"  Trading Capital:    {trading_capital:.2f} USDT")
        logger.info(f"  Max Drawdown:       {max_dd:.2%}")
        logger.info(f"  Daily Max Loss:     {daily_loss:.2%}")
        logger.info("=" * 70)

    def _build_report(
        self,
        status: str,
        preflight: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        traceback: Optional[str] = None,
    ) -> Dict[str, Any]:
        """构建 JSON 安全的运行报告。"""
        end_time = datetime.now()
        duration = 0.0
        if self._start_time is not None:
            duration = safe_finite((end_time - self._start_time).total_seconds(), 0.0)
        report = {
            "status": status,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "preflight": preflight,
            "error": error,
            "traceback": traceback,
        }
        return _sanitize_for_json(report)


__all__ = ["LiveTradingRunner", "PreflightCheck"]
