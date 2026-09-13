"""
多实例沙盒管理器
================
管理多个策略沙盒实例的生命周期：创建、启动、停止、监控、对比分析

核心定位：
- 统一管理多个独立沙盒实例，支持并行策略测试
- 沙盒配置模板化，快速创建新沙盒
- 支持沙盒克隆（A/B测试：同一策略不同参数对比）
- 沙盒实例间独立运行，互不干扰
- 沙盒性能排名和交叉对比分析
- 沙盒状态持久化（暂停后恢复）

使用场景：
  1. 并行测试：同时测试趋势、网格、抢单策略各自表现
  2. A/B对比：同一策略不同参数的两组沙盒对比
  3. 币种筛选：多币种并行测试，筛选最佳交易对
  4. 策略迁移：沙盒验证通过后迁移到实盘
"""

import asyncio
import threading
import json
import os
import shutil
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from loguru import logger

from .sandbox_engine import (
    SandboxEngine, SandboxConfig, SandboxState, SandboxTrade
)
from .virtual_account import VirtualAccount, create_virtual_account, \
    get_virtual_account, remove_virtual_account, list_virtual_accounts


class SandboxRunMode(Enum):
    """沙盒运行模式"""
    MANUAL = "manual"         # 手动触发（步进模式）
    CONTINUOUS = "continuous"  # 持续运行（自动行情驱动）
    SCHEDULED = "scheduled"    # 定时运行


class SandboxRankMetric(Enum):
    """沙盒排名指标"""
    ROI = "roi_pct"
    SHARPE = "sharpe_ratio"
    WIN_RATE = "win_rate_pct"
    PROFIT_FACTOR = "profit_factor"
    MAX_DRAWDOWN = "max_drawdown_pct"
    CALMAR = "calmar_ratio"
    TOTAL_PNL = "total_pnl"


@dataclass
class SandboxTemplate:
    """沙盒配置模板"""
    template_id: str
    name: str
    description: str
    symbols: List[str] = field(default_factory=list)
    strategies: List[str] = field(default_factory=list)
    initial_capital: float = 10000.0
    max_leverage: int = 10
    max_positions: int = 4
    risk_per_trade_pct: float = 0.02
    max_daily_loss_pct: float = 0.10
    max_drawdown_pct: float = 0.25
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.0
    min_signal_strength: float = 0.4
    run_mode: SandboxRunMode = SandboxRunMode.CONTINUOUS
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "symbols": self.symbols,
            "strategies": self.strategies,
            "initial_capital": self.initial_capital,
            "max_leverage": self.max_leverage,
            "max_positions": self.max_positions,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "min_signal_strength": self.min_signal_strength,
            "run_mode": self.run_mode.value,
            "tags": self.tags,
        }


@dataclass
class SandboxInstance:
    """沙盒实例元数据"""
    sandbox_id: str
    name: str
    config: SandboxConfig
    engine: SandboxEngine
    run_mode: SandboxRunMode = SandboxRunMode.CONTINUOUS
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    template_id: str = ""
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    cloned_from: Optional[str] = None

    def to_summary(self) -> Dict[str, Any]:
        perf = self.engine.get_performance_summary()
        return {
            "sandbox_id": self.sandbox_id,
            "name": self.name,
            "state": self.engine.state.value,
            "run_mode": self.run_mode.value,
            "template_id": self.template_id,
            "tags": self.tags,
            "notes": self.notes,
            "cloned_from": self.cloned_from,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "config": self.config.to_dict(),
            "performance": {
                "roi_pct": perf.get("roi_pct", 0),
                "win_rate_pct": perf.get("win_rate_pct", 0),
                "profit_factor": perf.get("profit_factor", 0),
                "total_trades": perf.get("total_trades_count", 0),
                "max_drawdown_pct": perf.get("max_drawdown_pct", 0),
                "sharpe_ratio": perf.get("sharpe_ratio", 0),
                "calmar_ratio": perf.get("calmar_ratio", 0),
                "avg_hold_minutes": perf.get("avg_hold_minutes", 0),
            },
            "risk": {
                "is_frozen": perf.get("is_frozen", False),
                "freeze_reason": perf.get("freeze_reason", ""),
                "margin_ratio": perf.get("margin_ratio", 0),
                "daily_pnl": perf.get("daily_pnl", 0),
            },
        }


@dataclass
class SandboxComparison:
    """沙盒对比结果"""
    metric: str
    rankings: List[Dict[str, Any]] = field(default_factory=list)
    best_sandbox_id: str = ""
    worst_sandbox_id: str = ""
    avg_value: float = 0.0
    std_value: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "best": self.best_sandbox_id,
            "worst": self.worst_sandbox_id,
            "avg": round(self.avg_value, 4),
            "std": round(self.std_value, 4),
            "rankings": self.rankings,
        }


# ── 预设模板 ────────────────────────────────────────────────────

PRESET_TEMPLATES = {
    "aggressive_trend": SandboxTemplate(
        template_id="aggressive_trend",
        name="激进趋势策略",
        description="高杠杆趋势跟随，适合大波动行情",
        symbols=["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"],
        strategies=["trend"],
        initial_capital=10000.0,
        max_leverage=15,
        max_positions=3,
        risk_per_trade_pct=0.03,
        max_daily_loss_pct=0.12,
        max_drawdown_pct=0.30,
        min_signal_strength=0.45,
        tags=["aggressive", "trend", "high_volatility"],
    ),
    "balanced_grid": SandboxTemplate(
        template_id="balanced_grid",
        name="均衡网格策略",
        description="中等杠杆网格交易，适合震荡行情",
        symbols=["ADA-USDT-SWAP", "AVAX-USDT-SWAP", "ARB-USDT-SWAP"],
        strategies=["grid"],
        initial_capital=5000.0,
        max_leverage=8,
        max_positions=5,
        risk_per_trade_pct=0.015,
        max_daily_loss_pct=0.08,
        max_drawdown_pct=0.20,
        min_signal_strength=0.35,
        tags=["balanced", "grid", "ranging_market"],
    ),
    "fast_scalping": SandboxTemplate(
        template_id="fast_scalping",
        name="快速抢单策略",
        description="高频抢单，快速止盈止损",
        symbols=["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP",
                  "XRP-USDT-SWAP", "BNB-USDT-SWAP"],
        strategies=["scalping"],
        initial_capital=3000.0,
        max_leverage=5,
        max_positions=4,
        risk_per_trade_pct=0.01,
        max_daily_loss_pct=0.06,
        max_drawdown_pct=0.15,
        min_signal_strength=0.45,
        tags=["scalping", "high_frequency", "small_profit"],
    ),
    "comprehensive": SandboxTemplate(
        template_id="comprehensive",
        name="综合策略组合",
        description="多策略并行测试",
        symbols=["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP",
                  "XRP-USDT-SWAP", "ADA-USDT-SWAP", "AVAX-USDT-SWAP"],
        strategies=["trend", "grid", "scalping"],
        initial_capital=10000.0,
        max_leverage=10,
        max_positions=6,
        risk_per_trade_pct=0.02,
        max_daily_loss_pct=0.10,
        max_drawdown_pct=0.25,
        min_signal_strength=0.40,
        tags=["comprehensive", "multi_strategy", "all_weather"],
    ),
    "conservative_arbitrage": SandboxTemplate(
        template_id="conservative_arbitrage",
        name="保守套利策略",
        description="低杠杆套利，严格风控",
        symbols=["BTC-USDT-SWAP", "ETH-USDT-SWAP"],
        strategies=["arbitrage"],
        initial_capital=20000.0,
        max_leverage=5,
        max_positions=2,
        risk_per_trade_pct=0.01,
        max_daily_loss_pct=0.04,
        max_drawdown_pct=0.10,
        min_signal_strength=0.55,
        tags=["conservative", "arbitrage", "low_risk"],
    ),
}


class SandboxManager:
    """
    多实例沙盒管理器

    统一管理所有沙盒实例，提供：
    - 实例生命周期管理（创建、启动、暂停、停止、删除）
    - 配置模板系统（快速创建/克隆沙盒）
    - 多方盒并行运行和对比分析
    - 状态持久化和恢复
    - 性能排名
    """

    def __init__(self, system_config: Dict[str, Any] = None):
        self.system_config = system_config or {}
        self._instances: Dict[str, SandboxInstance] = {}
        self._templates: Dict[str, SandboxTemplate] = dict(PRESET_TEMPLATES)
        self._lock = threading.RLock()
        self._instance_counter = 0
        self._persist_dir = self.system_config.get(
            "sandbox_persist_dir", "./data/sandboxes"
        )
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # 回调
        self._instance_callback: Optional[Callable] = None

        # 确保持久化目录存在
        os.makedirs(self._persist_dir, exist_ok=True)

        # 尝试恢复已保存的沙盒
        self._restore_saved_instances()

        logger.info(f"SandboxManager initialized: {len(self._templates)} templates, "
                    f"{len(self._instances)} instances")

    # ── 事件循环 ─────────────────────────────────────────────────

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """设置事件循环"""
        self._event_loop = loop

    def _run_async(self, coro):
        """在事件循环中运行协程"""
        if self._event_loop and self._event_loop.is_running():
            import asyncio as asyncio_mod
            future = asyncio_mod.run_coroutine_threadsafe(coro, self._event_loop)
            return future.result(timeout=30)
        else:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import asyncio as asyncio_mod
                    future = asyncio_mod.run_coroutine_threadsafe(coro, loop)
                    return future.result(timeout=30)
            except RuntimeError:
                pass
            return asyncio.run(coro)

    # ── 实例创建 ─────────────────────────────────────────────────

    def _generate_sandbox_id(self, prefix: str = "sb") -> str:
        """生成唯一沙盒ID"""
        with self._lock:
            self._instance_counter += 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{prefix}_{ts}_{self._instance_counter:04d}"

    def create_from_template(self, template_id: str, name: str = None,
                             overrides: Dict[str, Any] = None,
                             notes: str = "") -> Optional[str]:
        """
        基于模板创建沙盒

        Args:
            template_id: 模板ID
            name: 自定义名称（可选）
            overrides: 配置覆盖项（可选）
            notes: 备注说明

        Returns:
            新沙盒的 sandbox_id，失败返回 None
        """
        template = self._templates.get(template_id)
        if not template:
            logger.error(f"Template '{template_id}' not found")
            return None

        overrides = overrides or {}
        sandbox_id = self._generate_sandbox_id()

        # 合并配置
        config = SandboxConfig(
            sandbox_id=sandbox_id,
            name=name or f"{template.name} ({sandbox_id})",
            initial_capital=overrides.get("initial_capital", template.initial_capital),
            symbols=overrides.get("symbols", template.symbols),
            strategies=overrides.get("strategies", template.strategies),
            timeframes=overrides.get("timeframes", ["5m", "15m"]),
            max_leverage=overrides.get("max_leverage", template.max_leverage),
            max_positions=overrides.get("max_positions", template.max_positions),
            risk_per_trade_pct=overrides.get("risk_per_trade_pct", template.risk_per_trade_pct),
            max_daily_loss_pct=overrides.get("max_daily_loss_pct", template.max_daily_loss_pct),
            max_drawdown_pct=overrides.get("max_drawdown_pct", template.max_drawdown_pct),
            stop_loss_atr_mult=overrides.get("stop_loss_atr_mult", template.stop_loss_atr_mult),
            take_profit_atr_mult=overrides.get("take_profit_atr_mult", template.take_profit_atr_mult),
            min_signal_strength=overrides.get("min_signal_strength", template.min_signal_strength),
            description=overrides.get("description", template.description),
        )

        try:
            engine = SandboxEngine(config, self.system_config)
            engine.initialize()

            instance = SandboxInstance(
                sandbox_id=sandbox_id,
                name=config.name,
                config=config,
                engine=engine,
                run_mode=template.run_mode,
                template_id=template_id,
                tags=list(template.tags),
                notes=notes,
            )

            with self._lock:
                self._instances[sandbox_id] = instance

            logger.info(f"Sandbox '{sandbox_id}' created from template '{template_id}'")
            self._save_instance_metadata(sandbox_id)

            if self._instance_callback:
                try:
                    self._instance_callback("created", sandbox_id)
                except Exception:
                    pass

            return sandbox_id

        except Exception as e:
            logger.error(f"Failed to create sandbox from template '{template_id}': {e}")
            return None

    def create_custom(self, config: Dict[str, Any]) -> Optional[str]:
        """自定义配置创建沙盒"""
        sandbox_id = self._generate_sandbox_id()

        sandbox_config = SandboxConfig(
            sandbox_id=sandbox_id,
            name=config.get("name", f"Sandbox {sandbox_id}"),
            initial_capital=config.get("initial_capital", 10000.0),
            symbols=config.get("symbols", []),
            strategies=config.get("strategies", []),
            timeframes=config.get("timeframes", ["5m", "15m"]),
            max_leverage=config.get("max_leverage", 10),
            max_positions=config.get("max_positions", 4),
            risk_per_trade_pct=config.get("risk_per_trade_pct", 0.02),
            max_daily_loss_pct=config.get("max_daily_loss_pct", 0.10),
            max_drawdown_pct=config.get("max_drawdown_pct", 0.25),
            min_signal_strength=config.get("min_signal_strength", 0.4),
            description=config.get("description", ""),
        )

        try:
            engine = SandboxEngine(sandbox_config, self.system_config)
            engine.initialize()

            instance = SandboxInstance(
                sandbox_id=sandbox_id,
                name=sandbox_config.name,
                config=sandbox_config,
                engine=engine,
                run_mode=SandboxRunMode(config.get("run_mode", "continuous")),
                notes=config.get("notes", ""),
            )

            with self._lock:
                self._instances[sandbox_id] = instance

            logger.info(f"Custom sandbox '{sandbox_id}' created")
            self._save_instance_metadata(sandbox_id)
            return sandbox_id

        except Exception as e:
            logger.error(f"Failed to create custom sandbox: {e}")
            return None

    def clone(self, source_id: str, overrides: Dict[str, Any] = None,
              name: str = None) -> Optional[str]:
        """
        克隆沙盒（用于A/B测试）

        创建一个与源沙盒配置相同的新沙盒，
        可以用 overrides 调整部分参数进行对比测试。

        Args:
            source_id: 源沙盒ID
            overrides: 需要修改的配置项
            name: 新沙盒名称
        """
        source = self._instances.get(source_id)
        if not source:
            logger.error(f"Source sandbox '{source_id}' not found")
            return None

        new_id = self._generate_sandbox_id("sb_clone")
        overrides = overrides or {}

        cloned_config = SandboxConfig(
            sandbox_id=new_id,
            name=name or f"{source.config.name} (clone)",
            initial_capital=overrides.get("initial_capital", source.config.initial_capital),
            symbols=overrides.get("symbols", list(source.config.symbols)),
            strategies=overrides.get("strategies", list(source.config.strategies)),
            timeframes=overrides.get("timeframes", list(source.config.timeframes)),
            max_leverage=overrides.get("max_leverage", source.config.max_leverage),
            max_positions=overrides.get("max_positions", source.config.max_positions),
            risk_per_trade_pct=overrides.get("risk_per_trade_pct",
                                              source.config.risk_per_trade_pct),
            max_daily_loss_pct=overrides.get("max_daily_loss_pct",
                                              source.config.max_daily_loss_pct),
            max_drawdown_pct=overrides.get("max_drawdown_pct",
                                            source.config.max_drawdown_pct),
            min_signal_strength=overrides.get("min_signal_strength",
                                               source.config.min_signal_strength),
            description=overrides.get("description",
                                       f"Cloned from {source_id}"),
        )

        try:
            engine = SandboxEngine(cloned_config, self.system_config)
            engine.initialize()

            instance = SandboxInstance(
                sandbox_id=new_id,
                name=cloned_config.name,
                config=cloned_config,
                engine=engine,
                run_mode=source.run_mode,
                template_id=source.template_id,
                tags=list(source.tags),
                notes=overrides.get("notes", f"Cloned from {source_id}"),
                cloned_from=source_id,
            )

            with self._lock:
                self._instances[new_id] = instance

            logger.info(f"Sandbox '{new_id}' cloned from '{source_id}'")
            self._save_instance_metadata(new_id)
            return new_id

        except Exception as e:
            logger.error(f"Failed to clone sandbox '{source_id}': {e}")
            return None

    # ── 生命周期管理 ─────────────────────────────────────────────

    def start(self, sandbox_id: str) -> bool:
        """启动沙盒"""
        instance = self._instances.get(sandbox_id)
        if not instance:
            logger.error(f"Sandbox '{sandbox_id}' not found")
            return False

        try:
            self._run_async(instance.engine.start())
            instance.started_at = datetime.now()
            instance.stopped_at = None
            logger.info(f"Sandbox '{sandbox_id}' started")
            return True
        except Exception as e:
            logger.error(f"Failed to start sandbox '{sandbox_id}': {e}")
            return False

    def stop(self, sandbox_id: str) -> bool:
        """停止沙盒"""
        instance = self._instances.get(sandbox_id)
        if not instance:
            logger.error(f"Sandbox '{sandbox_id}' not found")
            return False

        try:
            self._run_async(instance.engine.stop())
            instance.stopped_at = datetime.now()
            logger.info(f"Sandbox '{sandbox_id}' stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop sandbox '{sandbox_id}': {e}")
            return False

    def pause(self, sandbox_id: str) -> bool:
        """暂停沙盒"""
        instance = self._instances.get(sandbox_id)
        if not instance:
            logger.error(f"Sandbox '{sandbox_id}' not found")
            return False

        instance.engine.pause()
        return True

    def resume(self, sandbox_id: str) -> bool:
        """恢复沙盒"""
        instance = self._instances.get(sandbox_id)
        if not instance:
            logger.error(f"Sandbox '{sandbox_id}' not found")
            return False

        instance.engine.resume()
        return True

    def delete(self, sandbox_id: str) -> bool:
        """删除沙盒（含清理虚拟账户）"""
        instance = self._instances.get(sandbox_id)
        if not instance:
            logger.error(f"Sandbox '{sandbox_id}' not found")
            return False

        try:
            # 先停止
            if instance.engine.state == SandboxState.RUNNING:
                self.stop(sandbox_id)

            # 清理虚拟账户
            remove_virtual_account(sandbox_id)

            with self._lock:
                del self._instances[sandbox_id]

            # 清理持久化文件
            self._clean_persisted_files(sandbox_id)

            logger.info(f"Sandbox '{sandbox_id}' deleted")
            return True
        except Exception as e:
            logger.error(f"Failed to delete sandbox '{sandbox_id}': {e}")
            return False

    def delete_all(self) -> int:
        """删除所有沙盒"""
        ids = list(self._instances.keys())
        count = 0
        for sid in ids:
            if self.delete(sid):
                count += 1
        return count

    def start_all(self) -> int:
        """启动所有非运行沙盒"""
        count = 0
        for sid in list(self._instances.keys()):
            instance = self._instances[sid]
            if instance.engine.state not in (SandboxState.RUNNING, SandboxState.FROZEN):
                if self.start(sid):
                    count += 1
        return count

    def stop_all(self) -> int:
        """停止所有运行中沙盒"""
        count = 0
        for sid in list(self._instances.keys()):
            if self._instances[sid].engine.state == SandboxState.RUNNING:
                if self.stop(sid):
                    count += 1
        return count

    # ── 行情广播 ─────────────────────────────────────────────────

    def broadcast_bar(self, symbol: str, bar: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        向所有相关沙盒广播K线数据

        Returns:
            {sandbox_id: [signals]}
        """
        results = {}
        for sandbox_id, instance in self._instances.items():
            if instance.engine.state != SandboxState.RUNNING:
                continue
            if symbol not in instance.config.symbols:
                continue
            if instance.run_mode != SandboxRunMode.CONTINUOUS:
                continue

            try:
                signals = self._run_async(instance.engine.on_bar(symbol, bar))
                if signals:
                    results[sandbox_id] = signals
            except Exception as e:
                logger.error(f"Broadcast bar to '{sandbox_id}' error: {e}")

        return results

    def broadcast_tick(self, symbol: str, tick: Dict[str, Any]) -> None:
        """向所有相关沙盒广播tick数据"""
        for sandbox_id, instance in self._instances.items():
            if instance.engine.state != SandboxState.RUNNING:
                continue
            if symbol not in instance.config.symbols:
                continue

            try:
                self._run_async(instance.engine.on_tick(symbol, tick))
            except Exception as e:
                logger.error(f"Broadcast tick to '{sandbox_id}' error: {e}")

    def broadcast_prices(self, prices: Dict[str, float]) -> None:
        """向所有沙盒广播价格更新"""
        for sandbox_id, instance in self._instances.items():
            if instance.engine.state != SandboxState.RUNNING:
                continue
            filtered = {s: p for s, p in prices.items()
                       if s in instance.config.symbols}
            if filtered:
                instance.engine.update_prices(filtered)

    # ── 步进模式 ─────────────────────────────────────────────────

    def step_bar(self, sandbox_id: str, symbol: str,
                 bar: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        手动步进一个K线（手动模式用）
        适用于精细回测和参数调优
        """
        instance = self._instances.get(sandbox_id)
        if not instance:
            logger.error(f"Sandbox '{sandbox_id}' not found")
            return []

        if instance.run_mode != SandboxRunMode.MANUAL:
            return []

        return self._run_async(instance.engine.on_bar(symbol, bar))

    # ── 查询 ─────────────────────────────────────────────────────

    def get_instance(self, sandbox_id: str) -> Optional[SandboxInstance]:
        """获取沙盒实例"""
        return self._instances.get(sandbox_id)

    def get_engine(self, sandbox_id: str) -> Optional[SandboxEngine]:
        """获取沙盒引擎"""
        instance = self._instances.get(sandbox_id)
        return instance.engine if instance else None

    def list_instances(self, state: str = None,
                       tags: List[str] = None) -> List[Dict[str, Any]]:
        """列出沙盒实例（支持状态和标签过滤）"""
        result = []
        for instance in self._instances.values():
            if state and instance.engine.state.value != state:
                continue
            if tags:
                if not any(t in instance.tags for t in tags):
                    continue
            result.append(instance.to_summary())
        return result

    def list_active(self) -> List[Dict[str, Any]]:
        """列出活跃沙盒"""
        return self.list_instances(state="running")

    def get_instance_count(self) -> Dict[str, int]:
        """统计各状态沙盒数量"""
        counts = {"total": 0, "running": 0, "paused": 0, "frozen": 0,
                   "stopped": 0, "created": 0, "error": 0}
        for instance in self._instances.values():
            counts["total"] += 1
            state = instance.engine.state.value
            counts[state] = counts.get(state, 0) + 1
        return counts

    # ── 对比分析 ─────────────────────────────────────────────────

    def compare(self, sandbox_ids: List[str] = None,
                metric: str = "roi_pct") -> SandboxComparison:
        """
        沙盒对比分析

        Args:
            sandbox_ids: 要对比的沙盒ID列表（None=所有）
            metric: 排名指标
        """
        instances = {}
        if sandbox_ids:
            for sid in sandbox_ids:
                if sid in self._instances:
                    instances[sid] = self._instances[sid]
        else:
            instances = dict(self._instances)

        if not instances:
            return SandboxComparison(metric=metric)

        # 收集指标值
        values = []
        for sid, inst in instances.items():
            perf = inst.engine.get_performance_summary()
            val = perf.get(metric, 0)
            values.append((sid, val))

        # 按值排序（对于回撤类指标，值越小越好）
        reverse = metric not in ("max_drawdown_pct", "max_drawdown")
        values.sort(key=lambda x: x[1], reverse=reverse)

        # 构建排名
        rankings = []
        for rank, (sid, val) in enumerate(values, 1):
            inst = instances[sid]
            rankings.append({
                "rank": rank,
                "sandbox_id": sid,
                "name": inst.name,
                "value": round(val, 4),
                "state": inst.engine.state.value,
            })

        # 统计
        numeric = [v[1] for v in values]
        import math
        avg_val = sum(numeric) / len(numeric)
        variance = sum((x - avg_val) ** 2 for x in numeric) / len(numeric)
        std_val = math.sqrt(variance)

        return SandboxComparison(
            metric=metric,
            rankings=rankings,
            best_sandbox_id=values[0][0] if values else "",
            worst_sandbox_id=values[-1][0] if len(values) > 1 else "",
            avg_value=avg_val,
            std_value=std_val,
        )

    def compare_multi(self, sandbox_ids: List[str] = None) -> Dict[str, SandboxComparison]:
        """多维对比（ROI、胜率、盈亏比、回撤）"""
        metrics = ["roi_pct", "win_rate_pct", "profit_factor",
                    "max_drawdown_pct", "sharpe_ratio"]
        return {m: self.compare(sandbox_ids, m) for m in metrics}

    def rank_all(self, metric: str = "roi_pct") -> List[Dict[str, Any]]:
        """全局排名"""
        comparison = self.compare(metric=metric)
        return comparison.rankings

    def find_best(self, metric: str = "roi_pct",
                  min_trades: int = 5) -> Optional[str]:
        """
        找最优沙盒

        Args:
            metric: 排名指标
            min_trades: 最少交易笔数
        """
        comparison = self.compare(metric=metric)
        for entry in comparison.rankings:
            sid = entry["sandbox_id"]
            instance = self._instances.get(sid)
            if not instance:
                continue
            perf = instance.engine.get_performance_summary()
            if perf.get("total_trades_count", 0) >= min_trades:
                return sid
        return None

    # ── 聚合统计 ─────────────────────────────────────────────────

    def aggregate_stats(self) -> Dict[str, Any]:
        """所有沙盒聚合统计"""
        all_roi = []
        all_win_rates = []
        all_profit_factors = []
        all_drawdowns = []
        total_trades = 0
        total_pnl = 0.0

        for instance in self._instances.values():
            perf = instance.engine.get_performance_summary()
            all_roi.append(perf.get("roi_pct", 0))
            all_win_rates.append(perf.get("win_rate_pct", 0))
            all_profit_factors.append(perf.get("profit_factor", 0))
            all_drawdowns.append(perf.get("max_drawdown_pct", 0))
            total_trades += perf.get("total_trades_count", 0)
            total_pnl += perf.get("realized_pnl_total", 0)

        n = len(all_roi) or 1

        return {
            "total_instances": len(self._instances),
            "active_instances": len(self.list_active()),
            "total_trades": total_trades,
            "total_pnl": round(total_pnl, 4),
            "avg_roi_pct": round(sum(all_roi) / n, 2),
            "best_roi_pct": round(max(all_roi) if all_roi else 0, 2),
            "worst_roi_pct": round(min(all_roi) if all_roi else 0, 2),
            "avg_win_rate_pct": round(sum(all_win_rates) / n, 2),
            "avg_profit_factor": round(sum(all_profit_factors) / n, 4),
            "avg_max_drawdown_pct": round(sum(all_drawdowns) / n, 2),
            "instance_counts": self.get_instance_count(),
        }

    # ── 模板管理 ─────────────────────────────────────────────────

    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """获取模板"""
        template = self._templates.get(template_id)
        return template.to_dict() if template else None

    def list_templates(self, tags: List[str] = None) -> List[Dict[str, Any]]:
        """列出所有模板"""
        result = []
        for t in self._templates.values():
            if tags and not any(tag in t.tags for tag in tags):
                continue
            result.append(t.to_dict())
        return result

    def register_template(self, template: SandboxTemplate) -> bool:
        """注册自定义模板"""
        with self._lock:
            self._templates[template.template_id] = template
        logger.info(f"Template '{template.template_id}' registered")
        return True

    # ── 持久化 ───────────────────────────────────────────────────

    def _get_instance_dir(self, sandbox_id: str) -> str:
        """获取沙盒实例目录"""
        return os.path.join(self._persist_dir, sandbox_id)

    def _save_instance_metadata(self, sandbox_id: str) -> None:
        """保存沙盒元数据到磁盘"""
        instance = self._instances.get(sandbox_id)
        if not instance:
            return

        data = {
            "sandbox_id": sandbox_id,
            "name": instance.name,
            "template_id": instance.template_id,
            "config": instance.config.to_dict(),
            "run_mode": instance.run_mode.value,
            "tags": instance.tags,
            "notes": instance.notes,
            "cloned_from": instance.cloned_from,
            "created_at": instance.created_at.isoformat(),
            "started_at": instance.started_at.isoformat() if instance.started_at else None,
            "stopped_at": instance.stopped_at.isoformat() if instance.stopped_at else None,
        }

        instance_dir = self._get_instance_dir(sandbox_id)
        os.makedirs(instance_dir, exist_ok=True)

        meta_file = os.path.join(instance_dir, "metadata.json")
        try:
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata for '{sandbox_id}': {e}")

    def _clean_persisted_files(self, sandbox_id: str) -> None:
        """清理持久化文件"""
        instance_dir = self._get_instance_dir(sandbox_id)
        if os.path.exists(instance_dir):
            try:
                shutil.rmtree(instance_dir)
            except Exception as e:
                logger.error(f"Failed to clean files for '{sandbox_id}': {e}")

    def _restore_saved_instances(self) -> None:
        """恢复已保存的沙盒"""
        if not os.path.exists(self._persist_dir):
            return

        for dir_name in os.listdir(self._persist_dir):
            instance_dir = os.path.join(self._persist_dir, dir_name)
            if not os.path.isdir(instance_dir):
                continue

            meta_file = os.path.join(instance_dir, "metadata.json")
            if not os.path.exists(meta_file):
                continue

            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read metadata for '{dir_name}': {e}")
                continue

            sandbox_id = data.get("sandbox_id", dir_name)
            logger.info(f"Restored sandbox metadata: {sandbox_id} ({data.get('name', '')})")
            # 注：恢复时仅记录元数据，引擎需要重新初始化
            # 实际的 Engine 恢复在 initialize_restored 中进行

    def restore_instance(self, sandbox_id: str) -> bool:
        """恢复单个已保存的沙盒（重新创建引擎）"""
        instance_dir = self._get_instance_dir(sandbox_id)
        meta_file = os.path.join(instance_dir, "metadata.json")

        if not os.path.exists(meta_file):
            logger.error(f"Metadata not found for '{sandbox_id}'")
            return False

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            config = SandboxConfig(
                sandbox_id=sandbox_id,
                name=data.get("name", sandbox_id),
                initial_capital=data.get("config", {}).get("initial_capital", 10000.0),
                symbols=data.get("config", {}).get("symbols", []),
                strategies=data.get("config", {}).get("strategies", []),
                max_leverage=data.get("config", {}).get("max_leverage", 10),
                max_positions=data.get("config", {}).get("max_positions", 4),
                risk_per_trade_pct=data.get("config", {}).get("risk_per_trade_pct", 0.02),
            )

            engine = SandboxEngine(config, self.system_config)
            engine.initialize()

            instance = SandboxInstance(
                sandbox_id=sandbox_id,
                name=config.name,
                config=config,
                engine=engine,
                run_mode=SandboxRunMode(data.get("run_mode", "continuous")),
                template_id=data.get("template_id", ""),
                tags=data.get("tags", []),
                notes=data.get("notes", ""),
                cloned_from=data.get("cloned_from"),
            )

            with self._lock:
                self._instances[sandbox_id] = instance

            logger.info(f"Sandbox '{sandbox_id}' restored")
            return True

        except Exception as e:
            logger.error(f"Failed to restore sandbox '{sandbox_id}': {e}")
            return False

    def persist_all(self) -> int:
        """持久化所有沙盒元数据"""
        count = 0
        for sandbox_id in self._instances:
            self._save_instance_metadata(sandbox_id)
            count += 1
        return count

    # ── 回调 ─────────────────────────────────────────────────────

    def register_instance_callback(self, callback: Callable) -> None:
        """注册实例变更回调"""
        self._instance_callback = callback


# ── 全局单例 ────────────────────────────────────────────────────

_manager: Optional[SandboxManager] = None
_manager_lock = threading.Lock()


def get_sandbox_manager(system_config: Dict[str, Any] = None) -> SandboxManager:
    """获取全局沙盒管理器"""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = SandboxManager(system_config)
        elif system_config:
            _manager.system_config.update(system_config)
        return _manager


def reset_sandbox_manager() -> None:
    """重置管理器（测试用）"""
    global _manager
    with _manager_lock:
        _manager = None
