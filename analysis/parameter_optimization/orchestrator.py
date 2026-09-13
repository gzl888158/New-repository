"""
参数优化编排器 (Parameter Optimization Orchestrator)

统一的参数优化编排层，整合：
  - 遗传算法优化 (GeneticOptimizer)
  - 贝叶斯优化 (BayesianOptimizer)
  - 前向行走分析 (WalkForwardAnalyzer)
  - 蒙特卡洛验证 (MonteCarloValidator)

提供统一的优化流程：
  1. GA 全局探索 → 2. BO 局部精调 → 3. WF 稳健性分析 → 4. MC 验证评估
"""
import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from functools import partial
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Callable
import numpy as np
from loguru import logger

from analysis.parameter_optimization.genetic_optimizer import (
    GeneticOptimizer, ParameterDef, GAOptimizationResult, SelectionMethod, CrossoverMethod,
)
from analysis.parameter_optimization.bayesian_optimizer import (
    BayesianOptimizer, BOOptimizationResult, AcquisitionFunction, KernelType,
)
from analysis.parameter_optimization.walk_forward_analyzer import (
    WalkForwardAnalyzer, WalkForwardResult, WindowMode,
)
from analysis.parameter_optimization.monte_carlo_validator import (
    MonteCarloValidator, MCValidationResult, ValidationMethod,
)


class OptimizationPhase(Enum):
    """优化阶段"""
    IDLE = "idle"
    GA_GLOBAL = "ga_global"
    BO_LOCAL = "bo_local"
    WF_ANALYSIS = "wf_analysis"
    MC_VALIDATION = "mc_validation"
    COMPLETE = "complete"
    FAILED = "failed"


class OptimizationStrategy(Enum):
    """优化策略"""
    FULL_PIPELINE = "full_pipeline"     # 完整流水线 GA→BO→WF→MC
    GA_ONLY = "ga_only"
    BO_ONLY = "bo_only"
    WF_ONLY = "wf_only"
    MC_ONLY = "mc_only"
    GA_BO_ONLY = "ga_bo_only"          # GA+BO 跳过验证
    LIGHTWEIGHT = "lightweight"        # 仅BO+MC


@dataclass
class OptimizationPipelineResult:
    """完整优化流水线结果"""
    strategy_name: str = ""
    strategy: str = ""
    phase: OptimizationPhase = OptimizationPhase.IDLE
    # 各阶段结果
    ga_result: Optional[GAOptimizationResult] = None
    bo_result: Optional[BOOptimizationResult] = None
    wf_result: Optional[WalkForwardResult] = None
    mc_result: Optional[MCValidationResult] = None
    # 最终最优参数
    best_params: Dict[str, float] = field(default_factory=dict)
    final_fitness: float = 0.0
    # 元信息
    total_time_seconds: float = 0.0
    total_evaluations: int = 0
    errors: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "phase": self.phase.value,
            "best_params": self.best_params,
            "final_fitness": round(self.final_fitness, 6),
            "ga_result": self.ga_result.to_dict() if self.ga_result else None,
            "bo_result": self.bo_result.to_dict() if self.bo_result else None,
            "wf_result": self.wf_result.to_dict() if self.wf_result else None,
            "mc_result": self.mc_result.to_dict() if self.mc_result else None,
            "total_time_seconds": round(self.total_time_seconds, 2),
            "total_evaluations": self.total_evaluations,
            "errors": self.errors,
            "timestamp": self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════
# 参数优化编排器
# ═══════════════════════════════════════════════════════════════

class ParameterOptimizationOrchestrator:
    """统一参数优化编排器"""

    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("parameter_optimization", {})
        self._config = config
        self._enabled = cfg.get("enabled", True)
        self._default_strategy = OptimizationStrategy(cfg.get("default_strategy", "full_pipeline"))
        self._persist_dir = cfg.get("persist_dir", "./data/parameter_optimization")
        self._max_concurrent = cfg.get("max_concurrent", 1)
        self._timeout_seconds = cfg.get("timeout_seconds", 3600)
        self._seed = cfg.get("seed", None)
        # 沉淤防护：内存与磁盘历史记录上限，防止参数优化结果无限累积拖垮系统
        self._max_history = int(cfg.get("max_history", 100))
        self._max_persist_files = int(cfg.get("max_persist_files", 50))

        # 子优化器
        self._ga: Optional[GeneticOptimizer] = None
        self._bo: Optional[BayesianOptimizer] = None
        self._wf: Optional[WalkForwardAnalyzer] = None
        self._mc: Optional[MonteCarloValidator] = None

        # 状态
        self._current_phase = OptimizationPhase.IDLE
        self._history: List[OptimizationPipelineResult] = []
        self._fitness_fn: Optional[Callable] = None
        self._price_data: Optional[np.ndarray] = None
        self._param_defs: List[ParameterDef] = []
        self._lock = asyncio.Lock()

        os.makedirs(self._persist_dir, exist_ok=True)
        logger.info(f"ParameterOptimizationOrchestrator initialized: strategy={self._default_strategy.value}")

    # ── 依赖注入 ──────────────────────────────────────────────

    def set_fitness_fn(self, fn: Callable[[Dict[str, float]], float]):
        """设置适应度函数"""
        self._fitness_fn = fn

    def _eval_fitness(self, params: Dict[str, float], data=None) -> float:
        """统一调用适应度函数，兼容 fn(params) 与 fn(params, data) 两种签名。"""
        if not self._fitness_fn:
            return 0.0
        try:
            val = self._fitness_fn(params, data)
        except TypeError:
            val = self._fitness_fn(params)
        try:
            return float(val) if val is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def set_price_data(self, data: np.ndarray):
        """设置价格数据（用于WF和MC验证）"""
        self._price_data = data

    def set_param_defs(self, param_defs: List[ParameterDef]):
        """设置搜索空间"""
        self._param_defs = param_defs

    def define_strategy_params(self, strategy_type: str) -> List[ParameterDef]:
        """为指定策略类型生成标准参数定义"""
        param_sets = {
            "trend": [
                ParameterDef("atr_period", "int", 10, 30, description="ATR周期"),
                ParameterDef("atr_multiplier", "float", 1.2, 3.5, step=0.1, description="ATR止损倍数"),
                ParameterDef("ema_fast", "int", 5, 30, description="快EMA周期"),
                ParameterDef("ema_slow", "int", 30, 120, description="慢EMA周期"),
                ParameterDef("rsi_period", "int", 7, 28, description="RSI周期"),
                ParameterDef("rsi_overbought", "int", 55, 80, description="RSI超买线"),
                ParameterDef("rsi_oversold", "int", 20, 45, description="RSI超卖线"),
                ParameterDef("min_signal_strength", "float", 0.15, 0.60, step=0.05, description="最小信号强度"),
            ],
            "scalping": [
                ParameterDef("atr_period", "int", 5, 20, description="ATR周期"),
                ParameterDef("atr_multiplier", "float", 0.5, 2.0, step=0.1, description="ATR倍数"),
                ParameterDef("rsi_period", "int", 3, 14, description="RSI周期"),
                ParameterDef("rsi_lower", "int", 15, 35, description="RSI下轨"),
                ParameterDef("rsi_upper", "int", 65, 85, description="RSI上轨"),
                ParameterDef("target_pct", "float", 0.003, 0.015, step=0.001, description="目标止盈%"),
                ParameterDef("stop_pct", "float", 0.002, 0.010, step=0.001, description="止损%"),
                ParameterDef("max_hold_seconds", "int", 30, 300, description="最大持仓秒数"),
            ],
            "grid": [
                ParameterDef("grid_count", "int", 3, 15, description="网格层数"),
                ParameterDef("grid_spacing_pct", "float", 0.005, 0.05, step=0.001, description="网格间距%"),
                ParameterDef("base_order_qty", "float", 1, 20, step=0.5, description="基础订单量"),
                ParameterDef("take_profit_pct", "float", 0.005, 0.03, step=0.001, description="止盈%"),
                ParameterDef("stop_loss_pct", "float", 0.01, 0.08, step=0.002, description="整体止损%"),
                ParameterDef("max_positions", "int", 3, 12, description="最大持仓层数"),
                ParameterDef("leverage", "int", 1, 10, description="杠杆倍数"),
            ],
            "arbitrage": [
                ParameterDef("min_spread_pct", "float", 0.0005, 0.005, step=0.0001, description="最小价差%", log_scale=True),
                ParameterDef("max_slippage_pct", "float", 0.0001, 0.002, step=0.0001, description="最大滑点%"),
                ParameterDef("max_hold_seconds", "int", 5, 120, description="最大持仓秒数"),
                ParameterDef("order_size_pct", "float", 0.01, 0.20, step=0.01, description="订单量占比"),
                ParameterDef("funding_rate_threshold", "float", 0.0001, 0.005, step=0.0001, description="资金费率阈值", log_scale=True),
            ],
        }
        return param_sets.get(strategy_type, [])

    # ── 单阶段执行 ────────────────────────────────────────────

    async def _run_ga_phase(self, result: OptimizationPipelineResult):
        """GA全局探索阶段"""
        self._current_phase = OptimizationPhase.GA_GLOBAL
        logger.info(f"Starting GA global exploration for {result.strategy_name}")

        self._ga = GeneticOptimizer(self._config)
        self._ga.set_param_defs(self._param_defs)
        self._ga.set_fitness_fn(self._fitness_fn)

        try:
            ga_result = await asyncio.wait_for(
                self._ga.optimize(),
                timeout=self._timeout_seconds
            )
            result.ga_result = ga_result
            result.best_params = ga_result.best_params
            result.final_fitness = ga_result.best_fitness
            result.total_evaluations += ga_result.total_evaluations
            logger.info(f"GA phase complete: best_fitness={ga_result.best_fitness:.4f}, "
                       f"gen={ga_result.convergence_generation}, evals={ga_result.total_evaluations}")

        except asyncio.TimeoutError:
            result.errors.append("GA phase timed out")
            logger.error("GA phase timed out")

    async def _run_bo_phase(self, result: OptimizationPipelineResult, use_ga_best: bool = False):
        """BO局部精调阶段"""
        self._current_phase = OptimizationPhase.BO_LOCAL
        logger.info(f"Starting BO local refinement for {result.strategy_name}")

        self._bo = BayesianOptimizer(self._config)
        self._bo.set_param_defs(self._param_defs)
        self._bo.set_fitness_fn(self._fitness_fn)

        # 热启动：将 GA 最优参数注入为 BO 初始观测，加速局部精调收敛
        if use_ga_best and result.ga_result and result.ga_result.best_params:
            self._bo.set_warm_start(dict(result.ga_result.best_params))
            logger.info(f"BO warm-started from GA best: {result.ga_result.best_fitness:.4f}")

        try:
            bo_result = await asyncio.wait_for(
                self._bo.optimize(),
                timeout=self._timeout_seconds
            )
            result.bo_result = bo_result
            result.total_evaluations += bo_result.total_iterations

            # 如果BO结果更好就替换
            if bo_result.best_value > result.final_fitness:
                result.best_params = bo_result.best_params
                result.final_fitness = bo_result.best_value

            logger.info(f"BO phase complete: best_value={bo_result.best_value:.4f}, "
                       f"iter={bo_result.total_iterations}")

        except asyncio.TimeoutError:
            result.errors.append("BO phase timed out")

    async def _run_wf_phase(self, result: OptimizationPipelineResult):
        """WF稳健性分析阶段"""
        if self._price_data is None or len(self._price_data) == 0:
            result.errors.append("WF skipped: no price data")
            return

        self._current_phase = OptimizationPhase.WF_ANALYSIS
        logger.info(f"Starting Walk-Forward analysis for {result.strategy_name}")

        self._wf = WalkForwardAnalyzer(self._config)
        self._wf.set_param_defs(self._param_defs)
        self._wf.set_data(self._price_data)

        # 使用BO/GA的最优参数附近的网格搜索作为优化函数
        best = dict(result.best_params)
        def wf_optimizer(param_defs, train_data):
            # 简化：以当前最优为中心扰动搜索
            params = dict(best)
            for pd in param_defs:
                v = params.get(pd.name, (pd.low + pd.high) / 2)
                # 加小扰动
                noise = v * np.random.normal(0, 0.05)
                params[pd.name] = pd.clamp(v + noise)
            return params

        self._wf.set_optimize_fn(wf_optimizer)
        # 注入评估函数：复用统一 fitness 评估参数在验证集上的表现（OOS）
        def wf_eval(params, data):
            return self._eval_fitness(params, data)
        self._wf.set_eval_fn(wf_eval)

        try:
            wf_result = await asyncio.wait_for(
                self._wf.analyze(),
                timeout=self._timeout_seconds
            )
            result.wf_result = wf_result
            logger.info(f"WF analysis complete: robustness={wf_result.robustness_score:.2f}, "
                       f"overfit={wf_result.overfit_risk}")

        except asyncio.TimeoutError:
            result.errors.append("WF phase timed out")

    async def _run_mc_phase(self, result: OptimizationPipelineResult):
        """MC验证阶段"""
        if self._price_data is None or len(self._price_data) == 0:
            result.errors.append("MC skipped: no price data")
            return

        self._current_phase = OptimizationPhase.MC_VALIDATION
        logger.info(f"Starting Monte Carlo validation for {result.strategy_name}")

        self._mc = MonteCarloValidator(self._config)
        self._mc.set_param_defs(self._param_defs)
        self._mc.set_best_params(result.best_params)
        self._mc.set_data(self._price_data)

        def mc_eval(params, data):
            # 使用统一 fitness 评估，data 传入以支持基于行情的独立验证
            return self._eval_fitness(params, data)

        self._mc.set_eval_fn(mc_eval)

        try:
            mc_result = await asyncio.wait_for(
                self._mc.validate(),
                timeout=self._timeout_seconds
            )
            result.mc_result = mc_result
            logger.info(f"MC validation complete: robustness={mc_result.robustness_index:.1f}/100, "
                       f"risk={mc_result.risk_level}")

        except asyncio.TimeoutError:
            result.errors.append("MC phase timed out")

    # ── 完整流水线 ────────────────────────────────────────────

    async def optimize(self, strategy_name: str,
                       strategy: OptimizationStrategy = None,
                       param_defs: List[ParameterDef] = None,
                       price_data: np.ndarray = None) -> OptimizationPipelineResult:
        """执行完整参数优化流水线"""
        if not self._fitness_fn:
            raise ValueError("No fitness function set")

        self._param_defs = param_defs or self._param_defs
        if not self._param_defs:
            self._param_defs = self.define_strategy_params(strategy_name)

        if price_data is not None:
            self._price_data = price_data

        strategy = strategy or self._default_strategy
        start_time = time.time()

        # 确定性：设置随机种子（若配置），保证结果可复现
        if self._seed is not None:
            random.seed(self._seed)
            np.random.seed(self._seed)

        result = OptimizationPipelineResult(
            strategy_name=strategy_name,
            strategy=strategy.value,
        )

        try:
            phases = []

            if strategy in (OptimizationStrategy.FULL_PIPELINE, OptimizationStrategy.GA_ONLY,
                          OptimizationStrategy.GA_BO_ONLY):
                phases.append(self._run_ga_phase)

            if strategy in (OptimizationStrategy.FULL_PIPELINE, OptimizationStrategy.BO_ONLY,
                          OptimizationStrategy.GA_BO_ONLY, OptimizationStrategy.LIGHTWEIGHT):
                # 仅当 GA 阶段已执行时，用 GA 最优参数热启动 BO
                use_ga_best = strategy in (OptimizationStrategy.FULL_PIPELINE, OptimizationStrategy.GA_BO_ONLY)
                phases.append(partial(self._run_bo_phase, use_ga_best=use_ga_best))

            if strategy in (OptimizationStrategy.FULL_PIPELINE, OptimizationStrategy.WF_ONLY):
                phases.append(self._run_wf_phase)

            if strategy in (OptimizationStrategy.FULL_PIPELINE, OptimizationStrategy.MC_ONLY,
                          OptimizationStrategy.LIGHTWEIGHT):
                phases.append(self._run_mc_phase)

            if not phases:
                raise ValueError(f"No phases defined for strategy {strategy.value}")

            async with self._lock:
                for phase_fn in phases:
                    await phase_fn(result)

            result.phase = OptimizationPhase.COMPLETE
            result.total_time_seconds = time.time() - start_time

        except Exception as e:
            result.phase = OptimizationPhase.FAILED
            result.errors.append(str(e))
            logger.error(f"Optimization pipeline failed for {strategy_name}: {e}")

        # 持久化
        self._history.append(result)
        # 内存沉淤防护：限制内存历史记录上限
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        self._persist_result(result)

        logger.info(f"Optimization pipeline complete for {strategy_name}: "
                    f"phase={result.phase.value}, best_fitness={result.final_fitness:.4f}, "
                    f"time={result.total_time_seconds:.1f}s")

        return result

    # ── 持久化 ────────────────────────────────────────────────

    def _persist_result(self, result: OptimizationPipelineResult):
        """持久化优化结果"""
        try:
            filename = f"opt_{result.strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
            filepath = os.path.join(self._persist_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            logger.debug(f"Optimization result persisted: {filepath}")
            # 磁盘沉淤防护：清理过期优化结果文件，仅保留最近 N 个
            self._prune_persist_files()
        except Exception as e:
            logger.warning(f"Failed to persist optimization result: {e}")

    def _prune_persist_files(self):
        """清理参数优化历史文件，只保留最近 _max_persist_files 个，防止磁盘沉淤。"""
        try:
            if not os.path.isdir(self._persist_dir):
                return
            files = sorted(
                (f for f in os.listdir(self._persist_dir)
                 if f.startswith("opt_") and f.endswith(".json")),
                key=lambda f: os.path.getmtime(os.path.join(self._persist_dir, f)),
                reverse=True,
            )
            for fname in files[self._max_persist_files:]:
                try:
                    os.remove(os.path.join(self._persist_dir, fname))
                except OSError:
                    pass
        except Exception as e:
            logger.debug(f"Prune optimization files error: {e}")

    def load_history(self, strategy_name: str = None) -> List[Dict[str, Any]]:
        """加载历史优化记录"""
        results = []
        try:
            if not os.path.exists(self._persist_dir):
                return results
            for fname in sorted(os.listdir(self._persist_dir), reverse=True):
                if not fname.endswith('.json'):
                    continue
                if strategy_name and strategy_name not in fname:
                    continue
                filepath = os.path.join(self._persist_dir, fname)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append(data)
        except Exception as e:
            logger.warning(f"Failed to load optimization history: {e}")
        return results

    # ── 查询接口 ──────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        latest = self.get_latest_result()
        latest_summary = None
        if latest:
            latest_summary = {
                "strategy_name": latest.get("strategy_name"),
                "phase": latest.get("phase"),
                "best_params": latest.get("best_params"),
                "final_fitness": latest.get("final_fitness"),
                "total_evaluations": latest.get("total_evaluations"),
                "errors": latest.get("errors"),
                "timestamp": latest.get("timestamp"),
            }
        persisted_count = 0
        try:
            persisted_count = len([f for f in os.listdir(self._persist_dir) if f.endswith('.json')])
        except Exception:
            persisted_count = 0
        return {
            "enabled": self._enabled,
            "current_phase": self._current_phase.value,
            "default_strategy": self._default_strategy.value,
            "param_count": len(self._param_defs),
            "history_count": len(self._history),
            "in_memory_history_count": len(self._history),
            "persisted_history_count": persisted_count,
            "last_optimization": self._history[-1].timestamp if self._history else None,
            "latest_result": latest_summary,
            "ga_status": self._ga.get_status() if self._ga else None,
            "bo_status": self._bo.get_status() if self._bo else None,
        }

    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        if not self._history:
            history = self.load_history()
            if history:
                return history[0]
            return None
        return self._history[-1].to_dict()
