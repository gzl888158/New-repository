"""
自动寻优闭环协调器 AutoOptimizationCoordinator
================================================

把分散在参数优化、回测、上线、反馈调参中的四个环节串成一条带反馈的自动寻优闭环，
而不重写它们内部的优化/回测/应用算法。

闭环流程（一个策略 = 一次 run_cycle）：
    优化 Optimize  → 回测 Backtest  → 上线 Deploy  → 反馈 Feedback

设计约束：
  - 纯编排：只调用 param_optimizer / optimizer / performance_feedback 既有接口，
    不复制 GA/BO/WF/MC 优化与回测适应度计算。
  - fail-closed：上线（写回生产参数）是高风险动作，仅当优化结果通过全部门控
    （COMPLETE + 无错误 + 正适应度 + 可构建上线建议）才自动应用；否则只记录拒绝原因。
  - 幂等 + 冷却：同一策略在冷却期内跳过，避免重复寻优与日志轰炸。
  - JSON 安全：所有统计/结果经 utils.helpers 的 safe_* 清洗，json.dumps 无 NaN/Inf。
  - 可观测：维护优化次数、上线次数、拒绝分布、反馈评分，供 Dashboard / AGI 协调器
    观察「优化→回测→上线→反馈」闭环效果。
"""
import copy
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from loguru import logger

from utils.helpers import safe_float, safe_finite, safe_int


@dataclass
class OptimizationCycleResult:
    """一次自动寻优闭环的统一结果（JSON 安全）。"""
    strategy: str = ""
    cycle: int = 0
    status: str = "skipped"            # optimized / deployed / deploy_rejected / skipped / degraded
    best_fitness: float = 0.0
    best_params: Dict[str, Any] = field(default_factory=dict)
    deploy_decision: str = "skipped"   # approved / rejected / skipped
    deploy_reason: str = ""
    deploy_applied: int = 0
    feedback_score: float = 0.0
    feedback_recommendations: list = field(default_factory=list)
    message: str = ""


class AutoOptimizationCoordinator:
    """统一自动寻优协调器：编排 param_optimizer + optimizer + performance_feedback。"""

    def __init__(
        self,
        param_optimizer=None,
        optimizer=None,
        performance_feedback=None,
        recommendation_builder: Optional[Callable] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._param_optimizer = param_optimizer
        self._optimizer = optimizer                       # StrategyOptimizer（apply/persist 上线）
        self._performance_feedback = performance_feedback  # PerformanceFeedback（反馈调参）
        self._recommendation_builder = recommendation_builder  # callable(strategy, result) -> recommendation
        self._config = dict(config or {})

        self._cooldown_seconds = safe_float(self._config.get("cooldown_seconds"), 3600.0)
        if self._cooldown_seconds < 0:
            self._cooldown_seconds = 3600.0

        self._stats: Dict[str, Any] = {
            "total_cycles": 0,
            "optimized": 0,
            "deployed": 0,
            "deploy_rejected": 0,
            "skipped": 0,
            "by_strategy": {},
        }
        self._cooldown: Dict[str, float] = {}

        logger.info(
            f"AutoOptimizationCoordinator initialized: cooldown={self._cooldown_seconds:.0f}s, "
            f"param_optimizer={'on' if param_optimizer else 'off'}, "
            f"optimizer={'on' if optimizer else 'off'}, "
            f"performance_feedback={'on' if performance_feedback else 'off'}"
        )

    # ─────────────────────────────────────────────────────────────
    # 自动寻优闭环入口
    # ─────────────────────────────────────────────────────────────

    async def run_cycle(
        self,
        strategy_name: str,
        fitness_fn: Optional[Callable] = None,
        param_defs: Optional[list] = None,
        price_data=None,
    ) -> OptimizationCycleResult:
        """对单个策略执行一次自动寻优闭环。

        Args:
            strategy_name: 策略名
            fitness_fn: 回测适应度函数（callable(params[, data]) -> float）
            param_defs: 参数搜索空间
            price_data: 价格数据（用于 WF/MC 稳健性验证）
        """
        strategy = str(strategy_name or "")

        # 幂等 + 冷却
        now = time.monotonic()
        last = self._cooldown.get(strategy)
        if last is not None and (now - last) < self._cooldown_seconds:
            self._stats["skipped"] = safe_int(self._stats.get("skipped"), 0) + 1
            return OptimizationCycleResult(strategy=strategy, status="skipped", message="cooldown active")

        self._cooldown[strategy] = now
        self._stats["total_cycles"] = safe_int(self._stats.get("total_cycles"), 0) + 1

        result = OptimizationCycleResult(strategy=strategy)

        # 依赖缺失：降级跳过
        if self._param_optimizer is None:
            result.status = "degraded"
            result.message = "no param optimizer"
            return result

        # 1. 优化 Optimize + 2. 回测 Backtest（fitness_fn 内部即回测评估）
        try:
            if fitness_fn is not None:
                self._param_optimizer.set_fitness_fn(fitness_fn)
            if param_defs is not None:
                self._param_optimizer.set_param_defs(param_defs)
            if price_data is not None:
                self._param_optimizer.set_price_data(price_data)

            opt_result = await self._param_optimizer.optimize(strategy_name)
            phase = getattr(getattr(opt_result, "phase", None), "value", "unknown")
            result.best_fitness = safe_finite(getattr(opt_result, "final_fitness", 0.0), 0.0)
            result.best_params = self._sanitize_params(getattr(opt_result, "best_params", {}) or {})
            history = getattr(self._param_optimizer, "_history", None)
            result.cycle = safe_int(len(history) if history else 0, 0)

            errors = list(getattr(opt_result, "errors", []) or [])
            logger.info(
                f"[AutoOpt] optimize [{strategy}] phase={phase} fitness={result.best_fitness:.4f} errors={len(errors)}"
            )
        except Exception as e:
            result.status = "degraded"
            result.message = f"optimization failed: {e}"
            logger.warning(f"[AutoOpt] optimization failed for {strategy}: {e}")
            return result

        # 3. 上线 Deploy（fail-closed 门控）
        decision, reason, applied = await self._deploy(strategy, opt_result, phase, errors)
        result.deploy_decision = decision
        result.deploy_reason = reason
        result.deploy_applied = applied

        # 4. 反馈 Feedback
        feedback_score, recommendations = await self._feedback(strategy)
        result.feedback_score = feedback_score
        result.feedback_recommendations = recommendations

        # 终态判定
        if decision == "approved":
            result.status = "deployed" if applied > 0 else "optimized"
            result.message = f"deployed ({applied} changes)"
            self._stats["deployed"] = safe_int(self._stats.get("deployed"), 0) + 1
        elif decision == "rejected":
            result.status = "deploy_rejected"
            result.message = reason
            self._stats["deploy_rejected"] = safe_int(self._stats.get("deploy_rejected"), 0) + 1
        else:
            result.status = "optimized"
            result.message = reason
            self._stats["optimized"] = safe_int(self._stats.get("optimized"), 0) + 1

        self._record_strategy_stat(strategy, result.status)
        return result

    # ─────────────────────────────────────────────────────────────
    # 上线 Deploy（fail-closed 门控）
    # ─────────────────────────────────────────────────────────────

    async def _deploy(self, strategy, opt_result, phase, errors) -> tuple:
        """上线门控：仅通过全部检查才自动写回生产参数。"""
        # 门控 1：优化阶段必须 COMPLETE
        if phase != "complete":
            return "rejected", f"phase={phase} not complete", 0
        # 门控 2：管线无错误
        if errors:
            return "rejected", f"{len(errors)} pipeline errors", 0
        # 门控 3：正适应度
        if safe_float(getattr(opt_result, "final_fitness", 0.0), 0.0) <= 0:
            return "rejected", "non-positive fitness", 0
        # 门控 4：可构建上线建议（recommendation_builder 回调）
        recommendation = None
        if self._recommendation_builder is not None:
            try:
                recommendation = self._recommendation_builder(strategy, opt_result)
            except Exception as e:
                logger.warning(f"[AutoOpt] recommendation build failed for {strategy}: {e}")
        if not recommendation:
            return "rejected", "no deployable recommendation", 0

        # 门控 5：上线执行器可用
        if self._optimizer is None:
            return "rejected", "no strategy optimizer", 0

        try:
            apply_res = await self._optimizer.apply_optimizations(recommendation)
            applied = safe_int(apply_res.get("total_applied"), 0) if isinstance(apply_res, dict) else 0
            if applied <= 0:
                return "rejected", "no changes applied", 0
            try:
                await self._optimizer.persist_config()
            except Exception as e:
                logger.warning(f"[AutoOpt] persist config failed for {strategy}: {e}")
            return "approved", f"applied {applied} changes", applied
        except Exception as e:
            logger.warning(f"[AutoOpt] deploy failed for {strategy}: {e}")
            return "rejected", f"deploy error: {e}", 0

    # ─────────────────────────────────────────────────────────────
    # 反馈 Feedback
    # ─────────────────────────────────────────────────────────────

    async def _feedback(self, strategy) -> tuple:
        """拉取 performance_feedback 的评分与反馈建议，形成闭环反馈。"""
        if self._performance_feedback is None:
            return 0.0, []

        score = 0.0
        recommendations = []
        try:
            score_data = await self._performance_feedback.get_score(strategy, "30d")
            if isinstance(score_data, dict) and "error" not in score_data:
                # score 可能是 dict（滚动评分），取复合分或默认 0
                score = safe_float(score_data.get("score", score_data.get("composite", 0.0)), 0.0)
        except Exception as e:
            logger.debug(f"[AutoOpt] feedback score failed for {strategy}: {e}")

        try:
            fb = await self._performance_feedback.get_feedback(strategy)
            if isinstance(fb, dict):
                recommendations = list(fb.get("recommendations", []) or [])
        except Exception as e:
            logger.debug(f"[AutoOpt] feedback recommendations failed for {strategy}: {e}")

        logger.info(f"[AutoOpt] feedback [{strategy}] score={score:.3f} recs={len(recommendations)}")
        return score, recommendations

    # ─────────────────────────────────────────────────────────────
    # 统计与可观测
    # ─────────────────────────────────────────────────────────────

    def _record_strategy_stat(self, strategy: str, status: str):
        try:
            sb = self._stats["by_strategy"].setdefault(
                strategy, {"total": 0, "deployed": 0, "optimized": 0, "deploy_rejected": 0}
            )
            sb["total"] = safe_int(sb.get("total"), 0) + 1
            if status in ("deployed", "optimized", "deploy_rejected"):
                sb[status] = safe_int(sb.get(status), 0) + 1
        except Exception as e:
            logger.debug(f"[AutoOpt] strategy stat failed: {e}")

    @staticmethod
    def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """清洗参数值为 JSON 安全（消除 NaN/Inf）。"""
        out = {}
        for k, v in (params or {}).items():
            if isinstance(v, float):
                out[str(k)] = safe_finite(v, 0.0)
            elif isinstance(v, (int, str, bool)):
                out[str(k)] = v
            else:
                try:
                    out[str(k)] = safe_finite(float(v), 0.0)
                except (TypeError, ValueError):
                    out[str(k)] = str(v)
        return out

    def get_stats(self) -> Dict[str, Any]:
        """返回自动寻优闭环统计（深拷贝，不暴露内部引用）。"""
        return copy.deepcopy(self._stats)

    def get_summary(self) -> Dict[str, Any]:
        """聚合寻优统计 + 性能反馈摘要，供 Dashboard / AGI 观察闭环效果。"""
        summary = {"stats": copy.deepcopy(self._stats)}
        if self._performance_feedback is not None:
            try:
                summary["performance_feedback"] = self._performance_feedback.get_summary()
            except Exception as e:
                logger.debug(f"[AutoOpt] performance feedback summary failed: {e}")
        return summary


__all__ = ["AutoOptimizationCoordinator", "OptimizationCycleResult"]
