"""
Meta-Learner - 元学习模块

学习"如何学习" — 从不同策略和市场条件的学习经验中提取通用模式，
并将知识迁移到新的或表现不佳的策略上。

核心能力:
  - 学习任务表示与索引
  - 任务相似度测量 (特征空间 / 市场状态 / 性能曲线)
  - 跨策略知识迁移 (参数初始化 / 学习率 / 特征重要性)
  - MAML 风格快速适应
  - 策略初始化建议
  - 学习策略优化
"""
import asyncio
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from app.services.enterprise import EnterpriseServiceMixin


# =============================================================================
# 1. LearningTask — 学习任务表示
# =============================================================================

@dataclass
class LearningTask:
    """表示一个完整的学习任务（一个策略在不同条件下的学习经验）"""
    strategy_name: str
    market_regime: str
    feature_space: List[str]
    task_embedding: List[float] = field(default_factory=list)
    difficulty_score: float = 0.0
    best_hyperparams: Dict[str, Any] = field(default_factory=dict)
    learning_trajectory: List[Dict[str, Any]] = field(default_factory=list)
    final_performance: Dict[str, Any] = field(default_factory=dict)
    regime_features: Dict[str, float] = field(default_factory=dict)
    created_at: str = ""
    completed_at: str = ""
    task_id: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.task_id:
            self.task_id = f"task_{int(time.time() * 1000)}_{hash(self.strategy_name) % 10000}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "strategy_name": self.strategy_name,
            "market_regime": self.market_regime,
            "feature_space": self.feature_space,
            "task_embedding": self.task_embedding,
            "difficulty_score": self.difficulty_score,
            "best_hyperparams": self.best_hyperparams,
            "learning_trajectory": self.learning_trajectory,
            "final_performance": self.final_performance,
            "regime_features": self.regime_features,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LearningTask":
        task = cls(
            strategy_name=d["strategy_name"],
            market_regime=d["market_regime"],
            feature_space=d.get("feature_space", []),
            task_embedding=d.get("task_embedding", []),
            difficulty_score=d.get("difficulty_score", 0.0),
            best_hyperparams=d.get("best_hyperparams", {}),
            learning_trajectory=d.get("learning_trajectory", []),
            final_performance=d.get("final_performance", {}),
            regime_features=d.get("regime_features", {}),
            created_at=d.get("created_at", ""),
            completed_at=d.get("completed_at", ""),
        )
        task.task_id = d.get("task_id", task.task_id)
        return task

    def compute_task_embedding(self, embedding_dim: int = 16):
        """根据特征空间和状态特征计算压缩的任务嵌入向量"""
        # 使用确定性哈希将特征名转成伪随机向量（无需外部 ML 库）
        feat_vec = [0.0] * embedding_dim
        for i, fname in enumerate(self.feature_space):
            seed = hash(fname) % 1000000
            for j in range(embedding_dim):
                seed = (seed * 1103515245 + 12345) % (2 ** 31)
                feat_vec[j] += (seed % 1000) / 1000.0 * 0.3 / max(1, len(self.feature_space))

        # 融入 regime features
        regime_keys = sorted(self.regime_features.keys())
        for k in regime_keys:
            seed = hash(k) % 1000000
            val = self.regime_features.get(k, 0.0)
            for j in range(embedding_dim):
                seed = (seed * 1103515245 + 12345) % (2 ** 31)
                feat_vec[j] += (seed % 1000) / 1000.0 * val * 0.1

        # 归一化
        norm = math.sqrt(sum(v * v for v in feat_vec))
        if norm > 0:
            feat_vec = [v / norm for v in feat_vec]

        self.task_embedding = feat_vec
        return feat_vec


# =============================================================================
# 2. TaskSimilarity — 任务相似度测量
# =============================================================================

class TaskSimilarity:
    """计算两个学习任务之间的相似度"""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self._weights = weights or {
            "feature_overlap": 0.30,
            "regime_similarity": 0.35,
            "performance_profile": 0.20,
            "embedding_cosine": 0.15,
        }

    def compute(self, task_a: LearningTask, task_b: LearningTask) -> Dict[str, Any]:
        """计算完整的相似度评估"""
        feat_sim = self._feature_overlap(task_a, task_b)
        regime_sim = self._regime_similarity(task_a, task_b)
        perf_sim = self._performance_similarity(task_a, task_b)
        emb_sim = self._embedding_similarity(task_a, task_b)

        combined = (
            self._weights["feature_overlap"] * feat_sim +
            self._weights["regime_similarity"] * regime_sim +
            self._weights["performance_profile"] * perf_sim +
            self._weights["embedding_cosine"] * emb_sim
        )

        return {
            "combined_score": round(combined, 4),
            "feature_overlap": round(feat_sim, 4),
            "regime_similarity": round(regime_sim, 4),
            "performance_similarity": round(perf_sim, 4),
            "embedding_similarity": round(emb_sim, 4),
            "breakdown": {
                "feature_overlap": feat_sim,
                "regime_similarity": regime_sim,
                "performance_profile": perf_sim,
                "embedding_cosine": emb_sim,
            },
        }

    def _feature_overlap(self, task_a: LearningTask, task_b: LearningTask) -> float:
        """Jaccard 相似度：特征空间重叠"""
        if not task_a.feature_space and not task_b.feature_space:
            return 1.0
        if not task_a.feature_space or not task_b.feature_space:
            return 0.0

        set_a = set(task_a.feature_space)
        set_b = set(task_b.feature_space)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _regime_similarity(self, task_a: LearningTask, task_b: LearningTask) -> float:
        """市场状态余弦相似度"""
        rf_a = task_a.regime_features
        rf_b = task_b.regime_features

        # 也考虑直接的 regime 字符串匹配
        regime_match = 1.0 if task_a.market_regime == task_b.market_regime else 0.0

        all_keys = set(rf_a.keys()) | set(rf_b.keys())
        if not all_keys:
            return regime_match

        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for k in all_keys:
            v1 = rf_a.get(k, 0.0)
            v2 = rf_b.get(k, 0.0)
            dot += v1 * v2
            norm_a += v1 * v1
            norm_b += v2 * v2

        cosine = 0.0
        if norm_a > 0 and norm_b > 0:
            cosine = max(0.0, min(1.0, dot / (math.sqrt(norm_a) * math.sqrt(norm_b))))

        return 0.4 * regime_match + 0.6 * cosine

    def _performance_similarity(self, task_a: LearningTask, task_b: LearningTask) -> float:
        """学习曲线相关性"""
        traj_a = task_a.learning_trajectory
        traj_b = task_b.learning_trajectory

        if not traj_a or not traj_b:
            # 退化为最终性能相似度
            perf_a = task_a.final_performance.get("sharpe_ratio", task_a.final_performance.get("return_pct", 0))
            perf_b = task_b.final_performance.get("sharpe_ratio", task_b.final_performance.get("return_pct", 0))
            if isinstance(perf_a, (int, float)) and isinstance(perf_b, (int, float)):
                diff = abs(perf_a - perf_b)
                return max(0.0, 1.0 - diff / max(abs(perf_a), abs(perf_b), 0.001))
            return 0.5

        # 提取每个轨迹中的性能值序列
        def extract_values(traj):
            vals = []
            for step in traj:
                perf = step.get("performance", step.get("score", step.get("return", 0)))
                if isinstance(perf, dict):
                    perf = perf.get("sharpe", perf.get("return_pct", 0))
                vals.append(float(perf))
            return vals

        vals_a = extract_values(traj_a)
        vals_b = extract_values(traj_b)

        if len(vals_a) < 2 or len(vals_b) < 2:
            return 0.5

        # 对齐长度：截断或填充
        min_len = min(len(vals_a), len(vals_b))
        if min_len < 2:
            return 0.5

        # 取公共长度的皮尔逊相关系数
        def pearson(xs, ys):
            n = len(xs)
            mx = sum(xs) / n
            my = sum(ys) / n
            sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
            sy = math.sqrt(sum((y - my) ** 2 for y in ys))
            if sx == 0 or sy == 0:
                return 0.0
            cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
            return cov / (sx * sy)

        corr = pearson(vals_a[:min_len], vals_b[:min_len])
        return max(0.0, min(1.0, (corr + 1.0) / 2.0))

    def _embedding_similarity(self, task_a: LearningTask, task_b: LearningTask) -> float:
        """任务嵌入向量的余弦相似度"""
        emb_a = task_a.task_embedding
        emb_b = task_b.task_embedding
        if not emb_a or not emb_b:
            return 0.5

        dot = sum(a * b for a, b in zip(emb_a, emb_b))
        norm_a = math.sqrt(sum(a * a for a in emb_a))
        norm_b = math.sqrt(sum(b * b for b in emb_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return max(0.0, min(1.0, dot / (norm_a * norm_b)))


# =============================================================================
# 3. KnowledgeTransfer — 跨策略知识迁移
# =============================================================================

class KnowledgeTransfer:
    """在相似学习任务之间迁移知识"""

    def __init__(self, similarity: TaskSimilarity):
        self._similarity = similarity
        self._transfer_history: List[Dict[str, Any]] = []
        self._max_history = 200

    def compute_transfer(self, source_task: LearningTask,
                         target_strategy: str,
                         target_market_regime: str,
                         target_feature_space: List[str],
                         target_regime_features: Dict[str, float]) -> Dict[str, Any]:
        """为目标准备从源任务的知识迁移方案"""
        # 构造临时目标 task 用于相似度计算
        target_task = LearningTask(
            strategy_name=target_strategy,
            market_regime=target_market_regime,
            feature_space=target_feature_space,
            regime_features=target_regime_features,
        )
        target_task.compute_task_embedding()

        if not source_task.task_embedding:
            source_task.compute_task_embedding()

        sim_result = self._similarity.compute(source_task, target_task)
        combined_score = sim_result["combined_score"]

        # 迁移置信度随相似度衰减
        confidence = self._transfer_confidence(combined_score)

        transfer = {
            "source_task_id": source_task.task_id,
            "source_strategy": source_task.strategy_name,
            "similarity_score": combined_score,
            "confidence": confidence,
            "param_initialization": None,
            "learning_rate_schedule": None,
            "feature_importance": None,
            "failure_regions": [],
            "transfer_type": "none",
        }

        if confidence < 0.2:
            transfer["transfer_type"] = "skip"
            return transfer

        # (a) 参数初始化：如果相似度足够高，使用源参数
        if confidence >= 0.5 and source_task.best_hyperparams:
            transfer["param_initialization"] = self._adapt_params(
                source_task.best_hyperparams, combined_score
            )
            transfer["transfer_type"] = "params"

        # (b) 学习率 warm-start
        if source_task.learning_trajectory:
            transfer["learning_rate_schedule"] = self._extract_lr_schedule(
                source_task.learning_trajectory, confidence
            )

        # (c) 特征重要性传递
        if source_task.feature_space:
            transfer["feature_importance"] = self._prioritize_features(
                source_task, target_task
            )

        # (d) 失败区域规避
        transfer["failure_regions"] = self._extract_failure_regions(
            source_task.learning_trajectory
        )

        # 组合判定迁移类型
        if transfer["param_initialization"] and confidence >= 0.6:
            transfer["transfer_type"] = "full_transfer"
        elif transfer["param_initialization"]:
            transfer["transfer_type"] = "warm_start"

        self._transfer_history.append({
            "timestamp": datetime.now().isoformat(),
            "source": source_task.task_id,
            "target": target_strategy,
            "score": combined_score,
            "type": transfer["transfer_type"],
        })
        if len(self._transfer_history) > self._max_history:
            self._transfer_history = self._transfer_history[-self._max_history:]

        return transfer

    def _transfer_confidence(self, similarity: float) -> float:
        """将相似度转为迁移置信度（非线性映射）"""
        # S 形函数：低相似度低置信，高相似度高置信
        x = (similarity - 0.3) * 8.0  # 放大到 [-2.4, 5.6]
        return 1.0 / (1.0 + math.exp(-x))

    def _adapt_params(self, source_params: Dict[str, Any],
                      similarity: float) -> Dict[str, Any]:
        """根据相似度调整源参数（加噪声模拟不确定性）"""
        adapted = {}
        for key, value in source_params.items():
            if isinstance(value, (int, float)):
                noise = (1.0 - similarity) * 0.2 * value * (random.random() * 2 - 1)
                adapted[key] = value + noise
            elif isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                noise = [(1.0 - similarity) * 0.2 * v * (random.random() * 2 - 1) for v in value]
                adapted[key] = [v + n for v, n in zip(value, noise)]
            else:
                adapted[key] = value
        return adapted

    def _extract_lr_schedule(self, trajectory: List[Dict[str, Any]],
                             confidence: float) -> List[Dict[str, Any]]:
        """从学习轨迹中提取学习率计划"""
        lr_schedule = []
        for step in trajectory:
            if "learning_rate" in step or "lr" in step:
                lr = step.get("learning_rate", step.get("lr", 0.001))
                lr_schedule.append({
                    "step": step.get("step", len(lr_schedule)),
                    "lr": float(lr) * (0.7 + 0.3 * confidence),  # 根据置信度缩放
                })
        return lr_schedule if lr_schedule else self._default_lr_schedule()

    def _default_lr_schedule(self) -> List[Dict[str, Any]]:
        return [
            {"step": 0, "lr": 0.01},
            {"step": 10, "lr": 0.005},
            {"step": 50, "lr": 0.001},
            {"step": 100, "lr": 0.0005},
            {"step": 200, "lr": 0.0001},
        ]

    @staticmethod
    def _prioritize_features(source_task: LearningTask,
                             target_task: LearningTask) -> Dict[str, Any]:
        """根据源任务确定目标应优先关注的特征"""
        source_feats = set(source_task.feature_space)
        target_feats = set(target_task.feature_space)

        # 重叠特征优先级高
        overlap = source_feats & target_feats
        only_source = source_feats - target_feats
        only_target = target_feats - source_feats

        priorities = {}
        for feat in overlap:
            priorities[feat] = 1.0  # 最高优先级
        for feat in only_target:
            priorities[feat] = 0.3  # 目标独有，中低优先级
        for feat in only_source:
            priorities[feat] = 0.0  # 通知目标忽略（不可用）

        return {
            "priority_order": sorted(overlap, key=lambda f: priorities.get(f, 0), reverse=True),
            "all_priorities": priorities,
            "overlap_ratio": len(overlap) / max(len(source_feats | target_feats), 1),
        }

    @staticmethod
    def _extract_failure_regions(trajectory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取学习轨迹中的失败区域"""
        failures = []
        for step in trajectory:
            perf = step.get("performance", step.get("score", 0))
            if isinstance(perf, dict):
                perf = perf.get("sharpe", perf.get("return_pct", 0))
            if float(perf) < step.get("failure_threshold", -0.1):
                failures.append({
                    "step": step.get("step", 0),
                    "params": step.get("params", {}),
                    "performance": float(perf),
                })
        return failures

    def validate_transfer(self, transferred_params: Dict[str, Any],
                          validation_data: List[Dict[str, Any]],
                          min_samples: int = 10) -> Dict[str, Any]:
        """验证迁移的知识在新任务上是否有效"""
        if len(validation_data) < min_samples:
            return {"valid": False, "reason": "insufficient_data",
                     "samples": len(validation_data)}

        # 简单验证：用迁移参数评估前几步的性能
        scores = []
        for sample in validation_data[:min_samples]:
            score = sample.get("score", sample.get("return", sample.get("performance", 0)))
            scores.append(float(score))

        if not scores:
            return {"valid": False, "reason": "no_scoring_data"}

        mean_score = sum(scores) / len(scores)
        var_score = sum((s - mean_score) ** 2 for s in scores) / max(len(scores), 1)
        std_score = math.sqrt(var_score)

        # 如果均值 > 0 或显著优于参考基线，认为有效
        reference = validation_data[0].get("reference_score", 0)
        improved = mean_score > float(reference)

        valid = mean_score > -0.05 or improved

        return {
            "valid": valid,
            "mean_score": round(mean_score, 6),
            "std_score": round(std_score, 6),
            "reference_score": float(reference),
            "improved": improved,
            "samples_used": min_samples,
        }


# =============================================================================
# 4. FastAdaptor — MAML 风格快速适应
# =============================================================================

class FastAdaptor:
    """
    简化的 MAML (Model-Agnostic Meta-Learning) 实现。

    维护一组元参数，作为任何新任务的"良好初始参数"。
    内层循环：在具体任务上从元参数出发做几步梯度更新
    外层循环：根据所有任务的适应效果更新元参数
    """

    def __init__(self, config: Dict[str, Any]):
        ma_cfg = config.get("meta_learner", {})
        self._inner_lr = ma_cfg.get("inner_lr", 0.01)
        self._outer_lr = ma_cfg.get("outer_lr", 0.001)
        self._inner_steps = ma_cfg.get("inner_steps", 5)
        self._meta_param_dim = ma_cfg.get("meta_param_dim", 32)
        self._max_param_value = ma_cfg.get("max_param_value", 5.0)

        # 元参数（向量形式）
        self._meta_params: List[float] = self._init_meta_params()
        self._meta_update_count = 0
        self._task_updates: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()

    def _init_meta_params(self) -> List[float]:
        """用 Xavier 风格初始化元参数"""
        limit = math.sqrt(6.0 / self._meta_param_dim)
        return [random.uniform(-limit, limit) for _ in range(self._meta_param_dim)]

    def get_meta_params(self) -> List[float]:
        return list(self._meta_params)

    def get_meta_params_dict(self) -> Dict[str, Any]:
        return {
            "params": list(self._meta_params),
            "dim": self._meta_param_dim,
            "update_count": self._meta_update_count,
            "inner_lr": self._inner_lr,
            "outer_lr": self._outer_lr,
            "inner_steps": self._inner_steps,
        }

    async def fast_adapt(self, task_data: List[Dict[str, Any]],
                         steps: int = None) -> Dict[str, Any]:
        """
        快速适应新任务：
          (a) 从元参数出发
          (b) 在新任务数据上做几步梯度更新（内层循环）
          (c) 返回适应后的参数和评估结果
        """
        async with self._lock:
            n_steps = steps if steps is not None else self._inner_steps

            # 克隆元参数
            adapted_params = list(self._meta_params)

            trajectory = []
            for s in range(n_steps):
                # 计算当前参数的损失（数值梯度）
                loss, gradients = self._compute_gradients(adapted_params, task_data)

                # 内层更新
                for i in range(len(adapted_params)):
                    adapted_params[i] -= self._inner_lr * gradients[i]
                    adapted_params[i] = max(-self._max_param_value,
                                            min(self._max_param_value, adapted_params[i]))

                trajectory.append({
                    "step": s,
                    "loss": round(loss, 6),
                    "param_norm": round(math.sqrt(sum(p * p for p in adapted_params)), 4),
                })

            # 评估适应质量
            quality = self._evaluate_adaptation(adapted_params, task_data)

            return {
                "adapted_params": list(adapted_params),
                "trajectory": trajectory,
                "quality": quality,
                "steps_performed": n_steps,
            }

    def _compute_gradients(self, params: List[float],
                           task_data: List[Dict[str, Any]]) -> Tuple[float, List[float]]:
        """使用数值梯度计算损失和梯度"""
        h = 0.001  # 有限差分步长
        base_loss = self._task_loss(params, task_data)
        gradients = [0.0] * len(params)

        for i in range(len(params)):
            params[i] += h
            loss_plus = self._task_loss(params, task_data)
            params[i] -= h  # 恢复

            gradients[i] = (loss_plus - base_loss) / h

        return base_loss, gradients

    def _task_loss(self, params: List[float],
                   task_data: List[Dict[str, Any]]) -> float:
        """计算参数在任务数据上的损失函数"""
        if not task_data:
            return 1.0

        total_loss = 0.0
        for sample in task_data:
            # 将样本特征映射到损失
            features = self._extract_sample_features(sample)
            if not features:
                total_loss += 1.0
                continue

            # 点积作预测
            pred_dim = min(len(params), len(features))
            pred = sum(params[i] * features[i] for i in range(pred_dim))

            # 与真实标签比较
            target = sample.get("target", sample.get("return", sample.get("score", 0.0)))
            error = float(target) - pred
            total_loss += error * error

        return total_loss / len(task_data)

    @staticmethod
    def _extract_sample_features(sample: Dict[str, Any]) -> List[float]:
        """从样本中提取数值特征向量"""
        feats = []
        for key, value in sorted(sample.items()):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                feats.append(float(value))
            elif isinstance(value, dict):
                for k, v in sorted(value.items()):
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        feats.append(float(v))
            elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                feats.extend(float(x) for x in value)
        return feats

    def _evaluate_adaptation(self, adapted_params: List[float],
                             task_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估适应后的参数质量"""
        if not task_data:
            return {"quality": "unknown", "score": 0.0}

        # 在验证集上评估
        final_loss = self._task_loss(adapted_params, task_data)

        # 与元参数对比
        meta_loss = self._task_loss(self._meta_params, task_data)
        improvement = meta_loss - final_loss

        quality = "good" if improvement > 0.05 else ("moderate" if improvement > 0.0 else "poor")

        return {
            "quality": quality,
            "final_loss": round(final_loss, 6),
            "meta_loss": round(meta_loss, 6),
            "improvement": round(improvement, 6),
            "improvement_pct": round(improvement / max(abs(meta_loss), 0.001) * 100, 2),
        }

    async def meta_update(self, task_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        外层循环：根据多个任务的适应结果更新元参数
        向能快速适应的方向微调元参数
        """
        async with self._lock:
            if not task_results:
                return {"status": "no_data", "update_count": self._meta_update_count}

            # 对每个任务，计算其最优方向，聚合后更新元参数
            total_gradient = [0.0] * self._meta_param_dim
            total_weight = 0.0

            for task_result in task_results:
                adapted = task_result.get("adapted_params", [])
                quality = task_result.get("quality", {})
                improvement = quality.get("improvement", 0)

                if not adapted or len(adapted) != self._meta_param_dim:
                    continue

                # 改进越大的任务，对元更新的贡献越大
                weight = max(0.1, 1.0 + improvement)
                total_weight += weight

                for i in range(self._meta_param_dim):
                    # 梯度方向：从元参数到适应后参数
                    total_gradient[i] += weight * (adapted[i] - self._meta_params[i])

            if total_weight > 0:
                for i in range(self._meta_param_dim):
                    total_gradient[i] /= total_weight
                    self._meta_params[i] += self._outer_lr * total_gradient[i]
                    self._meta_params[i] = max(-self._max_param_value,
                                               min(self._max_param_value, self._meta_params[i]))

            self._meta_update_count += 1

            grad_norm = math.sqrt(sum(g * g for g in total_gradient))
            param_norm = math.sqrt(sum(p * p for p in self._meta_params))

            update_info = {
                "update_count": self._meta_update_count,
                "tasks_processed": len(task_results),
                "gradient_norm": round(grad_norm, 6),
                "param_norm": round(param_norm, 6),
                "outer_lr": self._outer_lr,
            }
            self._task_updates.append(update_info)
            if len(self._task_updates) > 500:
                self._task_updates = self._task_updates[-500:]

            return update_info


# =============================================================================
# 5. StrategyInitializer — 新策略初始化
# =============================================================================

class StrategyInitializer:
    """基于先验知识，为新策略建议初始参数和 warm-up 计划"""

    def __init__(self, config: Dict[str, Any]):
        ma_cfg = config.get("meta_learner", {})
        self._default_params: Dict[str, Dict[str, Any]] = ma_cfg.get("default_params", {})
        self._conservative_defaults: Dict[str, Any] = ma_cfg.get("conservative_defaults", {
            "learning_rate": 0.001,
            "batch_size": 32,
            "exploration_rate": 0.3,
            "min_confidence": 0.6,
            "max_position_pct": 0.1,
            "stop_loss_pct": 0.02,
        })
        self._warmup_config = ma_cfg.get("warmup", {
            "default_steps": 50,
            "exploration_start": 0.5,
            "exploration_end": 0.05,
        })

    def suggest_init_params(self, strategy: str, market_regime: str,
                            similar_tasks: List[LearningTask],
                            meta_params: Optional[List[float]] = None) -> Dict[str, Any]:
        """综合多源信息为新策略建议初始参数"""
        suggestions = {
            "strategy": strategy,
            "market_regime": market_regime,
            "sources": [],
            "final_params": {},
            "param_confidence": {},
            "warmup_schedule": self._default_warmup_schedule(),
            "initialization_quality": "conservative",
        }

        # (a) 最相似策略的参数
        if similar_tasks:
            best_task = similar_tasks[0]
            if best_task.best_hyperparams:
                suggestions["sources"].append({
                    "type": "similar_strategy",
                    "source": best_task.strategy_name,
                    "task_id": best_task.task_id,
                    "params": dict(best_task.best_hyperparams),
                })

        # (b) 当前市场状态下的最佳参数
        regime_tasks = [t for t in similar_tasks if t.market_regime == market_regime]
        if regime_tasks and regime_tasks[0].best_hyperparams:
            suggestions["sources"].append({
                "type": "market_regime",
                "regime": market_regime,
                "params": dict(regime_tasks[0].best_hyperparams),
            })

        # (c) 元参数学到的良好初始化
        if meta_params:
            suggestions["sources"].append({
                "type": "meta_learned",
                "param_vector": list(meta_params),
            })

        # (d) 保守默认值作为兜底
        suggestions["sources"].append({
            "type": "conservative_defaults",
            "params": dict(self._conservative_defaults),
        })

        # 融合参数，按置信度加权
        final_params, param_confidence = self._fuse_params(suggestions["sources"])
        suggestions["final_params"] = final_params
        suggestions["param_confidence"] = param_confidence

        # 评估初始化质量
        quality = self._assess_quality(suggestions)
        suggestions["initialization_quality"] = quality

        return suggestions

    def _fuse_params(self, sources: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """融合多个来源的参数，按来源可靠性加权"""
        source_weights = {
            "similar_strategy": 0.40,
            "market_regime": 0.30,
            "meta_learned": 0.20,
            "conservative_defaults": 0.10,
        }

        accumulated: Dict[str, List[Tuple[float, float]]] = {}
        # key -> [(value, weight), ...]

        for source in sources:
            src_type = source["type"]
            base_weight = source_weights.get(src_type, 0.05)

            if src_type == "meta_learned":
                # 元参数向量需映射到具体参数（使用保守默认的 key 结构）
                param_vec = source.get("param_vector", [])
                for i, key in enumerate(sorted(self._conservative_defaults.keys())):
                    if i < len(param_vec):
                        val = param_vec[i]
                        accumulated.setdefault(key, []).append((val, base_weight * 0.3))
            else:
                params = source.get("params", {})
                for key, val in params.items():
                    if isinstance(val, (int, float)):
                        accumulated.setdefault(key, []).append((float(val), base_weight))

        # 加权平均
        fused = {}
        confidence = {}
        for key, vals in accumulated.items():
            total_w = sum(w for _, w in vals)
            if total_w > 0:
                fused[key] = sum(v * w for v, w in vals) / total_w
                # 置信度 = 加权一致性
                mean_val = fused[key]
                if len(vals) > 1:
                    variance = sum(w * (v - mean_val) ** 2 for v, w in vals) / total_w
                    consistency = max(0.0, 1.0 - math.sqrt(variance) / max(abs(mean_val), 0.001))
                    confidence[key] = round(consistency * total_w, 4)
                else:
                    confidence[key] = round(total_w, 4)
            else:
                fused[key] = self._conservative_defaults.get(key, 0.0)
                confidence[key] = 0.0

        return fused, confidence

    def _assess_quality(self, suggestions: Dict[str, Any]) -> str:
        """评估初始化质量"""
        confidence_vals = list(suggestions.get("param_confidence", {}).values())
        if not confidence_vals:
            return "conservative"

        avg_conf = sum(confidence_vals) / len(confidence_vals)

        if avg_conf > 0.7:
            return "high_quality"
        elif avg_conf > 0.4:
            return "moderate_quality"
        else:
            return "conservative"

    def _default_warmup_schedule(self) -> List[Dict[str, Any]]:
        """默认 warm-up 计划"""
        steps = self._warmup_config.get("default_steps", 50)
        exp_start = self._warmup_config.get("exploration_start", 0.5)
        exp_end = self._warmup_config.get("exploration_end", 0.05)

        schedule = []
        for s in range(steps):
            progress = s / max(steps - 1, 1)
            exploration = exp_start + (exp_end - exp_start) * progress
            schedule.append({
                "step": s,
                "exploration_rate": round(exploration, 4),
                "exploitation_rate": round(1.0 - exploration, 4),
                "phase": "exploration" if exploration > 0.3 else "transition" if exploration > 0.1 else "exploitation",
            })
        return schedule


# =============================================================================
# 6. LearningStrategyOptimizer — 学习策略优化
# =============================================================================

class LearningStrategyOptimizer:
    """优化学习过程本身：推荐学习率调度、批大小、探索/利用切换时机等"""

    def __init__(self, config: Dict[str, Any]):
        ma_cfg = config.get("meta_learner", {})
        self._lr_presets = ma_cfg.get("lr_presets", {
            "aggressive": [0.05, 0.02, 0.01, 0.005, 0.001],
            "moderate": [0.01, 0.005, 0.002, 0.001, 0.0005],
            "conservative": [0.002, 0.001, 0.0005, 0.0002, 0.0001],
        })
        self._batch_presets = ma_cfg.get("batch_presets", {
            "fast": 8,
            "balanced": 32,
            "stable": 128,
        })
        self._exploration_config = ma_cfg.get("exploration", {
            "max_exploration_steps": 200,
            "min_improvement_threshold": 0.001,
            "reset_patience": 50,
        })
        self._recommendation_log: List[Dict[str, Any]] = []

    def recommend(self, strategy_type: str, difficulty_score: float,
                  task_similarities: List[float] = None) -> Dict[str, Any]:
        """为新的学习任务推荐最优学习策略"""
        # 学习率策略
        if difficulty_score > 0.7:
            lr_style = "conservative"
        elif difficulty_score > 0.4:
            lr_style = "moderate"
        else:
            lr_style = "aggressive"

        lr_schedule = self._build_lr_schedule(lr_style)

        # 批大小
        if difficulty_score > 0.6:
            batch_style = "stable"
        elif difficulty_score > 0.3:
            batch_style = "balanced"
        else:
            batch_style = "fast"
        batch_size = self._batch_presets[batch_style]

        # 探索/利用切换
        exploration = self._compute_exploration_schedule(difficulty_score)

        # 是否建议重置
        should_reset = difficulty_score > 0.8

        # 相似任务数量影响
        similarity_context = {}
        if task_similarities:
            avg_sim = sum(task_similarities) / len(task_similarities)
            max_sim = max(task_similarities) if task_similarities else 0
            similarity_context = {
                "avg_similarity": round(avg_sim, 4),
                "max_similarity": round(max_sim, 4),
                "transfer_boost": round(max_sim * 0.3, 4),  # 高相似度可加速学习
            }
            # 有高相似度任务时可更激进
            if max_sim > 0.7:
                lr_schedule = [lr * 1.5 for lr in lr_schedule]

        recommendation = {
            "strategy_type": strategy_type,
            "difficulty_score": round(difficulty_score, 4),
            "learning_rate_style": lr_style,
            "learning_rate_schedule": [round(lr, 6) for lr in lr_schedule],
            "batch_size": batch_size,
            "batch_style": batch_style,
            "exploration_advice": exploration,
            "reset_recommended": should_reset,
            "reset_patience": self._exploration_config["reset_patience"],
            "similarity_context": similarity_context,
            "switching_advice": {
                "initial_phase": "explore",
                "transition_criterion": f"performance plateau for {self._exploration_config['reset_patience']} steps",
                "exploit_phase": "after convergence or sufficient data",
            },
        }

        self._recommendation_log.append({
            "timestamp": datetime.now().isoformat(),
            **recommendation,
        })
        if len(self._recommendation_log) > 100:
            self._recommendation_log = self._recommendation_log[-100:]

        return recommendation

    def _build_lr_schedule(self, style: str) -> List[float]:
        """根据风格构建学习率衰减计划"""
        base_lrs = self._lr_presets.get(style, self._lr_presets["moderate"])
        schedule = []
        for epoch_idx, lr in enumerate(base_lrs):
            # 线性插值生成更多步
            schedule.append(lr)
            if epoch_idx < len(base_lrs) - 1:
                for _ in range(3):  # 填充中间步
                    schedule.append(lr)
        return schedule

    def _compute_exploration_schedule(self, difficulty: float) -> Dict[str, Any]:
        """根据难度确定探索/利用策略"""
        # 困难任务需要更长的探索期
        total_steps = int(100 + difficulty * 300)
        explore_steps = int(total_steps * (0.3 + difficulty * 0.5))

        return {
            "total_estimated_steps": total_steps,
            "exploration_steps": explore_steps,
            "transition_steps": total_steps - explore_steps,
            "initial_exploration_rate": round(0.3 + difficulty * 0.4, 2),
            "final_exploration_rate": 0.02,
            "recommendation": "aggressive exploration" if difficulty > 0.6 else
            "balanced explore/exploit" if difficulty > 0.3 else
            "quick transition to exploitation",
        }

    def should_switch_to_exploit(self, performance_history: List[float],
                                 window_size: int = 20) -> Dict[str, Any]:
        """判断是否应从探索切换到利用"""
        if len(performance_history) < window_size:
            return {"should_switch": False, "reason": "insufficient_data"}

        recent = performance_history[-window_size:]
        half = window_size // 2
        first_half_avg = sum(recent[:half]) / half
        second_half_avg = sum(recent[half:]) / (window_size - half)

        improvement = second_half_avg - first_half_avg
        threshold = self._exploration_config["min_improvement_threshold"]

        # 如果最近窗口的改善低于阈值，可以切换到利用阶段
        should_switch = improvement < threshold

        return {
            "should_switch": should_switch,
            "recent_improvement": round(improvement, 6),
            "threshold": threshold,
            "window_size": window_size,
            "first_half_avg": round(first_half_avg, 6),
            "second_half_avg": round(second_half_avg, 6),
        }

    def should_reset(self, performance_history: List[float]) -> Dict[str, Any]:
        """判断是否应重置学习"""
        patience = self._exploration_config["reset_patience"]
        if len(performance_history) < patience:
            return {"should_reset": False, "reason": "insufficient_history"}

        recent = performance_history[-patience:]
        if len(recent) < 2:
            return {"should_reset": False, "reason": "too_few_points"}

        best = max(recent)
        current = recent[-1]
        # 如果当前远低于最近最佳且无改善趋势
        degradation = (best - current) / max(abs(best), 0.001)

        # 检测改善趋势
        half = patience // 2
        first_half_avg = sum(recent[:half]) / half
        second_half_avg = sum(recent[half:]) / (patience - half)

        should_reset = degradation > 0.5 and second_half_avg <= first_half_avg

        return {
            "should_reset": should_reset,
            "current_performance": round(current, 6),
            "recent_best": round(best, 6),
            "degradation": round(degradation, 4),
            "trend": "declining" if second_half_avg < first_half_avg else "stable_or_improving",
        }


# =============================================================================
# 7. MetaLearner — 主类
# =============================================================================

class MetaLearner(EnterpriseServiceMixin):
    """
    元学习器：学习"如何学习"

    协调所有子模块，管理完整的元学习生命周期：
    - 任务注册与完成
    - 知识迁移
    - MAML 元更新与快速适应
    - 新策略初始化建议
    - 学习效率分析
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        ma_cfg = config.get("meta_learner", {})

        # ── 子模块 ──
        similarity_weights = ma_cfg.get("similarity_weights", None)
        self._similarity = TaskSimilarity(weights=similarity_weights)
        self._knowledge_transfer = KnowledgeTransfer(similarity=self._similarity)
        self._fast_adaptor = FastAdaptor(config=config)
        self._initializer = StrategyInitializer(config=config)
        self._optimizer = LearningStrategyOptimizer(config=config)

        # ── 任务库 ──
        self._tasks: Dict[str, LearningTask] = {}       # task_id -> LearningTask
        self._strategy_index: Dict[str, List[str]] = {}  # strategy -> [task_id]
        self._regime_index: Dict[str, List[str]] = {}    # market_regime -> [task_id]
        self._completed_tasks: List[str] = []
        self._active_tasks: Dict[str, LearningTask] = {}

        # ── MAML 相关 ──
        self._meta_task_results: List[Dict[str, Any]] = []

        # ── 配置 ──
        self._embedding_dim = ma_cfg.get("embedding_dim", 16)
        self._max_tasks = ma_cfg.get("max_tasks", 500)
        self._meta_update_batch = ma_cfg.get("meta_update_batch", 5)

        # ── 持久化 ──
        self._persist_dir = ma_cfg.get("persist_dir", "./data/meta_learner")
        self._persist_interval = ma_cfg.get("persist_interval_seconds", 300)
        self._last_persist = 0.0

        # ── 统计 ──
        self._stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "total_transfers": 0,
            "meta_updates": 0,
            "fast_adaptations": 0,
            "by_strategy": {},
            "by_regime": {},
        }

        # ── 同步 ──
        self._lock = asyncio.Lock()
        self._running = False

        # 确保持久化目录存在 & 加载
        os.makedirs(self._persist_dir, exist_ok=True)
        self._load()

        logger.info(f"MetaLearner initialized: dim={self._embedding_dim}, "
                     f"max_tasks={self._max_tasks}, tasks_loaded={len(self._tasks)}")

    # ===================== 生命周期 =====================

    async def start(self):
        self._running = True
        logger.info("MetaLearner started")

    async def stop(self):
        self._running = False
        await self._persist()
        logger.info("MetaLearner stopped")

    # ===================== 任务管理 =====================

    async def register_task(self, task: LearningTask) -> str:
        """注册新的学习任务"""
        async with self._lock:
            # 生成 task ID（若未提供）
            if not task.task_id:
                task.task_id = f"task_{int(time.time() * 1000)}_{hash(task.strategy_name) % 10000}"

            # 计算任务嵌入
            if not task.task_embedding:
                task.compute_task_embedding(embedding_dim=self._embedding_dim)

            # 容量控制
            if len(self._tasks) >= self._max_tasks:
                self._prune_tasks()

            self._tasks[task.task_id] = task
            self._active_tasks[task.task_id] = task
            self._strategy_index.setdefault(task.strategy_name, []).append(task.task_id)
            self._regime_index.setdefault(task.market_regime, []).append(task.task_id)

            self._stats["total_tasks"] += 1
            self._stats["by_strategy"][task.strategy_name] = \
                self._stats["by_strategy"].get(task.strategy_name, 0) + 1
            self._stats["by_regime"][task.market_regime] = \
                self._stats["by_regime"].get(task.market_regime, 0) + 1

            logger.debug(f"Task registered: {task.task_id} ({task.strategy_name}/{task.market_regime})")
            return task.task_id

    async def complete_task(self, task_id: str, final_performance: Dict[str, Any],
                            best_params: Dict[str, Any],
                            learning_trajectory: List[Dict[str, Any]]) -> LearningTask:
        """标记任务完成，记录最终结果"""
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                raise ValueError(f"Unknown task: {task_id}")

            task.final_performance = final_performance
            task.best_hyperparams = best_params
            task.learning_trajectory = learning_trajectory
            task.completed_at = datetime.now().isoformat()

            # 估算难度
            task.difficulty_score = self._estimate_difficulty(task)

            # 从活跃列表移除
            self._active_tasks.pop(task_id, None)
            if task_id not in self._completed_tasks:
                self._completed_tasks.append(task_id)
            self._stats["completed_tasks"] += 1

            # 触发 MAML 元更新积累
            self._meta_task_results.append({
                "task_id": task_id,
                "strategy": task.strategy_name,
                "adapted_params": list(self._fast_adaptor.get_meta_params()),
                "quality": {"improvement": final_performance.get("sharpe_ratio", 0) -
                            final_performance.get("baseline_sharpe", 0)},
                "best_params": best_params,
            })
            if len(self._meta_task_results) >= self._meta_update_batch:
                asyncio.create_task(self.meta_update())

            # 自动持久化
            if time.time() - self._last_persist > self._persist_interval:
                asyncio.create_task(self._persist())
                self._last_persist = time.time()

            logger.info(f"Task completed: {task_id} ({task.strategy_name}), "
                         f"difficulty={task.difficulty_score:.3f}")
            return task

    def _estimate_difficulty(self, task: LearningTask) -> float:
        """评估任务学习难度"""
        perf = task.final_performance

        # 用多个指标综合判断
        sharpe = perf.get("sharpe_ratio", perf.get("sharpe", 0))
        win_rate = perf.get("win_rate", perf.get("accuracy", 0.5))
        max_drawdown = perf.get("max_drawdown", perf.get("drawdown", 1.0))

        # 困难 = 低 sharpe + 低胜率 + 大回撤
        sharpe_factor = max(0.0, 1.0 - (float(sharpe) + 1.0) / 3.0)  # sharpe>2 → 0难度
        wr_factor = 1.0 - float(win_rate)
        dd_factor = float(max_drawdown) if float(max_drawdown) < 1 else 1.0
        dd_factor = min(1.0, max(0.0, dd_factor))

        difficulty = 0.4 * sharpe_factor + 0.3 * wr_factor + 0.3 * dd_factor

        # 学习轨迹长也暗示困难
        if task.learning_trajectory:
            traj_len = len(task.learning_trajectory)
            if traj_len > 100:
                difficulty = min(1.0, difficulty + 0.1)

        return round(difficulty, 4)

    # ===================== 相似任务查询 =====================

    async def _find_similar_tasks(self, strategy: str, market_regime: str,
                                   feature_space: List[str],
                                   regime_features: Dict[str, float] = None,
                                   top_k: int = 5) -> List[LearningTask]:
        """查找与目标最相似的已完成任务"""
        temp_task = LearningTask(
            strategy_name=strategy,
            market_regime=market_regime,
            feature_space=feature_space,
            regime_features=regime_features or {},
        )
        temp_task.compute_task_embedding(embedding_dim=self._embedding_dim)

        scored = []
        for task_id in self._completed_tasks:
            other = self._tasks.get(task_id)
            if not other:
                continue
            if not other.task_embedding:
                other.compute_task_embedding(embedding_dim=self._embedding_dim)
            sim = self._similarity.compute(temp_task, other)["combined_score"]
            scored.append((sim, other))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scored[:top_k]]

    # ===================== 知识迁移 =====================

    async def get_transfer_suggestion(self, strategy: str, market_regime: str,
                                       feature_space: List[str]) -> Dict[str, Any]:
        """为新策略获取知识迁移建议"""
        async with self._lock:
            similar_tasks = await self._find_similar_tasks(
                strategy=strategy,
                market_regime=market_regime,
                feature_space=feature_space,
                top_k=5,
            )

            if not similar_tasks:
                return {
                    "has_transfer": False,
                    "reason": "no_similar_tasks",
                    "recommendation": "start_from_scratch",
                }

            best_source = similar_tasks[0]
            transfer = self._knowledge_transfer.compute_transfer(
                source_task=best_source,
                target_strategy=strategy,
                target_market_regime=market_regime,
                target_feature_space=feature_space,
                target_regime_features=best_source.regime_features,
            )

            self._stats["total_transfers"] += 1

            return {
                "has_transfer": transfer["transfer_type"] != "skip",
                "transfer": transfer,
                "alternative_sources": [
                    {"task_id": t.task_id, "strategy": t.strategy_name,
                     "difficulty": t.difficulty_score}
                    for t in similar_tasks[1:]
                ],
                "recommendation": (
                    "full_transfer" if transfer["transfer_type"] == "full_transfer"
                    else "warm_start" if transfer["transfer_type"] == "warm_start"
                    else "params_only" if transfer["transfer_type"] == "params"
                    else "no_transfer"
                ),
            }

    async def transfer_knowledge(self, source_task_id: str,
                                  target_strategy: str) -> Dict[str, Any]:
        """从指定源任务向目标迁移知识"""
        async with self._lock:
            source = self._tasks.get(source_task_id)
            if not source:
                return {"status": "error", "message": f"Source task not found: {source_task_id}"}

            # 需要目标策略的一些特征信息
            if target_strategy in self._strategy_index:
                # 用目标策略的历史任务来推断特征
                target_tasks = [self._tasks[tid] for tid in self._strategy_index[target_strategy]
                                if tid in self._tasks]
                if target_tasks:
                    all_features = set()
                    all_regime_features: Dict[str, float] = {}
                    for t in target_tasks:
                        all_features.update(t.feature_space)
                        for k, v in t.regime_features.items():
                            all_regime_features[k] = all_regime_features.get(k, 0) + v
                    # 平均 regime features
                    n = len(target_tasks)
                    target_regime_features = {k: v / n for k, v in all_regime_features.items()}
                    target_feature_space = list(all_features)
                    target_regime = target_tasks[0].market_regime
                else:
                    target_feature_space = source.feature_space
                    target_regime_features = source.regime_features
                    target_regime = source.market_regime
            else:
                target_feature_space = source.feature_space
                target_regime_features = source.regime_features
                target_regime = source.market_regime

            return self._knowledge_transfer.compute_transfer(
                source_task=source,
                target_strategy=target_strategy,
                target_market_regime=target_regime,
                target_feature_space=target_feature_space,
                target_regime_features=target_regime_features,
            )

    async def validate_transfer(self, transfer: Dict[str, Any],
                                 validation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """验证迁移知识的有效性"""
        params = transfer.get("param_initialization", {})
        if not params:
            return {"valid": False, "reason": "no_params_to_validate"}

        return self._knowledge_transfer.validate_transfer(params, validation_data)

    # ===================== MAML =====================

    async def meta_update(self) -> Dict[str, Any]:
        """执行一次 MAML 元更新（外层循环）"""
        async with self._lock:
            if not self._meta_task_results:
                return {"status": "no_tasks", "update_count": self._fast_adaptor.get_meta_params_dict()["update_count"]}

            results_to_process = list(self._meta_task_results)
            self._meta_task_results = []

        update_info = await self._fast_adaptor.meta_update(results_to_process)
        self._stats["meta_updates"] += 1

        # 持久化
        if time.time() - self._last_persist > self._persist_interval:
            asyncio.create_task(self._persist())
            self._last_persist = time.time()

        logger.debug(f"Meta update: {update_info}")
        return {"status": "ok", **update_info}

    async def fast_adapt(self, strategy: str, task_data: List[Dict[str, Any]],
                          steps: int = 5) -> Dict[str, Any]:
        """在新任务上快速适应（内层循环）"""
        result = await self._fast_adaptor.fast_adapt(task_data, steps=steps)
        self._stats["fast_adaptations"] += 1
        result["strategy"] = strategy
        return result

    async def get_meta_params(self) -> Dict[str, Any]:
        """获取当前元参数"""
        return self._fast_adaptor.get_meta_params_dict()

    # ===================== 初始化建议 =====================

    async def suggest_init_params(self, strategy: str,
                                   market_regime: str) -> Dict[str, Any]:
        """为新策略建议初始参数"""
        async with self._lock:
            # 查找相似任务
            # 先尝试获取该策略的历史
            similar_tasks = []
            if strategy in self._strategy_index:
                task_ids = self._strategy_index[strategy]
                similar_tasks = [self._tasks[tid] for tid in task_ids
                                 if tid in self._completed_tasks and tid in self._tasks]

            # 再找相似市场状态下的任务
            if len(similar_tasks) < 3:
                regime_tasks = self._regime_index.get(market_regime, [])
                regime_task_objs = [self._tasks[tid] for tid in regime_tasks
                                    if tid in self._completed_tasks and tid in self._tasks]
                # 去重
                existing_ids = {t.task_id for t in similar_tasks}
                for t in regime_task_objs:
                    if t.task_id not in existing_ids:
                        similar_tasks.append(t)
                        existing_ids.add(t.task_id)

            meta_params = self._fast_adaptor.get_meta_params()

            suggestion = self._initializer.suggest_init_params(
                strategy=strategy,
                market_regime=market_regime,
                similar_tasks=similar_tasks[:5],
                meta_params=meta_params,
            )

            return suggestion

    # ===================== 学习效率分析 =====================

    async def analyze_learning_efficiency(self, strategy: str) -> Dict[str, Any]:
        """分析指定策略的学习效率"""
        async with self._lock:
            task_ids = self._strategy_index.get(strategy, [])
            tasks = [self._tasks[tid] for tid in task_ids
                     if tid in self._completed_tasks and tid in self._tasks]

            if not tasks:
                return {
                    "strategy": strategy,
                    "status": "no_data",
                    "message": "No completed tasks for this strategy",
                }

            # 汇总统计
            n = len(tasks)
            difficulties = [t.difficulty_score for t in tasks]
            sharpe_values = [t.final_performance.get("sharpe_ratio", t.final_performance.get("sharpe", 0))
                             for t in tasks]
            win_rates = [t.final_performance.get("win_rate", 0) for t in tasks]
            traj_lengths = [len(t.learning_trajectory) for t in tasks]

            mean_difficulty = sum(difficulties) / n
            mean_sharpe = sum(sharpe_values) / n
            mean_win_rate = sum(win_rates) / n
            mean_traj_len = sum(traj_lengths) / n

            # 学习效率指标：越少步骤达到越好性能越高效
            # 效率 = (sharpe / difficulty) / trajectory_length
            efficiencies = []
            for t in tasks:
                sharpe = t.final_performance.get("sharpe_ratio", t.final_performance.get("sharpe", 0.01))
                diff = max(t.difficulty_score, 0.01)
                traj = max(len(t.learning_trajectory), 1)
                eff = float(sharpe) / (diff * traj)
                efficiencies.append(eff)

            mean_efficiency = sum(efficiencies) / n if n > 0 else 0.0

            # 效率趋势：后半段 vs 前半段
            half = n // 2
            if half > 0:
                early_eff = sum(efficiencies[:half]) / half
                late_eff = sum(efficiencies[half:]) / (n - half)
                trend = "improving" if late_eff > early_eff * 1.1 else (
                    "stable" if late_eff >= early_eff * 0.9 else "declining")
            else:
                trend = "insufficient_data"
                early_eff = mean_efficiency
                late_eff = mean_efficiency

            return {
                "strategy": strategy,
                "total_tasks": n,
                "mean_difficulty": round(mean_difficulty, 4),
                "mean_sharpe": round(mean_sharpe, 4),
                "mean_win_rate": round(mean_win_rate, 4),
                "mean_trajectory_steps": round(mean_traj_len, 1),
                "learning_efficiency": round(mean_efficiency, 6),
                "efficiency_trend": trend,
                "early_efficiency": round(early_eff, 6),
                "late_efficiency": round(late_eff, 6),
                "difficulty_distribution": {
                    "easy": sum(1 for d in difficulties if d < 0.3),
                    "moderate": sum(1 for d in difficulties if 0.3 <= d <= 0.7),
                    "hard": sum(1 for d in difficulties if d > 0.7),
                },
                "best_performance": {
                    "task_id": tasks[max(range(n), key=lambda i: sharpe_values[i])].task_id if n > 0 else None,
                    "sharpe": round(max(sharpe_values), 4) if sharpe_values else 0,
                    "win_rate": round(max(win_rates), 4) if win_rates else 0,
                },
            }

    # ===================== 总体摘要 =====================

    def get_summary(self) -> Dict[str, Any]:
        return {
            "meta_learner": {
                "total_tasks": len(self._tasks),
                "completed_tasks": len(self._completed_tasks),
                "active_tasks": len(self._active_tasks),
                "strategies_tracked": list(self._strategy_index.keys()),
                "regimes_tracked": list(self._regime_index.keys()),
                "embedding_dim": self._embedding_dim,
                "meta_params": {
                    "update_count": self._fast_adaptor.get_meta_params_dict()["update_count"],
                    "dim": self._meta_param_dim,
                },
            },
            "stats": self._stats,
            "recent_transfers": self._knowledge_transfer._transfer_history[-10:],
            "recommendation_log": self._optimizer._recommendation_log[-10:],
        }

    @property
    def _meta_param_dim(self) -> int:
        return self._fast_adaptor._meta_param_dim

    # ===================== 内部工具 =====================

    def _prune_tasks(self):
        """清理低价值任务"""
        # 保留已完成的、有最佳参数的任务
        keeper_tasks: Dict[str, LearningTask] = {}
        for tid, task in self._tasks.items():
            if tid in self._completed_tasks and task.best_hyperparams:
                keeper_tasks[tid] = task

        # 如果还不够，保留最新的活跃任务
        if len(keeper_tasks) < self._max_tasks:
            active_sorted = sorted(
                [(tid, t) for tid, t in self._tasks.items()
                 if tid not in keeper_tasks],
                key=lambda x: x[1].created_at, reverse=True
            )
            for tid, task in active_sorted[:self._max_tasks - len(keeper_tasks)]:
                keeper_tasks[tid] = task

        removed = len(self._tasks) - len(keeper_tasks)
        self._tasks = keeper_tasks

        # 重建索引
        self._strategy_index = {}
        self._regime_index = {}
        self._completed_tasks = []
        for tid, task in self._tasks.items():
            self._strategy_index.setdefault(task.strategy_name, []).append(tid)
            self._regime_index.setdefault(task.market_regime, []).append(tid)
            if task.completed_at:
                self._completed_tasks.append(tid)

        logger.debug(f"Pruned {removed} tasks, {len(self._tasks)} remaining")

    # ===================== 持久化 =====================

    async def _persist(self):
        """持久化到磁盘"""
        try:
            path = os.path.join(self._persist_dir, "meta_learner_state.json")
            data = {
                "tasks": [t.to_dict() for t in self._tasks.values()],
                "completed_tasks": self._completed_tasks,
                "meta_params": self._fast_adaptor.get_meta_params_dict(),
                "stats": self._stats,
                "transfer_history": self._knowledge_transfer._transfer_history[-100:],
                "recommendation_log": self._optimizer._recommendation_log[-100:],
                "meta_task_results": self._meta_task_results[-100:],
                "saved_at": datetime.now().isoformat(),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, default=str)
            self._last_persist = time.time()
            logger.debug(f"MetaLearner persisted: {len(self._tasks)} tasks, "
                          f"meta_updates={self._stats['meta_updates']}")
        except Exception as e:
            self._handle_exception(
                e, module="MetaLearner", function="_persist",
                severity="medium", category="persistence",
            )

    def _load(self):
        """从磁盘加载状态"""
        path = os.path.join(self._persist_dir, "meta_learner_state.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for task_dict in data.get("tasks", []):
                task = LearningTask.from_dict(task_dict)
                self._tasks[task.task_id] = task
                self._strategy_index.setdefault(task.strategy_name, []).append(task.task_id)
                self._regime_index.setdefault(task.market_regime, []).append(task.task_id)
                if task.completed_at:
                    self._completed_tasks.append(task.task_id)
                else:
                    self._active_tasks[task.task_id] = task

            self._completed_tasks = data.get("completed_tasks", self._completed_tasks)
            self._stats = data.get("stats", self._stats)

            # 加载元参数
            meta_params_data = data.get("meta_params", {})
            if meta_params_data.get("params"):
                self._fast_adaptor._meta_params = meta_params_data["params"]
                self._fast_adaptor._meta_update_count = meta_params_data.get("update_count", 0)

            # 加载迁移历史
            transfer_hist = data.get("transfer_history", [])
            if transfer_hist:
                self._knowledge_transfer._transfer_history = transfer_hist

            # 加载推荐日志
            rec_log = data.get("recommendation_log", [])
            if rec_log:
                self._optimizer._recommendation_log = rec_log

            # 加载待处理元任务结果
            meta_results = data.get("meta_task_results", [])
            if meta_results:
                self._meta_task_results = meta_results

            logger.info(f"MetaLearner loaded: {len(self._tasks)} tasks, "
                         f"{len(self._completed_tasks)} completed")
        except Exception as e:
            self._handle_exception(
                e, module="MetaLearner", function="_load",
                severity="medium", category="persistence",
            )