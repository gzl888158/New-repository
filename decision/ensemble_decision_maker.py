"""集成决策器：融合多信号源（加权投票/排序聚合等）生成综合交易决策。"""
from collections import deque
from enum import Enum
import math
import time
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
import statistics


class EnsembleMethod(Enum):
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    RANK_AGGREGATION = "rank_aggregation"
    BORDA_COUNT = "borda_count"
    HYBRID = "hybrid"


class SignalSource:
    """信号源 — 增强版：Kalman滤波器追踪可靠性 + 方向历史记录"""

    def __init__(self, name: str, weight: float = 1.0, reliability: float = 0.5):
        self.name = name
        self.weight = weight
        self.reliability = reliability
        self.signal_count = 0
        self.correct_count = 0
        self.last_updated = time.time()
        # ── Kalman滤波器状态 ──
        self._kalman_state = reliability      # 后验估计 x̂
        self._kalman_P = 1.0                   # 后验误差协方差 P
        self._kalman_Q = 0.001                 # 过程噪声（可靠性漂移速率）
        self._kalman_R = 0.15                  # 测量噪声（correctness观测噪声）
        # ── 方向历史（用于多样性度量）──
        self._direction_history: deque = deque(maxlen=50)

    def touch(self):
        self.last_updated = time.time()

    def update_reliability(self, was_correct: bool, direction: str = ""):
        """Kalman滤波器追踪信号源可靠性"""
        self.signal_count += 1
        if was_correct:
            self.correct_count += 1

        # ── Kalman预测 ──
        x_pred = self._kalman_state
        P_pred = self._kalman_P + self._kalman_Q

        # ── Kalman更新 ──
        z = 1.0 if was_correct else 0.0
        K = P_pred / (P_pred + self._kalman_R)
        self._kalman_state = x_pred + K * (z - x_pred)
        self._kalman_P = (1 - K) * P_pred

        # 同步reliability为Kalman估计值（钳制到[0,1]）
        self.reliability = max(0.0, min(1.0, self._kalman_state))
        self.last_updated = time.time()

        # 记录方向
        if direction:
            self._direction_history.append(direction)

    def get_effective_weight(self) -> float:
        staleness = time.time() - self.last_updated
        if staleness > 60:
            decay = 2.0 ** (-(staleness - 60) / 300)
        else:
            decay = 1.0
        return self.weight * self.reliability * decay

    def get_kalman_uncertainty(self) -> float:
        """返回Kalman估计的不确定性（P值）"""
        return self._kalman_P


class EnsembleDecisionMaker:
    def __init__(self, config=None):
        self.config = config or {}
        self._sources: Dict[str, SignalSource] = {}
        self._method = EnsembleMethod.WEIGHTED_AVERAGE
        self._confidence_threshold = 0.6
        self._min_votes = 2
        self._history: List[Dict[str, Any]] = []
        self._performance_history: deque = deque(maxlen=200)

    def register_source(self, name: str, weight: float = 1.0, reliability: float = 0.5):
        self._sources[name] = SignalSource(name, weight, reliability)
        logger.info(f"Registered signal source: {name}, weight: {weight}, reliability: {reliability}")

    def unregister_source(self, name: str):
        if name in self._sources:
            del self._sources[name]
            logger.info(f"Unregistered signal source: {name}")

    def set_ensemble_method(self, method: EnsembleMethod):
        self._method = method
        logger.info(f"Ensemble method changed to: {method.value}")

    def set_confidence_threshold(self, threshold: float):
        self._confidence_threshold = threshold
        logger.info(f"Confidence threshold set to: {threshold}")

    async def make_decision(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not signals:
            return {"decision": "no_signal", "confidence": 0.0, "reason": "No signals received"}

        filtered_signals = self._validate_signals(signals)
        if not filtered_signals:
            return {"decision": "no_valid_signal", "confidence": 0.0, "reason": "No valid signals"}

        if self._method == EnsembleMethod.MAJORITY_VOTE:
            return self._majority_vote(filtered_signals)
        elif self._method == EnsembleMethod.WEIGHTED_AVERAGE:
            return self._weighted_average(filtered_signals)
        elif self._method == EnsembleMethod.CONFIDENCE_THRESHOLD:
            return self._confidence_threshold_vote(filtered_signals)
        elif self._method == EnsembleMethod.RANK_AGGREGATION:
            return self._rank_aggregation(filtered_signals)
        elif self._method == EnsembleMethod.BORDA_COUNT:
            return self._borda_count(filtered_signals)
        elif self._method == EnsembleMethod.HYBRID:
            return self._hybrid_method(filtered_signals)

        return {"decision": "no_signal", "confidence": 0.0, "reason": "Unknown method"}

    def _validate_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid = []
        for signal in signals:
            if all(k in signal for k in ["source", "symbol", "direction", "confidence"]):
                if signal["source"] in self._sources:
                    self._sources[signal["source"]].touch()
                    valid.append(signal)
                else:
                    logger.warning(f"Unknown signal source: {signal['source']}")
            else:
                logger.warning(f"Invalid signal format: {signal}")
        return valid

    def _majority_vote(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        votes = {}
        for signal in signals:
            direction = signal["direction"].lower()
            source = signal["source"]
            source_weight = self._sources[source].get_effective_weight()

            if direction not in votes:
                votes[direction] = 0
            votes[direction] += source_weight

        if not votes:
            return {"decision": "no_signal", "confidence": 0.0, "reason": "No votes",
                    "consensus_strength": 0.0}

        max_votes = max(votes.values())
        winners = [d for d, v in votes.items() if v == max_votes]
        consensus = self._compute_consensus_strength(signals)

        if len(winners) == 1:
            total_votes = sum(votes.values())
            confidence = max_votes / total_votes if total_votes > 0 else 0.0
            return {
                "decision": winners[0],
                "confidence": confidence,
                "reason": f"Majority vote: {winners[0]} ({max_votes:.2f} votes)",
                "votes": votes,
                "consensus_strength": consensus,
            }
        else:
            return {"decision": "conflict", "confidence": 0.0,
                    "reason": f"Vote tie: {winners}", "consensus_strength": consensus}

    def _weighted_average(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0

        for signal in signals:
            direction = signal["direction"].lower()
            confidence = signal.get("confidence", 0.5)
            source = signal["source"]
            weight = self._sources[source].get_effective_weight()

            if direction in ("buy", "long"):
                buy_score += confidence * weight
            elif direction in ("sell", "short"):
                sell_score += confidence * weight

            total_weight += weight

        if total_weight == 0:
            return {"decision": "no_signal", "confidence": 0.0, "reason": "No weights",
                    "consensus_strength": 0.0}

        buy_score /= total_weight
        sell_score /= total_weight

        if buy_score > sell_score:
            confidence = buy_score
            decision = "buy" if buy_score - sell_score > 0.1 else "hold"
        elif sell_score > buy_score:
            confidence = sell_score
            decision = "sell" if sell_score - buy_score > 0.1 else "hold"
        else:
            confidence = (buy_score + sell_score) / 2
            decision = "hold"

        return {
            "decision": decision,
            "confidence": confidence,
            "reason": f"Weighted average: buy={buy_score:.3f}, sell={sell_score:.3f}",
            "scores": {"buy": buy_score, "sell": sell_score},
            "consensus_strength": self._compute_consensus_strength(signals),
        }

    def _confidence_threshold_vote(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        high_confidence_signals = [
            s for s in signals if s.get("confidence", 0) >= self._confidence_threshold
        ]

        if not high_confidence_signals:
            return {"decision": "no_signal", "confidence": 0.0,
                    "reason": "No high confidence signals", "consensus_strength": 0.0}

        buy_count = sum(1 for s in high_confidence_signals if s["direction"].lower() in ("buy", "long"))
        sell_count = sum(1 for s in high_confidence_signals if s["direction"].lower() in ("sell", "short"))
        consensus = self._compute_consensus_strength(high_confidence_signals)

        # P0: 避免空generator导致statistics.mean异常
        buy_confidences = [s["confidence"] for s in high_confidence_signals if s["direction"].lower() in ("buy", "long")]
        sell_confidences = [s["confidence"] for s in high_confidence_signals if s["direction"].lower() in ("sell", "short")]

        if buy_count >= self._min_votes and buy_count > sell_count:
            return {
                "decision": "buy",
                "confidence": statistics.mean(buy_confidences) if buy_confidences else 0.0,
                "reason": f"{buy_count} high confidence buy signals",
                "consensus_strength": consensus,
            }
        elif sell_count >= self._min_votes and sell_count > buy_count:
            return {
                "decision": "sell",
                "confidence": statistics.mean(sell_confidences) if sell_confidences else 0.0,
                "reason": f"{sell_count} high confidence sell signals",
                "consensus_strength": consensus,
            }
        else:
            return {"decision": "hold", "confidence": 0.0,
                    "reason": "Insufficient high confidence signals", "consensus_strength": consensus}

    def _rank_aggregation(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        ranked_signals = sorted(signals, key=lambda s: s.get("confidence", 0), reverse=True)

        buy_rank_sum = 0
        sell_rank_sum = 0
        buy_count = 0
        sell_count = 0

        for idx, signal in enumerate(ranked_signals):
            rank = idx + 1
            direction = signal["direction"].lower()
            source_weight = self._sources[signal["source"]].get_effective_weight()

            if direction in ("buy", "long"):
                buy_rank_sum += rank * source_weight
                buy_count += 1
            elif direction in ("sell", "short"):
                sell_rank_sum += rank * source_weight
                sell_count += 1

        consensus = self._compute_consensus_strength(signals)

        if buy_count == 0 and sell_count == 0:
            return {"decision": "no_signal", "confidence": 0.0, "reason": "No signals",
                    "consensus_strength": consensus}

        if buy_count > 0 and (sell_count == 0 or buy_rank_sum < sell_rank_sum):
            buy_confs = [s["confidence"] for s in ranked_signals if s["direction"].lower() in ("buy", "long")]
            return {
                "decision": "buy",
                "confidence": statistics.mean(buy_confs) if buy_confs else 0.0,
                "reason": f"Rank aggregation: buy rank={buy_rank_sum:.2f}, sell rank={sell_rank_sum:.2f}",
                "consensus_strength": consensus,
            }
        elif sell_count > 0 and (buy_count == 0 or sell_rank_sum < buy_rank_sum):
            sell_confs = [s["confidence"] for s in ranked_signals if s["direction"].lower() in ("sell", "short")]
            return {
                "decision": "sell",
                "confidence": statistics.mean(sell_confs) if sell_confs else 0.0,
                "reason": f"Rank aggregation: sell rank={sell_rank_sum:.2f}, buy rank={buy_rank_sum:.2f}",
                "consensus_strength": consensus,
            }
        else:
            return {"decision": "hold", "confidence": 0.0, "reason": "Rank tie",
                    "consensus_strength": consensus}

    async def update_reliability(self, source_name: str, was_correct: bool, direction: str = ""):
        if source_name in self._sources:
            self._sources[source_name].update_reliability(was_correct, direction)
            logger.info(f"Updated reliability for {source_name}: {self._sources[source_name].reliability:.3f}")

    # ═══════════════════════════════════════════════════════
    # 信号多样性度量
    # ═══════════════════════════════════════════════════════

    def _compute_direction_vector(self, source_name: str, window: int = 20) -> List[float]:
        """将信号源的方向历史转换为数值向量（buy=1, sell=-1, 其他=0）"""
        source = self._sources.get(source_name)
        if not source or not source._direction_history:
            return []
        recent = list(source._direction_history)[-window:]
        vec = []
        for d in recent:
            d_lower = d.lower()
            if d_lower in ("buy", "long"):
                vec.append(1.0)
            elif d_lower in ("sell", "short"):
                vec.append(-1.0)
            else:
                vec.append(0.0)
        return vec

    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """计算两个等长向量的Pearson相关系数"""
        n = len(x)
        if n < 3:
            return 0.0
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)
        denom = (var_x * var_y) ** 0.5
        if denom < 1e-10:
            return 0.0
        return cov / denom

    def compute_signal_diversity(self, window: int = 20) -> Dict[str, Any]:
        """
        信号多样性度量

        Returns:
            diversity_score: 0=完全一致, 1=完全多样（高分表示信号源独立）
            pairwise_correlations: 两两相关系数矩阵
            redundant_pairs: 高度冗余的信号源对 (corr > 0.7)
            entropy: 方向分布的香农熵
            effective_sources: 去冗余后的有效信号源数量
        """
        if len(self._sources) < 2:
            return {
                "diversity_score": 0.0 if len(self._sources) == 0 else 1.0,
                "pairwise_correlations": {},
                "redundant_pairs": [],
                "entropy": 0.0,
                "effective_sources": len(self._sources),
            }

        # 提取各信号源方向向量
        source_names = list(self._sources.keys())
        direction_vectors: Dict[str, List[float]] = {}
        for name in source_names:
            vec = self._compute_direction_vector(name, window)
            if vec:
                direction_vectors[name] = vec

        if len(direction_vectors) < 2:
            return {
                "diversity_score": 0.5,
                "pairwise_correlations": {},
                "redundant_pairs": [],
                "entropy": 0.0,
                "effective_sources": len(direction_vectors),
            }

        # 两两相关系数
        pairwise_correlations: Dict[str, float] = {}
        redundant_pairs: List[Dict[str, Any]] = []
        corr_values = []

        active_names = list(direction_vectors.keys())
        for i in range(len(active_names)):
            for j in range(i + 1, len(active_names)):
                a, b = active_names[i], active_names[j]
                # 对齐向量长度
                min_len = min(len(direction_vectors[a]), len(direction_vectors[b]))
                if min_len < 3:
                    continue
                va = direction_vectors[a][-min_len:]
                vb = direction_vectors[b][-min_len:]
                corr = self._pearson_correlation(va, vb)
                key = f"{a}:{b}"
                pairwise_correlations[key] = round(corr, 4)
                corr_values.append(abs(corr))
                if abs(corr) > 0.7:
                    redundant_pairs.append({
                        "source_a": a, "source_b": b,
                        "correlation": round(corr, 4),
                        "severity": "high" if abs(corr) > 0.85 else "medium",
                    })

        # 多样性分数：1 - avg(|correlation|)
        if corr_values:
            avg_abs_corr = sum(corr_values) / len(corr_values)
            diversity_score = round(1.0 - avg_abs_corr, 4)
        else:
            diversity_score = 0.5

        # 方向分布熵
        all_dirs: List[float] = []
        for vec in direction_vectors.values():
            all_dirs.extend(vec)
        buy_count = sum(1 for v in all_dirs if v > 0.5)
        sell_count = sum(1 for v in all_dirs if v < -0.5)
        hold_count = len(all_dirs) - buy_count - sell_count
        total = max(len(all_dirs), 1)
        probs = [c / total for c in (buy_count, sell_count, hold_count) if c > 0]
        if probs:
            entropy = -sum(p * math.log2(p) for p in probs)
        else:
            entropy = 0.0

        # 有效信号源数：去冗余后的独立信号源估算
        # N_eff = N / (1 + (N-1)*avg_corr)  (基于平均相关系数的等效独立样本数)
        n = len(active_names)
        if corr_values:
            avg_corr = sum(abs(c) for c in corr_values) / len(corr_values)
            effective = n / (1 + (n - 1) * avg_corr)
        else:
            effective = float(n)
        effective_sources = round(effective, 2)

        return {
            "diversity_score": diversity_score,
            "pairwise_correlations": pairwise_correlations,
            "redundant_pairs": redundant_pairs,
            "entropy": round(entropy, 4),
            "effective_sources": effective_sources,
            "total_sources": n,
        }

    def apply_diversity_penalty(
        self, source_name: str, penalty_map: Optional[Dict[str, float]] = None
    ) -> float:
        """
        对冗余信号源施加权重惩罚

        当 detect_redundant 返回 true 时，自动降低冗余源的权重。
        返回惩罚因子 (0~1)，惩罚越重因子越小。
        """
        diversity = self.compute_signal_diversity()
        if not diversity["redundant_pairs"]:
            return 1.0

        penalty = penalty_map or {}
        min_factor = 1.0
        for pair in diversity["redundant_pairs"]:
            if source_name in (pair["source_a"], pair["source_b"]):
                corr = abs(pair["correlation"])
                # 冗余惩罚：corr=0.7→factor=0.8, corr=0.9→factor=0.5
                factor = max(0.3, 1.0 - (corr - 0.6) * 1.5)
                other = pair["source_b"] if source_name == pair["source_a"] else pair["source_a"]
                custom = penalty.get(other, 1.0)
                factor = min(factor, custom)
                min_factor = min(min_factor, factor)

        return min_factor

    def _compute_consensus_strength(self, signals: List[Dict[str, Any]]) -> float:
        buy_count = sum(1 for s in signals if s["direction"].lower() in ("buy", "long"))
        sell_count = sum(1 for s in signals if s["direction"].lower() in ("sell", "short"))
        total = len(signals)
        if total == 0:
            return 0.0
        return abs(buy_count - sell_count) / total

    def _borda_count(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        ranked = sorted(signals, key=lambda s: s.get("confidence", 0), reverse=True)
        n = len(ranked)
        buy_points = 0.0
        sell_points = 0.0

        for idx, signal in enumerate(ranked):
            points = n - idx
            direction = signal["direction"].lower()
            source_weight = self._sources[signal["source"]].get_effective_weight()

            if direction in ("buy", "long"):
                buy_points += points * source_weight
            elif direction in ("sell", "short"):
                sell_points += points * source_weight

        consensus = self._compute_consensus_strength(signals)

        if buy_points == 0.0 and sell_points == 0.0:
            return {"decision": "no_signal", "confidence": 0.0, "reason": "No signals",
                    "consensus_strength": consensus}

        total_points = buy_points + sell_points
        if buy_points > sell_points:
            confidence = buy_points / total_points if total_points > 0 else 0.0
            return {
                "decision": "buy",
                "confidence": confidence,
                "reason": f"Borda count: buy={buy_points:.2f}, sell={sell_points:.2f}",
                "borda_points": {"buy": round(buy_points, 2), "sell": round(sell_points, 2)},
                "consensus_strength": consensus,
            }
        elif sell_points > buy_points:
            confidence = sell_points / total_points if total_points > 0 else 0.0
            return {
                "decision": "sell",
                "confidence": confidence,
                "reason": f"Borda count: sell={sell_points:.2f}, buy={buy_points:.2f}",
                "borda_points": {"buy": round(buy_points, 2), "sell": round(sell_points, 2)},
                "consensus_strength": consensus,
            }
        else:
            return {"decision": "hold", "confidence": 0.0, "reason": "Borda count tie",
                    "consensus_strength": consensus}

    def _hybrid_method(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        surviving = [
            s for s in signals if s.get("confidence", 0) >= self._confidence_threshold
        ]

        if not surviving:
            return self._weighted_average(signals)

        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0

        for signal in surviving:
            direction = signal["direction"].lower()
            confidence = signal.get("confidence", 0.5)
            source = signal["source"]
            weight = self._sources[source].get_effective_weight()

            if direction in ("buy", "long"):
                buy_score += confidence * weight
            elif direction in ("sell", "short"):
                sell_score += confidence * weight
            total_weight += weight

        if total_weight == 0:
            return {"decision": "no_signal", "confidence": 0.0,
                    "reason": "No surviving signals", "consensus_strength": 0.0}

        buy_score /= total_weight
        sell_score /= total_weight

        if buy_score > sell_score:
            decision = "buy" if buy_score - sell_score > 0.1 else "hold"
            confidence = buy_score
        elif sell_score > buy_score:
            decision = "sell" if sell_score - buy_score > 0.1 else "hold"
            confidence = sell_score
        else:
            decision = "hold"
            confidence = (buy_score + sell_score) / 2

        return {
            "decision": decision,
            "confidence": confidence,
            "reason": f"Hybrid: buy={buy_score:.3f}, sell={sell_score:.3f} (threshold={self._confidence_threshold})",
            "scores": {"buy": buy_score, "sell": sell_score},
            "consensus_strength": self._compute_consensus_strength(surviving),
        }

    def record_outcome(self, decision_result: Dict[str, Any], was_correct: bool, pnl_impact: float = 0.0):
        entry = {
            "timestamp": time.time(),
            "decision": decision_result.get("decision", "unknown"),
            "confidence": decision_result.get("confidence", 0.0),
            "was_correct": was_correct,
            "pnl_impact": pnl_impact,
        }
        self._performance_history.append(entry)
        logger.info(f"Recorded outcome: decision={entry['decision']}, correct={was_correct}, pnl={pnl_impact}")

    def get_recent_performance(self, minutes: int = 60) -> Dict[str, Any]:
        cutoff = time.time() - minutes * 60
        recent = [e for e in self._performance_history if e["timestamp"] >= cutoff]
        total = len(recent)
        if total == 0:
            return {"win_rate": 0.0, "avg_confidence": 0.0, "decision_count": 0}
        correct = sum(1 for e in recent if e["was_correct"])
        avg_confidence = statistics.mean(e["confidence"] for e in recent)
        return {
            "win_rate": correct / total,
            "avg_confidence": round(avg_confidence, 4),
            "decision_count": total,
        }

    def get_source_stats(self) -> Dict[str, Any]:
        stats = {}
        for name, source in self._sources.items():
            stats[name] = {
                "weight": source.weight,
                "reliability": source.reliability,
                "kalman_uncertainty": round(source.get_kalman_uncertainty(), 6),
                "effective_weight": source.get_effective_weight(),
                "signal_count": source.signal_count,
                "correct_count": source.correct_count,
            }
        return stats

    def get_stats(self) -> Dict[str, Any]:
        return {
            "method": self._method.value,
            "confidence_threshold": self._confidence_threshold,
            "min_votes": self._min_votes,
            "num_sources": len(self._sources),
            "sources": self.get_source_stats(),
            "diversity": self.compute_signal_diversity(),
        }
