"""策略相关性分析模块 (Strategy Correlation Analyzer)

提供策略收益序列之间的多维度相关性分析：
1. Pearson / Spearman / Kendall Tau 相关性
2. 滚动窗口相关性与 EWMA 加权相关性
3. 尾部依赖分析（基于经验 Copula 的上下尾依赖系数）
4. 层次聚类与聚类稳定性评估
5. 有效 N（分散化度量）
6. 市场状态（regime）感知的相关性追踪

仅依赖 Python 标准库 + numpy + loguru + asyncio，所有统计量手工实现。
"""
import asyncio
import math
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


# ==============================================================================
# 内部辅助函数：纯统计计算，不依赖外部 ML/SciPy
# ==============================================================================

def _mean(values: np.ndarray) -> float:
    """数组均值"""
    if len(values) == 0:
        return 0.0
    return float(np.sum(values) / len(values))


def _std(values: np.ndarray, ddof: int = 0) -> float:
    """数组标准差"""
    n = len(values)
    if n <= ddof:
        return 0.0
    m = _mean(values)
    var = float(np.sum((values - m) ** 2) / (n - ddof))
    return math.sqrt(max(var, 0.0))


def _rank(arr: np.ndarray) -> np.ndarray:
    """对数组元素进行排名（平均值法处理平局）"""
    n = len(arr)
    if n == 0:
        return np.array([], dtype=float)
    sorted_indices = np.argsort(arr)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[sorted_indices[j + 1]] == arr[sorted_indices[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based average
        for k in range(i, j + 1):
            ranks[sorted_indices[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson 相关系数（手工实现）"""
    n = len(x)
    if n < 3:
        return 0.0
    mx, my = _mean(x), _mean(y)
    sx, sy = _std(x, ddof=0), _std(y, ddof=0)
    if sx == 0.0 or sy == 0.0:
        return 0.0
    cov = float(np.sum((x - mx) * (y - my)) / n)
    r = cov / (sx * sy)
    return max(-1.0, min(1.0, r))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman 等级相关系数（手工实现：Pearson on ranks）"""
    return _pearson(_rank(x), _rank(y))


def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall's tau-b 相关系数（手工实现）"""
    n = len(x)
    if n < 2:
        return 0.0

    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0

    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            if dx == 0:
                ties_x += 1
            if dy == 0:
                ties_y += 1
            if dx > 0 and dy > 0:
                concordant += 1
            elif dx < 0 and dy < 0:
                concordant += 1
            elif dx > 0 and dy < 0:
                discordant += 1
            elif dx < 0 and dy > 0:
                discordant += 1
            # dx==0 or dy==0 → neither concordant nor discordant

    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


def _ema_weighted_covariance(x: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[float, float, float, float]:
    """计算指数加权下的协方差和方差

    Returns: (ewm_cov, ewm_var_x, ewm_var_y, sum_weights)
    """
    n = len(x)
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0

    # 从最后一个元素开始（近期权重高）
    weights = np.array([(1 - alpha) ** i for i in range(n)][::-1], dtype=float)
    w_sum = float(np.sum(weights))
    if w_sum == 0:
        return 0.0, 0.0, 0.0, 0.0

    wx = np.sum(weights * x) / w_sum
    wy = np.sum(weights * y) / w_sum

    ewm_cov = float(np.sum(weights * (x - wx) * (y - wy)) / w_sum)
    ewm_var_x = float(np.sum(weights * (x - wx) ** 2) / w_sum)
    ewm_var_y = float(np.sum(weights * (y - wy) ** 2) / w_sum)

    return ewm_cov, ewm_var_x, ewm_var_y, w_sum


def _quantile(values: np.ndarray, q: float) -> float:
    """计算分位数（手工实现）"""
    n = len(values)
    if n == 0:
        return 0.0
    sorted_vals = np.sort(values)
    idx = q * (n - 1)
    low = int(idx)
    high = min(low + 1, n - 1)
    frac = idx - low
    return float(sorted_vals[low] * (1 - frac) + sorted_vals[high] * frac)


def _upper_triangle_mean(matrix: np.ndarray) -> float:
    """计算相关系数矩阵上三角的均值（不含对角线）"""
    n = matrix.shape[0]
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            val = matrix[i, j]
            if not np.isnan(val):
                total += val
                count += 1
    return total / count if count > 0 else 0.0


def _upper_triangle_max(matrix: np.ndarray) -> float:
    """计算相关系数矩阵上三角的最大值（不含对角线）"""
    n = matrix.shape[0]
    if n < 2:
        return 0.0
    max_val = -1.0
    for i in range(n):
        for j in range(i + 1, n):
            val = matrix[i, j]
            if not np.isnan(val) and val > max_val:
                max_val = val
    return max_val if max_val > -1.0 else 0.0


def _pairwise_matrix(matrix: np.ndarray, names: List[str]) -> Dict[str, float]:
    """将矩阵上三角转为 pairwise 字典 {(name_i, name_j): correlation}"""
    n = matrix.shape[0]
    result = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = f"{names[i]}:{names[j]}"
            result[key] = float(matrix[i, j])
    return result


def _significance_flag(corr: float, n: int) -> str:
    """基于样本量判断相关性的显著程度"""
    abs_r = abs(corr)
    if n < 10:
        return "uncertain"
    # 使用近似 t-test: t = r * sqrt((n-2)/(1-r^2))，转换为定性标签
    if abs_r < 0.1:
        return "negligible"
    elif abs_r < 0.3:
        return "weak"
    elif abs_r < 0.5:
        return "moderate"
    elif abs_r < 0.7:
        return "strong"
    elif abs_r < 0.85:
        return "very_strong"
    else:
        return "near_perfect"


# ==============================================================================
# 数据类
# ==============================================================================

@dataclass
class CorrelationResult:
    """相关性分析完整结果"""
    pearson: float = 0.0
    spearman: float = 0.0
    kendall_tau: float = 0.0
    rolling_correlation: List[float] = field(default_factory=list)
    correlation_matrix: Dict[str, float] = field(default_factory=dict)
    average_correlation: float = 0.0
    max_pair_correlation: float = 0.0
    correlation_regime: str = "low"  # low / medium / high / extreme
    tail_dependence_lower: float = 0.0
    tail_dependence_upper: float = 0.0
    effective_n: float = 0.0
    regime_correlations: Dict[str, float] = field(default_factory=dict)
    pairwise: Dict[str, Dict[str, float]] = field(default_factory=dict)
    clusters: Dict[str, Any] = field(default_factory=dict)
    diversification_erosion: bool = False
    timestamp: float = 0.0


# ==============================================================================
# RollingCorrelationAnalyzer
# ==============================================================================

class RollingCorrelationAnalyzer:
    """滚动窗口相关性分析器

    支持 Pearson / Spearman / Kendall Tau 三种相关系数，
    以及指数加权移动平均（EWMA）相关性和市场状态感知追踪。
    """

    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("strategy_correlation", {})
        self._rolling_window: int = int(cfg.get("rolling_window", 30))
        self._ewma_half_life: int = int(cfg.get("ewma_half_life", 20))
        self._regime_shift_threshold: float = float(cfg.get("regime_shift_threshold", 0.15))
        self._min_observations: int = int(cfg.get("min_observations", 10))

        # EWMA alpha: alpha = 1 - exp(-ln(2) / half_life)
        self._ewma_alpha = 1.0 - math.exp(-math.log(2) / max(self._ewma_half_life, 1))

        # 市场状态追踪
        self._regime_correlation_cache: Dict[str, Dict[str, float]] = {}
        self._previous_matrix_cache: Optional[np.ndarray] = None
        self._previous_names: List[str] = []

    def compute_pairwise(self, returns_x: List[float], returns_y: List[float]) -> float:
        """计算两个收益序列之间的 Pearson 滚动相关系数"""
        arr_x = np.array(returns_x, dtype=float)
        arr_y = np.array(returns_y, dtype=float)
        min_len = min(len(arr_x), len(arr_y))
        if min_len < self._min_observations:
            return 0.0
        window = min(self._rolling_window, min_len)
        arr_x = arr_x[-window:]
        arr_y = arr_y[-window:]
        return _pearson(arr_x, arr_y)

    def compute_spearman(self, returns_x: List[float], returns_y: List[float]) -> float:
        """计算两个收益序列之间的 Spearman 等级相关系数"""
        arr_x = np.array(returns_x, dtype=float)
        arr_y = np.array(returns_y, dtype=float)
        min_len = min(len(arr_x), len(arr_y))
        if min_len < self._min_observations:
            return 0.0
        window = min(self._rolling_window, min_len)
        return _spearman(arr_x[-window:], arr_y[-window:])

    def compute_kendall(self, returns_x: List[float], returns_y: List[float]) -> float:
        """计算两个收益序列之间的 Kendall tau 相关系数"""
        arr_x = np.array(returns_x, dtype=float)
        arr_y = np.array(returns_y, dtype=float)
        min_len = min(len(arr_x), len(arr_y))
        if min_len < self._min_observations:
            return 0.0
        window = min(self._rolling_window, min_len)
        return _kendall_tau(arr_x[-window:], arr_y[-window:])

    def compute_correlation_matrix(
        self, strategy_returns: Dict[str, List[float]]
    ) -> Tuple[np.ndarray, List[str], Dict[str, str]]:
        """计算完整的 N×N 相关系数矩阵，附带显著性标记

        Returns:
            matrix: (N, N) ndarray
            names: strategy names in order
            significance: {(name_i, name_j): significance_flag}
        """
        names = sorted(strategy_returns.keys())
        n = len(names)
        if n == 0:
            return np.array([[]]), [], {}

        matrix = np.eye(n, dtype=float)
        sig_flags: Dict[str, str] = {}

        for i in range(n):
            for j in range(i + 1, n):
                arr_i = np.array(strategy_returns[names[i]], dtype=float)
                arr_j = np.array(strategy_returns[names[j]], dtype=float)
                min_len = min(len(arr_i), len(arr_j))
                if min_len < self._min_observations:
                    r = 0.0
                else:
                    w = min(self._rolling_window, min_len)
                    r = _pearson(arr_i[-w:], arr_j[-w:])
                    if np.isnan(r):
                        r = 0.0
                matrix[i, j] = r
                matrix[j, i] = r
                sig_flags[f"{names[i]}:{names[j]}"] = _significance_flag(r, min_len)

        return matrix, names, sig_flags

    def compute_ewma_correlation(
        self, returns_x: List[float], returns_y: List[float]
    ) -> float:
        """计算指数加权移动平均相关系数（近期数据权重更高）"""
        arr_x = np.array(returns_x, dtype=float)
        arr_y = np.array(returns_y, dtype=float)
        min_len = min(len(arr_x), len(arr_y))
        if min_len < self._min_observations:
            return 0.0
        win = min(self._rolling_window, min_len)
        ewm_cov, ewm_vx, ewm_vy, _ = _ema_weighted_covariance(
            arr_x[-win:], arr_y[-win:], self._ewma_alpha
        )
        denom = math.sqrt(max(ewm_vx * ewm_vy, 0.0))
        if denom == 0:
            return 0.0
        r = ewm_cov / denom
        return max(-1.0, min(1.0, r))

    def compute_rolling_correlations(
        self, returns_x: List[float], returns_y: List[float]
    ) -> List[float]:
        """计算随时间变化的滚动相关性序列

        每一步利用过去 rolling_window 个观察值计算相关系数，生成序列。
        """
        arr_x = np.array(returns_x, dtype=float)
        arr_y = np.array(returns_y, dtype=float)
        n = min(len(arr_x), len(arr_y))
        if n < self._rolling_window + 1:
            return []
        result = []
        for end in range(self._rolling_window, n + 1):
            win_x = arr_x[end - self._rolling_window : end]
            win_y = arr_y[end - self._rolling_window : end]
            r = _pearson(win_x, win_y)
            result.append(float(r))
        return result

    def detect_regime_shifts(
        self, current_matrix: np.ndarray, current_names: List[str]
    ) -> bool:
        """检测相关性结构是否发生了显著变化

        比较当前矩阵与上一次缓存的矩阵之间的 Frobenius 范数差。
        """
        if self._previous_matrix_cache is None or self._previous_names != current_names:
            self._previous_matrix_cache = current_matrix.copy()
            self._previous_names = list(current_names)
            return False

        if current_matrix.shape != self._previous_matrix_cache.shape:
            self._previous_matrix_cache = current_matrix.copy()
            self._previous_names = list(current_names)
            return False

        diff = float(np.sqrt(np.sum((current_matrix - self._previous_matrix_cache) ** 2)))
        n = current_matrix.shape[0]
        if n == 0:
            return False
        # Normalize by theoretical max Frobenius diff
        max_diff = math.sqrt(n * n * 4.0)  # each element ∈ [-1,1], diff max = 2
        normalized_diff = diff / max_diff if max_diff > 0 else 0.0

        self._previous_matrix_cache = current_matrix.copy()
        self._previous_names = list(current_names)
        return normalized_diff > self._regime_shift_threshold

    def track_regime_correlation(
        self, market_regime: str, correlation_matrix: np.ndarray, names: List[str]
    ) -> Dict[str, float]:
        """维护每个市场状态下的独立相关性估计

        使用简单指数平滑更新该 regime 对应的 pairwise 相关性。
        """
        if market_regime not in self._regime_correlation_cache:
            self._regime_correlation_cache[market_regime] = {}

        cached = self._regime_correlation_cache[market_regime]
        n = correlation_matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                key = f"{names[i]}:{names[j]}"
                new_val = float(correlation_matrix[i, j])
                if key in cached:
                    cached[key] = 0.7 * cached[key] + 0.3 * new_val
                else:
                    cached[key] = new_val

        return dict(cached)

    def get_regime_correlations(self, market_regime: str) -> Dict[str, float]:
        """获取指定市场状态下的缓存相关性"""
        return self._regime_correlation_cache.get(market_regime, {})


# ==============================================================================
# TailDependenceAnalyzer
# ==============================================================================

class TailDependenceAnalyzer:
    """尾部依赖分析器

    基于经验 Copula 方法，计算上下尾依赖系数。
    尾部依赖衡量两个策略在极端收益（好或坏）时共同出现的概率。
    """

    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("strategy_correlation", {})
        self._tail_percentile: float = float(cfg.get("tail_percentile", 0.10))
        self._min_observations: int = int(cfg.get("min_observations", 10))

    def compute_lower_tail_dependence(
        self, returns_x: List[float], returns_y: List[float]
    ) -> float:
        """计算下尾依赖系数: P(X < q | Y < q)

        使用经验分位数，取底部 tail_percentile 的观测。
        """
        arr_x = np.array(returns_x, dtype=float)
        arr_y = np.array(returns_y, dtype=float)
        n = len(arr_x)
        if n < self._min_observations:
            return 0.0

        qx = _quantile(arr_x, self._tail_percentile)
        qy = _quantile(arr_y, self._tail_percentile)

        mask_y = arr_y < qy
        n_y_tail = int(np.sum(mask_y))
        if n_y_tail == 0:
            return 0.0

        mask_both = (arr_x < qx) & mask_y
        n_both = int(np.sum(mask_both))
        return n_both / n_y_tail

    def compute_upper_tail_dependence(
        self, returns_x: List[float], returns_y: List[float]
    ) -> float:
        """计算上尾依赖系数: P(X > q | Y > q)"""
        arr_x = np.array(returns_x, dtype=float)
        arr_y = np.array(returns_y, dtype=float)
        n = len(arr_x)
        if n < self._min_observations:
            return 0.0

        qx = _quantile(arr_x, 1.0 - self._tail_percentile)
        qy = _quantile(arr_y, 1.0 - self._tail_percentile)

        mask_y = arr_y > qy
        n_y_tail = int(np.sum(mask_y))
        if n_y_tail == 0:
            return 0.0

        mask_both = (arr_x > qx) & mask_y
        n_both = int(np.sum(mask_both))
        return n_both / n_y_tail

    def tail_risk_index(
        self, strategy_returns: Dict[str, List[float]]
    ) -> float:
        """计算组合尾部风险指数（0-1）

        对每一对策略，取上下尾依赖较大的值，再取所有 pair 的均值。
        """
        names = list(strategy_returns.keys())
        n = len(names)
        if n < 2:
            return 0.0

        tail_vals = []
        for i in range(n):
            for j in range(i + 1, n):
                rx = strategy_returns[names[i]]
                ry = strategy_returns[names[j]]
                lt = self.compute_lower_tail_dependence(rx, ry)
                ut = self.compute_upper_tail_dependence(rx, ry)
                tail_vals.append(max(lt, ut))

        if not tail_vals:
            return 0.0
        avg = sum(tail_vals) / len(tail_vals)
        # Scale: expected tail dependence under independence ≈ tail_percentile
        # Normalize so that independence → 0, high dependence → 1
        base = self._tail_percentile
        if base >= 1.0:
            return float(avg)
        normalized = max(0.0, min(1.0, (avg - base) / (1.0 - base)))
        return normalized


# ==============================================================================
# ClusterAnalyzer
# ==============================================================================

class ClusterAnalyzer:
    """层次聚类分析器

    基于相关系数距离矩阵，使用自底向上的层次聚类（average-linkage），
    支持按聚类数量或距离阈值切分。
    """

    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("strategy_correlation", {})
        self._rolling_window: int = int(cfg.get("rolling_window", 30))
        self._min_observations: int = int(cfg.get("min_observations", 10))
        # 历次聚类的 label 记录，用于稳定性评估
        self._cluster_history: List[Dict[str, int]] = []

    def _distance_matrix(self, correlation_matrix: np.ndarray) -> np.ndarray:
        """将相关系数矩阵转换为距离矩阵: d = 1 - |r|"""
        return 1.0 - np.abs(correlation_matrix)

    def _hierarchical_cluster(
        self, dist_matrix: np.ndarray
    ) -> Tuple[List[Tuple[int, int, float]], np.ndarray]:
        """自底向上的层次聚类（average-linkage）

        Returns:
            merge_steps: [(i, j, distance), ...] 合并步骤
            linkage_matrix: 供后续切分使用的 linkage 矩阵
        """
        n = dist_matrix.shape[0]
        if n <= 1:
            return [], np.array([])

        # 初始每个元素自成一簇
        clusters = [{i} for i in range(n)]
        dist = dist_matrix.copy()
        np.fill_diagonal(dist, np.inf)

        merge_steps: List[Tuple[int, int, float]] = []

        # 记录 linkage 矩阵: [cluster_i, cluster_j, distance, size]
        linkage_rows = []
        next_cluster_id = n

        for _ in range(n - 1):
            # 找到最小距离对
            min_dist = np.inf
            min_i, min_j = -1, -1
            for a in range(len(clusters)):
                for b in range(len(clusters)):
                    if a >= b:
                        continue
                    # 计算平均距离
                    total_d = 0.0
                    cnt = 0
                    for ia in clusters[a]:
                        for ib in clusters[b]:
                            total_d += dist_matrix[ia, ib]
                            cnt += 1
                    avg_d = total_d / cnt if cnt > 0 else np.inf
                    if avg_d < min_dist:
                        min_dist = avg_d
                        min_i, min_j = a, b

            if min_i == -1:
                break

            merge_steps.append((min_i, min_j, min_dist))
            linkage_rows.append([
                float(min_i if min_i < n else min_i),
                float(min_j if min_j < n else min_j),
                min_dist,
                float(len(clusters[min_i]) + len(clusters[min_j])),
            ])

            # 合并
            new_cluster = clusters[min_i] | clusters[min_j]
            # 删除 min_j 和 min_i（先删大的索引避免偏移）
            del clusters[max(min_i, min_j)]
            del clusters[min(min_i, min_j)]
            clusters.append(new_cluster)

        linkage_matrix = np.array(linkage_rows) if linkage_rows else np.array([])
        return merge_steps, linkage_matrix

    def cluster_by_count(
        self, correlation_matrix: np.ndarray, names: List[str], n_clusters: int
    ) -> Dict[str, Any]:
        """按指定聚类数量进行层次聚类"""
        n = correlation_matrix.shape[0]
        if n == 0:
            return {"clusters": [], "assignments": {}, "stability": 0.0}

        if n <= n_clusters:
            assignments = {names[i]: i for i in range(n)}
            clusters = {i: [names[i]] for i in range(n)}
            return {
                "clusters": [
                    {"id": i, "members": clusters[i], "size": len(clusters[i])}
                    for i in range(n)
                ],
                "assignments": assignments,
                "stability": 0.0,
                "intra_cluster_corr": {},
                "concentration_risk": False,
            }

        dist = self._distance_matrix(correlation_matrix)
        merge_steps, _ = self._hierarchical_cluster(dist)

        # 从 merge_steps 重建聚类结构
        # 初始每个元素一簇
        cluster_members = {i: {i} for i in range(n)}
        active_clusters = set(range(n))
        next_id = n

        for (a, b, d) in merge_steps:
            new_members = cluster_members[a] | cluster_members[b]
            cluster_members[next_id] = new_members
            active_clusters.discard(a)
            active_clusters.discard(b)
            active_clusters.add(next_id)
            next_id += 1

            if len(active_clusters) == n_clusters:
                break

        # 构建结果
        result_clusters = []
        assignments = {}
        for cid, idx in enumerate(sorted(active_clusters)):
            members = [names[i] for i in sorted(cluster_members[idx])]
            result_clusters.append({"id": cid, "members": members, "size": len(members)})
            for name in members:
                assignments[name] = cid

        # 计算簇内平均相关性
        intra = {}
        for c in result_clusters:
            members = c["members"]
            if len(members) < 2:
                intra[c["id"]] = 0.0
                continue
            vals = []
            for mi in range(len(members)):
                for mj in range(mi + 1, len(members)):
                    ii = names.index(members[mi])
                    jj = names.index(members[mj])
                    vals.append(float(correlation_matrix[ii, jj]))
            intra[c["id"]] = sum(vals) / len(vals) if vals else 0.0

        # 集中度风险：检查是否有某簇占比过大
        total_size = sum(c["size"] for c in result_clusters)
        max_cluster_ratio = max(c["size"] / total_size for c in result_clusters) if total_size > 0 else 0
        concentration_risk = max_cluster_ratio > 0.6

        # 记录聚类结果用于稳定性评估
        label_map = {names[i]: assignments.get(names[i], -1) for i in range(n)}
        self._cluster_history.append(label_map)
        if len(self._cluster_history) > 20:
            self._cluster_history = self._cluster_history[-20:]

        stability = self._compute_cluster_stability()

        return {
            "clusters": result_clusters,
            "assignments": assignments,
            "stability": stability,
            "intra_cluster_corr": intra,
            "concentration_risk": concentration_risk,
        }

    def cluster_by_threshold(
        self, correlation_matrix: np.ndarray, names: List[str], distance_threshold: float
    ) -> Dict[str, Any]:
        """按距离阈值进行层次聚类切分"""
        n = correlation_matrix.shape[0]
        if n <= 1:
            return {
                "clusters": [{"id": 0, "members": names, "size": len(names)}],
                "assignments": {name: 0 for name in names},
                "stability": 0.0,
                "intra_cluster_corr": {0: 0.0},
                "concentration_risk": False,
            }

        dist = self._distance_matrix(correlation_matrix)
        merge_steps, _ = self._hierarchical_cluster(dist)

        cluster_members = {i: {i} for i in range(n)}
        active_clusters = set(range(n))
        next_id = n

        for (a, b, d) in merge_steps:
            if d > distance_threshold:
                break
            new_members = cluster_members[a] | cluster_members[b]
            cluster_members[next_id] = new_members
            active_clusters.discard(a)
            active_clusters.discard(b)
            active_clusters.add(next_id)
            next_id += 1

        result_clusters = []
        assignments = {}
        for cid, idx in enumerate(sorted(active_clusters)):
            members = [names[i] for i in sorted(cluster_members[idx])]
            result_clusters.append({"id": cid, "members": members, "size": len(members)})
            for name in members:
                assignments[name] = cid

        intra = {}
        for c in result_clusters:
            members = c["members"]
            if len(members) < 2:
                intra[c["id"]] = 0.0
                continue
            vals = []
            for mi in range(len(members)):
                for mj in range(mi + 1, len(members)):
                    ii = names.index(members[mi])
                    jj = names.index(members[mj])
                    vals.append(float(correlation_matrix[ii, jj]))
            intra[c["id"]] = sum(vals) / len(vals) if vals else 0.0

        total_size = sum(c["size"] for c in result_clusters)
        max_cluster_ratio = max(c["size"] / total_size for c in result_clusters) if total_size > 0 else 0
        concentration_risk = max_cluster_ratio > 0.6

        return {
            "clusters": result_clusters,
            "assignments": assignments,
            "stability": 0.0,
            "intra_cluster_corr": intra,
            "concentration_risk": concentration_risk,
        }

    def _compute_cluster_stability(self) -> float:
        """计算聚类稳定性：比较最近两次聚类结果的一致性"""
        if len(self._cluster_history) < 2:
            return 1.0

        prev = self._cluster_history[-2]
        curr = self._cluster_history[-1]

        # Rand index simplified: count agreement
        names_prev = list(prev.keys())
        names_curr = list(curr.keys())

        # 找共同名称
        common = set(names_prev) & set(names_curr)
        if len(common) < 2:
            return 0.0

        agreements = 0
        total_pairs = 0
        common_list = list(common)
        for i in range(len(common_list)):
            for j in range(i + 1, len(common_list)):
                total_pairs += 1
                same_prev = prev[common_list[i]] == prev[common_list[j]]
                same_curr = curr[common_list[i]] == curr[common_list[j]]
                if same_prev == same_curr:
                    agreements += 1

        return agreements / total_pairs if total_pairs > 0 else 1.0


# ==============================================================================
# EffectiveN
# ==============================================================================

class EffectiveN:
    """分散化有效 N 计算器

    effective_n = (∑ w_i)² / (∑ w_i² + ∑_{i≠j} w_i·w_j·ρ_{ij})

    范围: 1（完全相关）到 N（完全独立）。
    """

    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("strategy_correlation", {})
        self._min_observations: int = int(cfg.get("min_observations", 10))
        self._history: List[float] = []

    def compute(
        self,
        correlation_matrix: np.ndarray,
        weights: Optional[List[float]] = None,
    ) -> float:
        """从相关系数矩阵计算有效 N（等权或自定义权重）"""
        n = correlation_matrix.shape[0]
        if n == 0:
            return 0.0

        if weights is None:
            weights = [1.0 / n] * n
        else:
            total_w = sum(weights)
            if total_w != 0:
                weights = [w / total_w for w in weights]

        # Denominator: sum(w_i^2) + sum_{i!=j} w_i * w_j * corr_{ij}
        sum_w_sq = sum(w ** 2 for w in weights)
        cross_term = 0.0
        for i in range(n):
            for j in range(n):
                if i != j:
                    cross_term += weights[i] * weights[j] * correlation_matrix[i, j]

        denom = sum_w_sq + cross_term
        if denom <= 0:
            return 1.0

        sum_w = sum(weights)
        effective_n = (sum_w ** 2) / denom

        # Clamp to [1, n]
        effective_n = max(1.0, min(float(n), effective_n))

        self._history.append(effective_n)
        if len(self._history) > 1000:
            self._history = self._history[-1000:]

        return effective_n

    def get_trend(self, lookback: int = 10) -> Dict[str, Any]:
        """获取有效 N 的变化趋势"""
        if len(self._history) < 2:
            return {
                "current": self._history[-1] if self._history else 0.0,
                "trend": "stable",
                "erosion_speed": 0.0,
            }

        recent = self._history[-lookback:] if len(self._history) >= lookback else self._history
        current = recent[-1]
        # 简单线性趋势
        xs = list(range(len(recent)))
        if len(recent) < 2:
            return {"current": current, "trend": "stable", "erosion_speed": 0.0}

        slope = _pearson(np.array(xs, dtype=float), np.array(recent, dtype=float))

        # Normalize slope as percentage change
        avg_val = _mean(np.array(recent))
        if avg_val > 0:
            erosion_speed = slope / avg_val
        else:
            erosion_speed = 0.0

        if erosion_speed < -0.02:
            trend = "eroding"
        elif erosion_speed > 0.02:
            trend = "improving"
        else:
            trend = "stable"

        return {
            "current": current,
            "trend": trend,
            "erosion_speed": float(erosion_speed),
        }


# ==============================================================================
# StrategyCorrelationAnalyzer (主类)
# ==============================================================================

class StrategyCorrelationAnalyzer:
    """策略相关性综合分析器

    整合所有分析维度：
    - RollingCorrelationAnalyzer: 滚动相关性与矩阵
    - TailDependenceAnalyzer: 尾部依赖
    - ClusterAnalyzer: 层次聚类
    - EffectiveN: 分散化度量
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        cfg = config.get("strategy_correlation", {})

        self._rolling_window: int = int(cfg.get("rolling_window", 30))
        self._high_corr_threshold: float = float(cfg.get("high_correlation_threshold", 0.7))
        self._extreme_corr_threshold: float = float(cfg.get("extreme_correlation_threshold", 0.85))
        self._min_observations: int = int(cfg.get("min_observations", 10))

        # 子分析器
        self._rolling_analyzer = RollingCorrelationAnalyzer(config)
        self._tail_analyzer = TailDependenceAnalyzer(config)
        self._cluster_analyzer = ClusterAnalyzer(config)
        self._effective_n_calc = EffectiveN(config)

        # 状态
        self._is_running = False
        self._lock = asyncio.Lock()
        self._strategy_returns: Dict[str, List[float]] = {}
        self._last_result: Optional[CorrelationResult] = None
        self._market_regime: str = "unknown"

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """启动分析器"""
        async with self._lock:
            if self._is_running:
                logger.warning("StrategyCorrelationAnalyzer already running")
                return
            self._is_running = True
            logger.info(
                f"StrategyCorrelationAnalyzer started: rolling_window={self._rolling_window}, "
                f"high_corr_threshold={self._high_corr_threshold}, "
                f"extreme_corr_threshold={self._extreme_corr_threshold}"
            )

    async def stop(self) -> None:
        """停止分析器"""
        async with self._lock:
            if not self._is_running:
                return
            self._is_running = False
            logger.info("StrategyCorrelationAnalyzer stopped")

    # ------------------------------------------------------------------
    # 数据输入
    # ------------------------------------------------------------------

    async def update_returns(
        self, strategy_returns: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """更新策略收益数据并触发关联分析

        Args:
            strategy_returns: {strategy_name: [return_1, return_2, ...]}

        Returns:
            简要的结果摘要
        """
        async with self._lock:
            # 合并数据
            for name, rets in strategy_returns.items():
                if name not in self._strategy_returns:
                    self._strategy_returns[name] = []
                self._strategy_returns[name].extend(rets)
                # 限制历史长度
                max_len = self._rolling_window * 5
                if len(self._strategy_returns[name]) > max_len:
                    self._strategy_returns[name] = self._strategy_returns[name][-max_len:]

        result = await self.analyze()
        return {
            "average_correlation": result.get("average_correlation", 0.0),
            "correlation_regime": result.get("correlation_regime", "low"),
            "effective_n": result.get("effective_n", 0.0),
            "tail_risk_index": result.get("tail_risk_index", 0.0),
        }

    # ------------------------------------------------------------------
    # 核心分析
    # ------------------------------------------------------------------

    async def analyze(self) -> Dict[str, Any]:
        """执行完整的多维度相关性分析"""
        async with self._lock:
            data = dict(self._strategy_returns)
            if not data:
                return self._empty_result()

            # 过滤数据不足的策略
            valid_data = {
                k: v for k, v in data.items() if len(v) >= self._min_observations
            }
            if len(valid_data) < 2:
                return self._empty_result_with_names(list(valid_data.keys()))

            names = sorted(valid_data.keys())
            n = len(names)

            # 1. 相关性矩阵
            matrix, mat_names, sig_flags = self._rolling_analyzer.compute_correlation_matrix(
                valid_data
            )
            avg_corr = _upper_triangle_mean(matrix)
            max_pair = _upper_triangle_max(matrix)
            pairwise_dict = _pairwise_matrix(matrix, mat_names)

            # 2. 滚动相关性（取第一对策略的序列作为代表）
            rolling_corr: List[float] = []
            if n >= 2:
                rolling_corr = self._rolling_analyzer.compute_rolling_correlations(
                    valid_data[names[0]], valid_data[names[1]]
                )

            # 3. 全套 pairwise 指标
            pairwise_detail: Dict[str, Dict[str, float]] = {}
            for i in range(n):
                for j in range(i + 1, n):
                    rx = valid_data[names[i]]
                    ry = valid_data[names[j]]
                    key = f"{names[i]}:{names[j]}"
                    pairwise_detail[key] = {
                        "pearson": self._rolling_analyzer.compute_pairwise(rx, ry),
                        "spearman": self._rolling_analyzer.compute_spearman(rx, ry),
                        "kendall": self._rolling_analyzer.compute_kendall(rx, ry),
                        "ewma_corr": self._rolling_analyzer.compute_ewma_correlation(rx, ry),
                        "tail_lower": self._tail_analyzer.compute_lower_tail_dependence(rx, ry),
                        "tail_upper": self._tail_analyzer.compute_upper_tail_dependence(rx, ry),
                        "significance": sig_flags.get(key, "uncertain"),
                    }

            # 4. 尾部依赖
            tail_dep = {}
            for i in range(n):
                for j in range(i + 1, n):
                    rx = valid_data[names[i]]
                    ry = valid_data[names[j]]
                    key = f"{names[i]}:{names[j]}"
                    lt = self._tail_analyzer.compute_lower_tail_dependence(rx, ry)
                    ut = self._tail_analyzer.compute_upper_tail_dependence(rx, ry)
                    tail_dep[key] = {"lower": lt, "upper": ut}
            tail_idx = self._tail_analyzer.tail_risk_index(valid_data)

            # 5. 聚类
            clusters = self._cluster_analyzer.cluster_by_count(
                matrix, mat_names, min(3, n)
            )

            # 6. 有效 N
            eff_n = self._effective_n_calc.compute(matrix)
            eff_n_trend = self._effective_n_calc.get_trend()

            # 7. 相关性 regime 判断
            regime = self._classify_correlation_regime(avg_corr, max_pair)

            # 8. 市场状态感知相关性
            regime_corr = self._rolling_analyzer.track_regime_correlation(
                self._market_regime, matrix, mat_names
            )

            # 9. 检测 regime shift
            regime_shifted = self._rolling_analyzer.detect_regime_shifts(matrix, mat_names)

            result = CorrelationResult(
                pearson=avg_corr,  # average pairwise
                spearman=0.0,  # computed per-pair in detail
                kendall_tau=0.0,
                rolling_correlation=rolling_corr,
                correlation_matrix=pairwise_dict,
                average_correlation=avg_corr,
                max_pair_correlation=max_pair,
                correlation_regime=regime,
                tail_dependence_lower=0.0,
                tail_dependence_upper=0.0,
                effective_n=eff_n,
                regime_correlations=regime_corr,
                pairwise=pairwise_detail,
                clusters=clusters,
                diversification_erosion=(eff_n_trend.get("trend", "stable") == "eroding"),
                timestamp=time.time(),
            )

            self._last_result = result

            return {
                "timestamp": result.timestamp,
                "n_strategies": n,
                "strategy_names": names,
                "average_correlation": result.average_correlation,
                "max_pair_correlation": result.max_pair_correlation,
                "correlation_regime": result.correlation_regime,
                "effective_n": result.effective_n,
                "effective_n_trend": eff_n_trend,
                "tail_risk_index": tail_idx,
                "correlation_matrix": pairwise_dict,
                "significance_flags": sig_flags,
                "pairwise": pairwise_detail,
                "tail_dependence": tail_dep,
                "clusters": clusters,
                "regime_correlations": regime_corr,
                "regime_shift_detected": regime_shifted,
                "market_regime": self._market_regime,
                "diversification_erosion": result.diversification_erosion,
                "rolling_correlation_sample": rolling_corr[-10:] if rolling_corr else [],
            }

    def _classify_correlation_regime(self, avg_corr: float, max_pair: float) -> str:
        """根据平均相关性和最大 pair 相关性判断 regime"""
        if avg_corr >= self._extreme_corr_threshold or max_pair >= 0.95:
            return "extreme"
        elif avg_corr >= self._high_corr_threshold:
            return "high"
        elif avg_corr >= 0.3:
            return "medium"
        else:
            return "low"

    async def check_correlation_risk(self, threshold: float = 0.7) -> Dict[str, Any]:
        """检查是否存在相关性风险

        Returns:
            包含风险告警信息的字典
        """
        result = await self.analyze()

        alerts = []
        risk_level = "low"

        avg_corr = result.get("average_correlation", 0.0)
        max_corr = result.get("max_pair_correlation", 0.0)
        regime = result.get("correlation_regime", "low")
        eff_n = result.get("effective_n", 0.0)
        eff_n_trend = result.get("effective_n_trend", {})
        tail_idx = result.get("tail_risk_index", 0.0)
        clusters = result.get("clusters", {})

        if regime in ("extreme",):
            alerts.append({
                "type": "extreme_correlation",
                "severity": "critical",
                "message": f"策略间存在极端相关性 (avg={avg_corr:.3f}, max={max_corr:.3f})",
            })
            risk_level = "critical"
        elif regime == "high" and avg_corr >= threshold:
            alerts.append({
                "type": "high_correlation",
                "severity": "high",
                "message": f"策略间存在高相关性 (avg={avg_corr:.3f})",
            })
            risk_level = "high"
        elif avg_corr >= threshold * 0.8:
            alerts.append({
                "type": "elevated_correlation",
                "severity": "medium",
                "message": f"策略间相关性偏高 (avg={avg_corr:.3f})",
            })
            risk_level = "medium"

        # 分散化侵蚀
        if eff_n_trend.get("trend") == "eroding":
            alerts.append({
                "type": "diversification_erosion",
                "severity": "medium",
                "message": f"分散化效应正在削弱 (effective_n={eff_n:.2f})",
            })
            if risk_level == "low":
                risk_level = "medium"

        # 尾部风险
        if tail_idx > 0.5:
            alerts.append({
                "type": "tail_risk",
                "severity": "high",
                "message": f"尾部依赖风险偏高 (tail_risk_index={tail_idx:.3f})",
            })
            if risk_level in ("low", "medium"):
                risk_level = "high"

        # 集中度风险
        if clusters.get("concentration_risk", False):
            alerts.append({
                "type": "cluster_concentration",
                "severity": "high",
                "message": "聚类集中度偏高，存在系统性风险",
            })
            if risk_level in ("low", "medium"):
                risk_level = "high"

        # regime shift
        if result.get("regime_shift_detected", False):
            alerts.append({
                "type": "regime_shift",
                "severity": "medium",
                "message": "相关性结构发生显著变化",
            })

        return {
            "risk_level": risk_level,
            "alerts": alerts,
            "alert_count": len(alerts),
            "threshold_used": threshold,
            "summary": {
                "average_correlation": avg_corr,
                "max_pair_correlation": max_corr,
                "correlation_regime": regime,
                "effective_n": eff_n,
                "tail_risk_index": tail_idx,
            },
        }

    async def get_pairwise_correlation(self, s1: str, s2: str) -> Dict[str, Any]:
        """获取两个策略间的详细 pairwise 相关性"""
        async with self._lock:
            data1 = self._strategy_returns.get(s1, [])
            data2 = self._strategy_returns.get(s2, [])

            if len(data1) < self._min_observations or len(data2) < self._min_observations:
                return {"error": "insufficient_data", "s1_len": len(data1), "s2_len": len(data2)}

        pearson = self._rolling_analyzer.compute_pairwise(data1, data2)
        spearman = self._rolling_analyzer.compute_spearman(data1, data2)
        kendall = self._rolling_analyzer.compute_kendall(data1, data2)
        ewma_c = self._rolling_analyzer.compute_ewma_correlation(data1, data2)
        rolling = self._rolling_analyzer.compute_rolling_correlations(data1, data2)

        lt = self._tail_analyzer.compute_lower_tail_dependence(data1, data2)
        ut = self._tail_analyzer.compute_upper_tail_dependence(data1, data2)

        return {
            "strategy_a": s1,
            "strategy_b": s2,
            "n_observations": min(len(data1), len(data2)),
            "pearson": pearson,
            "spearman": spearman,
            "kendall_tau": kendall,
            "ewma_correlation": ewma_c,
            "rolling_correlations": rolling[-20:] if rolling else [],
            "tail_dependence_lower": lt,
            "tail_dependence_upper": ut,
            "significance": _significance_flag(pearson, min(len(data1), len(data2))),
        }

    async def get_tail_dependence(self) -> Dict[str, Any]:
        """获取所有策略对的尾部依赖分析"""
        async with self._lock:
            data = dict(self._strategy_returns)

        valid_data = {k: v for k, v in data.items() if len(v) >= self._min_observations}
        names = sorted(valid_data.keys())
        n = len(names)

        if n < 2:
            return {"pairs": {}, "tail_risk_index": 0.0}

        tail_idx = self._tail_analyzer.tail_risk_index(valid_data)
        pairs = {}
        for i in range(n):
            for j in range(i + 1, n):
                rx = valid_data[names[i]]
                ry = valid_data[names[j]]
                key = f"{names[i]}:{names[j]}"
                pairs[key] = {
                    "lower": self._tail_analyzer.compute_lower_tail_dependence(rx, ry),
                    "upper": self._tail_analyzer.compute_upper_tail_dependence(rx, ry),
                }

        return {
            "pairs": pairs,
            "tail_risk_index": tail_idx,
            "n_strategies": n,
        }

    async def get_clusters(self, n_clusters: int = 3) -> Dict[str, Any]:
        """获取层次聚类结果"""
        async with self._lock:
            data = dict(self._strategy_returns)

        valid_data = {k: v for k, v in data.items() if len(v) >= self._min_observations}
        if len(valid_data) < 2:
            return {"clusters": [], "assignments": {}, "stability": 0.0}

        matrix, mat_names, _ = self._rolling_analyzer.compute_correlation_matrix(valid_data)
        return self._cluster_analyzer.cluster_by_count(matrix, mat_names, n_clusters)

    async def get_effective_n(self) -> Dict[str, Any]:
        """获取当前有效 N 及其趋势"""
        async with self._lock:
            data = dict(self._strategy_returns)

        valid_data = {k: v for k, v in data.items() if len(v) >= self._min_observations}
        if len(valid_data) < 2:
            return {"effective_n": len(valid_data), "trend": "stable", "erosion_speed": 0.0}

        matrix, _, _ = self._rolling_analyzer.compute_correlation_matrix(valid_data)
        eff_n = self._effective_n_calc.compute(matrix)
        trend = self._effective_n_calc.get_trend()

        return {
            "effective_n": eff_n,
            "n_strategies": len(valid_data),
            "n_strategies_ideal": len(valid_data),
            "diversification_ratio": eff_n / max(len(valid_data), 1),
            "trend": trend["trend"],
            "erosion_speed": trend["erosion_speed"],
        }

    def get_summary(self) -> Dict[str, Any]:
        """获取汇总快照（同步，不触发完整重算）"""
        if self._last_result is None:
            return {"status": "no_data", "message": "尚未执行分析"}

        return {
            "status": "ready",
            "is_running": self._is_running,
            "timestamp": self._last_result.timestamp,
            "n_strategies": len(self._strategy_returns),
            "average_correlation": self._last_result.average_correlation,
            "max_pair_correlation": self._last_result.max_pair_correlation,
            "correlation_regime": self._last_result.correlation_regime,
            "effective_n": self._last_result.effective_n,
            "diversification_erosion": self._last_result.diversification_erosion,
            "market_regime": self._market_regime,
        }

    # ------------------------------------------------------------------
    # 状态管理
    # ------------------------------------------------------------------

    async def set_market_regime(self, regime: str) -> None:
        """设置当前市场状态"""
        async with self._lock:
            if self._market_regime != regime:
                logger.info(f"Market regime changed: {self._market_regime} -> {regime}")
            self._market_regime = regime

    def get_market_regime(self) -> str:
        """获取当前市场状态"""
        return self._market_regime

    async def clear_history(self) -> None:
        """清除历史数据"""
        async with self._lock:
            self._strategy_returns.clear()
            self._last_result = None
            logger.info("StrategyCorrelationAnalyzer history cleared")

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "timestamp": time.time(),
            "n_strategies": 0,
            "strategy_names": [],
            "average_correlation": 0.0,
            "max_pair_correlation": 0.0,
            "correlation_regime": "low",
            "effective_n": 0.0,
            "effective_n_trend": {"current": 0.0, "trend": "stable", "erosion_speed": 0.0},
            "tail_risk_index": 0.0,
            "correlation_matrix": {},
            "significance_flags": {},
            "pairwise": {},
            "tail_dependence": {},
            "clusters": {},
            "regime_correlations": {},
            "regime_shift_detected": False,
            "market_regime": self._market_regime,
            "diversification_erosion": False,
            "rolling_correlation_sample": [],
        }

    def _empty_result_with_names(self, names: List[str]) -> Dict[str, Any]:
        r = self._empty_result()
        r["n_strategies"] = len(names)
        r["strategy_names"] = names
        r["effective_n"] = float(len(names)) if names else 0.0
        return r
