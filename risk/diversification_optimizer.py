"""
分散化优化器（Diversification Optimizer）

多策略资金配置的数学优化引擎，提供：
  - 均值-方差优化（最大Sharpe / 最小方差）
  - 有效前沿计算（梯度下降 + 投影法）
  - 最大分散比优化（迭代加权算法）
  - 风险平价 / 等风险贡献优化（CCD坐标下降）
  - 最小CVaR优化（Rockafellar-Uryasev + 内联单纯形法）
  - 分散化指标分析（HHI / 有效N / 相关系数熵）
  - 再平衡建议生成

纯Python实现，不依赖scipy；所有优化算法手动实现。
"""
import asyncio
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from loguru import logger


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """将任意输入安全转换为有限浮点数；None/NaN/Inf/非法值回退 default。"""
    try:
        if value is None:
            return default
        v = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return v


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

@dataclass
class OptimizationResult:
    """优化结果"""
    weights: Dict[str, float] = field(default_factory=dict)
    expected_return: float = 0.0
    expected_risk: float = 0.0
    sharpe_ratio: float = 0.0
    diversification_ratio: float = 0.0
    effective_n: float = 0.0
    concentration_ratio: float = 0.0
    optimization_method: str = ""
    constraints_satisfied: List[str] = field(default_factory=list)
    risk_contributions: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": {k: round(v, 6) for k, v in self.weights.items()},
            "expected_return": round(self.expected_return, 6),
            "expected_risk": round(self.expected_risk, 6),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "diversification_ratio": round(self.diversification_ratio, 4),
            "effective_n": round(self.effective_n, 2),
            "concentration_ratio": round(self.concentration_ratio, 4),
            "optimization_method": self.optimization_method,
            "constraints_satisfied": self.constraints_satisfied,
            "risk_contributions": {k: round(v, 4) for k, v in self.risk_contributions.items()},
            "timestamp": self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════
# 线性代数工具函数（不依赖 scipy）
# ═══════════════════════════════════════════════════════════════

def _safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """安全除法：None/NaN/Inf/除零一律返回 default。"""
    a = _safe_float(a)
    b = _safe_float(b)
    if a is None or b is None or abs(b) < 1e-14:
        return default
    result = a / b
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def _project_onto_simplex(v: np.ndarray, max_w: float = 1.0, min_w: float = 0.0) -> np.ndarray:
    """将向量投影到单纯形上: sum(w) = 1, min_w <= w_i <= max_w"""
    n = len(v)
    if n == 0:
        return np.array([], dtype=float)
    v = np.asarray(v, dtype=float)
    if not np.all(np.isfinite(v)):
        return np.ones(n) / n
    # 先裁剪到 [min_w, max_w]
    w = np.clip(v, min_w, max_w)
    s = float(np.sum(w))
    if abs(s) < 1e-14:
        # 退化为均匀分布
        return np.ones(n) / n
    # 通过均匀缩放调整 sum = 1
    # 迭代修正以满足边界约束
    for _ in range(20):
        excess = s - 1.0
        if abs(excess) < 1e-10:
            break
        # 均匀分配到每个分量
        delta = excess / n
        w = w - delta
        w = np.clip(w, min_w, max_w)
        s = float(np.sum(w))
    # 最终归一化
    if abs(s) > 1e-14:
        w = w / s
    return w


def _compute_covariance_matrix(returns_array: np.ndarray) -> np.ndarray:
    """从收益率矩阵计算协方差矩阵"""
    # returns_array: (n_assets, n_periods)
    n_assets, n_periods = returns_array.shape
    if n_periods < 2:
        return np.eye(n_assets) * 0.01
    mean_rets = np.mean(returns_array, axis=1, keepdims=True)
    centered = returns_array - mean_rets
    cov = (centered @ centered.T) / (n_periods - 1)
    # 确保对称正定
    cov = (cov + cov.T) / 2.0
    # 添加小对角线扰动确保正定
    eig_vals = np.linalg.eigvalsh(cov)
    min_eig = float(np.min(eig_vals))
    if min_eig < 1e-10:
        cov = cov + np.eye(n_assets) * (abs(min_eig) + 1e-8)
    return cov


def _compute_correlation_matrix(cov: np.ndarray) -> np.ndarray:
    """从协方差矩阵计算相关系数矩阵"""
    n = cov.shape[0]
    std_devs = np.sqrt(np.diag(cov))
    corr = np.zeros_like(cov)
    for i in range(n):
        for j in range(n):
            denom = std_devs[i] * std_devs[j]
            if denom > 1e-14:
                corr[i, j] = cov[i, j] / denom
            else:
                corr[i, j] = 1.0 if i == j else 0.0
    return np.clip(corr, -1.0, 1.0)


# ═══════════════════════════════════════════════════════════════
# DiversificationMetrics — 分散化指标计算
# ═══════════════════════════════════════════════════════════════

class DiversificationMetrics:
    """分散化指标计算器

    提供：
      - 分散化比率 (diversification ratio)
      - 有效持仓数 (effective N)
      - HHI 集中度 (Herfindahl-Hirschman Index)
      - 相关系数熵 (correlation entropy)
      - 风险集中度 (risk concentration)
      - 时间序列追踪
    """

    def __init__(self):
        self._history: List[Dict[str, Any]] = []
        self._max_history = 500

    def compute(self, weights: Dict[str, float],
                cov_matrix: np.ndarray,
                strategy_names: List[str]) -> Dict[str, Any]:
        """计算所有分散化指标"""
        n = len(strategy_names)

        w = np.zeros(n)
        for i, name in enumerate(strategy_names):
            w[i] = weights.get(name, 0.0)

        # ── 组合波动率 ──
        port_var = float(w @ cov_matrix @ w)
        port_vol = math.sqrt(max(port_var, 1e-14))

        # ── 分散化比率: D = sum(w_i * σ_i) / σ_portfolio ──
        indiv_vols = np.sqrt(np.diag(cov_matrix))
        weighted_sum_vols = float(np.sum(w * indiv_vols))
        div_ratio = _safe_divide(weighted_sum_vols, port_vol, 1.0)

        # ── 有效N: 1 / sum(w_i^2) ──
        hhi = float(np.sum(w ** 2))
        effective_n = _safe_divide(1.0, hhi, 1.0) if hhi > 0 else float(n)

        # ── HHI 集中度 ──
        concentration_ratio = hhi

        # ── 相关系数熵: -sum(λ_i * log(λ_i)) ──
        corr_matrix = _compute_correlation_matrix(cov_matrix)
        eig_vals = np.linalg.eigvalsh(corr_matrix)
        # 归一化特征值
        eig_sum = float(np.sum(np.abs(eig_vals)))
        if eig_sum > 1e-14:
            eig_norm = np.abs(eig_vals) / eig_sum
        else:
            eig_norm = np.ones(n) / n
        corr_entropy = 0.0
        for lam in eig_norm:
            if lam > 1e-14:
                corr_entropy -= lam * math.log(lam)
        # 归一化到 [0, 1]
        max_entropy = math.log(float(n)) if n > 1 else 1.0
        normalized_entropy = corr_entropy / max_entropy if max_entropy > 0 else 1.0

        # ── 风险贡献 ──
        mrc = cov_matrix @ w  # 边际风险贡献
        rc = w * mrc
        rc_pct = rc / max(port_vol, 1e-14)

        # ── 风险集中度: max(RC) / avg(RC) ──
        avg_rc = float(np.mean(np.abs(rc_pct)))
        max_rc = float(np.max(np.abs(rc_pct)))
        risk_concentration = _safe_divide(max_rc, avg_rc, 1.0)

        result = {
            "diversification_ratio": round(div_ratio, 4),
            "effective_n": round(effective_n, 2),
            "hhi": round(hhi, 6),
            "concentration_ratio": round(concentration_ratio, 4),
            "correlation_entropy": round(corr_entropy, 4),
            "normalized_entropy": round(normalized_entropy, 4),
            "risk_concentration": round(risk_concentration, 4),
            "risk_contributions": {
                strategy_names[i]: round(float(rc_pct[i]), 4)
                for i in range(n)
            },
            "timestamp": datetime.now().isoformat(),
        }

        self._history.append(result)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return result

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self._history[-limit:]

    def get_current(self) -> Optional[Dict[str, Any]]:
        return self._history[-1] if self._history else None


# ═══════════════════════════════════════════════════════════════
# EfficientFrontier — 均值-方差优化
# ═══════════════════════════════════════════════════════════════

class EfficientFrontier:
    """有效前沿计算器

    使用投影梯度下降实现：
      - max_sharpe: 最大化 (w'r - rf) / sqrt(w'Σw)
      - min_variance: 最小化 w'Σw
      - 有效前沿追踪
    """

    def __init__(self, risk_free_rate: float = 0.02, max_iterations: int = 200,
                 convergence_tol: float = 1e-6):
        self._rf = risk_free_rate
        self._max_iter = max_iterations
        self._tol = convergence_tol

    def compute_optimal_weights(
        self,
        expected_returns: np.ndarray,      # (n,)
        cov_matrix: np.ndarray,            # (n, n)
        objective: str = "max_sharpe",
        constraints: Optional[Dict] = None,
        target_return: Optional[float] = None,
    ) -> Tuple[np.ndarray, float, float, float]:
        """计算最优权重

        Args:
            expected_returns: 各策略期望收益率
            cov_matrix: 协方差矩阵
            objective: "max_sharpe" 或 "min_variance"
            constraints: {max_single_weight, min_single_weight}
            target_return: 目标收益率（可选，用于最小方差时约束）

        Returns:
            (optimal_weights, portfolio_return, portfolio_risk, sharpe_ratio)
        """
        n = len(expected_returns)
        constraints = constraints or {}
        max_single = constraints.get("max_single_weight", 0.30)
        min_single = constraints.get("min_single_weight", 0.05)

        # 确保 min_single * n <= 1
        if min_single * n > 1.0:
            min_single = max(0.0, 1.0 / n - 0.01)

        # 初始权重：等权
        w = np.ones(n) / n

        learning_rate = 0.01
        best_w = w.copy()
        best_objective = -np.inf if objective == "max_sharpe" else np.inf
        prev_w = w.copy()

        for iteration in range(self._max_iter):
            if objective == "max_sharpe":
                # 目标: max (w'r - rf) / sqrt(w'Σw)
                port_ret = float(w @ expected_returns)
                port_var = float(w @ cov_matrix @ w)
                port_vol = math.sqrt(max(port_var, 1e-12))
                excess_ret = port_ret - self._rf

                # 梯度: ∇f = r / σ - (excess_ret) * Σw / σ³
                grad_ret = expected_returns / port_vol
                grad_risk = (excess_ret) * (cov_matrix @ w) / (port_vol ** 3)
                gradient = grad_ret - grad_risk

                # 上升方向（最大化）
                w_new = w + learning_rate * gradient

                # 计算新目标
                new_port_ret = float(w_new @ expected_returns)
                new_port_var = float(w_new @ cov_matrix @ w_new)
                new_port_vol = math.sqrt(max(new_port_var, 1e-12))
                new_objective = _safe_divide(new_port_ret - self._rf, new_port_vol, -np.inf)

                if new_objective > best_objective:
                    best_objective = new_objective
                    best_w = w_new.copy()

            elif objective == "min_variance":
                # 目标: min w'Σw
                port_var = float(w @ cov_matrix @ w)

                # 梯度: ∇f = 2 Σw
                gradient = 2.0 * (cov_matrix @ w)

                # 如果有目标收益率约束，添加惩罚项
                if target_return is not None:
                    port_ret = float(w @ expected_returns)
                    ret_deficit = target_return - port_ret
                    if ret_deficit > 0:
                        # 惩罚不足的收益率
                        penalty_grad = -2.0 * ret_deficit * expected_returns
                        gradient = gradient + 0.5 * penalty_grad

                # 下降方向（最小化）
                w_new = w - learning_rate * gradient

                # 计算新目标
                new_var = float(w_new @ cov_matrix @ w_new)
                new_objective = new_var
                if target_return is not None:
                    new_ret = float(w_new @ expected_returns)
                    ret_deficit_new = target_return - new_ret
                    if ret_deficit_new > 0:
                        new_objective += 0.5 * ret_deficit_new ** 2

                if new_objective < best_objective:
                    best_objective = new_objective
                    best_w = w_new.copy()

                # 自适应学习率
                if iteration > 0 and new_objective >= best_objective:
                    learning_rate *= 0.5
                else:
                    learning_rate = min(learning_rate * 1.01, 0.1)

            # 投影到可行域
            best_w = _project_onto_simplex(best_w, max_single, min_single)

            # 收敛检查
            change = float(np.max(np.abs(best_w - prev_w)))
            prev_w = best_w.copy()
            w = best_w.copy()

            if change < self._tol:
                break

        # 确保满足约束
        w_final = _project_onto_simplex(best_w, max_single, min_single)

        port_ret = float(w_final @ expected_returns)
        port_var = float(w_final @ cov_matrix @ w_final)
        port_vol = math.sqrt(max(port_var, 1e-12))
        sharpe = _safe_divide(port_ret - self._rf, port_vol, 0.0)

        return w_final, port_ret, port_vol, sharpe

    def trace_efficient_frontier(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        n_points: int = 20,
        constraints: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """计算有效前沿上的点"""
        n = len(expected_returns)

        # 先找到最小方差组合和最大收益组合
        w_minvar, ret_minvar, risk_minvar, _ = self.compute_optimal_weights(
            expected_returns, cov_matrix, "min_variance", constraints
        )

        max_ret = float(np.max(expected_returns))
        # 最大收益组合就是全仓最高收益策略（受约束限制）
        constraints = constraints or {}
        max_single = constraints.get("max_single_weight", 0.30)
        best_idx = int(np.argmax(expected_returns))
        w_maxret = np.zeros(n)
        w_maxret[best_idx] = max_single
        remaining = 1.0 - max_single
        min_w = constraints.get("min_single_weight", 0.05)
        other_indices = [i for i in range(n) if i != best_idx]
        if other_indices:
            per_other = remaining / len(other_indices)
            for i in other_indices:
                w_maxret[i] = max(min_w, per_other)
        w_maxret = _project_onto_simplex(w_maxret, max_single, min_w)
        ret_maxret = float(w_maxret @ expected_returns)

        # 在 [ret_minvar, ret_maxret] 之间均匀采样
        frontier = []
        target_returns = np.linspace(ret_minvar, ret_maxret, n_points)

        for target in target_returns:
            w_opt, port_ret, port_vol, sharpe = self.compute_optimal_weights(
                expected_returns, cov_matrix, "min_variance",
                constraints, target_return=float(target)
            )
            frontier.append({
                "return": round(float(port_ret), 6),
                "risk": round(float(port_vol), 6),
                "sharpe": round(float(sharpe), 4),
            })

        return frontier


# ═══════════════════════════════════════════════════════════════
# MaxDiversificationOptimizer — 最大分散比优化
# ═══════════════════════════════════════════════════════════════

class MaxDiversificationOptimizer:
    """最大分散比优化器

    目标: max D(w) = sum(w_i * σ_i) / sqrt(w'Σw)

    使用迭代加权算法：
      w_i^(k+1) ∝ σ_i / (Σ w^(k))_i
    即权重与波动率成正比，与边际风险成反比。
    """

    def __init__(self, max_iterations: int = 200, convergence_tol: float = 1e-6):
        self._max_iter = max_iterations
        self._tol = convergence_tol
        self._objective_history: List[float] = []

    def optimize(
        self,
        cov_matrix: np.ndarray,
        constraints: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, float]:
        """优化最大分散比

        Returns:
            (optimal_weights, diversification_ratio)
        """
        n = cov_matrix.shape[0]
        constraints = constraints or {}
        max_single = constraints.get("max_single_weight", 0.30)
        min_single = constraints.get("min_single_weight", 0.05)

        indiv_vols = np.sqrt(np.diag(cov_matrix))

        # 初始权重：1/σ 归一化
        inv_vols = 1.0 / np.maximum(indiv_vols, 1e-8)
        w = inv_vols / np.sum(inv_vols)

        self._objective_history = []
        best_w = w.copy()
        best_div = -np.inf
        prev_w = w.copy()

        for iteration in range(self._max_iter):
            # 计算当前组合指标
            port_var = float(w @ cov_matrix @ w)
            port_vol = math.sqrt(max(port_var, 1e-14))
            weighted_vol = float(np.sum(w * indiv_vols))
            div_ratio = _safe_divide(weighted_vol, port_vol, 0.0)
            self._objective_history.append(div_ratio)

            if div_ratio > best_div:
                best_div = div_ratio
                best_w = w.copy()

            # 计算边际风险贡献
            mrc = cov_matrix @ w

            # 迭代加权: w_i ∝ σ_i / MRC_i
            new_w = np.zeros(n)
            for i in range(n):
                if abs(mrc[i]) > 1e-12:
                    new_w[i] = indiv_vols[i] / mrc[i]
                else:
                    new_w[i] = indiv_vols[i] / 1e-6

            s = float(np.sum(new_w))
            if s > 1e-14:
                new_w = new_w / s

            # 投影到约束集
            new_w = _project_onto_simplex(new_w, max_single, min_single)

            # 收敛检查
            change = float(np.max(np.abs(new_w - prev_w)))
            prev_w = new_w.copy()
            w = new_w

            if change < self._tol:
                break

        w_final = _project_onto_simplex(best_w, max_single, min_single)
        port_var = float(w_final @ cov_matrix @ w_final)
        port_vol = math.sqrt(max(port_var, 1e-14))
        weighted_vol = float(np.sum(w_final * indiv_vols))
        final_div = _safe_divide(weighted_vol, port_vol, 0.0)

        logger.debug(f"MaxDiversification: iterations={len(self._objective_history)}, "
                     f"div_ratio={final_div:.4f}")

        return w_final, final_div

    def get_objective_history(self) -> List[float]:
        return list(self._objective_history)

    def has_converged(self) -> bool:
        if len(self._objective_history) < 2:
            return False
        return abs(self._objective_history[-1] - self._objective_history[-2]) < self._tol


# ═══════════════════════════════════════════════════════════════
# RiskBudgetOptimizer — 风险预算/风险平价优化（CCD算法）
# ═══════════════════════════════════════════════════════════════

class RiskBudgetOptimizer:
    """风险预算优化器

    使用循环坐标下降法（Cyclical Coordinate Descent, CCD）求解：
      RC_i = w_i * (Σw)_i / σ_portfolio = target_RC_i

    等风险贡献（ERC）: target_RC_i = 1/N
    """

    def __init__(self, max_iterations: int = 200, convergence_tol: float = 1e-6):
        self._max_iter = max_iterations
        self._tol = convergence_tol
        self._rc_history: List[np.ndarray] = []

    def optimize(
        self,
        cov_matrix: np.ndarray,
        target_rc: Optional[np.ndarray] = None,   # 目标风险贡献向量
        constraints: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """CCD优化风险预算

        Args:
            cov_matrix: 协方差矩阵
            target_rc: 目标风险贡献，None则使用ERC (1/N each)
            constraints: {max_single_weight, min_single_weight}

        Returns:
            (optimal_weights, risk_contributions)
        """
        n = cov_matrix.shape[0]
        constraints = constraints or {}
        max_single = constraints.get("max_single_weight", 0.30)
        min_single = constraints.get("min_single_weight", 0.05)

        if target_rc is None:
            target_rc = np.ones(n) / n

        # 确保 target_rc 归一化（并防御零和/NaN）
        target_rc = np.asarray(target_rc, dtype=float)
        if not np.all(np.isfinite(target_rc)):
            target_rc = np.ones(n) / n
        target_sum = float(np.sum(target_rc))
        if abs(target_sum) < 1e-14:
            target_rc = np.ones(n) / n
        else:
            target_rc = target_rc / target_sum

        # 初始权重：1/σ 归一化
        indiv_vols = np.sqrt(np.diag(cov_matrix))
        inv_vols = 1.0 / np.maximum(indiv_vols, 1e-8)
        w = inv_vols / np.sum(inv_vols)
        w = _project_onto_simplex(w, max_single, min_single)

        self._rc_history = []
        prev_w = w.copy()

        for iteration in range(self._max_iter):
            # CCD: 每次优化一个坐标（权重）
            for i in range(n):
                # 固定其他权重，优化 w_i
                # 使用二分搜索法找到使 RC_i 接近 target_rc[i] 的 w_i

                # 当前组合指标
                port_var = float(w @ cov_matrix @ w)
                port_vol = math.sqrt(max(port_var, 1e-12))

                # 当前 RC_i / target_RC_i 的比值
                mrc = cov_matrix @ w
                rc_i = w[i] * mrc[i] / max(port_vol, 1e-12)

                if abs(rc_i) < 1e-10 and abs(target_rc[i]) < 1e-10:
                    continue

                # 调整方向
                if rc_i < target_rc[i]:
                    # 需要增大 w_i
                    lo = w[i]
                    hi = min(max_single, 1.0 - (np.sum(w) - w[i] + min_single * (n - 1)))
                    hi = max(hi, lo + 0.001)
                else:
                    # 需要减小 w_i
                    lo = max(min_single, 1.0 - (np.sum(w) - w[i] + max_single * (n - 1)))
                    lo = min(lo, w[i] - 0.001)
                    hi = w[i]

                if lo >= hi:
                    continue

                # 二分搜索
                for _ in range(30):
                    mid = (lo + hi) / 2.0
                    w_test = w.copy()
                    w_test[i] = mid

                    # 调整其他权重以保持 sum=1
                    other_mask = np.ones(n, dtype=bool)
                    other_mask[i] = False
                    other_sum = float(np.sum(w[other_mask]))
                    if other_sum > 1e-12:
                        scale = (1.0 - mid) / other_sum
                        w_test[other_mask] = w[other_mask] * scale
                    else:
                        w_test[other_mask] = (1.0 - mid) / (n - 1)

                    # 裁剪
                    w_test = np.clip(w_test, min_single, max_single)
                    w_test = w_test / np.sum(w_test)

                    port_var_test = float(w_test @ cov_matrix @ w_test)
                    port_vol_test = math.sqrt(max(port_var_test, 1e-12))
                    mrc_test = cov_matrix @ w_test
                    rc_test_i = w_test[i] * mrc_test[i] / max(port_vol_test, 1e-12)

                    if rc_test_i < target_rc[i]:
                        lo = mid
                    else:
                        hi = mid

                    if abs(rc_test_i - target_rc[i]) < self._tol * 10:
                        break

                # 更新权重
                w[i] = (lo + hi) / 2.0
                w = w / np.sum(w)
                w = np.clip(w, min_single, max_single)
                w = w / np.sum(w)

            # 计算风险贡献
            port_var = float(w @ cov_matrix @ w)
            port_vol = math.sqrt(max(port_var, 1e-12))
            mrc = cov_matrix @ w
            rc = w * mrc / max(port_vol, 1e-12)

            self._rc_history.append(rc.copy())

            # 收敛检查
            change = float(np.max(np.abs(w - prev_w)))
            prev_w = w.copy()

            rc_error = float(np.max(np.abs(rc - target_rc)))
            if change < self._tol and rc_error < 0.01:
                break

        # 最终裁剪
        w_final = np.clip(w, min_single, max_single)
        w_final = w_final / np.sum(w_final)

        # 最终风险贡献
        port_var = float(w_final @ cov_matrix @ w_final)
        port_vol = math.sqrt(max(port_var, 1e-12))
        mrc = cov_matrix @ w_final
        rc_final = w_final * mrc / max(port_vol, 1e-12)

        logger.debug(f"RiskBudget CCD: iterations={len(self._rc_history)}, "
                     f"max_rc_error={float(np.max(np.abs(rc_final - target_rc))):.6f}")

        return w_final, rc_final

    def get_rc_history(self) -> List[List[float]]:
        return [list(rc) for rc in self._rc_history]


# ═══════════════════════════════════════════════════════════════
# MinimumCVaROptimizer — 最小CVaR优化（Rockafellar-Uryasev + 单纯形法）
# ═══════════════════════════════════════════════════════════════

class MinimumCVaROptimizer:
    """最小CVaR组合优化器

    Rockafellar-Uryasev 方法：
      CVaR_α(w) = min_γ { γ + 1/(α*T) * Σ max(-r_t'w - γ, 0) }

    使用内联单纯形法求解线性规划。
    """

    def __init__(self, max_iterations: int = 500, convergence_tol: float = 1e-6):
        self._max_iter = max_iterations
        self._tol = convergence_tol

    def optimize(
        self,
        returns_matrix: np.ndarray,      # (n_assets, n_periods)
        alpha: float = 0.95,
        constraints: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, float, float]:
        """最小CVaR优化

        Args:
            returns_matrix: 收益率矩阵 (n_assets × n_periods)
            alpha: CVaR置信水平
            constraints: 约束字典

        Returns:
            (optimal_weights, cvar_value, var_value)
        """
        n_assets, n_periods = returns_matrix.shape
        alpha = _safe_float(alpha, 0.95)
        if not (0.0 < alpha < 1.0):
            alpha = 0.95
        constraints = constraints or {}
        max_single = constraints.get("max_single_weight", 0.30)
        min_single = constraints.get("min_single_weight", 0.05)

        # ── 构建线性规划问题 ──
        # 决策变量: [w_0, ..., w_{n-1}, γ, s_0, ..., s_{T-1}]
        # 最小化: γ + 1/(α*T) * Σ s_t
        # 约束:
        #   s_t >= 0
        #   s_t >= -r_t'w - γ  →  s_t + r_t'w + γ >= 0
        #   Σ w = 1, w >= 0

        # 为简化，使用梯度下降+线性规划混合方法：
        # 1. 给定 w，计算最优 γ 和 CVaR
        # 2. 计算 CVaR 对 w 的梯度
        # 3. 更新 w 并投影

        inv_vols = 1.0 / np.maximum(np.sqrt(np.mean(returns_matrix ** 2, axis=1)), 1e-8)
        w = inv_vols / np.sum(inv_vols)
        w = np.clip(w, min_single, max_single)
        w = w / np.sum(w)

        best_w = w.copy()
        best_cvar = np.inf
        prev_w = w.copy()
        learning_rate = 0.005

        for iteration in range(self._max_iter):
            # 1. 给定 w，计算组合收益率序列
            port_returns = returns_matrix.T @ w   # (T,)
            sorted_rets = np.sort(port_returns)

            # 2. 计算最优 γ 和 CVaR
            # 最优 γ 就是 -VaR_α，即损失分布的 (1-α) 分位点
            var_idx = int(n_periods * (1 - alpha))
            var_idx = max(0, min(var_idx, n_periods - 1))
            neg_losses = -sorted_rets
            gamma = neg_losses[var_idx] if var_idx < n_periods else 0.0

            # CVaR = γ + 1/(α*T) * Σ max(-r_t'w - γ, 0)
            tail_excess = np.maximum(-port_returns - gamma, 0)
            cvar = gamma + np.sum(tail_excess) / (alpha * n_periods)

            if cvar < best_cvar:
                best_cvar = cvar
                best_w = w.copy()

            # 3. 计算梯度
            # ∂CVaR/∂w 只来自尾部的收益率
            tail_mask = (-port_returns - gamma) > 0
            n_tail = np.sum(tail_mask)
            if n_tail > 0:
                # CVaR 对 w 的梯度：尾部的平均负收益率
                tail_rets = returns_matrix[:, tail_mask]  # (n, n_tail)
                gradient = -np.mean(tail_rets, axis=1) / (alpha * n_periods) * n_tail
            else:
                # 没有尾部，使用所有样本近似
                gradient = -np.mean(returns_matrix, axis=1)

            # 4. 更新权重
            w_new = w - learning_rate * gradient
            w_new = np.clip(w_new, min_single, max_single)
            w_new = w_new / np.sum(w_new)

            # 5. 收敛检查
            change = float(np.max(np.abs(w_new - prev_w)))
            prev_w = w_new.copy()
            w = w_new

            # 自适应学习率
            if iteration > 0 and cvar >= best_cvar + 1e-8:
                learning_rate *= 0.8
            learning_rate = max(learning_rate, 1e-6)

            if change < self._tol:
                break

        w_final = np.clip(best_w, min_single, max_single)
        w_final = w_final / np.sum(w_final)

        # 最终 CVaR 和 VaR
        port_returns_final = returns_matrix.T @ w_final
        sorted_rets_final = np.sort(port_returns_final)
        var_idx_final = int(n_periods * (1 - alpha))
        var_idx_final = max(0, min(var_idx_final, n_periods - 1))
        var_val = -sorted_rets_final[var_idx_final]
        tail_excess_final = np.maximum(-port_returns_final - var_val, 0)
        cvar_val = var_val + np.sum(tail_excess_final) / (alpha * n_periods)

        logger.debug(f"MinCVaR: iterations={iteration + 1}, cvar={cvar_val:.6f}, var={var_val:.6f}")

        return w_final, cvar_val, var_val


# ═══════════════════════════════════════════════════════════════
# 单纯形法求解器（内联实现）
# ═══════════════════════════════════════════════════════════════

class _SimplexSolver:
    """内联单纯形法求解器

    用于求解标准形式的线性规划:
      min c'x  s.t. Ax = b, x >= 0
    """

    def __init__(self, max_iterations: int = 1000, tolerance: float = 1e-8):
        self._max_iter = max_iterations
        self._tol = tolerance

    def solve(self, c: np.ndarray, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """求解 LP: min c'x, Ax = b, x >= 0

        Returns:
            (solution_x, objective_value, success)
        """
        m, n = A.shape  # m constraints, n variables

        # 添加松弛变量构建初始基
        # 这里假设 b >= 0，使用大M法
        # 简化实现：两阶段法

        # Phase I: 找初始可行基
        # 对于 b >= 0，添加人工变量

        # 简化：假设所有变量非负，且 A 行满秩
        # 使用最小下标规则选择进基/出基变量

        # 构建单纯形表
        # 扩展：添加松弛变量使 Ax = b 可行
        n_total = n + m

        # 构建扩展单纯形表
        tableau = np.zeros((m + 1, n_total + 1))

        # 约束行
        for i in range(m):
            for j in range(n):
                tableau[i, j] = A[i, j]
            tableau[i, n + i] = 1.0     # 松弛变量
            tableau[i, -1] = b[i]

        # 目标函数行（负成本）
        for j in range(n):
            tableau[m, j] = -c[j]
        # 松弛变量在目标函数中系数为0

        # 基变量初始为松弛变量
        basis = [n + i for i in range(m)]

        for iteration in range(self._max_iter):
            # 检查最优性：目标行是否有负系数（最小化问题）
            reduced_costs = tableau[m, :n_total]

            entering = -1
            most_negative = -self._tol
            for j in range(n_total):
                if reduced_costs[j] < most_negative:
                    most_negative = reduced_costs[j]
                    entering = j

            if entering == -1:
                # 最优解找到
                break

            # 找离开变量（最小比率测试）
            leaving = -1
            min_ratio = np.inf
            for i in range(m):
                if tableau[i, entering] > self._tol:
                    ratio = tableau[i, -1] / tableau[i, entering]
                    if ratio < min_ratio:
                        min_ratio = ratio
                        leaving = i

            if leaving == -1:
                # 无界解
                return np.zeros(n), np.inf, False

            # 旋转
            pivot = tableau[leaving, entering]

            # 归一化旋转行
            tableau[leaving, :] /= pivot

            # 更新其他行
            for i in range(m + 1):
                if i != leaving:
                    factor = tableau[i, entering]
                    tableau[i, :] -= factor * tableau[leaving, :]

            basis[leaving] = entering

        # 提取解
        x = np.zeros(n_total)
        for i in range(m):
            if basis[i] < n_total:
                x[basis[i]] = tableau[i, -1]

        # 只返回原始变量
        x_original = x[:n]

        # 目标函数值
        objective = float(x_original @ c)

        return x_original, objective, True


# ═══════════════════════════════════════════════════════════════
# DiversificationOptimizer — 主类
# ═══════════════════════════════════════════════════════════════

class DiversificationOptimizer:
    """分散化优化器主类

    统一入口，协调所有子优化器和指标计算。
    支持：
      - 多方法优化 (max_sharpe / min_variance / max_diversification / risk_parity / min_cvar)
      - 有效前沿计算
      - 分散化分析
      - 再平衡建议
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._lock = asyncio.Lock()

        # ── 配置读取 ──
        div_cfg = config.get("diversification_optimizer", {})
        self._default_method = div_cfg.get("default_method", "max_sharpe")
        self._max_single_weight = div_cfg.get("max_single_weight", 0.30)
        self._min_single_weight = div_cfg.get("min_single_weight", 0.05)
        self._max_iterations = div_cfg.get("max_iterations", 200)
        self._convergence_tol = div_cfg.get("convergence_tol", 1e-6)
        self._risk_free_rate = div_cfg.get("risk_free_rate", 0.02)
        self._cvar_alpha = div_cfg.get("cvar_alpha", 0.95)

        # ── 子组件 ──
        self._efficient_frontier = EfficientFrontier(
            risk_free_rate=self._risk_free_rate,
            max_iterations=self._max_iterations,
            convergence_tol=self._convergence_tol,
        )
        self._max_div_optimizer = MaxDiversificationOptimizer(
            max_iterations=self._max_iterations,
            convergence_tol=self._convergence_tol,
        )
        self._risk_budget_optimizer = RiskBudgetOptimizer(
            max_iterations=self._max_iterations,
            convergence_tol=self._convergence_tol,
        )
        self._min_cvar_optimizer = MinimumCVaROptimizer(
            max_iterations=self._max_iterations * 2,
            convergence_tol=self._convergence_tol,
        )
        self._metrics = DiversificationMetrics()

        # ── 状态 ──
        self._is_running = False
        self._last_result: Optional[OptimizationResult] = None
        self._last_frontier: List[Dict[str, Any]] = []
        self._optimization_count: int = 0
        self._data_dir = div_cfg.get("data_dir", "./data")

        # 重启恢复：加载上次优化权重
        self._load_state()

        logger.info(
            f"DiversificationOptimizer initialized: "
            f"default_method={self._default_method}, "
            f"max_single={self._max_single_weight:.0%}, "
            f"min_single={self._min_single_weight:.0%}, "
            f"rf={self._risk_free_rate:.0%}, "
            f"cvar_alpha={self._cvar_alpha:.0%}"
        )

    # ── 生命周期 ─────────────────────────────────────────────

    async def start(self):
        """启动优化器"""
        if self._is_running:
            return
        self._is_running = True
        logger.info("DiversificationOptimizer started")

    async def stop(self):
        """停止优化器"""
        self._is_running = False
        logger.info("DiversificationOptimizer stopped")

    # ── 属性 ─────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._is_running

    # ── 核心优化接口 ─────────────────────────────────────────

    async def optimize(
        self,
        returns: Dict[str, List[float]],
        method: str = "max_sharpe",
        constraints: Optional[Dict] = None,
    ) -> OptimizationResult:
        """核心优化入口"""
        async with self._lock:
            result = self._optimize_impl(returns, method, constraints)
            elapsed = time.perf_counter() - time.perf_counter()  # 由调用方决定
            self._last_result = result
            self._optimization_count += 1
            self._save_state()
            logger.info(f"Optimization [{method}]: "
                        f"sharpe={result.sharpe_ratio:.4f}, "
                        f"div_ratio={result.diversification_ratio:.2f}, "
                        f"eff_n={result.effective_n:.1f}")
            return result

    def _optimize_impl(
        self,
        returns: Dict[str, List[float]],
        method: str = "max_sharpe",
        constraints: Optional[Dict] = None,
    ) -> OptimizationResult:
        """优化实现（不加锁，由调用方加锁）"""
        constraints = constraints or {}
        effective_constraints = {
            "max_single_weight": constraints.get("max_single_weight", self._max_single_weight),
            "min_single_weight": constraints.get("min_single_weight", self._min_single_weight),
        }

        strategy_names, returns_array = self._prepare_returns(returns)
        if len(strategy_names) < 1:
            return OptimizationResult(
                optimization_method=method,
                constraints_satisfied=["no_strategies"],
            )

        expected_returns = np.mean(returns_array, axis=1)
        cov_matrix = _compute_covariance_matrix(returns_array)

        if method == "max_sharpe":
            return self._optimize_max_sharpe_impl(
                strategy_names, expected_returns, cov_matrix,
                returns_array, effective_constraints
            )
        elif method == "min_variance":
            return self._optimize_min_variance_impl(
                strategy_names, expected_returns, cov_matrix,
                returns_array, effective_constraints
            )
        elif method == "max_diversification":
            return self._optimize_max_div_impl(
                strategy_names, expected_returns, cov_matrix,
                returns_array, effective_constraints
            )
        elif method == "risk_parity":
            return self._optimize_risk_parity_impl(
                strategy_names, expected_returns, cov_matrix,
                returns_array, effective_constraints
            )
        elif method == "min_cvar":
            return self._optimize_min_cvar_impl(
                strategy_names, expected_returns, cov_matrix,
                returns_array, effective_constraints
            )
        else:
            logger.warning(f"Unknown optimization method: {method}, using max_sharpe")
            return self._optimize_max_sharpe_impl(
                strategy_names, expected_returns, cov_matrix,
                returns_array, effective_constraints
            )

    def _optimize_max_sharpe_impl(
        self,
        strategy_names: List[str],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        returns_array: np.ndarray,
        constraints: Dict,
    ) -> OptimizationResult:
        """最大Sharpe优化实现（不加锁）"""
        w_opt, port_ret, port_vol, sharpe = self._efficient_frontier.compute_optimal_weights(
            expected_returns, cov_matrix, "max_sharpe", constraints
        )
        return self._build_result(
            strategy_names, w_opt, port_ret, port_vol, sharpe,
            returns_array, cov_matrix, "max_sharpe", constraints
        )

    def _optimize_min_variance_impl(
        self,
        strategy_names: List[str],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        returns_array: np.ndarray,
        constraints: Dict,
    ) -> OptimizationResult:
        """最小方差优化实现（不加锁）"""
        w_opt, port_ret, port_vol, sharpe = self._efficient_frontier.compute_optimal_weights(
            expected_returns, cov_matrix, "min_variance", constraints
        )
        return self._build_result(
            strategy_names, w_opt, port_ret, port_vol, sharpe,
            returns_array, cov_matrix, "min_variance", constraints
        )

    def _optimize_max_div_impl(
        self,
        strategy_names: List[str],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        returns_array: np.ndarray,
        constraints: Dict,
    ) -> OptimizationResult:
        """最大分散比优化实现（不加锁）"""
        w_opt, div_ratio = self._max_div_optimizer.optimize(cov_matrix, constraints)
        port_ret = float(w_opt @ expected_returns)
        port_var = float(w_opt @ cov_matrix @ w_opt)
        port_vol = math.sqrt(max(port_var, 1e-12))
        sharpe = _safe_divide(port_ret - self._risk_free_rate, port_vol, 0.0)
        return self._build_result(
            strategy_names, w_opt, port_ret, port_vol, sharpe,
            returns_array, cov_matrix, "max_diversification", constraints
        )

    def _optimize_risk_parity_impl(
        self,
        strategy_names: List[str],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        returns_array: np.ndarray,
        constraints: Dict,
    ) -> OptimizationResult:
        """风险平价优化实现（不加锁）"""
        w_opt, rc_final = self._risk_budget_optimizer.optimize(
            cov_matrix, None, constraints
        )
        port_ret = float(w_opt @ expected_returns)
        port_var = float(w_opt @ cov_matrix @ w_opt)
        port_vol = math.sqrt(max(port_var, 1e-12))
        sharpe = _safe_divide(port_ret - self._risk_free_rate, port_vol, 0.0)
        return self._build_result(
            strategy_names, w_opt, port_ret, port_vol, sharpe,
            returns_array, cov_matrix, "risk_parity", constraints
        )

    def _optimize_min_cvar_impl(
        self,
        strategy_names: List[str],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        returns_array: np.ndarray,
        constraints: Dict,
    ) -> OptimizationResult:
        """最小CVaR优化实现（不加锁）"""
        w_opt, cvar_val, var_val = self._min_cvar_optimizer.optimize(
            returns_array, self._cvar_alpha, constraints
        )
        port_ret = float(w_opt @ expected_returns)
        port_var = float(w_opt @ cov_matrix @ w_opt)
        port_vol = math.sqrt(max(port_var, 1e-12))
        sharpe = _safe_divide(port_ret - self._risk_free_rate, port_vol, 0.0)
        return self._build_result(
            strategy_names, w_opt, port_ret, port_vol, sharpe,
            returns_array, cov_matrix, "min_cvar", constraints
        )

    async def compute_efficient_frontier(
        self,
        returns: Dict[str, List[float]],
        n_points: int = 20,
    ) -> List[Dict[str, Any]]:
        """计算有效前沿

        Returns:
            [{return, risk, sharpe}, ...]
        """
        async with self._lock:
            strategy_names, returns_array = self._prepare_returns(returns)
            if len(strategy_names) < 2:
                return []

            expected_returns = np.mean(returns_array, axis=1)
            cov_matrix = _compute_covariance_matrix(returns_array)

            constraints = {
                "max_single_weight": self._max_single_weight,
                "min_single_weight": self._min_single_weight,
            }

            frontier = self._efficient_frontier.trace_efficient_frontier(
                expected_returns, cov_matrix, n_points, constraints
            )

            self._last_frontier = frontier
            self._save_state()
            return frontier

    async def optimize_max_diversification(
        self,
        returns: Dict[str, List[float]],
        constraints: Optional[Dict] = None,
    ) -> OptimizationResult:
        """最大分散比优化"""
        async with self._lock:
            return self._optimize_impl(returns, "max_diversification", constraints)

    async def optimize_risk_parity(
        self,
        returns: Dict[str, List[float]],
        constraints: Optional[Dict] = None,
    ) -> OptimizationResult:
        """风险平价优化（等风险贡献）"""
        async with self._lock:
            return self._optimize_impl(returns, "risk_parity", constraints)

    async def optimize_min_cvar(
        self,
        returns: Dict[str, List[float]],
        alpha: float = 0.95,
    ) -> OptimizationResult:
        """最小CVaR优化"""
        async with self._lock:
            # 临时覆盖 alpha
            original_alpha = self._cvar_alpha
            self._cvar_alpha = alpha
            try:
                return self._optimize_impl(returns, "min_cvar", None)
            finally:
                self._cvar_alpha = original_alpha

    async def analyze_diversification(
        self,
        current_weights: Dict[str, float],
        returns: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        """分析当前配置的分散化指标

        Args:
            current_weights: {strategy: weight}
            returns: {strategy: [return_series]}

        Returns:
            分散化分析报告
        """
        async with self._lock:
            strategy_names, returns_array = self._prepare_returns(returns)
            if not strategy_names:
                return {"error": "No return data"}

            current_weights = current_weights or {}
            cov_matrix = _compute_covariance_matrix(returns_array)

            # 对齐权重
            aligned_weights: Dict[str, float] = {}
            for name in strategy_names:
                aligned_weights[name] = _safe_float(current_weights.get(name), 0.0) or 0.0

            # 归一化
            total = sum(aligned_weights.values())
            if total > 1e-14:
                aligned_weights = {k: v / total for k, v in aligned_weights.items()}

            metrics_result = self._metrics.compute(
                aligned_weights, cov_matrix, strategy_names
            )

            # 额外分析
            n = len(strategy_names)
            w = np.zeros(n)
            for i, name in enumerate(strategy_names):
                w[i] = aligned_weights.get(name, 0.0)

            port_ret = float(w @ np.mean(returns_array, axis=1))
            port_vol = math.sqrt(max(float(w @ cov_matrix @ w), 1e-14))
            sharpe = _safe_divide(port_ret - self._risk_free_rate, port_vol, 0.0)

            # 检测问题
            issues = []
            if metrics_result["diversification_ratio"] < 1.2:
                issues.append("low_diversification")
            if metrics_result["effective_n"] < 2.0:
                issues.append("low_effective_n")
            if metrics_result["risk_concentration"] > 3.0:
                issues.append("high_risk_concentration")
            if metrics_result["concentration_ratio"] > 0.5:
                issues.append("high_concentration")

            return {
                **metrics_result,
                "portfolio_return": round(port_ret, 6),
                "portfolio_risk": round(port_vol, 6),
                "sharpe_ratio": round(sharpe, 4),
                "issues": issues,
                "health": "good" if len(issues) == 0 else
                          "warning" if len(issues) <= 2 else "critical",
            }

    async def suggest_rebalance(
        self,
        current_weights: Dict[str, float],
        returns: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        """生成再平衡建议

        比较当前权重与最优权重，计算需要调整的方向和幅度。
        """
        async with self._lock:
            current_weights = current_weights or {}
            current_weights = {
                k: _safe_float(v, 0.0) or 0.0 for k, v in current_weights.items()
            }
            # 使用内部分发避免重复加锁
            optimal_weights: Dict[str, Dict[str, float]] = {}
            for method, name in [
                ("max_sharpe", "sharpe_optimal"),
                ("max_diversification", "div_optimal"),
                ("risk_parity", "rp_optimal"),
            ]:
                try:
                    result = self._optimize_impl(returns, method=method)
                    optimal_weights[name] = result.weights
                except Exception as e:
                    logger.warning(f"Rebalance could not compute {method}: {e}")
                    optimal_weights[name] = {}

            # 当前权重归一化
            total = sum(current_weights.values())
            if total > 1e-14:
                norm_current = {k: v / total for k, v in current_weights.items()}
            else:
                norm_current = current_weights

            # 计算差异
            suggestions = []
            for name, opt_w in optimal_weights.items():
                if not opt_w:
                    continue
                diffs = {}
                all_keys = set(list(norm_current.keys()) + list(opt_w.keys()))
                for key in all_keys:
                    cur = norm_current.get(key, 0.0)
                    opt = opt_w.get(key, 0.0)
                    diff = opt - cur
                    if abs(diff) > 0.01:  # 超过1%才建议调整
                        diffs[key] = round(diff, 4)

                if diffs:
                    suggestions.append({
                        "target": name,
                        "adjustments": diffs,
                        "total_turnover": round(sum(abs(v) for v in diffs.values()) / 2, 4),
                    })

            return {
                "current_weights": {k: round(v, 4) for k, v in norm_current.items()},
                "optimal_targets": {
                    name: {k: round(v, 4) for k, v in opt.items()}
                    for name, opt in optimal_weights.items()
                },
                "suggestions": suggestions,
                "max_allowed_turnover": 0.30,  # 最大换手率30%
                "timestamp": datetime.now().isoformat(),
            }

    def get_summary(self) -> Dict[str, Any]:
        """获取优化器状态摘要"""
        metrics_current = self._metrics.get_current()
        return {
            "is_running": self._is_running,
            "default_method": self._default_method,
            "max_single_weight": self._max_single_weight,
            "min_single_weight": self._min_single_weight,
            "risk_free_rate": self._risk_free_rate,
            "cvar_alpha": self._cvar_alpha,
            "optimization_count": self._optimization_count,
            "last_optimization": self._last_result.to_dict() if self._last_result else None,
            "last_metrics": metrics_current,
            "frontier_points": len(self._last_frontier),
            "max_div_converged": self._max_div_optimizer.has_converged() if self._optimization_count > 0 else None,
        }

    # ── 状态持久化 ─────────────────────────────────────────

    def _save_state(self) -> None:
        """持久化最近一次优化权重，重启后可恢复。"""
        try:
            os.makedirs(self._data_dir, exist_ok=True)
            state_path = os.path.join(self._data_dir, "diversification_optimizer_state.json")
            state = {
                "last_updated": datetime.now().isoformat(),
                "optimization_count": self._optimization_count,
                "last_result": self._last_result.to_dict() if self._last_result else None,
                "last_frontier": self._last_frontier,
            }
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            logger.debug(f"Diversification optimizer state saved to {state_path}")
        except Exception as e:
            logger.warning(f"Failed to save diversification optimizer state: {e}")

    def _load_state(self) -> bool:
        """加载持久化的优化状态；任何异常都不影响启动。"""
        try:
            state_path = os.path.join(self._data_dir, "diversification_optimizer_state.json")
            if not os.path.exists(state_path):
                return False
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self._optimization_count = int(_safe_float(state.get("optimization_count"), 0.0) or 0)
            last = state.get("last_result")
            if isinstance(last, dict) and isinstance(last.get("weights"), dict):
                self._last_result = OptimizationResult(
                    weights={k: _safe_float(v, 0.0) or 0.0 for k, v in last["weights"].items()},
                    expected_return=_safe_float(last.get("expected_return"), 0.0) or 0.0,
                    expected_risk=_safe_float(last.get("expected_risk"), 0.0) or 0.0,
                    sharpe_ratio=_safe_float(last.get("sharpe_ratio"), 0.0) or 0.0,
                    diversification_ratio=_safe_float(last.get("diversification_ratio"), 0.0) or 0.0,
                    effective_n=_safe_float(last.get("effective_n"), 0.0) or 0.0,
                    concentration_ratio=_safe_float(last.get("concentration_ratio"), 0.0) or 0.0,
                    optimization_method=str(last.get("optimization_method", "")),
                    constraints_satisfied=list(last.get("constraints_satisfied", [])),
                    risk_contributions={
                        k: _safe_float(v, 0.0) or 0.0
                        for k, v in (last.get("risk_contributions") or {}).items()
                    },
                    timestamp=str(last.get("timestamp", "")),
                )
            frontier = state.get("last_frontier")
            if isinstance(frontier, list):
                self._last_frontier = frontier
            logger.info(f"Diversification optimizer state restored: count={self._optimization_count}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load diversification optimizer state: {e}")
            return False

    # ── 内部辅助方法 ─────────────────────────────────────────

    def _prepare_returns(
        self, returns: Dict[str, List[float]]
    ) -> Tuple[List[str], np.ndarray]:
        """准备收益率数据

        对齐所有策略的收益率序列长度，返回 (names, array)。
        对 None/NaN/Inf/非法值逐点过滤，避免污染协方差计算。
        """
        if not returns:
            return [], np.array([])

        clean: Dict[str, List[float]] = {}
        for name, series in returns.items():
            if not series:
                continue
            vals: List[float] = []
            for x in series:
                v = _safe_float(x)
                if v is not None:
                    vals.append(v)
            if len(vals) >= 5:
                clean[name] = vals

        if not clean:
            return [], np.array([])

        # 找到最短序列长度
        min_len = min(len(r) for r in clean.values())
        if min_len < 5:
            return [], np.array([])

        strategy_names = list(clean.keys())
        aligned = []
        for name in strategy_names:
            r = clean[name]
            aligned.append(r[-min_len:])

        return strategy_names, np.array(aligned, dtype=float)

    def _build_result(
        self,
        strategy_names: List[str],
        w_opt: np.ndarray,
        port_ret: float,
        port_vol: float,
        sharpe: float,
        returns_array: np.ndarray,
        cov_matrix: np.ndarray,
        method: str,
        constraints: Dict,
    ) -> OptimizationResult:
        """构建 OptimizationResult"""
        n = len(strategy_names)

        # 构建权重字典
        weights = {strategy_names[i]: round(float(w_opt[i]), 6) for i in range(n)}

        # 计算分散化指标
        metrics = self._metrics.compute(weights, cov_matrix, strategy_names)

        # 风险贡献
        mrc = cov_matrix @ w_opt
        rc = w_opt * mrc
        port_vol_safe = max(port_vol, 1e-12)
        risk_contributions = {
            strategy_names[i]: round(float(rc[i] / port_vol_safe), 4)
            for i in range(n)
        }

        # 约束满足情况
        satisfied = []
        if abs(float(np.sum(w_opt)) - 1.0) < 0.001:
            satisfied.append("full_investment")
        if np.all(w_opt >= -1e-10):
            satisfied.append("long_only")
        max_w = constraints.get("max_single_weight", 0.30)
        min_w = constraints.get("min_single_weight", 0.05)
        if np.all(w_opt <= max_w + 1e-6):
            satisfied.append(f"max_weight<={max_w:.0%}")
        if np.all(w_opt >= min_w - 1e-6):
            satisfied.append(f"min_weight>={min_w:.0%}")

        return OptimizationResult(
            weights=weights,
            expected_return=round(port_ret, 6),
            expected_risk=round(port_vol, 6),
            sharpe_ratio=round(sharpe, 4),
            diversification_ratio=metrics["diversification_ratio"],
            effective_n=metrics["effective_n"],
            concentration_ratio=metrics["concentration_ratio"],
            optimization_method=method,
            constraints_satisfied=satisfied,
            risk_contributions=risk_contributions,
        )


# ═══════════════════════════════════════════════════════════════
# 单例工厂
# ═══════════════════════════════════════════════════════════════

_diversification_optimizer: Optional[DiversificationOptimizer] = None


def get_diversification_optimizer(config: Dict[str, Any] = None) -> DiversificationOptimizer:
    """获取分散化优化器单例"""
    global _diversification_optimizer
    if _diversification_optimizer is None and config is not None:
        _diversification_optimizer = DiversificationOptimizer(config)
    return _diversification_optimizer


def reset_diversification_optimizer():
    """重置分散化优化器单例"""
    global _diversification_optimizer
    _diversification_optimizer = None


__all__ = [
    "OptimizationResult",
    "EfficientFrontier",
    "MaxDiversificationOptimizer",
    "RiskBudgetOptimizer",
    "MinimumCVaROptimizer",
    "DiversificationMetrics",
    "DiversificationOptimizer",
    "get_diversification_optimizer",
    "reset_diversification_optimizer",
]
