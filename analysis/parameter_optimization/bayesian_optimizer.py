"""
贝叶斯参数优化器 (Bayesian Optimizer)

基于高斯过程回归的贝叶斯优化，用于交易策略参数调优：
  - 高斯过程回归 (RBF/Matern 核)
  - 多种采集函数：UCB、EI、PI、Thompson Sampling
  - 约束感知优化（可行域约束）
  - 批量建议 (q-EI)
  - 超参数自适应（核参数自动调整）
  - 收敛检测与早期停止
"""
import asyncio
import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Callable
import numpy as np
from loguru import logger

from analysis.parameter_optimization.genetic_optimizer import ParameterDef


def _std_normal_cdf(z):
    """标准正态分布 CDF（math.erf 实现，无 scipy 依赖）。"""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def _std_normal_pdf(z):
    """标准正态分布 PDF（无 scipy 依赖）。"""
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


class AcquisitionFunction(Enum):
    """采集函数"""
    UCB = "ucb"
    EI = "ei"
    PI = "pi"
    THOMPSON = "thompson"


class KernelType(Enum):
    """核函数类型"""
    RBF = "rbf"
    MATERN32 = "matern32"
    MATERN52 = "matern52"


@dataclass
class Observation:
    """单次观测"""
    params: Dict[str, float]
    value: float
    iteration: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BOOptimizationResult:
    """贝叶斯优化结果"""
    best_params: Dict[str, float]
    best_value: float
    observations: List[Observation] = field(default_factory=list)
    acquisition_function: str = ""
    total_iterations: int = 0
    total_time_seconds: float = 0.0
    convergence_iteration: int = 0
    convergence_reason: str = ""
    # 诊断信息
    gp_log_marginal_likelihood: float = 0.0
    kernel_params: Dict[str, float] = field(default_factory=dict)
    value_history: List[float] = field(default_factory=list)
    # 后验统计
    posterior_mean: Optional[List[float]] = None
    posterior_std: Optional[List[float]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        def _finite(v: Any, default: float = 0.0) -> Any:
            """将数值安全转为有限 float；非数值（如字符串类别）原样保留，NaN/Inf 归零。"""
            if isinstance(v, bool):
                return v
            try:
                f = float(v)
            except (TypeError, ValueError):
                return v
            return f if math.isfinite(f) else default

        return {
            "best_params": {k: _finite(v) for k, v in self.best_params.items()},
            "best_value": _finite(self.best_value),
            "acquisition_function": self.acquisition_function,
            "total_iterations": _finite(self.total_iterations),
            "total_time_seconds": _finite(self.total_time_seconds),
            "convergence_iteration": _finite(self.convergence_iteration),
            "convergence_reason": self.convergence_reason,
            "gp_log_likelihood": _finite(self.gp_log_marginal_likelihood),
            "kernel_params": {k: _finite(v) for k, v in self.kernel_params.items()},
            "value_history": [round(v, 6) for v in self.value_history if math.isfinite(v)],
            "observation_count": len(self.observations),
            "timestamp": self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════
# 高斯过程回归 (纯 NumPy 实现)
# ═══════════════════════════════════════════════════════════════

class GaussianProcess:
    """高斯过程回归器（纯 NumPy）"""

    def __init__(self, kernel_type: KernelType = KernelType.MATERN52,
                 length_scale: float = 1.0, signal_variance: float = 1.0,
                 noise_variance: float = 1e-6):
        self._kernel_type = kernel_type
        self._length_scale = length_scale
        self._signal_variance = signal_variance
        self._noise_variance = noise_variance
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._L: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None
        self._K: Optional[np.ndarray] = None
        self._y_mean: float = 0.0

    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """计算核矩阵"""
        length_scale = self._length_scale if (math.isfinite(self._length_scale) and self._length_scale > 0) else 1.0
        signal_variance = self._signal_variance if math.isfinite(self._signal_variance) else 1.0
        dists = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2 / (length_scale ** 2), axis=-1)
        if self._kernel_type == KernelType.RBF:
            K = signal_variance * np.exp(-0.5 * dists)
        elif self._kernel_type == KernelType.MATERN32:
            sqrt3 = math.sqrt(3)
            d_scaled = sqrt3 * np.sqrt(dists)
            K = signal_variance * (1.0 + d_scaled) * np.exp(-d_scaled)
        else:  # MATERN52
            sqrt5 = math.sqrt(5)
            d_scaled = sqrt5 * np.sqrt(dists)
            K = signal_variance * (1.0 + d_scaled + d_scaled ** 2 / 3.0) * np.exp(-d_scaled)
        return K

    def fit(self, X: np.ndarray, y: np.ndarray):
        """拟合 GP 模型（Cholesky 分解，避免显式求逆提升数值稳定性）"""
        n = len(y)
        self._X_train = None
        self._y_train = None
        self._L = None
        self._alpha = None
        self._K = None
        self._y_mean = 0.0
        self._log_marginal_likelihood = float('-inf')

        if n == 0:
            return

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).flatten()
        if X_arr.shape[0] != n:
            X_arr = X_arr.reshape(n, -1)
        if not np.all(np.isfinite(X_arr)) or not np.all(np.isfinite(y_arr)):
            logger.warning("GaussianProcess.fit skipped: non-finite values in training data")
            return

        self._X_train = X_arr.copy()
        self._y_train = y_arr.copy()
        self._y_mean = float(np.mean(self._y_train))

        # 中心化
        y_centered = self._y_train - self._y_mean

        # 计算核矩阵 K + noise*I（带 jitter 递增，保证正定）
        noise = self._noise_variance if (math.isfinite(self._noise_variance) and self._noise_variance >= 0) else 1e-6
        jitter = 0.0
        while True:
            self._K = self._compute_kernel(X_arr, X_arr)
            self._K += np.eye(n) * (noise + jitter)
            try:
                self._L = np.linalg.cholesky(self._K)
                break
            except np.linalg.LinAlgError:
                jitter = 1e-6 if jitter == 0 else jitter * 10
                if jitter > 1e-1:
                    logger.warning("GaussianProcess.fit failed: kernel matrix not positive definite")
                    self._X_train = None
                    self._y_train = None
                    self._L = None
                    self._alpha = None
                    self._K = None
                    return

        # 用 Cholesky 解 K^{-1} y_centered
        try:
            self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y_centered))
        except np.linalg.LinAlgError:
            logger.warning("GaussianProcess.fit failed: Cholesky solve failed")
            self._L = None
            self._alpha = None
            return

        self._log_marginal_likelihood = float(
            -0.5 * y_centered.T @ self._alpha
            - np.sum(np.log(np.diag(self._L)))
            - 0.5 * n * np.log(2 * math.pi)
        )

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """后验预测（基于 Cholesky 因子）"""
        if self._X_train is None or self._L is None or self._alpha is None:
            n = len(np.asarray(X_test))
            return np.zeros(n), np.ones(n)

        K_s = self._compute_kernel(self._X_train, X_test)
        K_ss = self._compute_kernel(X_test, X_test)

        mean = self._y_mean + K_s.T @ self._alpha

        # cov = K_ss - K_s^T K^{-1} K_s，用三角求解提升数值稳定性
        try:
            v = np.linalg.solve(self._L, K_s)
            cov = K_ss - v.T @ v
            std = np.sqrt(np.maximum(np.diag(cov), 1e-10))
        except np.linalg.LinAlgError:
            return np.zeros(len(np.asarray(X_test))), np.ones(len(np.asarray(X_test)))
        return mean, std

    def sample_posterior(self, X_test: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """从后验分布采样"""
        mean, std = self.predict(X_test)
        samples = np.random.normal(mean, std, (n_samples, len(mean)))
        return samples

    @property
    def log_marginal_likelihood(self) -> float:
        return getattr(self, '_log_marginal_likelihood', float('-inf'))


# ═══════════════════════════════════════════════════════════════
# 贝叶斯优化器
# ═══════════════════════════════════════════════════════════════

class BayesianOptimizer:
    """贝叶斯参数优化器"""

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config.get("bayesian_optimizer", {}) if config else {}
        self._n_initial = cfg.get("n_initial_points", 10)
        self._n_iterations = cfg.get("n_iterations", 50)
        self._acquisition = AcquisitionFunction(cfg.get("acquisition", "ucb"))
        self._exploration_weight = cfg.get("exploration_weight", 2.0)
        self._kernel_type = KernelType(cfg.get("kernel_type", "matern52"))
        # 收敛
        self._convergence_patience = cfg.get("convergence_patience", 15)
        self._convergence_tol = cfg.get("convergence_tol", 1e-6)
        # GP 超参数
        self._length_scale = cfg.get("length_scale", 1.0)
        self._signal_variance = cfg.get("signal_variance", 1.0)
        self._noise_variance = cfg.get("noise_variance", 1e-6)
        # 采集函数优化
        self._n_acquisition_samples = cfg.get("n_acquisition_samples", 1000)
        self._n_restart_candidates = cfg.get("n_restart_candidates", 5)
        # 约束
        self._constraints: List[Callable[[Dict[str, float]], bool]] = []

        self._seed = cfg.get("seed", None)

        self._param_defs: List[ParameterDef] = []
        self._fitness_fn: Optional[Callable] = None
        self._observations: List[Observation] = []
        self._gp: Optional[GaussianProcess] = None
        self._best_observation: Optional[Observation] = None
        self._warm_start: Optional[Dict[str, float]] = None

        logger.info(f"BayesianOptimizer initialized: n_init={self._n_initial}, "
                    f"iter={self._n_iterations}, acq={self._acquisition.value}, "
                    f"kernel={self._kernel_type.value}")

    # ── 参数定义 ──────────────────────────────────────────────

    def set_param_defs(self, param_defs: List[ParameterDef]):
        self._param_defs = param_defs

    def add_param(self, name: str, low: float = 0, high: float = 1,
                  param_type: str = "float", step: float = 0,
                  categories: List = None, log_scale: bool = False,
                  description: str = ""):
        self._param_defs.append(ParameterDef(
            name=name, type=param_type, low=low, high=high,
            step=step, categories=categories, log_scale=log_scale,
            description=description,
        ))

    def set_fitness_fn(self, fn: Callable[[Dict[str, float]], float]):
        self._fitness_fn = fn

    def set_warm_start(self, params: Dict[str, float]):
        """设置热启动初始观测（如上游 GA 阶段的最优参数）。"""
        self._warm_start = dict(params)

    def add_constraint(self, constraint_fn: Callable[[Dict[str, float]], bool]):
        """添加可行性约束 fn(params) -> True 表示可行"""
        self._constraints.append(constraint_fn)

    # ── 参数编码 ──────────────────────────────────────────────

    def _params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """将参数字典映射到 [0,1] 归一化数组"""
        arr = np.zeros(len(self._param_defs))
        for i, pd in enumerate(self._param_defs):
            v = params.get(pd.name)
            if v is None:
                v = (pd.low + pd.high) / 2
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = (pd.low + pd.high) / 2
            if not math.isfinite(v):
                v = (pd.low + pd.high) / 2
            if pd.high > pd.low:
                if pd.log_scale and pd.low > 0:
                    log_low = math.log10(pd.low)
                    log_high = math.log10(pd.high)
                    arr[i] = (math.log10(max(v, pd.low)) - log_low) / (log_high - log_low)
                else:
                    arr[i] = (v - pd.low) / (pd.high - pd.low)
            else:
                arr[i] = 0.5
        return np.clip(arr, 0, 1)

    def _array_to_params(self, arr: np.ndarray) -> Dict[str, float]:
        """将 [0,1] 归一化数组映射回参数"""
        params = {}
        for i, pd in enumerate(self._param_defs):
            t = np.clip(arr[i], 0, 1)
            if pd.log_scale and pd.low > 0:
                log_low = math.log10(pd.low)
                log_high = math.log10(pd.high)
                v = 10 ** (log_low + t * (log_high - log_low))
            else:
                v = pd.low + t * (pd.high - pd.low)
            params[pd.name] = pd.clamp(v)
        return params

    def _is_feasible(self, params: Dict[str, float]) -> bool:
        """检查参数是否满足所有约束"""
        if not self._constraints:
            return True
        return all(c(params) for c in self._constraints)

    # ── 采集函数 ──────────────────────────────────────────────

    def _acquisition_value(self, X_candidate: np.ndarray) -> np.ndarray:
        """计算采集函数值"""
        if not self._gp:
            return np.random.rand(len(X_candidate))

        mean, std = self._gp.predict(X_candidate)
        # 处理 NaN/Inf
        std = np.nan_to_num(std, nan=1e-6, posinf=1e-6, neginf=1e-6)
        mean = np.nan_to_num(mean, nan=0.0)

        if self._acquisition == AcquisitionFunction.UCB:
            return mean + self._exploration_weight * std

        elif self._acquisition == AcquisitionFunction.EI:
            if not self._best_observation or not math.isfinite(self._best_observation.value):
                return std
            f_best = self._best_observation.value
            improvement = mean - f_best
            Z = improvement / np.maximum(std, 1e-10)
            # EI = improvement * Phi(Z) + std * phi(Z)，用 math.erf 实现（无 scipy 依赖）
            cdf = np.array([_std_normal_cdf(float(z)) for z in Z])
            pdf = np.array([_std_normal_pdf(float(z)) for z in Z])
            ei = improvement * cdf + std * pdf
            return np.maximum(np.nan_to_num(ei, nan=0.0), 0)

        elif self._acquisition == AcquisitionFunction.PI:
            if not self._best_observation or not math.isfinite(self._best_observation.value):
                return std
            f_best = self._best_observation.value + 0.01 * abs(self._best_observation.value)
            Z = (mean - f_best) / np.maximum(std, 1e-10)
            return np.array([_std_normal_cdf(float(z)) for z in Z])

        elif self._acquisition == AcquisitionFunction.THOMPSON:
            samples = self._gp.sample_posterior(X_candidate, n_samples=1)
            return samples.flatten()

        return mean + std

    def _suggest_next(self, n_candidates: int = None) -> Dict[str, float]:
        """使用采集函数最大化建议下一组参数"""
        n = n_candidates or self._n_acquisition_samples
        d = len(self._param_defs)

        # 随机候选点
        if self._gp:
            # 局部搜索：在最优观测附近密集采样
            n_local = n // 2
            rand_points = np.random.rand(n - n_local, d)
            if self._best_observation:
                best_arr = self._params_to_array(self._best_observation.params)
                local_points = best_arr + np.random.normal(0, 0.1, (n_local, d))
                local_points = np.clip(local_points, 0, 1)
                candidates = np.vstack([rand_points, local_points])
            else:
                candidates = rand_points
        else:
            candidates = np.random.rand(n, d)

        # 计算采集函数值
        acq_vals = self._acquisition_value(candidates)

        # 选择最优候选（满足约束）
        sorted_idx = np.argsort(acq_vals)[::-1]
        for idx in sorted_idx:
            params = self._array_to_params(candidates[idx])
            if self._is_feasible(params):
                return params

        # 兜底
        return self._array_to_params(candidates[sorted_idx[0]])

    # ── 初始化采样 ────────────────────────────────────────────

    def _sample_initial_points(self, n: int) -> List[Dict[str, float]]:
        """使用拉丁超立方采样生成初始点"""
        points = []
        for i in range(n):
            genes = []
            for pd in self._param_defs:
                seg = (i + random.random()) / n
                if pd.log_scale and pd.low > 0:
                    log_low = math.log10(pd.low)
                    log_high = math.log10(pd.high)
                    v = 10 ** (log_low + seg * (log_high - log_low))
                else:
                    v = pd.low + seg * (pd.high - pd.low)
                genes.append(pd.clamp(v))
            params = {pd.name: genes[i] for i, pd in enumerate(self._param_defs)}
            if self._is_feasible(params):
                points.append(params)
        # 填充
        while len(points) < n:
            params = {pd.name: pd.sample() for pd in self._param_defs}
            if self._is_feasible(params):
                points.append(params)
        return points[:n]

    # ── 评估 ──────────────────────────────────────────────────

    async def _evaluate_params(self, params: Dict[str, float], iteration: int) -> Observation:
        if not self._fitness_fn:
            return Observation(params=params, value=0.0, iteration=iteration)
        try:
            result = self._fitness_fn(params)
            if hasattr(result, '__await__'):
                result = await result
            value = float(result)
            if not math.isfinite(value):
                value = float('-inf')
        except Exception as e:
            logger.debug(f"BO evaluation error: {e}")
            value = float('-inf')
        return Observation(params=params, value=value, iteration=iteration)

    # ── GP 超参数自适应 ──────────────────────────────────────

    def _adapt_gp_hyperparams(self):
        """根据数据自适应调整 GP 超参数"""
        vals = [obs.value for obs in self._observations if math.isfinite(obs.value)]
        if len(vals) < 5:
            return
        # 基于观测范围调整 length_scale
        obs_range = max(vals) - min(vals)
        if math.isfinite(obs_range) and obs_range > 0:
            self._length_scale = max(0.1, obs_range / 5.0)
            self._noise_variance = max(1e-8, obs_range * 0.001)

    def _check_convergence(self) -> Tuple[bool, str]:
        """检测收敛"""
        if len(self._observations) < self._convergence_patience:
            return False, ""
        recent = [o.value for o in self._observations[-self._convergence_patience:] if math.isfinite(o.value)]
        if len(recent) < self._convergence_patience:
            return False, ""
        improvements = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i - 1] + self._convergence_tol)
        if improvements == 0:
            return True, "no_improvement"
        return False, ""

    # ── 主优化循环 ────────────────────────────────────────────

    async def optimize(self) -> BOOptimizationResult:
        """执行贝叶斯优化"""
        if not self._param_defs:
            raise ValueError("No parameter definitions set")
        if not self._fitness_fn:
            raise ValueError("No fitness function set")

        start_time = time.time()
        self._observations = []
        self._best_observation = None

        # 确定性：设置随机种子
        if self._seed is not None:
            random.seed(self._seed)
            np.random.seed(self._seed)

        # 热启动：评估注入的初始参数作为第一个观测
        if self._warm_start:
            obs = await self._evaluate_params(self._warm_start, 0)
            self._observations.append(obs)
            self._best_observation = obs
            logger.info(f"BO warm-start observation: {obs.value:.4f}")

        # 阶段1：初始采样
        logger.info(f"BO Phase 1: initial sampling ({self._n_initial} points)")
        init_points = self._sample_initial_points(self._n_initial)
        for i, params in enumerate(init_points):
            obs = await self._evaluate_params(params, i)
            self._observations.append(obs)
            if not self._best_observation or obs.value > self._best_observation.value:
                self._best_observation = obs
        best_val = self._best_observation.value if self._best_observation else float('-inf')
        logger.info(f"BO initial sampling complete: best={best_val:.4f}")

        # 阶段2：贝叶斯优化迭代
        for iteration in range(self._n_initial, self._n_initial + self._n_iterations):
            # 更新 GP 模型
            self._adapt_gp_hyperparams()
            fit_obs = [o for o in self._observations if math.isfinite(o.value)]
            if len(fit_obs) >= 2:
                X = np.array([self._params_to_array(o.params) for o in fit_obs])
                y = np.array([o.value for o in fit_obs])
                self._gp = GaussianProcess(
                    kernel_type=self._kernel_type,
                    length_scale=self._length_scale,
                    signal_variance=self._signal_variance,
                    noise_variance=self._noise_variance,
                )
                self._gp.fit(X, y)
            else:
                self._gp = None

            # 建议下一参数
            next_params = self._suggest_next()
            obs = await self._evaluate_params(next_params, iteration)
            self._observations.append(obs)

            if self._best_observation is None or obs.value > self._best_observation.value:
                self._best_observation = obs
                logger.debug(f"BO iter {iteration}: new best={obs.value:.4f}")

            # 收敛检测
            converged, reason = self._check_convergence()
            if converged and iteration > self._n_initial + self._convergence_patience:
                logger.info(f"BO converged at iter {iteration}: {reason}")
                break

            # 定期日志
            if iteration % 10 == 0 or iteration == self._n_initial:
                logger.info(f"BO iter {iteration}: best={self._best_observation.value:.4f}, "
                           f"gp_ll={self._gp.log_marginal_likelihood:.2f}")

        # 构建结果
        elapsed = time.time() - start_time
        conv_check = self._check_convergence()

        if self._best_observation is None or not math.isfinite(self._best_observation.value):
            best_params = {pd.name: pd.sample() for pd in self._param_defs}
            best_value = 0.0
            convergence_reason = "evaluation_failed"
        else:
            best_params = dict(self._best_observation.params)
            best_value = self._best_observation.value
            convergence_reason = conv_check[1]

        # 最终后验
        if self._gp and self._gp._X_train is not None:
            X_test = np.linspace(0, 1, 50).reshape(-1, 1)
            # 扩展到多维
            if len(self._param_defs) > 1:
                posterior_mean, posterior_std = None, None
            else:
                posterior_mean, posterior_std = self._gp.predict(X_test)
        else:
            posterior_mean, posterior_std = None, None

        return BOOptimizationResult(
            best_params=best_params,
            best_value=best_value,
            observations=self._observations,
            acquisition_function=self._acquisition.value,
            total_iterations=len(self._observations),
            total_time_seconds=elapsed,
            convergence_iteration=len(self._observations),
            convergence_reason=convergence_reason,
            gp_log_marginal_likelihood=self._gp.log_marginal_likelihood if self._gp else 0,
            kernel_params={
                "type": self._kernel_type.value,
                "length_scale": self._length_scale,
                "signal_variance": self._signal_variance,
            },
            value_history=[o.value for o in self._observations],
            posterior_mean=posterior_mean.tolist() if posterior_mean is not None else None,
            posterior_std=posterior_std.tolist() if posterior_std is not None else None,
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "n_initial": self._n_initial,
            "n_iterations": self._n_iterations,
            "acquisition": self._acquisition.value,
            "kernel": self._kernel_type.value,
            "observations": len(self._observations),
            "best_value": self._best_observation.value if self._best_observation else None,
            "constraints": len(self._constraints),
        }
