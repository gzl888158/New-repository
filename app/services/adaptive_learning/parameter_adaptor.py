"""参数自适应器：基于贝叶斯优化等方法动态调整策略参数。"""
import asyncio
import math
import random
import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable
from loguru import logger

from app.services.enterprise import EnterpriseServiceMixin


# =============================================================================
# 1. BayesianOptimizer — Gaussian Process surrogate model
# =============================================================================

class BayesianOptimizer:
    """基于高斯过程回归的贝叶斯优化器。

    使用 GP 替代模型估计参数空间中的性能曲面，通过采集函数
    （UCB / EI / PI）建议下一个待评估的参数点。
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._lock = asyncio.Lock()

        # 观测数据：X (n×d), y (n,)
        self._X: List[List[float]] = []
        self._y: List[float] = []

        # GP 超参数
        self._kernel_type = config.get("gp_kernel", "matern")  # "rbf" | "matern"
        self._length_scale = config.get("gp_length_scale", 1.0)
        self._signal_variance = config.get("gp_signal_variance", 1.0)
        self._noise_variance = config.get("gp_noise_variance", 1e-6)

        # 采集函数配置
        self._acquisition = config.get("acquisition", "ucb")   # "ucb" | "ei" | "pi"
        self._kappa = config.get("ucb_kappa", 2.0)             # UCB exploration weight
        self._xi = config.get("ei_xi", 0.01)                   # EI exploration bonus

        # 参数边界
        self._bounds: List[Tuple[float, float]] = config.get("bounds", [(0.0, 1.0)])
        # 参数名顺序（与 bounds 一一对应），observe 时按此顺序提取值，避免 dict 插入顺序错位
        self._param_names: List[str] = []

        self._fitted = False
        self._K_inv: Optional[List[List[float]]] = None         # (K + σ²I)^(-1)
        self._alpha: Optional[List[float]] = None               # K^(-1) * y

    # ---- 核函数 ----

    def _kernel(self, x1: List[float], x2: List[float]) -> float:
        """计算两个点之间的核函数值。"""
        if len(x1) != len(x2):
            raise ValueError("Dimension mismatch in kernel")
        d = len(x1)
        sq_dist = sum((x1[i] - x2[i]) ** 2 for i in range(d)) / (self._length_scale ** 2)

        if self._kernel_type == "rbf":
            return self._signal_variance * math.exp(-0.5 * sq_dist)
        elif self._kernel_type == "matern":
            # Matern 3/2
            r = math.sqrt(3.0 * sq_dist)
            return self._signal_variance * (1.0 + r) * math.exp(-r)
        else:
            return self._signal_variance * math.exp(-0.5 * sq_dist)

    def _build_kernel_matrix(self, X: List[List[float]]) -> List[List[float]]:
        """构建核矩阵 K。"""
        n = len(X)
        K = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                val = self._kernel(X[i], X[j])
                K[i][j] = val
                K[j][i] = val
        return K

    # ---- Cholesky 分解 (原地) ----

    def _cholesky(self, A: List[List[float]]) -> List[List[float]]:
        """Cholesky 分解 A = L * L^T，返回下三角 L。"""
        n = len(A)
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    L[i][j] = math.sqrt(max(A[i][i] - s, 1e-12))
                else:
                    L[i][j] = (A[i][j] - s) / max(L[j][j], 1e-12)
        return L

    def _solve_triangular(self, L: List[List[float]], b: List[float], lower: bool = True) -> List[float]:
        """解三角方程组 L*x = b (lower=True) 或 L^T*x = b (lower=False)。"""
        n = len(L)
        x = [0.0] * n
        if lower:
            for i in range(n):
                s = sum(L[i][j] * x[j] for j in range(i))
                x[i] = (b[i] - s) / max(L[i][i], 1e-12)
        else:
            for i in range(n - 1, -1, -1):
                s = sum(L[j][i] * x[j] for j in range(i + 1, n))
                x[i] = (b[i] - s) / max(L[i][i], 1e-12)
        return x

    # ---- GP 拟合 ----

    def _fit_gp(self) -> None:
        """根据已有观测数据拟合 GP。"""
        n = len(self._X)
        if n == 0:
            self._fitted = False
            return

        K = self._build_kernel_matrix(self._X)
        # 加噪声对角
        for i in range(n):
            K[i][i] += self._noise_variance

        L = self._cholesky(K)
        # 解 K * alpha = y
        y = self._y
        temp = self._solve_triangular(L, y, lower=True)
        self._alpha = self._solve_triangular(L, temp, lower=False)
        self._K_inv_L = L   # 保存 L 以便预测
        self._fitted = True

    # ---- GP 预测 ----

    def _predict(self, x_star: List[float]) -> Tuple[float, float]:
        """返回 (均值, 方差) 预测。"""
        if not self._fitted or len(self._X) == 0:
            return 0.0, self._signal_variance + self._noise_variance

        n = len(self._X)
        k_star = [self._kernel(x_star, xi) for xi in self._X]

        # 均值 = k_star^T * alpha
        mean = sum(k_star[i] * self._alpha[i] for i in range(n))

        # 方差 = k(x*,x*) - k_star^T * K^(-1) * k_star
        v = self._solve_triangular(self._K_inv_L, k_star, lower=True)
        k_star_star = self._kernel(x_star, x_star) + self._noise_variance
        var = k_star_star - sum(v[i] * v[i] for i in range(n))
        var = max(var, 1e-12)

        return mean, var

    # ---- 采集函数 ----

    def _acquisition_value(self, x: List[float], best_y: float) -> float:
        """计算采集函数值（越大越值得探索）。"""
        mean, var = self._predict(x)
        std = math.sqrt(var)

        if self._acquisition == "ucb":
            return mean + self._kappa * std
        elif self._acquisition == "ei":
            if std < 1e-9:
                return 0.0
            z = (mean - best_y - self._xi) / std
            # EI = (mean - best_y - xi)*Φ(z) + σ*φ(z)
            phi_z = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z * z)
            # standard normal CDF approximation
            phi_cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            return (mean - best_y - self._xi) * phi_cdf + std * phi_z
        elif self._acquisition == "pi":
            if std < 1e-9:
                return 0.0
            z = (mean - best_y - self._xi) / std
            return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        else:
            return mean + self._kappa * math.sqrt(var)

    # ---- 公共接口 ----

    async def suggest_next(self, n_candidates: int = 1000,
                           param_names: List[str] = None) -> Dict[str, Any]:
        """建议下一组参数值。使用随机搜索 + 采集函数最大化。

        Returns:
            {"params": {name: value}, "acquisition": float, "predicted_mean": float, "predicted_std": float}
        """
        async with self._lock:
            if not self._bounds:
                return {"error": "No bounds configured"}

            d = len(self._bounds)
            best_y = max(self._y) if self._y else 0.0

            best_acq = float("-inf")
            best_candidate = None

            # 生成候选点 + 已有的点 = 更多候选
            all_candidates = []
            for _ in range(n_candidates):
                point = [random.uniform(self._bounds[i][0], self._bounds[i][1]) for i in range(d)]
                all_candidates.append(point)

            # 如果已有点太少，加一些之前观测过的点附近
            if len(self._X) >= 3:
                for xi in self._X:
                    for _ in range(max(1, n_candidates // max(len(self._X), 1))):
                        perturbed = [
                            xi[j] + random.gauss(0, (self._bounds[j][1] - self._bounds[j][0]) * 0.05)
                            for j in range(d)
                        ]
                        # clamp
                        perturbed = [
                            max(self._bounds[j][0], min(self._bounds[j][1], perturbed[j]))
                            for j in range(d)
                        ]
                        all_candidates.append(perturbed)

            if self._fitted:
                for candidate in all_candidates:
                    acq = self._acquisition_value(candidate, best_y)
                    if acq > best_acq:
                        best_acq = acq
                        best_candidate = candidate
            else:
                # 尚未拟合：随机选一个
                best_candidate = all_candidates[0] if all_candidates else [
                    random.uniform(self._bounds[i][0], self._bounds[i][1]) for i in range(d)
                ]

            if best_candidate is None:
                best_candidate = [random.uniform(self._bounds[i][0], self._bounds[i][1]) for i in range(d)]

            mean, var = self._predict(best_candidate) if self._fitted else (0.0, 1.0)

            params_out = {}
            if param_names and len(param_names) == d:
                for idx, name in enumerate(param_names):
                    params_out[name] = best_candidate[idx]
            else:
                for idx in range(d):
                    params_out[f"param_{idx}"] = best_candidate[idx]

            return {
                "params": params_out,
                "acquisition": best_acq if self._fitted else 0.0,
                "predicted_mean": mean,
                "predicted_std": math.sqrt(max(var, 0.0)),
            }

    async def observe(self, params_dict: Dict[str, float], score: float) -> None:
        """记录一次观测结果，更新 GP 模型。

        Args:
            params_dict: 参数名 -> 参数值映射
            score: 性能得分（越高越好）
        """
        async with self._lock:
            if isinstance(params_dict, dict):
                # 按 bounds 顺序提取值（优先遵循 set_bounds 传入的参数名顺序，避免 dict 插入顺序错位）
                if self._bounds:
                    if (self._param_names
                            and len(self._param_names) == len(self._bounds)
                            and all(k in params_dict for k in self._param_names)):
                        values = [params_dict[k] for k in self._param_names]
                    else:
                        values = list(params_dict.values())
                    if len(values) != len(self._bounds):
                        logger.warning(f"BayesianOptimizer: param count {len(values)} != bounds count {len(self._bounds)}")
                    self._X.append(values[:len(self._bounds)])
                    self._y.append(score)
            elif isinstance(params_dict, list):
                self._X.append(list(params_dict))
                self._y.append(score)

            # 限制历史长度
            max_hist = self._config.get("gp_max_history", 500)
            if len(self._X) > max_hist:
                self._X = self._X[-max_hist:]
                self._y = self._y[-max_hist:]

            self._fit_gp()
            logger.debug(f"BayesianOptimizer observed: score={score:.4f}, n={len(self._X)}")

    async def get_best_params(self) -> Dict[str, Any]:
        """返回当前已知的最优参数。"""
        async with self._lock:
            if not self._y:
                return {"params": {}, "score": None}
            best_idx = max(range(len(self._y)), key=lambda i: self._y[i])
            params = {}
            for j, v in enumerate(self._X[best_idx]):
                params[f"param_{j}"] = v
            return {"params": params, "score": self._y[best_idx]}

    def set_bounds(self, bounds: List[Tuple[float, float]], param_names: List[str] = None):
        self._bounds = bounds
        self._param_names = list(param_names) if param_names else []


# =============================================================================
# 2. ResponseSurface — 多项式响应面建模
# =============================================================================

class ResponseSurface:
    """二阶响应面模型：score = b0 + Σ(bi*xi) + Σ(bij*xi*xj) + Σ(bii*xi²)

    通过正规方程拟合系数，并在曲面上执行梯度上升以定位最优点。
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._lock = asyncio.Lock()

        self._X: List[List[float]] = []     # n×d 设计矩阵（原始值）
        self._y: List[float] = []            # n 维目标值

        self._coeffs: List[float] = []       # 拟合系数
        self._coeff_names: List[str] = []    # 系数名称（调试用）
        self._d: int = 0                     # 参数维度
        self._fitted = False

        self._regularization = config.get("rs_regularization", 1e-4)

    # ---- 构建设计矩阵 ----

    def _build_design_matrix(self, X: List[List[float]]) -> List[List[float]]:
        """构建二阶设计矩阵（含交互项和平方项）。

        列顺序：[1, x1, x2, ..., xi*xj (i<j), x1², x2², ...]
        """
        n = len(X)
        d = len(X[0]) if X else 0
        # 项数：1 (intercept) + d (linear) + d*(d-1)//2 (interactions) + d (quadratic)
        p = 1 + d + d * (d - 1) // 2 + d

        D = [[0.0] * p for _ in range(n)]
        for i in range(n):
            col = 0
            # 截距
            D[i][col] = 1.0
            col += 1
            # 线性项
            for j in range(d):
                D[i][col] = X[i][j]
                col += 1
            # 交互项
            for j in range(d):
                for k in range(j + 1, d):
                    D[i][col] = X[i][j] * X[i][k]
                    col += 1
            # 平方项
            for j in range(d):
                D[i][col] = X[i][j] * X[i][j]
                col += 1
        return D

    def _build_coeff_names(self, d: int) -> List[str]:
        names = ["b0"]
        for j in range(d):
            names.append(f"b_{j}")
        for j in range(d):
            for k in range(j + 1, d):
                names.append(f"b_{j}{k}")
        for j in range(d):
            names.append(f"b_{j}²")
        return names

    # ---- 矩阵运算工具 ----

    def _mat_mul(self, A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        """矩阵乘法 C = A * B。"""
        m, n, p = len(A), len(A[0]) if A else 0, len(B[0]) if B else 0
        C = [[0.0] * p for _ in range(m)]
        for i in range(m):
            for k in range(n):
                aik = A[i][k]
                if aik != 0:
                    for j in range(p):
                        C[i][j] += aik * B[k][j]
        return C

    def _mat_vec_mul(self, A: List[List[float]], v: List[float]) -> List[float]:
        """矩阵-向量乘法 y = A * v。"""
        m = len(A)
        n = len(A[0]) if A else 0
        y = [0.0] * m
        for i in range(m):
            s = 0.0
            for j in range(n):
                s += A[i][j] * v[j]
            y[i] = s
        return y

    def _transpose(self, A: List[List[float]]) -> List[List[float]]:
        """矩阵转置。"""
        if not A:
            return []
        m, n = len(A), len(A[0])
        return [[A[i][j] for i in range(m)] for j in range(n)]

    def _solve_normal_equations(self, Xt: List[List[float]], XtX: List[List[float]],
                                 Xty: List[float]) -> List[float]:
        """解正规方程 (X^T X + λI) β = X^T y。

        使用带正则化的 Cholesky 分解。
        """
        p = len(XtX)
        # 加正则化
        A = [row[:] for row in XtX]
        for i in range(p):
            A[i][i] += self._regularization

        # Cholesky
        L = [[0.0] * p for _ in range(p)]
        for i in range(p):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    L[i][j] = math.sqrt(max(A[i][i] - s, 1e-12))
                else:
                    L[i][j] = (A[i][j] - s) / max(L[j][j], 1e-12)

        # 前代
        y_temp = [0.0] * p
        for i in range(p):
            s = sum(L[i][j] * y_temp[j] for j in range(i))
            y_temp[i] = (Xty[i] - s) / max(L[i][i], 1e-12)

        # 回代
        coeffs = [0.0] * p
        for i in range(p - 1, -1, -1):
            s = sum(L[j][i] * coeffs[j] for j in range(i + 1, p))
            coeffs[i] = (y_temp[i] - s) / max(L[i][i], 1e-12)

        return coeffs

    # ---- 拟合 ----

    async def fit(self) -> Dict[str, Any]:
        """拟合二阶响应面。"""
        async with self._lock:
            n = len(self._X)
            if n < 3:
                return {"status": "insufficient_data", "n": n}

            d = len(self._X[0]) if self._X else 0
            self._d = d

            D = self._build_design_matrix(self._X)
            Dt = self._transpose(D)
            DtD = self._mat_mul(Dt, D)
            Dty = self._mat_vec_mul(Dt, self._y)

            self._coeffs = self._solve_normal_equations(Dt, DtD, Dty)
            self._coeff_names = self._build_coeff_names(d)
            self._fitted = True

            # 计算 R²
            y_pred = self._mat_vec_mul(D, self._coeffs)
            ss_res = sum((self._y[i] - y_pred[i]) ** 2 for i in range(n))
            y_mean = statistics.mean(self._y)
            ss_tot = sum((yi - y_mean) ** 2 for yi in self._y)
            r_squared = 1.0 - ss_res / max(ss_tot, 1e-12)

            logger.info(f"ResponseSurface fitted: d={d}, n={n}, R²={r_squared:.4f}")
            return {"status": "fitted", "n": n, "d": d, "r_squared": r_squared}

    def _predict_single(self, x: List[float]) -> float:
        """单点预测。"""
        if not self._fitted:
            return 0.0
        d = len(x)
        terms = [1.0]
        terms.extend(x)
        for j in range(d):
            for k in range(j + 1, d):
                terms.append(x[j] * x[k])
        terms.extend([v * v for v in x])
        if len(terms) != len(self._coeffs):
            return 0.0
        return sum(terms[i] * self._coeffs[i] for i in range(len(self._coeffs)))

    async def predict(self, x: List[float]) -> float:
        async with self._lock:
            return self._predict_single(x)

    # ---- 梯度上升求最优 ----

    def _gradient(self, x: List[float]) -> List[float]:
        """计算响应面在 x 处的梯度。"""
        d = len(x)
        grad = [0.0] * d

        # 线性项系数：coeffs[1:1+d]
        for j in range(d):
            grad[j] += self._coeffs[1 + j]

        # 交互项系数：coeffs[1+d : 1+d+d*(d-1)//2]
        idx = 1 + d
        for j in range(d):
            for k in range(j + 1, d):
                c = self._coeffs[idx]
                grad[j] += c * x[k]
                grad[k] += c * x[j]
                idx += 1

        # 平方项系数：coeffs[1+d+d*(d-1)//2 :]
        idx_sq = 1 + d + d * (d - 1) // 2
        for j in range(d):
            grad[j] += 2.0 * self._coeffs[idx_sq + j] * x[j]

        return grad

    async def find_optimal(self, bounds: List[Tuple[float, float]] = None,
                           n_restarts: int = 10, n_iterations: int = 200,
                           lr: float = 0.01) -> Dict[str, Any]:
        """在响应面上通过梯度上升找到最优点。

        Args:
            bounds: 每个维度 (min, max) 的列表
            n_restarts: 随机重启次数
            n_iterations: 每次重启的梯度上升迭代数
            lr: 学习率
        """
        async with self._lock:
            if not self._fitted:
                return {"status": "not_fitted"}

            d = self._d
            if bounds is None:
                bounds = [(0.0, 1.0)] * d

            best_x = None
            best_y = float("-inf")

            for restart in range(n_restarts):
                # 随机初始点
                x = [random.uniform(bounds[j][0], bounds[j][1]) for j in range(d)]

                for it in range(n_iterations):
                    grad = self._gradient(x)
                    grad_norm = math.sqrt(sum(g * g for g in grad))
                    if grad_norm < 1e-8:
                        break

                    # 自适应学习率：RMSProp 风格
                    step = lr / math.sqrt(it + 1)

                    for j in range(d):
                        x[j] += step * grad[j]
                        x[j] = max(bounds[j][0], min(bounds[j][1], x[j]))

                y_val = self._predict_single(x)
                if y_val > best_y:
                    best_y = y_val
                    best_x = x[:]

            # 置信区间：用 Hessian 对角近似
            confidence = {}
            if best_x:
                ci = self._confidence_intervals(best_x)
                confidence = {"lower": ci[0], "upper": ci[1]}

            return {
                "status": "success",
                "optimal_params": best_x,
                "optimal_score": best_y,
                "confidence_intervals": confidence,
            }

    def _confidence_intervals(self, x: List[float]) -> Tuple[List[float], List[float]]:
        """简单的一维置信区间估计（基于 Hessian 对角）。"""
        d = len(x)
        # 二阶导：平方项的 2*coeff
        idx_sq = 1 + d + d * (d - 1) // 2
        lower = []
        upper = []
        for j in range(d):
            h_ii = 2.0 * self._coeffs[idx_sq + j]
            if abs(h_ii) < 1e-8:
                lower.append(x[j] - 0.1)
                upper.append(x[j] + 0.1)
            else:
                se = math.sqrt(1.0 / abs(h_ii)) * 1.96  # 95% CI
                lower.append(x[j] - se)
                upper.append(x[j] + se)
        return lower, upper

    async def add_observation(self, params: List[float], score: float) -> None:
        """添加观测数据点。"""
        async with self._lock:
            self._X.append(list(params))
            self._y.append(score)
            max_hist = self._config.get("rs_max_history", 1000)
            if len(self._X) > max_hist:
                self._X = self._X[-max_hist:]
                self._y = self._y[-max_hist:]

    async def get_slice(self, dim1: int, dim2: int,
                        fixed_values: Dict[int, float] = None,
                        n_points: int = 50) -> Dict[str, Any]:
        """获取二维切片数据（用于可视化）。

        Args:
            dim1: 第一个维度的索引
            dim2: 第二个维度的索引
            fixed_values: 固定维度的值 {dim_index: value}
            n_points: 每维采样点数
        """
        async with self._lock:
            if not self._fitted:
                return {"status": "not_fitted"}

            d = self._d
            # 构建默认固定值
            defaults = [0.0] * d
            if fixed_values:
                for k, v in fixed_values.items():
                    defaults[k] = v

            grid_x = []
            grid_y = []
            grid_z = []

            x_min, x_max = defaults[dim1] - 1.0, defaults[dim1] + 1.0
            y_min, y_max = defaults[dim2] - 1.0, defaults[dim2] + 1.0

            for i in range(n_points):
                row = []
                xi = x_min + (x_max - x_min) * i / (n_points - 1)
                grid_x.append(xi)
                y_row = []
                for j in range(n_points):
                    yi = y_min + (y_max - y_min) * j / (n_points - 1)
                    if i == 0:
                        grid_y.append(yi)
                    point = defaults[:]
                    point[dim1] = xi
                    point[dim2] = yi
                    y_row.append(self._predict_single(point))
                grid_z.append(y_row)

            return {
                "status": "success",
                "dim1": dim1,
                "dim2": dim2,
                "grid_x": grid_x,
                "grid_y": grid_y,
                "grid_z": grid_z,
            }

    def get_coefficients(self) -> Dict[str, float]:
        """获取拟合系数。"""
        if not self._fitted or not self._coeff_names:
            return {}
        return {name: val for name, val in zip(self._coeff_names, self._coeffs)}


# =============================================================================
# 3. SafetyBoundary — 安全边界约束
# =============================================================================

class SafetyBoundary:
    """参数安全边界管理。

    - 硬边界：绝不可越过的 min/max
    - 软边界：惩罚区域，接近边界时施加惩罚
    - 关联约束：某些参数对必须同步移动
    - 自动学习：从历史安全运行数据中学习边界
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._lock = asyncio.Lock()

        # 硬边界：{param_name: (min, max)}
        self._hard_bounds: Dict[str, Tuple[float, float]] = {}
        # 软边界：{param_name: (soft_min, soft_max)} — 惩罚在 [soft_min, hard_min) 和 (hard_max, soft_max]
        self._soft_bounds: Dict[str, Tuple[float, float]] = {}
        # 软边界惩罚系数
        self._soft_penalty = config.get("soft_penalty", 10.0)
        # 关联约束：[(param_a, param_b, corr_factor)]
        # corr_factor > 0: 同向移动；< 0: 反向移动
        self._correlations: List[Tuple[str, str, float]] = []
        # 最小关联比：abs(Δa/Δb - factor) 超过此值视为违规
        self._corr_tolerance = config.get("corr_tolerance", 0.2)

        # 历史安全运行数据：用于自动学习边界
        self._safe_history: Dict[str, List[float]] = {}
        self._auto_learn_enabled = config.get("auto_learn_boundaries", False)
        self._auto_learn_samples = config.get("auto_learn_min_samples", 50)

    async def set_hard_bound(self, param_name: str, min_val: float, max_val: float) -> None:
        async with self._lock:
            self._hard_bounds[param_name] = (min_val, max_val)

    async def set_soft_bound(self, param_name: str, soft_min: float, soft_max: float) -> None:
        async with self._lock:
            self._soft_bounds[param_name] = (soft_min, soft_max)

    async def add_correlation(self, param_a: str, param_b: str, corr_factor: float) -> None:
        """添加参数关联约束。

        corr_factor = 1.0 表示 a 和 b 必须 1:1 同步移动；
        corr_factor = -0.5 表示 a 升 1 时 b 应降 0.5。
        """
        async with self._lock:
            self._correlations.append((param_a, param_b, corr_factor))

    async def record_safe_operation(self, params: Dict[str, float]) -> None:
        """记录一次安全运行时的参数值，用于自动边界学习。"""
        async with self._lock:
            for name, value in params.items():
                if name not in self._safe_history:
                    self._safe_history[name] = []
                self._safe_history[name].append(value)
                max_hist = self._config.get("safe_history_max", 10000)
                if len(self._safe_history[name]) > max_hist:
                    self._safe_history[name] = self._safe_history[name][-max_hist:]

    async def learn_boundaries(self) -> Dict[str, Any]:
        """从历史安全数据自动学习硬边界和软边界。"""
        async with self._lock:
            if not self._auto_learn_enabled:
                return {"status": "auto_learn_disabled"}

            learned = {}
            for name, values in self._safe_history.items():
                if len(values) < self._auto_learn_samples:
                    continue

                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0.0

                # 硬边界：μ ± 5σ
                hard_min = mean_val - 5.0 * std_val
                hard_max = mean_val + 5.0 * std_val
                self._hard_bounds[name] = (hard_min, hard_max)

                # 软边界：μ ± 2σ
                soft_min = mean_val - 2.0 * std_val
                soft_max = mean_val + 2.0 * std_val
                self._soft_bounds[name] = (soft_min, soft_max)

                learned[name] = {
                    "mean": mean_val,
                    "std": std_val,
                    "hard": (hard_min, hard_max),
                    "soft": (soft_min, soft_max),
                }

            logger.info(f"SafetyBoundary learned boundaries for {len(learned)} parameters")
            return {"status": "learned", "parameters": learned}

    async def validate_params(self, params: Dict[str, float],
                              previous_params: Dict[str, float] = None) -> Dict[str, Any]:
        """检查参数是否在安全区域内。

        Returns:
            {
                "safe": bool,
                "violations": [{"param": str, "type": "hard"|"soft"|"correlation", "detail": str}],
                "penalty": float,
            }
        """
        async with self._lock:
            violations = []
            total_penalty = 0.0

            for name, value in params.items():
                # 硬边界检查
                if name in self._hard_bounds:
                    h_min, h_max = self._hard_bounds[name]
                    if value < h_min or value > h_max:
                        violations.append({
                            "param": name,
                            "type": "hard",
                            "detail": f"value {value:.4f} outside [{h_min:.4f}, {h_max:.4f}]",
                        })
                        total_penalty += 100.0

                # 软边界检查
                if name in self._soft_bounds:
                    s_min, s_max = self._soft_bounds[name]
                    if value < s_min:
                        dist = (s_min - value) / max(abs(s_min), 1e-8)
                        violations.append({
                            "param": name,
                            "type": "soft",
                            "detail": f"value {value:.4f} below soft min {s_min:.4f}",
                        })
                        total_penalty += self._soft_penalty * dist
                    elif value > s_max:
                        dist = (value - s_max) / max(abs(s_max), 1e-8)
                        violations.append({
                            "param": name,
                            "type": "soft",
                            "detail": f"value {value:.4f} above soft max {s_max:.4f}",
                        })
                        total_penalty += self._soft_penalty * dist

            # 关联约束检查（需要 previous_params）
            if previous_params:
                for param_a, param_b, factor in self._correlations:
                    if param_a in params and param_b in params:
                        if param_a in previous_params and param_b in previous_params:
                            da = params[param_a] - previous_params[param_a]
                            db = params[param_b] - previous_params[param_b]
                            if abs(da) > 1e-8 or abs(db) > 1e-8:
                                denom = max(abs(da), 1e-8)
                                actual_ratio = db / denom if abs(da) > abs(db) else da / max(abs(db), 1e-8)
                                diff = abs(actual_ratio - factor)
                                if diff > self._corr_tolerance:
                                    violations.append({
                                        "param": f"{param_a},{param_b}",
                                        "type": "correlation",
                                        "detail": f"ratio {actual_ratio:.3f} vs expected {factor:.3f}",
                                    })
                                    total_penalty += 10.0 * diff

            safe = len(violations) == 0
            return {
                "safe": safe,
                "violations": violations,
                "penalty": total_penalty,
            }

    async def project_to_safe(self, params: Dict[str, float]) -> Dict[str, float]:
        """将不安全参数投影回安全区域。

        先处理硬边界截断，再处理软边界惩罚性回拉。
        """
        async with self._lock:
            safe_params = dict(params)

            for name, value in list(safe_params.items()):
                # 硬边界：直接截断
                if name in self._hard_bounds:
                    h_min, h_max = self._hard_bounds[name]
                    safe_params[name] = max(h_min, min(h_max, value))

                # 软边界：线性插值回拉到 soft 边界
                if name in self._soft_bounds:
                    s_min, s_max = self._soft_bounds[name]
                    if safe_params[name] < s_min:
                        # 从 hard_min 回拉到 soft_min
                        h_min = self._hard_bounds.get(name, (s_min - 1.0, s_max + 1.0))[0]
                        t = (safe_params[name] - h_min) / max(s_min - h_min, 1e-8)
                        safe_params[name] = h_min + t * (s_min - h_min) * 0.5 + (1 - t) * (s_min - h_min) * 0.0
                        safe_params[name] = max(h_min, min(s_max, s_min))  # clamp at soft_min
                    elif safe_params[name] > s_max:
                        h_max = self._hard_bounds.get(name, (0.0, 0.0))[1]
                        safe_params[name] = min(s_max, safe_params[name])

            return safe_params

    async def get_safety_status(self) -> Dict[str, Any]:
        """获取当前安全边界状态。"""
        async with self._lock:
            return {
                "hard_bounds": {k: list(v) for k, v in self._hard_bounds.items()},
                "soft_bounds": {k: list(v) for k, v in self._soft_bounds.items()},
                "correlations": [{"a": a, "b": b, "factor": f} for a, b, f in self._correlations],
                "auto_learn_enabled": self._auto_learn_enabled,
                "safe_history_sizes": {k: len(v) for k, v in self._safe_history.items()},
            }


# =============================================================================
# 4. MultiObjectiveAdaptor — 多目标优化
# =============================================================================

class MultiObjectiveAdaptor:
    """多目标参数优化器。

    支持：
    - Pareto 前沿计算
    - 加权和标量化
    - Epsilon-约束法
    - 目标间权衡分析
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._lock = asyncio.Lock()

        # 目标定义：{name: {"direction": "maximize"|"minimize", "weight": float}}
        self._objectives: Dict[str, Dict[str, Any]] = {}

        # 观测数据：[{params: {...}, objectives: {name: value, ...}}]
        self._observations: List[Dict[str, Any]] = []

        # 默认标量化方法
        self._method = config.get("mo_method", "weighted_sum")  # "weighted_sum" | "epsilon_constraint"
        self._epsilon_values: Dict[str, float] = {}  # epsilon-约束法的阈值

    async def set_objectives(self, objectives: Dict[str, Dict[str, Any]]) -> None:
        """设置优化目标。

        objectives = {
            "sharpe": {"direction": "maximize", "weight": 1.0},
            "drawdown": {"direction": "minimize", "weight": -0.5},
            "win_rate": {"direction": "maximize", "weight": 0.3},
        }
        """
        async with self._lock:
            self._objectives = objectives

    async def add_observation(self, params: Dict[str, float],
                              objectives: Dict[str, float]) -> None:
        """记录一次多目标观测。"""
        async with self._lock:
            self._observations.append({
                "params": dict(params),
                "objectives": dict(objectives),
            })
            max_hist = self._config.get("mo_max_history", 500)
            if len(self._observations) > max_hist:
                self._observations = self._observations[-max_hist:]

    def _normalize_objective(self, name: str, value: float) -> float:
        """将目标值标准化为越大越好的方向。"""
        obj = self._objectives.get(name, {})
        direction = obj.get("direction", "maximize")
        if direction == "minimize":
            return -value
        return value

    def _denormalize_objective(self, name: str, value: float) -> float:
        """反向标准化。"""
        obj = self._objectives.get(name, {})
        direction = obj.get("direction", "maximize")
        if direction == "minimize":
            return -value
        return value

    def _is_dominated(self, a_values: List[float], b_values: List[float]) -> bool:
        """判断 a 是否被 b Pareto 支配（所有目标都 <= b，且至少一个严格 < b）。"""
        # 值已经标准化为越大越好
        all_le = True
        any_lt = False
        for av, bv in zip(a_values, b_values):
            if av > bv:
                all_le = False
                break
            if av < bv:
                any_lt = True
        return all_le and any_lt

    async def get_pareto_frontier(self) -> Dict[str, Any]:
        """计算 Pareto 前沿。

        Returns:
            {
                "frontier": [{"params": {}, "objectives": {}, "scalarized": float}],
                "n_total": int,
                "n_frontier": int,
            }
        """
        async with self._lock:
            if not self._observations:
                return {"frontier": [], "n_total": 0, "n_frontier": 0}

            obj_names = list(self._objectives.keys())
            if not obj_names:
                return {"frontier": [], "n_total": len(self._observations), "n_frontier": 0}

            n = len(self._observations)
            dominated = [False] * n

            # 标准化值
            norm_values = []
            for obs in self._observations:
                vals = [self._normalize_objective(name, obs["objectives"].get(name, 0.0))
                        for name in obj_names]
                norm_values.append(vals)

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if self._is_dominated(norm_values[i], norm_values[j]):
                        dominated[i] = True
                        break

            frontier = []
            for i in range(n):
                if not dominated[i]:
                    obs = self._observations[i]
                    scalarized = self._scalarize(obs["objectives"])
                    frontier.append({
                        "params": obs["params"],
                        "objectives": obs["objectives"],
                        "scalarized": scalarized,
                    })

            # 按标量化值降序排列
            frontier.sort(key=lambda x: x["scalarized"], reverse=True)

            return {
                "frontier": frontier,
                "n_total": n,
                "n_frontier": len(frontier),
            }

    def _scalarize(self, objectives: Dict[str, float]) -> float:
        """加权和标量化。"""
        total = 0.0
        for name, value in objectives.items():
            obj_cfg = self._objectives.get(name, {})
            weight = obj_cfg.get("weight", 1.0)
            normalized = self._normalize_objective(name, value)
            total += weight * normalized
        return total

    async def weighted_sum_optimize(self) -> Dict[str, Any]:
        """使用加权和法选出最优参数。"""
        async with self._lock:
            if not self._observations:
                return {"status": "no_data"}

            best_obs = None
            best_score = float("-inf")

            for obs in self._observations:
                score = self._scalarize(obs["objectives"])
                if score > best_score:
                    best_score = score
                    best_obs = obs

            return {
                "status": "success",
                "method": "weighted_sum",
                "best_score": best_score,
                "best_params": best_obs["params"] if best_obs else {},
                "best_objectives": best_obs["objectives"] if best_obs else {},
            }

    async def epsilon_constraint_optimize(self, constraints: Dict[str, float]) -> Dict[str, Any]:
        """Epsilon-约束法优化。

        约束条件：objective[name] >= epsilon 或 <= epsilon（取决于 direction）

        Args:
            constraints: {objective_name: threshold}
        """
        async with self._lock:
            if not self._observations:
                return {"status": "no_data"}

            # 确认主目标：取第一个未约束的目标，或权重最大的
            constrained_names = set(constraints.keys())
            primary_obj = None
            for name, cfg in self._objectives.items():
                if name not in constrained_names:
                    primary_obj = name
                    break
            if primary_obj is None and self._objectives:
                primary_obj = list(self._objectives.keys())[0]

            feasible = []
            for obs in self._observations:
                ok = True
                for c_name, threshold in constraints.items():
                    actual = obs["objectives"].get(c_name, 0.0)
                    direction = self._objectives.get(c_name, {}).get("direction", "maximize")
                    if direction == "maximize":
                        if actual < threshold:
                            ok = False
                            break
                    else:
                        if actual > threshold:
                            ok = False
                            break
                if ok:
                    feasible.append(obs)

            if not feasible:
                return {"status": "no_feasible_solution", "constraints": constraints}

            # 按主目标排序
            feasible.sort(
                key=lambda o: self._normalize_objective(primary_obj, o["objectives"].get(primary_obj, 0.0)),
                reverse=True,
            )

            best = feasible[0]
            return {
                "status": "success",
                "method": "epsilon_constraint",
                "primary_objective": primary_obj,
                "constraints": constraints,
                "n_feasible": len(feasible),
                "best_params": best["params"],
                "best_objectives": best["objectives"],
            }

    async def trade_off_analysis(self, obj_a: str, obj_b: str) -> Dict[str, Any]:
        """分析两个目标之间的权衡关系。

        Returns:
            {
                "correlation": float,  # 正相关/负相关
                "trade_off_rate": float,  # 平均边际替代率
                "pareto_trade_offs": [{"obj_a": float, "obj_b": float}],
            }
        """
        async with self._lock:
            if len(self._observations) < 2:
                return {"status": "insufficient_data"}

            vals_a = [obs["objectives"].get(obj_a, 0.0) for obs in self._observations]
            vals_b = [obs["objectives"].get(obj_b, 0.0) for obs in self._observations]

            n = len(vals_a)
            mean_a = statistics.mean(vals_a)
            mean_b = statistics.mean(vals_b)
            std_a = statistics.stdev(vals_a) if n > 1 else 1.0
            std_b = statistics.stdev(vals_b) if n > 1 else 1.0

            # Pearson 相关系数
            cov = sum((vals_a[i] - mean_a) * (vals_b[i] - mean_b) for i in range(n)) / n
            corr = cov / (std_a * std_b) if std_a > 0 and std_b > 0 else 0.0

            # 从 Pareto 前沿上计算边际替代率
            frontier_result = await self.get_pareto_frontier()
            frontier = frontier_result.get("frontier", [])
            trade_offs = []
            if len(frontier) >= 2:
                # 按 obj_a 排序
                sorted_frontier = sorted(frontier,
                                         key=lambda x: x["objectives"].get(obj_a, 0.0))
                for i in range(len(sorted_frontier) - 1):
                    a1 = sorted_frontier[i]["objectives"].get(obj_a, 0.0)
                    a2 = sorted_frontier[i + 1]["objectives"].get(obj_a, 0.0)
                    b1 = sorted_frontier[i]["objectives"].get(obj_b, 0.0)
                    b2 = sorted_frontier[i + 1]["objectives"].get(obj_b, 0.0)
                    da = a2 - a1
                    if abs(da) > 1e-8:
                        trade_offs.append({
                            "obj_a": a2,
                            "obj_b": b2,
                            "marginal_rate": (b2 - b1) / da,
                        })

            avg_rate = statistics.mean([t["marginal_rate"] for t in trade_offs]) if trade_offs else 0.0

            return {
                "status": "success",
                "correlation": corr,
                "trade_off_rate": avg_rate,
                "pareto_trade_offs": trade_offs,
            }


# =============================================================================
# 5. ExplorationScheduler — 探索-利用调度器
# =============================================================================

class ExplorationScheduler:
    """自适应探索-利用调度器。

    方法：
    - Epsilon-Greedy：ε 从高衰减到低
    - Thompson Sampling：从后验分布中采样
    - 探索预算管理
    - 基于不确定性的探索
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._lock = asyncio.Lock()

        # Epsilon-greedy
        self._epsilon = config.get("explore_epsilon_start", 0.5)
        self._epsilon_start = config.get("explore_epsilon_start", 0.5)
        self._epsilon_end = config.get("explore_epsilon_end", 0.05)
        self._epsilon_decay = config.get("explore_epsilon_decay", 0.995)   # 每步衰减因子
        self._epsilon_type = config.get("explore_decay_type", "exponential")  # "exponential" | "linear" | "adaptive"

        # 步数计数
        self._step_count = 0
        self._total_steps_planned = config.get("explore_total_steps", 10000)

        # 探索预算
        self._exploration_budget = config.get("explore_budget", 0.3)       # 初始探索预算占总步数比例
        self._exploration_spent = 0                                         # 已用探索步数

        # Thompson Sampling 参数（用于 Beta 分布的参数）
        # 每个参数臂有 {successes, failures, mean, variance}
        self._arms: Dict[str, Dict[str, Any]] = {}

        # 不确定性追踪
        self._uncertainty_threshold = config.get("explore_uncertainty_threshold", 0.5)

    async def should_explore(self) -> bool:
        """判断当前步是否应该探索。"""
        async with self._lock:
            self._step_count += 1

            # 检查探索预算
            budget_remaining = self._exploration_budget * self._total_steps_planned - self._exploration_spent
            if budget_remaining <= 0:
                return False

            # 衰减 epsilon
            if self._epsilon_type == "exponential":
                self._epsilon = max(self._epsilon_end, self._epsilon * self._epsilon_decay)
            elif self._epsilon_type == "linear":
                progress = min(1.0, self._step_count / self._total_steps_planned)
                self._epsilon = self._epsilon_start + (self._epsilon_end - self._epsilon_start) * progress
            elif self._epsilon_type == "adaptive":
                # 自适应：根据最近表现调整
                recent_success_rate = self._compute_recent_success_rate()
                self._epsilon = self._epsilon_end + (self._epsilon_start - self._epsilon_end) * (1.0 - recent_success_rate)

            if random.random() < self._epsilon:
                self._exploration_spent += 1
                return True
            return False

    def _compute_recent_success_rate(self) -> float:
        """计算最近各臂的平均成功率。"""
        if not self._arms:
            return 0.0
        rates = []
        for arm in self._arms.values():
            s = arm.get("successes", 0)
            f = arm.get("failures", 0)
            total = s + f
            if total > 0:
                rates.append(s / total)
        return statistics.mean(rates) if rates else 0.0

    async def thompson_sample(self, arm_names: List[str],
                               param_pool: Dict[str, List[float]] = None) -> Dict[str, Any]:
        """Thompson 采样：从每个臂的后验 Beta 分布采样，选最优。

        Args:
            arm_names: 候选臂名称列表
            param_pool: {arm_name: [candidate values]} 可选，提供候选参数值
        """
        async with self._lock:
            best_arm = None
            best_sample = float("-inf")

            for name in arm_names:
                arm = self._arms.get(name, {"successes": 1.0, "failures": 1.0})
                # Beta(α+1, β+1) — 加平滑
                alpha = arm.get("successes", 0.0) + 1.0
                beta_param = arm.get("failures", 0.0) + 1.0
                # 使用 Gamma 分布近似 Beta 采样
                sample = self._sample_beta(alpha, beta_param)
                if sample > best_sample:
                    best_sample = sample
                    best_arm = name

            selected_params = None
            if param_pool and best_arm and best_arm in param_pool:
                candidates = param_pool[best_arm]
                selected_params = random.choice(candidates)

            return {
                "selected_arm": best_arm,
                "sample_value": best_sample,
                "params": {best_arm: selected_params} if selected_params else {},
            }

    def _sample_beta(self, alpha: float, beta_param: float) -> float:
        """从 Beta(alpha, beta) 采样（使用 Gamma 分布近似）。"""
        # Gamma 采样：Marsaglia and Tsang 方法简化版
        x = self._sample_gamma(alpha)
        y = self._sample_gamma(beta_param)
        return x / (x + y) if (x + y) > 0 else 0.5

    def _sample_gamma(self, shape: float) -> float:
        """从 Gamma(shape, 1) 采样（简单近似）。"""
        if shape < 1.0:
            # 使用 Ahrens-Dieter 方法
            u = random.random()
            return self._sample_gamma(shape + 1.0) * (u ** (1.0 / shape)) if u > 0 else 0.0
        # Marsaglia-Tsang 方法
        d = shape - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        while True:
            x = random.gauss(0, 1)
            v = (1.0 + c * x) ** 3
            if v > 0:
                u = random.random()
                if u < 1.0 - 0.0331 * (x ** 4) or math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v)):
                    return d * v

    async def update_arm(self, arm_name: str, success: bool,
                         params: Dict[str, float] = None) -> None:
        """更新臂的 Beta 分布参数。"""
        async with self._lock:
            if arm_name not in self._arms:
                self._arms[arm_name] = {"successes": 0.0, "failures": 0.0, "mean": 0.0, "variance": 0.0,
                                         "params": {}}
            arm = self._arms[arm_name]
            if success:
                arm["successes"] += 1
            else:
                arm["failures"] += 1
            total = arm["successes"] + arm["failures"]
            arm["mean"] = arm["successes"] / total if total > 0 else 0.5
            arm["variance"] = (arm["mean"] * (1.0 - arm["mean"])) / (total + 1) if total > 1 else 0.25
            if params:
                arm["params"] = dict(params)

    async def should_explore_uncertainty(self, arm_name: str) -> bool:
        """基于不确定性的探索决策：如果 arm 方差高则探索。"""
        async with self._lock:
            arm = self._arms.get(arm_name, {})
            variance = arm.get("variance", 1.0)
            return variance > self._uncertainty_threshold

    async def get_exploration_status(self) -> Dict[str, Any]:
        """获取当前探索状态。"""
        async with self._lock:
            budget_total = self._exploration_budget * self._total_steps_planned
            budget_pct = self._exploration_spent / max(budget_total, 1) * 100

            return {
                "epsilon": self._epsilon,
                "epsilon_start": self._epsilon_start,
                "epsilon_end": self._epsilon_end,
                "decay_type": self._epsilon_type,
                "step_count": self._step_count,
                "exploration_spent": self._exploration_spent,
                "exploration_budget_pct": budget_pct,
                "num_arms": len(self._arms),
                "arms": {
                    name: {
                        "successes": a["successes"],
                        "failures": a["failures"],
                        "mean": a["mean"],
                        "variance": a["variance"],
                    }
                    for name, a in self._arms.items()
                },
            }

    def reset(self):
        self._step_count = 0
        self._exploration_spent = 0
        self._epsilon = self._epsilon_start
        self._arms.clear()


# =============================================================================
# 6. ParameterAdaptor — 主类（集成所有模块）
# =============================================================================

class ParameterAdaptor(EnterpriseServiceMixin):
    """参数自适应器主类。

    保留原有 API 兼容的同时，新增 BayesianOptimizer / ResponseSurface /
    SafetyBoundary / MultiObjectiveAdaptor / ExplorationScheduler 功能。
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config

        # 主配置节（回退到顶层 config）
        pa_cfg = config.get("parameter_adaptor", config)

        self._parameters: Dict[str, Dict[str, Any]] = {}
        self._parameter_history: Dict[str, List[Dict[str, Any]]] = {}
        self._max_history = pa_cfg.get("max_history", 100)
        self._adaptation_threshold = pa_cfg.get("adaptation_threshold", 0.1)
        self._min_samples = pa_cfg.get("min_samples", 20)

        # 线程安全锁
        self._lock = asyncio.Lock()

        # ---- 新模块 ----
        # BayesianOptimizer
        gp_cfg = pa_cfg.get("bayesian", {})
        gp_cfg.setdefault("bounds", [(0.0, 1.0)])
        self._bayesian = BayesianOptimizer(gp_cfg)
        self._bayesian_enabled = pa_cfg.get("bayesian_enabled", True)

        # ResponseSurface
        rs_cfg = pa_cfg.get("response_surface", {})
        self._response_surface = ResponseSurface(rs_cfg)

        # SafetyBoundary
        sb_cfg = pa_cfg.get("safety", {})
        self._safety = SafetyBoundary(sb_cfg)

        # MultiObjectiveAdaptor
        mo_cfg = pa_cfg.get("multi_objective", {})
        self._multi_objective = MultiObjectiveAdaptor(mo_cfg)

        # ExplorationScheduler
        es_cfg = pa_cfg.get("exploration", {})
        self._exploration = ExplorationScheduler(es_cfg)

        logger.info("ParameterAdaptor initialized with all modules")

    # =====================================================================
    # 原有公开 API（保持兼容）
    # =====================================================================

    async def start(self):
        """启动参数自适应器（兼容 scheduler 生命周期管理）"""
        logger.info("Parameter adaptor started")

    async def stop(self):
        """停止参数自适应器"""
        logger.info("Parameter adaptor stopped")

    async def adapt_parameters(self) -> List[Dict[str, Any]]:
        """批量自适应参数（供 scheduler 调用）

        P0: 使用各参数的 target_metric，让每个参数有自己的性能目标。
        """
        results = []
        for key, param in self._parameters.items():
            try:
                target = param.get("target_value", param.get("current_value", 0.0))
                result = await self.adapt(
                    param["strategy_name"],
                    param["param_name"],
                    performance_metric=target,
                )
                if result.get("status") not in ("no_adaptation_needed", "error"):
                    results.append(result)
            except Exception as e:
                self._handle_exception(
                    e, {"parameter": key}, module="ParameterAdaptor",
                    function="adapt_parameters", severity="low", category="parameter_adaptation",
                )
        return results

    def register_parameter(self, strategy_name: str, param_name: str, current_value: float,
                           min_value: float, max_value: float, step: float = 0.01):
        key = f"{strategy_name}_{param_name}"
        self._parameters[key] = {
            "strategy_name": strategy_name,
            "param_name": param_name,
            "current_value": current_value,
            "min_value": min_value,
            "max_value": max_value,
            "step": step,
            "target_value": current_value,
            "last_adapted": datetime.now(),
        }
        self._parameter_history[key] = []
        logger.info(f"Registered parameter: {key} = {current_value}")

    async def adapt(self, strategy_name: str, param_name: str, performance_metric: float,
                    target_metric: float = None) -> Dict[str, Any]:
        key = f"{strategy_name}_{param_name}"
        if key not in self._parameters:
            return {"error": f"Parameter {key} not registered"}

        param = self._parameters[key]
        current_value = param["current_value"]

        self._record_history(key, current_value, performance_metric)

        if target_metric is None:
            pa_cfg = self._config.get("parameter_adaptor", {})
            target_metric = pa_cfg.get("default_target", self._config.get("default_target", 0.0))

        error = performance_metric - target_metric
        abs_error = abs(error)

        if abs_error < self._adaptation_threshold:
            return {"status": "no_adaptation_needed", "error": error}

        history = self._parameter_history[key]
        if len(history) < self._min_samples:
            return {"status": "waiting_for_samples", "samples": len(history)}

        gradient = self._compute_gradient(key, history)

        if gradient == 0:
            step_direction = -1 if error < 0 else 1
        else:
            step_direction = -1 if gradient > 0 else 1

        adaptation_amount = param["step"] * step_direction * (abs_error / self._adaptation_threshold)
        new_value = current_value + adaptation_amount
        new_value = max(param["min_value"], min(param["max_value"], new_value))

        param["current_value"] = new_value
        param["target_value"] = new_value
        param["last_adapted"] = datetime.now()

        logger.info(f"Adapted parameter {key}: {current_value:.4f} -> {new_value:.4f} (error={error:.4f})")

        return {
            "status": "adapted",
            "param_name": param_name,
            "strategy_name": strategy_name,
            "old_value": current_value,
            "new_value": new_value,
            "error": error,
            "gradient": gradient,
        }

    def _compute_gradient(self, key: str, history: List[Dict[str, Any]]) -> float:
        values = [h["value"] for h in history]
        metrics = [h["metric"] for h in history]

        if len(values) < 2:
            return 0

        n = len(values)
        sum_x = sum(values)
        sum_y = sum(metrics)
        sum_xy = sum(x * y for x, y in zip(values, metrics))
        sum_x2 = sum(x ** 2 for x in values)

        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0

        gradient = (n * sum_xy - sum_x * sum_y) / denominator
        return gradient

    def _record_history(self, key: str, value: float, metric: float):
        entry = {
            "timestamp": datetime.now(),
            "value": value,
            "metric": metric,
        }
        self._parameter_history[key].append(entry)
        if len(self._parameter_history[key]) > self._max_history:
            self._parameter_history[key] = self._parameter_history[key][-self._max_history:]

    def get_parameter(self, strategy_name: str, param_name: str) -> Optional[Dict[str, Any]]:
        key = f"{strategy_name}_{param_name}"
        return self._parameters.get(key)

    def get_parameters(self, strategy_name: str = None) -> Dict[str, Any]:
        if strategy_name:
            return {k: v for k, v in self._parameters.items() if v["strategy_name"] == strategy_name}
        return self._parameters

    def get_parameter_history(self, strategy_name: str, param_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        key = f"{strategy_name}_{param_name}"
        return self._parameter_history.get(key, [])[-limit:]

    def get_adaptation_summary(self) -> Dict[str, Any]:
        summary = {
            "total_parameters": len(self._parameters),
            "parameters": {},
        }

        for key, param in self._parameters.items():
            history = self._parameter_history.get(key, [])
            if history:
                recent_metrics = [h["metric"] for h in history[-20:]]
                avg_metric = statistics.mean(recent_metrics)
                std_metric = statistics.stdev(recent_metrics) if len(recent_metrics) > 1 else 0
            else:
                avg_metric = 0
                std_metric = 0

            summary["parameters"][key] = {
                "current_value": param["current_value"],
                "target_value": param["target_value"],
                "min_value": param["min_value"],
                "max_value": param["max_value"],
                "last_adapted": param["last_adapted"].isoformat(),
                "avg_metric": avg_metric,
                "std_metric": std_metric,
                "history_length": len(history),
            }

        return summary

    # =====================================================================
    # 新增公开 API
    # =====================================================================

    # ---- Bayesian Optimization ----

    async def bayesian_suggest(self, n_candidates: int = 1000,
                               param_names: List[str] = None) -> Dict[str, Any]:
        """Bayesian GP: 建议下一个待评估的参数点。

        用于在参数空间中智能搜索，平衡探索与利用。
        """
        async with self._lock:
            # 自动从注册参数同步 bounds
            if self._parameters:
                bounds = []
                names = []
                for key, p in sorted(self._parameters.items()):
                    bounds.append((p["min_value"], p["max_value"]))
                    names.append(key)
                self._bayesian.set_bounds(bounds, names)
                if param_names is None:
                    param_names = names

            return await self._bayesian.suggest_next(
                n_candidates=n_candidates,
                param_names=param_names,
            )

    async def update_observation(self, params: Dict[str, float], score: float) -> Dict[str, Any]:
        """记录一次观测结果，同时更新 GP 模型和响应面。

        Args:
            params: {param_key: value}
            score: 性能得分（越高越好）
        """
        async with self._lock:
            # 同步 GP 边界与参数名顺序，确保 observe 按注册顺序（sorted key）提取值，
            # 避免调用方传入 dict 的插入顺序与 bounds 顺序不一致导致维度错位
            if self._parameters:
                bounds = []
                names = []
                for key, p in sorted(self._parameters.items()):
                    bounds.append((p["min_value"], p["max_value"]))
                    names.append(key)
                self._bayesian.set_bounds(bounds, names)

            # 更新 GP
            await self._bayesian.observe(params, score)

            # 更新响应面
            if self._parameters:
                # 按注册顺序提取参数值列表
                sorted_keys = sorted(self._parameters.keys())
                values = []
                for k in sorted_keys:
                    if k in params:
                        values.append(params[k])
                    else:
                        values.append(self._parameters[k]["current_value"])
                await self._response_surface.add_observation(values, score)

            # 记录到探索调度器
            for name, value in params.items():
                await self._exploration.update_arm(name, score > 0, {name: value})

            return {"status": "observed", "n_gp": len(self._bayesian._X)}

    # ---- Safety ----

    async def get_safety_status(self) -> Dict[str, Any]:
        """获取安全边界状态。"""
        return await self._safety.get_safety_status()