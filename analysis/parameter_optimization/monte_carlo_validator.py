"""
蒙特卡洛验证器 (Monte Carlo Validator)

参数优化的蒙特卡洛验证，用于评估参数稳健性：
  - 参数扰动测试：在最优参数周围采样，评估性能衰减
  - 收益率重采样：Bootstrap 收益序列，评估绩效稳定性
  - 随机噪声注入：在回测中注入随机噪声，测试抗噪性
  - 交叉验证：K-Fold 时间序列交叉验证
  - 概率分布拟合：最优参数的经验分布
  - 压力场景模拟：极端市场条件下的参数表现
"""
import asyncio
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Callable
import numpy as np
from loguru import logger

from analysis.parameter_optimization.genetic_optimizer import ParameterDef


def _safe_float(v: Any, default: float = 0.0) -> float:
    """安全转换数值，None/非数值/NaN/Inf 返回默认值。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if np.isfinite(f) else default


def _round_metric(v: Any, ndigits: int = 4) -> Any:
    """对数值指标四舍五入，非标量（如置信区间 list）原样保留。"""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        f = float(v)
        if not np.isfinite(f):
            return 0.0
        return round(f, ndigits)
    return v


class ValidationMethod(Enum):
    """验证方法"""
    PARAMETER_PERTURBATION = "parameter_perturbation"  # 参数扰动
    RETURN_BOOTSTRAP = "return_bootstrap"               # 收益率Bootstrap
    NOISE_INJECTION = "noise_injection"                 # 噪声注入
    CROSS_VAL = "cross_validation"                      # 交叉验证


@dataclass
class PerturbationSample:
    """单个扰动样本"""
    params: Dict[str, float]
    performance: float                   # 评估指标值
    is_worse: bool = False               # 是否比基准差
    degradation_pct: float = 0.0         # 衰减百分比


@dataclass
class BootstrapSample:
    """单个Bootstrap样本"""
    sample_index: int
    performance: float
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0


@dataclass
class MCValidationResult:
    """蒙特卡洛验证结果"""
    # 基准性能（最优参数在完整数据上的表现）
    baseline_performance: float = 0.0
    baseline_metrics: Dict[str, float] = field(default_factory=dict)

    # 参数扰动结果
    perturbation_samples: List[PerturbationSample] = field(default_factory=list)
    perturbation_stats: Dict[str, float] = field(default_factory=dict)

    # Bootstrap结果
    bootstrap_samples: List[BootstrapSample] = field(default_factory=list)
    bootstrap_stats: Dict[str, float] = field(default_factory=dict)

    # 噪声注入结果
    noise_injection_results: Dict[str, Any] = field(default_factory=dict)

    # 交叉验证结果
    cross_val_folds: List[Dict[str, float]] = field(default_factory=list)
    cross_val_stats: Dict[str, float] = field(default_factory=dict)

    # 综合评分
    robustness_index: float = 0.0        # 0-100 稳健性指数
    stability_score: float = 0.0         # 0-1 稳定性评分
    generalization_score: float = 0.0    # 0-1 泛化能力评分
    anti_noise_score: float = 0.0        # 0-1 抗噪评分
    consistency_score: float = 0.0       # 0-1 一致性评分

    # 结论
    is_robust: bool = True
    risk_level: str = "low"              # low / moderate / high
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)

    total_time_seconds: float = 0.0
    total_simulations: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": {
                "performance": round(self.baseline_performance, 6),
                "metrics": {k: _round_metric(v) for k, v in self.baseline_metrics.items()},
            },
            "perturbation": {
                "n_samples": len(self.perturbation_samples),
                "stats": {k: _round_metric(v) for k, v in self.perturbation_stats.items()},
            },
            "bootstrap": {
                "n_samples": len(self.bootstrap_samples),
                "stats": {k: _round_metric(v) for k, v in self.bootstrap_stats.items()},
            },
            "noise_injection": self.noise_injection_results,
            "cross_validation": {
                "n_folds": len(self.cross_val_folds),
                "stats": {k: _round_metric(v) for k, v in self.cross_val_stats.items()},
            },
            "scores": {
                "robustness_index": round(self.robustness_index, 2),
                "stability_score": round(self.stability_score, 4),
                "generalization_score": round(self.generalization_score, 4),
                "anti_noise_score": round(self.anti_noise_score, 4),
                "consistency_score": round(self.consistency_score, 4),
            },
            "verdict": {
                "is_robust": self.is_robust,
                "risk_level": self.risk_level,
                "summary": self.summary,
            },
            "recommendations": self.recommendations,
            "total_simulations": self.total_simulations,
            "total_time_seconds": round(self.total_time_seconds, 2),
            "timestamp": self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════
# 蒙特卡洛验证器
# ═══════════════════════════════════════════════════════════════

class MonteCarloValidator:
    """蒙特卡洛参数验证器"""

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config.get("monte_carlo_validator", {}) if config else {}
        # 扰动参数
        self._n_perturbation = cfg.get("n_perturbation_samples", 200)
        self._perturbation_std = cfg.get("perturbation_std", 0.15)    # 扰动标准差(归一化)
        # Bootstrap参数
        self._n_bootstrap = cfg.get("n_bootstrap_samples", 500)
        self._bootstrap_ratio = cfg.get("bootstrap_ratio", 0.7)       # 每次采样比例
        # 噪声注入
        self._n_noise_levels = cfg.get("n_noise_levels", 5)
        self._noise_std_range = cfg.get("noise_std_range", [0.001, 0.05])
        # 交叉验证
        self._n_cv_folds = cfg.get("n_cv_folds", 5)
        self._cv_min_train_ratio = cfg.get("cv_min_train_ratio", 0.5)
        # 阈值
        self._stability_threshold = cfg.get("stability_threshold", 0.7)
        self._degradation_threshold = cfg.get("degradation_threshold", 0.20)

        self._seed = cfg.get("seed", None)

        self._param_defs: List[ParameterDef] = []
        self._best_params: Dict[str, float] = {}
        self._eval_fn: Optional[Callable] = None    # eval_fn(params, data) -> float
        self._price_data: Optional[np.ndarray] = None
        self._returns: Optional[np.ndarray] = None

        logger.info(f"MonteCarloValidator initialized: perturb={self._n_perturbation}, "
                    f"bootstrap={self._n_bootstrap}, cv_folds={self._n_cv_folds}")

    def set_param_defs(self, param_defs: List[ParameterDef]):
        self._param_defs = param_defs

    def set_best_params(self, params: Dict[str, float]):
        self._best_params = dict(params)

    def set_eval_fn(self, fn: Callable[[Dict[str, float], np.ndarray], float]):
        """设置评估函数 fn(params, data) -> float"""
        self._eval_fn = fn

    def set_data(self, price_data: np.ndarray):
        self._price_data = price_data
        if price_data is None or len(price_data) < 2:
            self._returns = None
            return
        arr = np.asarray(price_data, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            rets = np.diff(arr) / arr[:-1]
        self._returns = rets[np.isfinite(rets)]

    # ── 参数扰动 ──────────────────────────────────────────────

    async def _run_parameter_perturbation(self) -> Tuple[List[PerturbationSample], Dict[str, float]]:
        """参数扰动测试：在最优参数附近加扰动，评估性能衰减"""
        samples = []
        base_perf = self._eval(self._best_params, self._price_data)
        if not np.isfinite(base_perf):
            base_perf = 0.0

        # 编码参数到 [0,1]
        best_arr = self._params_to_array(self._best_params)
        if len(best_arr) == 0:
            return [], {
                "mean_performance": 0.0,
                "std_performance": 0.0,
                "min_performance": 0.0,
                "max_performance": 0.0,
                "mean_degradation_pct": 0.0,
                "worse_than_baseline_ratio": 0.0,
                "performance_ci_95_low": 0.0,
                "performance_ci_95_high": 0.0,
            }

        for i in range(self._n_perturbation):
            # 从最优参数正态采样
            perturbed = best_arr + np.random.normal(0, self._perturbation_std, len(best_arr))
            perturbed = np.clip(perturbed, 0, 1)
            params = self._array_to_params(perturbed)

            try:
                perf = self._eval(params, self._price_data)
            except Exception:
                perf = float('-inf')

            if np.isfinite(perf):
                degradation = (base_perf - perf) / max(abs(base_perf), 1e-10)
            else:
                degradation = float('nan')
            samples.append(PerturbationSample(
                params=params,
                performance=perf,
                is_worse=perf < base_perf * 0.95,
                degradation_pct=degradation,
            ))

        # 统计
        perfs = np.array([s.performance for s in samples])
        perfs = perfs[np.isfinite(perfs)]
        degradations = [s.degradation_pct for s in samples if np.isfinite(s.degradation_pct)]
        worse_ratio = sum(1 for s in samples if s.is_worse) / max(len(samples), 1)

        stats = {
            "mean_performance": float(np.mean(perfs)) if len(perfs) > 0 else 0,
            "std_performance": float(np.std(perfs, ddof=1)) if len(perfs) > 1 else 0,
            "min_performance": float(np.min(perfs)) if len(perfs) > 0 else 0,
            "max_performance": float(np.max(perfs)) if len(perfs) > 0 else 0,
            "mean_degradation_pct": float(np.mean(degradations)) if degradations else 0,
            "worse_than_baseline_ratio": worse_ratio,
            "performance_ci_95_low": float(np.percentile(perfs, 2.5)) if len(perfs) > 0 else 0,
            "performance_ci_95_high": float(np.percentile(perfs, 97.5)) if len(perfs) > 0 else 0,
        }

        return samples, stats

    # ── Bootstrap ──────────────────────────────────────────────

    async def _run_bootstrap(self) -> Tuple[List[BootstrapSample], Dict[str, float]]:
        """收益率 Bootstrap：重采样收益序列评估稳定性"""
        if self._returns is None or len(self._returns) < 10:
            return [], {}

        samples = []
        n = len(self._returns)
        sample_size = max(10, int(n * self._bootstrap_ratio))
        sample_size = min(sample_size, n)

        for i in range(self._n_bootstrap):
            # 随机采样（保持时间序列结构）
            start = random.randint(0, max(0, n - sample_size))
            boot_returns = self._returns[start:start + sample_size]

            # 用这些收益评估最优参数
            prices = np.cumprod(1 + boot_returns)
            try:
                perf = self._eval(self._best_params, prices)
            except Exception:
                perf = 0

            sharpe = self._compute_sharpe(boot_returns)
            max_dd = self._compute_max_drawdown(boot_returns)
            wr = self._compute_win_rate(boot_returns)

            samples.append(BootstrapSample(
                sample_index=i,
                performance=perf,
                sharpe=sharpe,
                max_drawdown=max_dd,
                win_rate=wr,
            ))

        # 统计
        perfs = [s.performance for s in samples]
        perfs_finite = [p for p in perfs if np.isfinite(p)]
        sharpes = [s.sharpe for s in samples]
        sharpes_finite = [s for s in sharpes if np.isfinite(s)]
        max_dds = [s.max_drawdown for s in samples]
        max_dds_finite = [d for d in max_dds if np.isfinite(d)]

        stats = {
            "mean_performance": float(np.mean(perfs_finite)) if perfs_finite else 0,
            "std_performance": float(np.std(perfs_finite, ddof=1)) if len(perfs_finite) > 1 else 0,
            "performance_ci_95": [float(np.percentile(perfs_finite, 2.5)), float(np.percentile(perfs_finite, 97.5))]
                if len(perfs_finite) > 1 else [0, 0],
            "mean_sharpe": float(np.mean(sharpes_finite)) if sharpes_finite else 0,
            "std_sharpe": float(np.std(sharpes_finite, ddof=1)) if len(sharpes_finite) > 1 else 0,
            "worst_sharpe": float(np.min(sharpes_finite)) if sharpes_finite else 0,
            "mean_max_drawdown": float(np.mean(max_dds_finite)) if max_dds_finite else 0,
            "worst_max_drawdown": float(np.max(max_dds_finite)) if max_dds_finite else 0,
            "negative_performance_ratio": sum(1 for p in perfs if np.isfinite(p) and p < 0) / max(len(perfs), 1),
        }

        return samples, stats

    # ── 噪声注入 ──────────────────────────────────────────────

    async def _run_noise_injection(self) -> Dict[str, Any]:
        """噪声注入测试：在价格数据中注入随机噪声，评估抗噪性"""
        if self._price_data is None or len(self._price_data) < 5:
            return {}

        base_perf = self._eval(self._best_params, self._price_data)
        if not np.isfinite(base_perf):
            base_perf = 0.0
        levels_results = []

        for level in range(self._n_noise_levels):
            noise_std = self._noise_std_range[0] + level * (
                self._noise_std_range[1] - self._noise_std_range[0]
            ) / max(self._n_noise_levels - 1, 1)

            # 多次测试每个噪声水平
            perfs_at_level = []
            for _ in range(20):
                noisy_data = self._price_data.copy()
                noise = np.random.normal(0, noise_std, len(noisy_data))
                # 相对噪声
                noisy_data = noisy_data * (1 + noise)
                noisy_data = np.maximum(noisy_data, 1e-8)

                try:
                    perf = self._eval(self._best_params, noisy_data)
                    perfs_at_level.append(perf)
                except Exception:
                    perfs_at_level.append(float('-inf'))

            valid_perfs = [p for p in perfs_at_level if np.isfinite(p)]
            if valid_perfs:
                mean_perf = float(np.mean(valid_perfs))
                std_perf = float(np.std(valid_perfs, ddof=1)) if len(valid_perfs) > 1 else 0.0
                levels_results.append({
                    "noise_std": round(noise_std, 6),
                    "mean_performance": mean_perf,
                    "std_performance": std_perf,
                    "degradation_pct": (base_perf - mean_perf) / max(abs(base_perf), 1e-10),
                    "min_performance": float(np.min(valid_perfs)),
                })

        # 找到性能下降50%时的噪声水平（耐受度）
        tolerance_noise = None
        for lr in levels_results:
            if lr["degradation_pct"] > 0.5:
                tolerance_noise = lr["noise_std"]
                break

        return {
            "base_performance": base_perf,
            "noise_levels": levels_results,
            "tolerance_noise_std": tolerance_noise,
            "n_noise_levels_tested": self._n_noise_levels,
        }

    # ── 交叉验证 ──────────────────────────────────────────────

    async def _run_cross_validation(self) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
        """K-Fold时间序列交叉验证"""
        if self._n_cv_folds <= 0:
            return [], {}
        if self._price_data is None or len(self._price_data) < self._n_cv_folds * 10:
            return [], {}

        n = len(self._price_data)
        fold_size = n // self._n_cv_folds
        folds = []

        for fold in range(self._n_cv_folds):
            # 时间序列CV：训练集在验证集之前
            split_point = min(fold_size * (fold + 1), int(n * self._cv_min_train_ratio) + fold_size * fold)
            split_point = min(split_point, n - fold_size)

            train = self._price_data[:split_point]
            test = self._price_data[split_point:split_point + fold_size]

            train_perf = self._eval(self._best_params, train)
            test_perf = self._eval(self._best_params, test)

            if np.isfinite(train_perf) and np.isfinite(test_perf):
                ratio = test_perf / max(abs(train_perf), 1e-10)
            else:
                ratio = float('nan')

            folds.append({
                "fold": fold,
                "train_size": len(train),
                "test_size": len(test),
                "train_performance": train_perf,
                "test_performance": test_perf,
                "performance_ratio": ratio,
            })

        # 统计
        test_perfs = [f["test_performance"] for f in folds]
        ratios = [f["performance_ratio"] for f in folds]
        test_perfs = [p for p in test_perfs if np.isfinite(p)]
        ratios = [r for r in ratios if np.isfinite(r)]

        stats = {
            "mean_test_performance": float(np.mean(test_perfs)) if test_perfs else 0,
            "std_test_performance": float(np.std(test_perfs, ddof=1)) if len(test_perfs) > 1 else 0,
            "mean_train_test_ratio": float(np.mean(ratios)) if ratios else 0,
            "worst_fold_performance": float(np.min(test_perfs)) if test_perfs else 0,
            "best_fold_performance": float(np.max(test_perfs)) if test_perfs else 0,
        }

        return folds, stats

    # ── 辅助方法 ──────────────────────────────────────────────

    def _params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        arr = np.zeros(len(self._param_defs))
        for i, pd in enumerate(self._param_defs):
            v = params.get(pd.name, (pd.low + pd.high) / 2)
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = (pd.low + pd.high) / 2
            if not np.isfinite(v):
                v = (pd.low + pd.high) / 2
            if pd.high > pd.low:
                if pd.log_scale and pd.low > 0:
                    log_low = math.log10(pd.low)
                    log_high = math.log10(pd.high)
                    arr[i] = (math.log10(max(v, pd.low)) - log_low) / (log_high - log_low)
                else:
                    arr[i] = (v - pd.low) / (pd.high - pd.low)
        return np.clip(arr, 0, 1)

    def _array_to_params(self, arr: np.ndarray) -> Dict[str, float]:
        result = {}
        for i, pd in enumerate(self._param_defs):
            raw = arr[i] if i < len(arr) else 0.5
            if not np.isfinite(raw):
                raw = 0.5
            t = np.clip(raw, 0, 1)
            if pd.log_scale and pd.low > 0:
                log_low = math.log10(pd.low)
                log_high = math.log10(pd.high)
                v = 10 ** (log_low + t * (log_high - log_low))
            else:
                v = pd.low + t * (pd.high - pd.low)
            result[pd.name] = pd.clamp(v)
        return result

    def _eval(self, params: Dict[str, float], data: np.ndarray) -> float:
        if not self._eval_fn:
            return 0.0
        try:
            val = float(self._eval_fn(params, data))
        except Exception:
            return float('-inf')
        if not np.isfinite(val):
            return float('-inf')
        return val

    @staticmethod
    def _compute_sharpe(returns: np.ndarray) -> float:
        if returns is None:
            return 0.0
        rets = np.asarray(returns, dtype=float)
        rets = rets[np.isfinite(rets)]
        if len(rets) < 2:
            return 0.0
        mean = float(np.mean(rets))
        std = float(np.std(rets, ddof=1))
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            return 0.0
        return mean / std * math.sqrt(365)

    @staticmethod
    def _compute_max_drawdown(returns: np.ndarray) -> float:
        if returns is None:
            return 0.0
        rets = np.asarray(returns, dtype=float)
        rets = rets[np.isfinite(rets)]
        if len(rets) < 2:
            return 0.0
        cum = np.cumprod(1 + rets)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / np.maximum(peak, 1e-10)
        dd = dd[np.isfinite(dd)]
        if len(dd) == 0:
            return 0.0
        return float(abs(np.min(dd)))

    @staticmethod
    def _compute_win_rate(returns: np.ndarray) -> float:
        if returns is None:
            return 0.0
        rets = np.asarray(returns, dtype=float)
        rets = rets[np.isfinite(rets)]
        if len(rets) == 0:
            return 0.0
        return float(np.sum(rets > 0) / len(rets))

    # ── 综合评分 ──────────────────────────────────────────────

    def _compute_scores(self, result: MCValidationResult):
        """计算综合稳健性评分"""
        # 1. 稳定性评分（扰动测试）
        worse_ratio = result.perturbation_stats.get("worse_than_baseline_ratio", 1.0)
        if worse_ratio is None or not np.isfinite(worse_ratio):
            worse_ratio = 1.0
        result.stability_score = float(np.clip(1.0 - worse_ratio / 0.5, 0.0, 1.0))

        # 2. 泛化能力评分（交叉验证）
        cv_ratio = result.cross_val_stats.get("mean_train_test_ratio", 0)
        if cv_ratio is None or not np.isfinite(cv_ratio):
            cv_ratio = 0.0
        result.generalization_score = float(np.clip(cv_ratio, 0.0, 1.0))

        # 3. 抗噪评分
        tolerance = result.noise_injection_results.get("tolerance_noise_std", 0)
        noise_range = self._noise_std_range[1] - self._noise_std_range[0]
        if tolerance is None or not np.isfinite(tolerance):
            result.anti_noise_score = 0.0
        else:
            result.anti_noise_score = float(np.clip(tolerance / max(noise_range, 1e-6), 0.0, 1.0))

        # 4. 一致性评分（Bootstrap）
        bs_negative = result.bootstrap_stats.get("negative_performance_ratio", 1.0)
        if bs_negative is None or not np.isfinite(bs_negative):
            bs_negative = 1.0
        result.consistency_score = float(np.clip(1.0 - bs_negative, 0.0, 1.0))

        # 综合稳健性指数 (0-100)
        result.robustness_index = float(np.clip(
            result.stability_score * 30 +
            result.generalization_score * 25 +
            result.anti_noise_score * 20 +
            result.consistency_score * 25,
            0.0, 100.0,
        ))

        # 风险评估
        if result.robustness_index >= 70:
            result.risk_level = "low"
            result.is_robust = True
            result.summary = f"参数稳健 (指数={result.robustness_index:.0f}/100)"
        elif result.robustness_index >= 50:
            result.risk_level = "moderate"
            result.is_robust = True
            result.summary = f"参数中等稳健 (指数={result.robustness_index:.0f}/100)"
        else:
            result.risk_level = "high"
            result.is_robust = False
            result.summary = f"参数不够稳健 (指数={result.robustness_index:.0f}/100), 建议扩大搜索范围"

        # 建议
        if result.stability_score < 0.5:
            result.recommendations.append("参数稳定性低：小扰动导致大性能波动，建议缩小搜索范围")
        if result.generalization_score < 0.6:
            result.recommendations.append("泛化能力弱：训练/测试表现差异大，建议增加正则化或减少参数数量")
        if result.anti_noise_score < 0.5:
            result.recommendations.append("抗噪能力差：建议在回测中加入噪声过滤或使用更稳健的指标")

    # ── 主验证流程 ────────────────────────────────────────────

    async def validate(self, methods: List[ValidationMethod] = None) -> MCValidationResult:
        """执行蒙特卡洛验证"""
        if not self._best_params:
            raise ValueError("No best parameters set")
        if not self._eval_fn:
            raise ValueError("No evaluation function set")
        if self._price_data is None or len(self._price_data) == 0:
            raise ValueError("No price data set")

        methods = methods or list(ValidationMethod)
        start_time = time.time()

        # 确定性：设置随机种子
        if self._seed is not None:
            random.seed(self._seed)
            np.random.seed(self._seed)

        result = MCValidationResult()

        # 基准性能
        result.baseline_performance = self._eval(self._best_params, self._price_data)
        if not np.isfinite(result.baseline_performance):
            result.baseline_performance = 0.0
            result.baseline_metrics = {"evaluation": 0.0}
            result.is_robust = False
            result.risk_level = "high"
            result.summary = "基线评估失败：评估函数返回非法值"
            result.total_time_seconds = time.time() - start_time
            logger.warning("MC validation baseline returned non-finite value; failing closed")
            return result
        result.baseline_metrics = {
            "evaluation": result.baseline_performance,
        }
        logger.info(f"MC validation baseline: {result.baseline_performance:.4f}")

        total_sims = 0

        # 1. 参数扰动
        if ValidationMethod.PARAMETER_PERTURBATION in methods:
            logger.info(f"Running parameter perturbation ({self._n_perturbation} samples)...")
            samples, stats = await self._run_parameter_perturbation()
            result.perturbation_samples = samples
            result.perturbation_stats = stats
            total_sims += len(samples)

        # 2. Bootstrap
        if ValidationMethod.RETURN_BOOTSTRAP in methods:
            logger.info(f"Running return bootstrap ({self._n_bootstrap} samples)...")
            samples, stats = await self._run_bootstrap()
            result.bootstrap_samples = samples
            result.bootstrap_stats = stats
            total_sims += len(samples)

        # 3. 噪声注入
        if ValidationMethod.NOISE_INJECTION in methods:
            logger.info(f"Running noise injection ({self._n_noise_levels} levels)...")
            noise_results = await self._run_noise_injection()
            result.noise_injection_results = noise_results
            total_sims += len(noise_results.get("noise_levels", [])) * 20

        # 4. 交叉验证
        if ValidationMethod.CROSS_VAL in methods:
            logger.info(f"Running cross-validation ({self._n_cv_folds} folds)...")
            folds, stats = await self._run_cross_validation()
            result.cross_val_folds = folds
            result.cross_val_stats = stats
            total_sims += len(folds)

        # 综合评分
        self._compute_scores(result)
        result.total_simulations = total_sims
        result.total_time_seconds = time.time() - start_time

        logger.info(f"MC validation complete: robustness={result.robustness_index:.1f}/100, "
                    f"risk={result.risk_level}, sims={total_sims}")

        return result

    def get_status(self) -> Dict[str, Any]:
        return {
            "n_perturbation": self._n_perturbation,
            "n_bootstrap": self._n_bootstrap,
            "n_noise_levels": self._n_noise_levels,
            "n_cv_folds": self._n_cv_folds,
            "params_count": len(self._param_defs),
            "data_length": len(self._price_data) if self._price_data is not None else 0,
        }
