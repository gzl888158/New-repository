"""
在线学习器 - 支持增量学习、概念漂移检测、特征重要性追踪和预测校准

核心能力:
  - SampleBuffer: 多维样本采集与特征提取
  - DriftDetector: KS-Test / PSI / ADWIN 三种概念漂移检测
  - IncrementalModel: SGD 在线模型（线性回归/逻辑回归/岭回归）
  - 特征重要性追踪（permutation importance）
  - 预测校准（Online Platt Scaling + ECE）
"""
import asyncio
import math
import statistics
import time
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
from loguru import logger

from app.services.enterprise import EnterpriseServiceMixin


# =============================================================================
# Enums
# =============================================================================

class LearningMode(Enum):
    ONLINE = "online"
    BATCH = "batch"
    HYBRID = "hybrid"


class LearningStatus(Enum):
    IDLE = "idle"
    LEARNING = "learning"
    UPDATING = "updating"
    PAUSED = "paused"


class DriftSeverity(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ModelType(Enum):
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    RIDGE_REGRESSION = "ridge_regression"


class LRSchedule(Enum):
    EXPONENTIAL_DECAY = "exponential_decay"
    STEP_DECAY = "step_decay"
    COSINE_ANNEALING = "cosine_annealing"
    CONSTANT = "constant"


# =============================================================================
# SampleBuffer
# =============================================================================

class SampleBuffer:
    """多维样本缓冲区，支持自动特征提取和样本加权"""

    def __init__(self, max_size: int = 5000, weight_method: str = "time_decay",
                 decay_factor: float = 0.99):
        self._max_size = max_size
        self._weight_method = weight_method
        self._decay_factor = decay_factor
        self._samples: deque = deque(maxlen=max_size)
        self._total_added = 0

    def add(self, market_features: Dict[str, float],
            strategy_features: Dict[str, float],
            predictions: Dict[str, float],
            actuals: Dict[str, float],
            outcome: Dict[str, Any],
            timestamp: datetime = None,
            weight: float = None):
        """添加样本"""
        ts = timestamp or datetime.now()
        if weight is None:
            if self._weight_method == "time_decay":
                weight = self._compute_time_weight(ts)
            else:
                weight = 1.0

        sample = {
            "market_features": dict(market_features),
            "strategy_features": dict(strategy_features),
            "predictions": dict(predictions),
            "actuals": dict(actuals),
            "outcome": dict(outcome),
            "timestamp": ts,
            "weight": weight,
            "_index": self._total_added,
        }
        self._samples.append(sample)
        self._total_added += 1

    def _compute_time_weight(self, ts: datetime) -> float:
        """基于时间的指数衰减权重（越新的样本权重越高）。"""
        if not self._samples:
            return 1.0
        latest_ts = self._samples[-1]["timestamp"]
        # 以小时为时间单位做指数衰减，避免浮点下溢
        delta_hours = max(0.0, (latest_ts - ts).total_seconds() / 3600.0)
        return self._decay_factor ** delta_hours

    @staticmethod
    def extract_market_features(market_data: Dict[str, Any]) -> Dict[str, float]:
        """从市场数据自动提取特征"""
        features: Dict[str, float] = {}

        # 价格变化率
        if "price" in market_data:
            features["price"] = float(market_data["price"])
        if "price_change_pct" in market_data:
            features["price_change_pct"] = float(market_data["price_change_pct"])
        elif "open_24h" in market_data and "last" in market_data:
            open_24h = float(market_data["open_24h"])
            last = float(market_data["last"])
            if open_24h != 0:
                features["price_change_pct"] = (last - open_24h) / open_24h * 100

        # 成交量
        if "volume_24h" in market_data:
            features["volume_24h"] = float(market_data["volume_24h"])
        if "volume" in market_data:
            features["volume"] = float(market_data["volume"])

        # 波动率
        if "volatility" in market_data:
            features["volatility"] = float(market_data["volatility"])
        if "high_24h" in market_data and "low_24h" in market_data:
            high = float(market_data["high_24h"])
            low = float(market_data["low_24h"])
            if low != 0:
                features["daily_range"] = (high - low) / low

        # 价差
        if "spread" in market_data:
            features["spread"] = float(market_data["spread"])
        if "bid" in market_data and "ask" in market_data:
            mid = (float(market_data["bid"]) + float(market_data["ask"])) / 2
            if mid != 0:
                features["spread_pct"] = (float(market_data["ask"]) - float(market_data["bid"])) / mid * 100

        # 流动性指标
        if "turnover" in market_data:
            features["turnover"] = float(market_data["turnover"])

        # 技术指标
        for key in ("rsi", "macd", "macd_signal", "bb_upper", "bb_lower", "bb_mid",
                     "ema_12", "ema_26", "sma_20", "sma_50"):
            if key in market_data:
                features[key] = float(market_data[key])

        return features

    def get_recent(self, n: int = None) -> List[Dict[str, Any]]:
        """获取最近的 n 条样本"""
        if n is None:
            return list(self._samples)
        return list(self._samples)[-n:]

    def get_weighted_features(self, n: int = None) -> List[List[float]]:
        """获取带权重的特征向量列表（合并 market + strategy features）"""
        samples = self.get_recent(n)
        result = []
        for s in samples:
            vec = []
            for d in (s["market_features"], s["strategy_features"]):
                for k in sorted(d.keys()):
                    vec.append(d[k])
            result.append(vec)
        return result

    def get_feature_names(self) -> List[str]:
        """获取所有特征名（有序）"""
        if not self._samples:
            return []
        names = []
        for d in (self._samples[-1]["market_features"], self._samples[-1]["strategy_features"]):
            for k in sorted(d.keys()):
                names.append(k)
        return names

    def get_feature_matrix(self, n: int = None) -> Tuple[List[List[float]], List[float], List[float]]:
        """
        提取训练数据矩阵
        Returns:
            X: 特征矩阵 [n_samples, n_features]
            y: 实际值列表 [n_samples]
            weights: 样本权重 [n_samples]
        """
        samples = self.get_recent(n)
        if not samples:
            return [], [], []

        feature_names = self.get_feature_names()
        fn_map = {name: i for i, name in enumerate(feature_names)}

        X = []
        y = []
        weights = []

        for s in samples:
            vec = [0.0] * len(feature_names)
            for d in (s["market_features"], s["strategy_features"]):
                for k, v in d.items():
                    if k in fn_map:
                        vec[fn_map[k]] = float(v)
            X.append(vec)

            # 使用第一个 actual 值作为目标（可根据需要扩展为多输出）
            if s["actuals"]:
                first_key = next(iter(s["actuals"]))
                y.append(float(s["actuals"][first_key]))
            else:
                y.append(0.0)

            weights.append(s["weight"])

        return X, y, weights

    def __len__(self) -> int:
        return len(self._samples)

    def clear(self):
        self._samples.clear()
        self._total_added = 0


# =============================================================================
# DriftDetector
# =============================================================================

class ADWIN:
    """Adaptive Windowing algorithm — 自适应窗口变化检测"""

    def __init__(self, delta: float = 0.002, min_window: int = 30, max_window: int = 1000):
        self._delta = delta
        self._min_window = min_window
        self._max_window = max_window
        self._window: deque = deque()
        self._sum = 0.0
        self._drift_count = 0

    def add(self, value: float) -> bool:
        """添加观测值，返回是否检测到 drift"""
        self._window.append(value)
        self._sum += value

        if len(self._window) > self._max_window:
            removed = self._window.popleft()
            self._sum -= removed

        if len(self._window) < self._min_window * 2:
            return False

        n = len(self._window)
        # Hoeffding bound: epsilon = sqrt(1/(2m) * ln(2/delta))
        # 检查所有可能的分割点
        for split in range(self._min_window, n - self._min_window):
            # 前半部分
            w1 = list(self._window)[:split]
            mu1 = sum(w1) / split
            # 后半部分
            w2 = list(self._window)[split:]
            mu2 = sum(w2) / (n - split)

            abs_diff = abs(mu1 - mu2)
            m = 1.0 / (1.0 / split + 1.0 / (n - split))
            epsilon = math.sqrt(1.0 / (2.0 * m) * math.log(2.0 / max(self._delta, 1e-15)))

            if abs_diff > epsilon:
                # Drift detected: 丢弃前半部分数据
                for _ in range(split):
                    removed = self._window.popleft()
                    self._sum -= removed
                self._drift_count += 1
                return True

        return False

    @property
    def window_size(self) -> int:
        return len(self._window)

    @property
    def drift_count(self) -> int:
        return self._drift_count

    def reset(self):
        self._window.clear()
        self._sum = 0.0


class DriftDetector:
    """概念漂移检测器 — 集成 KS-Test、PSI、ADWIN"""

    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("drift_detection", {})

        # KS-Test 配置
        ks_cfg = cfg.get("ks_test", {})
        self._ks_alpha = ks_cfg.get("alpha", 0.05)
        self._ks_window = ks_cfg.get("window_size", 200)

        # PSI 配置
        psi_cfg = cfg.get("psi", {})
        self._psi_threshold = psi_cfg.get("threshold", 0.25)
        self._psi_n_bins = psi_cfg.get("n_bins", 10)

        # ADWIN 配置
        adwin_cfg = cfg.get("adwin", {})
        delta = adwin_cfg.get("delta", 0.002)
        self._adwin = ADWIN(delta=delta)

        # 状态
        self._reference_window: List[float] = []
        self._current_window: List[float] = []
        self._last_psi: float = 0.0
        self._last_ks_stat: float = 0.0
        self._last_ks_pvalue: float = 1.0
        self._drift_history: List[Dict[str, Any]] = []

    def update(self, error: float, features: Dict[str, float] = None) -> Dict[str, Any]:
        """
        更新检测器并返回检测结果
        Returns:
            Dict with keys: drift_detected, severity, methods, details
        """
        methods_detected: List[str] = []
        details: Dict[str, Any] = {}
        max_severity = DriftSeverity.NONE

        # 维护误差窗口
        self._current_window.append(error)
        if len(self._current_window) > self._ks_window:
            self._current_window = self._current_window[-self._ks_window:]

        # 1) KS-Test
        ks_result = self._run_ks_test()
        details["ks_test"] = ks_result
        if ks_result.get("drift_detected"):
            methods_detected.append("ks_test")
            max_severity = self._max_severity(max_severity, ks_result["severity"])

        # 2) PSI
        if features is not None:
            psi_result = self._run_psi(features)
            details["psi"] = psi_result
            if psi_result.get("drift_detected"):
                methods_detected.append("psi")
                max_severity = self._max_severity(max_severity, psi_result["severity"])

        # 3) ADWIN
        adwin_detected = self._adwin.add(error)
        details["adwin"] = {
            "drift_detected": adwin_detected,
            "window_size": self._adwin.window_size,
            "total_drifts": self._adwin.drift_count,
        }
        if adwin_detected:
            methods_detected.append("adwin")
            max_severity = self._max_severity(max_severity,
                DriftSeverity.HIGH if self._adwin.drift_count > 3 else DriftSeverity.MEDIUM)

        # 每 ks_window 次切换到 reference window
        if len(self._current_window) >= self._ks_window:
            self._reference_window = list(self._current_window)
            self._current_window = []

        result = {
            "drift_detected": len(methods_detected) > 0,
            "severity": max_severity,
            "methods": methods_detected,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        }
        if result["drift_detected"]:
            self._drift_history.append(result)
            if len(self._drift_history) > 100:
                self._drift_history = self._drift_history[-100:]
        return result

    def _run_ks_test(self) -> Dict[str, Any]:
        """两样本 Kolmogorov-Smirnov 检验"""
        if len(self._reference_window) < 30 or len(self._current_window) < 30:
            return {"drift_detected": False, "statistic": 0.0, "p_value": 1.0,
                    "severity": DriftSeverity.NONE, "insufficient_data": True}

        ref = sorted(self._reference_window)
        cur = sorted(self._current_window)

        # 计算经验 CDF 最大差异
        n1, n2 = len(ref), len(cur)
        max_diff = 0.0
        i, j = 0, 0
        while i < n1 and j < n2:
            if ref[i] <= cur[j]:
                i += 1
            else:
                j += 1
            diff = abs(i / n1 - j / n2)
            if diff > max_diff:
                max_diff = diff

        # 近似 p-value
        n_eff = (n1 * n2) / (n1 + n2)
        lambda_stat = (math.sqrt(n_eff) + 0.12 + 0.11 / math.sqrt(n_eff)) * max_diff
        # Kolmogorov 分布近似
        p_value = 2.0 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lambda_stat * lambda_stat)
                            for k in range(1, 100))
        p_value = max(0.0, min(1.0, p_value))

        self._last_ks_stat = max_diff
        self._last_ks_pvalue = p_value
        drift_detected = p_value < self._ks_alpha

        severity = DriftSeverity.NONE
        if drift_detected:
            if p_value < 0.001:
                severity = DriftSeverity.CRITICAL
            elif p_value < 0.01:
                severity = DriftSeverity.HIGH
            elif p_value < 0.05:
                severity = DriftSeverity.MEDIUM
            else:
                severity = DriftSeverity.LOW

        return {
            "drift_detected": drift_detected,
            "statistic": round(max_diff, 6),
            "p_value": round(p_value, 6),
            "severity": severity,
        }

    def _run_psi(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Population Stability Index — 特征分布稳定性检测"""
        values = [v for v in features.values() if isinstance(v, (int, float))
                  and math.isfinite(v) and abs(v) < 1e10]
        if len(values) < 10:
            return {"drift_detected": False, "psi": 0.0,
                    "severity": DriftSeverity.NONE, "insufficient_data": True}

        v_min = min(values)
        v_max = max(values)
        if v_max == v_min:
            return {"drift_detected": False, "psi": 0.0,
                    "severity": DriftSeverity.NONE}

        bin_width = (v_max - v_min) / self._psi_n_bins
        bin_edges = [v_min + i * bin_width for i in range(self._psi_n_bins + 1)]

        # 使用均匀分布作为 reference 分布（实际使用时可对比历史分布）
        expected_ratio = 1.0 / self._psi_n_bins
        actual_counts = [0] * self._psi_n_bins

        for v in values:
            for i in range(self._psi_n_bins):
                if i == self._psi_n_bins - 1:
                    if bin_edges[i] <= v <= bin_edges[i + 1]:
                        actual_counts[i] += 1
                        break
                else:
                    if bin_edges[i] <= v < bin_edges[i + 1]:
                        actual_counts[i] += 1
                        break

        total = len(values)
        psi = 0.0
        for i in range(self._psi_n_bins):
            actual_ratio = actual_counts[i] / total if total > 0 else 0
            actual_ratio = max(actual_ratio, 0.001)  # 避免除零
            expected = expected_ratio
            expected = max(expected, 0.001)
            psi += (actual_ratio - expected) * math.log(actual_ratio / expected)

        self._last_psi = psi
        drift_detected = psi > self._psi_threshold

        severity = DriftSeverity.NONE
        if drift_detected:
            if psi > 0.5:
                severity = DriftSeverity.CRITICAL
            elif psi > 0.35:
                severity = DriftSeverity.HIGH
            elif psi > 0.25:
                severity = DriftSeverity.MEDIUM
            else:
                severity = DriftSeverity.LOW

        return {
            "drift_detected": drift_detected,
            "psi": round(psi, 6),
            "severity": severity,
            "threshold": self._psi_threshold,
        }

    @staticmethod
    def _max_severity(a: DriftSeverity, b: DriftSeverity) -> DriftSeverity:
        order = {DriftSeverity.NONE: 0, DriftSeverity.LOW: 1, DriftSeverity.MEDIUM: 2,
                 DriftSeverity.HIGH: 3, DriftSeverity.CRITICAL: 4}
        return a if order.get(a, 0) >= order.get(b, 0) else b

    def get_status(self) -> Dict[str, Any]:
        return {
            "last_ks_statistic": self._last_ks_stat,
            "last_ks_pvalue": self._last_ks_pvalue,
            "last_psi": self._last_psi,
            "adwin_window_size": self._adwin.window_size,
            "adwin_drift_count": self._adwin.drift_count,
            "drift_history_len": len(self._drift_history),
        }

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self._drift_history[-limit:]

    def reset(self):
        self._reference_window = []
        self._current_window = []
        self._adwin.reset()
        self._last_psi = 0.0
        self._last_ks_stat = 0.0
        self._last_ks_pvalue = 1.0


# =============================================================================
# IncrementalModel
# =============================================================================

class IncrementalModel:
    """在线增量学习模型 — 支持 SGD 训练的线性回归/逻辑回归/岭回归"""

    def __init__(self, n_features: int, model_type: ModelType = ModelType.LINEAR_REGRESSION,
                 learning_rate: float = 0.01, lr_schedule: LRSchedule = LRSchedule.EXPONENTIAL_DECAY,
                 l2_penalty: float = 0.0001, ema_alpha: float = 0.9,
                 lr_decay_rate: float = 0.999, lr_step_size: int = 100,
                 lr_step_drop: float = 0.5, lr_cosine_t_max: int = 1000,
                 lr_cosine_lr_min: float = 0.0001):
        """初始化增量模型"""
        self._n_features = n_features
        self._model_type = model_type
        self._l2_penalty = l2_penalty
        self._ema_alpha = ema_alpha

        # 学习率调度
        self._lr_schedule = lr_schedule
        self._base_lr = learning_rate
        self._lr_decay_rate = lr_decay_rate
        self._lr_step_size = lr_step_size
        self._lr_step_drop = lr_step_drop
        self._lr_cosine_t_max = lr_cosine_t_max
        self._lr_cosine_lr_min = lr_cosine_lr_min

        # 模型参数
        self._weights = [0.0] * n_features
        self._bias = 0.0
        self._ema_weights = [0.0] * n_features
        self._ema_bias = 0.0
        self._step_count = 0
        self._current_lr = learning_rate

    def predict(self, X: List[float]) -> float:
        """预测单个样本"""
        if len(X) != self._n_features:
            raise ValueError(f"Expected {self._n_features} features, got {len(X)}")
        raw = self._bias + sum(
            max(min(w * x, 1e6), -1e6) for w, x in zip(self._weights, X))

        if self._model_type == ModelType.LOGISTIC_REGRESSION:
            # Sigmoid
            raw = max(min(raw, 50), -50)
            return 1.0 / (1.0 + math.exp(-raw))
        # Clamp regression output
        return max(min(raw, 1e6), -1e6)

    def partial_fit(self, X: List[float], y: float, weight: float = 1.0):
        """单样本 SGD 更新"""
        self._step_count += 1
        lr = self._compute_lr()
        self._current_lr = lr

        pred = self.predict(X)

        # 梯度计算
        if self._model_type == ModelType.LOGISTIC_REGRESSION:
            # Log loss 梯度: (pred - y) * x
            error = pred - y
            grad_weights = [max(min(error * x, 1e3), -1e3) + self._l2_penalty * w
                            for x, w in zip(X, self._weights)]
            grad_bias = max(min(error, 1e3), -1e3)
        else:
            # MSE 梯度: (pred - y) * x (+ L2 for Ridge)
            error = max(min(pred - y, 1e3), -1e3)
            l2_term = self._l2_penalty if self._model_type == ModelType.RIDGE_REGRESSION else 0.0
            grad_weights = [error * x + l2_term * w for x, w in zip(X, self._weights)]
            grad_bias = error

        # 参数更新（带裁剪）
        for i in range(self._n_features):
            update = lr * weight * grad_weights[i]
            update = max(min(update, 10.0), -10.0)
            self._weights[i] -= update
            self._weights[i] = max(min(self._weights[i], 100.0), -100.0)
        bias_update = lr * weight * grad_bias
        bias_update = max(min(bias_update, 10.0), -10.0)
        self._bias -= bias_update
        self._bias = max(min(self._bias, 100.0), -100.0)

        # EMA 更新
        for i in range(self._n_features):
            self._ema_weights[i] = (self._ema_alpha * self._ema_weights[i] +
                                    (1 - self._ema_alpha) * self._weights[i])
        self._ema_bias = (self._ema_alpha * self._ema_bias +
                          (1 - self._ema_alpha) * self._bias)

    def batch_fit(self, X_list: List[List[float]], y_list: List[float],
                  weights: List[float] = None):
        """批量训练多个样本（每个样本单独 SGD 步）"""
        if weights is None:
            weights = [1.0] * len(X_list)
        for X, y, w in zip(X_list, y_list, weights):
            self.partial_fit(X, y, w)

    def _compute_lr(self) -> float:
        """根据 LR 调度策略计算当前学习率"""
        t = self._step_count
        if self._lr_schedule == LRSchedule.CONSTANT:
            return self._base_lr
        elif self._lr_schedule == LRSchedule.EXPONENTIAL_DECAY:
            return self._base_lr * (self._lr_decay_rate ** t)
        elif self._lr_schedule == LRSchedule.STEP_DECAY:
            epoch = t // max(self._lr_step_size, 1)
            return self._base_lr * (self._lr_step_drop ** epoch)
        elif self._lr_schedule == LRSchedule.COSINE_ANNEALING:
            progress = (t % max(self._lr_cosine_t_max, 1)) / max(self._lr_cosine_t_max, 1)
            return self._lr_cosine_lr_min + 0.5 * (self._base_lr - self._lr_cosine_lr_min) * \
                   (1 + math.cos(math.pi * progress))
        return self._base_lr

    def get_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        return {
            "weights": list(self._weights),
            "bias": self._bias,
            "ema_weights": list(self._ema_weights),
            "ema_bias": self._ema_bias,
            "n_features": self._n_features,
            "model_type": self._model_type.value,
            "step_count": self._step_count,
            "current_lr": self._current_lr,
        }

    def get_weights(self) -> List[float]:
        """返回 EMA 平滑后的权重"""
        return list(self._ema_weights)

    def reset(self):
        """重置模型"""
        self._weights = [0.0] * self._n_features
        self._bias = 0.0
        self._ema_weights = [0.0] * self._n_features
        self._ema_bias = 0.0
        self._step_count = 0


# =============================================================================
# Online Calibrator — Platt Scaling
# =============================================================================

class OnlineCalibrator:
    """在线 Platt Scaling 概率校准 + ECE 追踪"""

    def __init__(self, ece_threshold: float = 0.1, n_bins: int = 10, lr: float = 0.01):
        self._ece_threshold = ece_threshold
        self._n_bins = n_bins
        self._lr = lr

        # Platt scaling 参数: p_calib = sigmoid(A * score + B)
        self._A = 1.0
        self._B = 0.0

        # 校准历史
        self._predictions: List[float] = []
        self._actuals: List[int] = []  # 0 or 1
        self._ece_history: List[float] = []
        self._step = 0

    def calibrate(self, score: float) -> float:
        """对原始预测分数进行 Platt 校准"""
        x = self._A * score + self._B
        x = max(min(x, 50), -50)
        return 1.0 / (1.0 + math.exp(-x))

    def update(self, predictions: List[float], actuals: List[float]):
        """
        用预测值和实际值更新校准器
        actuals: 真实值（回归问题使用阈值二值化 or 保持连续）
        """
        if len(predictions) != len(actuals):
            return

        # 对于回归问题，我们基于正/负方向二值化
        for pred, act in zip(predictions, actuals):
            # 二值化：正值 -> 1，负值/零 -> 0
            binary_act = 1 if act > 0 else 0
            # 将预测值映射到 [0, 1]
            clipped_pred = max(min(pred, 1.0 - 1e-10), 1e-10)
            self._predictions.append(clipped_pred)
            self._actuals.append(binary_act)
            self._step += 1

            # 在线 Platt 更新：对 (A, B) 做 SGD
            # 使用 logistic loss: -y*log(sigmoid(A*s + B)) - (1-y)*log(1 - sigmoid(A*s + B))
            score = clipped_pred
            z = self._A * score + self._B
            z_clamped = max(min(z, 50), -50)
            sigmoid_z = 1.0 / (1.0 + math.exp(-z_clamped))
            error = sigmoid_z - binary_act

            self._A -= self._lr * error * score
            self._B -= self._lr * error

        # 限制内存
        max_len = 2000
        if len(self._predictions) > max_len:
            self._predictions = self._predictions[-max_len:]
            self._actuals = self._actuals[-max_len:]

    def compute_ece(self) -> float:
        """计算 Expected Calibration Error"""
        if len(self._predictions) < self._n_bins:
            return 0.0

        # 二值化实际值
        bin_size = 1.0 / self._n_bins
        bins = [[] for _ in range(self._n_bins)]

        for pred, act in zip(self._predictions, self._actuals):
            bin_idx = min(int(pred // bin_size), self._n_bins - 1)
            bins[bin_idx].append((pred, act))

        ece = 0.0
        total = len(self._predictions)
        for bin_data in bins:
            if not bin_data:
                continue
            avg_pred = sum(p for p, _ in bin_data) / len(bin_data)
            avg_act = sum(a for _, a in bin_data) / len(bin_data)
            ece += (len(bin_data) / total) * abs(avg_pred - avg_act)

        self._ece_history.append(ece)
        if len(self._ece_history) > 200:
            self._ece_history = self._ece_history[-200:]
        return ece

    def needs_calibration(self) -> bool:
        """检查是否需要校准"""
        ece = self.compute_ece()
        return ece > self._ece_threshold

    def get_ece(self) -> float:
        return self.compute_ece()

    def get_params(self) -> Dict[str, Any]:
        return {"A": self._A, "B": self._B, "step": self._step,
                "ece": self.get_ece()}

    def reset(self):
        self._A = 1.0
        self._B = 0.0
        self._predictions = []
        self._actuals = []
        self._ece_history = []
        self._step = 0


# =============================================================================
# OnlineLearner
# =============================================================================

class OnlineLearner(EnterpriseServiceMixin):
    """
    在线学习器 — 增强版
    集成: 多维样本采集、概念漂移检测、增量模型更新、特征重要性追踪、预测校准
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        ol_cfg = config.get("online_learner", {})

        # 基础配置（统一从 online_learner 段读取，避免污染顶层配置）
        self._mode = LearningMode(ol_cfg.get("learning_mode", "online"))
        self._status = LearningStatus.IDLE
        self._learning_rate = ol_cfg.get("learning_rate", 0.01)
        self._decay_rate = ol_cfg.get("decay_rate", 0.99)
        self._min_samples = ol_cfg.get("min_samples", 100)
        self._history: List[Dict[str, Any]] = []
        self._models: Dict[str, Any] = {}
        self._knowledge_base = None
        self._lock = asyncio.Lock()

        # ── SampleBuffer ──
        buf_cfg = ol_cfg.get("sample_buffer", {})
        self._sample_buffer = SampleBuffer(
            max_size=buf_cfg.get("max_size", 5000),
            weight_method=buf_cfg.get("weight_method", "time_decay"),
            decay_factor=buf_cfg.get("decay_factor", 0.99),
        )

        # ── DriftDetector ──
        drift_enabled = ol_cfg.get("drift_detection", {}).get("enabled", True)
        self._drift_detector = DriftDetector(config.get("online_learner", config)) if drift_enabled else None
        self._drift_learning_rate_multiplier = ol_cfg.get("drift_detection", {}).get(
            "lr_multiplier_on_drift", 3.0)

        # ── IncrementalModel 管理 ──
        model_cfg = ol_cfg.get("incremental_model", {})
        self._incremental_models: Dict[str, IncrementalModel] = {}
        self._default_model_type = ModelType(model_cfg.get("model_type", "linear_regression"))
        self._default_lr_schedule = LRSchedule(model_cfg.get("lr_schedule", "exponential_decay"))
        self._model_lr = model_cfg.get("learning_rate", self._learning_rate)
        self._model_l2 = model_cfg.get("l2_penalty", 0.0001)
        self._model_ema_alpha = model_cfg.get("ema_alpha", 0.9)
        self._lr_decay_rate = model_cfg.get("lr_decay_rate", 0.999)
        self._lr_step_size = model_cfg.get("lr_step_size", 100)
        self._lr_step_drop = model_cfg.get("lr_step_drop", 0.5)
        self._lr_cosine_t_max = model_cfg.get("lr_cosine_t_max", 1000)
        self._lr_cosine_lr_min = model_cfg.get("lr_cosine_lr_min", 0.0001)

        # ── 特征重要性 ──
        fi_cfg = ol_cfg.get("feature_importance", {})
        self._fi_enabled = fi_cfg.get("enabled", True)
        self._fi_window = fi_cfg.get("window_size", 500)
        self._fi_n_permutations = fi_cfg.get("n_permutations", 5)
        self._feature_importance: Dict[str, List[Dict[str, Any]]] = {}  # model_key -> [{name, score, timestamp}]

        # ── 校准器 ──
        cal_cfg = ol_cfg.get("calibration", {})
        self._cal_enabled = cal_cfg.get("enabled", True)
        self._calibrator = OnlineCalibrator(
            ece_threshold=cal_cfg.get("ece_threshold", 0.1),
            n_bins=cal_cfg.get("n_bins", 10),
            lr=cal_cfg.get("lr", 0.01),
        )

        # ── 统计 ──
        self._sample_count = 0
        self._drift_alerts: List[Dict[str, Any]] = []
        self._auto_reset_on_drift = ol_cfg.get("drift_detection", {}).get(
            "auto_reset_on_drift", False)

        logger.info(f"OnlineLearner initialized: mode={self._mode.value}, "
                     f"buffer_size={self._sample_buffer._max_size}, "
                     f"drift_enabled={drift_enabled}, "
                     f"fi_enabled={self._fi_enabled}, "
                     f"cal_enabled={self._cal_enabled}")

    # ============================
    # Lifecycle
    # ============================

    async def start(self):
        """启动在线学习器"""
        async with self._lock:
            if self._status == LearningStatus.PAUSED:
                self._status = LearningStatus.IDLE
        logger.info("Online learner started")

    async def stop(self):
        """停止在线学习器"""
        async with self._lock:
            self._status = LearningStatus.PAUSED
        logger.info("Online learner stopped")

    def pause(self):
        self._status = LearningStatus.PAUSED
        logger.info("Online learner paused")

    def resume(self):
        self._status = LearningStatus.IDLE
        logger.info("Online learner resumed")

    def reset(self):
        self._models = {}
        self._history = []
        self._sample_buffer.clear()
        self._incremental_models = {}
        self._feature_importance = {}
        self._calibrator.reset()
        self._drift_alerts = []
        self._sample_count = 0
        if self._drift_detector:
            self._drift_detector.reset()
        logger.info("Online learner reset")

    def set_knowledge_base(self, knowledge_base):
        self._knowledge_base = knowledge_base
        logger.info("Knowledge base set for online learner")

    # ============================
    # Public API — new
    # ============================

    async def add_sample(self, market_data: Dict[str, Any] = None,
                         strategy_features: Dict[str, float] = None,
                         predictions: Dict[str, float] = None,
                         actuals: Dict[str, float] = None,
                         outcome: Dict[str, Any] = None,
                         weight: float = None) -> str:
        """
        添加一个多维样本
        Returns: sample_id
        """
        async with self._lock:
            market_features = SampleBuffer.extract_market_features(market_data or {})
            sf = strategy_features or {}
            self._sample_buffer.add(
                market_features=market_features,
                strategy_features=sf,
                predictions=predictions or {},
                actuals=actuals or {},
                outcome=outcome or {},
                weight=weight,
            )
            self._sample_count += 1
            sample_id = f"sample_{self._sample_count}_{int(time.time() * 1000)}"
            logger.debug(f"Added sample {sample_id}, buffer_size={len(self._sample_buffer)}")
            return sample_id

    async def detect_drift(self) -> Dict[str, Any]:
        """检测概念漂移"""
        async with self._lock:
            if not self._drift_detector:
                return {"drift_detected": False, "reason": "drift_detection_disabled"}

            # 从 sample buffer 获取最近预测误差
            recent = self._sample_buffer.get_recent(100)
            if len(recent) < 30:
                return {"drift_detected": False, "reason": "insufficient_samples"}

            # 计算每个样本的预测误差
            errors = []
            for s in recent:
                for key in s["predictions"]:
                    if key in s["actuals"]:
                        errors.append(s["actuals"][key] - s["predictions"][key])

            if not errors:
                return {"drift_detected": False, "reason": "no_prediction_errors"}

            # 对每个 error 跑 drift detector
            features = {}
            if recent:
                features = recent[-1].get("market_features", {})

            result = None
            for err in errors:
                result = self._drift_detector.update(err, features)

            if result and result["drift_detected"]:
                severity = result["severity"]
                logger.warning(f"Concept drift detected! severity={severity.value}, "
                               f"methods={result['methods']}")
                self._drift_alerts.append(result)
                if len(self._drift_alerts) > 50:
                    self._drift_alerts = self._drift_alerts[-50:]

                # 自动处理：提高学习率或重置模型
                if self._auto_reset_on_drift:
                    if severity in (DriftSeverity.CRITICAL, DriftSeverity.HIGH):
                        await self._handle_drift_response(severity)

            return result or {"drift_detected": False}

    async def _handle_drift_response(self, severity: DriftSeverity):
        """处理漂移响应"""
        if severity in (DriftSeverity.CRITICAL, DriftSeverity.HIGH):
            # 重置增量模型
            for model in self._incremental_models.values():
                model.reset()
            logger.warning(f"Models reset due to {severity.value} drift")
        elif severity == DriftSeverity.MEDIUM:
            # 提高学习率
            self._model_lr = min(self._model_lr * self._drift_learning_rate_multiplier, 0.5)
            logger.info(f"Learning rate increased to {self._model_lr} due to drift")

    async def get_drift_status(self) -> Dict[str, Any]:
        """获取漂移检测状态"""
        async with self._lock:
            if not self._drift_detector:
                return {"enabled": False}
            status = self._drift_detector.get_status()
            status["recent_alerts"] = self._drift_alerts[-10:]
            return status

    async def get_feature_importance(self, strategy_name: str = "",
                                     top_k: int = 10) -> List[Dict[str, Any]]:
        """
        获取特征重要性排名（基于 permutation importance）
        """
        async with self._lock:
            model_key = f"{strategy_name}_combined" if strategy_name else "default"
            if model_key in self._feature_importance:
                return self._feature_importance[model_key][-top_k:]

            # 如果没有缓存，计算一次
            result = self._compute_feature_importance()
            return result[:top_k]

    async def calibrate_predictions(self, predictions: List[float],
                                    actuals: List[float] = None) -> List[float]:
        """
        校准预测值
        """
        async with self._lock:
            if not self._cal_enabled:
                return list(predictions)

            if actuals is not None:
                self._calibrator.update(predictions, actuals)

            calibrated = [self._calibrator.calibrate(p) for p in predictions]
            if self._calibrator.needs_calibration():
                logger.info(f"Calibration applied: ECE={self._calibrator.get_ece():.4f}")
            return calibrated

    # ============================
    # Public API — existing
    # ============================

    async def process_batch(self) -> Dict[str, Any]:
        """处理批量学习（供 scheduler 调用）"""
        async with self._lock:
            if self._status == LearningStatus.PAUSED:
                return {"status": "paused"}
            if len(self._sample_buffer) >= self._min_samples:
                result = await self._batch_update()
                # 批量学习后触发漂移检测
                drift_result = await self._run_drift_check()
                if drift_result:
                    result["drift_check"] = drift_result
                return result
            return {"status": "insufficient_samples",
                    "buffer_size": len(self._sample_buffer)}

    async def learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        学习核心方法 — 向后兼容
        data 格式: {
            predictions: Dict[str, float],
            actuals: Dict[str, float],
            strategy_name: str,
            market_data: Dict (optional),
            outcome: Dict (optional),
        }
        """
        async with self._lock:
            if self._status == LearningStatus.PAUSED:
                return {"status": "paused"}

            self._status = LearningStatus.LEARNING
            self._add_to_history(data)

            try:
                result: Dict[str, Any] = {}

                # 1) 添加样本到 SampleBuffer（内联，避免锁重入）
                market_data = data.get("market_data", {})
                predictions = data.get("predictions", {})
                actuals = data.get("actuals", {})
                outcome = data.get("outcome", {})
                strategy_name = data.get("strategy_name", "")

                market_features = SampleBuffer.extract_market_features(market_data)
                sf = {"strategy_id": hash(strategy_name) % 1000} if strategy_name else {}
                self._sample_buffer.add(
                    market_features=market_features,
                    strategy_features=sf,
                    predictions=predictions,
                    actuals=actuals,
                    outcome=outcome,
                )
                self._sample_count += 1

                # 2) 在线增量学习
                if self._mode in (LearningMode.ONLINE, LearningMode.HYBRID):
                    online_result = await self._online_update(data)
                    result["online_update"] = online_result

                # 3) 批量学习
                if self._mode in (LearningMode.BATCH, LearningMode.HYBRID):
                    if len(self._sample_buffer) >= self._min_samples:
                        batch_result = await self._batch_update()
                        result["batch_update"] = batch_result

                # 4) 特征重要性（定期计算）
                if self._fi_enabled and self._sample_count % 200 == 0:
                    result["feature_importance"] = self._compute_feature_importance()

                # 5) 校准更新
                if self._cal_enabled and predictions and actuals:
                    pred_list = list(predictions.values())
                    act_list = list(actuals.values())
                    if len(pred_list) == len(act_list):
                        self._calibrator.update(pred_list, act_list)

                # 6) 知识库存储
                if self._knowledge_base and outcome:
                    try:
                        pnl = outcome.get("pnl", 0)
                        await self._knowledge_base.store_trade_result(
                            strategy=strategy_name,
                            symbol=market_data.get("instId", ""),
                            market_state=market_data,
                            action=outcome.get("action", "unknown"),
                            pnl=float(pnl),
                            return_pct=float(outcome.get("return_pct", 0)),
                            duration=float(outcome.get("duration", 0)),
                        )
                    except Exception as e:
                        self._handle_exception(
                            e, module="OnlineLearner", function="learn",
                            severity="low", category="knowledge_base",
                        )

                self._status = LearningStatus.IDLE

                # 清理旧 history（保持向后兼容的 _history 行为）
                if len(self._history) >= self._min_samples:
                    self._history = self._history[self._min_samples:]
                return result

            except Exception as e:
                self._handle_exception(
                    e, module="OnlineLearner", function="learn",
                    severity="high", category="online_learning",
                )
                self._notify_sync(
                    "在线学习失败",
                    f"OnlineLearner.learn() failed: {e}",
                    priority="warning",
                    category="adaptive_learning",
                )
                self._status = LearningStatus.IDLE
                return {"status": "error", "message": str(e)}

    # ============================
    # Summary & Model Access
    # ============================

    def get_learning_summary(self) -> Dict[str, Any]:
        """获取学习摘要"""
        model_stats = {}
        for key, model in self._incremental_models.items():
            params = model.get_params()
            model_stats[key] = {
                "step_count": params["step_count"],
                "current_lr": params["current_lr"],
                "model_type": params["model_type"],
                "weight_norm": math.sqrt(sum(w * w for w in params["weights"])),
            }
        # 兼容旧 _models 格式
        for key, model in self._models.items():
            if key not in model_stats:
                model_stats[key] = {
                    "bias": model.get("bias", 0),
                    "variance": model.get("variance", 0),
                    "count": model.get("count", 0),
                }

        return {
            "status": self._status.value,
            "mode": self._mode.value,
            "learning_rate": self._learning_rate,
            "decay_rate": self._decay_rate,
            "num_models": len(self._incremental_models) + len(self._models),
            "buffer_size": len(self._sample_buffer),
            "total_samples": self._sample_count,
            "pending_samples": len(self._history),
            "model_stats": model_stats,
            "ece": self._calibrator.get_ece() if self._cal_enabled else None,
            "drift_alerts": len(self._drift_alerts),
        }

    def get_model(self, model_key: str) -> Optional[Dict[str, Any]]:
        """获取模型（向后兼容）"""
        # 优先查找增量模型
        if model_key in self._incremental_models:
            return self._incremental_models[model_key].get_params()
        # 兼容旧的 _models
        return self._models.get(model_key)

    def get_models(self) -> Dict[str, Any]:
        result = {}
        for key, model in self._incremental_models.items():
            result[key] = model.get_params()
        for key, model in self._models.items():
            result[key] = model
        return result

    def get_status(self) -> str:
        return self._status.value

    # ============================
    # Internal — Online Update
    # ============================

    async def _online_update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """使用增量模型进行在线更新"""
        predictions = data.get("predictions", {})
        actuals = data.get("actuals", {})
        strategy_name = data.get("strategy_name", "")

        # 提取特征向量
        market_data = data.get("market_data", {})
        market_features = SampleBuffer.extract_market_features(market_data)

        updates: Dict[str, Any] = {}

        for key, prediction in predictions.items():
            if key not in actuals:
                continue

            model_key = f"{strategy_name}_{key}"
            actual = actuals[key]
            error = actual - prediction

            # 确保增量模型存在
            if model_key not in self._incremental_models:
                # 估算特征维度
                feature_list = list(market_features.values()) if market_features else [prediction]
                n_features = len(feature_list)
                if n_features == 0:
                    n_features = 1
                self._incremental_models[model_key] = self._create_incremental_model(n_features)
                logger.info(f"Created incremental model: {model_key} ({n_features} features)")

            model = self._incremental_models[model_key]
            X = list(market_features.values()) if market_features else [prediction]
            # 确保特征维度匹配
            n_needed = model._n_features
            if len(X) < n_needed:
                X = X + [0.0] * (n_needed - len(X))
            elif len(X) > n_needed:
                X = X[:n_needed]

            # SGD 更新
            model.partial_fit(X, actual)

            # 后向兼容：也更新旧格式模型
            if model_key not in self._models:
                self._models[model_key] = {"bias": 0.0, "variance": 1.0, "count": 0}
            old_model = self._models[model_key]
            lr = self._learning_rate * (self._decay_rate ** old_model["count"])
            old_model["bias"] += lr * error
            old_model["variance"] += lr * (error ** 2 - old_model["variance"])
            old_model["count"] += 1

            updates[key] = {
                "error": error,
                "learning_rate": model._current_lr,
                "model_params": model.get_params(),
            }

        return {"status": "success", "updates": updates}

    async def _batch_update(self) -> Dict[str, Any]:
        """批量更新 — 使用 SampleBuffer 中所有样本"""
        X, y, weights = self._sample_buffer.get_feature_matrix(None)
        if not X:
            return {"status": "no_data"}

        n_features = len(X[0]) if X else 1
        batch_key = "_batch_model"

        if batch_key not in self._incremental_models:
            self._incremental_models[batch_key] = self._create_incremental_model(n_features)

        model = self._incremental_models[batch_key]
        # 确保维度匹配
        if model._n_features != n_features:
            model = self._create_incremental_model(n_features)
            self._incremental_models[batch_key] = model

        model.batch_fit(X, y, weights)

        # 统计
        errors = []
        for i, (xi, yi) in enumerate(zip(X, y)):
            pred = model.predict(xi)
            errors.append(yi - pred)

        batch_results = {
            "status": "success",
            "samples_processed": len(X),
            "mean_error": statistics.mean(errors) if errors else 0,
            "std_error": statistics.stdev(errors) if len(errors) > 1 else 0,
            "model_params": model.get_params(),
        }

        # 后向兼容
        self._history = []

        return batch_results

    async def _run_drift_check(self) -> Optional[Dict[str, Any]]:
        """运行漂移检查（内部）"""
        if not self._drift_detector:
            return None

        recent = self._sample_buffer.get_recent(200)
        if len(recent) < 50:
            return None

        all_errors = []
        for s in recent:
            for key in s["predictions"]:
                if key in s["actuals"]:
                    all_errors.append(s["actuals"][key] - s["predictions"][key])

        if not all_errors:
            return None

        features = recent[-1].get("market_features", {}) if recent else {}
        result = None
        for err in all_errors[-50:]:  # 只取最近50个
            result = self._drift_detector.update(err, features)

        if result and result["drift_detected"]:
            self._drift_alerts.append(result)
            if self._auto_reset_on_drift and result["severity"] in (
                    DriftSeverity.CRITICAL, DriftSeverity.HIGH):
                await self._handle_drift_response(result["severity"])

        return result

    def _compute_feature_importance(self) -> List[Dict[str, Any]]:
        """计算 permutation importance"""
        X, y, weights = self._sample_buffer.get_feature_matrix(self._fi_window)
        feature_names = self._sample_buffer.get_feature_names()

        if not X or not y or len(X) < 10:
            return []

        n_features = len(X[0])
        if n_features != len(feature_names):
            return []

        # 创建临时模型计算 baseline
        temp_model = IncrementalModel(n_features, ModelType.LINEAR_REGRESSION,
                                      learning_rate=0.001)
        temp_model.batch_fit(X, y, weights)

        # Baseline MSE
        baseline_errors = []
        for xi, yi in zip(X, y):
            pred = temp_model.predict(xi)
            err = yi - pred
            err = max(min(err, 1e6), -1e6)
            baseline_errors.append(err * err)
        baseline_mse = statistics.mean(baseline_errors) if baseline_errors else 0

        # Permutation importance
        import random
        scores: List[Dict[str, Any]] = []

        for fi in range(n_features):
            permuted_mse_total = 0.0
            n_perm = self._fi_n_permutations

            for _ in range(n_perm):
                # 拷贝 X 并打乱第 fi 列
                X_perm = [list(row) for row in X]
                col_values = [row[fi] for row in X_perm]
                random.shuffle(col_values)
                for row_idx, row in enumerate(X_perm):
                    row[fi] = col_values[row_idx]

                perm_errors = []
                for xi, yi in zip(X_perm, y):
                    pred = temp_model.predict(xi)
                    err = yi - pred
                    err = max(min(err, 1e6), -1e6)
                    perm_errors.append(err * err)
                perm_mse = statistics.mean(perm_errors) if perm_errors else 0
                permuted_mse_total += perm_mse

            avg_perm_mse = permuted_mse_total / n_perm
            importance = avg_perm_mse - baseline_mse
            scores.append({
                "feature": feature_names[fi] if fi < len(feature_names) else f"f{fi}",
                "importance": round(importance, 6),
                "baseline_mse": round(baseline_mse, 6),
                "permuted_mse": round(avg_perm_mse, 6),
                "timestamp": datetime.now().isoformat(),
            })

        # 排序：重要度从高到低
        scores.sort(key=lambda s: s["importance"], reverse=True)

        # 存储历史（键需与 get_feature_importance 的查询键一致）
        model_key = "default"
        if model_key not in self._feature_importance:
            self._feature_importance[model_key] = []
        self._feature_importance[model_key].extend(scores)
        if len(self._feature_importance[model_key]) > 200:
            self._feature_importance[model_key] = self._feature_importance[model_key][-200:]

        return scores

    def _create_incremental_model(self, n_features: int) -> IncrementalModel:
        """创建增量模型实例"""
        return IncrementalModel(
            n_features=n_features,
            model_type=self._default_model_type,
            learning_rate=self._model_lr,
            lr_schedule=self._default_lr_schedule,
            l2_penalty=self._model_l2,
            ema_alpha=self._model_ema_alpha,
            lr_decay_rate=self._lr_decay_rate,
            lr_step_size=self._lr_step_size,
            lr_step_drop=self._lr_step_drop,
            lr_cosine_t_max=self._lr_cosine_t_max,
            lr_cosine_lr_min=self._lr_cosine_lr_min,
        )

    def _add_to_history(self, data: Dict[str, Any]):
        """向后兼容：维护旧的 _history 列表"""
        entry = {"timestamp": datetime.now(), **data}
        self._history.append(entry)
        max_history = self._config.get("max_history", 10000)
        if len(self._history) > max_history:
            self._history = self._history[-max_history:]
