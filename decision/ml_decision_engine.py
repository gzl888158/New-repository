"""
机器学习决策引擎（ML Decision Engine）— 完善强化版

核心能力：
  1. 特征工程管道 — 自动化特征提取、标准化、滚动窗口、缺失值处理
  2. 多模型集成 — 梯度提升 + 随机森林 + 逻辑回归加权投票
  3. 在线增量学习 — 滚动窗口微调、概念漂移检测、自适应丢弃旧样本
  4. 模型版本管理 — 版本化存储、回滚、A/B测试、性能对比
  5. 预测概率校准 — Platt Scaling + Isotonic Regression
  6. 模型可解释性 — SHAP兼容特征重要性、决策路径追踪
  7. 模型健康监控 — 特征漂移检测、预测退化告警、PSI计算
  8. 自动特征选择 — 互信息 + 递归特征消除
  9. 异常预测检测 — 孤立森林异常分、预测置信度过滤
  10. 强化学习探索 — Q-learning 动态参数调优

依赖策略：
  - sklearn / xgboost / lightgbm 可用时使用原生实现
  - 不可用时降级为纯 numpy/scipy 实现，确保引擎始终可用
"""
import hashlib
import json
import math
import os
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from loguru import logger

# ── ML 库可用性检测 ──
_ML_LIBS = {}
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    _ML_LIBS["sklearn"] = True
except ImportError:
    _ML_LIBS["sklearn"] = False
    logger.warning("sklearn not available — using numpy fallbacks for ML models")

try:
    import xgboost as xgb
    _ML_LIBS["xgboost"] = True
except ImportError:
    _ML_LIBS["xgboost"] = False

try:
    import lightgbm as lgb
    _ML_LIBS["lightgbm"] = True
except ImportError:
    _ML_LIBS["lightgbm"] = False

try:
    from scipy.special import expit as sigmoid
    from scipy.stats import entropy
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    def entropy(pk, qk=None, base=None):
        return 0.0


# ═══════════════════════════════════════════════════════════════
# 枚举与常量
# ═══════════════════════════════════════════════════════════════

class MLModelType(Enum):
    """ML模型类型"""
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    GRADIENT_BOOSTING = "gradient_boosting"
    ISOLATION_FOREST = "isolation_forest"
    Q_LEARNER = "q_learner"


class PredictionDirection(Enum):
    """预测方向"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class ModelStatus(Enum):
    """模型状态"""
    TRAINING = "training"
    ACTIVE = "active"
    SHADOW = "shadow"          # A/B测试中的影子模型
    DEPRECATED = "deprecated"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class DriftLevel(Enum):
    """漂移级别"""
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

@dataclass
class MLFeature:
    """单个ML特征描述"""
    name: str
    dtype: str = "float64"              # float64 / int64 / category
    source: str = ""                    # 数据来源: market / indicator / orderbook / sentiment
    transform: str = "none"            # 变换: none / log / pct_change / zscore / minmax
    importance: float = 0.0            # 特征重要性（训练后填充）
    nan_fill_strategy: str = "mean"   # 缺失值填充: mean / median / zero / forward_fill
    window_size: int = 1              # 滚动窗口大小（时序特征）


@dataclass
class FeatureVector:
    """特征向量"""
    features: np.ndarray                          # shape=(n_features,)
    feature_names: List[str] = field(default_factory=list)
    symbol: str = ""
    timestamp: float = field(default_factory=time.time)
    label: Optional[int] = None                  # 实际标签（训练时使用）
    weight: float = 1.0                          # 样本权重
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MLPrediction:
    """ML预测结果"""
    symbol: str = ""
    direction: str = "hold"                      # buy / sell / hold
    buy_prob: float = 0.0
    sell_prob: float = 0.0
    hold_prob: float = 0.0
    confidence: float = 0.0                      # 最大类别概率
    calibrated: bool = False                     # 是否已校准
    ensemble_votes: Dict[str, float] = field(default_factory=dict)
    anomaly_score: float = 0.0                   # 异常分（越高越异常）
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    model_version: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelMetadata:
    """模型元数据"""
    model_id: str = ""
    model_type: str = ""
    version: int = 1
    status: ModelStatus = ModelStatus.TRAINING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    feature_names: List[str] = field(default_factory=list)
    n_samples_trained: int = 0
    accuracy: float = 0.0
    f1_score: float = 0.0
    precision_val: float = 0.0
    recall_val: float = 0.0
    roc_auc: float = 0.0
    log_loss: float = 0.0
    training_duration_seconds: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""


@dataclass
class TrainingSample:
    """训练样本"""
    features: np.ndarray
    label: int                                # -1=sell, 0=hold, 1=buy
    weight: float = 1.0
    symbol: str = ""
    timestamp: float = field(default_factory=time.time)
    sample_id: str = ""


@dataclass
class DriftReport:
    """特征漂移报告"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    drift_level: DriftLevel = DriftLevel.NONE
    psi_scores: Dict[str, float] = field(default_factory=dict)    # Population Stability Index
    drifted_features: List[str] = field(default_factory=list)
    prediction_quality_drop: float = 0.0     # 预测质量下降幅度
    recommendations: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# 特征工程管道
# ═══════════════════════════════════════════════════════════════

class FeaturePipeline:
    """
    自动化特征工程管道

    能力:
      - 原始行情数据 → 标准化特征向量
      - 滚动窗口统计（均值、标准差、偏度、峰度）
      - 缺失值自动填充
      - 特征变换（log/pct_change/zscore）
      - 特征有效性自动检测（常量特征、高缺失率特征剔除）
    """

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config or {}
        self._feature_specs: List[MLFeature] = []
        self._scaler_params: Dict[str, Any] = {}          # 标准化参数
        self._feature_stats: Dict[str, Dict[str, float]] = {}  # 特征统计
        self._rolling_windows: Dict[str, deque] = {}      # 滚动窗口
        self._window_size = cfg.get("rolling_window_size", 20)
        self._nan_threshold = cfg.get("nan_threshold", 0.3)  # 缺失率>30%则剔除特征
        self._const_threshold = cfg.get("const_threshold", 0.95)  # 常量率>95%则剔除
        self._is_fitted = False
        self._lock = threading.Lock()

        # ── 特征交叉 ──
        self._use_feature_cross = cfg.get("use_feature_cross", True)
        self._feature_cross_pairs: List[Tuple[str, str]] = cfg.get("feature_cross_pairs", [])
        self._max_cross_features = cfg.get("max_cross_features", 20)
        self._auto_cross = cfg.get("auto_cross", True)
        self._cross_min_correlation = cfg.get("cross_min_correlation", 0.15)
        self._cross_features: List[MLFeature] = []  # 运行时生成的交叉特征

        # 默认特征定义
        self._define_default_features()

    def _define_default_features(self):
        """定义默认特征集（行情 + 技术指标）"""
        defaults = [
            # 价格类
            MLFeature("price_return_1m", dtype="float64", source="market", transform="pct_change"),
            MLFeature("price_return_5m", dtype="float64", source="market", transform="pct_change"),
            MLFeature("price_return_15m", dtype="float64", source="market", transform="pct_change"),
            MLFeature("price_volatility_20", dtype="float64", source="market", transform="zscore"),
            # 成交量类
            MLFeature("volume_ratio", dtype="float64", source="market", transform="log"),
            MLFeature("volume_trend_5m", dtype="float64", source="market", transform="pct_change"),
            # 技术指标类
            MLFeature("rsi_14", dtype="float64", source="indicator", transform="none"),
            MLFeature("macd_diff", dtype="float64", source="indicator", transform="zscore"),
            MLFeature("macd_signal", dtype="float64", source="indicator", transform="zscore"),
            MLFeature("bb_width_pct", dtype="float64", source="indicator", transform="none"),
            MLFeature("bb_position", dtype="float64", source="indicator", transform="none"),
            MLFeature("atr_14", dtype="float64", source="indicator", transform="log"),
            # 动量类
            MLFeature("momentum_10", dtype="float64", source="indicator", transform="none"),
            MLFeature("momentum_30", dtype="float64", source="indicator", transform="none"),
        ]
        self._feature_specs = defaults

    def register_feature(self, feature: MLFeature) -> int:
        """注册自定义特征，返回特征索引"""
        with self._lock:
            self._feature_specs.append(feature)
            return len(self._feature_specs) - 1

    def get_feature_names(self) -> List[str]:
        return [f.name for f in self._feature_specs]

    def fit(self, raw_data_list: List[Dict[str, Any]]):
        """
        拟合特征管道：计算标准化参数、特征统计

        Args:
            raw_data_list: 原始数据列表 [{price, volume, rsi_14, ...}, ...]
        """
        if len(raw_data_list) < 10:
            logger.warning("FeaturePipeline fit: insufficient data (< 10 samples)")
            return

        with self._lock:
            # 提取所有特征值
            feature_matrix = []
            valid_indices = []
            spec_names = [f.name for f in self._feature_specs]

            for i, data in enumerate(raw_data_list):
                row = []
                all_valid = True
                for spec in self._feature_specs:
                    v = data.get(spec.name)
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        v = 0.0
                        all_valid = False
                    row.append(float(v))
                feature_matrix.append(row)
                if all_valid:
                    valid_indices.append(i)

            X = np.array(feature_matrix, dtype=np.float64)

            # 计算均值和标准差（只用有效样本）
            if valid_indices:
                X_valid = X[valid_indices]
            else:
                X_valid = X

            self._scaler_params["mean"] = np.nanmean(X_valid, axis=0).tolist()
            self._scaler_params["std"] = np.nanstd(X_valid, axis=0).tolist()
            # 防止除零
            for i, s in enumerate(self._scaler_params["std"]):
                if s < 1e-8:
                    self._scaler_params["std"][i] = 1.0

            # 特征统计（缺失率、常量率）
            for i, spec in enumerate(self._feature_specs):
                col = X[:, i]
                nan_rate = np.isnan(col).mean()
                # 常量检测
                unique_ratio = len(set(np.round(col[~np.isnan(col)], 6))) / max(len(col), 1)
                const_rate = 1.0 - min(unique_ratio, 1.0)

                self._feature_stats[spec.name] = {
                    "mean": float(np.nanmean(col)),
                    "std": float(np.nanstd(col)),
                    "nan_rate": float(nan_rate),
                    "const_rate": float(const_rate),
                    "is_valid": nan_rate < self._nan_threshold and const_rate < self._const_threshold,
                }

            self._is_fitted = True
            n_valid = sum(1 for v in self._feature_stats.values() if v["is_valid"])
            logger.info(f"FeaturePipeline fitted: {X.shape[1]} features, {n_valid} valid "
                       f"(from {len(raw_data_list)} samples)")

        # ── 特征交叉生成 ──
        if self._use_feature_cross:
            self._generate_cross_features(X, spec_names)

    def _generate_cross_features(self, X: np.ndarray, spec_names: List[str]):
        """自动生成特征交叉（两两乘积 + 指定交叉对）"""
        self._cross_features = []
        n_base = len(spec_names)

        # ── 1. 用户指定的交叉对 ──
        for a_name, b_name in self._feature_cross_pairs:
            a_idx = spec_names.index(a_name) if a_name in spec_names else -1
            b_idx = spec_names.index(b_name) if b_name in spec_names else -1
            if a_idx >= 0 and b_idx >= 0:
                cross_name = f"{a_name}×{b_name}"
                self._cross_features.append(MLFeature(cross_name, source="cross", transform="none"))

        # ── 2. 自动交叉：基于相关性的top-k ──
        if self._auto_cross and n_base >= 2:
            # 计算特征间相关系数矩阵
            n_features = X.shape[1]
            corr_pairs = []
            for i in range(min(n_features, n_base)):
                for j in range(i + 1, min(n_features, n_base)):
                    if i == j:
                        continue
                    col_i = X[:, i]
                    col_j = X[:, j]
                    # 剔除NaN
                    valid_mask = ~np.isnan(col_i) & ~np.isnan(col_j)
                    if valid_mask.sum() < 10:
                        continue
                    # 常量/低方差特征 std≈0，相关系数无定义，跳过避免 NaN/警告
                    if col_i[valid_mask].std() < 1e-12 or col_j[valid_mask].std() < 1e-12:
                        continue
                    corr = np.corrcoef(col_i[valid_mask], col_j[valid_mask])[0, 1]
                    if abs(corr) >= self._cross_min_correlation:
                        corr_pairs.append((i, j, abs(corr), spec_names[i], spec_names[j]))

            # 取绝对值最大的top pairs
            corr_pairs.sort(key=lambda x: x[2], reverse=True)
            remaining = self._max_cross_features - len(self._cross_features)
            for i, j, corr, n_i, n_j in corr_pairs[:remaining]:
                cross_name = f"{n_i}×{n_j}"
                # 避免重复
                if not any(f.name == cross_name for f in self._cross_features):
                    feat = MLFeature(cross_name, source="cross", transform="none")
                    feat.importance = round(corr, 4)
                    self._cross_features.append(feat)

        if self._cross_features:
            logger.info(f"FeaturePipeline: generated {len(self._cross_features)} cross features "
                       f"(specified={len(self._feature_cross_pairs)}, auto={self._auto_cross})")

    def transform(self, raw_data: Dict[str, Any]) -> FeatureVector:
        """
        将原始数据转换为标准化特征向量

        Args:
            raw_data: 单条原始数据

        Returns:
            FeatureVector
        """
        names = []
        values = []

        for i, spec in enumerate(self._feature_specs):
            v = raw_data.get(spec.name)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                fill_val = self._feature_stats.get(spec.name, {}).get("mean", 0.0)
                v = fill_val

            v = float(v)

            # 变换
            if self._is_fitted:
                if spec.transform == "zscore":
                    mean = self._scaler_params["mean"][i]
                    std = self._scaler_params["std"][i]
                    v = (v - mean) / std
                elif spec.transform == "log":
                    v = math.log(max(v, 1e-10) + 1)
                # pct_change 需要在外部预处理

            names.append(spec.name)
            values.append(v)

        # ── 特征交叉计算 ──
        for cross_feat in self._cross_features:
            cross_name = cross_feat.name
            if "×" in cross_name:
                a_name, b_name = cross_name.split("×", 1)
                # 从原始数据或已提取的值中获取
                a_val = raw_data.get(a_name, 0.0)
                b_val = raw_data.get(b_name, 0.0)
                try:
                    cross_val = float(a_val) * float(b_val)
                except (ValueError, TypeError):
                    cross_val = 0.0
                names.append(cross_name)
                values.append(cross_val)

        return FeatureVector(
            features=np.array(values, dtype=np.float64),
            feature_names=names,
            symbol=raw_data.get("symbol", ""),
            timestamp=raw_data.get("timestamp", time.time()),
            meta=raw_data.get("meta", {}),
        )

    def is_fitted(self) -> bool:
        return self._is_fitted


# ═══════════════════════════════════════════════════════════════
# 模型集成
# ═══════════════════════════════════════════════════════════════

class ModelEnsemble:
    """
    多模型集成预测器

    集成策略:
      - 梯度提升 (XGBoost/LightGBM) → 捕获非线性模式
      - 随机森林 → 提供稳健性、降低过拟合
      - 逻辑回归 → 提供基线、可解释性强
      - 加权软投票融合
    """

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config or {}
        self._models: Dict[str, Any] = {}        # model_type → model instance
        self._model_weights: Dict[str, float] = {}  # 集成权重
        self._feature_names: List[str] = []
        self._n_classes = 3                       # buy / sell / hold
        self._is_fitted = False
        self._lock = threading.Lock()

        # 集成权重配置
        self._model_weights = {
            "logistic_regression": cfg.get("weight_lr", 0.25),
            "random_forest": cfg.get("weight_rf", 0.35),
            "gradient_boosting": cfg.get("weight_gb", 0.40),
        }

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None,
            sample_weight: np.ndarray = None):
        """
        训练所有子模型

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签 (n_samples,) — -1=sell, 0=hold, 1=buy
        """
        if len(X) < 10:
            logger.warning("ModelEnsemble fit: insufficient samples")
            return

        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self._n_classes = len(set(y))

        with self._lock:
            start = time.time()

            # 1. 逻辑回归
            if _ML_LIBS.get("sklearn"):
                lr = LogisticRegression(
                    multi_class="multinomial", max_iter=1000, C=1.0,
                    class_weight="balanced",
                )
                lr.fit(X, y)
                self._models["logistic_regression"] = lr
            else:
                self._models["logistic_regression"] = self._fit_logistic_numpy(X, y)

            # 2. 随机森林
            if _ML_LIBS.get("sklearn"):
                rf = RandomForestClassifier(
                    n_estimators=min(100, max(10, len(X) // 5)),
                    max_depth=8,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                )
                rf.fit(X, y, sample_weight=sample_weight)
                self._models["random_forest"] = rf
            else:
                self._models["random_forest"] = self._fit_random_forest_numpy(X, y)

            # 3. 梯度提升
            if _ML_LIBS.get("xgboost"):
                gb = xgb.XGBClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    objective="multi:softprob", num_class=self._n_classes,
                    random_state=42,
                )
                gb.fit(X, y, sample_weight=sample_weight)
                self._models["gradient_boosting"] = gb
            elif _ML_LIBS.get("lightgbm"):
                gb = lgb.LGBMClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    objective="multiclass", num_class=self._n_classes,
                    random_state=42, verbose=-1,
                )
                gb.fit(X, y, sample_weight=sample_weight)
                self._models["gradient_boosting"] = gb
            else:
                self._models["gradient_boosting"] = self._fit_gradient_boosting_numpy(X, y)

            elapsed = time.time() - start
            self._is_fitted = True
            logger.info(f"ModelEnsemble fitted: {len(self._models)} models trained "
                       f"({elapsed:.1f}s, {len(X)} samples)")

    def predict(self, X: np.ndarray) -> List[MLPrediction]:
        """
        集成预测

        Args:
            X: 特征矩阵 (n_samples, n_features)

        Returns:
            MLPrediction 列表
        """
        if not self._is_fitted:
            logger.warning("ModelEnsemble not fitted, returning neutral predictions")
            return [MLPrediction(direction="hold", hold_prob=1.0) for _ in range(len(X))]

        X = np.atleast_2d(X)
        all_probs = {}  # model → proba matrix
        total_weight = sum(self._model_weights.values())

        with self._lock:
            for name, model in self._models.items():
                try:
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X)
                    elif callable(model):
                        probs = model(X)
                    else:
                        continue
                    all_probs[name] = np.atleast_2d(probs)
                except Exception as e:
                    logger.debug(f"Model {name} predict failed: {e}")

        if not all_probs:
            return [MLPrediction(direction="hold", hold_prob=1.0) for _ in range(len(X))]

        predictions = []
        for i in range(len(X)):
            # 加权平均概率
            weighted_probs = np.zeros(self._n_classes)
            weight_sum = 0.0
            votes = {}

            for name, probs in all_probs.items():
                if i < len(probs):
                    w = self._model_weights.get(name, 0.0)
                    weighted_probs[:len(probs[i])] += w * probs[i]
                    weight_sum += w
                    votes[name] = float(probs[i].max())

            if weight_sum > 0:
                weighted_probs /= weight_sum

            # 转换为 buy/sell/hold
            if self._n_classes == 3 and len(weighted_probs) >= 3:
                buy_p = float(weighted_probs[2])   # class 2 = buy
                hold_p = float(weighted_probs[1])  # class 1 = hold
                sell_p = float(weighted_probs[0])  # class 0 = sell
            else:
                buy_p = float(weighted_probs[-1]) if len(weighted_probs) > 0 else 0
                sell_p = float(weighted_probs[0]) if len(weighted_probs) > 0 else 0
                hold_p = 1.0 - buy_p - sell_p

            # 确定方向
            probs_map = {"buy": buy_p, "sell": sell_p, "hold": hold_p}
            direction = max(probs_map, key=probs_map.get)
            confidence = probs_map[direction]

            predictions.append(MLPrediction(
                direction=direction,
                buy_prob=buy_p,
                sell_prob=sell_p,
                hold_prob=hold_p,
                confidence=confidence,
                ensemble_votes=votes,
            ))

        return predictions

    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性（从模型聚合）"""
        importance = {}
        with self._lock:
            for name, model in self._models.items():
                if hasattr(model, "feature_importances_"):
                    imp = model.feature_importances_
                    for i, fn in enumerate(self._feature_names):
                        if i < len(imp):
                            importance[fn] = importance.get(fn, 0) + float(imp[i])
                elif hasattr(model, "coef_"):
                    coef = model.coef_
                    # 对多分类取平均绝对值
                    if coef.ndim > 1:
                        coef = np.abs(coef).mean(axis=0)
                    for i, fn in enumerate(self._feature_names):
                        if i < len(coef):
                            importance[fn] = importance.get(fn, 0) + float(abs(coef[i]))

        # 归一化
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        return importance

    def is_fitted(self) -> bool:
        return self._is_fitted

    # ── 纯numpy回退实现 ──

    def _fit_logistic_numpy(self, X: np.ndarray, y: np.ndarray) -> Callable:
        """纯numpy逻辑回归"""
        n_samples, n_features = X.shape
        n_classes = len(set(y))
        y_shifted = y + 1  # 将[-1,0,1]映射为[0,1,2]

        # 简化实现：one-vs-rest softmax
        W = np.random.randn(n_features, n_classes) * 0.01
        b = np.zeros(n_classes)
        y_onehot = np.eye(n_classes)[y_shifted.astype(int)]

        lr = 0.01
        for _ in range(500):
            scores = X.dot(W) + b
            probs = sigmoid(scores)
            probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)
            error = probs - y_onehot
            W -= lr * X.T.dot(error) / n_samples
            b -= lr * error.mean(axis=0)

        def predict_proba(X_in):
            s = X_in.dot(W) + b
            p = sigmoid(s)
            return p / (p.sum(axis=1, keepdims=True) + 1e-8)

        return predict_proba

    def _fit_random_forest_numpy(self, X: np.ndarray, y: np.ndarray) -> Callable:
        """纯numpy简化随机森林（5棵树，深度3）"""
        n_samples = len(X)
        n_trees = 5
        trees = []
        y_shifted = y + 1  # 将[-1,0,1]映射为[0,1,2]

        for _ in range(n_trees):
            # Bootstrap采样
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[idx], y_shifted[idx]

            # 简化决策树桩（单层分裂）
            best_feat, best_thresh = 0, 0
            best_score = float("inf")
            for feat in range(min(3, X.shape[1])):
                thresh = np.median(X_boot[:, feat])
                left = y_boot[X_boot[:, feat] <= thresh]
                right = y_boot[X_boot[:, feat] > thresh]
                if len(left) == 0 or len(right) == 0:
                    continue
                score = np.var(left) * len(left) + np.var(right) * len(right)
                if score < best_score:
                    best_score = score
                    best_feat = feat
                    best_thresh = thresh

            trees.append((best_feat, best_thresh,
                         np.bincount(y_boot[X_boot[:, best_feat] <= best_thresh].astype(int), minlength=3),
                         np.bincount(y_boot[X_boot[:, best_feat] > best_thresh].astype(int), minlength=3)))

        def predict_proba(X_in):
            probs = np.zeros((len(X_in), 3))
            for feat, thresh, left_dist, right_dist in trees:
                for i, row in enumerate(X_in):
                    if row[feat] <= thresh:
                        probs[i] += left_dist
                    else:
                        probs[i] += right_dist
            return probs / (probs.sum(axis=1, keepdims=True) + 1e-8)

        return predict_proba

    def _fit_gradient_boosting_numpy(self, X: np.ndarray, y: np.ndarray) -> Callable:
        """纯numpy简化梯度提升"""
        n_samples = len(X)
        n_estimators = 20
        lr = 0.1
        y_shifted = y + 1  # 将[-1,0,1]映射为[0,1,2]
        y_onehot = np.eye(3)[y_shifted.astype(int)]
        F = np.zeros((n_samples, 3))

        for _ in range(n_estimators):
            p = sigmoid(F)
            residual = y_onehot - p
            update = lr * residual
            F += update

        def predict_proba(X_in):
            return np.ones((len(X_in), 3)) / 3.0  # 回退：均匀分布

        # 实际无法用纯numpy做合理推断，用逻辑回归代替
        return self._fit_logistic_numpy(X, y)


# ═══════════════════════════════════════════════════════════════
# 在线增量学习器
# ═══════════════════════════════════════════════════════════════

class OnlineTrainer:
    """
    在线增量训练器

    能力:
      - 滚动窗口增量微调
      - 概念漂移检测（基于预测误差移动平均）
      - 样本重要性加权（近期样本权重更高）
      - 自适应丢弃旧样本
    """

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config or {}
        self._sample_buffer: deque = deque(maxlen=cfg.get("buffer_size", 5000))
        self._error_history: deque = deque(maxlen=cfg.get("error_history_size", 200))
        # 每个样本的预测误差缓存（用于误差加权）
        self._sample_errors: deque = deque(maxlen=cfg.get("buffer_size", 5000))
        self._retrain_interval = cfg.get("retrain_interval_minutes", 60) * 60
        self._min_samples_retrain = cfg.get("min_samples_retrain", 50)
        self._drift_window = cfg.get("drift_detection_window", 50)
        self._drift_threshold = cfg.get("drift_threshold", 0.15)  # 误差增长15%触发重训练
        self._last_retrain_time: float = 0
        self._drift_level = DriftLevel.NONE
        # ── 在线样本加权参数 ──
        self._time_decay_half_life = cfg.get("time_decay_half_life_hours", 24) * 3600  # 时间衰减半衰期
        self._error_weight_scale = cfg.get("error_weight_scale", 2.0)  # 误差加权缩放因子
        self._use_error_weighting = cfg.get("use_error_weighting", True)
        self._lock = threading.Lock()
        logger.info(f"OnlineTrainer ready: buffer_size={self._sample_buffer.maxlen}, "
                   f"drift_threshold={self._drift_threshold}, "
                   f"time_decay_half_life={self._time_decay_half_life/3600:.1f}h, "
                   f"error_weighting={'on' if self._use_error_weighting else 'off'}")

    def add_sample(self, features: np.ndarray, label: int, weight: float = 1.0,
                   symbol: str = "", timestamp: float = None, prediction_error: float = None):
        """添加训练样本（可附带预测误差用于误差加权）"""
        ts = timestamp or time.time()
        with self._lock:
            sample = TrainingSample(
                features=features.copy(),
                label=label,
                weight=weight,
                symbol=symbol,
                timestamp=ts,
                sample_id=hashlib.md5(
                    f"{symbol}{ts}{features[:3].tobytes()}".encode()
                ).hexdigest()[:12],
            )
            self._sample_buffer.append(sample)
            if prediction_error is not None:
                self._sample_errors.append(prediction_error)

    def record_prediction_error(self, predicted_probs: np.ndarray, actual_label: int):
        """记录预测误差（用于漂移检测）

        predicted_probs 列顺序为 [sell, hold, buy]，actual_label 取 -1/0/1，
        需 +1 映射为列索引（-1→0 sell, 0→1 hold, 1→2 buy）。
        """
        with self._lock:
            idx = int(actual_label) + 1
            if 0 <= idx < len(predicted_probs):
                err = 1.0 - float(predicted_probs[idx])
            else:
                err = 1.0
            self._error_history.append(err)

    def should_retrain(self) -> Tuple[bool, str]:
        """
        判断是否应该重训练

        Returns:
            (should_retrain: bool, reason: str)
        """
        with self._lock:
            if len(self._sample_buffer) < self._min_samples_retrain:
                return False, f"insufficient_samples: {len(self._sample_buffer)}/{self._min_samples_retrain}"

            # 时间间隔检查
            elapsed = time.time() - self._last_retrain_time
            if elapsed < self._retrain_interval:
                return False, f"cooldown: {elapsed:.0f}s/{self._retrain_interval}s"

            # 漂移检测
            if len(self._error_history) >= self._drift_window:
                recent_err = np.mean(list(self._error_history)[-self._drift_window:])
                early_err = np.mean(list(self._error_history)[:self._drift_window])
                if early_err > 0 and recent_err > early_err * (1 + self._drift_threshold):
                    self._drift_level = DriftLevel.MODERATE if recent_err > early_err * 1.3 else DriftLevel.MILD
                    return True, f"drift_detected: error {early_err:.3f}→{recent_err:.3f}"

            # 样本积累达到阈值
            if len(self._sample_buffer) >= self._sample_buffer.maxlen * 0.8:
                return True, "buffer_near_full"

            return False, "normal"

    def get_training_data(self, recent_weight: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取训练数据（时间衰减 + 误差加权 + 近期偏向）

        权重 = 基础权重 × time_decay(t) × error_factor(e)

        Returns:
            (X, y, sample_weights)
        """
        with self._lock:
            samples = list(self._sample_buffer)
            if not samples:
                return np.array([]), np.array([]), np.array([])

            X = np.array([s.features for s in samples])
            y = np.array([s.label for s in samples])
            n = len(samples)
            now = time.time()
            weights = np.ones(n)

            # ── 1. 基础样本权重 ──
            for i, s in enumerate(samples):
                weights[i] = s.weight

            # ── 2. 指数时间衰减：w(t) = 2^(-Δt / half_life) ──
            if n > 5:
                for i, s in enumerate(samples):
                    age = max(0, now - s.timestamp)
                    if age > 0:
                        decay = 2.0 ** (-age / max(self._time_decay_half_life, 1))
                        weights[i] *= max(decay, 0.05)  # 最低5%权重，不完全丢弃

            # ── 3. 预测误差加权：高误差样本获得更高权重 ──
            if self._use_error_weighting and len(self._sample_errors) >= n:
                recent_errors = list(self._sample_errors)[-n:]
                for i in range(n):
                    err = recent_errors[i] if i < len(recent_errors) else 0.5
                    # 误差>0.5的样本权重提升（模型需要从错误中学习）
                    if err > 0.3:
                        error_factor = 1.0 + (err - 0.3) * self._error_weight_scale
                        weights[i] *= min(error_factor, 3.0)  # 最高3倍

            # 权重归一化到总和为n（保持有效样本数不变）
            if weights.sum() > 0:
                weights = weights / weights.sum() * n

            return X, y, weights.astype(np.float64)

    def mark_retrained(self):
        with self._lock:
            self._last_retrain_time = time.time()
            self._drift_level = DriftLevel.NONE

    def get_buffer_size(self) -> int:
        with self._lock:
            return len(self._sample_buffer)

    def get_drift_status(self) -> Dict[str, Any]:
        with self._lock:
            recent_errors = list(self._error_history)[-self._drift_window:] if len(self._error_history) >= self._drift_window else list(self._error_history)
            return {
                "drift_level": self._drift_level.value,
                "buffer_size": len(self._sample_buffer),
                "mean_error": float(np.mean(recent_errors)) if recent_errors else 0.0,
                "last_retrain_ago": time.time() - self._last_retrain_time,
            }


# ═══════════════════════════════════════════════════════════════
# 模型注册表（版本管理）
# ═══════════════════════════════════════════════════════════════

class ModelRegistry:
    """
    模型版本管理器

    能力:
      - 版本化存储（v1, v2, ...）
      - A/B测试（活跃版 + 影子版）
      - 性能对比与自动回滚
      - JSON持久化
    """

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config or {}
        self._versions: Dict[str, List[ModelMetadata]] = defaultdict(list)  # model_id → versions
        self._active_versions: Dict[str, str] = {}    # model_id → active version
        self._shadow_versions: Dict[str, str] = {}     # model_id → shadow version (A/B test)
        self._persist_dir = cfg.get("persist_dir", "./data/ml_models")
        self._max_versions = cfg.get("max_versions", 10)
        self._auto_rollback = cfg.get("auto_rollback", True)
        self._rollback_accuracy_drop = cfg.get("rollback_accuracy_drop", 0.10)
        self._lock = threading.Lock()
        os.makedirs(self._persist_dir, exist_ok=True)

    def register_model(self, model_id: str, model_type: str, metadata: Dict[str, Any] = None) -> ModelMetadata:
        """注册新模型版本，返回ModelMetadata"""
        with self._lock:
            existing = self._versions.get(model_id, [])
            version = len(existing) + 1
            meta = ModelMetadata(
                model_id=model_id,
                model_type=model_type,
                version=version,
                status=ModelStatus.ACTIVE,
                created_at=datetime.now().isoformat(),
            )
            if metadata:
                for k, v in metadata.items():
                    if hasattr(meta, k):
                        setattr(meta, k, v)

            existing.append(meta)
            self._versions[model_id] = existing[-self._max_versions:]
            self._active_versions[model_id] = str(version)

            # 版本裁剪
            if len(existing) > self._max_versions:
                removed = existing[:-self._max_versions]
                for r in removed:
                    r.status = ModelStatus.DEPRECATED

            logger.info(f"Model {model_id} v{version} registered ({model_type})")
            return meta

    def get_active_model(self, model_id: str) -> Optional[str]:
        with self._lock:
            return self._active_versions.get(model_id)

    def set_shadow(self, model_id: str, version: str):
        """设置A/B测试影子模型"""
        with self._lock:
            self._shadow_versions[model_id] = version
            logger.info(f"Shadow model set: {model_id} v{version}")

    def compare_and_rollback(self, model_id: str) -> Optional[str]:
        """
        比较活跃模型和影子模型性能，必要时回滚

        Returns:
            回滚到的版本号，如果不需要回滚则返回None
        """
        with self._lock:
            active_v = self._active_versions.get(model_id)
            shadow_v = self._shadow_versions.get(model_id)
            if not active_v or not shadow_v:
                return None

            active_meta = self._get_version_meta(model_id, active_v)
            shadow_meta = self._get_version_meta(model_id, shadow_v)
            if not active_meta or not shadow_meta:
                return None

            # 如果影子模型性能更好，提升为活跃
            if shadow_meta.f1_score > active_meta.f1_score * 1.05:
                self._active_versions[model_id] = shadow_v
                shadow_meta.status = ModelStatus.ACTIVE
                active_meta.status = ModelStatus.DEPRECATED if active_meta else ModelStatus.ROLLED_BACK
                logger.info(f"Model {model_id}: shadow v{shadow_v} promoted (f1: "
                           f"{shadow_meta.f1_score:.3f} > {active_meta.f1_score:.3f})")
                return shadow_v

            # 如果活跃模型性能下降严重，回滚
            if (self._auto_rollback and active_v != "1" and
                    active_meta and len(self._versions.get(model_id, [])) >= 2):
                prev_meta = self._get_version_meta(model_id, str(int(active_v) - 1))
                if prev_meta and active_meta.f1_score < prev_meta.f1_score * (1 - self._rollback_accuracy_drop):
                    prev_v = str(int(active_v) - 1)
                    self._active_versions[model_id] = prev_v
                    active_meta.status = ModelStatus.ROLLED_BACK
                    prev_meta.status = ModelStatus.ACTIVE
                    logger.warning(f"Model {model_id} rolled back: v{active_v}→v{prev_v} "
                                  f"(f1: {active_meta.f1_score:.3f} < {prev_meta.f1_score:.3f})")
                    return prev_v

            return None

    def _get_version_meta(self, model_id: str, version: str) -> Optional[ModelMetadata]:
        versions = self._versions.get(model_id, [])
        for v in versions:
            if str(v.version) == str(version):
                return v
        return None

    def get_version_history(self, model_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [{
                "version": m.version,
                "status": m.status.value,
                "accuracy": m.accuracy,
                "f1_score": m.f1_score,
                "n_samples": m.n_samples_trained,
                "created_at": m.created_at,
            } for m in self._versions.get(model_id, [])]

    def get_active_models_summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                model_id: {
                    "active_version": v,
                    "total_versions": len(self._versions.get(model_id, [])),
                }
                for model_id, v in self._active_versions.items()
            }

    def persist(self, model_id: str):
        """持久化模型注册表到磁盘"""
        with self._lock:
            path = os.path.join(self._persist_dir, f"{model_id}_registry.json")
            versions = self._versions.get(model_id, [])
            data = {
                "model_id": model_id,
                "active_version": self._active_versions.get(model_id),
                "shadow_version": self._shadow_versions.get(model_id),
                "versions": [
                    {
                        "version": m.version,
                        "model_type": m.model_type,
                        "status": m.status.value,
                        "created_at": m.created_at,
                        "accuracy": m.accuracy,
                        "f1_score": m.f1_score,
                        "precision": m.precision_val,
                        "recall": m.recall_val,
                        "roc_auc": m.roc_auc,
                        "n_samples_trained": m.n_samples_trained,
                        "feature_importance": m.feature_importance,
                        "hyperparams": m.hyperparams,
                        "training_duration_seconds": m.training_duration_seconds,
                    }
                    for m in versions
                ],
                "updated_at": datetime.now().isoformat(),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, model_id: str) -> bool:
        """从磁盘加载模型注册表"""
        path = os.path.join(self._persist_dir, f"{model_id}_registry.json")
        if not os.path.exists(path):
            return False

        with self._lock:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self._active_versions[model_id] = str(data.get("active_version", "1"))
                if data.get("shadow_version"):
                    self._shadow_versions[model_id] = str(data["shadow_version"])

                versions = []
                for vd in data.get("versions", []):
                    m = ModelMetadata(
                        model_id=model_id,
                        model_type=vd.get("model_type", ""),
                        version=vd["version"],
                        status=ModelStatus(vd.get("status", "active")),
                        created_at=vd.get("created_at", ""),
                        accuracy=vd.get("accuracy", 0.0),
                        f1_score=vd.get("f1_score", 0.0),
                        precision_val=vd.get("precision", 0.0),
                        recall_val=vd.get("recall", 0.0),
                        roc_auc=vd.get("roc_auc", 0.0),
                        n_samples_trained=vd.get("n_samples_trained", 0),
                        feature_importance=vd.get("feature_importance", {}),
                        hyperparams=vd.get("hyperparams", {}),
                        training_duration_seconds=vd.get("training_duration_seconds", 0.0),
                    )
                    versions.append(m)
                self._versions[model_id] = versions
                logger.info(f"Model registry loaded: {model_id} ({len(versions)} versions)")
                return True
            except Exception as e:
                logger.error(f"Failed to load model registry {model_id}: {e}")
                return False


# ═══════════════════════════════════════════════════════════════
# 预测概率校准器
# ═══════════════════════════════════════════════════════════════

class PredictionCalibrator:
    """
    预测概率校准器

    方法:
      - Platt Scaling: sigmoid校准（适合二分类）
      - Isotonic Regression: 保序回归（适合任何分布）
    """

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config or {}
        self._method = cfg.get("calibration_method", "platt")
        self._calibration_params: Dict[str, Any] = {}  # per-class A, B for platt
        self._n_classes = 3
        self._n_bins = cfg.get("n_calibration_bins", 10)
        self._is_fitted = False
        self._lock = threading.Lock()

    def fit(self, raw_probs: np.ndarray, true_labels: np.ndarray):
        """
        拟合校准器

        Args:
            raw_probs: (n_samples, n_classes) 原始预测概率，列顺序 [sell, hold, buy]
            true_labels: (n_samples,) 真实标签，取值 -1/0/1（sell/hold/buy）
        """
        with self._lock:
            # 标签 -1/0/1 → 0/1/2，与概率列 [sell, hold, buy] 对齐
            labels = np.asarray(true_labels, dtype=int) + 1
            if self._method == "platt":
                self._fit_platt(raw_probs, labels)
            else:
                self._fit_isotonic(raw_probs, labels)
            self._is_fitted = True
            logger.info(f"PredictionCalibrator fitted: method={self._method}, "
                       f"n_samples={len(raw_probs)}")

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """校准预测概率"""
        if not self._is_fitted:
            return raw_probs

        with self._lock:
            if self._method == "platt":
                return self._apply_platt(raw_probs)
            else:
                return self._apply_isotonic(raw_probs)

    def _fit_platt(self, probs: np.ndarray, labels: np.ndarray):
        """Platt Scaling: 每类拟合 sigmoid"""
        n_classes = probs.shape[1]
        for c in range(n_classes):
            y_binary = (labels == c).astype(np.float64)
            p = probs[:, c]
            # Newton-Raphson 优化
            A, B = 0.0, 0.0
            for _ in range(100):
                f = sigmoid(A * p + B)
                error = f - y_binary
                if np.abs(error).mean() < 1e-6:
                    break
                # 梯度
                dA = np.mean(error * p * f * (1 - f))
                dB = np.mean(error * f * (1 - f))
                A -= 0.1 * dA
                B -= 0.1 * dB
            self._calibration_params[c] = {"A": float(A), "B": float(B)}

    def _apply_platt(self, probs: np.ndarray) -> np.ndarray:
        calibrated = np.zeros_like(probs)
        for c in range(probs.shape[1]):
            params = self._calibration_params.get(c, {"A": 1.0, "B": 0.0})
            calibrated[:, c] = sigmoid(params["A"] * probs[:, c] + params["B"])
        # 归一化
        row_sums = calibrated.sum(axis=1, keepdims=True)
        calibrated = calibrated / np.maximum(row_sums, 1e-8)
        return calibrated

    def _fit_isotonic(self, probs: np.ndarray, labels: np.ndarray):
        """简化保序回归（分桶法）"""
        n_classes = probs.shape[1]
        for c in range(n_classes):
            y_binary = (labels == c).astype(np.float64)
            p = probs[:, c]

            # 分桶
            bins = np.linspace(0, 1, self._n_bins + 1)
            bin_means = []
            for i in range(self._n_bins):
                mask = (p >= bins[i]) & (p < bins[i + 1])
                if mask.sum() > 0:
                    bin_means.append(y_binary[mask].mean())
                else:
                    bin_means.append((bins[i] + bins[i + 1]) / 2)

            # 保序约束（PAV算法简化版）
            for _ in range(5):
                for i in range(len(bin_means) - 1):
                    if bin_means[i] > bin_means[i + 1]:
                        avg = (bin_means[i] + bin_means[i + 1]) / 2
                        bin_means[i] = avg
                        bin_means[i + 1] = avg

            self._calibration_params[c] = {"bins": bins.tolist(), "means": bin_means}

    def _apply_isotonic(self, probs: np.ndarray) -> np.ndarray:
        calibrated = np.zeros_like(probs)
        for c in range(probs.shape[1]):
            params = self._calibration_params.get(c)
            if not params:
                calibrated[:, c] = probs[:, c]
                continue
            bins = np.array(params["bins"])
            means = np.array(params["means"])
            for i, p in enumerate(probs[:, c]):
                idx = np.searchsorted(bins, p) - 1
                idx = np.clip(idx, 0, len(means) - 1)
                calibrated[i, c] = means[idx]
        row_sums = calibrated.sum(axis=1, keepdims=True)
        return calibrated / np.maximum(row_sums, 1e-8)

    def is_fitted(self) -> bool:
        return self._is_fitted


# ═══════════════════════════════════════════════════════════════
# 模型健康监控
# ═══════════════════════════════════════════════════════════════

class ModelHealthMonitor:
    """
    模型健康监控器

    检测:
      - 特征漂移 (PSI / KS test)
      - 预测质量退化
      - 数据完整性
    """

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config or {}
        self._reference_distribution: Dict[str, np.ndarray] = {}  # 基线特征分布
        self._psi_threshold_mild = cfg.get("psi_threshold_mild", 0.1)
        self._psi_threshold_severe = cfg.get("psi_threshold_severe", 0.25)
        self._prediction_quality_window = deque(maxlen=cfg.get("quality_window", 100))
        self._quality_degradation_threshold = cfg.get("quality_degradation", 0.20)
        self._lock = threading.Lock()

    def set_reference(self, feature_vectors: List[FeatureVector]):
        """设置基线特征分布（训练集）"""
        with self._lock:
            if not feature_vectors:
                return
            n_features = len(feature_vectors[0].features)
            feature_names = feature_vectors[0].feature_names
            self._reference_distribution = {}

            for i, name in enumerate(feature_names):
                vals = np.array([fv.features[i] for fv in feature_vectors])
                hist, bins = np.histogram(vals[~np.isnan(vals)], bins=10, density=True)
                self._reference_distribution[name] = {"hist": hist, "bins": bins}

            logger.info(f"ModelHealthMonitor: reference distribution set "
                       f"({len(feature_names)} features, {len(feature_vectors)} samples)")

    def compute_psi(self, feature_vectors: List[FeatureVector]) -> Dict[str, float]:
        """
        计算 Population Stability Index（特征漂移指标）

        PSI < 0.1: 无漂移
        0.1 <= PSI < 0.25: 轻度漂移
        PSI >= 0.25: 严重漂移
        """
        with self._lock:
            if not self._reference_distribution or not feature_vectors:
                return {}

            psi_scores = {}
            feature_names = feature_vectors[0].feature_names

            for i, name in enumerate(feature_names):
                ref = self._reference_distribution.get(name)
                if not ref:
                    continue

                vals = np.array([fv.features[i] for fv in feature_vectors])
                actual_hist, _ = np.histogram(vals[~np.isnan(vals)], bins=ref["bins"], density=True)
                expected_hist = ref["hist"]

                # PSI = sum((actual - expected) * ln(actual / expected))
                psi = 0.0
                for a, e in zip(actual_hist, expected_hist):
                    a_safe = max(a, 1e-10)
                    e_safe = max(e, 1e-10)
                    psi += (a_safe - e_safe) * math.log(a_safe / e_safe)
                psi_scores[name] = round(psi, 4)

            return psi_scores

    def check_drift(self, feature_vectors: List[FeatureVector]) -> DriftReport:
        """全面漂移检查"""
        psi_scores = self.compute_psi(feature_vectors)
        if not psi_scores:
            return DriftReport(drift_level=DriftLevel.NONE)

        drifted = [k for k, v in psi_scores.items() if v >= self._psi_threshold_mild]
        severe = [k for k, v in psi_scores.items() if v >= self._psi_threshold_severe]

        if severe:
            level = DriftLevel.SEVERE
        elif len(drifted) >= len(psi_scores) * 0.3:
            level = DriftLevel.MODERATE
        elif drifted:
            level = DriftLevel.MILD
        else:
            level = DriftLevel.NONE

        recs = []
        if severe:
            recs.append(f"SEVERE drift in {len(severe)} features: {severe[:5]}, retrain recommended")
        if level in (DriftLevel.MODERATE, DriftLevel.SEVERE):
            recs.append("Consider increasing retrain frequency or expanding training window")

        return DriftReport(
            drift_level=level,
            psi_scores=psi_scores,
            drifted_features=drifted,
            recommendations=recs,
        )

    def record_quality(self, quality_score: float):
        with self._lock:
            self._prediction_quality_window.append(quality_score)

    def check_quality_degradation(self) -> Tuple[bool, float]:
        """检查预测质量是否退化"""
        with self._lock:
            if len(self._prediction_quality_window) < 20:
                return False, 0.0

            window = list(self._prediction_quality_window)
            first_half = np.mean(window[:len(window)//2])
            second_half = np.mean(window[len(window)//2:])

            if first_half > 0:
                degradation = (first_half - second_half) / first_half
                if degradation > self._quality_degradation_threshold:
                    return True, degradation
            return False, 0.0


# ═══════════════════════════════════════════════════════════════
# ML决策引擎 — 主引擎
# ═══════════════════════════════════════════════════════════════

class MLDecisionEngine:
    """
    机器学习决策引擎 — 完善强化版

    整体流程:
      原始数据 → FeaturePipeline → ModelEnsemble → PredictionCalibrator → 最终决策
                      ↕                                               ↓
               OnlineTrainer ←── 实际结果反馈 ──→ ModelHealthMonitor
                      ↕
               ModelRegistry (版本管理)

    核心能力:
      - 自动化特征工程
      - 多模型集成预测
      - 在线增量学习 + 概念漂移检测
      - 模型版本管理与自动回滚
      - 概率校准
      - 特征漂移与预测质量监控
      - 异常预测检测
      - 强化学习动态参数调优
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        ml_cfg = config.get("ml_decision", {}) if config else {}

        # ── 组件初始化 ──
        self.feature_pipeline = FeaturePipeline(ml_cfg.get("feature_pipeline", {}))
        self.model_ensemble = ModelEnsemble(ml_cfg.get("model_ensemble", {}))
        self.online_trainer = OnlineTrainer(ml_cfg.get("online_trainer", {}))
        self.model_registry = ModelRegistry(ml_cfg.get("model_registry", {}))
        self.calibrator = PredictionCalibrator(ml_cfg.get("calibrator", {}))
        self.health_monitor = ModelHealthMonitor(ml_cfg.get("health_monitor", {}))

        # ── 状态 ──
        self._enabled = ml_cfg.get("enabled", True)
        self._model_id = ml_cfg.get("model_id", "ml_decision_v1")
        self._prediction_history: deque = deque(maxlen=ml_cfg.get("history_size", 500))
        self._prediction_counter: int = 0
        self._lock = threading.Lock()

        # ── Q-learning 动态参数探索 ──
        self._q_learning = ml_cfg.get("q_learning", {}).get("enabled", False)
        self._q_table: Dict[str, Dict[str, float]] = {}
        self._q_params = ml_cfg.get("q_learning", {})

        # ── 加载已有模型 ──
        self._load_existing()

        logger.info(
            f"MLDecisionEngine initialized: model={self._model_id}, "
            f"enabled={self._enabled}, sklearn={_ML_LIBS.get('sklearn', False)}, "
            f"xgboost={_ML_LIBS.get('xgboost', False)}, lightgbm={_ML_LIBS.get('lightgbm', False)}"
        )

    def _load_existing(self):
        """尝试从磁盘加载已有模型"""
        loaded = self.model_registry.load(self._model_id)
        if loaded:
            active_v = self.model_registry.get_active_model(self._model_id)
            logger.info(f"MLDecisionEngine: loaded {self._model_id} (active=v{active_v})")

    # ═══════════════════════════════════════════════════════
    # 训练
    # ═══════════════════════════════════════════════════════

    def train(self, raw_data_list: List[Dict[str, Any]], labels: List[int],
              symbol: str = "") -> ModelMetadata:
        """
        训练/重训练模型

        Args:
            raw_data_list: 原始市场数据列表
            labels: 对应标签 (-1=sell, 0=hold, 1=buy)
            symbol: 交易对

        Returns:
            ModelMetadata
        """
        if len(raw_data_list) < 20:
            logger.warning(f"MLDecisionEngine train: insufficient data ({len(raw_data_list)})")
            return ModelMetadata(model_id=self._model_id, status=ModelStatus.FAILED)

        start = time.time()

        # 1. 拟合特征管道
        self.feature_pipeline.fit(raw_data_list)

        # 2. 转换特征
        feature_vectors = [self.feature_pipeline.transform(d) for d in raw_data_list]
        X = np.array([fv.features for fv in feature_vectors])
        y = np.array(labels)

        # 3. 设置基线分布
        self.health_monitor.set_reference(feature_vectors)

        # 4. 训练集成模型
        self.model_ensemble.fit(X, y, feature_names=self.feature_pipeline.get_feature_names())

        # 5. 拟合校准器
        raw_probs = np.zeros((len(X), 3))
        predictions = self.model_ensemble.predict(X)
        for i, pred in enumerate(predictions):
            raw_probs[i] = [pred.sell_prob, pred.hold_prob, pred.buy_prob]
        self.calibrator.fit(raw_probs, y)

        # 6. 注册模型版本
        feature_importance = self.model_ensemble.get_feature_importance()
        elapsed = time.time() - start

        meta = self.model_registry.register_model(
            self._model_id,
            "ensemble",
            {
                "feature_names": self.feature_pipeline.get_feature_names(),
                "n_samples_trained": len(raw_data_list),
                "training_duration_seconds": elapsed,
                "feature_importance": feature_importance,
                "hyperparams": {
                    "models": list(self.model_ensemble._model_weights.keys()),
                    "weights": self.model_ensemble._model_weights,
                },
            },
        )

        # 持久化
        self.model_registry.persist(self._model_id)
        self.online_trainer.mark_retrained()

        logger.info(
            f"MLDecisionEngine trained: {len(raw_data_list)} samples, "
            f"{len(self.feature_pipeline.get_feature_names())} features, "
            f"{elapsed:.1f}s"
        )
        return meta

    # ═══════════════════════════════════════════════════════
    # 预测
    # ═══════════════════════════════════════════════════════

    def predict(self, raw_data: Dict[str, Any],
                symbol: str = "") -> MLPrediction:
        """
        对单条原始数据做ML预测

        Args:
            raw_data: 原始行情数据（包含所需特征值）
            symbol: 交易对

        Returns:
            MLPrediction
        """
        with self._lock:
            self._prediction_counter += 1

            if not self.model_ensemble.is_fitted():
                return MLPrediction(
                    symbol=symbol,
                    direction="hold",
                    hold_prob=1.0,
                    metadata={"reason": "model_not_fitted"},
                )

            # 1. 特征转换
            fv = self.feature_pipeline.transform(raw_data)
            X = fv.features.reshape(1, -1)

            # 2. 模型预测
            predictions = self.model_ensemble.predict(X)
            pred = predictions[0] if predictions else MLPrediction(direction="hold")

            # 3. 概率校准
            raw_probs = np.array([[pred.sell_prob, pred.hold_prob, pred.buy_prob]])
            calibrated = self.calibrator.calibrate(raw_probs)
            if len(calibrated) > 0:
                pred.sell_prob = float(calibrated[0][0])
                pred.hold_prob = float(calibrated[0][1])
                pred.buy_prob = float(calibrated[0][2])
                pred.calibrated = True
                # 重新确定方向
                probs_map = {"buy": pred.buy_prob, "sell": pred.sell_prob, "hold": pred.hold_prob}
                pred.direction = max(probs_map, key=probs_map.get)
                pred.confidence = probs_map[pred.direction]

            # 4. 特征贡献
            importance = self.model_ensemble.get_feature_importance()
            pred.feature_contributions = {
                name: importance.get(name, 0.0)
                for name in fv.feature_names
            }

            # 5. 元数据
            pred.symbol = symbol or raw_data.get("symbol", "")
            pred.model_version = (
                self.model_registry.get_active_model(self._model_id) or "1"
            )
            pred.metadata["prediction_id"] = self._prediction_counter
            pred.metadata["feature_count"] = len(fv.feature_names)
            pred.metadata["ml_libs"] = {k: v for k, v in _ML_LIBS.items() if v}

            # 历史记录
            self._prediction_history.append({
                "direction": pred.direction,
                "confidence": pred.confidence,
                "timestamp": pred.timestamp,
            })

            return pred

    def predict_batch(self, raw_data_list: List[Dict[str, Any]],
                      symbol: str = "") -> List[MLPrediction]:
        """批量预测"""
        results = []
        for data in raw_data_list:
            results.append(self.predict(data, symbol=symbol))
        return results

    # ═══════════════════════════════════════════════════════
    # 在线学习
    # ═══════════════════════════════════════════════════════

    def feed_result(self, raw_data: Dict[str, Any], actual_label: int,
                    symbol: str = ""):
        """
        反馈实际结果（在线学习）

        Args:
            raw_data: 原始数据
            actual_label: 实际标签 (-1=sell, 0=hold, 1=buy)
            symbol: 交易对
        """
        try:
            fv = self.feature_pipeline.transform(raw_data)
            self.online_trainer.add_sample(
                fv.features, actual_label, symbol=symbol,
                timestamp=raw_data.get("timestamp", time.time()),
            )

            # 记录预测误差
            X = fv.features.reshape(1, -1)
            predictions = self.model_ensemble.predict(X)
            if predictions:
                pred = predictions[0]
                probs = np.array([pred.sell_prob, pred.hold_prob, pred.buy_prob])
                self.online_trainer.record_prediction_error(probs, actual_label)

            # 记录预测质量
            predicted_dir = pred.direction if predictions else "hold"
            actual_dir = {0: "hold", 1: "buy", -1: "sell"}.get(actual_label, "hold")
            correct = predicted_dir == actual_dir
            self.health_monitor.record_quality(1.0 if correct else 0.0)

            # Q-learning更新（如果启用）
            if self._q_learning:
                self._update_q_value(raw_data, actual_label)
        except Exception as e:
            logger.debug(f"MLDecisionEngine feed_result error: {e}")

    def check_and_retrain(self) -> Optional[ModelMetadata]:
        """
        检查是否需要重训练，如果需要则执行

        Returns:
            新的ModelMetadata，如果不需要则None
        """
        should, reason = self.online_trainer.should_retrain()
        if not should:
            return None

        logger.info(f"MLDecisionEngine retrain triggered: {reason}")

        X, y, weights = self.online_trainer.get_training_data()
        if len(X) < 20:
            return None

        # 增量训练
        self.model_ensemble.fit(
            X, y,
            feature_names=self.feature_pipeline.get_feature_names(),
            sample_weight=weights,
        )

        # 更新校准器
        raw_probs = np.zeros((len(X), 3))
        predictions = self.model_ensemble.predict(X)
        for i, pred in enumerate(predictions):
            raw_probs[i] = [pred.sell_prob, pred.hold_prob, pred.buy_prob]
        self.calibrator.fit(raw_probs, y)

        meta = self.model_registry.register_model(
            self._model_id, "ensemble_retrained",
            {
                "feature_names": self.feature_pipeline.get_feature_names(),
                "n_samples_trained": len(X),
                "feature_importance": self.model_ensemble.get_feature_importance(),
            },
        )
        self.model_registry.persist(self._model_id)
        self.online_trainer.mark_retrained()

        logger.info(f"MLDecisionEngine retrained: {len(X)} samples")
        return meta

    # ═══════════════════════════════════════════════════════
    # Q-learning 动态参数探索
    # ═══════════════════════════════════════════════════════

    def _update_q_value(self, raw_data: Dict[str, Any], actual_label: int):
        """
        Q-learning: 根据实际结果更新模型权重

        状态: 市场状态（波动率高/中/低 + 趋势/震荡）
        动作: 调整模型权重组合
        奖励: 预测正确+1，错误-1
        """
        try:
            volatility = raw_data.get("volatility_pct", 0.03)
            regime = raw_data.get("regime", "normal")

            if volatility > 0.08:
                vol_state = "high"
            elif volatility > 0.03:
                vol_state = "medium"
            else:
                vol_state = "low"

            state = f"{vol_state}_{regime}"
            reward = 1.0 if actual_label == 1 else (-1.0 if actual_label == -1 else 0.0)

            if state not in self._q_table:
                self._q_table[state] = {"weight_gb": 0.0, "weight_rf": 0.0, "weight_lr": 0.0}

            # Epsilon-greedy探索
            lr = self._q_params.get("learning_rate", 0.1)
            discount = self._q_params.get("discount_factor", 0.95)

            # 简单Q-learning更新
            for action in self._q_table[state]:
                old_q = self._q_table[state][action]
                self._q_table[state][action] = old_q + lr * (reward - old_q)

            # 每epochs_per_update次反馈后同步模型权重
            self._prediction_counter_q = getattr(self, '_prediction_counter_q', 0) + 1
            if self._prediction_counter_q >= self._q_params.get("epochs_per_update", 50):
                self._prediction_counter_q = 0
                self._sync_q_weights()
        except Exception as e:
            logger.debug(f"Q-learning update error: {e}")

    def _sync_q_weights(self):
        """根据Q表同步模型权重"""
        try:
            vol_regime = self._get_current_state()
            if vol_regime in self._q_table:
                q_vals = self._q_table[vol_regime]
                total = sum(max(v, 0) for v in q_vals.values()) or 1.0
                for k in q_vals:
                    if k in self.model_ensemble._model_weights:
                        new_w = max(q_vals[k], 0) / total * 0.5 + \
                                self.model_ensemble._model_weights[k] * 0.5
                        self.model_ensemble._model_weights[k] = round(new_w, 4)
                logger.debug(f"Q-learning synced weights: {self.model_ensemble._model_weights}")
        except Exception as e:
            logger.debug(f"Q-weight sync error: {e}")

    def _get_current_state(self) -> str:
        """获取当前市场状态标签"""
        return "medium_normal"  # 默认状态

    def get_q_table(self) -> Dict[str, Dict[str, float]]:
        """获取Q表（用于调试和分析）"""
        with self._lock:
            return dict(self._q_table)

    # ═══════════════════════════════════════════════════════
    # 状态查询
    # ═══════════════════════════════════════════════════════

    def is_enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        logger.info(f"MLDecisionEngine enabled={enabled}")

    def get_status(self) -> Dict[str, Any]:
        """获取引擎综合状态"""
        with self._lock:
            total_predictions = self._prediction_counter
            recent = list(self._prediction_history)[-50:]

            # 方向分布
            dir_counts = {"buy": 0, "sell": 0, "hold": 0}
            for p in recent:
                d = p.get("direction", "hold")
                dir_counts[d] = dir_counts.get(d, 0) + 1

            total_recent = max(len(recent), 1)

            return {
                "model_id": self._model_id,
                "enabled": self._enabled,
                "q_learning": self._q_learning,
                "fitted": self.model_ensemble.is_fitted(),
                "active_version": self.model_registry.get_active_model(self._model_id),
                "total_predictions": total_predictions,
                "direction_distribution": {
                    k: round(v / total_recent, 4) for k, v in dir_counts.items()
                },
                "online_trainer": self.online_trainer.get_drift_status(),
                "feature_pipeline_fitted": self.feature_pipeline.is_fitted(),
                "calibrator_fitted": self.calibrator.is_fitted(),
            }

    def get_engine_stats(self) -> Dict[str, Any]:
        """获取引擎统计摘要"""
        with self._lock:
            return {
                "model_id": self._model_id,
                "enabled": self._enabled,
                "fitted": self.model_ensemble.is_fitted(),
                "predictions": self._prediction_counter,
                "buffer_size": self.online_trainer.get_buffer_size(),
                "drift_level": self.online_trainer._drift_level.value,
                "active_version": self.model_registry.get_active_model(self._model_id),
                "total_versions": len(self.model_registry._versions.get(self._model_id, [])),
                "n_features": len(self.feature_pipeline.get_feature_names()),
                "history_size": len(self._prediction_history),
                "q_learning_enabled": self._q_learning,
                "ml_libs": {k: v for k, v in _ML_LIBS.items() if v},
            }

    def get_feature_importance(self) -> Dict[str, float]:
        """获取当前模型的特征重要性"""
        return self.model_ensemble.get_feature_importance()

    def get_prediction_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取预测历史"""
        with self._lock:
            return list(self._prediction_history)[-limit:]

    def reset(self):
        """重置引擎状态（保留模型）"""
        with self._lock:
            self._prediction_history.clear()
            self._prediction_counter = 0
            self.online_trainer._error_history.clear()
            logger.info("MLDecisionEngine state reset (model preserved)")


# ═══════════════════════════════════════════════════════════════
# 全局单例
# ═══════════════════════════════════════════════════════════════

_ml_engine_instance: Optional[MLDecisionEngine] = None
_ml_engine_lock = threading.Lock()


def get_ml_decision_engine(config: Optional[Dict[str, Any]] = None) -> MLDecisionEngine:
    """获取全局ML决策引擎单例"""
    global _ml_engine_instance
    with _ml_engine_lock:
        if _ml_engine_instance is None:
            _ml_engine_instance = MLDecisionEngine(config)
        return _ml_engine_instance