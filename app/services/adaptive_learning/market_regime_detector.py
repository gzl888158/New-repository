"""
市场状态检测器 - HMM市场状态分类、多时间周期分析、状态转移预测与策略映射

核心能力:
  - HMM隐马尔可夫模型进行市场状态分类 (自实现，无外部ML库)
  - 多时间周期联合分析 (短/中/长周期)
  - Markov链状态转移预测
  - 策略-市场状态最优映射
  - 早期预警信号检测
"""
import asyncio
import json
import math
import os
import time
from collections import OrderedDict, deque
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from app.services.enterprise import EnterpriseServiceMixin


# ==================== 市场状态枚举 ====================

class MarketRegime(Enum):
    TRENDING_UP = "trending_up"       # 强势上升
    TRENDING_DOWN = "trending_down"   # 强势下跌
    RANGING = "ranging"               # 横盘震荡
    HIGH_VOLATILITY = "high_vol"       # 高波动
    LOW_VOLATILITY = "low_vol"        # 低波动
    BREAKOUT = "breakout"             # 突破
    REVERSAL = "reversal"             # 反转
    UNKNOWN = "unknown"


# 状态到索引映射
_REGIME_ORDER = [
    MarketRegime.TRENDING_UP,
    MarketRegime.TRENDING_DOWN,
    MarketRegime.RANGING,
    MarketRegime.HIGH_VOLATILITY,
    MarketRegime.LOW_VOLATILITY,
    MarketRegime.BREAKOUT,
    MarketRegime.REVERSAL,
]
_REGIME_TO_IDX = {r: i for i, r in enumerate(_REGIME_ORDER)}
_NUM_REGIMES = len(_REGIME_ORDER)


# ==================== HMM 状态分类器 ====================

class HMMRegimeClassifier:
    """HMM隐马尔可夫模型 - 市场状态分类
    
    使用Gaussian发射概率的自实现HMM。
    隐藏状态映射到市场状态，通过Baum-Welch(前向-后向)算法训练。
    """

    def __init__(self, n_states: int = _NUM_REGIMES, n_features: int = 10):
        self.n_states = n_states
        self.n_features = n_features
        
        # 初始状态概率 π
        self.pi = [1.0 / n_states] * n_states
        
        # 状态转移矩阵 A (n_states x n_states)
        self.A = [[1.0 / n_states] * n_states for _ in range(n_states)]
        
        # Gaussian发射参数: mean (n_states x n_features), var (n_states x n_features)
        self.means = [[0.0] * n_features for _ in range(n_states)]
        self.vars = [[1.0] * n_features for _ in range(n_states)]
        
        # 状态到regime映射
        self._state_to_regime: Dict[int, MarketRegime] = {}
        self._regime_to_state: Dict[MarketRegime, int] = {}
        
        # 训练统计
        self._log_likelihood_history: List[float] = []
        self._trained = False
        self._n_iterations = 0

    def _gaussian_pdf(self, x: float, mean: float, var: float) -> float:
        """单变量Gaussian概率密度"""
        if var <= 0:
            var = 1e-6
        exponent = -0.5 * ((x - mean) ** 2) / var
        return (1.0 / math.sqrt(2 * math.pi * var)) * math.exp(exponent)

    def _log_gaussian_pdf(self, x: float, mean: float, var: float) -> float:
        """单变量Gaussian对数概率密度（避免exp下溢）"""
        if var <= 0:
            var = 1e-6
        return -0.5 * (math.log(2 * math.pi * var) + ((x - mean) ** 2) / var)

    def _emission_prob(self, obs: List[float], state: int) -> float:
        """计算给定状态下的观测发射概率（各特征独立Gaussian乘积）"""
        prob = 1.0
        for j in range(min(len(obs), self.n_features)):
            p = self._gaussian_pdf(obs[j], self.means[state][j], self.vars[state][j])
            prob *= max(p, 1e-300)
        return prob

    def _log_emission_prob(self, obs: List[float], state: int) -> float:
        """计算给定状态下的观测对数发射概率（各特征独立Gaussian对数和）"""
        log_prob = 0.0
        for j in range(min(len(obs), self.n_features)):
            log_prob += self._log_gaussian_pdf(obs[j], self.means[state][j], self.vars[state][j])
        return log_prob

    @staticmethod
    def _log_sum_exp(log_values: List[float]) -> float:
        """数值稳定的log-sum-exp：log(sum(exp(log_values)))"""
        if not log_values:
            return float('-inf')
        max_log = max(log_values)
        if max_log == float('-inf'):
            return float('-inf')
        return max_log + math.log(sum(math.exp(lv - max_log) for lv in log_values))

    def _forward(self, observations: List[List[float]]) -> Tuple[List[List[float]], List[float]]:
        """前向算法: 计算alpha和缩放因子"""
        T = len(observations)
        alpha = [[0.0] * self.n_states for _ in range(T)]
        scale = [0.0] * T
        
        # 初始化
        for i in range(self.n_states):
            alpha[0][i] = self.pi[i] * self._emission_prob(observations[0], i)
        scale[0] = sum(alpha[0])
        if scale[0] > 0:
            for i in range(self.n_states):
                alpha[0][i] /= scale[0]
        
        # 递推
        for t in range(1, T):
            for j in range(self.n_states):
                s = 0.0
                for i in range(self.n_states):
                    s += alpha[t-1][i] * self.A[i][j]
                alpha[t][j] = s * self._emission_prob(observations[t], j)
            scale[t] = sum(alpha[t])
            if scale[t] > 0:
                for j in range(self.n_states):
                    alpha[t][j] /= scale[t]
        
        return alpha, scale

    def _log_forward(self, observations: List[List[float]]) -> Tuple[List[List[float]], float]:
        """对数空间前向算法: 计算log_alpha和对数似然（数值稳定）"""
        T = len(observations)
        log_alpha = [[float('-inf')] * self.n_states for _ in range(T)]
        log_pi = [math.log(max(p, 1e-300)) for p in self.pi]
        log_A = [[math.log(max(self.A[i][j], 1e-300)) for j in range(self.n_states)] for i in range(self.n_states)]
        
        # 初始化
        for i in range(self.n_states):
            log_alpha[0][i] = log_pi[i] + self._log_emission_prob(observations[0], i)
        
        # 递推
        for t in range(1, T):
            for j in range(self.n_states):
                log_terms = [log_alpha[t-1][i] + log_A[i][j] for i in range(self.n_states)]
                log_alpha[t][j] = self._log_sum_exp(log_terms) + self._log_emission_prob(observations[t], j)
        
        # 对数似然 = log(sum(alpha[T-1]))
        log_likelihood = self._log_sum_exp(log_alpha[T-1])
        return log_alpha, log_likelihood

    def _log_backward(self, observations: List[List[float]]) -> List[List[float]]:
        """对数空间后向算法: 计算log_beta（数值稳定）"""
        T = len(observations)
        log_beta = [[float('-inf')] * self.n_states for _ in range(T)]
        log_A = [[math.log(max(self.A[i][j], 1e-300)) for j in range(self.n_states)] for i in range(self.n_states)]
        
        # 初始化
        for i in range(self.n_states):
            log_beta[T-1][i] = 0.0  # log(1) = 0
        
        # 递推
        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                log_terms = [
                    log_A[i][j] + self._log_emission_prob(observations[t+1], j) + log_beta[t+1][j]
                    for j in range(self.n_states)
                ]
                log_beta[t][i] = self._log_sum_exp(log_terms)
        
        return log_beta

    def _baum_welch_step(self, observations: List[List[float]]) -> float:
        """一步Baum-Welch更新，返回对数似然（使用对数空间避免下溢）"""
        T = len(observations)
        if T < 2:
            return 0.0
        
        log_alpha, log_likelihood = self._log_forward(observations)
        log_beta = self._log_backward(observations)
        
        # 计算log_gamma和log_xi（对数空间）
        log_gamma = [[float('-inf')] * self.n_states for _ in range(T)]
        log_xi = [[[float('-inf')] * self.n_states for _ in range(self.n_states)] for _ in range(T - 1)]
        
        log_A = [[math.log(max(self.A[i][j], 1e-300)) for j in range(self.n_states)] for i in range(self.n_states)]
        
        # 计算gamma
        for t in range(T):
            log_terms = [log_alpha[t][i] + log_beta[t][i] for i in range(self.n_states)]
            log_norm = self._log_sum_exp(log_terms)
            for i in range(self.n_states):
                log_gamma[t][i] = log_alpha[t][i] + log_beta[t][i] - log_norm
        
        # 计算xi
        for t in range(T - 1):
            log_terms_flat = []
            for i in range(self.n_states):
                for j in range(self.n_states):
                    log_val = (log_alpha[t][i] + log_A[i][j] + 
                              self._log_emission_prob(observations[t+1], j) + log_beta[t+1][j])
                    log_terms_flat.append(log_val)
            log_norm = self._log_sum_exp(log_terms_flat)
            idx = 0
            for i in range(self.n_states):
                for j in range(self.n_states):
                    log_xi[t][i][j] = (log_alpha[t][i] + log_A[i][j] + 
                                       self._log_emission_prob(observations[t+1], j) + log_beta[t+1][j] - log_norm)
                    idx += 1
        
        # 更新pi
        for i in range(self.n_states):
            self.pi[i] = math.exp(log_gamma[0][i])
        
        # 更新A
        for i in range(self.n_states):
            log_sum_gamma = self._log_sum_exp([log_gamma[t][i] for t in range(T - 1)])
            for j in range(self.n_states):
                if log_sum_gamma > float('-inf'):
                    log_xi_sum = self._log_sum_exp([log_xi[t][i][j] for t in range(T - 1)])
                    self.A[i][j] = math.exp(log_xi_sum - log_sum_gamma)
                else:
                    self.A[i][j] = 1.0 / self.n_states
        
        # 更新Gaussian参数
        for i in range(self.n_states):
            log_sum_gamma_i = self._log_sum_exp([log_gamma[t][i] for t in range(T)])
            if log_sum_gamma_i > math.log(1e-6):
                # 转换回概率空间用于加权平均（此时gamma已归一化，不会下溢）
                gamma_i = [math.exp(log_gamma[t][i] - log_sum_gamma_i) for t in range(T)]
                sum_gamma_i = sum(gamma_i)
                if sum_gamma_i > 1e-6:
                    for k in range(self.n_features):
                        # mean
                        self.means[i][k] = sum(gamma_i[t] * observations[t][k] for t in range(T)) / sum_gamma_i
                        # variance
                        self.vars[i][k] = sum(
                            gamma_i[t] * (observations[t][k] - self.means[i][k]) ** 2
                            for t in range(T)
                        ) / sum_gamma_i
                        self.vars[i][k] = max(self.vars[i][k], 1e-6)
        
        # 对数似然（已从log_forward返回）
        return log_likelihood

    def fit(self, observations: List[List[float]], n_iterations: int = 50,
             convergence_threshold: float = 1e-4) -> Dict[str, Any]:
        """训练HMM模型
        
        Args:
            observations: 观测序列，每行一个时间点，每列一个特征
            n_iterations: 最大迭代次数
            convergence_threshold: 收敛阈值
        
        Returns:
            训练结果 {log_likelihood, iterations, converged}
        """
        if len(observations) < self.n_states:
            logger.warning(f"Insufficient data for HMM training: {len(observations)} < {self.n_states}")
            return {"log_likelihood": 0, "iterations": 0, "converged": False}
        
        # 初始化Gaussian参数 (KMeans-like initialization)
        n_samples = len(observations)
        segment_size = max(1, n_samples // self.n_states)
        for i in range(self.n_states):
            start = i * segment_size
            end = min(start + segment_size, n_samples)
            segment = observations[start:end]
            if segment:
                for k in range(self.n_features):
                    vals = [obs[k] for obs in segment]
                    self.means[i][k] = sum(vals) / len(vals)
                    var = sum((v - self.means[i][k]) ** 2 for v in vals) / len(vals)
                    self.vars[i][k] = max(var, 1e-6)
        
        prev_ll = float('-inf')
        self._log_likelihood_history = []
        
        for it in range(n_iterations):
            ll = self._baum_welch_step(observations)
            self._log_likelihood_history.append(ll)
            self._n_iterations = it + 1
            
            if abs(ll - prev_ll) < convergence_threshold:
                self._trained = True
                self._map_states_to_regimes()
                return {"log_likelihood": ll, "iterations": it + 1, "converged": True}
            prev_ll = ll
        
        self._trained = True
        self._map_states_to_regimes()
        return {"log_likelihood": prev_ll, "iterations": n_iterations, "converged": False}

    def _map_states_to_regimes(self):
        """自动将HMM隐藏状态映射到市场状态"""
        self._state_to_regime = {}
        self._regime_to_state = {}
        
        mean_trend = [0.0] * self.n_states
        mean_vol = [0.0] * self.n_states
        for i in range(self.n_states):
            # 趋势 = 平均收益率特征 (index 0)
            mean_trend[i] = self.means[i][0] if self.n_features > 0 else 0.0
            # 波动 = 平均波动率特征 (index 1)
            mean_vol[i] = self.means[i][1] if self.n_features > 1 else 1.0
        
        n = self.n_states
        if n >= 7:
            available = set(range(n))

            def assign(regime: MarketRegime, score, maximize: bool = True) -> None:
                if not available:
                    return
                if maximize:
                    state = max(available, key=lambda candidate: (score(candidate), -candidate))
                else:
                    state = min(available, key=lambda candidate: (score(candidate), candidate))
                self._state_to_regime[state] = regime
                available.remove(state)

            assign(MarketRegime.TRENDING_UP, lambda state: mean_trend[state])
            assign(MarketRegime.TRENDING_DOWN, lambda state: mean_trend[state], maximize=False)
            assign(MarketRegime.HIGH_VOLATILITY, lambda state: mean_vol[state])
            assign(MarketRegime.LOW_VOLATILITY, lambda state: mean_vol[state], maximize=False)
            assign(
                MarketRegime.BREAKOUT,
                lambda state: mean_vol[state] + abs(mean_trend[state] - 0.5),
            )
            acceleration_index = 9 if self.n_features > 9 else 0
            assign(
                MarketRegime.REVERSAL,
                lambda state: abs(self.means[state][acceleration_index] - 0.5),
            )
            for state in available:
                self._state_to_regime[state] = MarketRegime.RANGING
        else:
            for i in range(n):
                if i not in self._state_to_regime:
                    self._state_to_regime[i] = _REGIME_ORDER[i % len(_REGIME_ORDER)]
        
        for state, regime in self._state_to_regime.items():
            self._regime_to_state[regime] = state

    def predict(self, features: List[float]) -> MarketRegime:
        """预测当前观测所属市场状态"""
        if not self._trained:
            return MarketRegime.UNKNOWN
        
        probs = self.predict_proba(features)
        if not probs:
            return MarketRegime.UNKNOWN
        
        best_regime = max(probs, key=lambda k: probs[k])
        return best_regime if probs[best_regime] > 0 else MarketRegime.UNKNOWN

    def predict_proba(self, features: List[float]) -> Dict[MarketRegime, float]:
        """返回各市场状态的概率分布"""
        return self.predict_proba_sequence([features])

    def predict_proba_sequence(self, observations: List[List[float]]) -> Dict[MarketRegime, float]:
        """对观测序列执行前向过滤，返回最后时点的市场状态后验概率。"""
        if not self._trained:
            return {r: 0.0 for r in _REGIME_ORDER}

        if not observations:
            return {r: 0.0 for r in _REGIME_ORDER}

        alpha, _ = self._forward(observations)
        state_probs = alpha[-1]
        total = sum(state_probs)
        result = {regime: 0.0 for regime in _REGIME_ORDER}
        if total > 0.0:
            for state, regime in self._state_to_regime.items():
                result[regime] += state_probs[state] / total
        elif self._state_to_regime:
            for regime in self._state_to_regime.values():
                result[regime] += 1.0 / len(self._state_to_regime)
        return result

    def get_transition_matrix(self) -> List[List[float]]:
        """获取状态转移矩阵"""
        return [row[:] for row in self.A]

    def get_regime_transition_matrix(self) -> List[List[float]]:
        """按市场状态枚举顺序返回转移矩阵，而非隐藏状态编号顺序。"""
        matrix = [[0.0] * _NUM_REGIMES for _ in range(_NUM_REGIMES)]
        for from_state, from_regime in self._state_to_regime.items():
            if from_regime not in _REGIME_TO_IDX or from_state >= len(self.A):
                continue
            from_index = _REGIME_TO_IDX[from_regime]
            for to_state, to_regime in self._state_to_regime.items():
                if to_regime not in _REGIME_TO_IDX or to_state >= len(self.A[from_state]):
                    continue
                to_index = _REGIME_TO_IDX[to_regime]
                matrix[from_index][to_index] += self.A[from_state][to_state]
        return matrix

    def get_state_means(self) -> List[List[float]]:
        """获取各状态均值"""
        return [row[:] for row in self.means]

    def get_state_vars(self) -> List[List[float]]:
        """获取各状态方差"""
        return [row[:] for row in self.vars]

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "n_states": self.n_states,
            "n_features": self.n_features,
            "pi": self.pi,
            "A": self.A,
            "means": self.means,
            "vars": self.vars,
            "state_to_regime": {str(k): v.value for k, v in self._state_to_regime.items()},
            "trained": self._trained,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HMMRegimeClassifier":
        """从字典反序列化"""
        hmm = cls(n_states=d["n_states"], n_features=d["n_features"])
        hmm.pi = d["pi"]
        hmm.A = d["A"]
        hmm.means = d["means"]
        hmm.vars = d["vars"]
        for state_str, regime_str in d.get("state_to_regime", {}).items():
            state = int(state_str)
            regime = MarketRegime(regime_str)
            hmm._state_to_regime[state] = regime
            hmm._regime_to_state[regime] = state
        hmm._trained = d.get("trained", False)
        return hmm


# ==================== 特征提取器 ====================

class RegimeFeatureExtractor:
    """从OHLCV数据提取市场状态特征"""

    def __init__(self, config: Dict[str, Any] = None):
        self._config = config or {}
        
        # 特征维度
        self._feature_names = [
            "log_return",           # 对数收益率
            "realized_volatility",  # 已实现波动率
            "volume_ratio",         # 成交量比率
            "price_ma_distance",    # 价格-MA距离
            "bollinger_position",   # 布林带位置
            "rsi",                  # RSI
            "atr_ratio",            # ATR比率
            "spread_proxy",         # 买卖价差代理
            "imbalance_proxy",      # 订单簿不平衡代理
            "price_acceleration",   # 价格加速度
        ]
        self.n_features = len(self._feature_names)
        
        # 滚动统计缓存
        self._feature_cache: Dict[str, List[float]] = {name: [] for name in self._feature_names}
        self._cache_size = self._config.get("feature_cache_size", 1000)

    def extract_features(
        self, ohlcv_data: List[Dict[str, Any]], cache_features: bool = True
    ) -> List[List[float]]:
        """从OHLCV序列提取特征向量序列
        
        Args:
            ohlcv_data: OHLCV数据列表，每项含 open, high, low, close, vol
            
        Returns:
            特征矩阵 [n_samples x n_features]
        """
        if len(ohlcv_data) < 30:
            return []
        
        closes = [float(c["close"]) for c in ohlcv_data]
        opens = [float(c["open"]) for c in ohlcv_data]
        highs = [float(c["high"]) for c in ohlcv_data]
        lows = [float(c["low"]) for c in ohlcv_data]
        volumes = [float(c.get("vol", 0)) for c in ohlcv_data]
        
        n = len(ohlcv_data)
        features = []
        
        for i in range(20, n):
            window_closes = closes[max(0, i-20):i+1]
            window_highs = highs[max(0, i-20):i+1]
            window_lows = lows[max(0, i-20):i+1]
            window_volumes = volumes[max(0, i-20):i+1]
            
            feat = self._extract_single(closes, highs, lows, volumes, i, cache_features=cache_features)
            features.append(feat)
        
        return features

    def extract_latest(self, ohlcv_data: List[Dict[str, Any]]) -> List[float]:
        """提取最新一根K线的特征向量"""
        if len(ohlcv_data) < 30:
            return [0.0] * self.n_features
        
        closes = [float(c["close"]) for c in ohlcv_data]
        highs = [float(c["high"]) for c in ohlcv_data]
        lows = [float(c["low"]) for c in ohlcv_data]
        volumes = [float(c.get("vol", 0)) for c in ohlcv_data]
        
        return self._extract_single(closes, highs, lows, volumes, len(closes) - 1)

    def _extract_single(
        self, closes: List[float], highs: List[float], lows: List[float],
        volumes: List[float], idx: int, cache_features: bool = True,
    ) -> List[float]:
        """提取单个时间点的特征向量"""
        n = idx + 1
        window_20 = max(2, min(20, n))
        window_14 = max(2, min(14, n))
        window_50 = max(2, min(50, n))
        
        close = closes[idx]
        prev_close = closes[idx - 1] if idx > 0 else close
        
        # 1. 对数收益率
        if prev_close > 0:
            log_return = math.log(close / prev_close)
        else:
            log_return = 0.0
        
        # 2. 已实现波动率 (20周期年化)
        returns_20 = []
        for j in range(max(1, idx-window_20+1), idx+1):
            if closes[j-1] > 0:
                returns_20.append(math.log(closes[j] / closes[j-1]))
        if returns_20:
            mean_r = sum(returns_20) / len(returns_20)
            realized_vol = math.sqrt(sum((r - mean_r) ** 2 for r in returns_20) / len(returns_20))
        else:
            realized_vol = 0.0
        
        # 3. 成交量比率 (当前成交量 / 20周期平均成交量)
        vol_20 = volumes[max(0, idx-window_20):idx+1]
        avg_vol = sum(vol_20) / len(vol_20) if vol_20 else 1.0
        volume_ratio = volumes[idx] / avg_vol if avg_vol > 0 else 1.0
        
        # 4. 价格-MA距离 (标准化)
        ma_20 = sum(closes[max(0, idx-window_20+1):idx+1]) / window_20
        ma_50 = sum(closes[max(0, idx-window_50+1):idx+1]) / window_50
        if ma_50 > 0:
            price_ma_distance = (close - ma_50) / ma_50
        else:
            price_ma_distance = 0.0
        
        # 5. 布林带位置
        if len(closes[max(0, idx-window_20+1):idx+1]) > 1:
            band_closes = closes[max(0, idx-window_20+1):idx+1]
            ma = sum(band_closes) / len(band_closes)
            variance = sum((c - ma) ** 2 for c in band_closes) / len(band_closes)
            std = math.sqrt(variance) if variance > 0 else 1e-6
            bollinger_position = (close - ma) / (2 * std) if std > 0 else 0.0
        else:
            bollinger_position = 0.0
        
        # 6. RSI
        rsi = self._compute_rsi(closes, idx, 14)
        
        # 7. ATR比率
        atr = self._compute_atr(highs, lows, closes, idx, window_14)
        atr_ratio = atr / close if close > 0 else 0.0
        
        # 8. 买卖价差代理 (用 high-low / close)
        spread_proxy = (highs[idx] - lows[idx]) / close if close > 0 else 0.0
        
        # 9. 订单簿不平衡代理 (用价格在高低区间的相对位置)
        h_l_range = highs[idx] - lows[idx]
        if h_l_range > 0:
            imbalance_proxy = (close - lows[idx]) / h_l_range - 0.5
        else:
            imbalance_proxy = 0.0
        
        # 10. 价格加速度 (收益率的一阶差分)
        if idx >= 2 and closes[idx-1] > 0 and closes[idx-2] > 0:
            r1 = math.log(closes[idx-1] / closes[idx-2])
            r2 = math.log(closes[idx] / closes[idx-1])
            price_acceleration = r2 - r1
        else:
            price_acceleration = 0.0
        
        return self._normalize_features([
            log_return, realized_vol, volume_ratio, price_ma_distance,
            bollinger_position, rsi, atr_ratio, spread_proxy, imbalance_proxy,
            price_acceleration,
        ], cache_features=cache_features)

    def _compute_rsi(self, closes: List[float], idx: int, period: int = 14) -> float:
        """计算RSI"""
        if idx < period + 1:
            return 50.0
        
        gains = 0.0
        losses = 0.0
        for j in range(idx - period + 1, idx + 1):
            change = closes[j] - closes[j - 1]
            if change > 0:
                gains += change
            else:
                losses -= change
        
        avg_gain = gains / period
        avg_loss = losses / period
        
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _compute_atr(self, highs: List[float], lows: List[float],
                     closes: List[float], idx: int, period: int = 14) -> float:
        """计算ATR"""
        if idx < 1:
            return highs[idx] - lows[idx]
        
        tr_list = []
        start = max(1, idx - period + 1)
        for j in range(start, idx + 1):
            tr = max(
                highs[j] - lows[j],
                abs(highs[j] - closes[j-1]),
                abs(lows[j] - closes[j-1])
            )
            tr_list.append(tr)
        
        return sum(tr_list) / len(tr_list) if tr_list else 0.0

    def _normalize_features(self, features: List[float], cache_features: bool = True) -> List[float]:
        """归一化特征到[0,1]范围 (使用tanh/linear组合)"""
        normalized = []
        # 不同特征使用不同归一化方案
        for i, val in enumerate(features):
            if i == 0:  # log_return: tanh to [-1,1] then shift
                normalized.append((math.tanh(val * 20) + 1) / 2)
            elif i == 1:  # realized_volatility: linear in [0, 0.05]
                normalized.append(min(1.0, max(0.0, val / 0.05)))
            elif i == 2:  # volume_ratio: [0, 5] -> [0, 1]
                normalized.append(min(1.0, max(0.0, (val - 0.5) / 4.5)))
            elif i == 3:  # price_ma_distance: tanh
                normalized.append((math.tanh(val * 10) + 1) / 2)
            elif i == 4:  # bollinger_position: [-1, 1] -> [0, 1]
                normalized.append(min(1.0, max(0.0, (val + 1) / 2)))
            elif i == 5:  # RSI: already [0, 100]
                normalized.append(min(1.0, max(0.0, val / 100.0)))
            elif i == 6:  # ATR ratio: [0, 0.1] -> [0, 1]
                normalized.append(min(1.0, max(0.0, val / 0.1)))
            elif i == 7:  # spread proxy
                normalized.append(min(1.0, max(0.0, val / 0.05)))
            elif i == 8:  # imbalance proxy: [-0.5, 0.5] -> [0, 1]
                normalized.append(min(1.0, max(0.0, val + 0.5)))
            elif i == 9:  # price_acceleration: tanh
                normalized.append((math.tanh(val * 100) + 1) / 2)
            else:
                # 通用tanh归一化
                normalized.append((math.tanh(val) + 1) / 2)
        
        if cache_features:
            self._update_cache(normalized)
        return normalized

    def _update_cache(self, features: List[float]):
        """更新特征缓存"""
        for i, name in enumerate(self._feature_names):
            if i < len(features):
                self._feature_cache[name].append(features[i])
                if len(self._feature_cache[name]) > self._cache_size:
                    self._feature_cache[name] = self._feature_cache[name][-self._cache_size:]

    def get_rolling_statistics(self) -> Dict[str, Dict[str, float]]:
        """获取各特征的滚动统计量"""
        stats = {}
        for name, values in self._feature_cache.items():
            if not values:
                stats[name] = {"mean": 0, "std": 0, "skewness": 0, "kurtosis": 0}
                continue
            
            n = len(values)
            mean = sum(values) / n
            if n > 1:
                variance = sum((v - mean) ** 2 for v in values) / n
                std = math.sqrt(variance)
                skewness = sum(((v - mean) / std) ** 3 for v in values) / n if std > 0 else 0
                kurtosis = sum(((v - mean) / std) ** 4 for v in values) / n - 3 if std > 0 else 0
            else:
                std = 0
                skewness = 0
                kurtosis = 0
            
            stats[name] = {"mean": mean, "std": std, "skewness": skewness, "kurtosis": kurtosis}
        
        return stats

    def get_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """获取特征间相关系数矩阵"""
        names = self._feature_names
        n_features = len(names)
        matrix = {}
        
        for i in range(n_features):
            vals_i = self._feature_cache[names[i]]
            if not vals_i:
                continue
            matrix[names[i]] = {}
            for j in range(n_features):
                vals_j = self._feature_cache[names[j]]
                if not vals_j:
                    matrix[names[i]][names[j]] = 0.0
                    continue
                # Pearson correlation
                n_vals = min(len(vals_i), len(vals_j))
                if n_vals < 2:
                    matrix[names[i]][names[j]] = 0.0
                    continue
                x = vals_i[-n_vals:]
                y = vals_j[-n_vals:]
                mean_x = sum(x) / n_vals
                mean_y = sum(y) / n_vals
                cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n_vals
                std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / n_vals)
                std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / n_vals)
                if std_x > 0 and std_y > 0:
                    matrix[names[i]][names[j]] = cov / (std_x * std_y)
                else:
                    matrix[names[i]][names[j]] = 0.0
        
        return matrix

    def get_feature_names(self) -> List[str]:
        return self._feature_names


# ==================== 多时间周期分析 ====================

class MultiTimeframeRegime:
    """多时间周期市场状态分析
    
    使用三个时间周期: short (5min/15min), medium (1H), long (4H/1D)
    加权集成各周期的市场状态判断。
    """

    def __init__(self, config: Dict[str, Any] = None):
        self._config = config or {}
        
        # 时间周期定义
        self._timeframes = ["short", "medium", "long"]
        
        # 各周期的HMM分类器
        self._classifiers: Dict[str, HMMRegimeClassifier] = {
            "short": HMMRegimeClassifier(n_features=10),
            "medium": HMMRegimeClassifier(n_features=10),
            "long": HMMRegimeClassifier(n_features=10),
        }
        
        # 集成权重
        self._trend_weights = {
            "short": 0.15, "medium": 0.35, "long": 0.50
        }
        self._breakout_weights = {
            "short": 0.55, "medium": 0.30, "long": 0.15
        }
        self._default_weights = {
            "short": 0.25, "medium": 0.35, "long": 0.40
        }
        
        # 初始化特征提取器
        self._extractors: Dict[str, RegimeFeatureExtractor] = {
            "short": RegimeFeatureExtractor(config),
            "medium": RegimeFeatureExtractor(config),
            "long": RegimeFeatureExtractor(config),
        }
        
        self._trained = {"short": False, "medium": False, "long": False}

    def train(self, timeframe: str, ohlcv_data: List[Dict[str, Any]],
              n_iterations: int = 50) -> Dict[str, Any]:
        """训练某时间周期的HMM"""
        if timeframe not in self._timeframes:
            return {"error": f"Invalid timeframe: {timeframe}"}
        
        features = self._extractors[timeframe].extract_features(ohlcv_data)
        if len(features) < self._classifiers[timeframe].n_states:
            return {"error": f"Insufficient features: {len(features)}"}
        
        result = self._classifiers[timeframe].fit(features, n_iterations=n_iterations)
        self._trained[timeframe] = True
        logger.info(f"HMM trained for {timeframe}: ll={result.get('log_likelihood', 0):.2f}, "
                     f"iterations={result.get('iterations', 0)}, "
                     f"converged={result.get('converged', False)}")
        return result

    def predict(self, timeframe: str, ohlcv_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """预测某时间周期的市场状态"""
        if timeframe not in self._timeframes or not self._trained.get(timeframe, False):
            return {"regime": MarketRegime.UNKNOWN, "probabilities": {}, "timeframe": timeframe}
        
        feature_sequence = self._extractors[timeframe].extract_features(
            ohlcv_data, cache_features=False
        )
        if not feature_sequence:
            return {"regime": MarketRegime.UNKNOWN, "probabilities": {}, "timeframe": timeframe}

        classifier = self._classifiers[timeframe]
        probs = classifier.predict_proba_sequence(feature_sequence[-100:])
        regime = max(probs, key=probs.get) if probs else MarketRegime.UNKNOWN
        
        return {
            "regime": regime,
            "probabilities": {r.value: p for r, p in probs.items()},
            "timeframe": timeframe,
        }

    def predict_ensemble(self, ohlcv_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """加权集成多时间周期预测"""
        timeframe_results = {}
        
        for tf in self._timeframes:
            if tf in ohlcv_data and self._trained.get(tf, False):
                timeframe_results[tf] = self.predict(tf, ohlcv_data[tf])
            else:
                timeframe_results[tf] = {
                    "regime": MarketRegime.UNKNOWN,
                    "probabilities": {r.value: 0.0 for r in _REGIME_ORDER},
                    "timeframe": tf,
                }
        
        # 加权投票
        weighted_scores = {r: 0.0 for r in _REGIME_ORDER}
        total_weight = 0.0
        
        for tf, result in timeframe_results.items():
            regime = result["regime"]
            probs = result["probabilities"]
            if regime == MarketRegime.UNKNOWN:
                continue
            
            # 根据检测到的regime类型选择权重
            if regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
                weights = self._trend_weights
            elif regime == MarketRegime.BREAKOUT:
                weights = self._breakout_weights
            else:
                weights = self._default_weights
            
            w = weights.get(tf, 0.25)
            total_weight += w
            
            for r in _REGIME_ORDER:
                p = probs.get(r.value, 0.0)
                weighted_scores[r] += w * p
        
        # 归一化
        if total_weight > 0:
            final_probs = {r: s / total_weight for r, s in weighted_scores.items()}
        else:
            final_probs = {r: 0.0 for r in _REGIME_ORDER}
            final_probs[MarketRegime.UNKNOWN] = 1.0
        
        best_regime = max(final_probs, key=lambda k: final_probs[k])
        
        # 一致性检测：所有时间周期是否一致
        regimes_set = set()
        for tf in self._timeframes:
            r = timeframe_results[tf]["regime"]
            if r != MarketRegime.UNKNOWN:
                regimes_set.add(r)
        confluence = len(regimes_set) == 1 and len(regimes_set) > 0
        
        return {
            "regime": best_regime,
            "probabilities": {r.value: p for r, p in final_probs.items()},
            "confluence": confluence,
            "timeframe_results": {
                tf: {
                    "regime": r["regime"].value,
                    "probabilities": r["probabilities"],
                }
                for tf, r in timeframe_results.items()
            },
        }

    def is_trained(self, timeframe: str) -> bool:
        return self._trained.get(timeframe, False)

    def are_all_trained(self) -> bool:
        return all(self._trained.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classifiers": {tf: clf.to_dict() for tf, clf in self._classifiers.items()},
            "trained": self._trained,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any], config: Dict[str, Any] = None) -> "MultiTimeframeRegime":
        mtf = cls(config)
        for tf, clf_dict in d.get("classifiers", {}).items():
            if tf in mtf._classifiers:
                mtf._classifiers[tf] = HMMRegimeClassifier.from_dict(clf_dict)
        mtf._trained = d.get("trained", {"short": False, "medium": False, "long": False})
        return mtf


# ==================== 状态转移预测器 ====================

class RegimeTransitionPredictor:
    """Markov链状态转移预测
    
    基于HMM的状态转移矩阵和历史状态序列，
    预测最可能的下一个状态和当前状态的持续性。
    """

    def __init__(self, config: Dict[str, Any] = None):
        self._config = config or {}
        
        # 历史状态序列
        self._state_history: deque = deque(maxlen=self._config.get("history_size", 500))
        
        # 每个状态的持续统计: {regime: [durations]}
        self._duration_stats: Dict[MarketRegime, List[int]] = {
            r: [] for r in _REGIME_ORDER
        }
        
        # 转移计数矩阵 (for empirical transition probabilities)
        self._transition_counts: Dict[MarketRegime, Dict[MarketRegime, int]] = {
            r1: {r2: 0 for r2 in _REGIME_ORDER} for r1 in _REGIME_ORDER
        }
        
        # 早期预警信号阈值
        self._warning_thresholds = {
            "vol_expansion_ratio": self._config.get("vol_expansion_ratio", 2.0),
            "volume_spike_ratio": self._config.get("volume_spike_ratio", 3.0),
            "price_compression_pct": self._config.get("price_compression_pct", 0.3),
        }

        # HMM 转移矩阵与经验转移计数合并权重（0.0~1.0，越大越依赖 HMM）
        self._hmm_weight = self._config.get("hmm_weight", 0.5)

    def update_history(self, regime: MarketRegime, features: Dict[str, float] = None):
        """更新状态历史"""
        if regime == MarketRegime.UNKNOWN:
            return
        
        self._state_history.append({
            "regime": regime,
            "features": features or {},
            "timestamp": datetime.now(),
        })
        
        # 更新转移计数
        if len(self._state_history) >= 2:
            prev = self._state_history[-2]["regime"]
            curr = self._state_history[-1]["regime"]
            if prev != curr:
                self._transition_counts[prev][curr] += 1
        
        # 更新持续统计
        durations = self._compute_durations()
        for r, d in durations.items():
            if d > 0 and d not in self._duration_stats[r]:
                self._duration_stats[r].append(d)
                # 限制最大存储
                if len(self._duration_stats[r]) > 100:
                    self._duration_stats[r] = self._duration_stats[r][-100:]

    def _compute_durations(self) -> Dict[MarketRegime, int]:
        """从历史序列计算各状态的当前持续长度"""
        if not self._state_history:
            return {r: 0 for r in _REGIME_ORDER}
        
        durations = {r: 0 for r in _REGIME_ORDER}
        current_regime = self._state_history[-1]["regime"]
        
        count = 0
        for entry in reversed(self._state_history):
            if entry["regime"] == current_regime:
                count += 1
            else:
                break
        durations[current_regime] = count
        
        return durations

    def predict_next_regime(self, transition_matrix: List[List[float]] = None) -> Dict[str, Any]:
        """预测最可能的下一个状态
        
        Args:
            transition_matrix: HMM状态转移矩阵 (可选，用于更精确的预测)
        """
        if not self._state_history:
            return {"status": "insufficient_data"}
        
        current = self._state_history[-1]["regime"]
        durations = self._compute_durations()
        current_duration = durations.get(current, 0)
        
        # 1. 从 HMM 转移矩阵预测：映射到各状态的概率分布
        hmm_probs: Dict[MarketRegime, float] = {}
        if transition_matrix and current in _REGIME_TO_IDX:
            cur_idx = _REGIME_TO_IDX[current]
            if cur_idx < len(transition_matrix):
                row = transition_matrix[cur_idx]
                for i, prob in enumerate(row):
                    if i < len(_REGIME_ORDER):
                        hmm_probs[_REGIME_ORDER[i]] = prob

        # 2. 从经验转移计数预测
        empirical_probs = self._get_empirical_probs(current)

        # 3. 持续性预测
        persistence = self._predict_persistence(current, current_duration)

        # 4. 合并预测：HMM 转移矩阵（更精确）与经验计数加权融合
        next_regime_probs = {r: 0.0 for r in _REGIME_ORDER}

        norm_empirical: Dict[MarketRegime, float] = {}
        if empirical_probs:
            total = sum(empirical_probs.values())
            if total > 0:
                norm_empirical = {r: p / total for r, p in empirical_probs.items()}

        if hmm_probs and norm_empirical:
            w = self._hmm_weight
            for r in _REGIME_ORDER:
                next_regime_probs[r] = w * hmm_probs.get(r, 0.0) + (1.0 - w) * norm_empirical.get(r, 0.0)
        elif hmm_probs:
            next_regime_probs.update(hmm_probs)
        elif norm_empirical:
            next_regime_probs.update(norm_empirical)
        
        most_likely = max(next_regime_probs, key=lambda k: next_regime_probs[k]) \
            if any(v > 0 for v in next_regime_probs.values()) else current
        
        return {
            "current_regime": current.value,
            "most_likely_next": most_likely.value,
            "next_probabilities": {r.value: p for r, p in next_regime_probs.items()},
            "current_duration": current_duration,
            "predicted_persistence": persistence,
            "transition_risk": self._compute_transition_risk(current, next_regime_probs),
        }

    def _get_empirical_probs(self, current: MarketRegime) -> Dict[MarketRegime, float]:
        """从经验计数计算转移概率"""
        counts = self._transition_counts[current]
        total = sum(counts.values())
        if total == 0:
            return {current: 1.0}
        return {r: c / total for r, c in counts.items()}

    def _predict_persistence(self, regime: MarketRegime, current_duration: int) -> Dict[str, Any]:
        """预测当前状态的持续性"""
        durations = self._duration_stats.get(regime, [])
        if not durations:
            return {"expected_duration": current_duration, "confidence": 0.0}
        
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)

        # 使用历史平均持续 vs 当前持续
        remaining_expected = max(0, avg_duration - current_duration)
        
        confidence = min(1.0, len(durations) / 30.0) if durations else 0.0
        
        return {
            "expected_duration": avg_duration,
            "current_duration": current_duration,
            "expected_remaining": remaining_expected,
            "max_historical_duration": max_duration,
            "regime_mature": current_duration > avg_duration * 1.5,
            "confidence": confidence,
        }

    def _compute_transition_risk(self, current: MarketRegime,
                                  next_probs: Dict[MarketRegime, float]) -> Dict[str, Any]:
        """计算转移风险 - 转移到不利状态的概率"""
        adverse_regimes = {
            MarketRegime.TRENDING_DOWN,
            MarketRegime.HIGH_VOLATILITY,
            MarketRegime.REVERSAL,
        }
        
        # 如果当前已经是不利状态，重新定义"不利"为更差
        if current in adverse_regimes:
            adverse_regimes = {MarketRegime.HIGH_VOLATILITY, MarketRegime.UNKNOWN}
        
        adverse_prob = sum(next_probs.get(r, 0) for r in adverse_regimes if r != current)
        
        return {
            "adverse_transition_probability": adverse_prob,
            "risk_level": "high" if adverse_prob > 0.4 else ("medium" if adverse_prob > 0.2 else "low"),
            "adverse_regimes": [r.value for r in adverse_regimes],
        }

    def detect_early_warnings(self, features_history: List[Dict[str, float]],
                               window: int = 20) -> List[Dict[str, Any]]:
        """检测早期预警信号
        
        检测:
        - 波动率扩张 (regime变化前兆)
        - 成交量异常放大
        - 价格压缩 (布林带收窄)
        """
        warnings = []
        
        if len(features_history) < window:
            return warnings
        
        recent = features_history[-window:]
        older = features_history[-window*2:-window] if len(features_history) >= window * 2 else recent
        
        # 1. 波动率扩张检测
        recent_vol = [f.get("realized_volatility", 0) for f in recent]
        older_vol = [f.get("realized_volatility", 0) for f in older]
        avg_recent_vol = sum(recent_vol) / len(recent_vol) if recent_vol else 0
        avg_older_vol = sum(older_vol) / len(older_vol) if older_vol else 1e-6
        
        if avg_older_vol > 0 and avg_recent_vol / avg_older_vol > self._warning_thresholds["vol_expansion_ratio"]:
            warnings.append({
                "type": "volatility_expansion",
                "severity": "high" if avg_recent_vol / avg_older_vol > 3 else "medium",
                "ratio": avg_recent_vol / avg_older_vol,
                "message": f"波动率急剧扩张: {avg_recent_vol/avg_older_vol:.1f}x",
            })
        
        # 2. 成交量异常放大
        recent_vol_ratio = [f.get("volume_ratio", 1.0) for f in recent]
        max_vol_ratio = max(recent_vol_ratio) if recent_vol_ratio else 1.0
        if max_vol_ratio > self._warning_thresholds["volume_spike_ratio"]:
            warnings.append({
                "type": "volume_spike",
                "severity": "high" if max_vol_ratio > 5 else "medium",
                "ratio": max_vol_ratio,
                "message": f"成交量异常放大: {max_vol_ratio:.1f}x",
            })
        
        # 3. 价格压缩检测 (布林带位置范围窄)
        boll_positions = [f.get("bollinger_position", 0.5) for f in recent]
        if boll_positions:
            boll_range = max(boll_positions) - min(boll_positions)
            if boll_range < self._warning_thresholds["price_compression_pct"]:
                warnings.append({
                    "type": "price_compression",
                    "severity": "medium",
                    "range": boll_range,
                    "message": f"价格压缩信号 (布林带收窄): range={boll_range:.3f}",
                })
        
        # 4. RSI极端
        recent_rsi = [f.get("rsi", 50) for f in recent[-5:]]
        avg_rsi = sum(recent_rsi) / len(recent_rsi) if recent_rsi else 50
        if avg_rsi > 75:
            warnings.append({
                "type": "rsi_extreme",
                "severity": "medium",
                "rsi": avg_rsi,
                "message": f"RSI超买: {avg_rsi:.1f}",
            })
        elif avg_rsi < 25:
            warnings.append({
                "type": "rsi_extreme",
                "severity": "medium",
                "rsi": avg_rsi,
                "message": f"RSI超卖: {avg_rsi:.1f}",
            })
        
        return warnings

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transition_counts": {
                r1.value: {r2.value: c for r2, c in counts.items()}
                for r1, counts in self._transition_counts.items()
            },
            "duration_stats": {
                r.value: d for r, d in self._duration_stats.items()
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegimeTransitionPredictor":
        predictor = cls()
        for r1_str, counts in d.get("transition_counts", {}).items():
            r1 = MarketRegime(r1_str)
            for r2_str, c in counts.items():
                r2 = MarketRegime(r2_str)
                predictor._transition_counts[r1][r2] = c
        for r_str, durations in d.get("duration_stats", {}).items():
            r = MarketRegime(r_str)
            predictor._duration_stats[r] = durations
        return predictor


# ==================== 策略-状态映射器 ====================

class RegimeStrategyMapper:
    """市场状态 → 最优策略映射
    
    根据当前市场状态推荐最优策略类型，并跟踪各策略在各状态下的历史表现。
    """

    # 策略-状态最优映射表
    DEFAULT_REGIME_STRATEGY_MAP = {
        MarketRegime.TRENDING_UP: {
            "primary": ["trend_following", "momentum"],
            "secondary": ["scalping"],
            "avoid": ["grid", "mean_reversion", "spot_martingale"],
        },
        MarketRegime.TRENDING_DOWN: {
            "primary": ["trend_following_short", "hedge"],
            "secondary": ["scalping_short"],
            "avoid": ["grid_long", "spot_martingale", "trend_following_long"],
        },
        MarketRegime.RANGING: {
            "primary": ["grid", "mean_reversion"],
            "secondary": ["scalping", "spot_grid"],
            "avoid": ["trend_following", "momentum", "arbitrage"],
        },
        MarketRegime.HIGH_VOLATILITY: {
            "primary": ["scalping", "momentum"],
            "secondary": ["trend_following"],
            "avoid": ["grid", "spot_grid", "mean_reversion", "spot_martingale"],
        },
        MarketRegime.LOW_VOLATILITY: {
            "primary": ["grid", "spot_grid", "spot_martingale"],
            "secondary": ["scalping"],
            "avoid": ["trend_following", "momentum"],
        },
        MarketRegime.BREAKOUT: {
            "primary": ["momentum", "trend_following"],
            "secondary": ["scalping"],
            "avoid": ["mean_reversion", "grid"],
        },
        MarketRegime.REVERSAL: {
            "primary": ["mean_reversion", "counter_trend"],
            "secondary": ["scalping"],
            "avoid": ["trend_following", "momentum", "grid"],
        },
        MarketRegime.UNKNOWN: {
            "primary": ["scalping"],
            "secondary": [],
            "avoid": ["trend_following", "grid", "momentum"],
        },
    }

    def __init__(self, config: Dict[str, Any] = None):
        self._config = config or {}
        self._strategy_map = dict(self.DEFAULT_REGIME_STRATEGY_MAP)
        
        # 策略在各状态下的历史表现: {strategy: {regime: {win_rate, avg_pnl, count}}}
        self._performance: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # 置信度衰减因子
        self._confidence_base = self._config.get("confidence_base", 0.5)

    def get_optimal_strategies(self, regime: MarketRegime,
                                regime_probs: Dict[MarketRegime, float] = None,
                                top_k: int = 5) -> List[Dict[str, Any]]:
        """获取当前市场状态下的最优策略列表
        
        Args:
            regime: 当前市场状态
            regime_probs: 各状态的概率分布 (用于加权)
            top_k: 返回前K个策略
        
        Returns:
            策略列表 [{strategy, type, confidence, performance}]
        """
        if regime_probs is None:
            regime_probs = {regime: 1.0}
        
        strategy_scores: Dict[str, float] = {}
        strategy_types: Dict[str, str] = {}
        
        for r, prob in regime_probs.items():
            if prob <= 0:
                continue
            
            mapping = self._strategy_map.get(r, self._strategy_map[MarketRegime.UNKNOWN])
            
            for s in mapping.get("primary", []):
                strategy_scores[s] = strategy_scores.get(s, 0) + prob * 1.0
                strategy_types[s] = "primary"
            
            for s in mapping.get("secondary", []):
                strategy_scores[s] = strategy_scores.get(s, 0) + prob * 0.6
                if s not in strategy_types:
                    strategy_types[s] = "secondary"
            
            for s in mapping.get("avoid", []):
                strategy_scores[s] = strategy_scores.get(s, 0) - prob * 0.5
                if s not in strategy_types:
                    strategy_types[s] = "avoid"
        
        # 根据历史表现调整分数
        for strategy, scores in self._performance.items():
            regime_perf = scores.get(regime.value, {})
            if regime_perf:
                win_rate = regime_perf.get("win_rate", 0)
                avg_pnl = regime_perf.get("avg_pnl", 0)
                count = regime_perf.get("count", 0)
                
                perf_bonus = (win_rate - 0.5) * 0.3 + (avg_pnl / max(abs(avg_pnl), 1)) * 0.1
                if count >= 5:
                    strategy_scores[strategy] = strategy_scores.get(strategy, 0) + perf_bonus
        
        # 排序
        sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for strategy, score in sorted_strategies[:top_k]:
            if score <= 0:
                continue
            
            # 置信度: 基于得分和历史数据
            conf = self._confidence_base + (score - 0.5) * 0.4
            conf = min(1.0, max(0.0, conf))
            
            # 历史表现
            perf = {}
            if strategy in self._performance and regime.value in self._performance[strategy]:
                perf = self._performance[strategy][regime.value]
            
            results.append({
                "strategy": strategy,
                "type": strategy_types.get(strategy, "unknown"),
                "score": round(score, 4),
                "confidence": round(conf, 4),
                "historical_performance": perf,
            })
        
        return results

    def update_performance(self, strategy: str, regime: MarketRegime,
                            win: bool, pnl: float):
        """更新策略在某状态下的历史表现"""
        if strategy not in self._performance:
            self._performance[strategy] = {}
        
        regime_key = regime.value
        if regime_key not in self._performance[strategy]:
            self._performance[strategy][regime_key] = {
                "wins": 0, "losses": 0, "total_pnl": 0.0,
                "count": 0, "win_rate": 0.0, "avg_pnl": 0.0,
            }
        
        perf = self._performance[strategy][regime_key]
        if win:
            perf["wins"] += 1
        else:
            perf["losses"] += 1
        perf["total_pnl"] += pnl
        perf["count"] += 1
        perf["win_rate"] = perf["wins"] / perf["count"] if perf["count"] > 0 else 0
        perf["avg_pnl"] = perf["total_pnl"] / perf["count"] if perf["count"] > 0 else 0

    def get_strategy_performance(self, strategy: str) -> Dict[str, Dict[str, Any]]:
        """获取策略在所有状态下的表现"""
        return self._performance.get(strategy, {})

    def get_regime_map(self, regime: MarketRegime) -> Dict[str, List[str]]:
        """获取某状态下的策略映射"""
        return self._strategy_map.get(regime, self._strategy_map[MarketRegime.UNKNOWN])

    def update_regime_map(self, regime: MarketRegime,
                           primary: List[str] = None,
                           secondary: List[str] = None,
                           avoid: List[str] = None):
        """动态更新策略-状态映射"""
        if regime not in self._strategy_map:
            self._strategy_map[regime] = {"primary": [], "secondary": [], "avoid": []}
        if primary is not None:
            self._strategy_map[regime]["primary"] = primary
        if secondary is not None:
            self._strategy_map[regime]["secondary"] = secondary
        if avoid is not None:
            self._strategy_map[regime]["avoid"] = avoid

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_map": {
                r.value: v for r, v in self._strategy_map.items()
            },
            "performance": self._performance,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegimeStrategyMapper":
        mapper = cls()
        for r_str, mapping in d.get("strategy_map", {}).items():
            r = MarketRegime(r_str)
            mapper._strategy_map[r] = mapping
        mapper._performance = d.get("performance", {})
        return mapper


# ==================== 主类：市场状态检测器 ====================

class MarketRegimeDetector(EnterpriseServiceMixin):
    """市场状态检测器 - 统一入口
    
    整合HMM分类、多时间周期分析、转移预测和策略映射。
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        detector_cfg = config.get("market_regime_detector", {})
        
        # ── 多时间周期分析 ──
        self._mtf = MultiTimeframeRegime(detector_cfg)
        self._symbol_mtfs: Dict[str, MultiTimeframeRegime] = {}
        
        # ── 特征提取器 ──
        self._feature_extractor = RegimeFeatureExtractor(detector_cfg)
        self._symbol_feature_extractors: Dict[str, RegimeFeatureExtractor] = {}
        
        # ── 状态转移预测器 ──
        self._transition_predictor = RegimeTransitionPredictor(detector_cfg)
        self._symbol_transition_predictors: Dict[str, RegimeTransitionPredictor] = {}
        
        # ── 策略映射器 ──
        self._strategy_mapper = RegimeStrategyMapper(detector_cfg)
        
        # ── 当前状态缓存 ──
        self._current_regimes: Dict[str, Dict[str, Any]] = {}
        self._regime_history: Dict[str, deque] = {}
        self._last_observation_keys: Dict[str, Any] = {}
        self._max_history = detector_cfg.get("max_history_per_symbol", 200)
        
        # ── 持久化 ──
        self._persist_dir = detector_cfg.get("persist_dir", "./data/regime_detector")
        self._persist_interval = detector_cfg.get("persist_interval_seconds", 600)
        self._last_persist = 0
        
        # ── 配置参数 ──
        self._train_min_samples = detector_cfg.get("train_min_samples", 50)
        self._auto_train = detector_cfg.get("auto_train", True)
        self._default_timeframes = detector_cfg.get("timeframes", ["short", "medium", "long"])
        self._fallback_timeframe = detector_cfg.get("fallback_timeframe", "medium")
        if self._fallback_timeframe not in ("short", "medium", "long"):
            self._fallback_timeframe = "medium"
        self._min_stable_updates = max(1, int(detector_cfg.get("min_stable_updates", 3)))
        self._stability_confidence_discount = max(
            0.0, min(1.0, float(detector_cfg.get("stability_confidence_discount", 0.7)))
        )
        
        # ── 线程安全 ──
        self._lock = asyncio.Lock()
        self._running = False
        
        os.makedirs(self._persist_dir, exist_ok=True)
        self._load()
        logger.info(f"MarketRegimeDetector initialized: timeframes={self._default_timeframes}, "
                     f"min_samples={self._train_min_samples}")

    async def start(self):
        """启动市场状态检测器"""
        self._running = True
        logger.info("MarketRegimeDetector started")

    async def stop(self):
        """停止市场状态检测器"""
        self._running = False
        await self._persist()
        logger.info("MarketRegimeDetector stopped")

    # ===================== 核心检测 =====================

    def _get_symbol_mtf(self, symbol: str) -> MultiTimeframeRegime:
        if symbol not in self._symbol_mtfs:
            self._symbol_mtfs[symbol] = MultiTimeframeRegime(self._config.get("market_regime_detector", {}))
        return self._symbol_mtfs[symbol]

    def _get_symbol_feature_extractor(self, symbol: str) -> RegimeFeatureExtractor:
        if symbol not in self._symbol_feature_extractors:
            self._symbol_feature_extractors[symbol] = RegimeFeatureExtractor(
                self._config.get("market_regime_detector", {})
            )
        return self._symbol_feature_extractors[symbol]

    def _get_symbol_transition_predictor(self, symbol: str) -> RegimeTransitionPredictor:
        if symbol not in self._symbol_transition_predictors:
            self._symbol_transition_predictors[symbol] = RegimeTransitionPredictor(
                self._config.get("market_regime_detector", {})
            )
        return self._symbol_transition_predictors[symbol]

    @staticmethod
    def _observation_key(ohlcv_data: List[Dict[str, Any]]):
        if not ohlcv_data or not isinstance(ohlcv_data[-1], dict):
            return None
        latest = ohlcv_data[-1]
        for timestamp_key in ("timestamp", "ts", "time"):
            if latest.get(timestamp_key) is not None:
                return (timestamp_key, str(latest[timestamp_key]))
        candle_values = tuple(
            (key, str(latest[key]))
            for key in ("open", "high", "low", "close", "vol", "volume")
            if key in latest
        )
        return (len(ohlcv_data), candle_values)

    def _infer_reversal_direction(self, ohlcv_data: List[Dict[str, Any]]) -> Optional[str]:
        """Require a prior directional leg and a multi-candle move against it."""
        detector_cfg = self._config.get("market_regime_detector", {})
        trend_lookback = max(3, int(detector_cfg.get("reversal_direction_lookback", 8)))
        confirmation_bars = max(
            2, int(detector_cfg.get("reversal_direction_confirmation_bars", 3))
        )
        min_trend_return = max(
            0.0, float(detector_cfg.get("reversal_direction_min_trend_pct", 0.004))
        )
        min_confirmation_return = max(
            0.0, float(detector_cfg.get("reversal_direction_min_confirm_pct", 0.0015))
        )

        required_candles = trend_lookback + confirmation_bars + 1
        if not isinstance(ohlcv_data, list) or len(ohlcv_data) < required_candles:
            return None

        try:
            closes = [float(candle["close"]) for candle in ohlcv_data[-required_candles:]]
        except (KeyError, TypeError, ValueError):
            return None
        if any(close <= 0 for close in closes):
            return None

        prior_start = closes[0]
        prior_end = closes[trend_lookback]
        confirmation_start = prior_end
        confirmation_closes = closes[trend_lookback:]
        prior_return = prior_end / prior_start - 1.0
        confirmation_return = confirmation_closes[-1] / confirmation_start - 1.0
        confirmation_changes = [
            right - left
            for left, right in zip(confirmation_closes, confirmation_closes[1:])
        ]
        required_direction_bars = (confirmation_bars * 2 + 2) // 3

        if (
            prior_return <= -min_trend_return
            and confirmation_return >= min_confirmation_return
            and sum(change > 0 for change in confirmation_changes) >= required_direction_bars
        ):
            return "long"
        if (
            prior_return >= min_trend_return
            and confirmation_return <= -min_confirmation_return
            and sum(change < 0 for change in confirmation_changes) >= required_direction_bars
        ):
            return "short"
        return None

    async def detect_regime(self, ohlcv_data: List[Dict[str, Any]],
                             symbol: str, timeframe: Optional[str] = None,
                             timeframe_data: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Dict[str, Any]:
        """检测单个币种的市场状态
        
        Args:
            ohlcv_data: OHLCV数据列表
            symbol: 交易对符号
            timeframe: 明确的周期类型（short/medium/long）；不提供时由K线时间戳间隔推断
            timeframe_data: 可选的按周期分组OHLCV数据；至少两个已训练周期有效时启用集成
        
        Returns:
            市场状态检测结果
        """
        async with self._lock:
            mtf = self._get_symbol_mtf(symbol)
            feature_extractor = self._get_symbol_feature_extractor(symbol)
            transition_predictor = self._get_symbol_transition_predictor(symbol)
            tf = timeframe if timeframe in ("short", "medium", "long") else self._estimate_timeframe(ohlcv_data)

            # Train each timeframe from its own candle sequence; never reuse one interval as another.
            training_inputs = {tf: ohlcv_data}
            if isinstance(timeframe_data, dict):
                training_inputs.update({
                    bucket: candles
                    for bucket, candles in timeframe_data.items()
                    if bucket in self._default_timeframes and isinstance(candles, list)
                })
            if self._auto_train:
                for train_tf, candles in training_inputs.items():
                    if len(candles) >= self._train_min_samples and not mtf.is_trained(train_tf):
                        result = mtf.train(train_tf, candles, n_iterations=30)
                        logger.debug(f"Auto-trained HMM for {symbol} ({train_tf}): {result}")

            prediction = mtf.predict(tf, ohlcv_data)
            if isinstance(timeframe_data, dict):
                ensemble_inputs = {
                    time_bucket: candles
                    for time_bucket, candles in timeframe_data.items()
                    if time_bucket in self._default_timeframes
                    and mtf.is_trained(time_bucket)
                    and isinstance(candles, list)
                    and candles
                }
                if len(ensemble_inputs) >= 2:
                    prediction = mtf.predict_ensemble(ensemble_inputs)

            candidate_regime = prediction["regime"]
            previous_regime = self._current_regimes.get(symbol, {})
            observation_key = self._observation_key(ohlcv_data)
            is_new_observation = (
                observation_key is not None
                and self._last_observation_keys.get(symbol) != observation_key
            )
            previous_regime_name = previous_regime.get("regime")
            pending_regime_name = previous_regime.get("pending_regime")
            pending_regime_updates = previous_regime.get("pending_regime_updates", 0)
            immediate_regimes = {
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.REVERSAL,
            }

            if previous_regime_name and candidate_regime == MarketRegime.UNKNOWN:
                regime = MarketRegime(previous_regime_name)
                pending_regime_name = None
                pending_regime_updates = 0
            elif (
                previous_regime_name
                and candidate_regime.value != previous_regime_name
                and candidate_regime not in immediate_regimes
            ):
                if pending_regime_name == candidate_regime.value:
                    if is_new_observation:
                        pending_regime_updates += 1
                else:
                    pending_regime_name = candidate_regime.value
                    pending_regime_updates = 1

                if pending_regime_updates < self._min_stable_updates:
                    regime = MarketRegime(previous_regime_name)
                else:
                    regime = candidate_regime
                    pending_regime_name = None
                    pending_regime_updates = 0
            else:
                regime = candidate_regime
                pending_regime_name = None
                pending_regime_updates = 0
            
            # 提取最新特征
            latest_features = feature_extractor.extract_latest(ohlcv_data)
            
            # 更新转移预测器
            transition_predictor.update_history(regime)
            
            # 检测早期预警信号
            features_history = []
            for feat_list in feature_extractor._feature_cache.values():
                if feat_list:
                    break
            if feature_extractor._feature_cache.get("log_return"):
                n_entries = min(40, len(feature_extractor._feature_cache["log_return"]))
                for i in range(-n_entries, 0):
                    entry = {}
                    for name in feature_extractor._feature_names:
                        cache = feature_extractor._feature_cache[name]
                        if len(cache) + i >= 0:
                            entry[name] = cache[i]
                    if entry:
                        features_history.append(entry)
            
            warnings = transition_predictor.detect_early_warnings(
                features_history[-40:] if features_history else []
            )
            
            # 获取最优策略
            probs = prediction.get("probabilities", {})
            regime_probs = {}
            for r_str, p in probs.items():
                try:
                    regime_probs[MarketRegime(r_str)] = p
                except ValueError:
                    pass
            optimal_strategies = self._strategy_mapper.get_optimal_strategies(
                regime, regime_probs
            )

            if previous_regime.get("regime") == regime.value:
                consecutive_updates = previous_regime.get("regime_consecutive_updates", 1)
                if is_new_observation:
                    consecutive_updates += 1
            else:
                consecutive_updates = 1
            self._last_observation_keys[symbol] = observation_key
            stable = consecutive_updates >= self._min_stable_updates
            confidence = float(probs.get(regime.value, 0.0))
            if not stable:
                confidence *= self._stability_confidence_discount
            
            # 缓存结果
            result = {
                "symbol": symbol,
                "regime": regime.value,
                "probabilities": probs,
                "confidence": confidence,
                "stable": stable,
                "regime_consecutive_updates": consecutive_updates,
                "pending_regime": pending_regime_name,
                "pending_regime_updates": pending_regime_updates,
                "reversal_direction": (
                    self._infer_reversal_direction(ohlcv_data)
                    if regime == MarketRegime.REVERSAL
                    else None
                ),
                "features": {
                    name: latest_features[i] if i < len(latest_features) else 0.0
                    for i, name in enumerate(feature_extractor._feature_names)
                },
                "optimal_strategies": optimal_strategies,
                "early_warnings": warnings,
                "timeframe": tf,
                "detected_at": datetime.now().isoformat(),
                "candidate_regime": prediction.get("candidate_regime", regime.value),
                "transition_pending": pending_regime_name is not None,
            }
            
            self._current_regimes[symbol] = result
            
            # 更新历史
            if symbol not in self._regime_history:
                self._regime_history[symbol] = deque(maxlen=self._max_history)
            self._regime_history[symbol].append(result)
            
            # 自动持久化
            if time.time() - self._last_persist > self._persist_interval:
                asyncio.create_task(self._persist())
                self._last_persist = time.time()
            
            return result

    def _estimate_timeframe(self, ohlcv_data: List[Dict[str, Any]]) -> str:
        """从K线时间戳间隔推断周期；缺少有效时间戳时使用明确配置的回退周期。"""
        timestamps = []
        for candle in ohlcv_data if isinstance(ohlcv_data, list) else []:
            raw_timestamp = candle.get("timestamp", candle.get("ts", candle.get("time")))
            if raw_timestamp is None:
                continue
            try:
                timestamp = float(raw_timestamp)
                if timestamp > 1e11:
                    timestamp /= 1000.0
            except (TypeError, ValueError):
                try:
                    timestamp = datetime.fromisoformat(str(raw_timestamp).replace("Z", "+00:00")).timestamp()
                except ValueError:
                    continue
            timestamps.append(timestamp)

        intervals = sorted(
            right - left for left, right in zip(timestamps, timestamps[1:]) if right > left
        )
        if not intervals:
            return self._fallback_timeframe

        interval_seconds = intervals[len(intervals) // 2]
        if interval_seconds <= 15 * 60:
            return "short"
        if interval_seconds <= 4 * 60 * 60:
            return "medium"
        return "long"

    async def detect_all_symbols(self, symbols_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """批量检测所有币种的市场状态"""
        results = {}
        
        tasks = [
            self.detect_regime(data, symbol)
            for symbol, data in symbols_data.items()
        ]
        
        detected = await asyncio.gather(*tasks, return_exceptions=True)
        
        for symbol, result in zip(symbols_data.keys(), detected):
            if isinstance(result, Exception):
                self._handle_exception(
                    result, {"symbol": symbol}, module="MarketRegimeDetector",
                    function="detect_all_symbols", severity="medium", category="market_regime",
                )
                results[symbol] = {
                    "symbol": symbol,
                    "regime": MarketRegime.UNKNOWN.value,
                    "error": str(result),
                }
            else:
                results[symbol] = result
        
        return {
            "timestamp": datetime.now().isoformat(),
            "symbols": results,
            "summary": self._summarize_market(results),
        }

    def _summarize_market(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """汇总整体市场状态"""
        regime_counts = {r.value: 0 for r in _REGIME_ORDER}
        symbols_with_warnings = []
        
        for symbol, data in results.items():
            regime = data.get("regime", "unknown")
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            if data.get("early_warnings"):
                symbols_with_warnings.append(symbol)
        
        dominant_regime = max(regime_counts, key=lambda k: regime_counts[k]) \
            if any(regime_counts.values()) else MarketRegime.UNKNOWN.value
        
        return {
            "total_symbols": len(results),
            "regime_distribution": regime_counts,
            "dominant_regime": dominant_regime,
            "symbols_with_warnings": symbols_with_warnings,
            "warning_count": len(symbols_with_warnings),
        }

    # ===================== 转移预测 =====================

    async def predict_transition(self, symbol: str) -> Dict[str, Any]:
        """预测某币种的下一个市场状态"""
        async with self._lock:
            if symbol not in self._current_regimes:
                return {"status": "no_data", "symbol": symbol}
            
            # 获取HMM转移矩阵
            tf = self._current_regimes[symbol].get("timeframe", "short")
            transition_matrix = None
            mtf = self._symbol_mtfs.get(symbol)
            transition_predictor = self._symbol_transition_predictors.get(symbol, self._transition_predictor)
            if mtf and mtf.is_trained(tf):
                transition_matrix = mtf._classifiers[tf].get_regime_transition_matrix()
            
            prediction = transition_predictor.predict_next_regime(transition_matrix)
            prediction["symbol"] = symbol
            
            return prediction

    async def get_optimal_strategies(self, symbol: str) -> List[Dict[str, Any]]:
        """获取某币种在当前市场状态下的最优策略"""
        async with self._lock:
            if symbol not in self._current_regimes:
                return []
            
            regime_data = self._current_regimes[symbol]
            regime = MarketRegime(regime_data["regime"])
            
            probs = regime_data.get("probabilities", {})
            regime_probs = {}
            for r_str, p in probs.items():
                try:
                    regime_probs[MarketRegime(r_str)] = p
                except ValueError:
                    pass
            
            return self._strategy_mapper.get_optimal_strategies(regime, regime_probs)

    def get_regime(self, symbol: str = None) -> Dict[str, Any]:
        """获取当前市场状态
        
        Args:
            symbol: 交易对符号，None时返回所有
        
        Returns:
            {symbol: regime_dict} 或 {regime_dict, symbols: ...}
        """
        if symbol:
            return self._current_regimes.get(symbol, {"symbol": symbol, "regime": MarketRegime.UNKNOWN.value})
        
        return {
            "timestamp": datetime.now().isoformat(),
            "symbols": dict(self._current_regimes),
            "count": len(self._current_regimes),
        }

    def get_regime_history(self, symbol: str, limit: int = 50) -> List[Dict]:
        """获取某币种的市场状态历史"""
        if symbol not in self._regime_history:
            return []
        
        history = list(self._regime_history[symbol])
        return history[-limit:]

    def get_summary(self) -> Dict[str, Any]:
        """获取检测器整体概览"""
        regime_counts = {r.value: 0 for r in _REGIME_ORDER}
        symbol_regimes = {}
        
        for symbol, data in self._current_regimes.items():
            regime = data.get("regime", "unknown")
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            symbol_regimes[symbol] = regime
        
        # 训练状态汇总所有币种模型，同时保留旧的无币种手动训练模型。
        trained_tfs = sorted({
            tf for tf in self._default_timeframes
            if self._mtf.is_trained(tf)
            or any(mtf.is_trained(tf) for mtf in self._symbol_mtfs.values())
        })
        trained_timeframes_by_symbol = {
            symbol: [tf for tf in self._default_timeframes if mtf.is_trained(tf)]
            for symbol, mtf in self._symbol_mtfs.items()
        }
        all_timeframes_trained = self._mtf.are_all_trained() or any(
            all(mtf.is_trained(tf) for tf in self._default_timeframes)
            for mtf in self._symbol_mtfs.values()
        )
        
        # 转移统计
        transition_stats = {}
        if self._transition_predictor._state_history:
            current = self._transition_predictor._state_history[-1]["regime"]
            durations = self._transition_predictor._compute_durations()
            transition_stats = {
                "current_global_regime": current.value,
                "current_duration": durations.get(current, 0),
            }
        transition_stats_by_symbol = {}
        for symbol, predictor in self._symbol_transition_predictors.items():
            if not predictor._state_history:
                continue
            current = predictor._state_history[-1]["regime"]
            transition_stats_by_symbol[symbol] = {
                "current_regime": current.value,
                "current_duration": predictor._compute_durations().get(current, 0),
            }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "tracked_symbols": len(self._current_regimes),
            "regime_distribution": regime_counts,
            "dominant_regime": max(regime_counts, key=lambda k: regime_counts[k]) if any(v > 0 for v in regime_counts.values()) else MarketRegime.UNKNOWN.value,
            "trained_timeframes": trained_tfs,
            "trained_timeframes_by_symbol": trained_timeframes_by_symbol,
            "all_timeframes_trained": all_timeframes_trained,
            "transition_stats": transition_stats,
            "transition_stats_by_symbol": transition_stats_by_symbol,
            "feature_stats": self._feature_extractor.get_rolling_statistics(),
            "feature_stats_by_symbol": {
                symbol: extractor.get_rolling_statistics()
                for symbol, extractor in self._symbol_feature_extractors.items()
            },
            "persist_dir": self._persist_dir,
            "running": self._running,
        }

    # ===================== 持久化 =====================

    async def _persist(self):
        """持久化HMM模型参数到磁盘"""
        try:
            data = {
                "mtf": self._mtf.to_dict(),
                "mtf_by_symbol": {
                    symbol: mtf.to_dict() for symbol, mtf in self._symbol_mtfs.items()
                },
                "transition_predictor": self._transition_predictor.to_dict(),
                "transition_predictors_by_symbol": {
                    symbol: predictor.to_dict()
                    for symbol, predictor in self._symbol_transition_predictors.items()
                },
                "strategy_mapper": self._strategy_mapper.to_dict(),
                "persisted_at": datetime.now().isoformat(),
            }
            path = os.path.join(self._persist_dir, "hmm_models.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, default=str, indent=2)
            self._last_persist = time.time()
            logger.debug(f"HMM models persisted to {path}")
        except Exception as e:
            self._handle_exception(
                e, module="MarketRegimeDetector", function="_persist",
                severity="medium", category="persistence",
            )

    def _load(self):
        """从磁盘加载HMM模型参数"""
        path = os.path.join(self._persist_dir, "hmm_models.json")
        if not os.path.exists(path):
            logger.info("No persisted HMM models found, starting fresh")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if "mtf" in data:
                self._mtf = MultiTimeframeRegime.from_dict(data["mtf"], self._config)
            self._symbol_mtfs = {
                symbol: MultiTimeframeRegime.from_dict(mtf_data, self._config)
                for symbol, mtf_data in data.get("mtf_by_symbol", {}).items()
            }
            if "transition_predictor" in data:
                self._transition_predictor = RegimeTransitionPredictor.from_dict(data["transition_predictor"])
            self._symbol_transition_predictors = {
                symbol: RegimeTransitionPredictor.from_dict(predictor_data)
                for symbol, predictor_data in data.get("transition_predictors_by_symbol", {}).items()
            }
            if "strategy_mapper" in data:
                self._strategy_mapper = RegimeStrategyMapper.from_dict(data["strategy_mapper"])
            trained_by_symbol = {
                symbol: [tf for tf in self._default_timeframes if mtf.is_trained(tf)]
                for symbol, mtf in self._symbol_mtfs.items()
            }
            logger.info(f"HMM models loaded from {path} "
                         f"(trained: {trained_by_symbol})")
        except Exception as e:
            self._handle_exception(
                e, module="MarketRegimeDetector", function="_load",
                severity="medium", category="persistence",
            )

    # ===================== 手动训练与配置 =====================

    async def train_timeframe(self, timeframe: str, ohlcv_data: List[Dict[str, Any]],
                               n_iterations: int = 50, symbol: Optional[str] = None) -> Dict[str, Any]:
        """手动训练指定时间周期的HMM模型；提供 symbol 时使用币种专属模型。"""
        async with self._lock:
            mtf = self._get_symbol_mtf(symbol) if symbol else self._mtf
            return mtf.train(timeframe, ohlcv_data, n_iterations=n_iterations)

    async def update_strategy_performance(self, strategy: str, symbol: str,
                                            win: bool, pnl: float):
        """更新策略在市场状态下的表现统计"""
        async with self._lock:
            regime_data = self._current_regimes.get(symbol, {})
            regime_str = regime_data.get("regime", "unknown")
            try:
                regime = MarketRegime(regime_str)
            except ValueError:
                regime = MarketRegime.UNKNOWN
            self._strategy_mapper.update_performance(strategy, regime, win, pnl)

    def get_strategy_performance_summary(self) -> Dict[str, Any]:
        """获取所有策略在各状态下的表现摘要"""
        summary = {}
        for strategy, perf_by_regime in self._strategy_mapper._performance.items():
            total_count = sum(p.get("count", 0) for p in perf_by_regime.values())
            total_wins = sum(p.get("wins", 0) for p in perf_by_regime.values())
            total_pnl = sum(p.get("total_pnl", 0) for p in perf_by_regime.values())
            summary[strategy] = {
                "total_trades": total_count,
                "total_wins": total_wins,
                "win_rate": total_wins / max(total_count, 1),
                "total_pnl": total_pnl,
                "by_regime": perf_by_regime,
            }
        return summary

    def verify_regime_accuracy_offline(
        self,
        ohlcv_data: List[Dict[str, Any]],
        symbol: str = "BTC-USDT-SWAP",
        validation_window: int = 20,
        step_size: int = 5,
    ) -> Dict[str, Any]:
        """离线验证 regime 分类准确率：用历史 K 线滚动预测，再对照后续实际走势判定对错。

        Args:
            ohlcv_data: 按时间升序的 OHLCV 序列（至少 100 根）
            validation_window: 预测后等待多少根 K 线再验证
            step_size: 滚动步长（每 step_size 根采样一次预测）
        """
        if len(ohlcv_data) < 100:
            return {"error": "insufficient data", "min_required": 100, "got": len(ohlcv_data)}

        extractor = RegimeFeatureExtractor(self._config)
        mtf = self._mtf

        # 找到已训练的 short 周期 HMM
        hmm = None
        for tf_key in ("short", "medium", "long"):
            clf = mtf._classifiers.get(tf_key)
            if clf and clf._trained:
                hmm = clf
                break
        if hmm is None:
            return {"error": "no trained HMM model available"}

        predictions = []
        closes = [float(c.get("close", 0)) for c in ohlcv_data]

        for i in range(50, len(ohlcv_data) - validation_window, step_size):
            window = ohlcv_data[max(0, i - 60):i + 1]
            try:
                features = extractor.extract_features(window, cache_features=False)
                if not features:
                    continue
                last_features = features[-1]
                regime = hmm.predict(last_features)
                pred_price = closes[i]
                if pred_price <= 0:
                    continue

                future_idx = min(i + validation_window, len(closes) - 1)
                future_price = closes[future_idx]
                actual_change = (future_price - pred_price) / pred_price

                correct = self._judge_regime_prediction(regime, actual_change)
                predictions.append({
                    "index": i,
                    "regime": regime.value if isinstance(regime, MarketRegime) else str(regime),
                    "price": pred_price,
                    "actual_change": round(actual_change, 6),
                    "correct": correct,
                })
            except Exception:
                continue

        if not predictions:
            return {"error": "no valid predictions generated", "data_length": len(ohlcv_data)}

        # 汇总统计
        total = len(predictions)
        correct_total = sum(1 for p in predictions if p["correct"])
        by_regime: Dict[str, Dict[str, Any]] = {}
        for p in predictions:
            r = p["regime"]
            if r not in by_regime:
                by_regime[r] = {"total": 0, "correct": 0, "avg_actual_change": 0.0, "_changes": []}
            by_regime[r]["total"] += 1
            if p["correct"]:
                by_regime[r]["correct"] += 1
            by_regime[r]["_changes"].append(abs(p["actual_change"]))

        for r, stats in by_regime.items():
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] else 0
            stats["avg_abs_change"] = sum(stats["_changes"]) / len(stats["_changes"]) if stats["_changes"] else 0
            del stats["_changes"]

        return {
            "total_predictions": total,
            "correct": correct_total,
            "accuracy": correct_total / total if total else 0,
            "validation_window": validation_window,
            "by_regime": by_regime,
        }

    @staticmethod
    def _judge_regime_prediction(regime, actual_change: float) -> bool:
        """根据实际价格变化判断 regime 预测是否正确。"""
        regime_val = regime.value if isinstance(regime, MarketRegime) else str(regime)
        if regime_val == "trending_up":
            return actual_change > 0.02
        elif regime_val == "trending_down":
            return actual_change < -0.02
        elif regime_val == "ranging":
            return abs(actual_change) < 0.03
        elif regime_val == "breakout":
            return abs(actual_change) > 0.05
        elif regime_val == "high_vol":
            return abs(actual_change) > 0.04
        elif regime_val == "low_vol":
            return abs(actual_change) < 0.015
        elif regime_val == "reversal":
            return abs(actual_change) > 0.03
        else:
            return abs(actual_change) < 0.10