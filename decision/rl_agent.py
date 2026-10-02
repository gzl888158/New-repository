"""
强化学习智能体（RL Agent）— 完善强化版

.. deprecated::
    实验性模块，未接入生产交易链路。保留供架构演进参考。

核心能力:
  1. Q-learning/SARSA 动态参数调优 — 根据市场状态自适应调整策略参数
  2. 多臂老虎机探索 — 自适应选择最优策略/参数组合
  3. 策略参数空间探索 — 杠杆、仓位大小、止损距离等参数的在线优化
  4. 奖励塑形 — 复合奖励函数（PnL + Sharpe + 回撤惩罚）
  5. 经验回放 — 优先级经验回放防止灾难性遗忘
  6. 市场状态编码 — 波动率+趋势+流动性多维状态空间
  7. 多策略联合优化 — 协同优化多个策略的参数配置
  8. 在线vs离线模式 — 支持离线回测训练 + 在线微调
"""
import hashlib
import json
import math
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from loguru import logger
from utils.helpers import safe_finite


# ═══════════════════════════════════════════════════════════════
# 枚举与常量
# ═══════════════════════════════════════════════════════════════

class RLAction(Enum):
    """RL动作类型"""
    INCREASE_LEVERAGE = "increase_leverage"
    DECREASE_LEVERAGE = "decrease_leverage"
    INCREASE_POSITION = "increase_position"
    DECREASE_POSITION = "decrease_position"
    TIGHTEN_STOP = "tighten_stop"
    LOOSEN_STOP = "loosen_stop"
    INCREASE_TP = "increase_tp"
    DECREASE_TP = "decrease_tp"
    NO_CHANGE = "no_change"


class AgentMode(Enum):
    """智能体运行模式"""
    OFFLINE_TRAIN = "offline_train"    # 离线回测训练
    ONLINE_FINETUNE = "online_finetune"  # 在线微调
    EVALUATE = "evaluate"              # 评估模式
    DISABLED = "disabled"


class MarketState(Enum):
    """市场状态编码"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRASH = "crash"
    RECOVERY = "recovery"


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

@dataclass
class StateEncoding:
    """状态编码"""
    market_regime: str = "normal"       # 市场状态
    volatility_percentile: float = 0.5  # 波动率分位数
    trend_strength: float = 0.0         # 趋势强度 [-1, 1]
    liquidity_score: float = 0.5        # 流动性评分
    time_of_day: int = 0                # 小时 (0-23)
    current_drawdown_pct: float = 0.0   # 当前回撤
    position_count: int = 0             # 当前持仓数
    strategy_id: str = ""               # 策略标识
    symbol: str = ""                    # 币种标识（用于融合 MarketRegimeEngine 真实状态）
    factor_score: float = 0.0            # 多因子综合得分 [-1, 1]
    regime_confidence: float = 0.0       # regime 置信度 [0, 1]
    utilization_rate: float = 0.0        # 资金利用率 [0, 1]
    avg_correlation: float = 0.0         # 组合平均相关性 [-1, 1]
    decision_id: str = ""                # 跨 agent 决策链标识，不进入网络特征


@dataclass
class Experience:
    """经验元组"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    priority: float = 1.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentStats:
    """智能体统计"""
    total_steps: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    epsilon: float = 1.0
    episodes_completed: int = 0
    best_episode_reward: float = float("-inf")
    exploration_rate: float = 1.0
    action_distribution: Dict[str, int] = field(default_factory=dict)
    last_update: str = field(default_factory=lambda: datetime.now().isoformat())


# ═══════════════════════════════════════════════════════════════
# SumTree — O(log n) 优先级采样
# ═══════════════════════════════════════════════════════════════

class SumTree:
    """
    二叉SumTree，用于优先级经验回放的O(log n)采样

    叶子节点存储 (priority, data) 对。
    内部节点存储子节点优先级之和。

    支持:
      - add(priority, data): O(log n) 插入
      - update(idx, priority): O(log n) 更新优先级
      - sample(n, beta): O(n log n) 批量采样，返回 (batch, indices, is_weights)
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self._ptr = 0    # 循环写入指针
        self._size = 0   # 当前实际元素数
        self._min_priority = 1e-6

    def __len__(self) -> int:
        return self._size

    def _propagate(self, idx: int, change: float):
        """向上传播优先级变更"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def add(self, priority: float, data):
        """添加新经验（循环覆盖）"""
        leaf_idx = self._ptr + self.capacity - 1
        self.data[self._ptr] = data
        self.update_leaf(leaf_idx, max(priority, self._min_priority))
        self._ptr = (self._ptr + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def update_leaf(self, leaf_idx: int, priority: float):
        """更新叶子节点优先级"""
        change = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        if leaf_idx != 0:
            self._propagate(leaf_idx, change)

    def get_leaf(self, value: float) -> Tuple[int, float, Any]:
        """
        根据采样值查找叶子节点

        Args:
            value: 在 [0, total_priority) 范围内的随机值

        Returns:
            (leaf_idx, priority, data)
        """
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right

        leaf_idx = idx
        data_idx = idx - (self.capacity - 1)
        return leaf_idx, self.tree[idx], self.data[data_idx]

    def total_priority(self) -> float:
        return self.tree[0]

    def sample(self, n: int, beta: float) -> Tuple[List[Any], List[int], np.ndarray]:
        """
        批量优先级采样

        Returns:
            batch: 采样数据列表
            indices: 叶子索引列表（用于更新优先级）
            is_weights: 重要性采样权重 [n]
        """
        batch = []
        indices = []
        total_p = self.total_priority()
        if total_p <= 0:
            # 退化为均匀采样
            valid_data = [d for d in self.data[:self._size] if d is not None]
            if not valid_data:
                return [], [], np.array([])
            n = min(n, len(valid_data))
            batch = list(np.random.choice(valid_data, n, replace=False))
            indices = [0] * n
            is_weights = np.ones(n)
            return batch, indices, is_weights

        segment = total_p / n
        priorities_list = []

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            value = np.random.uniform(a, b)
            leaf_idx, priority, data = self.get_leaf(value)
            batch.append(data)
            indices.append(leaf_idx)
            priorities_list.append(priority)

        # IS权重
        probs = np.array(priorities_list) / total_p
        is_weights = (1.0 / (self._size * probs)) ** beta
        is_weights /= is_weights.max()

        return batch, indices, is_weights

    def update(self, indices: List[int], priorities: np.ndarray):
        """批量更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.update_leaf(idx, max(float(priority), self._min_priority))


# ═══════════════════════════════════════════════════════════════
# 强化学习智能体
# ═══════════════════════════════════════════════════════════════

class TradingRLAgent:
    """
    交易强化学习智能体

    算法: Double DQN + 优先级经验回放(SumTree) + Dueling架构 + N-step TD + 动作掩码
    状态空间: 市场状态 + 策略状态编码
    动作空间: 参数调整动作（杠杆/仓位/止损/止盈）
    奖励: PnL复合奖励（考虑风险调整）
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        rl_cfg = cfg.get("rl_agent", {}) if cfg else {}

        # ── 智能体配置 ──
        self.name = rl_cfg.get("name", "trading_rl_agent")
        self.mode = AgentMode(rl_cfg.get("mode", "online_finetune"))
        self._enabled = rl_cfg.get("enabled", True)

        # ── 状态/动作空间 ──
        self._n_states = rl_cfg.get("n_states", 12)
        self._decision_enabled = bool(rl_cfg.get("decision_enabled", False))
        self._online_training_enabled = bool(rl_cfg.get("online_training_enabled", False))
        self._drift_detection_enabled = bool(rl_cfg.get("drift_detection_enabled", True))
        self._drift_window_size = max(10, int(rl_cfg.get("drift_window_size", 20)))
        self._drift_min_samples = max(10, int(rl_cfg.get("drift_min_samples", 20)))
        self._drift_drop_threshold = max(0.0, float(rl_cfg.get("drift_drop_threshold", 0.5)))
        self._training_frozen = False
        self._training_freeze_reason = ""
        self._actions = list(RLAction)
        self._n_actions = len(self._actions)

        # ── 学习参数 ──
        self._learning_rate = rl_cfg.get("learning_rate", 0.001)
        self._discount_factor = rl_cfg.get("discount_factor", 0.95)
        self._epsilon = rl_cfg.get("epsilon_start", 1.0)
        self._epsilon_min = rl_cfg.get("epsilon_min", 0.05)
        self._epsilon_decay = rl_cfg.get("epsilon_decay", 0.995)
        self._batch_size = rl_cfg.get("batch_size", 32)
        self._target_update_freq = rl_cfg.get("target_update_freq", 100)

        # ── 经验回放 ──
        replay_size = rl_cfg.get("replay_size", 10000)
        self._use_sumtree = rl_cfg.get("use_sumtree", True)
        if self._use_sumtree:
            self._replay_buffer = SumTree(replay_size)
        else:
            self._replay_buffer: deque = deque(maxlen=replay_size)
        self._priority_alpha = rl_cfg.get("priority_alpha", 0.6)
        self._priority_beta = rl_cfg.get("priority_beta", 0.4)
        self._min_priority = 1e-6
        self._priority_beta_increment = rl_cfg.get("priority_beta_increment", 0.001)  # beta从0.4逐渐升温到1.0

        # ── N-step TD学习 ──
        self._n_step = rl_cfg.get("n_step", 3)  # N-step返回
        self._n_step_buffer: deque = deque(maxlen=self._n_step)  # 暂存最近N步

        # ── 梯度裁剪 ──
        self._grad_clip = rl_cfg.get("grad_clip", 10.0)

        # ── 奖励塑形参数 ──
        self._reward_sharpe_weight = rl_cfg.get("reward_sharpe_weight", 0.1)
        self._reward_winrate_weight = rl_cfg.get("reward_winrate_weight", 0.05)
        self._reward_efficiency_weight = rl_cfg.get("reward_efficiency_weight", 0.05)

        # ── Dueling Network ──
        self._use_dueling = rl_cfg.get("use_dueling", True)

        # ── Q网络（简化神经网络用numpy实现） ──
        hidden_size = rl_cfg.get("hidden_size", 64)
        self._q_network = self._build_network(self._n_states, hidden_size, self._n_actions)
        self._target_network = self._build_network(self._n_states, hidden_size, self._n_actions)
        self._sync_target_network()

        # ── 统计 ──
        self._stats = AgentStats()
        self._step_counter = 0
        self._episode_rewards: List[float] = []

        # ── 策略参数空间 ──
        # 从config获取风控约束，确保RL探索空间与实际风控一致
        max_leverage = rl_cfg.get("max_leverage", cfg.get("risk", {}).get("max_leverage", 20))
        max_position_pct = float(cfg.get("risk", {}).get("max_position_per_trade", 0.25))
        max_stop_loss = float(cfg.get("risk", {}).get("max_stop_loss_pct", 0.05))
        max_take_profit = float(cfg.get("strategies", {}).get("grid", {}).get("take_profit_pct", 0.10))

        self._param_bounds: Dict[str, Tuple[float, float]] = {
            "leverage": (1.0, float(max_leverage)),
            "position_pct": (0.01, max_position_pct),
            "stop_loss_pct": (0.005, max_stop_loss),
            "take_profit_pct": (0.01, max_take_profit),
            "trailing_stop_pct": (0.005, 0.04),
        }

        # ── 线程安全 ──
        self._lock = threading.Lock()
        self._persist_dir = rl_cfg.get("persist_dir", "./data/rl_models")
        os.makedirs(self._persist_dir, exist_ok=True)

        # ── 统一市场状态引擎（可选注入，用于真实 regime 状态编码） ──
        self._regime_engine = None
        self._learning_memory = None
        self._last_shared_context: Dict[str, Any] = {}
        self._last_shared_trace_id = ""

        # ── 加载已有模型 ──
        self._load()

        logger.info(
            f"TradingRLAgent '{self.name}' initialized: mode={self.mode.value}, "
            f"states={self._n_states}, actions={self._n_actions}, "
            f"epsilon={self._epsilon:.3f}"
        )

    def set_regime_engine(self, engine) -> None:
        """注入 MarketRegimeEngine，使 RL 状态编码融合真实多因子市场状态。

        未注入时，StateEncoding.market_regime 直接沿用调用方传入的字符串，
        保持向后兼容。
        """
        self._regime_engine = engine
        logger.info("MarketRegimeEngine injected into TradingRLAgent")

    def set_learning_memory(self, memory) -> None:
        """Inject shared cross-agent decision and outcome memory."""
        self._learning_memory = memory

    # ═══════════════════════════════════════════════════════
    # 网络构建
    # ═══════════════════════════════════════════════════════

    def _build_network(self, input_size: int, hidden_size: int, output_size: int) -> Dict[str, np.ndarray]:
        """
        构建网络（Dueling架构：V(s) + A(s,a)）

        Dueling分解:
          共享特征层 → 分离为Value流和Advantage流
          Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))   [identifiability constraint]
        """
        if not self._use_dueling:
            return {
                "W1": np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size),
                "b1": np.zeros(hidden_size),
                "W2": np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size),
                "b2": np.zeros(hidden_size),
                "W3": np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size),
                "b3": np.zeros(output_size),
            }

        net: Dict[str, np.ndarray] = {
            # ── 共享特征层 ──
            "W_shared1": np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size),
            "b_shared1": np.zeros(hidden_size),
            "W_shared2": np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size),
            "b_shared2": np.zeros(hidden_size),
            # ── Value流 V(s) ──
            "W_value": np.random.randn(hidden_size, hidden_size // 2) * np.sqrt(2.0 / hidden_size),
            "b_value": np.zeros(hidden_size // 2),
            "W_value_out": np.random.randn(hidden_size // 2, 1) * np.sqrt(2.0 / (hidden_size // 2)),
            "b_value_out": np.zeros(1),
            # ── Advantage流 A(s,a) ──
            "W_adv": np.random.randn(hidden_size, hidden_size // 2) * np.sqrt(2.0 / hidden_size),
            "b_adv": np.zeros(hidden_size // 2),
            "W_adv_out": np.random.randn(hidden_size // 2, output_size) * np.sqrt(2.0 / (hidden_size // 2)),
            "b_adv_out": np.zeros(output_size),
        }
        return net

    def _forward(self, net: Dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
        """前向传播（Dueling：V(s) + A(s,a) - mean(A)）"""
        if not self._use_dueling:
            h1 = np.maximum(0, x @ net["W1"] + net["b1"])
            h2 = np.maximum(0, h1 @ net["W2"] + net["b2"])
            return h2 @ net["W3"] + net["b3"]

        # ── 共享特征 ──
        sh1 = np.maximum(0, x @ net["W_shared1"] + net["b_shared1"])
        sh2 = np.maximum(0, sh1 @ net["W_shared2"] + net["b_shared2"])

        # ── Value流 ──
        v_h = np.maximum(0, sh2 @ net["W_value"] + net["b_value"])
        value = v_h @ net["W_value_out"] + net["b_value_out"]

        # ── Advantage流 ──
        a_h = np.maximum(0, sh2 @ net["W_adv"] + net["b_adv"])
        advantage = a_h @ net["W_adv_out"] + net["b_adv_out"]

        # ── 组合：Q = V + A - mean(A) ──
        if x.ndim == 1:
            return value + advantage - advantage.mean()
        else:
            return value + advantage - advantage.mean(axis=1, keepdims=True)

    def _sync_target_network(self):
        """同步目标网络"""
        for k in self._q_network:
            self._target_network[k] = self._q_network[k].copy()

    # ═══════════════════════════════════════════════════════
    # 状态编码
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _map_engine_regime_to_state_str(regime_str: str, trend_strength: float = 0.0) -> str:
        """将 MarketRegimeEngine 的 regime 字符串映射为 StateEncoding.market_regime 的键。

        MarketRegimeEngine 输出 trend_bullish / trend_bearish / range_bound /
        extreme_volatility / funding_crush / liquidity_crisis，而 StateEncoding 使用
        trending_up / trending_down / ranging / high_volatility / crash 等键。
        """
        if not regime_str or regime_str == "unknown":
            return "normal"
        if regime_str == "trend_bullish":
            return "trending_up"
        if regime_str == "trend_bearish":
            return "trending_down"
        if regime_str == "range_bound":
            return "ranging"
        if regime_str == "extreme_volatility":
            return "high_volatility"
        if regime_str in ("funding_crush", "liquidity_crisis"):
            return "crash"
        return "normal"

    def _fuse_regime_into_state(self, se: StateEncoding) -> StateEncoding:
        """融合真实 MarketRegimeEngine 数据到 StateEncoding（有引擎且带 symbol 时）。

        用多因子融合结果覆盖 market_regime / trend_strength / volatility_percentile /
        liquidity_score，消除调用方传入陈旧或 naive 状态导致的状态漂移。
        """
        engine = self._regime_engine
        symbol = getattr(se, "symbol", "")
        if engine is None or not symbol:
            return se
        try:
            data = engine.get_symbol_regime(symbol)
        except Exception as e:
            logger.debug(f"RL regime fusion unavailable for {symbol}: {e}")
            return se
        if not data:
            return se
        regime_str = data.get("regime")
        if not regime_str or regime_str == "unknown":
            return se
        factor_scores = data.get("factor_scores")
        factor_weights = data.get("factor_weights") or {}
        factor_score = se.factor_score
        if isinstance(factor_scores, dict) and factor_scores:
            weighted = []
            for name, value in factor_scores.items():
                score = safe_finite(value, 0.0)
                weight = safe_finite(factor_weights.get(name), 1.0)
                if weight > 0:
                    weighted.append((score, weight))
            total_weight = sum(weight for _, weight in weighted)
            if total_weight > 0:
                factor_score = sum(score * weight for score, weight in weighted) / total_weight
        return StateEncoding(
            market_regime=self._map_engine_regime_to_state_str(
                regime_str, float(data.get("trend_strength", 0.0))),
            volatility_percentile=float(data.get("volatility_percentile", se.volatility_percentile)),
            trend_strength=float(data.get("trend_strength", se.trend_strength)),
            liquidity_score=float(data.get("liquidity_score", se.liquidity_score)),
            time_of_day=se.time_of_day,
            current_drawdown_pct=se.current_drawdown_pct,
            position_count=se.position_count,
            strategy_id=se.strategy_id,
            symbol=symbol,
            factor_score=float(np.clip(factor_score, -1.0, 1.0)),
            regime_confidence=float(np.clip(
                safe_finite(data.get("confidence"), se.regime_confidence), 0.0, 1.0
            )),
            utilization_rate=se.utilization_rate,
            avg_correlation=se.avg_correlation,
        )

    def encode_state(self, se: StateEncoding) -> np.ndarray:
        """将StateEncoding编码为神经网络输入向量（融合真实 regime 引擎数据）"""
        se = self._fuse_regime_into_state(se)
        regime_map = {
            "trending_up": 1.0, "trending_down": -1.0, "ranging": 0.0,
            "high_volatility": 0.5, "low_volatility": -0.5,
            "crash": -1.0, "recovery": 0.8, "normal": 0.0,
        }
        state = np.array([
            regime_map.get(se.market_regime, 0.0),
            np.clip(se.volatility_percentile, 0, 1),
            np.clip(se.trend_strength, -1, 1),
            np.clip(se.liquidity_score, 0, 1),
            se.time_of_day / 24.0,
            np.clip(se.current_drawdown_pct * 10, -1, 1),
            np.clip(se.position_count / 10.0, 0, 1),
            1.0 if se.strategy_id else 0.0,
            np.clip(se.factor_score, -1, 1),
            np.clip(se.regime_confidence, 0, 1),
            np.clip(se.utilization_rate, 0, 1),
            np.clip(se.avg_correlation, -1, 1),
        ], dtype=np.float64)
        # 确保维度匹配
        if len(state) < self._n_states:
            state = np.pad(state, (0, self._n_states - len(state)))
        return state[:self._n_states]

    def decode_action(self, action_idx: int) -> RLAction:
        """将动作索引解码为RLAction"""
        if 0 <= action_idx < len(self._actions):
            return self._actions[action_idx]
        return RLAction.NO_CHANGE

    # ═══════════════════════════════════════════════════════
    # 动作选择
    # ═══════════════════════════════════════════════════════

    def select_action(self, state: np.ndarray, explore: bool = True,
                      action_context: Optional[Dict[str, float]] = None) -> Tuple[int, RLAction, float]:
        """
        Epsilon-greedy动作选择（支持动作掩码）

        Args:
            state: 状态向量
            explore: 是否探索
            action_context: 当前参数状态，用于生成动作掩码 {'leverage':5,'position_pct':0.1,...}

        Returns:
            (action_idx, RLAction, q_value)
        """
        mask = self._get_action_mask(action_context)

        with self._lock:
            if explore and np.random.random() < self._epsilon:
                # 掩码探索：仅从有效动作中随机选择
                valid_actions = [i for i in range(self._n_actions) if mask[i]]
                if valid_actions:
                    action_idx = int(np.random.choice(valid_actions))
                else:
                    action_idx = 0  # 无有效动作时选NO_CHANGE
                q_val = 0.0
            else:
                q_values = self._forward(self._q_network, state)
                # 掩码贪心：无效动作的Q值设为 -inf
                masked_q = np.where(mask, q_values, -np.inf)
                action_idx = int(np.argmax(masked_q))
                q_val = float(q_values[action_idx])

            action = self.decode_action(action_idx)
            return action_idx, action, q_val

    def _get_action_mask(self, context: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        根据当前参数状态生成动作有效性掩码

        防止探索时的无效动作：
          - 已达上限时禁止 INCREASE_*
          - 已达下限时禁止 DECREASE_*
          - 无持仓时禁止仓位调整

        Args:
            context: 当前参数状态，如 {'leverage': 5, 'position_pct': 0.1, ...}

        Returns:
            布尔掩码 [n_actions], True=有效
        """
        if context is None:
            return np.ones(self._n_actions, dtype=bool)

        mask = np.ones(self._n_actions, dtype=bool)

        # 获取各参数的当前值和边界
        for param_key, (act_inc, act_dec) in {
            "leverage": (RLAction.INCREASE_LEVERAGE, RLAction.DECREASE_LEVERAGE),
            "position_pct": (RLAction.INCREASE_POSITION, RLAction.DECREASE_POSITION),
            "stop_loss_pct": (RLAction.LOOSEN_STOP, RLAction.TIGHTEN_STOP),
            "take_profit_pct": (RLAction.INCREASE_TP, RLAction.DECREASE_TP),
        }.items():
            current = context.get(param_key)
            if current is None:
                continue
            bounds = self._param_bounds.get(param_key)
            if bounds is None:
                continue
            low, high = bounds
            inc_idx = self._actions.index(act_inc) if act_inc in self._actions else -1
            dec_idx = self._actions.index(act_dec) if act_dec in self._actions else -1
            if current >= high and inc_idx >= 0:
                mask[inc_idx] = False
            if current <= low and dec_idx >= 0:
                mask[dec_idx] = False

        # 无持仓时禁止仓位调整
        if context.get("position_pct", 0.1) == 0:
            inc_pos = self._actions.index(RLAction.INCREASE_POSITION) if RLAction.INCREASE_POSITION in self._actions else -1
            dec_pos = self._actions.index(RLAction.DECREASE_POSITION) if RLAction.DECREASE_POSITION in self._actions else -1
            if inc_pos >= 0:
                mask[inc_pos] = False
            if dec_pos >= 0:
                mask[dec_pos] = False

        return mask

    def select_best_action(self, state: np.ndarray,
                           action_context: Optional[Dict[str, float]] = None) -> Tuple[int, RLAction, float]:
        """选择最优动作（不探索）"""
        return self.select_action(state, explore=False, action_context=action_context)

    # ═══════════════════════════════════════════════════════
    # 训练
    # ═══════════════════════════════════════════════════════

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool, priority: float = None):
        """存储经验到回放缓冲区（支持N-step TD）"""
        with self._lock:
            exp = Experience(
                state=state.copy(),
                action=action,
                reward=reward,
                next_state=next_state.copy(),
                done=done,
                priority=priority or self._min_priority,
            )
            self._n_step_buffer.append(exp)

            # N-step TD: 当缓冲区满或episode结束时，计算N-step return并存入主缓冲区
            if len(self._n_step_buffer) == self._n_step or done:
                # 计算N-step discounted return
                n_step_return = 0.0
                n_step_done = False
                n_step_next_state = self._n_step_buffer[-1].next_state
                for i, e in enumerate(reversed(self._n_step_buffer)):
                    n_step_return = e.reward + self._discount_factor * n_step_return
                    if e.done:
                        n_step_done = True
                        n_step_next_state = e.next_state
                        # 清除已使用的经验
                        while len(self._n_step_buffer) > i + 1:
                            self._n_step_buffer.popleft()
                        break

                # 存储N-step经验到主缓冲区
                first_exp = self._n_step_buffer[0]
                n_exp = Experience(
                    state=first_exp.state,
                    action=first_exp.action,
                    reward=n_step_return,
                    next_state=n_step_next_state,
                    done=n_step_done or done,
                    priority=priority or self._min_priority,
                )
                if hasattr(self._replay_buffer, 'add'):  # SumTree
                    self._replay_buffer.add(n_exp.priority, n_exp)
                else:
                    self._replay_buffer.append(n_exp)

                # 清除已处理的第一个经验
                self._n_step_buffer.popleft()

                # episode结束：清空剩余缓冲区
                if done:
                    # 处理缓冲区剩余的每一步（作为1-step）
                    for e in list(self._n_step_buffer):
                        e.priority = e.priority or self._min_priority
                        if hasattr(self._replay_buffer, 'add'):
                            self._replay_buffer.add(e.priority, e)
                        else:
                            self._replay_buffer.append(e)
                    self._n_step_buffer.clear()

    def train_step(self) -> Optional[float]:
        """单步批量训练（支持Dueling + SumTree IS权重 + 矩阵运算 + 梯度裁剪 + N-step TD）"""
        if self._training_frozen and self.mode == AgentMode.ONLINE_FINETUNE:
            return None
        if len(self._replay_buffer) < self._batch_size:
            return None

        with self._lock:
            # ── 优先级采样（SumTree O(log n) 或 线性 O(n)）──
            if hasattr(self._replay_buffer, 'sample'):  # SumTree
                batch, indices, is_weights = self._replay_buffer.sample(
                    self._batch_size, self._priority_beta
                )
            else:
                priorities = np.array([e.priority for e in self._replay_buffer])
                probs = priorities ** self._priority_alpha
                probs /= probs.sum()
                N = len(self._replay_buffer)
                indices = np.random.choice(N, self._batch_size, p=probs, replace=False)
                batch = [list(self._replay_buffer)[i] for i in indices]
                batch_probs = probs[indices]
                is_weights = (1.0 / (N * batch_probs)) ** self._priority_beta
                is_weights /= is_weights.max()

            # 逐步增加beta到1.0
            self._priority_beta = min(1.0, self._priority_beta + self._priority_beta_increment)

            # ── 构建批次 ──
            states = np.array([e.state for e in batch])
            actions = np.array([e.action for e in batch])
            rewards = np.array([e.reward for e in batch])
            next_states = np.array([e.next_state for e in batch])
            dones = np.array([e.done for e in batch]).astype(np.float64)

            if self._use_dueling:
                loss = self._train_step_dueling(states, actions, rewards, next_states, dones, is_weights, indices)
            else:
                loss = self._train_step_standard(states, actions, rewards, next_states, dones, is_weights, indices)

            # ── 更新目标网络 ──
            self._step_counter += 1
            if self._step_counter % self._target_update_freq == 0:
                self._sync_target_network()

            # ── 自适应epsilon衰减 ──
            self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)
            self._stats.epsilon = self._epsilon
            self._stats.total_steps += 1

            return loss

    def _train_step_standard(self, states, actions, rewards, next_states, dones, is_weights, indices):
        """标准DQN训练（无Dueling）"""
        # 当前Q值
        z1 = states @ self._q_network["W1"] + self._q_network["b1"]
        h1 = np.maximum(0, z1)
        z2 = h1 @ self._q_network["W2"] + self._q_network["b2"]
        h2 = np.maximum(0, z2)
        z3 = h2 @ self._q_network["W3"] + self._q_network["b3"]
        q_current = z3[np.arange(self._batch_size), actions]

        # 目标Q值（Double DQN）
        nz1 = next_states @ self._q_network["W1"] + self._q_network["b1"]
        nh1 = np.maximum(0, nz1)
        nz2 = nh1 @ self._q_network["W2"] + self._q_network["b2"]
        nh2 = np.maximum(0, nz2)
        nz3_online = nh2 @ self._q_network["W3"] + self._q_network["b3"]
        best_actions = np.argmax(nz3_online, axis=1)

        tnz1 = next_states @ self._target_network["W1"] + self._target_network["b1"]
        tnh1 = np.maximum(0, tnz1)
        tnz2 = tnh1 @ self._target_network["W2"] + self._target_network["b2"]
        tnh2 = np.maximum(0, tnz2)
        tnz3 = tnh2 @ self._target_network["W3"] + self._target_network["b3"]
        q_target_next = tnz3[np.arange(self._batch_size), best_actions]
        q_target = rewards + self._discount_factor * q_target_next * (1 - dones)

        td_errors = q_target - q_current
        weighted_td = td_errors * is_weights

        # 梯度计算
        dz3 = np.zeros_like(z3)
        dz3[np.arange(self._batch_size), actions] = weighted_td
        dW3 = h2.T @ dz3 * self._learning_rate / self._batch_size
        db3 = dz3.sum(axis=0) * self._learning_rate / self._batch_size

        dh2 = dz3 @ self._q_network["W3"].T
        dz2 = dh2 * (z2 > 0).astype(float)
        dW2 = h1.T @ dz2 * self._learning_rate / self._batch_size
        db2 = dz2.sum(axis=0) * self._learning_rate / self._batch_size

        dh1 = dz2 @ self._q_network["W2"].T
        dz1 = dh1 * (z1 > 0).astype(float)
        dW1 = states.T @ dz1 * self._learning_rate / self._batch_size
        db1 = dz1.sum(axis=0) * self._learning_rate / self._batch_size

        for grad in (dW3, db3, dW2, db2, dW1, db1):
            np.clip(grad, -self._grad_clip, self._grad_clip, out=grad)

        self._q_network["W3"] += dW3
        self._q_network["b3"] += db3
        self._q_network["W2"] += dW2
        self._q_network["b2"] += db2
        self._q_network["W1"] += dW1
        self._q_network["b1"] += db1

        # 更新优先级
        for i, idx in enumerate(indices):
            new_prio = abs(float(td_errors[i])) + self._min_priority
            if hasattr(self._replay_buffer, 'update_leaf'):  # SumTree
                self._replay_buffer.update_leaf(idx, new_prio)
            else:
                replay_list = list(self._replay_buffer)
                replay_list[idx].priority = new_prio

        return float(np.mean(np.abs(td_errors)))

    def _train_step_dueling(self, states, actions, rewards, next_states, dones, is_weights, indices):
        """Dueling DQN训练（Value流 + Advantage流 分离梯度）"""
        net = self._q_network
        tnet = self._target_network

        # ═══ 当前Q值：Q = V + A - mean(A) ═══
        # 共享特征
        sh1 = states @ net["W_shared1"] + net["b_shared1"]
        sh1_relu = np.maximum(0, sh1)
        sh2 = sh1_relu @ net["W_shared2"] + net["b_shared2"]
        sh2_relu = np.maximum(0, sh2)

        # Value流
        vh = sh2_relu @ net["W_value"] + net["b_value"]
        vh_relu = np.maximum(0, vh)
        value = vh_relu @ net["W_value_out"] + net["b_value_out"]  # (batch, 1)

        # Advantage流
        ah = sh2_relu @ net["W_adv"] + net["b_adv"]
        ah_relu = np.maximum(0, ah)
        advantage = ah_relu @ net["W_adv_out"] + net["b_adv_out"]  # (batch, n_actions)

        # Q = V + A - mean(A)
        adv_mean = advantage.mean(axis=1, keepdims=True)
        q_current_all = value + advantage - adv_mean  # (batch, n_actions)
        q_current = q_current_all[np.arange(self._batch_size), actions]  # (batch,)

        # ═══ 目标Q值（Double DQN，用target网络） ═══
        ns1 = next_states @ net["W_shared1"] + net["b_shared1"]
        ns1_relu = np.maximum(0, ns1)
        ns2 = ns1_relu @ net["W_shared2"] + net["b_shared2"]
        ns2_relu = np.maximum(0, ns2)

        nvh = ns2_relu @ net["W_value"] + net["b_value"]
        nvh_relu = np.maximum(0, nvh)
        nvalue = nvh_relu @ net["W_value_out"] + net["b_value_out"]

        nah = ns2_relu @ net["W_adv"] + net["b_adv"]
        nah_relu = np.maximum(0, nah)
        nadvantage = nah_relu @ net["W_adv_out"] + net["b_adv_out"]

        na_mean = nadvantage.mean(axis=1, keepdims=True)
        nq_online = nvalue + nadvantage - na_mean
        best_actions = np.argmax(nq_online, axis=1)

        # Target网络评估
        tns1 = next_states @ tnet["W_shared1"] + tnet["b_shared1"]
        tns1_relu = np.maximum(0, tns1)
        tns2 = tns1_relu @ tnet["W_shared2"] + tnet["b_shared2"]
        tns2_relu = np.maximum(0, tns2)

        tnvh = tns2_relu @ tnet["W_value"] + tnet["b_value"]
        tnvh_relu = np.maximum(0, tnvh)
        tnvalue = tnvh_relu @ tnet["W_value_out"] + tnet["b_value_out"]

        tnah = tns2_relu @ tnet["W_adv"] + tnet["b_adv"]
        tnah_relu = np.maximum(0, tnah)
        tnadvantage = tnah_relu @ tnet["W_adv_out"] + tnet["b_adv_out"]

        tna_mean = tnadvantage.mean(axis=1, keepdims=True)
        tnq = tnvalue + tnadvantage - tna_mean
        q_target_next = tnq[np.arange(self._batch_size), best_actions]
        q_target = rewards + self._discount_factor * q_target_next * (1 - dones)

        # ═══ TD误差 ═══
        td_errors = q_target - q_current
        weighted_td = td_errors * is_weights  # (batch,)

        # ═══ 反向传播：Q = V + A - mean(A) ═══
        # dQ/dV = 1, dQ/dA_i = 1 - 1/n, dQ/dA_j = -1/n (j≠i)
        n = self._n_actions

        # Advantage输出层梯度
        d_adv_out = np.zeros((self._batch_size, n))
        for b_idx in range(self._batch_size):
            act = actions[b_idx]
            wt = weighted_td[b_idx]
            d_adv_out[b_idx, :] = -wt / n          # 对所有action的mean惩罚
            d_adv_out[b_idx, act] += wt             # 选中的action: 1 - 1/n

        # Value输出层梯度
        d_value_out = weighted_td.reshape(-1, 1)  # dQ/dV = 1

        lr_batch = self._learning_rate / self._batch_size

        # ── Advantage head回传 ──
        dW_adv_out = ah_relu.T @ d_adv_out * lr_batch
        db_adv_out = d_adv_out.sum(axis=0) * lr_batch

        d_ah_relu = d_adv_out @ net["W_adv_out"].T
        d_ah = d_ah_relu * (ah > 0).astype(float)
        dW_adv = sh2_relu.T @ d_ah * lr_batch
        db_adv = d_ah.sum(axis=0) * lr_batch

        d_sh2_adv = d_ah @ net["W_adv"].T

        # ── Value head回传 ──
        dW_value_out = vh_relu.T @ d_value_out * lr_batch
        db_value_out = d_value_out.sum(axis=0) * lr_batch

        d_vh_relu = d_value_out @ net["W_value_out"].T
        d_vh = d_vh_relu * (vh > 0).astype(float)
        dW_value = sh2_relu.T @ d_vh * lr_batch
        db_value = d_vh.sum(axis=0) * lr_batch

        d_sh2_value = d_vh @ net["W_value"].T

        # ── 共享层回传（合并Value和Advantage的梯度） ──
        d_sh2 = d_sh2_adv + d_sh2_value
        d_sh2_relu = d_sh2 * (sh2 > 0).astype(float)
        dW_shared2 = sh1_relu.T @ d_sh2_relu * lr_batch
        db_shared2 = d_sh2_relu.sum(axis=0) * lr_batch

        d_sh1 = d_sh2_relu @ net["W_shared2"].T
        d_sh1_relu = d_sh1 * (sh1 > 0).astype(float)
        dW_shared1 = states.T @ d_sh1_relu * lr_batch
        db_shared1 = d_sh1_relu.sum(axis=0) * lr_batch

        # ── 梯度裁剪 ──
        all_grads = [
            dW_shared1, db_shared1, dW_shared2, db_shared2,
            dW_value, db_value, dW_value_out, db_value_out,
            dW_adv, db_adv, dW_adv_out, db_adv_out,
        ]
        for grad in all_grads:
            np.clip(grad, -self._grad_clip, self._grad_clip, out=grad)

        # ── 权重更新 ──
        net["W_shared1"] += dW_shared1
        net["b_shared1"] += db_shared1
        net["W_shared2"] += dW_shared2
        net["b_shared2"] += db_shared2
        net["W_value"] += dW_value
        net["b_value"] += db_value
        net["W_value_out"] += dW_value_out
        net["b_value_out"] += db_value_out
        net["W_adv"] += dW_adv
        net["b_adv"] += db_adv
        net["W_adv_out"] += dW_adv_out
        net["b_adv_out"] += db_adv_out

        # ── 更新优先级 ──
        for i, idx in enumerate(indices):
            new_prio = abs(float(td_errors[i])) + self._min_priority
            if hasattr(self._replay_buffer, 'update_leaf'):  # SumTree
                self._replay_buffer.update_leaf(idx, new_prio)
            else:
                replay_list = list(self._replay_buffer)
                replay_list[idx].priority = new_prio

        return float(np.mean(np.abs(td_errors)))

    def end_episode(self, total_reward: float):
        """结束一个episode"""
        with self._lock:
            self._episode_rewards.append(total_reward)
            if len(self._episode_rewards) > self._drift_window_size:
                self._episode_rewards = self._episode_rewards[-self._drift_window_size:]
            self._stats.episodes_completed += 1
            self._stats.total_reward += total_reward
            self._stats.avg_reward = self._stats.total_reward / max(self._stats.episodes_completed, 1)
            new_best = total_reward > self._stats.best_episode_reward
            if new_best:
                self._stats.best_episode_reward = total_reward
            was_frozen = self._training_frozen
            self._check_training_drift()
            if new_best or (not was_frozen and self._training_frozen):
                self._save()
            self._stats.last_update = datetime.now().isoformat()
            shared_context = dict(self._last_shared_context)
            shared_trace_id = self._last_shared_trace_id
            self._last_shared_context = {}
            self._last_shared_trace_id = ""
        if self._learning_memory is not None:
            try:
                self._learning_memory.record(
                    agent="rl_agent",
                    kind="episode_outcome",
                    trace_id=shared_trace_id or None,
                    context=shared_context,
                    outcome={"reward": total_reward},
                )
            except Exception as exc:
                logger.debug(f"Shared RL memory write failed: {exc}")

    def _check_training_drift(self) -> bool:
        """Freeze online fine-tuning when recent episode rewards materially regress."""
        if (not self._drift_detection_enabled
            or not self._online_training_enabled
                or self.mode != AgentMode.ONLINE_FINETUNE
                or self._training_frozen
                or len(self._episode_rewards) < self._drift_min_samples):
            return self._training_frozen
        half = max(5, self._drift_min_samples // 2)
        rewards = self._episode_rewards
        baseline = float(np.mean(rewards[:-half]))
        recent = float(np.mean(rewards[-half:]))
        if baseline - recent >= self._drift_drop_threshold:
            self._training_frozen = True
            self._training_freeze_reason = (
                f"reward_drift baseline={baseline:.4f} recent={recent:.4f}"
            )
            logger.error(f"RL online fine-tuning frozen: {self._training_freeze_reason}")
        return self._training_frozen

    def reset_training_drift_freeze(self) -> None:
        """Require an explicit operator action to resume drift-frozen online training."""
        self._training_frozen = False
        self._training_freeze_reason = ""

    # ═══════════════════════════════════════════════════════
    # 奖励计算
    # ═══════════════════════════════════════════════════════

    def compute_reward(self, pnl: float, drawdown_penalty: float = 0.0,
                       risk_adjusted: bool = True, trade_duration: float = 0.0,
                       win_streak: int = 0, loss_streak: int = 0) -> float:
        """
        复合奖励函数（增强版）

        组件:
          - PnL归一化（核心）
          - 回撤惩罚（风险保护）
          - 交易频率惩罚（避免过度交易）
          - Sharpe贡献（盈利质量）
          - 胜率贡献（一致性奖励）
          - 持仓效率（快速获利奖励）

        Args:
            pnl: 单笔PnL
            drawdown_penalty: 回撤惩罚系数
            risk_adjusted: 是否风险调整
            trade_duration: 持仓持续时间（秒），0表示未知
            win_streak: 连续盈利次数
            loss_streak: 连续亏损次数
        """
        # ── 1. PnL归一化（假设单笔盈亏在-50到+50 USDT范围） ──
        pnl_reward = np.clip(pnl / 20.0, -2.5, 2.5)

        # ── 2. 回撤惩罚 ──
        dd_penalty = drawdown_penalty * 2.0 if drawdown_penalty > 0.03 else 0

        # ── 3. 交易频率惩罚（每步小惩罚鼓励少交易） ──
        action_penalty = 0.01

        # ── 4. Sharpe-like奖励（盈利时为正值增加奖励） ──
        sharpe_bonus = 0.0
        if pnl > 0 and risk_adjusted:
            sharpe_bonus = self._reward_sharpe_weight * min(pnl / max(abs(pnl), 1e-6), 2.0)

        # ── 5. 胜率/持续性奖励 ──
        winrate_bonus = 0.0
        if win_streak >= 3:
            winrate_bonus = self._reward_winrate_weight * min(win_streak, 5)
        elif loss_streak >= 3:
            winrate_bonus = -self._reward_winrate_weight * min(loss_streak, 3)

        # ── 6. 持仓效率奖励（快速获利 > 慢速获利） ──
        efficiency_bonus = 0.0
        if pnl > 0 and trade_duration > 0:
            # 单位时间收益率越高越好（年化概念）
            hourly_return = (pnl / max(trade_duration / 3600.0, 0.01))
            efficiency_bonus = self._reward_efficiency_weight * np.clip(hourly_return, -1.0, 2.0)

        # ── 组合 ──
        reward = pnl_reward - dd_penalty - action_penalty + sharpe_bonus + winrate_bonus + efficiency_bonus

        return round(reward, 4)

    # ═══════════════════════════════════════════════════════
    # 参数调优
    # ═══════════════════════════════════════════════════════

    def get_parameter_adjustment(self, param_name: str,
                                  current_value: float,
                                  state: Optional[StateEncoding] = None) -> float:
        """
        根据Q值推荐参数调整

        Args:
            param_name: 参数名称
            current_value: 当前值
            state: 当前市场状态；缺失时不输出参数建议

        Returns:
            调整后的参数值
        """
        if not self._decision_enabled or state is None:
            return current_value

        bounds = self._param_bounds.get(param_name)
        actions = {
            "leverage": (RLAction.INCREASE_LEVERAGE, RLAction.DECREASE_LEVERAGE),
            "position_pct": (RLAction.INCREASE_POSITION, RLAction.DECREASE_POSITION),
            "stop_loss_pct": (RLAction.LOOSEN_STOP, RLAction.TIGHTEN_STOP),
            "trailing_stop_pct": (RLAction.LOOSEN_STOP, RLAction.TIGHTEN_STOP),
            "take_profit_pct": (RLAction.INCREASE_TP, RLAction.DECREASE_TP),
        }
        if bounds is None or param_name not in actions:
            return current_value

        shared_context = {
            "symbol": state.symbol,
            "strategy": state.strategy_id,
            "regime": state.market_regime,
            "confidence": state.regime_confidence,
            "factor_score": state.factor_score,
            "utilization_rate": state.utilization_rate,
            "avg_correlation": state.avg_correlation,
        }
        self._last_shared_context = shared_context
        self._last_shared_trace_id = state.decision_id
        if (self._learning_memory is not None
                and self._learning_memory.has_veto(
                    shared_context, trace_id=state.decision_id or None
                )):
            self._learning_memory.record(
                agent="rl_agent",
                kind="shared_veto_applied",
                trace_id=state.decision_id or None,
                context=shared_context,
                decision=param_name,
                outcome="recent_intelligent_agent_rejection",
            )
            return current_value

        _, action, _ = self.select_action(
            self.encode_state(state),
            explore=False,
            action_context={param_name: current_value},
        )
        increase_action, decrease_action = actions[param_name]
        direction = 1.0 if action == increase_action else -1.0 if action == decrease_action else 0.0
        scale = (bounds[1] - bounds[0]) * 0.05
        adjusted = current_value + direction * scale
        adjusted = round(max(bounds[0], min(bounds[1], adjusted)), 4)
        if self._learning_memory is not None:
            try:
                self._learning_memory.record(
                    agent="rl_agent",
                    kind="parameter_recommendation",
                    trace_id=state.decision_id or None,
                    context=shared_context,
                    decision={"parameter": param_name, "action": action.value},
                    outcome={"current": current_value, "recommended": adjusted},
                )
            except Exception as exc:
                logger.debug(f"Shared RL recommendation write failed: {exc}")
        return adjusted

    def explore_parameter_space(self, state: Optional[StateEncoding] = None) -> Dict[str, float]:
        """
        多臂老虎机探索：在参数空间中探索最优配置

        Args:
            state: 当前市场状态编码（None则使用保守默认状态）

        Returns:
            探索的参数组合
        """
        # 使用实际市场状态或保守默认状态
        if state is not None:
            state_vec = self.encode_state(state)
        else:
            # 保守默认状态：低波动、无趋势、中性市场
            default_state = StateEncoding(
                market_regime="ranging",
                volatility_percentile=0.5,
                trend_strength=0.0,
                liquidity_score=0.5,
            )
            state_vec = self.encode_state(default_state)
        
        params = {}
        for name, (low, high) in self._param_bounds.items():
            action_idx, _, _ = self.select_action(state_vec)
            action_type = self.decode_action(action_idx)

            mid = (low + high) / 2
            if "increase" in action_type.value:
                val = mid + (high - mid) * np.random.random() * 0.3
            elif "decrease" in action_type.value:
                val = mid - (mid - low) * np.random.random() * 0.3
            else:
                val = mid
            params[name] = round(max(low, min(high, val)), 4)

        return params

    # ═══════════════════════════════════════════════════════
    # 批量优化
    # ═══════════════════════════════════════════════════════

    def optimize_strategy_params(self, current_params: Dict[str, float],
                                  state: StateEncoding,
                                  n_iterations: int = 10) -> Dict[str, float]:
        """
        多策略联合参数优化

        Args:
            current_params: 当前参数配置
            state: 当前市场状态
            n_iterations: 优化迭代次数

        Returns:
            优化后的参数配置
        """
        state_vec = self.encode_state(state)
        optimized = dict(current_params)

        for _ in range(n_iterations):
            for param_name in self._param_bounds:
                action_idx, action, _ = self.select_action(state_vec, explore=True)
                current = optimized.get(param_name, 0.5)

                adjustment = self.get_parameter_adjustment(param_name, current)

                # 根据动作微调
                if action in (RLAction.INCREASE_LEVERAGE, RLAction.INCREASE_POSITION,
                              RLAction.INCREASE_TP, RLAction.LOOSEN_STOP):
                    adjustment = current * 1.02
                elif action in (RLAction.DECREASE_LEVERAGE, RLAction.DECREASE_POSITION,
                                RLAction.DECREASE_TP, RLAction.TIGHTEN_STOP):
                    adjustment = current * 0.98

                bounds = self._param_bounds.get(param_name, (0, float("inf")))
                optimized[param_name] = round(max(bounds[0], min(bounds[1], adjustment)), 4)

        return optimized

    # ═══════════════════════════════════════════════════════
    # MAB (Multi-Armed Bandit) 策略选择
    # ═══════════════════════════════════════════════════════

    def __init_mab(self):
        """MAB内部初始化"""
        pass  # MAB状态在__init__后通过属性设置

    def select_strategy_mab(self, strategy_names: List[str],
                            strategy_scores: Dict[str, float]) -> str:
        """
        多臂老虎机选择最优策略

        使用UCB1算法平衡探索与利用

        Args:
            strategy_names: 候选策略名列表
            strategy_scores: 策略历史得分

        Returns:
            选中的策略名
        """
        if not hasattr(self, '_mab_counts'):
            self._mab_counts: Dict[str, int] = defaultdict(int)
            self._mab_values: Dict[str, float] = defaultdict(float)

        if not strategy_names:
            return ""

        # 确保所有策略都已探索过
        for name in strategy_names:
            if self._mab_counts[name] == 0:
                self._mab_counts[name] = 1
                self._mab_values[name] = strategy_scores.get(name, 0.5)
                return name

        total_counts = sum(self._mab_counts[n] for n in strategy_names)
        ucb_scores = {}

        for name in strategy_names:
            avg_reward = self._mab_values[name]
            exploration_bonus = math.sqrt(2 * math.log(total_counts + 1) / self._mab_counts[name])
            ucb_scores[name] = avg_reward + exploration_bonus

        return max(ucb_scores, key=ucb_scores.get)

    def update_mab(self, strategy_name: str, reward: float):
        """更新MAB奖励"""
        if not hasattr(self, '_mab_counts'):
            self._mab_counts = defaultdict(int)
            self._mab_values = defaultdict(float)

        self._mab_counts[strategy_name] += 1
        n = self._mab_counts[strategy_name]
        self._mab_values[strategy_name] += (reward - self._mab_values[strategy_name]) / n

    # ═══════════════════════════════════════════════════════
    # 持久化
    # ═══════════════════════════════════════════════════════

    def _save(self):
        """保存模型到磁盘"""
        path = os.path.join(self._persist_dir, f"{self.name}_model.npz")
        try:
            np.savez(path, **self._q_network)
            # 保存元数据
            meta_path = os.path.join(self._persist_dir, f"{self.name}_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "epsilon": self._epsilon,
                    "steps": self._stats.total_steps,
                    "episodes": self._stats.episodes_completed,
                    "avg_reward": self._stats.avg_reward,
                    "best_reward": self._stats.best_episode_reward,
                    "training_frozen": self._training_frozen,
                    "training_freeze_reason": self._training_freeze_reason,
                    "updated_at": datetime.now().isoformat(),
                }, f, indent=2)
            logger.debug(f"RL agent '{self.name}' model saved")
        except Exception as e:
            logger.warning(f"Failed to save RL model: {e}")

    def _load(self) -> bool:
        """从磁盘加载模型"""
        path = os.path.join(self._persist_dir, f"{self.name}_model.npz")
        if not os.path.exists(path):
            return False
        try:
            data = np.load(path, allow_pickle=True)
            loaded = {k: data[k] for k in data.files}
            if not all(k in loaded for k in self._q_network):
                logger.warning("RL model keys do not match current network; using fresh model")
                return False
            if any(loaded[k].shape != self._q_network[k].shape for k in self._q_network):
                logger.warning("RL model shape mismatch; using fresh model for new state schema")
                return False
            self._q_network = loaded
            self._sync_target_network()
            # 加载元数据
            meta_path = os.path.join(self._persist_dir, f"{self.name}_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self._epsilon = meta.get("epsilon", self._epsilon)
                self._stats.total_steps = meta.get("steps", 0)
                self._stats.episodes_completed = meta.get("episodes", 0)
                self._stats.avg_reward = meta.get("avg_reward", 0)
                self._stats.best_episode_reward = meta.get("best_reward", float("-inf"))
                self._training_frozen = bool(meta.get("training_frozen", False))
                self._training_freeze_reason = str(meta.get("training_freeze_reason", ""))
            logger.info(f"RL agent '{self.name}' model loaded")
            return True
        except Exception as e:
            logger.warning(f"Failed to load RL model: {e}")
            return False

    # ═══════════════════════════════════════════════════════
    # 状态查询
    # ═══════════════════════════════════════════════════════

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "mode": self.mode.value,
                "enabled": self._enabled,
                "decision_enabled": self._decision_enabled,
                "online_training_enabled": self._online_training_enabled,
                "training_frozen": self._training_frozen,
                "training_freeze_reason": self._training_freeze_reason,
                "epsilon": self._epsilon,
                "total_steps": self._stats.total_steps,
                "episodes_completed": self._stats.episodes_completed,
                "avg_reward": self._stats.avg_reward,
                "best_episode_reward": self._stats.best_episode_reward,
                "replay_size": len(self._replay_buffer),
                "n_actions": self._n_actions,
                "n_states": self._n_states,
                "mab_strategies": len(getattr(self, '_mab_counts', {})),
                "last_update": self._stats.last_update,
            }

    def get_q_values(self, state: StateEncoding) -> Dict[str, float]:
        """获取当前状态下各动作的Q值"""
        state_vec = self.encode_state(state)
        with self._lock:
            q_values = self._forward(self._q_network, state_vec)
        return {
            self._actions[i].value: float(q_values[i])
            for i in range(self._n_actions)
        }

    def is_enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool):
        self._enabled = enabled

    def reset(self):
        """重置智能体（保留网络权重）"""
        with self._lock:
            if hasattr(self._replay_buffer, 'add'):
                capacity = self._replay_buffer.capacity
                self._replay_buffer = SumTree(capacity)
            else:
                self._replay_buffer.clear()
            self._epsilon = 1.0
            self._stats = AgentStats()
            self._episode_rewards.clear()
            logger.info(f"RL agent '{self.name}' reset")


# ═══════════════════════════════════════════════════════════════
# 全局单例
# ═══════════════════════════════════════════════════════════════

_rl_agent_instance: Optional[TradingRLAgent] = None
_rl_agent_lock = threading.Lock()


def get_rl_agent(config: Optional[Dict[str, Any]] = None) -> TradingRLAgent:
    """获取全局RL智能体单例"""
    global _rl_agent_instance
    with _rl_agent_lock:
        if _rl_agent_instance is None:
            _rl_agent_instance = TradingRLAgent(config)
        return _rl_agent_instance
