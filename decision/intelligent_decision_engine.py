"""
智能决策核心引擎（Intelligent Decision Engine）

完善强化的智能决策系统，提供：
  - 多时间框架贝叶斯决策融合（MTF Bayesian Fusion）
  - 决策成本收益分析（Expected Value + Fee/Slippage）
  - 元决策器（Meta-Decision Maker：决策是否该做决策）
  - 自适应决策阈值（市场状态感知）
  - 决策异常检测与回滚
  - 决策上下文丰富化（订单簿、跨策略、历史模式）
  - 决策延迟监控与优化
"""
import asyncio
import hashlib
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
from loguru import logger


# ═══════════════════════════════════════════════════════════════
# 枚举与数据模型
# ═══════════════════════════════════════════════════════════════

class DecisionAction(Enum):
    """决策动作"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    REDUCE = "reduce"
    CLOSE = "close"
    INCREASE = "increase"
    NO_ACTION = "no_action"


class DecisionUrgency(Enum):
    """决策紧急程度"""
    IMMEDIATE = "immediate"      # 立即执行（如止损）
    HIGH = "high"                # 高优先级
    NORMAL = "normal"
    LOW = "low"
    DEFERRED = "deferred"        # 可延迟


class MetaDecisionVerdict(Enum):
    """元决策结论"""
    PROCEED = "proceed"           # 可以决策
    DEFER = "defer"               # 暂缓决策
    REDUCE_SIZE = "reduce_size"   # 减仓决策
    ABSTAIN = "abstain"           # 放弃决策
    EMERGENCY_ONLY = "emergency_only"  # 仅紧急决策


class FusionMethod(Enum):
    """融合方法"""
    BAYESIAN = "bayesian"                 # 贝叶斯融合
    DEMPSTER_SHAFER = "dempster_shafer"   # D-S证据理论
    KALMAN = "kalman"                     # 卡尔曼滤波
    WEIGHTED = "weighted"                 # 加权融合
    VOTING = "voting"                     # 投票


@dataclass
class TimeFrameSignal:
    """多时间框架信号"""
    timeframe: str                       # 5m / 15m / 1h / 4h / 1d
    direction: str                       # buy / sell / hold
    strength: float = 0.0               # 信号强度 [0, 1]
    confidence: float = 0.0             # 置信度 [0, 1]
    source: str = ""                    # 信号来源
    indicators: Dict[str, float] = field(default_factory=dict)  # 指标快照
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MTFFusionResult:
    """MTF融合结果"""
    final_direction: str = "hold"
    final_confidence: float = 0.0
    fusion_method: str = "bayesian"
    # 各时间框架权重
    tf_weights: Dict[str, float] = field(default_factory=dict)
    # 贝叶斯后验概率（buy/sell/hold 三元，和为 1）
    posterior_buy: float = 0.0
    posterior_sell: float = 0.0
    posterior_hold: float = 0.0
    # 时间框架一致性
    tf_agreement: float = 0.0           # 多时间框架一致度 [0, 1]
    tf_divergence: bool = False         # 是否存在时间框架背离
    # 融合信号列表
    fused_signals: List[Dict[str, Any]] = field(default_factory=list)
    # 决策建议
    recommendation: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class CostBenefitAnalysis:
    """决策成本收益分析"""
    decision_id: str = ""
    # 预期收益
    expected_profit: float = 0.0         # 预期利润 (USDT)
    expected_profit_pct: float = 0.0     # 预期利润率
    # 成本
    taker_fee: float = 0.0              # Taker手续费
    maker_fee: float = 0.0              # Maker手续费
    estimated_slippage: float = 0.0     # 预估滑点
    funding_cost: float = 0.0           # 预计资金费率成本
    total_cost: float = 0.0             # 总成本
    # 风险调整
    risk_adjusted_return: float = 0.0   # 风险调整后收益
    sharpe_contribution: float = 0.0    # 对组合夏普的边际贡献
    # 机会成本
    opportunity_cost: float = 0.0       # 机会成本（若不执行）
    # 净收益
    net_expected_value: float = 0.0     # 净期望值
    # 判断
    is_profitable: bool = False
    breakeven_move_pct: float = 0.0     # 盈亏平衡所需移动%
    recommendation: str = ""


@dataclass
class DecisionContext:
    """决策上下文"""
    symbol: str = ""
    # 市场微观结构
    bid_ask_spread_pct: float = 0.0
    order_book_imbalance: float = 0.0    # 订单簿不平衡度 [-1, 1]
    depth_10_bids: float = 0.0
    depth_10_asks: float = 0.0
    # 波动率环境
    current_volatility: float = 0.0
    volatility_percentile: float = 0.0
    # 市场状态
    market_regime: str = "unknown"
    trend_strength: float = 0.0
    # 跨策略上下文
    active_strategies_on_symbol: List[str] = field(default_factory=list)
    conflicting_positions: bool = False
    total_exposure_pct: float = 0.0
    # 历史模式
    similar_pattern_count: int = 0
    pattern_success_rate: float = 0.0


@dataclass
class DecisionAuditEntry:
    """决策审计条目"""
    decision_id: str = ""
    timestamp: str = ""
    decision_type: str = ""
    symbol: str = ""
    direction: str = ""
    # 决策链信息
    source_signals: List[Dict[str, Any]] = field(default_factory=list)
    fusion_result: Dict[str, Any] = field(default_factory=dict)
    cost_benefit: Dict[str, Any] = field(default_factory=dict)
    meta_verdict: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    # 执行结果
    executed: bool = False
    execution_latency_ms: float = 0.0
    outcome: str = "pending"
    outcome_pnl: float = 0.0
    # 归因
    attribution: Dict[str, float] = field(default_factory=dict)
    # 审计链
    prev_hash: str = ""
    entry_hash: str = ""
    # 异常标记
    is_anomaly: bool = False
    anomaly_reason: str = ""

    def compute_hash(self) -> str:
        """
        计算审计条目哈希（企业级防篡改）

        覆盖决策时点的全部不可变字段，保证任何决策内容的篡改都会破坏哈希链。
        注意：executed / execution_latency_ms / outcome / outcome_pnl 是决策后的
        结果回写字段，不纳入哈希，以便结果闭环回写时不破坏链完整性。
        """
        content = json.dumps({
            "decision_id": self.decision_id,
            "timestamp": self.timestamp,
            "decision_type": self.decision_type,
            "symbol": self.symbol,
            "direction": self.direction,
            "source_signals": self.source_signals,
            "fusion_result": self.fusion_result,
            "cost_benefit": self.cost_benefit,
            "meta_verdict": self.meta_verdict,
            "context": self.context,
            "attribution": self.attribution,
            "is_anomaly": self.is_anomaly,
            "anomaly_reason": self.anomaly_reason,
            "prev_hash": self.prev_hash,
        }, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp,
            "decision_type": self.decision_type,
            "symbol": self.symbol,
            "direction": self.direction,
            "source_signals": self.source_signals,
            "fusion_result": self.fusion_result,
            "cost_benefit": self.cost_benefit,
            "meta_verdict": self.meta_verdict,
            "context": self.context,
            "executed": self.executed,
            "execution_latency_ms": self.execution_latency_ms,
            "outcome": self.outcome,
            "outcome_pnl": self.outcome_pnl,
            "attribution": self.attribution,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
            "is_anomaly": self.is_anomaly,
            "anomaly_reason": self.anomaly_reason,
        }


# ═══════════════════════════════════════════════════════════════
# 智能决策引擎
# ═══════════════════════════════════════════════════════════════

class IntelligentDecisionEngine:
    """
    智能决策核心引擎

    核心功能：
      1. 多时间框架贝叶斯决策融合（MTF Bayesian Fusion）
      2. 决策成本收益分析（含手续费/滑点/资金费率）
      3. 元决策器（是否应该做决策）
      4. 自适应决策阈值
      5. 决策异常检测
      6. 决策上下文丰富化
      7. 决策延迟监控
      8. 不可变决策审计链
    """

    def __init__(self, config: Dict[str, Any] = None):
        self._config = config or {}
        ide_cfg = self._config.get("intelligent_decision_engine", {}) if self._config else {}
        self._lock = asyncio.Lock()

        # ── 融合配置 ──
        method_str = ide_cfg.get("fusion_method", "bayesian")
        self._fusion_method = FusionMethod(method_str) if method_str in [m.value for m in FusionMethod] else FusionMethod.BAYESIAN
        # 时间框架权重：越长周期权重越高（支持配置覆盖）
        self._tf_weights = ide_cfg.get("tf_weights", {"5m": 0.10, "15m": 0.15, "1h": 0.25, "4h": 0.30, "1d": 0.20})
        # 贝叶斯先验概率（支持配置覆盖）
        self._prior_buy = ide_cfg.get("prior_buy", 0.33)
        self._prior_sell = ide_cfg.get("prior_sell", 0.33)
        self._prior_hold = ide_cfg.get("prior_hold", 0.34)
        # TF一致性阈值
        self._tf_agreement_threshold = ide_cfg.get("tf_agreement_threshold", 0.60)

        # ── 决策阈值（自适应，支持配置覆盖）──
        self._base_confidence_threshold = ide_cfg.get("base_confidence_threshold", 0.43)
        self._min_confidence_threshold = ide_cfg.get("min_confidence_threshold", 0.30)
        self._max_confidence_threshold = ide_cfg.get("max_confidence_threshold", 0.70)
        self._current_threshold = self._base_confidence_threshold

        # ── 元决策配置（支持配置覆盖）──
        self._min_decision_interval_ms = ide_cfg.get("min_decision_interval_ms", 200)
        self._max_decisions_per_minute = ide_cfg.get("max_decisions_per_minute", 30)
        self._cooldown_after_loss_ms = ide_cfg.get("cooldown_after_loss_ms", 5000)
        self._min_expected_value_usdt = ide_cfg.get("min_expected_value_usdt", 0.50)

        # ── 成本参数（支持配置覆盖）──
        self._taker_fee_rate = ide_cfg.get("taker_fee_rate", 0.0005)
        self._maker_fee_rate = ide_cfg.get("maker_fee_rate", 0.0002)
        self._base_slippage_pct = ide_cfg.get("base_slippage_pct", 0.0002)
        self._funding_rate_annual = ide_cfg.get("funding_rate_annual", 0.10)

        # ── 状态追踪 ──
        self._decision_history: deque = deque(maxlen=500)
        self._last_decision_time: Optional[datetime] = None
        self._decision_count_1m: deque = deque(maxlen=60)  # 滑动窗口计数
        self._consecutive_losses: int = 0
        self._consecutive_wins: int = 0
        self._recent_latencies: deque = deque(maxlen=100)  # 延迟记录(ms)
        self._anomaly_detector_state: Dict[str, Any] = {}

        # ── 不可变审计链 ──
        self._audit_chain: List[DecisionAuditEntry] = []
        self._last_audit_hash: str = "0" * 16  # 创世哈希
        self._max_audit_entries = 10000
        self._audit_persist_path: str = ""

        # ── 上下文缓存 ──
        self._market_contexts: Dict[str, DecisionContext] = {}

        # ── 依赖注入 ──
        self._ensemble_maker = None
        self._risk_gate = None
        self._portfolio_optimizer = None

        logger.info(
            f"IntelligentDecisionEngine initialized: fusion={self._fusion_method.value}, "
            f"base_threshold={self._base_confidence_threshold:.0%}"
        )

    # ── 依赖注入 ─────────────────────────────────────────────

    def set_ensemble_maker(self, maker) -> None:
        self._ensemble_maker = maker

    def set_risk_gate(self, gate) -> None:
        self._risk_gate = gate

    def set_portfolio_optimizer(self, optimizer) -> None:
        self._portfolio_optimizer = optimizer

    def set_audit_persist_path(self, path: str) -> None:
        self._audit_persist_path = path

    # ═══════════════════════════════════════════════════════════
    # 核心1: 多时间框架贝叶斯决策融合
    # ═══════════════════════════════════════════════════════════

    async def fuse_multi_timeframe(
        self,
        tf_signals: List[TimeFrameSignal],
        method: FusionMethod = None,
    ) -> MTFFusionResult:
        """
        多时间框架贝叶斯融合

        原理：
          将每个时间框架的信号视为独立的"证据"，
          使用贝叶斯定理融合成统一的后验概率分布。

        P(direction | signals) ∝ P(signals | direction) * P(direction)
        """
        if method is None:
            method = self._fusion_method

        result = MTFFusionResult(fusion_method=method.value)

        if not tf_signals:
            result.warnings.append("No timeframe signals provided")
            return result

        # ── 按时间框架分组权重 ──
        tf_weighted: Dict[str, List[TimeFrameSignal]] = {}
        for sig in tf_signals:
            tf_weighted.setdefault(sig.timeframe, []).append(sig)

        # ── 计算各TF的综合信号 ──
        tf_aggregated: Dict[str, Dict[str, float]] = {}
        for tf, signals in tf_weighted.items():
            buy_strength = sum(s.strength * s.confidence for s in signals if s.direction in ("buy", "long"))
            sell_strength = sum(s.strength * s.confidence for s in signals if s.direction in ("sell", "short"))
            hold_strength = sum(s.strength * s.confidence for s in signals if s.direction == "hold")
            total = buy_strength + sell_strength + hold_strength or 1.0
            tf_aggregated[tf] = {
                "buy": buy_strength / total,
                "sell": sell_strength / total,
                "hold": hold_strength / total,
            }

        result.tf_weights = {
            tf: self._tf_weights.get(tf, 0.1) for tf in tf_aggregated
        }

        # ── 贝叶斯融合 ──
        if method == FusionMethod.BAYESIAN:
            self._bayesian_fusion(result, tf_aggregated)
        elif method == FusionMethod.DEMPSTER_SHAFER:
            self._dempster_shafer_fusion(result, tf_aggregated)
        elif method == FusionMethod.KALMAN:
            self._kalman_fusion(result, tf_aggregated)
        elif method == FusionMethod.WEIGHTED:
            self._weighted_fusion(result, tf_aggregated)
        elif method == FusionMethod.VOTING:
            self._voting_fusion(result, tf_aggregated)
        else:
            self._bayesian_fusion(result, tf_aggregated)

        # ── TF一致性计算 ──
        directions = []
        for tf, agg in tf_aggregated.items():
            best = max(agg, key=agg.get)
            directions.append(best)

        most_common = max(set(directions), key=directions.count)
        result.tf_agreement = directions.count(most_common) / max(len(directions), 1)

        if result.tf_agreement < self._tf_agreement_threshold:
            result.tf_divergence = True
            result.warnings.append(
                f"Timeframe divergence detected (agreement={result.tf_agreement:.1%})"
            )

        # 确定最终方向（buy/sell/hold 三元后验）
        probs = {
            "buy": result.posterior_buy,
            "sell": result.posterior_sell,
            "hold": result.posterior_hold,
        }
        best_dir = max(probs, key=probs.get)
        best_prob = probs[best_dir]

        # 仅当 buy/sell 显著占优时才开仓，否则 hold 以避免过度交易
        if best_dir in ("buy", "sell") and best_prob > 0.40:
            result.final_direction = best_dir
            result.final_confidence = best_prob
        else:
            result.final_direction = "hold"
            result.final_confidence = best_prob

        # 背离时降低置信度
        if result.tf_divergence:
            result.final_confidence *= 0.70

        result.recommendation = (
            f"Fused {len(tf_signals)} signals across {len(tf_aggregated)} timeframes; "
            f"direction={result.final_direction}, confidence={result.final_confidence:.2%}"
        )

        return result

    def _bayesian_fusion(self, result: MTFFusionResult, tf_aggregated: Dict[str, Dict[str, float]]):
        """贝叶斯融合：逐TF更新后验概率"""
        # 先验
        posterior_buy = self._prior_buy
        posterior_sell = self._prior_sell
        posterior_hold = self._prior_hold

        for tf, agg in tf_aggregated.items():
            weight = self._tf_weights.get(tf, 0.1)

            # 似然 ≈ 该TF的信号强度
            likelihood_buy = agg.get("buy", 0.3) * weight + 0.3 * (1 - weight)
            likelihood_sell = agg.get("sell", 0.3) * weight + 0.3 * (1 - weight)
            likelihood_hold = agg.get("hold", 0.3) * weight + 0.3 * (1 - weight)

            # 贝叶斯更新
            posterior_buy *= likelihood_buy
            posterior_sell *= likelihood_sell
            posterior_hold *= likelihood_hold

            # 归一化
            total = posterior_buy + posterior_sell + posterior_hold
            if total > 0:
                posterior_buy /= total
                posterior_sell /= total
                posterior_hold /= total

        result.posterior_buy = posterior_buy
        result.posterior_sell = posterior_sell
        result.posterior_hold = posterior_hold

    def _weighted_fusion(self, result: MTFFusionResult, tf_aggregated: Dict[str, Dict[str, float]]):
        """加权融合：按TF权重直接加权平均"""
        buy_score = sell_score = hold_score = 0.0
        total_weight = 0.0

        for tf, agg in tf_aggregated.items():
            w = self._tf_weights.get(tf, 0.1)
            buy_score += agg.get("buy", 0) * w
            sell_score += agg.get("sell", 0) * w
            hold_score += agg.get("hold", 0) * w
            total_weight += w

        if total_weight > 0:
            result.posterior_buy = buy_score / total_weight
            result.posterior_sell = sell_score / total_weight
            result.posterior_hold = hold_score / total_weight

    def _voting_fusion(self, result: MTFFusionResult, tf_aggregated: Dict[str, Dict[str, float]]):
        """投票融合：每个TF投给最强方向"""
        buy_votes = sell_votes = hold_votes = 0.0
        for tf, agg in tf_aggregated.items():
            w = self._tf_weights.get(tf, 0.1)
            best = max(agg, key=agg.get)
            if best == "buy":
                buy_votes += w
            elif best == "sell":
                sell_votes += w
            else:
                hold_votes += w

        total = buy_votes + sell_votes + hold_votes or 1.0
        result.posterior_buy = buy_votes / total
        result.posterior_sell = sell_votes / total
        result.posterior_hold = hold_votes / total

    def _dempster_shafer_fusion(self, result: MTFFusionResult, tf_aggregated: Dict[str, Dict[str, float]]):
        """
        Dempster-Shafer证据理论融合

        将每个时间框架视为独立的证据体（body of evidence），
        使用Dempster组合规则融合多个证据源对 {buy, sell, hold} 假设的支持度。

        步骤：
          1. 每个TF构建基本概率分配(BPA): m({buy}), m({sell}), m({hold}), m(Ω)
          2. 依次使用Dempster组合规则融合：m12(A) = Σ(B∩C=A) m1(B)*m2(C) / (1-K)
          3. 归一化得到最终mass函数

        修复：mass字典显式包含omega键，从完全无知状态开始(omega=1.0)，
              不确定性通过Dempster规则正确传播。
        """
        if not tf_aggregated:
            return

        # 框架 Ω = {buy, sell, hold}
        # 从完全无知状态开始：mass(Ω) = 1.0, mass({buy}) = mass({sell}) = mass({hold}) = 0
        mass = {"buy": 0.0, "sell": 0.0, "hold": 0.0, "omega": 1.0}

        for tf, agg in tf_aggregated.items():
            weight = self._tf_weights.get(tf, 0.1)
            # 该TF的BPA：信号强度作为mass，保留不确定性
            m_buy = agg.get("buy", 0.0) * weight
            m_sell = agg.get("sell", 0.0) * weight
            m_hold = agg.get("hold", 0.0) * weight
            # 未分配的质量 = 该TF的不确定性（omega）
            m_omega = 1.0 - (m_buy + m_sell + m_hold)
            if m_omega < 0:
                m_omega = 0.0
                total_m = m_buy + m_sell + m_hold
                if total_m > 0:
                    m_buy /= total_m
                    m_sell /= total_m
                    m_hold /= total_m

            # TF的BPA映射
            tf_mass = {"buy": m_buy, "sell": m_sell, "hold": m_hold, "omega": m_omega}

            # Dempster组合规则: mass12(A) = Σ(B∩C=A) mass1(B) * mass2(C) / (1-K)
            new_mass = {"buy": 0.0, "sell": 0.0, "hold": 0.0, "omega": 0.0}
            conflict = 0.0

            for h1 in ("buy", "sell", "hold", "omega"):
                m1_val = mass[h1]
                if m1_val == 0:
                    continue
                for h2 in ("buy", "sell", "hold", "omega"):
                    m2_val = tf_mass[h2]
                    if m2_val == 0:
                        continue
                    # 确定交集
                    if h1 == "omega":
                        intersection = h2  # Ω ∩ A = A
                    elif h2 == "omega":
                        intersection = h1  # A ∩ Ω = A
                    elif h1 == h2:
                        intersection = h1  # A ∩ A = A
                    else:
                        intersection = None  # 不相交的假设为冲突

                    prod = m1_val * m2_val
                    if intersection is None:
                        conflict += prod
                    elif intersection in new_mass:
                        new_mass[intersection] += prod

            # 归一化: 除以 (1 - K)
            if conflict >= 1.0 - 1e-10:
                # 完全冲突，保留原有mass（忽略此TF）
                continue
            norm = 1.0 - conflict
            if norm > 0:
                for k in new_mass:
                    new_mass[k] /= norm
                mass = new_mass

        # 移除omega（不确定性），只保留buy/sell/hold的分配
        total_decision_mass = mass["buy"] + mass["sell"] + mass["hold"]
        if total_decision_mass > 0:
            result.posterior_buy = mass["buy"] / total_decision_mass
            result.posterior_sell = mass["sell"] / total_decision_mass
            result.posterior_hold = mass["hold"] / total_decision_mass
        else:
            # 全是不确定性 → 均匀先验
            result.posterior_buy = 1/3
            result.posterior_sell = 1/3
            result.posterior_hold = 1/3

        result.fusion_method = "dempster_shafer"

    def _kalman_fusion(self, result: MTFFusionResult, tf_aggregated: Dict[str, Dict[str, float]]):
        """
        卡尔曼滤波融合

        将各时间框架的信号视为对真实方向信号的带噪声观测，
        使用卡尔曼滤波器递归估计最优方向信号。

        状态空间模型：
          x_k = x_{k-1} + w_k    (状态转移，随机游走)
          z_k = x_k + v_k        (观测，TF信号)
          其中 w_k ~ N(0, Q), v_k ~ N(0, R_k)

        各时间框架的观测噪声 R_k 由TF权重决定（权重低=噪声大）
        """
        if not tf_aggregated:
            return

        # 初始化卡尔曼滤波器
        # 状态：标量方向信号 [-1, 1]，正=多头，负=空头
        x_est = 0.0          # 初始状态估计
        p_est = 1.0          # 初始估计协方差
        q = 0.001             # 过程噪声（状态不确定性随时间增加）

        # 按权重从低到高排序TF（低权重=高噪声，先观测；高权重=低噪声，后校正）
        sorted_tfs = sorted(tf_aggregated.items(), key=lambda x: self._tf_weights.get(x[0], 0.1))

        for tf, agg in sorted_tfs:
            weight = self._tf_weights.get(tf, 0.1)

            # 观测值: 将buy/sell信号转换为标量 [-1, 1]
            z = agg.get("buy", 0.0) - agg.get("sell", 0.0)
            z = np.clip(z, -1.0, 1.0)

            # 观测噪声: 权重越低，噪声越大
            r = (1.0 - weight) * 0.5 + 0.05  # R ∈ [0.05, 0.55]

            # 预测步骤
            x_pred = x_est
            p_pred = p_est + q

            # 更新步骤
            k_gain = p_pred / (p_pred + r)      # 卡尔曼增益
            x_est = x_pred + k_gain * (z - x_pred)  # 状态更新
            p_est = (1 - k_gain) * p_pred       # 协方差更新

        # 将估计的状态映射回buy/sell概率
        # sigmoid风格的映射: x_est ∈ [-1, 1] → probabilities
        x_clipped = np.clip(x_est, -1.0, 1.0)
        if x_clipped > 0:
            # 偏buy
            result.posterior_buy = 0.5 + x_clipped * 0.4  # ∈ [0.5, 0.9]
            result.posterior_sell = 0.5 - x_clipped * 0.4
        else:
            # 偏sell
            result.posterior_buy = 0.5 + x_clipped * 0.4  # ∈ [0.1, 0.5]
            result.posterior_sell = 0.5 - x_clipped * 0.4
        # 加入估计不确定性
        uncertainty = min(p_est, 0.5)
        result.posterior_buy = max(0.05, min(0.95, result.posterior_buy * (1.0 - uncertainty) + 0.33 * uncertainty))
        result.posterior_sell = max(0.05, min(0.95, result.posterior_sell * (1.0 - uncertainty) + 0.33 * uncertainty))
        # 剩余质量归为 hold（卡尔曼状态为标量方向，hold 表示无方向倾向）
        result.posterior_hold = max(0.0, 1.0 - result.posterior_buy - result.posterior_sell)

    # ═══════════════════════════════════════════════════════════
    # 核心2: 决策成本收益分析
    # ═══════════════════════════════════════════════════════════

    def analyze_cost_benefit(
        self,
        decision_id: str,
        symbol: str,
        direction: str,
        quantity: float,
        entry_price: float,
        target_price: float,
        stop_price: float,
        leverage: float = 5.0,
        expected_hold_hours: float = 4.0,
        win_probability: float = 0.5,
    ) -> CostBenefitAnalysis:
        """
        决策成本收益分析

        计算完整的经济学期望值：
          EV = P(win) * Reward - P(loss) * Risk - Total_Cost

        Returns:
            CostBenefitAnalysis: 含净期望值、盈亏平衡点等
        """
        cba = CostBenefitAnalysis(decision_id=decision_id)
        notional = quantity * entry_price
        margin = notional / leverage

        # ── 收益/风险 ──
        if direction in ("buy", "long"):
            reward_pct = (target_price - entry_price) / entry_price
            risk_pct = (entry_price - stop_price) / entry_price
        else:
            reward_pct = (entry_price - target_price) / entry_price
            risk_pct = (stop_price - entry_price) / entry_price

        reward_usdt = notional * abs(reward_pct)
        risk_usdt = notional * abs(risk_pct)

        # ── 成本计算 ──
        # 开平仓手续费
        open_fee = notional * self._taker_fee_rate
        close_fee = notional * self._taker_fee_rate
        cba.taker_fee = open_fee + close_fee

        # 滑点估计（波动率越大滑点越大）
        vol_factor = 1.0 + abs(reward_pct) * 5  # 波动大的币种滑点更大
        cba.estimated_slippage = notional * self._base_slippage_pct * vol_factor

        # 资金费率成本（预估）
        cba.funding_cost = notional * (self._funding_rate_annual / 365 / 24) * expected_hold_hours

        # 总成本
        cba.total_cost = cba.taker_fee + cba.estimated_slippage + cba.funding_cost

        # ── 期望值 ──
        loss_probability = 1.0 - win_probability
        cba.expected_profit = win_probability * reward_usdt - loss_probability * risk_usdt
        cba.net_expected_value = cba.expected_profit - cba.total_cost

        # 期望利润率
        cba.expected_profit_pct = cba.expected_profit / max(margin, 1)

        # 风险调整收益
        if risk_usdt > 0:
            cba.risk_adjusted_return = cba.net_expected_value / risk_usdt

        # 对组合夏普的边际贡献（简化）
        cba.sharpe_contribution = cba.net_expected_value / max(notional * 0.02, 1)

        # 机会成本：若不执行，资金可获无风险收益
        cba.opportunity_cost = margin * 0.03 / 365 * expected_hold_hours / 24  # 3%年化

        # ── 盈亏平衡移动 ──
        if notional > 0:
            cba.breakeven_move_pct = cba.total_cost / notional

        # ── 判断 ──
        if cba.net_expected_value > self._min_expected_value_usdt and cba.risk_adjusted_return > 0.5:
            cba.is_profitable = True
            cba.recommendation = "Execute: positive expected value"
        elif cba.net_expected_value > 0:
            cba.is_profitable = True
            cba.recommendation = "Marginal: consider reducing size"
        else:
            cba.is_profitable = False
            cba.recommendation = (
                f"Avoid: negative EV (cost={cba.total_cost:.2f}, "
                f"breakeven requires {cba.breakeven_move_pct:.4%} move)"
            )

        return cba

    # ═══════════════════════════════════════════════════════════
    # 核心3: 元决策器 — "决策是否该做决策"
    # ═══════════════════════════════════════════════════════════

    async def meta_decide(
        self,
        symbol: str,
        decision_urgency: DecisionUrgency = DecisionUrgency.NORMAL,
        market_volatility: float = 0.02,
        current_exposure_pct: float = 0.0,
    ) -> Tuple[MetaDecisionVerdict, str, Dict[str, Any]]:
        """
        元决策：判断当前是否应该进行决策

        检查维度：
          1. 决策频率是否过高（防抖）
          2. 是否处于连续亏损冷却期
          3. 市场波动率是否过高/过低
          4. 当前暴露是否已达上限
          5. 系统负载/延迟是否正常
        """
        reasons = []
        details = {}

        # ── 1. 紧急决策始终放行 ──
        if decision_urgency in (DecisionUrgency.IMMEDIATE, DecisionUrgency.HIGH):
            return MetaDecisionVerdict.PROCEED, "Emergency/high urgency decision", details

        # ── 2. 决策频率防抖 ──
        now = datetime.now()
        if self._last_decision_time:
            elapsed_ms = (now - self._last_decision_time).total_seconds() * 1000
            details["elapsed_since_last_ms"] = elapsed_ms
            if elapsed_ms < self._min_decision_interval_ms:
                return MetaDecisionVerdict.DEFER, f"Too frequent: {elapsed_ms:.0f}ms", details

        # ── 3. 分钟决策数限制 ──
        cutoff = now - timedelta(minutes=1)
        recent_count = sum(1 for t in self._decision_count_1m if t > cutoff)
        details["decisions_last_minute"] = recent_count
        if recent_count >= self._max_decisions_per_minute:
            return MetaDecisionVerdict.DEFER, f"Rate limit: {recent_count}/{self._max_decisions_per_minute}", details

        # ── 4. 连续亏损冷却 ──
        if self._consecutive_losses >= 3:
            since_last = (now - self._last_decision_time).total_seconds() * 1000 if self._last_decision_time else float('inf')
            details["consecutive_losses"] = self._consecutive_losses
            if since_last < self._cooldown_after_loss_ms:
                return MetaDecisionVerdict.DEFER, f"Loss cooldown: {self._consecutive_losses} consecutive losses", details

        # ── 5. 暴露上限检查 ──
        details["current_exposure_pct"] = current_exposure_pct
        if current_exposure_pct > 0.90:
            return MetaDecisionVerdict.ABSTAIN, f"Exposure at {current_exposure_pct:.0%}", details
        elif current_exposure_pct > 0.70:
            return MetaDecisionVerdict.REDUCE_SIZE, f"High exposure: {current_exposure_pct:.0%}", details

        # ── 6. 极端波动率 ──
        details["market_volatility"] = market_volatility
        if market_volatility > 0.08:
            return MetaDecisionVerdict.EMERGENCY_ONLY, f"Extreme volatility: {market_volatility:.1%}", details

        # ── 7. 延迟检查 ──
        avg_latency = self.get_average_latency()
        details["avg_latency_ms"] = avg_latency
        if avg_latency > 500:
            reasons.append(f"High latency: {avg_latency:.0f}ms")
            return MetaDecisionVerdict.DEFER, f"System latency high: {avg_latency:.0f}ms", details

        # ── 通过 ──
        return MetaDecisionVerdict.PROCEED, "All checks passed", details

    # ═══════════════════════════════════════════════════════════
    # 核心4: 自适应决策阈值
    # ═══════════════════════════════════════════════════════════

    def adapt_threshold(
        self,
        market_regime: str,
        recent_win_rate: float = 0.5,
        volatility_percentile: float = 0.5,
        equity_drawdown: float = 0.0,
    ) -> float:
        """
        根据市场状态和账户状态自适应调整决策置信度阈值

        高波动 → 提高阈值（更保守）
        趋势明确 → 降低阈值（更积极）
        胜率低 → 提高阈值（更保守）
        高回撤 → 降低阈值（更积极，需要回本）
        """
        base = self._base_confidence_threshold

        # 市场状态调整
        regime_adjustments = {
            "trending_up": -0.05,       # 趋势时降低阈值
            "trending_down": -0.03,
            "ranging": 0.0,
            "range_bound": 0.0,         # P5: 添加range_bound市场状态
            "high_volatility": 0.10,    # 高波动提高阈值
            "low_volatility": -0.05,
            "unknown": 0.0,
        }
        regime_adj = regime_adjustments.get(market_regime, 0.0)

        # 波动率分位数调整
        vol_adj = (volatility_percentile - 0.5) * 0.10

        # 胜率调整
        wr_adj = (0.5 - recent_win_rate) * 0.15

        # P5: 账户回撤调整 - 高回撤时降低阈值，让系统更积极地捕捉交易机会
        # 回撤>50%: 降0.10, 回撤>30%: 降0.07, 回撤>15%: 降0.04
        if equity_drawdown > 0.50:
            drawdown_adj = -0.10
        elif equity_drawdown > 0.30:
            drawdown_adj = -0.07
        elif equity_drawdown > 0.15:
            drawdown_adj = -0.04
        else:
            drawdown_adj = 0.0

        new_threshold = base + regime_adj + vol_adj + wr_adj + drawdown_adj
        new_threshold = max(self._min_confidence_threshold,
                           min(self._max_confidence_threshold, new_threshold))

        self._current_threshold = new_threshold
        return new_threshold

    def get_current_threshold(self) -> float:
        return self._current_threshold

    # ═══════════════════════════════════════════════════════════
    # 核心5: 决策上下文丰富化
    # ═══════════════════════════════════════════════════════════

    def enrich_context(
        self,
        symbol: str,
        order_book: Dict[str, Any] = None,
        market_data: Dict[str, Any] = None,
        funding_data: Dict[str, Any] = None,
        oi_data: Dict[str, Any] = None,
    ) -> DecisionContext:
        """
        丰富决策上下文，注入市场微观结构信息

        Args:
            symbol: 交易对
            order_book: 订单簿数据
            market_data: 市场数据（波动率、市场状态等）
            funding_data: 资金费率数据
            oi_data: 持仓量数据
        """
        ctx = DecisionContext(symbol=symbol)

        # ── 订单簿分析 ──
        if order_book:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])

            if bids and asks:
                best_bid = float(bids[0][0]) if bids else 0
                best_ask = float(asks[0][0]) if asks else 0
                mid = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
                if mid > 0:
                    ctx.bid_ask_spread_pct = (best_ask - best_bid) / mid

                # 订单簿不平衡度
                depth_bids = sum(float(b[1]) for b in bids[:10])
                depth_asks = sum(float(a[1]) for a in asks[:10])
                total_depth = depth_bids + depth_asks
                if total_depth > 0:
                    ctx.order_book_imbalance = (depth_bids - depth_asks) / total_depth

                ctx.depth_10_bids = depth_bids
                ctx.depth_10_asks = depth_asks

        # ── 市场数据 ──
        if market_data:
            ctx.current_volatility = market_data.get("volatility", 0.02)
            ctx.volatility_percentile = market_data.get("volatility_percentile", 0.5)
            ctx.market_regime = market_data.get("regime", "unknown")
            ctx.trend_strength = market_data.get("trend_strength", 0.0)

        # ── 资金费率数据 ──
        if funding_data:
            funding_rate = float(funding_data.get("fundingRate", funding_data.get("funding_rate", 0)) or 0)
            # 极端资金费率可能预示市场情绪
            if abs(funding_rate) > 0.001:  # 超过0.1%为极端
                logger.debug(f"{symbol}: Extreme funding rate {funding_rate:.4%}")

        # ── 持仓量(OI)分析 ──
        if oi_data:
            oi_current = float(oi_data.get("oi", oi_data.get("openInterest", 0)) or 0)
            oi_24h_change = float(oi_data.get("oiChange24h", oi_data.get("hOIChange", 0)) or 0)
            # OI剧烈变化可能预示趋势反转
            if oi_current > 0:
                oi_change_pct = oi_24h_change / oi_current
                if abs(oi_change_pct) > 0.20:
                    logger.debug(f"{symbol}: Significant OI change {oi_change_pct:.1%}")

        # ── 历史模式快查 ──
        recent_entries = self._audit_chain[-50:] if self._audit_chain else []
        symbol_entries = [e for e in recent_entries if e.symbol == symbol]
        if symbol_entries:
            ctx.similar_pattern_count = len(symbol_entries)
            ctx.pattern_success_rate = sum(1 for e in symbol_entries if e.outcome_pnl > 0) / max(ctx.similar_pattern_count, 1)

        # 缓存上下文
        self._market_contexts[symbol] = ctx
        return ctx

    # ═══════════════════════════════════════════════════════════
    # 核心6: 决策异常检测
    # ═══════════════════════════════════════════════════════════

    def detect_anomaly(
        self,
        decision: Dict[str, Any],
        recent_decisions: List[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        检测异常决策

        检测维度：
          1. 方向突变（连续同向N次后突然反向）
          2. 仓位跳跃（仓位突然变大超过正常范围）
          3. 置信度异常（异常高或异常低）
          4. 价格偏离（决策价格远离当前市价）
          5. 频率异常（短时间内大量同向决策）
          6. 夜间异常（低流动性时段的大单）
          7. 盈亏比异常（风险回报比不合理）
        """
        reasons = []

        # 网格/剥头皮为均值回归策略，连续同向开仓是其设计行为，不应判定为"追涨杀跌"异常
        strategy_name = str(decision.get("strategy_name", "") or decision.get("source", "") or "").lower()
        is_grid_like = strategy_name in ("grid", "grid_trade", "scalping")

        # ── 1. 仓位大小异常 ──
        quantity = decision.get("quantity", 0)
        if quantity > 0:
            recent_qties = [d.get("quantity", 0) for d in (recent_decisions or [])[-20:]]
            if recent_qties:
                mean_qty = sum(recent_qties) / len(recent_qties)
                std_qty = (sum((q - mean_qty) ** 2 for q in recent_qties) / len(recent_qties)) ** 0.5
                if std_qty > 0 and quantity > mean_qty + 3 * std_qty:
                    reasons.append(f"Quantity spike: {quantity} vs mean {mean_qty:.2f}±{std_qty:.2f}")

        # ── 2. 置信度异常 ──
        confidence = decision.get("confidence", 0.5)
        if confidence > 0.99:
            reasons.append(f"Suspiciously high confidence: {confidence:.3f}")
        elif confidence < 0.10 and decision.get("direction") != "hold":
            reasons.append(f"Suspiciously low confidence: {confidence:.3f}")

        # ── 3. 方向一致性检查 ──
        direction = decision.get("direction", "")
        if recent_decisions and direction not in ("hold", "") and not is_grid_like:
            recent_dirs = [d.get("direction", "") for d in recent_decisions[-5:]]
            same_dir = sum(1 for d in recent_dirs if d == direction)
            if same_dir >= 4 and len(recent_dirs) >= 4:
                symbols = [d.get("symbol") for d in recent_decisions[-4:]]
                if decision.get("symbol") in symbols:
                    reasons.append(f"Momentum chasing: {same_dir}/5 same direction")

        # ── 4. 决策频率异常 ──
        if self._last_decision_time:
            elapsed = (datetime.now() - self._last_decision_time).total_seconds()
            if elapsed < 0.05:  # 50ms内两次决策
                reasons.append(f"Flash decision: {elapsed*1000:.0f}ms apart")

        # ── 5. 夜间低流动时段的异常大单 ──
        if quantity > 0:
            now_hour = datetime.now().hour
            is_night = now_hour < 7 or now_hour >= 23
            recent_qties = [d.get("quantity", 0) for d in (recent_decisions or [])[-10:] if d.get("quantity", 0) > 0]
            if is_night and recent_qties:
                avg_night_qty = sum(recent_qties) / len(recent_qties)
                if avg_night_qty > 0 and quantity > avg_night_qty * 2.5:
                    reasons.append(f"Oversized order in low-liquidity hours: {quantity} vs avg {avg_night_qty:.2f}")

        # ── 6. 盈亏比异常 ──
        # 兼容两种键命名：标准决策字段 (take_profit/stop_loss) 与信号字段 (take_profit_price/stop_loss_price)
        tp_price = (
            decision.get("take_profit")
            or decision.get("tp_price")
            or decision.get("take_profit_price")
            or decision.get("target_price")
        )
        sl_price = (
            decision.get("stop_loss")
            or decision.get("sl_price")
            or decision.get("stop_loss_price")
        )
        entry_price = decision.get("price", decision.get("entry_price"))
        if tp_price and sl_price and entry_price and entry_price > 0:
            try:
                if direction in ("buy", "long"):
                    reward = abs(float(tp_price) - float(entry_price))
                    risk = abs(float(entry_price) - float(sl_price))
                else:
                    reward = abs(float(entry_price) - float(tp_price))
                    risk = abs(float(sl_price) - float(entry_price))
                if risk > 0 and reward / risk < 0.3:
                    reasons.append(f"Poor risk/reward ratio: {reward/risk:.2f} (RR={reward:.4f}/{risk:.4f})")
                if risk > 0 and reward / risk > 20:
                    reasons.append(f"Suspiciously high risk/reward: {reward/risk:.1f} (possible data error)")
            except (TypeError, ValueError):
                pass

        is_anomaly = len(reasons) > 0
        reason_str = "; ".join(reasons) if reasons else "Normal"

        return is_anomaly, reason_str

    # ═══════════════════════════════════════════════════════════
    # 核心7: 不可变审计链
    # ═══════════════════════════════════════════════════════════

    def record_decision_audit(self, entry: DecisionAuditEntry) -> str:
        """
        将决策记录到不可变审计链
        每个条目包含前一条的哈希，形成防篡改链
        """
        entry.prev_hash = self._last_audit_hash
        entry.entry_hash = entry.compute_hash()

        self._audit_chain.append(entry)
        self._last_audit_hash = entry.entry_hash

        # 限制链长度
        if len(self._audit_chain) > self._max_audit_entries:
            self._audit_chain = self._audit_chain[-self._max_audit_entries:]

        return entry.entry_hash

    def verify_audit_chain(self) -> Tuple[bool, int]:
        """
        验证审计链完整性
        Returns:
            (is_valid, first_invalid_index)
        """
        prev_hash = "0" * 16
        for i, entry in enumerate(self._audit_chain):
            if entry.prev_hash != prev_hash:
                return False, i
            expected = entry.compute_hash()
            if entry.entry_hash != expected:
                return False, i
            prev_hash = entry.entry_hash
        return True, -1

    def get_audit_entries(
        self,
        symbol: str = None,
        limit: int = 100,
        outcome: str = None,
    ) -> List[Dict[str, Any]]:
        """查询审计条目"""
        entries = self._audit_chain
        if symbol:
            entries = [e for e in entries if e.symbol == symbol]
        if outcome:
            entries = [e for e in entries if e.outcome == outcome]
        return [e.to_dict() for e in entries[-limit:]]

    # ═══════════════════════════════════════════════════════════
    # 核心8: 决策归因分析
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _normalize_direction(direction: str) -> str:
        """归一化方向：buy/long -> long, sell/short -> short"""
        d = str(direction or "").strip().lower()
        if d in ("buy", "long"):
            return "long"
        if d in ("sell", "short"):
            return "short"
        return d

    @staticmethod
    def _regime_direction(regime: str) -> str:
        """市场状态隐含方向"""
        mapping = {
            "trend_bullish": "long", "trending_up": "long", "bullish": "long",
            "trend_bearish": "short", "trending_down": "short", "bearish": "short",
        }
        return mapping.get(str(regime or "").strip().lower(), "")

    def attribute_decision(
        self,
        decision_result: Dict[str, Any],
        fusion_result: MTFFusionResult,
        context: DecisionContext,
    ) -> Dict[str, float]:
        """
        决策因果归因分析（方向一致性因果贡献）

        计算每个因子对最终决策方向的因果推力：
          正贡献 = 该因子推动决策方向（与最终方向一致）
          负贡献 = 该因子反对决策方向（与最终方向相反）

        相比旧的静态 bonus 表，这里用「因子方向 × 强度」的真实因果方向，
        回答"是哪个因子、以多大力、推动/反对了这个决策"。

        Returns:
            {factor_name: contribution}  # 正值=推动，负值=反对
        """
        attribution: Dict[str, float] = {}
        direction = self._normalize_direction(decision_result.get("direction", ""))

        # ── 1. 信号源因果贡献：信号方向一致性 × 信号强度 ──
        signals = decision_result.get("signals", []) or []
        for s in signals[:8]:
            src = str(s.get("source", "unknown"))
            s_dir = self._normalize_direction(s.get("direction", ""))
            s_strength = float(s.get("strength", s.get("confidence", 0.0)) or 0.0)
            if not direction or not s_dir or s_strength <= 0:
                continue
            contrib = s_strength if s_dir == direction else -s_strength
            attribution[f"signal_{src}"] = round(contrib, 4)

        # ── 2. 市场状态因果贡献：regime 隐含方向 vs 决策方向 ──
        regime_dir = self._regime_direction(context.market_regime)
        if direction and regime_dir:
            regime_strength = abs(context.trend_strength) if abs(context.trend_strength) > 0 else 0.3
            contrib = regime_strength if regime_dir == direction else -regime_strength
            attribution["market_regime"] = round(contrib, 4)
        else:
            attribution["market_regime"] = 0.0

        # ── 3. 订单簿失衡因果贡献：买卖压力方向 vs 决策方向 ──
        imb = float(context.order_book_imbalance or 0.0)
        if direction and abs(imb) > 0.05:
            imb_dir = "long" if imb > 0 else "short"
            contrib = abs(imb) if imb_dir == direction else -abs(imb)
            attribution["order_book_imbalance"] = round(contrib, 4)

        return attribution

    def build_and_record_audit_entry(
        self,
        decision_id: str,
        symbol: str,
        strategy_name: str,
        direction: str,
        decision_type: str = "",
        confidence: float = 0.5,
        source_signals: List[Dict[str, Any]] = None,
        context: DecisionContext = None,
        cost_benefit: CostBenefitAnalysis = None,
        meta_verdict: str = "",
        is_anomaly: bool = False,
        anomaly_reason: str = "",
    ) -> str:
        """
        构建并写入不可变决策因果链条目（企业级决策因果链）

        将决策全生命周期的因果要素（信号源、上下文、成本收益、元决策、异常、归因）
        统一打包为一个防篡改条目写入哈希链，支持后续审计回溯与根因分析。

        Returns:
            entry_hash
        """
        ctx = context or DecisionContext(symbol=symbol)
        src_signals = source_signals or [{
            "source": strategy_name,
            "direction": direction,
            "strength": confidence,
            "confidence": confidence,
        }]

        # 因果归因：各因子对最终决策方向的因果推力（正=推动，负=反对）
        attribution = self.attribute_decision(
            {"direction": direction, "signals": src_signals},
            MTFFusionResult(),
            ctx,
        )

        entry = DecisionAuditEntry(
            decision_id=decision_id,
            timestamp=datetime.now().isoformat(),
            decision_type=decision_type or strategy_name,
            symbol=symbol,
            direction=direction,
            source_signals=src_signals,
            fusion_result={},
            cost_benefit=asdict(cost_benefit) if cost_benefit else {},
            meta_verdict=meta_verdict,
            context=asdict(ctx),
            attribution=attribution,
            is_anomaly=is_anomaly,
            anomaly_reason=anomaly_reason,
        )
        return self.record_decision_audit(entry)

    # ═══════════════════════════════════════════════════════════
    # 决策执行追踪
    # ═══════════════════════════════════════════════════════════

    def record_decision_execution(
        self,
        decision_id: str,
        latency_ms: float,
        success: bool,
    ):
        """记录决策执行结果"""
        self._decision_history.append({
            "decision_id": decision_id,
            "timestamp": datetime.now().isoformat(),
            "latency_ms": latency_ms,
            "success": success,
        })
        self._recent_latencies.append(latency_ms)
        self._decision_count_1m.append(datetime.now())
        self._last_decision_time = datetime.now()

    def record_decision_outcome(self, pnl: float, was_win: bool, decision_id: str = None):
        """
        记录决策盈亏结果，并回写结果到不可变因果链（结果闭环）

        结果字段（outcome / outcome_pnl）不纳入哈希，因此回写不会破坏链完整性。
        """
        if was_win:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            self._consecutive_wins = 0

        # 回写结果到审计链（因果闭环：决策 -> 结果可回溯）
        if decision_id:
            for entry in reversed(self._audit_chain):
                if entry.decision_id == decision_id:
                    entry.outcome = "win" if was_win else "loss"
                    entry.outcome_pnl = pnl
                    entry.executed = True
                    break

    def get_average_latency(self) -> float:
        """获取平均决策延迟(ms)"""
        if not self._recent_latencies:
            return 0.0
        return sum(self._recent_latencies) / len(self._recent_latencies)

    def get_latency_percentile(self, p: float = 95) -> float:
        """获取决策延迟分位数"""
        if not self._recent_latencies:
            return 0.0
        sorted_lat = sorted(self._recent_latencies)
        idx = int(len(sorted_lat) * p / 100)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    # ═══════════════════════════════════════════════════════════
    # 审计持久化
    # ═══════════════════════════════════════════════════════════

    def persist_audit_chain(self) -> bool:
        """持久化审计链到文件"""
        if not self._audit_persist_path:
            return False
        try:
            os.makedirs(os.path.dirname(self._audit_persist_path), exist_ok=True)
            data = {
                "last_hash": self._last_audit_hash,
                "entries": [e.to_dict() for e in self._audit_chain],
                "updated_at": datetime.now().isoformat(),
            }
            with open(self._audit_persist_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.warning(f"Failed to persist audit chain: {e}")
            return False

    def load_audit_chain(self) -> bool:
        """加载审计链"""
        if not self._audit_persist_path or not os.path.exists(self._audit_persist_path):
            return False
        try:
            with open(self._audit_persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._last_audit_hash = data.get("last_hash", "0" * 16)
            self._audit_chain = []
            for e_dict in data.get("entries", [])[-self._max_audit_entries:]:
                entry = DecisionAuditEntry(
                    decision_id=e_dict.get("decision_id", ""),
                    timestamp=e_dict.get("timestamp", ""),
                    decision_type=e_dict.get("decision_type", ""),
                    symbol=e_dict.get("symbol", ""),
                    direction=e_dict.get("direction", ""),
                    source_signals=e_dict.get("source_signals", []),
                    fusion_result=e_dict.get("fusion_result", {}),
                    cost_benefit=e_dict.get("cost_benefit", {}),
                    meta_verdict=e_dict.get("meta_verdict", ""),
                    context=e_dict.get("context", {}),
                    executed=e_dict.get("executed", False),
                    execution_latency_ms=e_dict.get("execution_latency_ms", 0),
                    outcome=e_dict.get("outcome", "pending"),
                    outcome_pnl=e_dict.get("outcome_pnl", 0),
                    attribution=e_dict.get("attribution", {}),
                    prev_hash=e_dict.get("prev_hash", ""),
                    entry_hash=e_dict.get("entry_hash", ""),
                    is_anomaly=e_dict.get("is_anomaly", False),
                    anomaly_reason=e_dict.get("anomaly_reason", ""),
                )
                self._audit_chain.append(entry)
            logger.info(f"Loaded {len(self._audit_chain)} audit entries (chain integrity: TBD)")
            return True
        except Exception as e:
            logger.warning(f"Failed to load audit chain: {e}")
            return False

    # ═══════════════════════════════════════════════════════════
    # 查询接口
    # ═══════════════════════════════════════════════════════════

    def get_stats(self) -> Dict[str, Any]:
        """获取决策引擎统计"""
        avg_latency = self.get_average_latency()
        p95_latency = self.get_latency_percentile(95)
        p99_latency = self.get_latency_percentile(99)

        recent = [d for d in self._decision_history
                 if (datetime.now() - datetime.fromisoformat(d["timestamp"])).total_seconds() < 3600]
        success_rate = sum(1 for d in recent if d["success"]) / max(len(recent), 1)

        return {
            "total_decisions": len(self._decision_history),
            "decisions_last_hour": len(recent),
            "success_rate_1h": round(success_rate, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "p99_latency_ms": round(p99_latency, 2),
            "current_threshold": round(self._current_threshold, 4),
            "consecutive_losses": self._consecutive_losses,
            "consecutive_wins": self._consecutive_wins,
            "audit_chain_length": len(self._audit_chain),
            "audit_chain_valid": self.verify_audit_chain()[0],
            "fusion_method": self._fusion_method.value,
        }

    def get_recent_decisions(self, limit: int = 20) -> List[Dict[str, Any]]:
        return list(self._decision_history)[-limit:]

    def get_consecutive_stats(self) -> Dict[str, int]:
        return {
            "consecutive_wins": self._consecutive_wins,
            "consecutive_losses": self._consecutive_losses,
        }


# ═══════════════════════════════════════════════════════════════
# 单例工厂
# ═══════════════════════════════════════════════════════════════

_intelligent_decision_engine: Optional[IntelligentDecisionEngine] = None


def get_intelligent_decision_engine(config: Dict[str, Any] = None) -> IntelligentDecisionEngine:
    global _intelligent_decision_engine
    if _intelligent_decision_engine is None and config is not None:
        _intelligent_decision_engine = IntelligentDecisionEngine(config)
    return _intelligent_decision_engine


def reset_intelligent_decision_engine():
    global _intelligent_decision_engine
    _intelligent_decision_engine = None


__all__ = [
    "IntelligentDecisionEngine",
    "DecisionAction",
    "DecisionUrgency",
    "MetaDecisionVerdict",
    "FusionMethod",
    "TimeFrameSignal",
    "MTFFusionResult",
    "CostBenefitAnalysis",
    "DecisionContext",
    "DecisionAuditEntry",
    "get_intelligent_decision_engine",
    "reset_intelligent_decision_engine",
]
