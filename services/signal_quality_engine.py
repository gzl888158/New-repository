"""
信号质量评分与解释系统
负责对交易信号进行多因子综合评估，输出评分和可解释的决策依据。
遵循经验回顾中的核心教训：建立可解释计算链，确保每个决策都有明确依据。
"""
import asyncio
import numpy as np
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger


class SignalQuality(Enum):
    """信号质量等级"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    REJECTED = "rejected"


class SignalFactor(Enum):
    """信号因子类型"""
    TREND_ALIGNMENT = "trend_alignment"
    VOLATILITY_ADAPTATION = "volatility_adaptation"
    LIQUIDITY_RISK = "liquidity_risk"
    FUNDING_IMPACT = "funding_impact"
    TIMING_QUALITY = "timing_quality"
    CONFIDENCE_CONSISTENCY = "confidence_consistency"
    DIVERSIFICATION_BENEFIT = "diversification_benefit"
    REGIME_COMPATIBILITY = "regime_compatibility"
    VOLUME_CONFIRMATION = "volume_confirmation"
    RSI_CONDITION = "rsi_condition"
    MACD_ALIGNMENT = "macd_alignment"
    BOLLINGER_BAND = "bollinger_band"
    ATR_QUALITY = "atr_quality"
    CORRELATION_RISK = "correlation_risk"
    ACCOUNT_HEALTH = "account_health"
    DIVERGENCE_DETECTION = "divergence_detection"
    MARKET_STRUCTURE = "market_structure"
    VOLUME_PROFILE = "volume_profile"
    ORDER_BOOK_DEPTH = "order_book_depth"


class SignalQualityEngine:
    """信号质量评分与解释引擎"""

    def __init__(self, config: Dict[str, Any], regime_engine=None, trade_journal=None, account_manager=None):
        self.config = config
        self._regime_engine = regime_engine
        self._trade_journal = trade_journal
        self._account_manager = account_manager
        
        self._factor_weights: Dict[str, float] = {
            "trend_alignment": 0.11,
            "volatility_adaptation": 0.07,
            "liquidity_risk": 0.07,
            "funding_impact": 0.06,
            "timing_quality": 0.07,
            "confidence_consistency": 0.05,
            "diversification_benefit": 0.05,
            "regime_compatibility": 0.05,
            "volume_confirmation": 0.05,
            "rsi_condition": 0.05,
            "macd_alignment": 0.04,
            "bollinger_band": 0.04,
            "atr_quality": 0.04,
            "correlation_risk": 0.04,
            "account_health": 0.03,
            "divergence_detection": 0.07,
            "market_structure": 0.06,
            "volume_profile": 0.05,
            "order_book_depth": 0.04,
        }
        
        self._quality_thresholds = {
            SignalQuality.EXCELLENT: 0.85,
            SignalQuality.GOOD: 0.70,
            SignalQuality.FAIR: 0.50,
            SignalQuality.POOR: 0.30,
            SignalQuality.REJECTED: 0.0,
        }
        
        self._score_history: List[Dict[str, Any]] = []
        self._recent_signals: Dict[str, List[Dict[str, Any]]] = {}
        self._factor_performance: Dict[str, List[float]] = {}
        self._trade_results: Dict[str, float] = {}  # signal_id -> pnl
        self._adaptive_weight_period = 100
        self._min_weight = 0.02
        self._max_weight = 0.25
        
        logger.info("SignalQualityEngine initialized with 15 evaluation factors")

    def set_regime_engine(self, engine):
        """注入MarketRegimeEngine"""
        self._regime_engine = engine

    def set_trade_journal(self, journal):
        """注入TradeJournal"""
        self._trade_journal = journal

    def set_account_manager(self, manager):
        """注入AccountManager"""
        self._account_manager = manager

    def _resolve_strategy_weights(self, strategy_name: str) -> Dict[str, float]:
        """按策略类型调整权重，使质量评分不再纯通用，而是绑定到策略语义。"""
        strategy = (strategy_name or "").lower()
        base = dict(self._factor_weights)

        profiles = {
            "trend": {
                "trend_alignment": 1.5,
                "regime_compatibility": 1.4,
                "divergence_detection": 1.3,
                "market_structure": 1.2,
                "rsi_condition": 1.1,
            },
            "arbitrage": {
                "market_structure": 1.4,
                "timing_quality": 1.3,
                "liquidity_risk": 1.3,
                "volume_confirmation": 1.2,
                "order_book_depth": 1.2,
            },
            "grid": {
                "rsi_condition": 1.5,
                "bollinger_band": 1.5,
                "volatility_adaptation": 1.4,
                "liquidity_risk": 1.2,
                "atr_quality": 1.2,
            },
            "scalping": {
                "timing_quality": 1.5,
                "volatility_adaptation": 1.4,
                "liquidity_risk": 1.3,
                "order_book_depth": 1.3,
                "rsi_condition": 1.1,
            },
            "spot_grid": {
                "rsi_condition": 1.5,
                "bollinger_band": 1.4,
                "volatility_adaptation": 1.4,
                "liquidity_risk": 1.2,
                "confidence_consistency": 1.1,
            },
            "spot_martingale": {
                "account_health": 1.5,
                "liquidity_risk": 1.4,
                "timing_quality": 1.3,
                "funding_impact": 1.2,
                "correlation_risk": 1.2,
            },
            "breakout": {
                "trend_alignment": 1.6,
                "market_structure": 1.6,
                "volume_confirmation": 1.5,
                "timing_quality": 1.3,
                "regime_compatibility": 1.2,
            },
            "reversal": {
                "divergence_detection": 1.7,
                "rsi_condition": 1.5,
                "macd_alignment": 1.4,
                "bollinger_band": 1.3,
                "market_structure": 1.2,
            },
        }
        selected = profiles.get(strategy, {})
        adjusted = {}
        for factor, weight in base.items():
            multiplier = selected.get(factor, 1.0)
            adjusted[factor] = max(0.01, weight * multiplier)
        return adjusted

    def evaluate_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估信号质量，返回评分和解释"""
        try:
            factor_scores = self._calculate_factor_scores(signal_data)
            strategy_weights = self._resolve_strategy_weights(signal_data.get("strategy_name", ""))

            self._update_adaptive_weights(factor_scores, signal_data)

            overall_score = self._compute_overall_score(factor_scores, strategy_weights)

            quality = self._map_score_to_quality(overall_score)

            explanation = self._generate_explanation(signal_data, factor_scores, overall_score)

            breakdown = {
                "overall_score": overall_score,
                "quality": quality.value,
                "factor_scores": factor_scores,
                "explanation": explanation,
                "timestamp": datetime.now().isoformat(),
                "signal_id": signal_data.get("signal_id", ""),
                "strategy": signal_data.get("strategy_name", ""),
                "symbol": signal_data.get("symbol", ""),
                "side": signal_data.get("side", ""),
                "factor_weights": strategy_weights,
                "strategy_profile": (signal_data.get("strategy_name", "") or "").lower(),
            }

            self._last_signal_confidence = signal_data.get("confidence", 0.5)
            self._record_score(breakdown)
            return breakdown
        except Exception as e:
            logger.error(f"Signal quality evaluation failed: {e}, returning default FAIR rating")
            return {
                "overall_score": 0.5,
                "quality": SignalQuality.FAIR.value,
                "factor_scores": {},
                "explanation": f"Evaluation error: {str(e)[:100]}",
                "timestamp": datetime.now().isoformat(),
                "signal_id": signal_data.get("signal_id", ""),
                "strategy": signal_data.get("strategy_name", ""),
                "symbol": signal_data.get("symbol", ""),
                "side": signal_data.get("side", ""),
                "factor_weights": dict(self._factor_weights),
            }

    def _calculate_factor_scores(self, signal_data: Dict[str, Any]) -> Dict[str, float]:
        """计算各因子得分（0-1）"""
        scores = {}
        
        scores["trend_alignment"] = self._score_trend_alignment(signal_data)
        scores["volatility_adaptation"] = self._score_volatility_adaptation(signal_data)
        scores["liquidity_risk"] = self._score_liquidity_risk(signal_data)
        scores["funding_impact"] = self._score_funding_impact(signal_data)
        scores["timing_quality"] = self._score_timing_quality(signal_data)
        scores["confidence_consistency"] = self._score_confidence_consistency(signal_data)
        scores["diversification_benefit"] = self._score_diversification_benefit(signal_data)
        scores["regime_compatibility"] = self._score_regime_compatibility(signal_data)
        scores["volume_confirmation"] = self._score_volume_confirmation(signal_data)
        scores["rsi_condition"] = self._score_rsi_condition(signal_data)
        scores["macd_alignment"] = self._score_macd_alignment(signal_data)
        scores["bollinger_band"] = self._score_bollinger_band(signal_data)
        scores["atr_quality"] = self._score_atr_quality(signal_data)
        scores["correlation_risk"] = self._score_correlation_risk(signal_data)
        scores["account_health"] = self._score_account_health(signal_data)
        scores["divergence_detection"] = self._score_divergence_detection(signal_data)
        scores["market_structure"] = self._score_market_structure(signal_data)
        scores["volume_profile"] = self._score_volume_profile(signal_data)
        scores["order_book_depth"] = self._score_order_book_depth(signal_data)
        
        return scores

    def _score_trend_alignment(self, signal_data: Dict[str, Any]) -> float:
        """趋势对齐得分：信号方向与大趋势的一致性"""
        signal_side = str(signal_data.get("direction") or signal_data.get("side", "")).lower()
        timeframe = signal_data.get("timeframe", "1H")
        
        if not self._regime_engine:
            return 0.5
        
        regime = self._regime_engine.get_regime()
        regime_name = str(regime.get("regime", "unknown")).lower()
        strength = max(0.0, min(1.0, float(regime.get("strength", 0.0) or 0.0)))
        bullish_side = signal_side in ("buy", "long")
        bearish_side = signal_side in ("sell", "short")

        if regime_name == "breakout":
            return min(1.0, 0.7 + strength * 0.3) if bullish_side else (
                max(0.1, 0.3 - strength * 0.2) if bearish_side else 0.5
            )
        if regime_name == "breakdown":
            return min(1.0, 0.7 + strength * 0.3) if bearish_side else (
                max(0.1, 0.3 - strength * 0.2) if bullish_side else 0.5
            )
        if regime_name == "reversal":
            return 0.35
        
        if regime_name == "trend_bullish":
            if bullish_side:
                return min(1.0, 0.7 + strength * 0.3)
            else:
                return max(0.1, 0.3 - strength * 0.2)
        elif regime_name == "trend_bearish":
            if bearish_side:
                return min(1.0, 0.7 + strength * 0.3)
            else:
                return max(0.1, 0.3 - strength * 0.2)
        else:
            return 0.5

    def _score_volatility_adaptation(self, signal_data: Dict[str, Any]) -> float:
        """波动率适配得分：仓位大小是否适应当前波动水平"""
        if not self._regime_engine:
            return 0.5
        
        regime = self._regime_engine.get_regime()
        volatility_score = regime.get("factor_scores", {}).get("volatility", 0)
        position_size = signal_data.get("quantity", 0) or signal_data.get("margin", 0)
        
        if volatility_score > 0.5:
            if position_size > 0:
                if position_size < 0.05:
                    return 0.8
                elif position_size < 0.1:
                    return 0.6
                else:
                    return max(0.2, 0.5 - position_size * 2)
            return 0.5
        elif volatility_score < -0.3:
            if position_size > 0 and position_size > 0.05:
                return 0.7
            return 0.5
        else:
            return 0.6

    def _score_liquidity_risk(self, signal_data: Dict[str, Any]) -> float:
        """流动性风险得分：滑点风险评估"""
        if not self._regime_engine:
            return 0.5
        
        regime = self._regime_engine.get_regime()
        liquidity_score = regime.get("factor_scores", {}).get("liquidity", 0)
        symbol = signal_data.get("symbol", "")
        
        if liquidity_score < -0.4:
            return max(0.1, 0.6 + liquidity_score)
        elif liquidity_score > 0.5:
            return min(1.0, 0.8 + liquidity_score * 0.2)
        else:
            return 0.7

    def _score_funding_impact(self, signal_data: Dict[str, Any]) -> float:
        """资金费率影响得分：持仓成本评估"""
        if not self._regime_engine:
            return 0.5
        
        regime = self._regime_engine.get_regime()
        funding_score = regime.get("factor_scores", {}).get("funding_rate", 0)
        signal_side = signal_data.get("side", "").lower()
        
        if abs(funding_score) > 0.4:
            if (funding_score > 0 and signal_side in ("sell", "short")) or \
               (funding_score < 0 and signal_side in ("buy", "long")):
                return min(1.0, 0.8 + abs(funding_score) * 0.2)
            else:
                return max(0.2, 0.5 - abs(funding_score) * 0.3)
        else:
            return 0.7

    def _score_timing_quality(self, signal_data: Dict[str, Any]) -> float:
        """时机质量得分：信号发出时机评估"""
        confidence = signal_data.get("confidence", 0.5)
        timeframe = signal_data.get("timeframe", "1H")
        is_signal_confirmed = signal_data.get("confirmed", False)
        
        score = confidence
        
        if is_signal_confirmed:
            score += 0.1
        
        if timeframe in ("15m", "30m"):
            score *= 0.85
        elif timeframe in ("4H", "1D"):
            score *= 1.05
        
        return min(1.0, max(0.1, score))

    def _score_confidence_consistency(self, signal_data: Dict[str, Any]) -> float:
        """置信度一致性得分：与策略历史置信度的一致性"""
        strategy_name = signal_data.get("strategy_name", "")
        confidence = signal_data.get("confidence", 0.5)
        
        if strategy_name not in self._recent_signals or not self._recent_signals[strategy_name]:
            return 0.5
        
        recent_confidences = [s.get("confidence", 0.5) for s in self._recent_signals[strategy_name][-10:]]
        
        if not recent_confidences:
            return 0.5
        
        avg_confidence = np.mean(recent_confidences)
        std_confidence = np.std(recent_confidences)
        
        deviation = abs(confidence - avg_confidence)
        
        if deviation < std_confidence * 0.5:
            return 0.8
        elif deviation < std_confidence * 1.0:
            return 0.6
        else:
            return max(0.2, 0.5 - deviation * 0.5)

    def _score_diversification_benefit(self, signal_data: Dict[str, Any]) -> float:
        """分散化收益得分：是否增加投资组合多样性"""
        if not self._trade_journal:
            return 0.5
        
        symbol = signal_data.get("symbol", "")
        signal_side = signal_data.get("side", "").lower()
        
        open_positions = self._trade_journal._open_positions

        if not open_positions:
            return 0.8

        # _open_positions 是 Dict[str, PositionSnapshot]，遍历 values() 获取持仓对象
        existing_symbols = set(open_positions.keys())
        existing_sides = set()
        for pos in open_positions.values():
            if pos is None:
                continue
            try:
                if hasattr(pos, 'side'):
                    existing_sides.add(pos.side)
            except Exception:
                continue
        
        if symbol not in existing_symbols:
            return 0.8
        elif signal_side not in existing_sides:
            return 0.6
        else:
            return 0.3

    def _score_regime_compatibility(self, signal_data: Dict[str, Any]) -> float:
        """市场状态兼容性得分"""
        if not self._regime_engine:
            return 0.5
        
        strategy_name = signal_data.get("strategy_name", "")
        recommendation = self._regime_engine.get_strategy_recommendation(strategy_name)
        
        factor = recommendation.get("adjustment_factor", 1.0)
        
        if factor > 1.0:
            return min(1.0, 0.7 + (factor - 1.0))
        elif factor < 0.9:
            return max(0.2, 0.6 - (0.9 - factor))
        else:
            return 0.6

    def _compute_overall_score(
        self,
        factor_scores: Dict[str, float],
        strategy_weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """计算综合得分。默认使用通用权重；若指定 strategy_weights，则按策略类型裁剪。"""
        weights = strategy_weights or self._factor_weights
        total_weight = sum(weights.values())
        overall = 0.0

        for factor, score in factor_scores.items():
            weight = weights.get(factor, 0.0)
            overall += score * weight / total_weight if total_weight > 0 else 0.0

        return round(overall, 4)

    def _map_score_to_quality(self, score: float) -> SignalQuality:
        """将得分映射到质量等级"""
        if score >= self._quality_thresholds[SignalQuality.EXCELLENT]:
            return SignalQuality.EXCELLENT
        elif score >= self._quality_thresholds[SignalQuality.GOOD]:
            return SignalQuality.GOOD
        elif score >= self._quality_thresholds[SignalQuality.FAIR]:
            return SignalQuality.FAIR
        elif score >= self._quality_thresholds[SignalQuality.POOR]:
            return SignalQuality.POOR
        else:
            return SignalQuality.REJECTED

    def _generate_explanation(self, signal_data: Dict[str, Any], 
                               factor_scores: Dict[str, float], 
                               overall_score: float) -> List[Dict[str, Any]]:
        """生成可解释的决策依据"""
        explanations = []
        
        sorted_factors = sorted(factor_scores.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)
        
        for factor, score in sorted_factors[:5]:
            reason = self._get_factor_reason(factor, score, signal_data)
            if reason:
                explanations.append({
                    "factor": factor,
                    "score": score,
                    "reason": reason,
                    "impact": "positive" if score > 0.6 else "negative" if score < 0.4 else "neutral",
                })
        
        return explanations

    def _get_factor_reason(self, factor: str, score: float, signal_data: Dict[str, Any]) -> Optional[str]:
        """获取因子得分的具体原因"""
        strategy = signal_data.get("strategy_name", "")
        side = signal_data.get("side", "")
        symbol = signal_data.get("symbol", "")
        
        if factor == "trend_alignment":
            if score > 0.7:
                return f"信号方向({side})与当前市场趋势一致"
            elif score < 0.4:
                return f"信号方向({side})与当前市场趋势相反"
        
        elif factor == "volatility_adaptation":
            if score > 0.7:
                return "仓位大小适应当前波动率水平"
            elif score < 0.4:
                return "仓位大小与当前波动率不匹配"
        
        elif factor == "liquidity_risk":
            if score > 0.7:
                return f"{symbol}流动性良好，滑点风险低"
            elif score < 0.4:
                return f"{symbol}流动性不足，注意滑点风险"
        
        elif factor == "funding_impact":
            if score > 0.7:
                return "资金费率对持仓有利"
            elif score < 0.4:
                return "资金费率不利于当前持仓方向"
        
        elif factor == "timing_quality":
            if score > 0.7:
                return "信号置信度高且已确认"
            elif score < 0.4:
                return "信号置信度较低或未确认"
        
        elif factor == "confidence_consistency":
            if score > 0.7:
                return f"{strategy}策略置信度与历史一致"
            elif score < 0.4:
                return f"{strategy}策略置信度偏离历史水平"
        
        elif factor == "diversification_benefit":
            if score > 0.7:
                return f"{symbol}为新增持仓，增加分散化"
            elif score < 0.4:
                return f"{symbol}已有类似持仓，分散化收益有限"
        
        elif factor == "regime_compatibility":
            if score > 0.7:
                return f"{strategy}策略与当前市场状态高度兼容"
            elif score < 0.4:
                return f"{strategy}策略与当前市场状态兼容性差"
        
        elif factor == "bollinger_band":
            if score > 0.7:
                return f"{symbol}价格在布林带合理位置，突破/回调信号有效"
            elif score < 0.4:
                return f"{symbol}价格在布林带极端位置，追涨杀跌风险高"
        
        elif factor == "atr_quality":
            if score > 0.7:
                return "波动率适中且止损距离合理（1-3倍ATR）"
            elif score < 0.4:
                return "波动率异常或止损距离不合理，风险控制不足"
        
        elif factor == "divergence_detection":
            if score > 0.7:
                return f"{side}信号与RSI/MACD背离方向一致，信号可靠"
            elif score < 0.4:
                return f"检测到RSI/MACD背离，{side}信号与背离方向冲突"
        
        elif factor == "market_structure":
            if score > 0.7:
                return f"市场结构({side})与信号方向一致(HH/HL或LH/LL)"
            elif score < 0.4:
                return f"市场结构与信号方向相反，逆势交易风险高"
        
        elif factor == "volume_profile":
            if score > 0.7:
                return f"价格在VWAP合理区域且成交量确认信号"
            elif score < 0.4:
                return f"价格偏离VWAP较远或成交量不支持信号方向"
        
        elif factor == "order_book_depth":
            if score > 0.7:
                return f"盘口深度充足，价差小，{side}方向有盘口支撑"
            elif score < 0.4:
                return f"盘口深度不足或价差过大，执行风险高"
        
        return None

    def _score_divergence_detection(self, signal_data: Dict[str, Any]) -> float:
        """背离检测得分：RSI和MACD背离检测，预警趋势衰竭和假突破
        - 顶背离：价格创新高但RSI/MACD走低 → 做空信号增强，做多信号减弱
        - 底背离：价格创新低但RSI/MACD走高 → 做多信号增强，做空信号减弱
        """
        metadata = signal_data.get("metadata", {})
        signal_side = signal_data.get("side", "").lower()
        
        # 从metadata获取背离标识
        rsi_divergence = metadata.get("rsi_divergence")  # "bearish"/"bullish"/None
        macd_divergence = metadata.get("macd_divergence")  # "bearish"/"bullish"/None
        price = signal_data.get("price", 0)
        
        if rsi_divergence is None and macd_divergence is None:
            # 尝试从历史价格数据检测背离
            hist_prices = metadata.get("hist_prices")
            hist_rsi = metadata.get("hist_rsi")
            hist_macd = metadata.get("hist_macd")
            
            if hist_prices and len(hist_prices) >= 20:
                rsi_divergence = self._detect_rsi_divergence(hist_prices, hist_rsi)
            if hist_prices and len(hist_prices) >= 26:
                macd_divergence = self._detect_macd_divergence(hist_prices, hist_macd)
        
        # 综合背离评分
        divergence_strength = 0.0  # 负=顶背离, 正=底背离
        count = 0
        
        if rsi_divergence == "bearish":
            divergence_strength -= 0.6
            count += 1
        elif rsi_divergence == "bullish":
            divergence_strength += 0.6
            count += 1
        
        if macd_divergence == "bearish":
            divergence_strength -= 0.8
            count += 1
        elif macd_divergence == "bullish":
            divergence_strength += 0.8
            count += 1
        
        if count == 0:
            return 0.5  # 无背离，中性
        
        # 根据信号方向评分
        if signal_side in ("buy", "long"):
            if divergence_strength > 0:
                # 底背离 + 做多 = 高质量信号
                return min(1.0, 0.6 + divergence_strength * 0.4)
            else:
                # 顶背离 + 做多 = 风险信号
                return max(0.1, 0.4 + divergence_strength * 0.3)
        elif signal_side in ("sell", "short"):
            if divergence_strength < 0:
                # 顶背离 + 做空 = 高质量信号
                return min(1.0, 0.6 - divergence_strength * 0.4)
            else:
                # 底背离 + 做空 = 风险信号
                return max(0.1, 0.4 - divergence_strength * 0.3)
        else:
            return 0.5
    
    def _detect_rsi_divergence(self, prices: list, rsi_values: list = None) -> Optional[str]:
        """检测RSI背离：比较最近两个高/低点
        
        Args:
            prices: 历史价格列表 [..., p1, p2]
            rsi_values: 历史RSI列表 [..., rsi1, rsi2]
        Returns: "bearish"(顶背离)/"bullish"(底背离)/None
        """
        try:
            if len(prices) < 20:
                return None
            
            # 取最近20根K线分前后两段
            mid = len(prices) // 2
            front_prices = prices[:mid]
            back_prices = prices[mid:]
            
            front_high = max(front_prices) if front_prices else 0
            back_high = max(back_prices) if back_prices else 0
            front_low = min(front_prices) if front_prices else float('inf')
            back_low = min(back_prices) if back_prices else float('inf')
            
            if rsi_values and len(rsi_values) >= 20:
                front_rsi = rsi_values[:mid]
                back_rsi = rsi_values[mid:]
                front_rsi_high = max(front_rsi) if front_rsi else 0
                back_rsi_high = max(back_rsi) if back_rsi else 0
                front_rsi_low = min(front_rsi) if front_rsi else 100
                back_rsi_low = min(back_rsi) if back_rsi else 100
                
                # 顶背离：价格新高但RSI走低
                if back_high > front_high and back_rsi_high < front_rsi_high:
                    return "bearish"
                # 底背离：价格新低但RSI走高
                if back_low < front_low and back_rsi_low > front_rsi_low:
                    return "bullish"
            else:
                # 无RSI数据时用简化版：价格新高但涨幅收敛
                if back_high > front_high:
                    front_range = front_high - min(front_prices)
                    back_range = back_high - min(back_prices)
                    if back_range < front_range * 0.7:
                        return "bearish"
                if back_low < front_low:
                    front_range = max(front_prices) - front_low
                    back_range = max(back_prices) - back_low
                    if back_range < front_range * 0.7:
                        return "bullish"
        except Exception:
            pass
        return None
    
    def _detect_macd_divergence(self, prices: list, macd_values: list = None) -> Optional[str]:
        """检测MACD背离"""
        try:
            if len(prices) < 26:
                return None
            
            mid = len(prices) // 2
            front_prices = prices[:mid]
            back_prices = prices[mid:]
            
            front_high = max(front_prices) if front_prices else 0
            back_high = max(back_prices) if back_prices else 0
            front_low = min(front_prices) if front_prices else float('inf')
            back_low = min(back_prices) if back_prices else float('inf')
            
            if macd_values and len(macd_values) >= 26:
                front_macd = macd_values[:mid]
                back_macd = macd_values[mid:]
                front_macd_high = max(front_macd) if front_macd else 0
                back_macd_high = max(back_macd) if back_macd else 0
                front_macd_low = min(front_macd) if front_macd else 100
                back_macd_low = min(back_macd) if back_macd else 100
                
                if back_high > front_high and back_macd_high < front_macd_high:
                    return "bearish"
                if back_low < front_low and back_macd_low > front_macd_low:
                    return "bullish"
        except Exception:
            pass
        return None
    
    def _score_market_structure(self, signal_data: Dict[str, Any]) -> float:
        """市场结构得分：HH/HL/LH/LL模式分析
        - 上升趋势：Higher Highs + Higher Lows
        - 下降趋势：Lower Highs + Lower Lows
        - 震荡：交替出现
        信号方向与市场结构一致性越高，得分越高
        """
        metadata = signal_data.get("metadata", {})
        signal_side = signal_data.get("side", "").lower()
        
        structure = metadata.get("market_structure", "")
        swing_highs = metadata.get("swing_highs", [])
        swing_lows = metadata.get("swing_lows", [])
        price = signal_data.get("price", 0)
        
        # 如果没有预设结构，尝试分析
        if not structure and swing_highs and len(swing_highs) >= 3:
            structure = self._analyze_market_structure(swing_highs, swing_lows)
        elif not structure and price > 0:
            hist_prices = metadata.get("hist_prices", [])
            if hist_prices and len(hist_prices) >= 30:
                highs, lows = self._find_swing_points(hist_prices)
                structure = self._analyze_market_structure(highs, lows)
        
        if not structure:
            return 0.5
        
        if signal_side in ("buy", "long"):
            if structure == "uptrend":
                return 0.85  # 上升趋势做多
            elif structure == "strong_uptrend":
                return 0.95
            elif structure == "downtrend":
                return 0.2   # 下降趋势做多 = 逆势
            elif structure == "strong_downtrend":
                return 0.1
            else:
                return 0.5   # 震荡
        elif signal_side in ("sell", "short"):
            if structure == "downtrend":
                return 0.85  # 下降趋势做空
            elif structure == "strong_downtrend":
                return 0.95
            elif structure == "uptrend":
                return 0.2   # 上升趋势做空 = 逆势
            elif structure == "strong_uptrend":
                return 0.1
            else:
                return 0.5
        else:
            return 0.5
    
    def _find_swing_points(self, prices: list, window: int = 5) -> tuple:
        """寻找摆动高点和低点
        
        Args:
            prices: 价格序列
            window: 摆动点窗口（左右各window根K线）
        Returns: (swing_highs, swing_lows)
        """
        try:
            highs = []
            lows = []
            n = len(prices)
            
            for i in range(window, n - window):
                # 检查是否是局部高点
                is_high = all(prices[i] >= prices[j] for j in range(i - window, i + window + 1) if j != i)
                # 检查是否是局部低点
                is_low = all(prices[i] <= prices[j] for j in range(i - window, i + window + 1) if j != i)
                
                if is_high:
                    highs.append(prices[i])
                if is_low:
                    lows.append(prices[i])
            
            return highs, lows
        except Exception:
            return [], []
    
    def _analyze_market_structure(self, swing_highs: list, swing_lows: list) -> str:
        """分析市场结构：基于摆动高低点的趋势判断
        
        Returns: strong_uptrend/uptrend/ranging/downtrend/strong_downtrend
        """
        try:
            if len(swing_highs) < 3 or len(swing_lows) < 3:
                return "ranging"
            
            # 取最近3个摆动高点和低点
            recent_highs = swing_highs[-3:]
            recent_lows = swing_lows[-3:]
            
            hh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] > recent_highs[i-1])
            hl_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] > recent_lows[i-1])
            lh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] < recent_highs[i-1])
            ll_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] < recent_lows[i-1])
            
            total = len(recent_highs) - 1
            
            if hh_count == total and hl_count == total:
                return "strong_uptrend"
            elif hh_count >= total * 0.7 and hl_count >= total * 0.7:
                return "uptrend"
            elif lh_count == total and ll_count == total:
                return "strong_downtrend"
            elif lh_count >= total * 0.7 and ll_count >= total * 0.7:
                return "downtrend"
            else:
                return "ranging"
        except Exception:
            return "ranging"
    
    def _score_volume_profile(self, signal_data: Dict[str, Any]) -> float:
        """成交量画像得分：基于VWAP和成交量分布评估信号质量
        - 价格在POC(Point of Control)附近：高流动性区域，信号更可靠
        - 价格远离VWAP：可能过度延伸，回归概率高
        - 放量突破价值区域：确认信号强度
        """
        metadata = signal_data.get("metadata", {})
        signal_side = signal_data.get("side", "").lower()
        price = signal_data.get("price", 0)
        
        vwap = metadata.get("vwap", 0)
        vwap_upper = metadata.get("vwap_upper", 0)
        vwap_lower = metadata.get("vwap_lower", 0)
        poc_price = metadata.get("poc_price", 0)
        volume_ratio = metadata.get("volume_ratio", 1.0)
        
        if price <= 0:
            return 0.5
        
        score = 0.5
        
        # VWAP偏差评估
        if vwap > 0:
            vwap_dev = (price - vwap) / vwap
            
            if signal_side in ("buy", "long"):
                if -0.02 <= vwap_dev <= 0.01:
                    # 略低于或接近VWAP，好的买入区
                    score += 0.2
                elif vwap_dev < -0.05:
                    # 大幅低于VWAP，可能超卖
                    score += 0.1
                elif vwap_dev > 0.03:
                    # 大幅高于VWAP，追高风险
                    score -= 0.15
            elif signal_side in ("sell", "short"):
                if -0.01 <= vwap_dev <= 0.02:
                    # 略高于或接近VWAP，好的做空区
                    score += 0.2
                elif vwap_dev > 0.05:
                    # 大幅高于VWAP，可能超买
                    score += 0.1
                elif vwap_dev < -0.03:
                    # 大幅低于VWAP，追空风险
                    score -= 0.15
        
        # POC（成交量控制点）评估
        if poc_price > 0:
            poc_dev = abs(price - poc_price) / poc_price
            if poc_dev < 0.01:
                # 价格在POC附近，高流动性区域
                score += 0.1
                if volume_ratio > 1.5:
                    # POC附近放量，确认信号
                    score += 0.1
        
        # 成交量确认
        if volume_ratio > 2.0:
            score += 0.1
        elif volume_ratio < 0.5:
            score -= 0.1
        
        return max(0.1, min(1.0, score))
    
    def _score_order_book_depth(self, signal_data: Dict[str, Any]) -> float:
        """订单簿深度得分：基于买卖盘口深度和价差评估
        - 价差小：流动性好
        - 深度大：滑点风险低
        - 盘口失衡：预示短期方向
        """
        metadata = signal_data.get("metadata", {})
        signal_side = signal_data.get("side", "").lower()
        
        spread_pct = metadata.get("spread_pct")
        bid_depth = metadata.get("bid_depth")
        ask_depth = metadata.get("ask_depth")
        imbalance = metadata.get("order_imbalance")  # -1到1, 正=买方强势
        
        if spread_pct is None:
            return 0.5
        
        score = 0.5
        
        # 价差评估
        spread_pct = float(spread_pct)
        if spread_pct < 0.0005:
            score += 0.15  # 极低滑点
        elif spread_pct < 0.001:
            score += 0.08
        elif spread_pct > 0.005:
            score -= 0.15  # 高滑点风险
        elif spread_pct > 0.003:
            score -= 0.08
        
        # 盘口深度评估
        if bid_depth is not None and ask_depth is not None:
            bid_depth = float(bid_depth)
            ask_depth = float(ask_depth)
            min_depth = min(bid_depth, ask_depth)
            
            if min_depth > 50000:
                score += 0.1  # 深度充足
            elif min_depth < 10000:
                score -= 0.1  # 深度不足
        
        # 盘口失衡 - 确认信号方向
        if imbalance is not None:
            imbalance = float(imbalance)
            if signal_side in ("buy", "long") and imbalance > 0.2:
                score += 0.1  # 买方强势+做多
            elif signal_side in ("sell", "short") and imbalance < -0.2:
                score += 0.1  # 卖方强势+做空
            elif signal_side in ("buy", "long") and imbalance < -0.2:
                score -= 0.1  # 卖方强势+做多=逆向
            elif signal_side in ("sell", "short") and imbalance > 0.2:
                score -= 0.1  # 买方强势+做空=逆向
        
        return max(0.1, min(1.0, score))

    def _record_score(self, breakdown: Dict[str, Any]):
        """记录评分历史"""
        self._score_history.append(breakdown)
        
        if len(self._score_history) > 500:
            self._score_history = self._score_history[-500:]
        
        strategy = breakdown.get("strategy", "")
        if strategy not in self._recent_signals:
            self._recent_signals[strategy] = []
        
        # P0: 修复 confidence 键名 —— breakdown 中不存在 signal_confidence，应使用 signal_data 中的值
        self._recent_signals[strategy].append({
            "confidence": self._last_signal_confidence if strategy in self._recent_signals else 0.5,
            "score": breakdown.get("overall_score", 0.5),
            "timestamp": datetime.now(),
        })
        
        if len(self._recent_signals[strategy]) > 50:
            self._recent_signals[strategy] = self._recent_signals[strategy][-50:]

    def _score_volume_confirmation(self, signal_data: Dict[str, Any]) -> float:
        """成交量确认得分：信号是否有成交量支持"""
        metadata = signal_data.get("metadata", {})
        volume = metadata.get("volume")
        avg_volume = metadata.get("avg_volume")
        
        if not volume or not avg_volume or avg_volume == 0:
            return 0.5
        
        volume_ratio = volume / avg_volume
        
        if volume_ratio > 2.0:
            return min(1.0, 0.7 + (volume_ratio - 2.0) * 0.15)
        elif volume_ratio > 1.5:
            return 0.7
        elif volume_ratio > 0.8:
            return 0.5
        else:
            return max(0.2, 0.5 - (0.8 - volume_ratio) * 0.3)

    def _score_rsi_condition(self, signal_data: Dict[str, Any]) -> float:
        """RSI条件得分：当前RSI是否支持信号方向"""
        metadata = signal_data.get("metadata", {})
        rsi = metadata.get("rsi")
        signal_side = signal_data.get("side", "").lower()
        
        if not rsi:
            return 0.5
        
        rsi = float(rsi)
        
        if signal_side in ("buy", "long"):
            if rsi < 30:
                return min(1.0, 0.7 + (30 - rsi) * 0.01)
            elif rsi < 45:
                return 0.6
            elif rsi < 60:
                return 0.5
            elif rsi < 70:
                return 0.4
            else:
                return max(0.1, 0.3 - (rsi - 70) * 0.02)
        else:
            if rsi > 70:
                return min(1.0, 0.7 + (rsi - 70) * 0.01)
            elif rsi > 55:
                return 0.6
            elif rsi > 40:
                return 0.5
            elif rsi > 30:
                return 0.4
            else:
                return max(0.1, 0.3 - (30 - rsi) * 0.02)

    def _score_macd_alignment(self, signal_data: Dict[str, Any]) -> float:
        """MACD对齐得分：MACD信号是否与交易方向一致"""
        metadata = signal_data.get("metadata", {})
        macd_line = metadata.get("macd_line")
        signal_line = metadata.get("signal_line")
        signal_side = signal_data.get("side", "").lower()
        
        if macd_line is None or signal_line is None:
            return 0.5
        
        macd_line = float(macd_line)
        signal_line = float(signal_line)
        
        macd_diff = macd_line - signal_line
        
        if signal_side in ("buy", "long"):
            if macd_diff > 0:
                return min(1.0, 0.7 + macd_diff * 0.1)
            else:
                return max(0.2, 0.5 + macd_diff * 0.3)
        else:
            if macd_diff < 0:
                return min(1.0, 0.7 - macd_diff * 0.1)
            else:
                return max(0.2, 0.5 - macd_diff * 0.3)

    def _score_bollinger_band(self, signal_data: Dict[str, Any]) -> float:
        """布林带突破得分：信号价格在布林带中的位置"""
        metadata = signal_data.get("metadata", {})
        bb_upper = metadata.get("bb_upper")
        bb_lower = metadata.get("bb_lower")
        bb_middle = metadata.get("bb_middle")
        price = signal_data.get("price", 0)
        signal_side = signal_data.get("side", "").lower()
        
        if bb_upper is None or bb_lower is None or price <= 0:
            return 0.5
        
        bb_upper = float(bb_upper)
        bb_lower = float(bb_lower)
        bb_middle = float(bb_middle) if bb_middle else (bb_upper + bb_lower) / 2
        price = float(price)
        
        band_width = bb_upper - bb_lower
        if band_width <= 0:
            return 0.5
        
        # 价格在布林带中的位置 (0=下轨, 1=上轨)
        position = (price - bb_lower) / band_width
        
        if signal_side in ("buy", "long"):
            # 做多：价格在中轨以下（回调买入）或突破上轨（突破买入）为佳
            if position < 0.2:
                # 接近下轨，超卖买入机会
                return min(1.0, 0.8 + (0.2 - position) * 0.5)
            elif position < 0.5:
                # 中轨以下，合理买入区
                return 0.7
            elif position < 0.8:
                # 中轨以上，偏高风险
                return 0.5
            elif position <= 1.0:
                # 接近上轨，追高风险
                return max(0.2, 0.4 - (position - 0.8) * 0.5)
            else:
                # 突破上轨，强势突破信号
                return min(1.0, 0.6 + (position - 1.0) * 0.3)
        else:
            # 做空：价格在中轨以上（回调做空）或突破下轨（突破做空）为佳
            if position > 0.8:
                # 接近上轨，超买卖入机会
                return min(1.0, 0.8 + (position - 0.8) * 0.5)
            elif position > 0.5:
                # 中轨以上，合理做空区
                return 0.7
            elif position > 0.2:
                # 中轨以下，偏高风险
                return 0.5
            elif position >= 0:
                # 接近下轨，追空风险
                return max(0.2, 0.4 - (0.2 - position) * 0.5)
            else:
                # 突破下轨，强势突破信号
                return min(1.0, 0.6 + (0 - position) * 0.3)

    def _score_atr_quality(self, signal_data: Dict[str, Any]) -> float:
        """ATR质量得分：基于ATR的止损合理性评估"""
        metadata = signal_data.get("metadata", {})
        atr = metadata.get("atr")
        price = signal_data.get("price", 0)
        stop_loss = signal_data.get("stop_loss")
        signal_side = signal_data.get("side", "").lower()
        
        if atr is None or price <= 0:
            return 0.5
        
        atr = float(atr)
        price = float(price)
        
        # ATR占比（波动率占价格比例）
        atr_ratio = atr / price if price > 0 else 0
        
        # 评估波动率水平是否适合交易
        if atr_ratio > 0.08:
            # 波动率过高，风险大
            base_score = max(0.2, 0.5 - (atr_ratio - 0.08) * 5)
        elif atr_ratio < 0.005:
            # 波动率过低，可能流动性不足
            base_score = max(0.3, 0.5 - (0.005 - atr_ratio) * 20)
        else:
            # 正常波动率范围
            base_score = 0.7
        
        # 如果有止损，评估止损距离是否合理
        if stop_loss and stop_loss > 0:
            stop_loss = float(stop_loss)
            sl_distance = abs(price - stop_loss) / price if price > 0 else 0
            sl_atr_ratio = abs(price - stop_loss) / atr if atr > 0 else 0
            
            # 止损距离应该在 1-3 倍 ATR 之间
            if 1.0 <= sl_atr_ratio <= 3.0:
                # 合理的止损距离
                stop_score = 0.9
            elif 0.5 <= sl_atr_ratio < 1.0:
                # 止损过近，容易被噪音触发
                stop_score = 0.5
            elif 3.0 < sl_atr_ratio <= 5.0:
                # 止损过远，风险较大
                stop_score = 0.5
            else:
                # 止损极不合理
                stop_score = 0.2
            
            # 综合波动率和止损评分
            return round(base_score * 0.4 + stop_score * 0.6, 4)
        
        return base_score

    def _score_correlation_risk(self, signal_data: Dict[str, Any]) -> float:
        """相关性风险得分：与现有持仓的相关性风险"""
        if not self._trade_journal:
            return 0.5
        
        symbol = signal_data.get("symbol", "")
        open_positions = self._trade_journal._open_positions
        
        if not open_positions:
            return 0.8
        
        high_corr_count = 0
        total_count = len(open_positions)
        
        for pos in open_positions.values():
            if pos.symbol != symbol:
                high_corr_count += 1
        
        if high_corr_count == 0:
            return 0.8
        elif high_corr_count <= total_count * 0.3:
            return 0.6
        else:
            return max(0.2, 0.5 - (high_corr_count / total_count) * 0.3)

    def _score_account_health(self, signal_data: Dict[str, Any]) -> float:
        """账户健康度得分：当前账户状态是否适合交易"""
        if not self._account_manager:
            return 0.5
        
        try:
            account_info = self._account_manager.get_account_info()
            if not account_info:
                return 0.5
            
            total_eq = float(account_info.get("totalEq", 0))
            details = account_info.get("details", [])
            avail_bal = 0.0
            for detail in details:
                if detail.get("ccy") == "USDT":
                    avail_bal = float(detail.get("availBal", 0))
                    break
            
            if total_eq > 0 and avail_bal > 0:
                avail_ratio = avail_bal / total_eq
                if avail_ratio > 0.5:
                    return 0.8
                elif avail_ratio > 0.3:
                    return 0.6
                elif avail_ratio > 0.1:
                    return 0.4
                else:
                    return 0.2
            return 0.5
        except Exception:
            return 0.5

    def record_trade_result(self, signal_id: str, pnl: float):
        """记录信号对应的实际交易盈亏结果"""
        self._trade_results[signal_id] = pnl
        # 限制缓存大小
        if len(self._trade_results) > 1000:
            oldest_keys = sorted(self._trade_results.keys())[:200]
            for k in oldest_keys:
                del self._trade_results[k]

    def _update_adaptive_weights(self, factor_scores: Dict[str, float], signal_data: Dict[str, Any]):
        """自适应权重更新：根据因子得分与实际交易盈亏结果的相关性动态调整权重"""
        if len(self._score_history) < self._adaptive_weight_period:
            return
        
        recent_scores = self._score_history[-self._adaptive_weight_period:]
        
        if not recent_scores:
            return
        
        # P1-Fix: 使用因子得分与实际交易盈亏结果的相关性，而非与综合得分的相关性
        # 提取有P&L结果的信号
        scored_signals = []
        for s in recent_scores:
            sid = s.get("signal_id", "")
            if sid in self._trade_results:
                scored_signals.append((s, self._trade_results[sid]))
        
        if len(scored_signals) < 10:
            return
        
        factor_correlations = {}
        for factor in factor_scores:
            factor_values = [s["factor_scores"].get(factor, 0.5) for s, _ in scored_signals if "factor_scores" in s]
            pnl_values = [pnl for s, pnl in scored_signals if "factor_scores" in s]
            
            if len(factor_values) >= 10:
                # 使用因子得分与PnL的相关性（绝对值），相关性越高说明因子越能预测盈亏
                correlation = np.corrcoef(factor_values, pnl_values)[0, 1]
                factor_correlations[factor] = abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                factor_correlations[factor] = 0.0
        
        total_correlation = sum(factor_correlations.values())
        if total_correlation == 0:
            return
        
        for factor, corr in factor_correlations.items():
            target_weight = corr / total_correlation
            current_weight = self._factor_weights.get(factor, 0.1)
            
            new_weight = current_weight * 0.9 + target_weight * 0.1
            new_weight = max(self._min_weight, min(self._max_weight, new_weight))
            
            self._factor_weights[factor] = round(new_weight, 4)

    def get_score_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取评分历史"""
        return self._score_history[-limit:]

    def get_strategy_quality_stats(self, strategy_name: str) -> Dict[str, Any]:
        """获取策略的质量统计"""
        strategy_scores = [s for s in self._score_history if s.get("strategy") == strategy_name]
        
        if not strategy_scores:
            return {
                "strategy": strategy_name,
                "total_signals": 0,
                "avg_score": 0.0,
                "quality_distribution": {},
            }
        
        scores = [s["overall_score"] for s in strategy_scores]
        avg_score = np.mean(scores)
        
        distribution = {}
        for quality in SignalQuality:
            count = sum(1 for s in strategy_scores if s["quality"] == quality.value)
            distribution[quality.value] = count
        
        return {
            "strategy": strategy_name,
            "total_signals": len(strategy_scores),
            "avg_score": round(avg_score, 4),
            "quality_distribution": distribution,
            "recent_scores": scores[-10:],
        }

    def is_signal_acceptable(self, signal_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """判断信号是否可接受"""
        breakdown = self.evaluate_signal(signal_data)
        
        min_quality = self.config.get("trading", {}).get("min_signal_quality", 0.30)
        
        is_acceptable = breakdown["overall_score"] >= min_quality
        
        return is_acceptable, breakdown
