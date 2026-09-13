"""
企业级智能交易记录分析智能体
============================
定位：统一编排交易记录、市场状态、策略方向、信号质量与 ADX 确认，
产出两份结构化报告：
  1. 综合分析报告（analysis report）
  2. 优化方向分析报告（optimization direction report）

设计原则：
  - 只编排与融合，不重复实现底层指标（ADX 复用 services.trend_vote，
    市场状态复用 MarketRegimeEngine，贡献度复用 ContributionAnalyzer，
    交易统计复用 HistoricalAnalyzer，参数优化复用 StrategyOptimizer，
    信号质量复用 DecisionQualityEvaluator）。
  - 所有依赖均为可选注入：缺失时优雅降级，报告相应字段标记为 unavailable，
    保证在任意运行环境下都能生成一份可用报告。
  - 关键指标统一四舍五入，避免超大 JSON 浮点噪声。
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from analysis.contribution_analyzer import ContributionAnalyzer
from analysis.historical_analyzer import HistoricalAnalyzer
from analysis.strategy_optimizer import StrategyOptimizer
from services.trend_vote import compute_trend_vote


class IntelligentAnalysisAgent:
    """企业级智能交易记录分析智能体

    依赖（全部可选，按需注入）：
      - trade_journal:          TradeJournal，交易记录数据源
      - sqlite_storage:         SQLiteStorage，原始交易记录（供贡献度分析）
      - market_regime_engine:   MarketRegimeEngine，市场状态
      - decision_quality_evaluator: DecisionQualityEvaluator，信号质量
      - okx_client:             OKX 客户端，用于拉取 K 线做 ADX 确认
    """

    # 默认监控币种（用于 ADX 确认，可被 config 覆盖）
    DEFAULT_SYMBOLS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]

    # ADX 强度分级阈值
    ADX_STRONG = 25.0
    ADX_MODERATE = 20.0

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        trade_journal=None,
        sqlite_storage=None,
        market_regime_engine=None,
        decision_quality_evaluator=None,
        okx_client=None,
        historical_analyzer: Optional[HistoricalAnalyzer] = None,
        contribution_analyzer: Optional[ContributionAnalyzer] = None,
        strategy_optimizer: Optional[StrategyOptimizer] = None,
    ):
        self.config = config or {}
        self.trade_journal = trade_journal
        self.sqlite_storage = sqlite_storage
        self.market_regime_engine = market_regime_engine
        self.decision_quality_evaluator = decision_quality_evaluator
        self.okx_client = okx_client

        # 编排依赖，缺失时按需实例化
        self.historical_analyzer = historical_analyzer
        if self.historical_analyzer is None and self.trade_journal is not None:
            self.historical_analyzer = HistoricalAnalyzer(self.trade_journal)

        self.contribution_analyzer = contribution_analyzer
        if self.contribution_analyzer is None:
            self.contribution_analyzer = ContributionAnalyzer(
                sqlite_storage=self.sqlite_storage,
                trade_journal=self.trade_journal,
                config=self.config,
            )

        self.strategy_optimizer = strategy_optimizer
        if self.strategy_optimizer is None and self.trade_journal is not None:
            self.strategy_optimizer = StrategyOptimizer(self.trade_journal, self.config)

        # ADX 确认默认币种
        self._adx_symbols: List[str] = self.config.get("market_regime", {}).get(
            "watched_symbols", self.DEFAULT_SYMBOLS
        )

        # 分析时间线：固定为首次设定的当前时刻，之前的交易不计入智能分析报告
        self._analysis_timeline_path = os.path.join("data", "analysis_timeline.json")
        self._analysis_since = self._load_analysis_timeline()

        logger.info(
            "IntelligentAnalysisAgent initialized "
            f"(regime={market_regime_engine is not None}, "
            f"dqe={decision_quality_evaluator is not None}, "
            f"okx={okx_client is not None}, "
            f"analysis_since={self._analysis_since.isoformat()})"
        )

    # ==================================================================
    # 分析时间线
    # ==================================================================

    def _load_analysis_timeline(self) -> datetime:
        """加载或创建分析时间线起点。

        时间线语义：智能分析报告只统计该时间点（含）之后平仓（exit_time/close_time）
        的交易，之前的交易不计入分析。首次设定固定为当前时刻并持久化，后续加载复用
        已持久化起点，保证重启后不丢失已累计的新交易。
        """
        try:
            if os.path.exists(self._analysis_timeline_path):
                with open(self._analysis_timeline_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                start_str = data.get("analysis_start_time")
                if start_str:
                    start = datetime.fromisoformat(start_str)
                    logger.info(f"Loaded analysis timeline start: {start.isoformat()}")
                    return start
        except Exception as e:
            logger.warning(f"Failed to load analysis timeline, will recreate: {e}")

        # 首次设定：固定为当前时刻并持久化
        start = datetime.now()
        try:
            os.makedirs(os.path.dirname(self._analysis_timeline_path), exist_ok=True)
            with open(self._analysis_timeline_path, "w", encoding="utf-8") as f:
                json.dump({
                    "analysis_start_time": start.isoformat(),
                    "created_at": start.isoformat(),
                    "filter_scope": "仅智能分析报告（不含实时风控/黑名单/策略退出/学习循环）",
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"Created analysis timeline start: {start.isoformat()}")
        except Exception as e:
            logger.warning(f"Failed to persist analysis timeline: {e}")
        return start

    def _timeline_info(self) -> Dict[str, Any]:
        """输出时间线信息，便于在报告中确认过滤生效。"""
        return {
            "analysis_start_time": self._analysis_since.isoformat(),
            "filter_scope": "仅智能分析报告（不含实时风控/黑名单/策略退出/学习循环）",
        }


    # ==================================================================
    # 1. 市场状态
    # ==================================================================

    def analyze_market_state(self) -> Dict[str, Any]:
        """聚合市场状态：主状态、因子得分、策略级仓位调整建议。"""
        if self.market_regime_engine is None:
            return {"available": False, "reason": "MarketRegimeEngine 未注入"}

        try:
            regime = self.market_regime_engine.get_regime()
            adjustment = self.market_regime_engine.get_position_adjustment()
            symbol_regimes = self.market_regime_engine.get_all_symbol_regimes()

            return {
                "available": True,
                "regime": regime.get("regime", "unknown"),
                "subtype": regime.get("subtype", "unknown"),
                "strength": self._round(regime.get("strength", 0.0)),
                "confidence": self._round(regime.get("confidence", 0.0)),
                "factor_scores": self._round_dict(regime.get("factor_scores", {})),
                "strategy_adjustment": self._round_dict(adjustment),
                "symbol_regimes": symbol_regimes,
                "last_update": regime.get("last_update"),
            }
        except Exception as e:
            logger.warning(f"analyze_market_state failed: {e}")
            return {"available": False, "reason": str(e)}

    # ==================================================================
    # 2. ADX 确认
    # ==================================================================

    async def confirm_adx(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Tuple[str, ...] = ("1H", "4H"),
    ) -> Dict[str, Any]:
        """ADX 趋势确认。

        对每个币种拉取多周期 K 线，复用 compute_trend_vote 计算
        ADX / direction / adx_strength，并输出多周期一致性与确认结论。

        okx_client 缺失或拉取失败时，回退到 MarketRegimeEngine 的
        trend 因子得分（其底层同样是 ADX 驱动的趋势判断）。
        """
        symbols = symbols or self._adx_symbols
        results: List[Dict[str, Any]] = []

        for symbol in symbols:
            entry: Dict[str, Any] = {"symbol": symbol, "available": False}
            if self.okx_client is not None:
                try:
                    tf_results = []
                    for tf in timeframes:
                        klines = await self._fetch_klines(symbol, tf)
                        if not klines:
                            continue
                        closes, highs, lows = self._extract_ohlc(klines)
                        if len(closes) < 20:
                            continue
                        vote = compute_trend_vote(closes, highs, lows)
                        tf_results.append({
                            "timeframe": tf,
                            "adx": self._round(vote["adx"]),
                            "direction": self._round(vote["direction"]),
                            "adx_strength": self._round(vote["adx_strength"]),
                            "votes": vote.get("votes", {}),
                        })

                    if tf_results:
                        entry.update(self._build_adx_entry(symbol, tf_results))
                        results.append(entry)
                        continue
                except Exception as e:
                    logger.debug(f"ADX confirm failed for {symbol}: {e}")

            # 回退：使用市场状态的 trend 因子得分
            fallback = self._adx_fallback(symbol)
            if fallback is not None:
                results.append(fallback)
            else:
                entry["reason"] = "no kline data / no regime data"
                results.append(entry)

        return {
            "available": any(r.get("available") for r in results),
            "symbols": results,
            "summary": self._summarize_adx(results),
        }

    async def _fetch_klines(self, symbol: str, timeframe: str) -> Optional[List[Any]]:
        """拉取 K 线（异步优先，回退同步）。"""
        try:
            if hasattr(self.okx_client, "get_kline_async"):
                return await self.okx_client.get_kline_async(symbol, timeframe, limit=60)
            return self.okx_client.get_kline(symbol, timeframe, limit=60)
        except Exception as e:
            logger.debug(f"fetch klines failed for {symbol} {timeframe}: {e}")
            return None

    @staticmethod
    def _extract_ohlc(klines: List[Any]) -> Tuple[List[float], List[float], List[float]]:
        """提取 OHLC，兼容 list-of-lists 与 dict 两种 K 线格式。"""
        closes: List[float] = []
        highs: List[float] = []
        lows: List[float] = []
        for k in klines:
            try:
                if isinstance(k, dict):
                    closes.append(float(k.get("close", 0)))
                    highs.append(float(k.get("high", 0)))
                    lows.append(float(k.get("low", 0)))
                else:
                    closes.append(float(k[4]))
                    highs.append(float(k[2]))
                    lows.append(float(k[3]))
            except (IndexError, TypeError, ValueError):
                continue
        return closes, highs, lows

    def _build_adx_entry(self, symbol: str, tf_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """基于多周期 ADX 结果生成单币种确认条目。"""
        primary = tf_results[0]
        adx = primary["adx"]
        direction = primary["direction"]

        # 多周期方向一致性
        directions = [t["direction"] for t in tf_results]
        positive = sum(1 for d in directions if d > 0.1)
        negative = sum(1 for d in directions if d < -0.1)
        if positive == len(directions):
            mtf_agreement = "bullish"
        elif negative == len(directions):
            mtf_agreement = "bearish"
        elif positive == 0 and negative == 0:
            mtf_agreement = "sideways"
        else:
            mtf_agreement = "divergent"

        if adx >= self.ADX_STRONG:
            confirmation = "strong_trend"
            regime_hint = "trend"
        elif adx >= self.ADX_MODERATE:
            confirmation = "moderate_trend"
            regime_hint = "transition"
        else:
            confirmation = "weak_no_trend"
            regime_hint = "range"

        return {
            "symbol": symbol,
            "available": True,
            "adx": self._round(adx),
            "direction": self._round(direction),
            "adx_strength": primary["adx_strength"],
            "confirmation": confirmation,
            "regime_hint": regime_hint,
            "mtf_agreement": mtf_agreement,
            "timeframes": tf_results,
        }

    def _adx_fallback(self, symbol: str) -> Optional[Dict[str, Any]]:
        """无 K 线时，用市场状态 trend 因子得分近似 ADX 确认。"""
        if self.market_regime_engine is None:
            return None
        try:
            sym_regime = self.market_regime_engine.get_symbol_regime(symbol)
            trend = float(sym_regime.get("trend_strength", 0.0) or 0.0)
            regime = sym_regime.get("regime", "unknown")
            abs_trend = abs(trend)
            if abs_trend > 0.5:
                confirmation, regime_hint = "strong_trend", "trend"
            elif abs_trend > 0.2:
                confirmation, regime_hint = "moderate_trend", "transition"
            else:
                confirmation, regime_hint = "weak_no_trend", "range"
            return {
                "symbol": symbol,
                "available": True,
                "adx": None,
                "direction": self._round(trend),
                "adx_strength": self._round(abs_trend),
                "confirmation": confirmation,
                "regime_hint": regime_hint,
                "mtf_agreement": "unknown",
                "source": "regime_trend_factor",
                "timeframes": [],
            }
        except Exception as e:
            logger.debug(f"ADX fallback failed for {symbol}: {e}")
            return None

    def _summarize_adx(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总 ADX 确认结论。"""
        available = [r for r in results if r.get("available")]
        if not available:
            return {"confirmed_symbols": 0, "dominant_regime_hint": "unknown"}

        strong = [r for r in available if r.get("confirmation") == "strong_trend"]
        hints = {}
        for r in available:
            hint = r.get("regime_hint", "unknown")
            hints[hint] = hints.get(hint, 0) + 1
        dominant_hint = max(hints, key=hints.get) if hints else "unknown"

        return {
            "confirmed_symbols": len(available),
            "strong_trend_symbols": len(strong),
            "dominant_regime_hint": dominant_hint,
            "regime_hint_distribution": hints,
        }

    # ==================================================================
    # 3. 策略方向
    # ==================================================================

    def analyze_strategy_direction(self) -> Dict[str, Any]:
        """聚合策略方向：历史健康度 + 市场状态适配 → 每策略方向建议。"""
        health_summary = {}
        ranking = []
        reallocations = []
        regime_adjustment = {}

        if self.contribution_analyzer is not None:
            try:
                health_summary = self.contribution_analyzer.get_health_summary(
                    since_override=self._analysis_since
                )
                ranking = self.contribution_analyzer.get_strategy_ranking(
                    "total_pnl", since_override=self._analysis_since
                )
                reallocations = self.contribution_analyzer.get_capital_reallocation_suggestions(
                    since_override=self._analysis_since
                )
            except Exception as e:
                logger.warning(f"contribution analysis failed: {e}")

        if self.market_regime_engine is not None:
            try:
                regime_adjustment = self.market_regime_engine.get_position_adjustment()
            except Exception as e:
                logger.warning(f"regime adjustment failed: {e}")

        # 以贡献度健康度为基准，叠加市场状态适配
        directions = []
        health_by_strategy = {
            h["strategy"]: h for h in health_summary.get("strategies", [])
        }
        realloc_by_strategy = {r["strategy"]: r for r in reallocations}

        for sname, health in health_by_strategy.items():
            realloc = realloc_by_strategy.get(sname, {})
            regime_factor = regime_adjustment.get(sname, regime_adjustment.get("overall", 1.0))

            direction, confidence = self._resolve_direction(
                health, realloc, regime_factor
            )
            directions.append({
                "strategy": sname,
                "health_score": health.get("health_score", 0.0),
                "health_grade": health.get("health_grade", "N/A"),
                "lifecycle": health.get("lifecycle", "unknown"),
                "trend": health.get("trend", "stable"),
                "win_rate": self._round(health.get("win_rate", 0.0)),
                "profit_factor": self._round(health.get("profit_factor", 0.0)),
                "total_pnl": self._round(health.get("total_pnl", 0.0)),
                "regime_factor": self._round(regime_factor),
                "direction": direction,
                "confidence": self._round(confidence),
            })

        directions.sort(key=lambda x: x["health_score"])

        return {
            "available": bool(directions),
            "overall_health": health_summary.get("overall_health"),
            "overall_health_grade": health_summary.get("overall_health_grade"),
            "synergy_score": health_summary.get("synergy_score"),
            "diversification_score": health_summary.get("diversification_score"),
            "ranking": ranking,
            "capital_reallocation_suggestions": reallocations,
            "strategy_directions": directions,
        }

    def _resolve_direction(
        self,
        health: Dict[str, Any],
        realloc: Dict[str, Any],
        regime_factor: float,
    ) -> Tuple[str, float]:
        """综合健康度、资金重分配建议与市场适配，输出方向与置信度。"""
        grade = health.get("health_grade", "N/A")
        trend = health.get("trend", "stable")
        realloc_action = realloc.get("action", "hold")

        # 市场状态适配优先（极端波动/流动性危机下强制降仓）
        if regime_factor <= 0.6:
            return "decrease", 0.85
        if regime_factor >= 1.2:
            base = "increase"
        elif regime_factor <= 0.8:
            base = "decrease"
        else:
            base = "maintain"

        # 贡献度建议覆盖
        if realloc_action == "reclaim":
            return "reclaim", 0.9
        if realloc_action == "increase" and base != "decrease":
            base = "increase"
        if realloc_action == "reduce":
            base = "decrease"

        # 健康度微调
        if grade in ("D", "F") and base == "maintain":
            base = "decrease"
        elif grade in ("A", "B") and trend == "improving" and base == "maintain":
            base = "increase"

        confidence = 0.6
        if realloc_action in ("increase", "reduce", "reclaim"):
            confidence = 0.75
        if grade in ("A", "F"):
            confidence = max(confidence, 0.8)

        return base, confidence

    # ==================================================================
    # 4. 信号质量
    # ==================================================================

    def analyze_signal_quality(self) -> Dict[str, Any]:
        """聚合信号质量：DQS 指标 + 交易记录维度的信号质量分布。"""
        result: Dict[str, Any] = {"available": False, "dqs": {}, "trade_based": {}}

        if self.decision_quality_evaluator is not None:
            try:
                metrics = self.decision_quality_evaluator.get_metrics()
                if metrics.get("total_decisions", 0) > 0:
                    result["dqs"] = {
                        "metrics": metrics,
                        "avg_dqs": self.decision_quality_evaluator.get_avg_dqs(),
                        "expectancy": self.decision_quality_evaluator.get_expectancy(),
                        "rolling_metrics": self.decision_quality_evaluator.get_all_rolling_metrics(),
                        "regime_attribution": self.decision_quality_evaluator.get_regime_attribution(),
                    }
                    result["available"] = True
                else:
                    result["dqs"] = {"available": False, "reason": "no decision records"}
            except Exception as e:
                logger.warning(f"signal quality (dqe) failed: {e}")

        trade_based = self._trade_based_signal_quality()
        if trade_based:
            result["trade_based"] = trade_based
            result["available"] = True

        result["summary"] = self._summarize_signal_quality(result)
        return result

    def _trade_based_signal_quality(self) -> Dict[str, Any]:
        """从交易记录按入场信号类型统计质量。"""
        if self.trade_journal is None:
            return {}
        try:
            trades = self.trade_journal.get_recent_trades(limit=1000)
        except Exception as e:
            logger.warning(f"trade-based signal quality failed: {e}")
            return {}

        if not trades:
            return {}

        by_signal: Dict[str, Dict[str, float]] = {}
        for t in trades:
            sig = t.get("entry_signal_type") or t.get("exit_reason") or "unknown"
            bucket = by_signal.setdefault(sig, {"count": 0, "wins": 0, "pnl": 0.0})
            bucket["count"] += 1
            bucket["pnl"] += t.get("pnl_usdt", 0.0)
            if t.get("win"):
                bucket["wins"] += 1

        signals = []
        for sig, b in by_signal.items():
            count = int(b["count"])
            signals.append({
                "signal_type": sig,
                "count": count,
                "win_rate": self._round(b["wins"] / count if count else 0.0),
                "total_pnl": self._round(b["pnl"]),
                "avg_pnl": self._round(b["pnl"] / count if count else 0.0),
            })
        signals.sort(key=lambda x: x["total_pnl"], reverse=True)

        return {"by_signal_type": signals}

    def _summarize_signal_quality(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """信号质量综合评级（优先 DQS，无 DQS 时回退交易记录胜率）。"""
        dqs = result.get("dqs", {})
        metrics = dqs.get("metrics") or {}
        avg_dqs = dqs.get("avg_dqs")
        win_rate = metrics.get("win_rate")

        # DQS 无数据 → 回退到交易记录维度
        if avg_dqs is None:
            signals = result.get("trade_based", {}).get("by_signal_type", [])
            total = sum(int(s["count"]) for s in signals)
            if total == 0:
                return {
                    "grade": "N/A",
                    "avg_dqs": None,
                    "win_rate": None,
                    "expectancy": None,
                    "comment": "无信号质量数据",
                }
            wins = sum(s["win_rate"] * s["count"] for s in signals)
            win_rate = wins / total
            if win_rate >= 0.6:
                grade, comment = "B", "信号质量良好（基于交易记录胜率）"
            elif win_rate >= 0.4:
                grade, comment = "C", "信号质量一般（基于交易记录胜率）"
            else:
                grade, comment = "D", "信号质量差（基于交易记录胜率）"
            return {
                "grade": grade,
                "avg_dqs": None,
                "win_rate": self._round(win_rate),
                "expectancy": None,
                "comment": comment,
            }

        # 有 DQS 数据
        if avg_dqs >= 70:
            grade, comment = "A", "信号质量优秀，DQS 稳定"
        elif avg_dqs >= 50:
            grade, comment = "B", "信号质量良好，可继续观察"
        elif avg_dqs >= 30:
            grade, comment = "C", "信号质量一般，建议收紧入场条件"
        else:
            grade, comment = "D", "信号质量差，建议暂停并复盘入场逻辑"

        return {
            "grade": grade,
            "avg_dqs": self._round(avg_dqs),
            "win_rate": self._round(win_rate),
            "expectancy": self._round(dqs.get("expectancy", 0.0)),
            "comment": comment,
        }

    # ==================================================================
    # 5. 交易记录分析
    # ==================================================================

    def analyze_trade_records(self) -> Dict[str, Any]:
        """交易记录多维分析（短板识别 + 成功模式提取）。"""
        if self.historical_analyzer is None:
            return {"available": False, "reason": "TradeJournal 未注入"}
        try:
            analysis = self.historical_analyzer.analyze_all_strategies(
                since=self._analysis_since
            )
            analysis["available"] = True
            return analysis
        except Exception as e:
            logger.warning(f"trade record analysis failed: {e}")
            return {"available": False, "reason": str(e)}

    # ==================================================================
    # 6. 报告生成
    # ==================================================================

    async def generate_analysis_report(self, include_adx: bool = True) -> Dict[str, Any]:
        """生成综合分析报告：市场状态 + ADX 确认 + 策略方向 + 信号质量 + 交易记录。"""
        adx_confirmation: Dict[str, Any] = {}
        if include_adx:
            adx_confirmation = await self.confirm_adx()

        report = {
            "report_type": "intelligent_analysis",
            "generated_at": datetime.now().isoformat(),
            "analysis_timeline": self._timeline_info(),
            "market_state": self.analyze_market_state(),
            "adx_confirmation": adx_confirmation,
            "strategy_direction": self.analyze_strategy_direction(),
            "signal_quality": self.analyze_signal_quality(),
            "trade_record_analysis": self.analyze_trade_records(),
        }
        report["summary"] = self._summarize_analysis(report)
        return report

    async def generate_optimization_report(self) -> Dict[str, Any]:
        """生成优化方向分析报告：参数优化 + 资金重分配 + 信号/ADX/市场自适应。"""
        strategy_optimization: Dict[str, Any] = {}
        if self.strategy_optimizer is not None:
            try:
                strategy_optimization = await self.strategy_optimizer.optimize_all_strategies(
                    since=self._analysis_since
                )
            except Exception as e:
                logger.warning(f"strategy optimization failed: {e}")
                strategy_optimization = {"error": str(e)}

        reallocations = []
        if self.contribution_analyzer is not None:
            try:
                reallocations = self.contribution_analyzer.get_capital_reallocation_suggestions(
                    since_override=self._analysis_since
                )
            except Exception as e:
                logger.warning(f"capital reallocation failed: {e}")

        signal_opt = self._signal_quality_optimization()
        adx_opt = self._adx_filter_optimization()
        market_opt = self._market_adaptive_optimization()

        report = {
            "report_type": "optimization_direction",
            "generated_at": datetime.now().isoformat(),
            "analysis_timeline": self._timeline_info(),
            "strategy_parameter_optimization": strategy_optimization,
            "capital_reallocation": reallocations,
            "signal_quality_optimization": signal_opt,
            "adx_filter_optimization": adx_opt,
            "market_adaptive_optimization": market_opt,
        }
        report["priority_actions"] = self._build_priority_actions(report)
        report["summary"] = self._summarize_optimization(report)
        return report

    async def generate_full_report(self, include_adx: bool = True) -> Dict[str, Any]:
        """一次生成分析报告与优化方向报告。"""
        analysis = await self.generate_analysis_report(include_adx=include_adx)
        optimization = await self.generate_optimization_report()
        return {
            "generated_at": datetime.now().isoformat(),
            "analysis_report": analysis,
            "optimization_report": optimization,
        }

    # ==================================================================
    # 优化方向细分
    # ==================================================================

    def _signal_quality_optimization(self) -> Dict[str, Any]:
        """基于信号质量产出优化方向。"""
        sq = self.analyze_signal_quality()
        summary = sq.get("summary", {})
        avg_dqs = summary.get("avg_dqs", 0.0)
        win_rate = summary.get("win_rate", 0.0)

        actions: List[Dict[str, Any]] = []
        if avg_dqs and avg_dqs < 30:
            actions.append({
                "dimension": "signal_quality",
                "action": "tighten",
                "detail": "DQS 过低，收紧入场信号门槛（提高 min_signal_quality）",
            })
        elif win_rate and win_rate < 0.4:
            actions.append({
                "dimension": "win_rate",
                "action": "add_confirmation",
                "detail": "胜率偏低，增加多时间框架/ADX 确认条件过滤低质信号",
            })
        else:
            actions.append({
                "dimension": "signal_quality",
                "action": "maintain",
                "detail": "信号质量处于可接受区间，维持当前门槛",
            })

        return {
            "avg_dqs": self._round(avg_dqs),
            "win_rate": self._round(win_rate),
            "actions": actions,
        }

    def _adx_filter_optimization(self) -> Dict[str, Any]:
        """基于市场状态给出 ADX 过滤优化方向。"""
        market = self.analyze_market_state()
        if not market.get("available"):
            return {"available": False}

        regime = market.get("regime", "unknown")
        regime_hint = "range" if regime in ("range_bound", "unknown") else "trend"

        if regime_hint == "range":
            recommendation = {
                "action": "prefer_range",
                "adx_threshold": 20.0,
                "detail": "震荡市 ADX 低，趋势策略应提高 ADX 门槛（>=20）过滤假突破，网格/剥头皮优先",
            }
        else:
            recommendation = {
                "action": "prefer_trend",
                "adx_threshold": 25.0,
                "detail": "趋势市 ADX 高，趋势策略以 ADX>=25 作为强确认，网格降低敞口",
            }

        return {
            "available": True,
            "regime": regime,
            "recommendation": recommendation,
        }

    def _market_adaptive_optimization(self) -> Dict[str, Any]:
        """基于市场状态给出策略权重自适应优化方向。"""
        market = self.analyze_market_state()
        if not market.get("available"):
            return {"available": False}

        regime = market.get("regime", "unknown")
        adjustment = market.get("strategy_adjustment", {})

        weight_map = {
            "trend_bullish": {"trend": 0.8, "grid": 0.2, "scalping": 0.2},
            "trend_bearish": {"trend": 0.6, "grid": 0.2, "arbitrage": 0.2},
            "range_bound": {"grid": 0.7, "scalping": 0.3, "trend": 0.0},
            "extreme_volatility": {"scalping": 0.5, "trend": 0.0, "grid": 0.0},
            "funding_crush": {"arbitrage": 0.5, "grid": 0.2},
            "liquidity_crisis": {"arbitrage": 0.1, "trend": 0.0},
        }

        return {
            "available": True,
            "regime": regime,
            "suggested_weights": weight_map.get(regime, {}),
            "position_adjustment": adjustment,
        }

    def _build_priority_actions(self, report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """汇总高优先级行动项。"""
        actions: List[Dict[str, Any]] = []

        for realloc in report.get("capital_reallocation", []):
            if realloc.get("action") in ("reclaim", "reduce"):
                actions.append({
                    "priority": "high",
                    "type": "capital",
                    "strategy": realloc.get("strategy"),
                    "action": realloc.get("action"),
                    "reason": realloc.get("reason", ""),
                })

        sq_actions = report.get("signal_quality_optimization", {}).get("actions", [])
        for a in sq_actions:
            if a.get("action") in ("tighten", "add_confirmation"):
                actions.append({
                    "priority": "medium",
                    "type": "signal",
                    "action": a.get("action"),
                    "reason": a.get("detail", ""),
                })

        actions.sort(key=lambda x: 0 if x["priority"] == "high" else 1)
        return actions

    # ==================================================================
    # 摘要
    # ==================================================================

    def _summarize_analysis(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """综合分析摘要。"""
        market = report.get("market_state", {})
        adx = report.get("adx_confirmation", {})
        direction = report.get("strategy_direction", {})
        signal_quality = report.get("signal_quality", {})
        sq = signal_quality.get("summary", {})

        overall_health_grade = (
            direction.get("overall_health_grade")
            if direction.get("available")
            else "N/A"
        )
        signal_quality_grade = (
            sq.get("grade", "N/A") if signal_quality.get("available") else "N/A"
        )

        return {
            "market_regime": market.get("regime", "unknown"),
            "market_strength": market.get("strength", 0.0),
            "adx_summary": adx.get("summary", {}),
            "overall_health_grade": overall_health_grade,
            "signal_quality_grade": signal_quality_grade,
            "dominant_direction": self._dominant_direction(
                direction.get("strategy_directions", [])
            ),
        }

    def _summarize_optimization(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """优化方向摘要。"""
        realloc = report.get("capital_reallocation", [])
        increase = sum(1 for r in realloc if r.get("action") == "increase")
        reduce = sum(1 for r in realloc if r.get("action") in ("reduce", "reclaim"))
        hold = sum(1 for r in realloc if r.get("action") == "hold")

        return {
            "strategy_count": len(realloc),
            "increase_count": increase,
            "reduce_count": reduce,
            "hold_count": hold,
            "priority_action_count": len(report.get("priority_actions", [])),
            "market_regime": report.get("market_adaptive_optimization", {}).get(
                "regime", "unknown"
            ),
        }

    @staticmethod
    def _dominant_direction(directions: List[Dict[str, Any]]) -> str:
        """统计策略方向主基调。"""
        if not directions:
            return "unknown"
        counts: Dict[str, int] = {}
        for d in directions:
            counts[d["direction"]] = counts.get(d["direction"], 0) + 1
        return max(counts, key=counts.get)

    # ==================================================================
    # 工具
    # ==================================================================

    @staticmethod
    def _round(value: Any, ndigits: int = 4) -> Any:
        """安全四舍五入（非数值原样返回）。"""
        if isinstance(value, float):
            return round(value, ndigits)
        if isinstance(value, int):
            return value
        return value

    def _round_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        return {k: self._round(v) for k, v in (d or {}).items()}
