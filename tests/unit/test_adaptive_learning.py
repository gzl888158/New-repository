"""自适应学习系统（adaptive_learning）— 正式单元测试。

覆盖本次「强化企业级自适应学习系统」改动的核心修复点：
  P0  SampleBuffer 时间衰减（旧实现权重恒为 1）
  P0  SampleBuffer.clear() 重置 _total_added 计数
  P0  OnlineLearner 特征重要性缓存键（存储键与查询键统一为 "default"）
  P0  LearningStrategyOptimizer.should_reset 后半段均值计算
  P0  StrategyEvolver._tournament_select_crowded 使用当前进化策略参数
  P0  PerformanceFeedback.apply_feedback 前后指标深拷贝（闭环效果跟踪）
  P0  IntelligentTradingAgent.learn_from_history 返回 exit_updates
  P0  config 中 adaptive_learning 各子模块配置段透传
"""

import asyncio
import copy
from datetime import datetime, timedelta

import pytest

from app.services.adaptive_learning.online_learner import (
    OnlineLearner,
    SampleBuffer,
)
from app.services.adaptive_learning.meta_learner import LearningStrategyOptimizer
from app.services.adaptive_learning.strategy_evolver import StrategyEvolver
from app.services.adaptive_learning.parameter_adaptor import (
    BayesianOptimizer,
    ParameterAdaptor,
)
from app.services.adaptive_learning.market_regime_detector import (
    MarketRegime,
    RegimeTransitionPredictor,
)
from app.services.adaptive_learning.performance_feedback import (
    PerformanceFeedback,
    PerformanceMetrics,
)
from core.intelligent_agent import IntelligentTradingAgent


def _run(coro):
    """在独立事件循环中运行协程，避免依赖 pytest-asyncio。"""
    return asyncio.run(coro)


def _empty_sample(**overrides) -> dict:
    sample = {
        "market_features": {},
        "strategy_features": {},
        "predictions": {},
        "actuals": {},
        "outcome": {},
        "timestamp": datetime.now(),
    }
    sample.update(overrides)
    return sample


# ═══════════════════════════════════════════════════════════════
# P0: SampleBuffer 时间衰减
# ═══════════════════════════════════════════════════════════════

class TestSampleBufferTimeDecay:
    def test_empty_buffer_weight_is_one(self):
        buf = SampleBuffer(max_size=10, weight_method="time_decay", decay_factor=0.5)
        assert buf._compute_time_weight(datetime.now()) == 1.0

    def test_same_timestamp_weight_is_one(self):
        buf = SampleBuffer(max_size=10, weight_method="time_decay", decay_factor=0.5)
        t0 = datetime(2026, 8, 19, 10, 0, 0)
        buf.add(**_empty_sample(timestamp=t0))
        assert buf._compute_time_weight(t0) == 1.0

    def test_older_sample_weight_decays_by_hours(self):
        buf = SampleBuffer(max_size=10, weight_method="time_decay", decay_factor=0.5)
        t_now = datetime(2026, 8, 19, 12, 0, 0)
        buf.add(**_empty_sample(timestamp=t_now))
        # 早 2 小时的样本权重应为 0.5 ** 2 = 0.25
        older = t_now - timedelta(hours=2)
        assert abs(buf._compute_time_weight(older) - 0.25) < 1e-9

    def test_add_applies_decayed_weight(self):
        buf = SampleBuffer(max_size=10, weight_method="time_decay", decay_factor=0.5)
        t_now = datetime(2026, 8, 19, 12, 0, 0)
        buf.add(**_empty_sample(timestamp=t_now))
        older = t_now - timedelta(hours=1)
        buf.add(**_empty_sample(timestamp=older))
        # 最后一条（较旧）样本被赋予衰减后的权重 0.5
        assert abs(buf._samples[-1]["weight"] - 0.5) < 1e-9


# ═══════════════════════════════════════════════════════════════
# P0: SampleBuffer.clear() 重置计数
# ═══════════════════════════════════════════════════════════════

class TestSampleBufferClear:
    def test_clear_resets_total_added(self):
        buf = SampleBuffer(max_size=10)
        for _ in range(3):
            buf.add(**_empty_sample())
        assert len(buf) == 3
        assert buf._total_added == 3

        buf.clear()
        assert len(buf) == 0
        assert buf._total_added == 0

    def test_index_restarts_after_clear(self):
        buf = SampleBuffer(max_size=10)
        for _ in range(3):
            buf.add(**_empty_sample())
        buf.clear()
        buf.add(**_empty_sample())
        # 清除后新样本的 _index 应从 0 重新开始
        assert buf._samples[-1]["_index"] == 0


# ═══════════════════════════════════════════════════════════════
# P0: OnlineLearner 特征重要性缓存键
# ═══════════════════════════════════════════════════════════════

class TestOnlineLearnerFeatureImportance:
    def test_default_cache_key_hits(self):
        learner = OnlineLearner({"online_learner": {}})
        cached = [{"name": "price", "score": 0.9, "timestamp": "x"}]
        learner._feature_importance = {"default": cached}

        result = _run(learner.get_feature_importance())
        assert result == cached

    def test_strategy_specific_key_hits(self):
        learner = OnlineLearner({"online_learner": {}})
        cached = [{"name": "rsi", "score": 0.6, "timestamp": "x"}]
        learner._feature_importance = {"trend_combined": cached}

        result = _run(learner.get_feature_importance(strategy_name="trend"))
        assert result == cached


# ═══════════════════════════════════════════════════════════════
# P0: LearningStrategyOptimizer.should_reset 均值计算
# ═══════════════════════════════════════════════════════════════

class TestLearningStrategyOptimizerReset:
    def _make_optimizer(self, patience: int) -> LearningStrategyOptimizer:
        return LearningStrategyOptimizer(
            {"meta_learner": {"exploration": {"reset_patience": patience}}}
        )

    def test_declining_history_triggers_reset(self):
        opt = self._make_optimizer(patience=10)
        history = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        result = opt.should_reset(history)
        assert result["should_reset"] is True
        assert result["trend"] == "declining"

    def test_improving_history_does_not_reset(self):
        opt = self._make_optimizer(patience=10)
        history = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = opt.should_reset(history)
        assert result["should_reset"] is False

    def test_improving_second_half_marks_stable_trend(self):
        # 关键：后半段均值高于前半段时，修复后的均值计算应标记为改善趋势；
        # 旧实现 second_half_avg = sum/patience - half 恒为负，会误判为 declining。
        opt = self._make_optimizer(patience=10)
        history = [1, 1, 1, 1, 1, 10, 10, 10, 10, 10]
        result = opt.should_reset(history)
        assert result["trend"] == "stable_or_improving"
        assert result["should_reset"] is False

    def test_insufficient_history(self):
        opt = self._make_optimizer(patience=50)
        result = opt.should_reset([1.0, 2.0])
        assert result["should_reset"] is False
        assert result["reason"] == "insufficient_history"


# ═══════════════════════════════════════════════════════════════
# P0: StrategyEvolver._tournament_select_crowded 参数传递
# ═══════════════════════════════════════════════════════════════

class TestStrategyEvolverTournamentSelect:
    def _make_evolver(self) -> StrategyEvolver:
        return StrategyEvolver({"strategy_evolver": {}})

    def test_select_returns_member_of_population(self):
        evolver = self._make_evolver()
        parameters = {
            "a": {"min": 0.0, "max": 10.0},
            "b": {"min": 0.0, "max": 10.0},
        }
        population = [
            {"a": 1.0, "b": 2.0},
            {"a": 3.0, "b": 4.0},
            {"a": 5.0, "b": 6.0},
            {"a": 7.0, "b": 8.0},
        ]
        fitnesses = [1.0, 2.0, 3.0, 4.0]

        selected = evolver._tournament_select_crowded(
            population, fitnesses, parameters
        )
        assert set(selected.keys()) == {"a", "b"}
        assert selected in population

    def test_select_with_single_param_dimension(self):
        evolver = self._make_evolver()
        parameters = {"a": {"min": 0.0, "max": 100.0}}
        population = [{"a": 10.0}, {"a": 20.0}, {"a": 30.0}]
        fitnesses = [1.0, 2.0, 3.0]

        selected = evolver._tournament_select_crowded(
            population, fitnesses, parameters
        )
        assert selected in population


# ═══════════════════════════════════════════════════════════════
# P0: PerformanceFeedback.apply_feedback 深拷贝效果跟踪
# ═══════════════════════════════════════════════════════════════

class TestPerformanceFeedbackDeepCopy:
    def test_apply_feedback_tracks_effect_with_deepcopy(self, monkeypatch):
        pf = PerformanceFeedback({"performance_feedback": {}})
        pf._strategy_metrics["s"] = PerformanceMetrics(
            sharpe_ratio=1.0,
            win_rate=0.5,
            profit_factor=2.0,
            max_drawdown=0.1,
            consistency_score=0.8,
        )

        # 模拟反馈应用会改变策略指标，验证 before/after 深拷贝生效
        async def fake_apply_feedback(strategy, recommendations):
            pf._strategy_metrics[strategy] = PerformanceMetrics(
                sharpe_ratio=1.5,
                win_rate=0.6,
                profit_factor=2.5,
                max_drawdown=0.08,
                consistency_score=0.9,
            )
            applied = [{"id": rec["id"]} for rec in recommendations]
            for rec in applied:
                pf._feedback._applied_feedback[strategy].append(rec)
            return {
                "strategy": strategy,
                "applied": applied,
                "total_applied": len(applied),
                "total_skipped": 0,
            }

        monkeypatch.setattr(pf._feedback, "apply_feedback", fake_apply_feedback)

        recommendations = [{"id": "r1", "metric": "win_rate"}]
        result = _run(pf.apply_feedback("s", recommendations))

        assert result["total_applied"] == 1
        applied = pf._feedback._applied_feedback["s"]
        assert len(applied) == 1
        effect = applied[0].get("effect")
        assert effect is not None
        # 深拷贝修复后，before/after 差异应体现在 effect.changes 中
        assert effect["changes"]["sharpe_ratio"] == pytest.approx(0.5)
        assert effect["changes"]["win_rate"] == pytest.approx(0.1)
        assert effect["net_effect"] == "positive"

    def test_apply_feedback_no_recommendations(self):
        pf = PerformanceFeedback({"performance_feedback": {}})
        result = _run(pf.apply_feedback("s", []))
        assert result["applied"] == []
        assert "No recommendations" in result["message"]


# ═══════════════════════════════════════════════════════════════
# P0: IntelligentTradingAgent.learn_from_history 返回 exit_updates
# ═══════════════════════════════════════════════════════════════

class TestIntelligentAgentExitUpdates:
    def _make_agent(self, tmp_path) -> IntelligentTradingAgent:
        return IntelligentTradingAgent({
            "data_dir": str(tmp_path),
            "total_capital": 1000.0,
        })

    def test_learn_from_history_returns_exit_updates(self, tmp_path):
        agent = self._make_agent(tmp_path)
        # 50 笔全亏交易：累计 -50 USDT，胜率 0%，触发策略永久退出条件
        trades = [
            {
                "strategy_name": "trend",
                "symbol": "BTC-USDT-SWAP",
                "pnl_usdt": -1.0,
                "timestamp": "2026-08-19T00:00:00",
            }
            for _ in range(50)
        ]
        result = agent.learn_from_history(trades)

        assert "exit_updates" in result
        assert isinstance(result["exit_updates"], list)
        assert len(result["exit_updates"]) >= 1
        exited = result["exit_updates"][0]
        assert exited["strategy"] == "trend"
        assert exited["action"] == "exited"
        assert exited["trade_count"] == 50
        assert agent.is_strategy_exited("trend") is True

    def test_empty_history_returns_no_exit_updates(self, tmp_path):
        agent = self._make_agent(tmp_path)
        result = agent.learn_from_history([])
        assert result["learned"] is False
        assert "exit_updates" not in result or result.get("exit_updates") == []


# ═══════════════════════════════════════════════════════════════
# P0: config 中 adaptive_learning 子模块配置段透传
# ═══════════════════════════════════════════════════════════════

class TestAdaptiveLearningConfigSections:
    def test_sections_present_in_loaded_config(self, config):
        for section in (
            "online_learner",
            "parameter_adaptor",
            "performance_feedback",
            "knowledge_base",
            "market_regime_detector",
            "meta_learner",
            "strategy_evolver",
        ):
            assert section in config, f"missing config section: {section}"
            assert isinstance(config[section], dict), f"{section} is not a dict"

    def test_online_learner_section_has_expected_keys(self, config):
        ol = config["online_learner"]
        assert "learning_mode" in ol
        assert "learning_rate" in ol
        assert "sample_buffer" in ol
        assert "drift_detection" in ol
        assert "feature_importance" in ol
        assert "calibration" in ol

    def test_strategy_evolver_section_has_expected_keys(self, config):
        se = config["strategy_evolver"]
        assert "evolution_strategy" in se
        assert "population_size" in se
        assert "generations" in se


# ═══════════════════════════════════════════════════════════════
# P0: BayesianOptimizer.observe 参数顺序（按 param_names 提取，避免 dict 插入顺序错位）
# ═══════════════════════════════════════════════════════════════

class TestBayesianOptimizerParamOrder:
    def test_observe_extracts_by_param_names_order(self):
        bo = BayesianOptimizer({})
        bo.set_bounds([(0.0, 10.0), (0.0, 10.0)], ["a", "b"])
        # 故意以与参数名顺序相反的插入顺序传入
        _run(bo.observe({"b": 9.0, "a": 1.0}, score=2.0))

        # 应严格按 ["a", "b"] 提取为 [1.0, 9.0]，而非按插入顺序 [9.0, 1.0]
        assert bo._X[0] == [1.0, 9.0]
        assert bo._y[0] == 2.0

    def test_observe_falls_back_to_insertion_order_without_names(self):
        bo = BayesianOptimizer({})
        bo.set_bounds([(0.0, 10.0), (0.0, 10.0)])
        _run(bo.observe({"b": 9.0, "a": 1.0}, score=2.0))
        # 未设置 param_names 时回退到旧行为（按 dict 插入顺序）
        assert bo._X[0] == [9.0, 1.0]


class TestParameterAdaptorUpdateObservationOrder:
    def test_update_observation_preserves_sorted_key_order(self):
        pa = ParameterAdaptor({"parameter_adaptor": {}})
        pa.register_parameter("grid", "b", current_value=0.5, min_value=0.0, max_value=1.0)
        pa.register_parameter("grid", "a", current_value=0.5, min_value=0.0, max_value=1.0)
        # sorted keys: ["grid_a", "grid_b"]；传入 dict 用相反插入顺序
        result = _run(pa.update_observation({"grid_b": 0.9, "grid_a": 0.1}, score=1.0))

        assert result["status"] == "observed"
        # GP 观测值应按 sorted key 顺序提取为 [0.1, 0.9]
        assert pa._bayesian._X[0] == [0.1, 0.9]


# ═══════════════════════════════════════════════════════════════
# P0: RegimeTransitionPredictor.predict_next_regime 使用 HMM 转移矩阵
# ═══════════════════════════════════════════════════════════════

class TestRegimeTransitionPredictorHMM:
    def test_predict_next_regime_uses_hmm_transition_matrix(self):
        rtp = RegimeTransitionPredictor({"hmm_weight": 1.0})
        rtp.update_history(MarketRegime.TRENDING_UP)
        # 7x7 转移矩阵：当前 TRENDING_UP(idx 0) 高概率转向 TRENDING_DOWN(idx 1)
        tm = [[0.0] * 7 for _ in range(7)]
        tm[0][1] = 1.0

        result = rtp.predict_next_regime(tm)

        assert result["most_likely_next"] == "trending_down"
        assert result["next_probabilities"]["trending_down"] == 1.0

    def test_predict_next_regime_falls_back_to_empirical_without_matrix(self):
        rtp = RegimeTransitionPredictor({})
        rtp.update_history(MarketRegime.TRENDING_UP)

        result = rtp.predict_next_regime(None)

        # 无转移矩阵时回退经验计数（无历史转移 → 保持当前状态）
        assert result["most_likely_next"] == "trending_up"

    def test_predict_next_regime_blends_hmm_and_empirical(self):
        rtp = RegimeTransitionPredictor({"hmm_weight": 0.5})
        rtp.update_history(MarketRegime.TRENDING_UP)
        tm = [[0.0] * 7 for _ in range(7)]
        tm[0][1] = 1.0  # HMM 指向 TRENDING_DOWN

        result = rtp.predict_next_regime(tm)

        # 加权融合：HMM(0.5 * trending_down) + 经验(0.5 * trending_up)
        assert result["next_probabilities"]["trending_down"] == pytest.approx(0.5)
        assert result["next_probabilities"]["trending_up"] == pytest.approx(0.5)

    def test_predict_next_regime_insufficient_data(self):
        rtp = RegimeTransitionPredictor({})
        result = rtp.predict_next_regime()
        assert result["status"] == "insufficient_data"
