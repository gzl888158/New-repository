"""企业级「事前」震荡磨损防护 单元测试。

覆盖三类新增能力：
1. TradeCostAnalyzer.should_open_position 检查4：24h 振幅是否足以容纳
   「预期盈利 + 全链路成本」，窄幅震荡下拒绝无利可图的磨损型开仓。
2. OrderExecutor._derive_expected_profit_pct：从策略止盈价推导真实预期盈利。
3. IntelligentTradingAgent.get_symbol_volatility_amplitude / OrderExecutor
   ._get_market_volatility：真实 24h 振幅提取的量纲边界（0.0 < vol <= 0.5）。
"""
import os
import pytest


@pytest.fixture
def cost_analyzer():
    from core.trade_cost_analyzer import TradeCostAnalyzer
    # 用 Lv1 taker 0.05% 构造成本分析器，成本可控且可复现
    return TradeCostAnalyzer({"taker_fee": 0.0005, "maker_fee": 0.0002})


def _open(analyzer, vol, tp_pct=0.01, hold_hours=24.0):
    return analyzer.should_open_position(
        symbol="SOL-USDT-SWAP", side="buy", price=100.0, quantity=10.0,
        leverage=1.0, pos_side="long",
        expected_profit_pct=tp_pct,
        expected_hold_hours=hold_hours,
        market_volatility=vol,
    )


class TestShouldOpenPositionVolatilityGuard:
    def test_narrow_range_rejects(self, cost_analyzer):
        # 止盈 1%，全链路成本约 0.21%，required≈1.21%；24h 振幅 0.5% 明显不足 -> 拒绝
        allowed, reason, _ = _open(cost_analyzer, vol=0.005, tp_pct=0.01)
        assert allowed is False
        assert "震荡空间不足" in reason

    def test_wide_range_allows(self, cost_analyzer):
        # 24h 振幅 5% 远大于 required -> 放行
        allowed, reason, _ = _open(cost_analyzer, vol=0.05, tp_pct=0.01)
        assert allowed is True
        assert reason == "OK"

    def test_long_horizon_scales_24h_amplitude(self, cost_analyzer):
        allowed, reason, _ = _open(
            cost_analyzer, vol=0.03, tp_pct=0.039381, hold_hours=72.0
        )
        assert allowed is True
        assert reason == "OK"

    def test_short_horizon_scales_24h_amplitude_down(self, cost_analyzer):
        allowed, reason, _ = _open(
            cost_analyzer, vol=0.03, tp_pct=0.02, hold_hours=6.0
        )
        assert allowed is False
        assert "震荡空间不足" in reason

    def test_none_volatility_skips_guard(self, cost_analyzer):
        # 未传振幅：跳过检查4（仍执行检查1~3），止盈 1% 可覆盖成本 -> 放行
        allowed, reason, _ = _open(cost_analyzer, vol=None, tp_pct=0.01)
        assert allowed is True

    def test_small_tp_still_blocked_by_cost_check(self, cost_analyzer):
        # 微利止盈（0.3%）< 3x 成本：无论振幅多大，先被检查2（盈利覆盖成本）拦截
        allowed, reason, _ = _open(cost_analyzer, vol=0.10, tp_pct=0.003)
        assert allowed is False
        assert "预期盈利不足以覆盖成本" in reason


class TestDeriveExpectedProfitPct:
    @pytest.fixture
    def executor(self):
        from execution.order_executor import OrderExecutor
        # 纯方法，无需完整初始化
        return OrderExecutor.__new__(OrderExecutor)

    def test_explicit_field(self, executor):
        assert executor._derive_expected_profit_pct(
            {"expected_profit_pct": 0.02, "take_profit": 100.3}, 100.0
        ) == pytest.approx(0.02)

    def test_derive_from_take_profit(self, executor):
        assert executor._derive_expected_profit_pct(
            {"take_profit": 100.3}, 100.0
        ) == pytest.approx(0.003)

    def test_default_fallback(self, executor):
        assert executor._derive_expected_profit_pct({}, 100.0) == pytest.approx(0.01)


class TestDeriveExpectedHoldHours:
    def test_signal_horizon_takes_precedence(self):
        from execution.order_executor import OrderExecutor
        executor = OrderExecutor.__new__(OrderExecutor)
        executor.config = {"strategies": {"grid": {"max_hold_hours": 72.0}}}

        assert executor._derive_expected_hold_hours(
            {"strategy_name": "grid", "expected_hold_hours": 12.0}
        ) == pytest.approx(12.0)

    def test_strategy_horizon_falls_back_to_max_hold(self):
        from execution.order_executor import OrderExecutor
        executor = OrderExecutor.__new__(OrderExecutor)
        executor.config = {"strategies": {"grid": {"max_hold_hours": 72.0}}}

        assert executor._derive_expected_hold_hours(
            {"strategy_name": "grid"}
        ) == pytest.approx(72.0)

    def test_planned_time_exit_precedes_max_hold_cap(self):
        from execution.order_executor import OrderExecutor
        executor = OrderExecutor.__new__(OrderExecutor)
        executor.config = {
            "strategies": {
                "scalping": {
                    "time_exit_after_hours": 2.0,
                    "max_hold_hours": 4.0,
                }
            }
        }

        assert executor._derive_expected_hold_hours(
            {"strategy_name": "scalping"}
        ) == pytest.approx(2.0)


class TestGetMarketVolatility:
    def _executor_with_agent(self, amplitude):
        from execution.order_executor import OrderExecutor
        ex = OrderExecutor.__new__(OrderExecutor)

        class _Agent:
            def get_symbol_volatility_amplitude(self, symbol):
                return amplitude

        ex._intelligent_agent = _Agent()
        return ex

    def test_returns_amplitude(self):
        ex = self._executor_with_agent(0.02)
        assert ex._get_market_volatility("SOL-USDT-SWAP") == pytest.approx(0.02)

    def test_no_agent_returns_none(self):
        from execution.order_executor import OrderExecutor
        ex = OrderExecutor.__new__(OrderExecutor)
        assert ex._get_market_volatility("SOL-USDT-SWAP") is None

    def test_agent_returns_none(self):
        ex = self._executor_with_agent(None)
        assert ex._get_market_volatility("SOL-USDT-SWAP") is None


class TestGetSymbolVolatilityAmplitude:
    @pytest.fixture
    def agent(self):
        from core.intelligent_agent import IntelligentTradingAgent
        return IntelligentTradingAgent.__new__(IntelligentTradingAgent)

    def _set_fused(self, agent, volatility):
        agent._get_fused_regime_data = lambda symbol: {"volatility": volatility}

    def test_valid_amplitude(self, agent):
        self._set_fused(agent, 0.02)
        assert agent.get_symbol_volatility_amplitude("X") == pytest.approx(0.02)

    def test_too_large_rejected(self, agent):
        # > 0.5 的量纲视为波动率得分，不可靠 -> None
        self._set_fused(agent, 0.6)
        assert agent.get_symbol_volatility_amplitude("X") is None

    def test_non_positive_rejected(self, agent):
        self._set_fused(agent, -0.3)
        assert agent.get_symbol_volatility_amplitude("X") is None
        self._set_fused(agent, 0.0)
        assert agent.get_symbol_volatility_amplitude("X") is None

    def test_no_data_returns_none(self, agent):
        agent._get_fused_regime_data = lambda symbol: None
        assert agent.get_symbol_volatility_amplitude("X") is None


class TestClassifyCostReason:
    @pytest.fixture
    def executor(self):
        from execution.order_executor import OrderExecutor
        return OrderExecutor.__new__(OrderExecutor)

    def test_insufficient_volatility(self, executor):
        assert executor._classify_cost_reason(
            "震荡空间不足: 24h振幅=0.50% < 预期盈利+成本=1.21%"
        ) == "insufficient_volatility"

    def test_profit_below_cost(self, executor):
        assert executor._classify_cost_reason(
            "预期盈利不足以覆盖成本: profit=3.0000 < cost*3.0=6.3000"
        ) == "profit_below_cost"

    def test_notional_too_low(self, executor):
        assert executor._classify_cost_reason(
            "名义价值过低 10.00 < 63.00"
        ) == "notional_too_low"

    def test_move_too_large(self, executor):
        assert executor._classify_cost_reason(
            "所需价格变动过大: 7.00% > 5%"
        ) == "move_too_large"

    def test_fallback(self, executor):
        assert executor._classify_cost_reason("未知成本拦截原因") == "cost_blocked"
        assert executor._classify_cost_reason("") == "cost_blocked"
