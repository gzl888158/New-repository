"""反转落袋引擎集成层测试：StopLossManager.compute_reversal_take_profit + GridStrategy 趋势模式。"""

import pytest

from core.reversal_take_profit_engine import ReversalTakeProfitEngine, ReversalTakeProfitResult
from core.stop_loss_manager import StopLossManager
from strategies.grid_strategy import GridStrategy


def _make_engine(**overrides) -> ReversalTakeProfitEngine:
    cfg = {"reversal_take_profit": {}}
    for key, value in overrides.items():
        cfg["reversal_take_profit"][key] = value
    return ReversalTakeProfitEngine(cfg)


def _result(exit_action, score=0.8, direction="long", pnl_pct=0.01):
    return ReversalTakeProfitResult(
        symbol="BTC-USDT-SWAP",
        strategy_name="grid",
        reversal_score=score,
        reversal_source="hmm" if exit_action != "none" else "none",
        hmm_reversal=exit_action != "none",
        adaptive_tp_price=None,
        base_tp_price=None,
        tp_tighten_factor=1.0,
        exit_action=exit_action,
        partial_ratio=1.0 if exit_action == "close" else (0.5 if exit_action == "partial" else 0.0),
        exit_reason=f"reversal_{exit_action}",
        details={"pnl_pct": pnl_pct},
    )


class FakeDetector:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def get_regime(self, symbol):
        self.calls.append(symbol)
        return self.result


# ═══════════════════════════════════════════════════════════════
# StopLossManager.compute_reversal_take_profit（纯计算，不执行平仓）
# ═══════════════════════════════════════════════════════════════

class TestComputeReversalTakeProfit:
    def _slm(self, engine=None, detector=None):
        slm = StopLossManager.__new__(StopLossManager)
        slm._reversal_tp_engine = engine
        slm._market_regime_detector = detector
        slm._okx_client = None
        return slm

    @pytest.mark.asyncio
    async def test_returns_none_without_engine(self):
        slm = self._slm(engine=None)
        assert await slm.compute_reversal_take_profit("BTC", "grid", "long", 100.0, 101.0) is None

    @pytest.mark.asyncio
    async def test_hmm_reversal_drives_result(self):
        engine = _make_engine()
        detector = FakeDetector({"regime": "reversal", "probabilities": {}, "early_warnings": []})
        slm = self._slm(engine=engine, detector=detector)
        result = await slm.compute_reversal_take_profit("BTC", "grid", "long", 100.0, 101.0)
        assert result is not None
        assert result.hmm_reversal is True
        assert result.reversal_score == pytest.approx(0.80)
        assert detector.calls == ["BTC"]

    @pytest.mark.asyncio
    async def test_explicit_hmm_skips_detector(self):
        engine = _make_engine()
        detector = FakeDetector({"regime": "reversal", "probabilities": {}, "early_warnings": []})
        slm = self._slm(engine=engine, detector=detector)
        result = await slm.compute_reversal_take_profit(
            "BTC", "grid", "long", 100.0, 101.0,
            hmm_result={"regime": "reversal", "probabilities": {}, "early_warnings": []},
        )
        assert result is not None
        assert detector.calls == []


# ═══════════════════════════════════════════════════════════════
# GridStrategy._check_reversal_take_profit（趋势模式退出）
# ═══════════════════════════════════════════════════════════════

class FakeSLM:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def compute_reversal_take_profit(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class FakeTicker:
    def get_ticker(self, symbol):
        return {"last": "101"}

    async def get_ticker_async(self, symbol):
        return {"last": "101"}


class TestGridReversalTakeProfit:
    def _grid(self, slm, pos_side="buy", entry_price=100.0):
        grid = GridStrategy.__new__(GridStrategy)
        grid._stop_loss_manager = slm
        grid._position_side = {"BTC-USDT-SWAP": pos_side}
        grid._trailing_state = {"BTC-USDT-SWAP": {"entry_price": entry_price}}
        grid.okx_client = FakeTicker()
        grid._exited = []

        async def _exit(symbol):
            grid._exited.append(symbol)

        grid._exit_trend_mode = _exit
        return grid

    @pytest.mark.asyncio
    async def test_exit_on_reversal_signal(self):
        slm = FakeSLM(_result("partial"))
        grid = self._grid(slm)
        await grid._check_reversal_take_profit("BTC-USDT-SWAP")
        assert grid._exited == ["BTC-USDT-SWAP"]
        assert slm.calls[0]["direction"] == "long"

    @pytest.mark.asyncio
    async def test_short_direction_mapping(self):
        slm = FakeSLM(_result("close"))
        grid = self._grid(slm, pos_side="sell")
        await grid._check_reversal_take_profit("BTC-USDT-SWAP")
        assert slm.calls[0]["direction"] == "short"
        assert grid._exited == ["BTC-USDT-SWAP"]

    @pytest.mark.asyncio
    async def test_no_exit_on_none_action(self):
        slm = FakeSLM(_result("none", score=0.2))
        grid = self._grid(slm)
        await grid._check_reversal_take_profit("BTC-USDT-SWAP")
        assert grid._exited == []

    @pytest.mark.asyncio
    async def test_no_slm_returns_early(self):
        grid = self._grid(None)
        await grid._check_reversal_take_profit("BTC-USDT-SWAP")
        assert grid._exited == []

    @pytest.mark.asyncio
    async def test_missing_entry_price_returns_early(self):
        slm = FakeSLM(_result("partial"))
        grid = self._grid(slm, entry_price=0.0)
        await grid._check_reversal_take_profit("BTC-USDT-SWAP")
        assert grid._exited == []
