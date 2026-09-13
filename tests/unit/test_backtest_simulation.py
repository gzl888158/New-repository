"""
回测 / 模拟验证

覆盖用户确认的验证方式之一：回测 / 模拟验证（mini_backtester / paper）。

两部分：
1. MiniBacktester 滚动回测管线（合成K线 + 交易记录 → 指标 + 健康评估）
2. TrendStrategy._detect_trend 端到端趋势判断（合成上涨/下跌/震荡K线 → 方向 + 置信度）
"""
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from core.mini_backtester import MiniBacktester, TradeRecord
from strategies.trend_strategy import TrendStrategy


# ---------------------------------------------------------------------------
# 合成K线生成
# ---------------------------------------------------------------------------
def _make_klines(direction: str = "up", n: int = 120, base: float = 100.0):
    """生成 [timestamp, open, high, low, close, volume] 格式的合成K线。

    direction:
      - "up":   持续上涨，ma20 > ma50，recent_close 突破近期高点
      - "down": 持续下跌，ma20 < ma50，recent_close 跌破近期低点
      - "flat": 震荡横盘，均线纠缠，无明确方向
    """
    rng = np.random.default_rng(42)
    klines = []
    ts = int(datetime(2026, 1, 1).timestamp() * 1000)
    prev_close = base

    for i in range(n):
        if direction == "up":
            close = base + i * 0.25 + rng.normal(0, 0.15)
        elif direction == "down":
            close = base + (n - i) * 0.25 + rng.normal(0, 0.15)
        else:
            close = base + rng.normal(0, 0.3)

        open_ = prev_close
        high = max(open_, close) * 1.001
        low = min(open_, close) * 0.999
        volume = 1000 + abs(rng.normal(0, 100))

        klines.append([ts + i * 3600_000, open_, high, low, close, volume])
        prev_close = close

    return klines


def _make_trend():
    """构造轻量 TrendStrategy 实例（跳过 __init__），仅用于趋势判断逻辑。"""
    s = TrendStrategy.__new__(TrendStrategy)
    s.config = {}
    s.okx_client = None
    s.redis_cache = None

    # _detect_trend 依赖
    s._confirmation_periods = ["1H", "4H"]
    s._multi_timeframe_confirmation = True
    s._trend_strength_filter = True
    s._breakout_confirmation = False
    s._pullback_entry = False
    s._indicator_cache = {}
    s._market_state = {}
    s._dynamic_adx_threshold = True
    s._adx_threshold = 25
    s._adx_vol_high = 30
    s._adx_vol_low = 20
    s._adx_vol_normal = 25

    # 指标计算依赖
    s._rsi_period = 14
    s._rsi_overbought = 70
    s._rsi_oversold = 30
    s._adx_period = 14
    s._atr_period = 14
    s._dx_cache = {}

    # _analyze_period_with_indicators 依赖
    s._divergence_detection = False
    s._market_structure_enabled = False

    # 共享趋势判断投票依赖（_detect_trend 末尾的共享投票校准）
    s._shared_trend_vote_enabled = True
    s._shared_trend_vote_weight = 0.15
    s._shared_vote_adx_floor = 15.0
    s._shared_vote_adx_saturation = 40.0

    return s


# ---------------------------------------------------------------------------
# 1. MiniBacktester 管线验证
# ---------------------------------------------------------------------------
class TestMiniBacktesterPipeline:
    def test_register_bars_trades_and_backtest(self):
        bt = MiniBacktester()
        bt.register_strategy("trend", "BTC-USDT-SWAP")

        for i in range(30):
            bt.add_bar("trend", {
                "timestamp": datetime(2026, 1, 1) + timedelta(hours=i),
                "open": 100 + i * 0.1,
                "high": 101 + i * 0.1,
                "low": 99 + i * 0.1,
                "close": 100 + i * 0.1,
                "volume": 1000,
            })

        trades = [
            ("buy", 100.0, 101.0, 0.05, 0.10),   # win
            ("sell", 101.0, 99.5, 0.05, 0.10),   # win
            ("buy", 99.5, 100.5, 0.05, 0.10),    # win
            ("sell", 100.5, 102.0, 0.05, -0.10), # loss
            ("buy", 102.0, 103.0, 0.05, 0.10),   # win
            ("sell", 103.0, 104.5, 0.05, -0.10), # loss
        ]
        for i, (side, entry, exit_, qty, pnl) in enumerate(trades):
            bt.add_trade("trend", TradeRecord(
                timestamp=datetime(2026, 1, 1) + timedelta(hours=i),
                symbol="BTC-USDT-SWAP",
                side=side,
                entry_price=entry,
                exit_price=exit_,
                quantity=qty,
                pnl=pnl,
                pnl_pct=pnl / entry,
                hold_bars=5,
                signal_source="trend",
                entry_reason="test",
                exit_reason="test",
            ))

        result = bt.run_backtest("trend")

        assert result is not None
        assert result.total_trades == 6
        assert result.winning_trades == 4
        assert result.losing_trades == 2
        assert abs(result.win_rate - 4 / 6) < 1e-9
        assert abs(result.total_pnl - 0.20) < 1e-9  # 0.10+0.10+0.10-0.10+0.10-0.10
        assert result.profit_factor > 0

    def test_backtest_insufficient_trades_returns_none(self):
        bt = MiniBacktester()
        bt.register_strategy("trend", "BTC-USDT-SWAP")
        bt.add_bar("trend", {"close": 100, "open": 100, "high": 101, "low": 99, "volume": 1000})
        bt.add_trade("trend", TradeRecord(
            timestamp=datetime.now(), symbol="BTC-USDT-SWAP", side="buy",
            entry_price=100, exit_price=101, quantity=0.1, pnl=0.1, pnl_pct=0.001,
            hold_bars=1, signal_source="trend", entry_reason="t", exit_reason="t",
        ))
        # 交易数 < min_trades_for_evaluation(5) -> None
        assert bt.run_backtest("trend") is None

    def test_stats_reporting(self):
        bt = MiniBacktester()
        bt.register_strategy("trend", "BTC-USDT-SWAP")
        stats = bt.get_stats()
        assert stats["strategies_registered"] == 1
        assert stats["backtests_run"] == 0


# ---------------------------------------------------------------------------
# 2. 趋势判断端到端验证（合成K线 → _detect_trend）
# ---------------------------------------------------------------------------
class TestTrendDetectSimulation:
    def test_uptrend_detects_long(self):
        async def _run():
            s = _make_trend()
            klines = _make_klines("up")
            s.okx_client = AsyncMock()
            s.okx_client.get_kline_async.return_value = klines
            return await s._detect_trend("BTC-USDT-SWAP")

        direction, confidence = asyncio.run(_run())
        assert direction == "long"
        assert 0.4 <= confidence <= 0.95

    def test_downtrend_detects_short(self):
        async def _run():
            s = _make_trend()
            klines = _make_klines("down")
            s.okx_client = AsyncMock()
            s.okx_client.get_kline_async.return_value = klines
            return await s._detect_trend("BTC-USDT-SWAP")

        direction, confidence = asyncio.run(_run())
        assert direction == "short"
        assert 0.4 <= confidence <= 0.95

    def test_sideways_no_direction(self):
        async def _run():
            s = _make_trend()
            klines = _make_klines("flat")
            s.okx_client = AsyncMock()
            s.okx_client.get_kline_async.return_value = klines
            return await s._detect_trend("BTC-USDT-SWAP")

        direction, confidence = asyncio.run(_run())
        # 震荡市：无明确方向（可能返回 None，或置信度极低）
        assert direction is None or confidence < 0.5

    def test_invalid_symbol_rejected(self):
        async def _run():
            s = _make_trend()
            return await s._detect_trend("INVALID")

        direction, confidence = asyncio.run(_run())
        assert direction is None
        assert confidence == 0

    def test_kline_failure_returns_none(self):
        async def _run():
            s = _make_trend()
            s.okx_client = AsyncMock()
            # 模拟K线获取失败
            s.okx_client.get_kline_async.side_effect = Exception("network error")
            return await s._detect_trend("BTC-USDT-SWAP")

        direction, confidence = asyncio.run(_run())
        assert direction is None
        assert confidence == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
