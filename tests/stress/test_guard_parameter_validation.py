"""
守卫参数回测有效性验证（guard parameter backtest validation）
=============================================================
针对策略级守卫参数（信号质量门槛 / 单日交易上限 / 磨损过滤器 / 趋势方向过滤），
用合成 K 线 A/B 回测「守卫开 vs 守卫关」，验证每个守卫参数是否真实有效：

  - min_signal_quality  → 过滤低质量噪声信号（交易数只减不增，降低手续费侵蚀）
  - max_daily_trades    → 抑制高频（单日交易数封顶）
  - wear_fee_multiple   → 结构性无利可图的止盈目标直接弃单（0 交易）
  - trend_filter_threshold → 强趋势下禁止逆势开仓（消除追涨杀跌）

与 tests/perf/（吞吐/延迟）和 tests/stress/test_agi_failure_scenarios.py（失效韧性）
区分：本套聚焦「守卫参数的策略效果」，用 BacktestEngine 的合成数据 A/B 回测定量验证。
"""
from datetime import datetime, timedelta

import numpy as np

from backtest.backtest_engine import BacktestEngine


def _engine() -> BacktestEngine:
    return BacktestEngine({
        "execution": {
            "taker_fee": 0.0005,
            "maker_fee": 0.0002,
            "funding_rate": 0.0001,
            "funding_interval_hours": 8,
        },
    })


def _candles(n: int = 400, base: float = 100.0, mode: str = "oscillate", seed: int = 42):
    """生成 Dict 格式合成 K 线（BacktestEngine 接口）。

    mode:
      - oscillate: 正弦震荡 + 噪声 → 均值回归，RSI 超买超卖反复触发（scalping/grid 主战场）
      - uptrend / downtrend: 单边趋势 → 用于验证趋势方向过滤（禁止逆势）
    """
    rng = np.random.default_rng(seed)
    candles = []
    ts = datetime(2026, 1, 1)
    price = base
    for i in range(n):
        if mode == "oscillate":
            price = base + 6.0 * np.sin(i / 6.0) + rng.normal(0, 0.15)
        elif mode == "uptrend":
            # 指数上行：百分比趋势强度恒定（线性趋势的百分比强度会随价格升高而衰减，
            # 导致 |EMA20-EMA50|/EMA50 最终跌破 0.05 使过滤器失效）
            price = base * (1.01 ** i) + rng.normal(0, base * 0.002)
        elif mode == "downtrend":
            price = base * (0.99 ** i) + rng.normal(0, base * 0.002)
        open_ = price
        close = price + rng.normal(0, 0.05)
        high = max(open_, close) * 1.002
        low = min(open_, close) * 0.998
        volume = 1000 + abs(rng.normal(0, 50))
        candles.append({
            "timestamp": ts + timedelta(hours=i),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })
    return candles


def _closed(trades):
    return [t for t in trades if t.status == "closed"]


def _total_fee(trades):
    return sum(t.fee + t.funding_fee for t in trades)


# ─────────────────────────────────────────────────────────────
# 1. 信号质量门槛（min_signal_quality）
# ─────────────────────────────────────────────────────────────
class TestSignalQualityGuard:
    def test_guard_never_increases_trades(self):
        """门槛只减不增：信号质量过滤只会移除噪声信号，绝不会新增交易。"""
        eng = _engine()
        candles = _candles(mode="oscillate")
        base = eng.run_scalping_with_candles(
            candles, "X-USDT-SWAP", min_signal_quality=0.0, max_daily_trades=0, wear_fee_multiple=0.0)
        guarded = eng.run_scalping_with_candles(
            candles, "X-USDT-SWAP", min_signal_quality=0.35, max_daily_trades=0, wear_fee_multiple=0.0)
        n_base = len(_closed(base.trades))
        n_guarded = len(_closed(guarded.trades))
        print(f"\n[信号质量门槛] 无门槛 {n_base} 笔 → 门槛0.35 {n_guarded} 笔")
        assert n_guarded <= n_base


# ─────────────────────────────────────────────────────────────
# 2. 单日交易上限（max_daily_trades）
# ─────────────────────────────────────────────────────────────
class TestMaxDailyTradesGuard:
    def test_guard_caps_daily_trades(self):
        """单日交易上限：封顶后每自然日交易数不超过阈值（高频抑制）。"""
        eng = _engine()
        candles = _candles(mode="oscillate")
        capped = eng.run_scalping_with_candles(
            candles, "X-USDT-SWAP", min_signal_quality=0.0, max_daily_trades=1, wear_fee_multiple=0.0)
        unlimited = eng.run_scalping_with_candles(
            candles, "X-USDT-SWAP", min_signal_quality=0.0, max_daily_trades=0, wear_fee_multiple=0.0)

        # 按入场日统计每日交易数，验证封顶后无单日超过 1 笔
        per_day = {}
        for t in _closed(capped.trades):
            day = t.timestamp.date()
            per_day[day] = per_day.get(day, 0) + 1
        print(f"\n[单日上限] 无上限 {len(_closed(unlimited.trades))} 笔 → 上限1 {len(_closed(capped.trades))} 笔")
        assert all(v <= 1 for v in per_day.values()), f"存在单日超过1笔: {per_day}"
        assert len(_closed(capped.trades)) <= len(_closed(unlimited.trades))


# ─────────────────────────────────────────────────────────────
# 3. 磨损过滤器（wear_fee_multiple）
# ─────────────────────────────────────────────────────────────
class TestWearFeeGuard:
    def test_unprofitable_target_skipped(self):
        """磨损过滤器：止盈目标 ≤ N× 往返手续费时结构性无利可图，必须弃单（0 交易）。"""
        eng = _engine()
        candles = _candles(mode="oscillate")
        # 止盈 0.1% 远小于 3× 往返 taker 手续费（3×0.1%=0.3%）→ 应整体跳过
        result = eng.run_scalping_with_candles(
            candles, "X-USDT-SWAP", profit_target_min=0.001, wear_fee_multiple=3.0,
            min_signal_quality=0.0, max_daily_trades=0)
        print(f"\n[磨损过滤器] 止盈0.1% (≤3×手续费) → {len(_closed(result.trades))} 笔")
        assert len(_closed(result.trades)) == 0

    def test_profitable_target_runs(self):
        """止盈目标高于磨损阈值时应正常交易（过滤器不误伤）。"""
        eng = _engine()
        candles = _candles(mode="oscillate")
        result = eng.run_scalping_with_candles(
            candles, "X-USDT-SWAP", profit_target_min=0.008, wear_fee_multiple=3.0,
            min_signal_quality=0.0, max_daily_trades=0)
        print(f"[磨损过滤器] 止盈0.8% (>3×手续费) → {len(_closed(result.trades))} 笔")
        assert len(_closed(result.trades)) > 0


# ─────────────────────────────────────────────────────────────
# 4. 趋势方向过滤（trend_filter_threshold，grid）
# ─────────────────────────────────────────────────────────────
class TestTrendFilterGuard:
    def test_blocks_counter_trend_short_in_uptrend(self):
        """强上涨趋势下禁止逆势做空：有过滤器时 0 笔空单。"""
        eng = _engine()
        candles = _candles(mode="uptrend")
        no_filter = eng.run_grid_with_candles(
            candles, "X-USDT-SWAP", grid_spacing=0.03, min_signal_quality=0.0,
            trend_filter_threshold=0.0)
        filtered = eng.run_grid_with_candles(
            candles, "X-USDT-SWAP", grid_spacing=0.03, min_signal_quality=0.0,
            trend_filter_threshold=0.05)
        shorts_no = sum(1 for t in _closed(no_filter.trades) if t.direction == "short")
        shorts_filtered = sum(1 for t in _closed(filtered.trades) if t.direction == "short")
        print(f"\n[趋势过滤] 强上涨趋势逆势空单：无过滤 {shorts_no} 笔 → 过滤 {shorts_filtered} 笔")
        # 有过滤时强上涨趋势下不应存在逆势空单
        assert shorts_filtered == 0
        # 无过滤时应确能观察到逆势空单（证明数据确实产生了逆势信号，过滤有真实作用）
        assert shorts_no > 0

    def test_blocks_counter_trend_long_in_downtrend(self):
        """强下跌趋势下禁止逆势做多：有过滤器时 0 笔多单。"""
        eng = _engine()
        candles = _candles(mode="downtrend")
        filtered = eng.run_grid_with_candles(
            candles, "X-USDT-SWAP", grid_spacing=0.03, min_signal_quality=0.0,
            trend_filter_threshold=0.05)
        longs_filtered = sum(1 for t in _closed(filtered.trades) if t.direction == "long")
        print(f"[趋势过滤] 强下跌趋势逆势多单：过滤后 {longs_filtered} 笔")
        assert longs_filtered == 0
