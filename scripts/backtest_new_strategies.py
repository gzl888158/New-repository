"""
新策略体系信号级离线回测（A/B/C/E）—— 含 ATR 止盈止损模拟

对最新系统化策略框架做只读信号级回测，不触碰实盘：
  A. 多周期 EMA 趋势跟踪          —— evaluate_ema_trend
  B. 唐奇安通道突破（海龟）       —— evaluate_donchian
  C. 多币种动量轮动（资金分散）   —— rank_momentum / rolling_return（横截面）
  E. 带趋势过滤器的布林均值回归   —— 内联复刻 BollingerMeanReversionStrategy._evaluate

两个评估维度：
  1) 方向命中率：long 命中 = 未来收益 > 0；short 命中 = 未来收益 < 0。
  2) ATR 止盈止损模拟：以信号收盘价入场，逐 bar 检查 SL/TP 触发，统计
     胜率 / 平均每笔收益 / 盈亏比。用于评估趋势策略（A/B）的赔率而非入场胜率。

用法：
  py -3 scripts/backtest_new_strategies.py [--symbols ...] [--limit 400] [--bar 1H] [--forward 4,8,24]
      [--ema_atr 0.003] [--ema_tp 3.0] [--don_period 20] [--don_atr 0.002] [--don_tp 3.0]
"""
import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import requests

sys.path.insert(0, ".")

from strategies.trend_sub_strategies import (
    evaluate_ema_trend,
    evaluate_donchian,
    rank_momentum,
    rolling_return,
    _ema,
    _atr,
)

DEFAULT_SYMBOLS = [
    "BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "XRP-USDT-SWAP",
    "DOGE-USDT-SWAP", "ADA-USDT-SWAP", "AVAX-USDT-SWAP", "NEAR-USDT-SWAP",
    "LINK-USDT-SWAP", "UNI-USDT-SWAP", "ATOM-USDT-SWAP", "DOT-USDT-SWAP",
    "LTC-USDT-SWAP", "BCH-USDT-SWAP", "ETC-USDT-SWAP", "FIL-USDT-SWAP",
    "APT-USDT-SWAP", "ARB-USDT-SWAP", "OP-USDT-SWAP", "SUI-USDT-SWAP",
    "SEI-USDT-SWAP", "TIA-USDT-SWAP", "INJ-USDT-SWAP", "PEPE-USDT-SWAP",
    "WIF-USDT-SWAP", "AAVE-USDT-SWAP", "TRX-USDT-SWAP", "ENA-USDT-SWAP",
]
PROXY = "http://127.0.0.1:7897"


# ------------------------------------------------------------
# 数据拉取（分页，突破 OKX 单次 limit=300 上限）
# ------------------------------------------------------------
def fetch_klines(symbol: str, bar: str, limit: int):
    proxies = {"http": PROXY, "https": PROXY}
    rows = []  # 倒序累积（最新在前）
    after = ""
    while len(rows) < limit:
        page_limit = min(300, limit - len(rows))
        url = (f"https://www.okx.com/api/v5/market/history-candles"
               f"?instId={symbol}&bar={bar}&limit={page_limit}")
        if after:
            url += f"&after={after}"
        try:
            r = requests.get(url, proxies=proxies, timeout=20)
        except Exception:
            r = requests.get(url, timeout=20)
        data = r.json()
        if data.get("code") != "0":
            raise RuntimeError(f"OKX error for {symbol}: {data}")
        page = data["data"]
        if not page:
            break
        earliest_ts = page[-1][0]
        rows.extend(page)
        if earliest_ts == after:
            break  # 没有更早的数据
        after = earliest_ts

    # 倒序 → 正序，并按时间戳去重
    seen = set()
    ordered = []
    for row in reversed(rows):
        ts = row[0]
        if ts in seen:
            continue
        seen.add(ts)
        ordered.append(row)

    ts = np.array([int(x[0]) for x in ordered])
    closes = np.array([float(x[4]) for x in ordered])
    highs = np.array([float(x[2]) for x in ordered])
    lows = np.array([float(x[3]) for x in ordered])
    volumes = np.array([float(x[5]) for x in ordered])
    return ts, closes, highs, lows, volumes


# ------------------------------------------------------------
# 策略 E 内联复刻（与策略类 _evaluate 保持一致）
# ------------------------------------------------------------
def evaluate_bollinger(closes, highs, lows, boll_period=20, boll_std_mult=2.0,
                       trend_ema_period=50, trend_lookback=10, atr_sl_mult=2.0):
    if len(closes) < boll_period + 1:
        return None
    closes = np.asarray(closes, dtype=float)
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)

    mid = float(np.mean(closes[-boll_period:]))
    std = float(np.std(closes[-boll_period:]))
    if std <= 0:
        return None
    upper = mid + boll_std_mult * std
    lower = mid - boll_std_mult * std
    price = float(closes[-1])

    atr = _atr(highs, lows, closes, 14)
    atr_pct = (atr / price) if price > 0 and atr > 0 else 0.0

    ema_now = _ema(closes, trend_ema_period)
    ema_prev = None
    if len(closes) > trend_ema_period + trend_lookback:
        ema_prev = _ema(closes[:-trend_lookback], trend_ema_period)

    trend = "neutral"
    if ema_now is not None and ema_prev is not None and ema_prev > 0:
        slope = (ema_now - ema_prev) / ema_prev
        trend = "up" if slope > 0 else ("down" if slope < 0 else "neutral")

    direction = None
    if trend == "up" and price <= lower:
        direction = "long"
    elif trend == "down" and price >= upper:
        direction = "short"

    sl_offset = atr * atr_sl_mult if atr > 0 else 0.0
    stop_loss = (price - sl_offset) if direction == "long" else (price + sl_offset) if direction == "short" else None

    return {"signal": direction, "trend": trend, "atr_pct": atr_pct,
            "mid": mid, "upper": upper, "lower": lower,
            "stop_loss": stop_loss, "take_profit": mid}


# ------------------------------------------------------------
# ATR 止盈止损模拟
# ------------------------------------------------------------
def simulate_trade(direction, entry, sl, tp, i, highs, lows, closes, max_hold):
    """以 entry 入场，逐 bar 检查 SL/TP，返回 (收益率, 出场原因)。"""
    end = min(i + max_hold + 1, len(closes))
    for j in range(i + 1, end):
        hi = float(highs[j])
        lo = float(lows[j])
        if direction == "long":
            if sl and lo <= sl:
                return (sl - entry) / entry, "stop_loss"
            if tp and hi >= tp:
                return (tp - entry) / entry, "take_profit"
        else:
            if sl and hi >= sl:
                return (entry - sl) / entry, "stop_loss"
            if tp and lo <= tp:
                return (entry - tp) / entry, "take_profit"
    exit_px = float(closes[end - 1])
    if direction == "long":
        return (exit_px - entry) / entry, "time_exit"
    return (entry - exit_px) / entry, "time_exit"


# ------------------------------------------------------------
# B. 唐奇安参数网格扫描（预计算 ATR，复用突破入场点）
# ------------------------------------------------------------
def _atr_series(highs, lows, closes, period=14):
    """返回与 _atr 同口径（简单均值）的 ATR 序列，atr[i] = mean(TR[i-period+1..i])。"""
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]))
    atr = np.zeros(n)
    for i in range(period, n):
        atr[i] = float(np.mean(tr[i - period + 1: i + 1]))
    return atr


def sweep_donchian(symbol_klines, forwards, periods, sl_mults, tp_mults, atr_thresholds):
    """对 B 唐奇安做参数网格扫描，输出按平均每笔收益排序的汇总表。"""
    max_fwd = max(forwards)
    # 预计算每个 symbol 的 ATR 序列
    prep = {}
    for sym, (ts, closes, highs, lows, _vols) in symbol_klines.items():
        prep[sym] = (np.asarray(closes, dtype=float), np.asarray(highs, dtype=float),
                     np.asarray(lows, dtype=float), _atr_series(highs, lows, closes, 14))

    results = []
    for period in periods:
        # 该周期下所有 symbol 的突破入场点（信号只依赖 period，与 SL/TP 无关）
        entries = []  # (i, side, entry, atr, highs, lows, closes)
        for sym, (closes, highs, lows, atr) in prep.items():
            n = len(closes)
            for i in range(period + 1, n - max_fwd):
                upper = float(np.max(highs[i - period: i]))
                lower = float(np.min(lows[i - period: i]))
                c = float(closes[i])
                if upper > 0 and c > upper:
                    entries.append((i, "long", c, float(atr[i]), highs, lows, closes))
                elif lower > 0 and c < lower:
                    entries.append((i, "short", c, float(atr[i]), highs, lows, closes))
        if not entries:
            continue

        for sl_mult in sl_mults:
            for tp_mult in tp_mults:
                for atr_thr in atr_thresholds:
                    pnls = []
                    for i, side, entry, atr_val, highs, lows, closes in entries:
                        if atr_val <= 0:
                            continue
                        if atr_thr > 0 and (atr_val / entry) < atr_thr:
                            continue
                        if side == "long":
                            sl = entry - sl_mult * atr_val
                            tp = entry + tp_mult * atr_val
                        else:
                            sl = entry + sl_mult * atr_val
                            tp = entry - tp_mult * atr_val
                        ret, _ = simulate_trade(side, entry, sl, tp, i, highs, lows, closes, max_fwd)
                        pnls.append(ret)
                    if len(pnls) < 50:
                        continue
                    wins = [p for p in pnls if p > 0]
                    losses = [p for p in pnls if p <= 0]
                    avg_win = np.mean(wins) if wins else 0.0
                    avg_loss = np.mean(losses) if losses else 0.0
                    ratio = (avg_win / abs(avg_loss)) if avg_loss else float("inf")
                    results.append({
                        "period": period, "sl": sl_mult, "tp": tp_mult, "atr_thr": atr_thr,
                        "n": len(pnls), "wr": len(wins) / len(pnls) * 100,
                        "avg": np.mean(pnls) * 100, "ratio": ratio,
                    })

    results.sort(key=lambda r: r["avg"], reverse=True)
    print("\n" + "=" * 78)
    print("B_唐奇安 参数网格扫描（按平均每笔收益降序，样本 ≥50 笔）")
    print("=" * 78)
    print(f"{'period':>7} {'SL':>5} {'TP':>5} {'ATR%':>6} {'笔数':>7} "
          f"{'胜率':>7} {'平均每笔':>9} {'盈亏比':>7}")
    for r in results[:30]:
        print(f"{r['period']:>7} {r['sl']:>5.1f} {r['tp']:>5.1f} {r['atr_thr']:>6.3f} "
              f"{r['n']:>7} {r['wr']:>6.1f}% {r['avg']:>+8.3f}% {r['ratio']:>7.2f}")
    pos = [r for r in results if r["avg"] > 0]
    print("-" * 78)
    print(f"共 {len(results)} 组参数（样本≥50笔），其中正期望 {len(pos)} 组。")
    if pos:
        best = pos[0]
        print(f"最优正期望：period={best['period']} SL={best['sl']} TP={best['tp']} "
              f"ATR%={best['atr_thr']}  →  {best['n']}笔 / 胜率{best['wr']:.1f}% / "
              f"平均每笔{best['avg']:+.3f}% / 盈亏比{best['ratio']:.2f}")
    else:
        print("结论：在当前网格范围内未找到正期望参数组合，建议弃用 B 唐奇安。")


# ------------------------------------------------------------
# 组合回测 + 日线稳健性验证
# ------------------------------------------------------------
BAR_HOURS = {"1H": 1, "4H": 4, "1D": 24}

DON_OPT = {"donchian_period": 70, "atr_sl_multiplier": 3.0, "atr_tp_multiplier": 8.0}


def fetch_avg_funding_rate(symbol, limit=100):
    """拉取品种近期历史资金费率均值（每 8h 一次）。失败返回 None。"""
    proxies = {"http": PROXY, "https": PROXY}
    url = (f"https://www.okx.com/api/v5/public/funding-rate-history"
           f"?instId={symbol}&limit={min(limit, 100)}")
    try:
        r = requests.get(url, proxies=proxies, timeout=20)
    except Exception:
        r = requests.get(url, timeout=20)
    try:
        data = r.json()
    except Exception:
        return None
    if data.get("code") != "0":
        return None
    rates = [float(x["fundingRate"]) for x in data.get("data", []) if x.get("fundingRate")]
    if not rates:
        return None
    return float(np.mean(rates))


def simulate_trade_exit(direction, entry, sl, tp, i, highs, lows, closes, max_hold):
    """同 simulate_trade，但额外返回出场 bar 下标（用于持仓时长/资金费率）。"""
    end = min(i + max_hold + 1, len(closes))
    for j in range(i + 1, end):
        hi = float(highs[j])
        lo = float(lows[j])
        if direction == "long":
            if sl and lo <= sl:
                return (sl - entry) / entry, "stop_loss", j
            if tp and hi >= tp:
                return (tp - entry) / entry, "take_profit", j
        else:
            if sl and hi >= sl:
                return (entry - sl) / entry, "stop_loss", j
            if tp and lo <= tp:
                return (entry - tp) / entry, "take_profit", j
    j = end - 1
    exit_px = float(closes[j])
    if direction == "long":
        return (exit_px - entry) / entry, "time_exit", j
    return (entry - exit_px) / entry, "time_exit", j


def _funding_cost(direction, avg_funding, hold_bars, bar_hours):
    """持仓期间资金费率成本（收益率口径）。多头支付为正、空头收取为负。"""
    if avg_funding is None:
        return 0.0
    n_funding = hold_bars * bar_hours / 8.0
    sign = 1.0 if direction == "long" else -1.0
    return sign * avg_funding * n_funding


def collect_trades(symbol_klines, forwards, bar, funding_map, don_params=None):
    """收集 B 唐奇安 + E 布林 的逐笔交易（含净值，已扣资金费率）。"""
    don_params = don_params or DON_OPT
    max_fwd = max(forwards)
    bar_hours = BAR_HOURS.get(bar, 24)
    trades = []
    for sym, (ts, closes, highs, lows, _vols) in symbol_klines.items():
        avg_funding = funding_map.get(sym)
        closes = np.asarray(closes, dtype=float)
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)
        n = len(closes)
        for i in range(70, n - max_fwd):
            c = closes[: i + 1]
            h = highs[: i + 1]
            l = lows[: i + 1]
            cur = float(closes[i])

            b = evaluate_donchian(h, l, c, don_params)
            if b.get("signal") in ("long", "short"):
                side = b["signal"]
                ret, reason, exit_idx = simulate_trade_exit(
                    side, cur, b.get("stop_loss"), b.get("take_profit"),
                    i, highs, lows, closes, max_fwd)
                cost = _funding_cost(side, avg_funding, exit_idx - i, bar_hours)
                trades.append({"ts": int(ts[i]), "strategy": "B_唐奇安", "symbol": sym,
                               "side": side, "ret": ret, "ret_net": ret - cost,
                               "hold": exit_idx - i, "reason": reason})

            e = evaluate_bollinger(c, h, l)
            if e and e.get("signal") in ("long", "short") and e.get("atr_pct", 0.0) >= 0.002:
                side = e["signal"]
                ret, reason, exit_idx = simulate_trade_exit(
                    side, cur, e.get("stop_loss"), e.get("take_profit"),
                    i, highs, lows, closes, max_fwd)
                cost = _funding_cost(side, avg_funding, exit_idx - i, bar_hours)
                trades.append({"ts": int(ts[i]), "strategy": "E_布林回归", "symbol": sym,
                               "side": side, "ret": ret, "ret_net": ret - cost,
                               "hold": exit_idx - i, "reason": reason})
    return trades


def collect_momentum_trades(symbol_klines, forwards, lookback=24, top_n=3, bottom_n=2):
    """收集 C 动量轮动的逐 bar 多空对冲收益（横截面）。"""
    symbols = list(symbol_klines.keys())
    ts_map = {s: v[0] for s, v in symbol_klines.items()}
    closes_map = {s: np.asarray(v[1], dtype=float) for s, v in symbol_klines.items()}
    min_len = min(len(c) for c in closes_map.values())
    forward = forwards[0]
    trades = []
    ref_ts = ts_map[symbols[0]]
    for i in range(70, min_len - forward):
        returns = {}
        for s in symbols:
            c = closes_map[s][: i + 1]
            if len(c) < lookback + 1:
                continue
            returns[s] = rolling_return(c, lookback)
        if len(returns) < top_n + bottom_n:
            continue
        ranked = rank_momentum(returns, top_n=top_n, bottom_n=bottom_n)
        longs = [s for s, _ in ranked["long"]]
        shorts = [s for s, _ in ranked["short"]]
        if not longs or not shorts:
            continue

        def fwd_ret(s):
            return float(closes_map[s][i + forward]) / float(closes_map[s][i]) - 1.0

        spread = float(np.mean([fwd_ret(s) for s in longs]) -
                       np.mean([fwd_ret(s) for s in shorts]))
        trades.append({"ts": int(ref_ts[i]), "strategy": "C_动量轮动", "symbol": "组合",
                       "side": "long_short", "ret": spread, "ret_net": spread,
                       "hold": forward, "reason": "time_exit"})
    return trades


def _summary(trades, label):
    pnls = [t["ret_net"] for t in trades]
    if not pnls:
        return f"{label}: 无样本"
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    ratio = (avg_win / abs(avg_loss)) if avg_loss else float("inf")
    return (f"{label}: {len(pnls)}笔 胜率{len(wins)/len(pnls)*100:.1f}% "
            f"平均每笔{np.mean(pnls)*100:+.3f}% 盈亏比{ratio:.2f} "
            f"总收益{np.sum(pnls)*100:+.1f}%")


def run_combo_backtest(symbol_klines, forwards, bar, funding_map):
    """B+E+C 组合回测：分策略统计 + 等权组合权益曲线（2% 仓位）。"""
    b_e = collect_trades(symbol_klines, forwards, bar, funding_map)
    c = collect_momentum_trades(symbol_klines, forwards)
    all_trades = sorted(b_e + c, key=lambda t: t["ts"])

    print("\n" + "=" * 78)
    print(f"B+E+C 组合回测（{bar}，已扣资金费率）")
    print("=" * 78)
    for strat in ("B_唐奇安", "E_布林回归", "C_动量轮动"):
        sub = [t for t in all_trades if t["strategy"] == strat]
        print("  " + _summary(sub, strat))

    w = 0.02
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    combo_rets = []
    for t in all_trades:
        r = w * t["ret_net"]
        combo_rets.append(r)
        equity *= (1.0 + r)
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)
    mean_r = float(np.mean(combo_rets)) if combo_rets else 0.0
    std_r = float(np.std(combo_rets)) if combo_rets else 0.0
    sharpe = mean_r / std_r if std_r > 0 else 0.0
    print("-" * 78)
    print(f"  组合合计：{len(all_trades)}笔  总收益{equity - 1.0:+.2%}  "
          f"最大回撤{max_dd:.2%}  每笔夏普{sharpe:.2f}")


def validate_daily_donchian(symbol_klines, forwards, funding_map):
    """日线 B 唐奇安稳健性验证：扣费率前后对比 + 按年份/币种分段。"""
    trades = [t for t in collect_trades(symbol_klines, forwards, "1D", funding_map)
              if t["strategy"] == "B_唐奇安"]
    if not trades:
        print("无日线 B 唐奇安样本")
        return

    print("\n" + "=" * 78)
    print("日线 B 唐奇安 稳健性验证（period=70/SL=3/TP=8）")
    print("=" * 78)

    gross = [t["ret"] for t in trades]
    net = [t["ret_net"] for t in trades]
    print(f"  扣费率前：{len(trades)}笔 平均每笔 {np.mean(gross)*100:+.3f}%")
    print(f"  扣费率后：{len(trades)}笔 平均每笔 {np.mean(net)*100:+.3f}%  "
          f"(费率侵蚀 {np.mean(gross)*100 - np.mean(net)*100:.3f}%/笔)")

    year_pnl = defaultdict(list)
    for t in trades:
        year_pnl[_year(t["ts"])].append(t["ret_net"])
    print("\n  按年份分段（净收益）：")
    for y in sorted(year_pnl):
        p = year_pnl[y]
        w = sum(1 for x in p if x > 0)
        print(f"    {y}: {len(p):>5}笔  胜率{w/len(p)*100:5.1f}%  "
              f"平均每笔{np.mean(p)*100:+7.3f}%  累计{np.sum(p)*100:+8.1f}%")

    sym_pnl = defaultdict(list)
    for t in trades:
        sym_pnl[t["symbol"]].append(t["ret_net"])
    total_net = np.sum(net)
    ranked = sorted(sym_pnl.items(), key=lambda kv: np.sum(kv[1]), reverse=True)
    print("\n  按币种分组（净收益，Top10）：")
    for sym, p in ranked[:10]:
        w = sum(1 for x in p if x > 0)
        print(f"    {sym:<16} {len(p):>4}笔  胜率{w/len(p)*100:5.1f}%  "
              f"累计{np.sum(p)*100:+8.1f}%  占比{np.sum(p)/total_net*100:5.1f}%")
    neg = [s for s, p in ranked if np.sum(p) < 0]
    print(f"\n  共 {len(sym_pnl)} 个币种，其中累计亏损 {len(neg)} 个；"
          f"Top3 币种贡献 {np.sum([np.sum(p) for _, p in ranked[:3]])/total_net*100:.1f}% 净收益。")


def validate_daily_donchian_segments(symbol_klines, forwards, funding_map):
    """日线 B 唐奇安 深化稳健性验证：按季度 + 按 ADX 趋势/震荡环境 分段。"""
    don_params = DON_OPT
    max_fwd = max(forwards)
    bar_hours = BAR_HOURS["1D"]
    trades = []
    for sym, (ts, closes, highs, lows, _vols) in symbol_klines.items():
        avg_funding = funding_map.get(sym)
        closes = np.asarray(closes, dtype=float)
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)
        adx = _adx_series(highs, lows, closes, 14)
        n = len(closes)
        for i in range(70, n - max_fwd):
            b = evaluate_donchian(highs[: i + 1], lows[: i + 1], closes[: i + 1], don_params)
            if b.get("signal") not in ("long", "short"):
                continue
            side = b["signal"]
            ret, _reason, exit_idx = simulate_trade_exit(
                side, float(closes[i]), b.get("stop_loss"), b.get("take_profit"),
                i, highs, lows, closes, max_fwd)
            cost = _funding_cost(side, avg_funding, exit_idx - i, bar_hours)
            trades.append({"ts": int(ts[i]), "symbol": sym, "side": side,
                           "ret_net": ret - cost, "adx": float(adx[i]),
                           "quarter": _quarter(int(ts[i]))})

    if not trades:
        print("无日线 B 唐奇安分段样本")
        return

    print("\n" + "=" * 78)
    print("日线 B 唐奇安 深化分段验证（按季度 + 按 ADX 环境）")
    print("=" * 78)

    def _line(p):
        w = sum(1 for x in p if x > 0)
        return (f"{len(p):>5}笔  胜率{w/len(p)*100:5.1f}%  "
                f"平均每笔{np.mean(p)*100:+7.3f}%  累计{np.sum(p)*100:+8.1f}%")

    # 按季度分段
    q_pnl = defaultdict(list)
    for t in trades:
        q_pnl[t["quarter"]].append(t["ret_net"])
    print("\n  按季度分段（净收益）：")
    for q in sorted(q_pnl):
        print(f"    {q}: " + _line(q_pnl[q]))

    # 按 ADX 环境分段
    buckets = [("震荡 ADX<20", 0, 20), ("过渡 20≤ADX<25", 20, 25),
               ("趋势 ADX≥25", 25, 1e9)]
    print("\n  按入场 ADX 环境分段（净收益）：")
    for label, lo, hi in buckets:
        p = [t["ret_net"] for t in trades if lo <= t["adx"] < hi]
        if not p:
            print(f"    {label}: 无样本")
            continue
        print(f"    {label:<16}: " + _line(p))

    # 关键交叉：2025 年（亏损年）内，趋势 vs 震荡环境
    print("\n  2025 年内部（亏损年）按 ADX 环境拆分：")
    for label, lo, hi in buckets:
        p = [t["ret_net"] for t in trades
             if _year(t["ts"]) == 2025 and lo <= t["adx"] < hi]
        if not p:
            print(f"    {label}: 无样本")
            continue
        print(f"    {label:<16}: " + _line(p))

    # 若只在 ADX≥25 趋势环境入场，整体表现
    trend_p = [t["ret_net"] for t in trades if t["adx"] >= 25]
    range_p = [t["ret_net"] for t in trades if t["adx"] < 20]
    print("\n  对比：仅 ADX≥25 趋势入场 vs 仅 ADX<20 震荡入场")
    print(f"    趋势入场(ADX≥25): " + _line(trend_p))
    print(f"    震荡入场(ADX<20): " + _line(range_p))


def benchmark_adx_filter(symbol_klines, forwards, funding_map):
    """量化 ADX 上限过滤器对 B 唐奇安及 B+E+C 组合的改善。"""
    don_params = DON_OPT
    max_fwd = max(forwards)
    bar_hours = BAR_HOURS["1D"]

    # 独立收集 B 唐奇安（带入场 ADX，不在此过滤）
    b_trades = []
    for sym, (ts, closes, highs, lows, _vols) in symbol_klines.items():
        avg_funding = funding_map.get(sym)
        closes = np.asarray(closes, dtype=float)
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)
        adx = _adx_series(highs, lows, closes, 14)
        n = len(closes)
        for i in range(70, n - max_fwd):
            b = evaluate_donchian(highs[: i + 1], lows[: i + 1], closes[: i + 1], don_params)
            if b.get("signal") not in ("long", "short"):
                continue
            side = b["signal"]
            ret, _reason, exit_idx = simulate_trade_exit(
                side, float(closes[i]), b.get("stop_loss"), b.get("take_profit"),
                i, highs, lows, closes, max_fwd)
            cost = _funding_cost(side, avg_funding, exit_idx - i, bar_hours)
            b_trades.append({"ts": int(ts[i]), "strategy": "B_唐奇安", "symbol": sym,
                             "side": side, "ret": ret, "ret_net": ret - cost,
                             "hold": exit_idx - i, "reason": _reason, "adx": float(adx[i])})

    # 复用现有逻辑收集 E 布林回归（只取 E）与 C 动量轮动
    e_trades = [t for t in collect_trades(symbol_klines, forwards, "1D", funding_map)
                if t["strategy"] == "E_布林回归"]
    c_trades = collect_momentum_trades(symbol_klines, forwards)

    def _combo_metrics(trades):
        w = 0.02
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        rets = []
        for t in trades:
            r = w * t["ret_net"]
            rets.append(r)
            equity *= (1.0 + r)
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak)
        mean_r = float(np.mean(rets)) if rets else 0.0
        std_r = float(np.std(rets)) if rets else 0.0
        sharpe = mean_r / std_r if std_r > 0 else 0.0
        return equity - 1.0, max_dd, sharpe, len(rets)

    print("\n" + "=" * 78)
    print("B 唐奇安 ADX 上限过滤器 对比（1D，已扣资金费率）")
    print("=" * 78)
    print(f"{'方案':<14} {'B笔数':>6} {'B胜率':>7} {'B平均每笔':>9} "
          f"{'组合收益':>9} {'组合回撤':>9} {'每笔夏普':>8}")

    for label, filt in [("无过滤", None), ("ADX<25", lambda a: a < 25),
                        ("ADX<20", lambda a: a < 20)]:
        sub = [t for t in b_trades if filt is None or filt(t["adx"])]
        if not sub:
            print(f"{label:<14}  无样本")
            continue
        pnls = [t["ret_net"] for t in sub]
        wins = [p for p in pnls if p > 0]
        wr = len(wins) / len(pnls) * 100
        avg = np.mean(pnls) * 100
        combo = sorted(sub + e_trades + c_trades, key=lambda t: t["ts"])
        gain, mdd, sharpe, _n = _combo_metrics(combo)
        print(f"{label:<14} {len(sub):>6} {wr:>6.1f}% {avg:>+8.3f}% "
              f"{gain:>+8.2%} {mdd:>8.2%} {sharpe:>8.2f}")

    # 各方案下 B 唐奇安单策略的年分段（验证是否消除 2025 亏损）
    print("\n  各方案 B 唐奇安 按年份净收益累计：")
    for label, filt in [("无过滤", None), ("ADX<25", lambda a: a < 25),
                        ("ADX<20", lambda a: a < 20)]:
        sub = [t for t in b_trades if filt is None or filt(t["adx"])]
        year_p = defaultdict(list)
        for t in sub:
            year_p[_year(t["ts"])].append(t["ret_net"])
        seg = "  ".join(f"{y}:{np.sum(p)*100:+.0f}%" for y, p in sorted(year_p.items()))
        print(f"    {label:<8} {seg}")


def benchmark_symbol_concentration(symbol_klines, forwards, funding_map):
    """验证单币种同时持仓数上限对 B(ADX<25)+E+C 收益集中度与稳健性的影响。"""
    don_params = DON_OPT
    max_fwd = max(forwards)
    bar_hours = BAR_HOURS["1D"]

    b_trades = []
    for sym, (ts, closes, highs, lows, _vols) in symbol_klines.items():
        avg_funding = funding_map.get(sym)
        closes = np.asarray(closes, dtype=float)
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)
        adx = _adx_series(highs, lows, closes, 14)
        n = len(closes)
        for i in range(70, n - max_fwd):
            b = evaluate_donchian(highs[: i + 1], lows[: i + 1], closes[: i + 1], don_params)
            if b.get("signal") not in ("long", "short"):
                continue
            if float(adx[i]) >= 25:  # 仅保留 ADX<25 的突破（上限过滤）
                continue
            side = b["signal"]
            ret, _reason, exit_idx = simulate_trade_exit(
                side, float(closes[i]), b.get("stop_loss"), b.get("take_profit"),
                i, highs, lows, closes, max_fwd)
            cost = _funding_cost(side, avg_funding, exit_idx - i, bar_hours)
            b_trades.append({"ts": int(ts[i]), "strategy": "B_唐奇安", "symbol": sym,
                             "side": side, "ret": ret, "ret_net": ret - cost,
                             "hold": exit_idx - i})

    e_trades = [t for t in collect_trades(symbol_klines, forwards, "1D", funding_map)
                if t["strategy"] == "E_布林回归"]
    c_trades = collect_momentum_trades(symbol_klines, forwards)
    all_trades = sorted(b_trades + e_trades + c_trades, key=lambda t: t["ts"])

    def _run(cap):
        w = 0.02
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        rets = []
        active = defaultdict(list)
        sym_contrib = defaultdict(float)
        taken = 0
        for t in all_trades:
            ts = t["ts"]
            sym = t["symbol"]
            is_combo = (sym == "组合")
            if not is_combo:
                active[sym] = [e for e in active[sym] if e > ts]
                if cap is not None and len(active[sym]) >= cap:
                    continue
            r = w * t["ret_net"]
            rets.append(r)
            equity *= (1.0 + r)
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak)
            sym_contrib[sym] += r
            taken += 1
            if not is_combo:
                active[sym].append(ts + t["hold"] * 86400_000)
        mean_r = float(np.mean(rets)) if rets else 0.0
        std_r = float(np.std(rets)) if rets else 0.0
        sharpe = mean_r / std_r if std_r > 0 else 0.0
        total = sum(sym_contrib.values())
        ranked = sorted(sym_contrib.values(), reverse=True)
        top1 = ranked[0] / total if (total and ranked) else 0.0
        top3 = sum(ranked[:3]) / total if (total and ranked) else 0.0
        return equity - 1.0, max_dd, sharpe, taken, top1, top3

    print("\n" + "=" * 78)
    print("B(ADX<25)+E+C 单币种同时持仓数上限 对比（1D，已扣资金费率）")
    print("=" * 78)
    print(f"{'单币种上限':<12} {'笔数':>6} {'组合收益':>9} {'组合回撤':>9} "
          f"{'每笔夏普':>8} {'Top1集中':>9} {'Top3集中':>9}")
    for cap in (None, 3, 2, 1):
        gain, mdd, sharpe, taken, top1, top3 = _run(cap)
        label = "无限" if cap is None else str(cap)
        print(f"{label:<12} {taken:>6} {gain:>+8.2%} {mdd:>8.2%} "
              f"{sharpe:>8.2f} {top1:>8.1%} {top3:>8.1%}")


def benchmark_symbol_cap(symbol_klines, forwards, funding_map):
    """验证单币种收益贡献 cap（占组合累计收益比例）对集中度与稳健性的影响。"""
    don_params = DON_OPT
    max_fwd = max(forwards)
    bar_hours = BAR_HOURS["1D"]

    b_trades = []
    for sym, (ts, closes, highs, lows, _vols) in symbol_klines.items():
        avg_funding = funding_map.get(sym)
        closes = np.asarray(closes, dtype=float)
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)
        adx = _adx_series(highs, lows, closes, 14)
        n = len(closes)
        for i in range(70, n - max_fwd):
            b = evaluate_donchian(highs[: i + 1], lows[: i + 1], closes[: i + 1], don_params)
            if b.get("signal") not in ("long", "short"):
                continue
            if float(adx[i]) >= 25:
                continue
            side = b["signal"]
            ret, _reason, exit_idx = simulate_trade_exit(
                side, float(closes[i]), b.get("stop_loss"), b.get("take_profit"),
                i, highs, lows, closes, max_fwd)
            cost = _funding_cost(side, avg_funding, exit_idx - i, bar_hours)
            b_trades.append({"ts": int(ts[i]), "strategy": "B_唐奇安", "symbol": sym,
                             "side": side, "ret": ret, "ret_net": ret - cost,
                             "hold": exit_idx - i})

    e_trades = [t for t in collect_trades(symbol_klines, forwards, "1D", funding_map)
                if t["strategy"] == "E_布林回归"]
    c_trades = collect_momentum_trades(symbol_klines, forwards)
    all_trades = sorted(b_trades + e_trades + c_trades, key=lambda t: t["ts"])

    def _run(cap_frac):
        w = 0.02
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        rets = []
        sym_contrib = defaultdict(float)
        running_total = 0.0
        taken = 0
        for t in all_trades:
            sym = t["symbol"]
            r = w * t["ret_net"]
            if cap_frac is not None and running_total > 0:
                if sym_contrib[sym] / running_total >= cap_frac:
                    continue
            rets.append(r)
            equity *= (1.0 + r)
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak)
            sym_contrib[sym] += r
            running_total += r
            taken += 1
        mean_r = float(np.mean(rets)) if rets else 0.0
        std_r = float(np.std(rets)) if rets else 0.0
        sharpe = mean_r / std_r if std_r > 0 else 0.0
        total = sum(sym_contrib.values())
        ranked = sorted(sym_contrib.values(), reverse=True)
        top1 = ranked[0] / total if (total and ranked) else 0.0
        top3 = sum(ranked[:3]) / total if (total and ranked) else 0.0
        return equity - 1.0, max_dd, sharpe, taken, top1, top3

    print("\n" + "=" * 78)
    print("B(ADX<25)+E+C 单币种收益贡献 cap 对比（1D，已扣资金费率）")
    print("=" * 78)
    print(f"{'贡献cap':<12} {'笔数':>6} {'组合收益':>9} {'组合回撤':>9} "
          f"{'每笔夏普':>8} {'Top1集中':>9} {'Top3集中':>9}")
    for cap_frac in (None, 0.20, 0.15, 0.10):
        gain, mdd, sharpe, taken, top1, top3 = _run(cap_frac)
        label = "无上限" if cap_frac is None else f"{cap_frac*100:.0f}%"
        print(f"{label:<12} {taken:>6} {gain:>+8.2%} {mdd:>8.2%} "
              f"{sharpe:>8.2f} {top1:>8.1%} {top3:>8.1%}")


def _year(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).year


def _quarter(ts_ms):
    d = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def _adx_series(highs, lows, closes, period=14):
    """标准 Wilder ADX 序列，adx[i] 对应第 i 根 K 线（前 2*period 根为 0）。"""
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]))
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0

    def _wilder(x):
        s = np.zeros(n)
        if n <= period:
            return s
        s[period] = float(np.sum(x[1:period + 1]))
        for i in range(period + 1, n):
            s[i] = s[i - 1] - s[i - 1] / period + x[i]
        return s

    atr = _wilder(tr)
    s_plus = _wilder(plus_dm)
    s_minus = _wilder(minus_dm)

    dx = np.zeros(n)
    for i in range(period, n):
        if atr[i] <= 0:
            continue
        pdi = 100.0 * s_plus[i] / atr[i]
        mdi = 100.0 * s_minus[i] / atr[i]
        denom = pdi + mdi
        dx[i] = 100.0 * abs(pdi - mdi) / denom if denom > 0 else 0.0

    adx = np.zeros(n)
    start = 2 * period
    if n > start:
        adx[start] = float(np.mean(dx[period + 1:start + 1]))
        for i in range(start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx


# ------------------------------------------------------------
# 回测引擎
# ------------------------------------------------------------
class NewStrategyBacktester:
    def __init__(self, forwards: List[int], ema_atr=0.003, ema_tp=3.0,
                 don_period=20, don_atr=0.002, don_tp=3.0):
        self.forwards = forwards
        self.ema_atr = ema_atr
        self.ema_tp = ema_tp
        self.don_period = don_period
        self.don_atr = don_atr
        self.don_tp = don_tp
        self.stats: Dict[str, Dict[str, Dict[int, List[int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: [0, 0]))
        )
        self.pnls: Dict[str, List[float]] = defaultdict(list)
        self.max_hold = max(forwards)

    def _record_dir(self, name, side, correct, forward):
        cell = self.stats[name][side][forward]
        cell[1] += 1
        if correct:
            cell[0] += 1

    def _record_pnl(self, name, ret):
        self.pnls[name].append(ret)

    def run_symbol(self, closes, highs, lows):
        n = len(closes)
        max_fwd = max(self.forwards)
        for i in range(70, n - max_fwd):
            c = closes[: i + 1]
            h = highs[: i + 1]
            l = lows[: i + 1]
            cur = float(closes[i])

            # A. EMA 趋势
            a = evaluate_ema_trend(c, h, l, {"ema_fast": 9, "ema_slow": 21,
                                             "atr_pct_threshold": self.ema_atr,
                                             "atr_tp_mult": self.ema_tp})
            if a.get("signal") in ("long", "short"):
                side = a["signal"]
                for fwd in self.forwards:
                    fwd_ret = float(closes[i + fwd]) / cur - 1.0
                    self._record_dir("A_EMA趋势", side,
                                     (side == "long" and fwd_ret > 0) or (side == "short" and fwd_ret < 0), fwd)
                ret, _ = simulate_trade(side, cur, a.get("stop_loss"), a.get("take_profit"),
                                        i, highs, lows, closes, self.max_hold)
                self._record_pnl("A_EMA趋势", ret)

            # B. 唐奇安突破（含趋势判定：ATR% 过低暂停）
            b = evaluate_donchian(h, l, c, {"donchian_period": self.don_period,
                                            "atr_tp_multiplier": self.don_tp})
            if b.get("signal") in ("long", "short"):
                atr = b.get("atr", 0.0)
                atr_pct = (atr / cur) if cur > 0 and atr > 0 else 0.0
                if atr_pct < self.don_atr:
                    continue
                side = b["signal"]
                for fwd in self.forwards:
                    fwd_ret = float(closes[i + fwd]) / cur - 1.0
                    self._record_dir("B_唐奇安", side,
                                     (side == "long" and fwd_ret > 0) or (side == "short" and fwd_ret < 0), fwd)
                ret, _ = simulate_trade(side, cur, b.get("stop_loss"), b.get("take_profit"),
                                        i, highs, lows, closes, self.max_hold)
                self._record_pnl("B_唐奇安", ret)

            # E. 布林均值回归（带趋势过滤）
            e = evaluate_bollinger(c, h, l)
            if e and e.get("signal") in ("long", "short") and e.get("atr_pct", 0.0) >= 0.002:
                side = e["signal"]
                for fwd in self.forwards:
                    fwd_ret = float(closes[i + fwd]) / cur - 1.0
                    self._record_dir("E_布林回归", side,
                                     (side == "long" and fwd_ret > 0) or (side == "short" and fwd_ret < 0), fwd)
                ret, _ = simulate_trade(side, cur, e.get("stop_loss"), e.get("take_profit"),
                                        i, highs, lows, closes, self.max_hold)
                self._record_pnl("E_布林回归", ret)

    def report(self):
        print("\n" + "=" * 78)
        print("新策略体系信号级回测结果（方向命中率 + ATR 止盈止损模拟）")
        print("=" * 78)
        for name in ("A_EMA趋势", "B_唐奇安", "E_布林回归"):
            print(f"\n【{name}】")
            for side in ("long", "short"):
                for fwd in self.forwards:
                    hit, tot = self.stats[name][side][fwd]
                    if tot == 0:
                        continue
                    wr = hit / tot * 100
                    print(f"  方向 {side:<6} fwd={fwd:>2}根  {wr:5.1f}%  ({hit}/{tot})")

            pnls = self.pnls.get(name, [])
            if not pnls:
                print("  TP/SL模拟：无样本")
                continue
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            avg_win = np.mean(wins) if wins else 0.0
            avg_loss = np.mean(losses) if losses else 0.0
            ratio = (avg_win / abs(avg_loss)) if avg_loss else float("inf")
            print(f"  TP/SL模拟：{len(pnls)}笔  胜率 {len(wins)/len(pnls)*100:.1f}%  "
                  f"平均每笔 {np.mean(pnls)*100:+.3f}%  盈亏比 {ratio:.2f}  "
                  f"(avg_win {avg_win*100:.3f}% / avg_loss {avg_loss*100:.3f}%)")
        print("\n" + "=" * 78)


# ------------------------------------------------------------
# C. 多币种动量轮动（横截面）
# ------------------------------------------------------------
def run_momentum_rotation(symbol_klines: Dict[str, Tuple], lookback: int,
                          top_n: int, bottom_n: int, forward: int):
    symbols = list(symbol_klines.keys())
    if not symbols:
        return
    closes_map = {s: v[1] for s, v in symbol_klines.items()}
    min_len = min(len(c) for c in closes_map.values())
    win_spread, win_count = 0.0, 0

    for i in range(70, min_len - forward):
        returns = {}
        for s in symbols:
            c = closes_map[s][: i + 1]
            if len(c) < lookback + 1:
                continue
            returns[s] = rolling_return(c, lookback)
        if len(returns) < top_n + bottom_n:
            continue
        ranked = rank_momentum(returns, top_n=top_n, bottom_n=bottom_n)
        longs = [s for s, _ in ranked["long"]]
        shorts = [s for s, _ in ranked["short"]]

        def fwd_ret(s):
            return float(closes_map[s][i + forward]) / float(closes_map[s][i]) - 1.0

        long_avg = np.mean([fwd_ret(s) for s in longs]) if longs else 0.0
        short_avg = np.mean([fwd_ret(s) for s in shorts]) if shorts else 0.0
        win_spread += (long_avg - short_avg)
        win_count += 1

    print("\n【C_动量轮动（横截面）】")
    if win_count == 0:
        print("  样本不足")
        return
    avg_spread = win_spread / win_count
    print(f"  fwd={forward:>2}根  强势组-弱势组平均收益差 = {avg_spread * 100:+.3f}%  "
          f"(样本 {win_count})  {'✓ 动量溢价为正' if avg_spread > 0 else '✗ 无动量溢价'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--bar", type=str, default="1H")
    ap.add_argument("--forward", type=str, default="4,8,24")
    ap.add_argument("--ema_atr", type=float, default=0.003)
    ap.add_argument("--ema_tp", type=float, default=3.0)
    ap.add_argument("--don_period", type=int, default=20)
    ap.add_argument("--don_atr", type=float, default=0.002)
    ap.add_argument("--don_tp", type=float, default=3.0)
    ap.add_argument("--sweep-b", action="store_true",
                    help="仅对 B 唐奇安做参数网格扫描（period/SL/TP/ATR% 阈值）")
    ap.add_argument("--combo", action="store_true",
                    help="B+E+C 组合回测（--bar 1D 时附带日线稳健性验证）")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    forwards = [int(x) for x in args.forward.split(",") if x.strip()]

    # B 唐奇安参数扫描模式：先拉全量数据，再扫参数，不跑 A/E/C
    if args.sweep_b:
        symbol_klines: Dict[str, Tuple] = {}
        for sym in symbols:
            try:
                ts, closes, highs, lows, volumes = fetch_klines(sym, args.bar, args.limit)
                if len(closes) < 90:
                    print(f"[跳过] {sym}: 数据不足 ({len(closes)} 根)")
                    continue
                symbol_klines[sym] = (ts, closes, highs, lows, volumes)
                print(f"[完成] {sym}: {len(closes)} 根 {args.bar} K线")
            except Exception as e:
                print(f"[失败] {sym}: {e}")
        sweep_donchian(symbol_klines, forwards,
                       periods=[10, 20, 40, 55, 70],
                       sl_mults=[1.0, 1.5, 2.0, 3.0],
                       tp_mults=[3.0, 4.0, 6.0, 8.0],
                       atr_thresholds=[0.0, 0.002, 0.004])
        return

    # 组合回测模式：B+E+C（1D 时附带日线稳健性验证）
    if args.combo:
        funding_map = {}
        for sym in symbols:
            try:
                fr = fetch_avg_funding_rate(sym)
                if fr is not None:
                    funding_map[sym] = fr
            except Exception:
                pass
        if funding_map:
            avg_fr = np.mean(list(funding_map.values()))
            print(f"[资金费率] 已获取 {len(funding_map)}/{len(symbols)} 个币种，均值 {avg_fr*100:.4f}%/8h")

        symbol_klines: Dict[str, Tuple] = {}
        for sym in symbols:
            try:
                ts, closes, highs, lows, volumes = fetch_klines(sym, args.bar, args.limit)
                if len(closes) < 90:
                    print(f"[跳过] {sym}: 数据不足 ({len(closes)} 根)")
                    continue
                symbol_klines[sym] = (ts, closes, highs, lows, volumes)
                print(f"[完成] {sym}: {len(closes)} 根 {args.bar} K线")
            except Exception as e:
                print(f"[失败] {sym}: {e}")
        run_combo_backtest(symbol_klines, forwards, args.bar, funding_map)
        if args.bar == "1D":
            validate_daily_donchian(symbol_klines, forwards, funding_map)
            validate_daily_donchian_segments(symbol_klines, forwards, funding_map)
            benchmark_adx_filter(symbol_klines, forwards, funding_map)
            benchmark_symbol_concentration(symbol_klines, forwards, funding_map)
            benchmark_symbol_cap(symbol_klines, forwards, funding_map)
        return

    bt = NewStrategyBacktester(forwards, ema_atr=args.ema_atr, ema_tp=args.ema_tp,
                               don_period=args.don_period, don_atr=args.don_atr, don_tp=args.don_tp)

    symbol_klines: Dict[str, Tuple] = {}
    for sym in symbols:
        try:
            ts, closes, highs, lows, volumes = fetch_klines(sym, args.bar, args.limit)
            if len(closes) < 90:
                print(f"[跳过] {sym}: 数据不足 ({len(closes)} 根)")
                continue
            bt.run_symbol(closes, highs, lows)
            symbol_klines[sym] = (ts, closes, highs, lows, volumes)
            print(f"[完成] {sym}: {len(closes)} 根 {args.bar} K线")
        except Exception as e:
            print(f"[失败] {sym}: {e}")

    bt.report()
    run_momentum_rotation(symbol_klines, lookback=24, top_n=3, bottom_n=2, forward=forwards[0])


if __name__ == "__main__":
    main()
