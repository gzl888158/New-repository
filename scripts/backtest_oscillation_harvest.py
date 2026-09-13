"""
震荡收割策略（OscillationHarvestStrategy）信号级离线回测

离线环境无法获取实盘 MarketRegimeEngine 的 range_bound 状态，故用 Wilder ADX 作为
震荡代理：ADX < 阈值 视为震荡区间（与 range_bound 语义一致）。默认做多档对比，验证
「仅在震荡市入场」这一核心过滤是否真的有效。

回测逻辑与 strategies/oscillation_harvest_strategy.py 的 _evaluate 保持一致：
  1. 支撑 = 近 lookback 根最低价，阻力 = 近 lookback 根最高价（排除当前未完成 bar）
  2. 区间宽度 < min_band_width_pct 判为磨损型，放弃
  3. RSI 超卖(<oversold)在支撑位做多 / 超买(>overbought)在阻力位做空
  4. ATR 止损（区间外侧）、中轨止盈（mid_tp_ratio 插值）
  5. 逐 bar 检查 SL/TP，扣除双边 taker 手续费（0.1%）

用法：
  .venv/Scripts/python.exe scripts/backtest_oscillation_harvest.py [--bar 1H] [--limit 800]
      [--lookback 48] [--min-band 0.003] [--rsi-oversold 30] [--rsi-overbought 70]
      [--atr-sl 1.5] [--max-hold 24]
"""
import argparse
import sys
from collections import defaultdict

import numpy as np
import requests

sys.path.insert(0, ".")

from strategies.trend_sub_strategies import _atr

PROXY = "http://127.0.0.1:7897"

TIER12_SYMBOLS = [
    "BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "XRP-USDT-SWAP",
    "DOGE-USDT-SWAP", "ADA-USDT-SWAP", "AVAX-USDT-SWAP", "NEAR-USDT-SWAP",
    "APT-USDT-SWAP", "SUI-USDT-SWAP", "ARB-USDT-SWAP", "OP-USDT-SWAP",
]

FEE = 0.001  # 双边 taker 手续费（0.05% x 2）


# ------------------------------------------------------------
# 数据拉取（分页，突破 OKX 单次 limit=300 上限）
# ------------------------------------------------------------
def fetch_klines(symbol: str, bar: str, limit: int):
    proxies = {"http": PROXY, "https": PROXY}
    rows = []
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
            break
        after = earliest_ts

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
# 指标（与策略类口径一致）
# ------------------------------------------------------------
def _rsi(closes, period: int = 14) -> float:
    closes = np.asarray(closes, dtype=float)
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[-period:]))
    avg_loss = float(np.mean(losses[-period:]))
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _adx_series(highs, lows, closes, period=14):
    """Wilder ADX 序列，adx[i] 对应第 i 根 K 线。"""
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


def _rsi_series(closes, period=14):
    """SMA 口径 RSI 序列（与 _rsi 一致）。"""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    for i in range(period, n):
        avg_gain = float(np.mean(gains[i - period:i]))
        avg_loss = float(np.mean(losses[i - period:i]))
        if avg_loss == 0:
            rsi[i] = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _atr_series(highs, lows, closes, period=14):
    """SMA 口径 ATR 序列（与 _atr 一致）。"""
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
        atr[i] = float(np.mean(tr[i - period + 1:i + 1]))
    return atr


def _precompute(symbol_klines, lookback, rsi_period):
    """预计算每个 symbol 的滚动支撑/阻力、RSI、ATR、ADX 序列。"""
    prep = {}
    for sym, (ts, closes, highs, lows, _v) in symbol_klines.items():
        closes = np.asarray(closes, dtype=float)
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)
        n = len(closes)
        support = np.full(n, np.nan)
        resistance = np.full(n, np.nan)
        # 滚动窗口 [i-lookback, i)，排除当前未完成 bar
        for i in range(lookback, n):
            support[i] = float(np.min(lows[i - lookback:i]))
            resistance[i] = float(np.max(highs[i - lookback:i]))
        prep[sym] = {
            "ts": ts, "closes": closes, "highs": highs, "lows": lows,
            "support": support, "resistance": resistance,
            "rsi": _rsi_series(closes, rsi_period),
            "atr": _atr_series(highs, lows, closes, 14),
            "adx": _adx_series(highs, lows, closes, 14),
        }
    return prep


def sweep_params(symbol_klines, lookbacks, mid_tp_ratios, min_bands,
                 adx_thresholds, rsi_oversold, rsi_overbought, rsi_period,
                 atr_sl, max_hold, support_band_pct, min_samples=50):
    """参数网格扫描：按平均每笔净收益降序返回正期望组合。"""
    results = []
    for lookback in lookbacks:
        prep = _precompute(symbol_klines, lookback, rsi_period)
        for mid_tp in mid_tp_ratios:
            for min_band in min_bands:
                for adx_thr in adx_thresholds:
                    net = []
                    for sym, d in prep.items():
                        closes = d["closes"]; highs = d["highs"]; lows = d["lows"]
                        support = d["support"]; resistance = d["resistance"]
                        rsi_s = d["rsi"]; atr_s = d["atr"]; adx_s = d["adx"]
                        n = len(closes)
                        start = lookback + 50
                        for i in range(start, n - max_hold):
                            if adx_thr is not None and float(adx_s[i]) >= adx_thr:
                                continue
                            sup = float(support[i]); res = float(resistance[i])
                            if np.isnan(sup) or res <= sup:
                                continue
                            mid = (sup + res) / 2.0
                            range_pct = (res - sup) / mid if mid > 0 else 0.0
                            if range_pct < min_band:
                                continue
                            price = float(closes[i])
                            rsi = float(rsi_s[i])
                            direction = None
                            if price <= sup * (1.0 + support_band_pct) and rsi < rsi_oversold:
                                direction = "long"
                            elif price >= res * (1.0 - support_band_pct) and rsi > rsi_overbought:
                                direction = "short"
                            if direction is None:
                                continue
                            atr = float(atr_s[i])
                            sl_offset = atr * atr_sl if atr > 0 else sup * support_band_pct
                            if direction == "long":
                                sl = sup - sl_offset
                                tp = sup + (mid - sup) * mid_tp
                            else:
                                sl = res + sl_offset
                                tp = res - (res - mid) * mid_tp
                            ret, _reason, _j = simulate_trade(
                                direction, price, sl, tp, i, highs, lows, closes, max_hold)
                            net.append(ret - FEE)
                    if len(net) < min_samples:
                        continue
                    wins = [p for p in net if p > 0]
                    losses = [p for p in net if p <= 0]
                    avg_win = np.mean(wins) if wins else 0.0
                    avg_loss = np.mean(losses) if losses else 0.0
                    ratio = (avg_win / abs(avg_loss)) if avg_loss else float("inf")
                    results.append({
                        "lookback": lookback, "mid_tp": mid_tp, "min_band": min_band,
                        "adx": adx_thr, "n": len(net), "wr": len(wins) / len(net) * 100,
                        "avg": np.mean(net) * 100, "ratio": ratio, "total": np.sum(net) * 100,
                    })
    results.sort(key=lambda r: r["avg"], reverse=True)
    return results


def print_sweep(results, top_n=30):
    print("\n" + "=" * 92)
    print("震荡收割 参数网格扫描（按平均每笔净收益降序，已扣 0.1% 双边手续费）")
    print("=" * 92)
    print(f"{'lookback':>8} {'TP比':>5} {'带宽':>7} {'ADX':>6} {'笔数':>6} "
          f"{'胜率':>7} {'平均每笔':>9} {'盈亏比':>7} {'累计':>9}")
    for r in results[:top_n]:
        adx_lbl = "无" if r["adx"] is None else f"<{r['adx']}"
        print(f"{r['lookback']:>8} {r['mid_tp']:>5.1f} {r['min_band']:>7.3f} "
              f"{adx_lbl:>6} {r['n']:>6} {r['wr']:>6.1f}% {r['avg']:>+8.3f}% "
              f"{r['ratio']:>7.2f} {r['total']:>+8.2f}%")
    pos = [r for r in results if r["avg"] > 0]
    print("-" * 92)
    print(f"共 {len(results)} 组参数，其中正期望 {len(pos)} 组。")
    if pos:
        b = pos[0]
        adx_lbl = "无" if b["adx"] is None else f"<{b['adx']}"
        print(f"最优：lookback={b['lookback']} TP比={b['mid_tp']} 带宽={b['min_band']} "
              f"ADX={adx_lbl} → {b['n']}笔 / 胜率{b['wr']:.1f}% / "
              f"平均每笔{b['avg']:+.3f}% / 盈亏比{b['ratio']:.2f}")
    else:
        print("结论：当前网格内未找到正期望组合。")


# ------------------------------------------------------------
# 震荡收割信号评估（复刻策略 _evaluate）
# ------------------------------------------------------------
def evaluate_oscillation(closes, highs, lows, params):
    lookback = params["lookback_bars"]
    if len(closes) < lookback + 1:
        return None
    support = float(np.min(lows[-lookback:-1]))
    resistance = float(np.max(highs[-lookback:-1]))
    if resistance <= support:
        return None
    mid = (support + resistance) / 2.0
    range_pct = (resistance - support) / mid if mid > 0 else 0.0
    if range_pct < params["min_band_width_pct"]:
        return None

    price = float(closes[-1])
    rsi = _rsi(closes, params["rsi_period"])
    atr = _atr(highs, lows, closes, 14)

    direction = None
    if price <= support * (1.0 + params["support_band_pct"]) and rsi < params["rsi_oversold"]:
        direction = "long"
    elif price >= resistance * (1.0 - params["support_band_pct"]) and rsi > params["rsi_overbought"]:
        direction = "short"
    if direction is None:
        return {"signal": None, "range_pct": range_pct, "rsi": rsi}

    sl_offset = atr * params["atr_sl_mult"] if atr > 0 else support * params["support_band_pct"]
    if direction == "long":
        stop_loss = support - sl_offset
        take_profit = support + (mid - support) * params["mid_tp_ratio"]
    else:
        stop_loss = resistance + sl_offset
        take_profit = resistance - (resistance - mid) * params["mid_tp_ratio"]

    return {"signal": direction, "stop_loss": stop_loss, "take_profit": take_profit,
            "range_pct": range_pct, "rsi": rsi, "support": support, "resistance": resistance}


def simulate_trade(direction, entry, sl, tp, i, highs, lows, closes, max_hold):
    """以 entry 入场，逐 bar 检查 SL/TP，返回 (收益率, 出场原因, 出场 bar)。"""
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
    exit_px = float(closes[end - 1])
    if direction == "long":
        return (exit_px - entry) / entry, "time_exit", end - 1
    return (entry - exit_px) / entry, "time_exit", end - 1


# ------------------------------------------------------------
# 回测
# ------------------------------------------------------------
def run_symbol(sym, ts, closes, highs, lows, params, adx_threshold):
    n = len(closes)
    max_hold = params["max_hold"]
    adx = _adx_series(highs, lows, closes, 14)
    trades = []
    start = params["lookback_bars"] + 50
    for i in range(start, n - max_hold):
        if adx_threshold is not None and float(adx[i]) >= adx_threshold:
            continue
        c = closes[: i + 1]
        h = highs[: i + 1]
        l = lows[: i + 1]
        r = evaluate_oscillation(c, h, l, params)
        if not r or r.get("signal") not in ("long", "short"):
            continue
        side = r["signal"]
        entry = float(closes[i])
        ret, reason, exit_idx = simulate_trade(
            side, entry, r["stop_loss"], r["take_profit"], i, highs, lows, closes, max_hold)
        trades.append({"ts": int(ts[i]), "symbol": sym, "side": side,
                       "ret": ret, "ret_net": ret - FEE,
                       "adx": float(adx[i]), "range_pct": r["range_pct"],
                       "reason": reason})
    return trades


def _summary(trades, label):
    if not trades:
        return f"{label}: 无样本"
    net = [t["ret_net"] for t in trades]
    wins = [p for p in net if p > 0]
    losses = [p for p in net if p <= 0]
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    ratio = (avg_win / abs(avg_loss)) if avg_loss else float("inf")
    return (f"{label}: {len(net)}笔 胜率{len(wins)/len(net)*100:.1f}% "
            f"平均每笔{np.mean(net)*100:+.3f}% 盈亏比{ratio:.2f} "
            f"累计{np.sum(net)*100:+.2f}%")


def run_comparison(symbol_klines, params):
    print("\n" + "=" * 80)
    print("震荡收割策略 离线回测（ADX 震荡代理对比，已扣 0.1% 双边手续费）")
    print("=" * 80)

    thresholds = [("无过滤", None), ("ADX<25", 25), ("ADX<20", 20)]
    for label, thr in thresholds:
        all_trades = []
        for sym, (ts, closes, highs, lows, _v) in symbol_klines.items():
            all_trades.extend(run_symbol(sym, ts, closes, highs, lows, params, thr))
        print("\n" + _summary(all_trades, f"[{label}] 整体"))

        # 按币种
        sym_pnl = defaultdict(list)
        for t in all_trades:
            sym_pnl[t["symbol"]].append(t["ret_net"])
        ranked = sorted(sym_pnl.items(), key=lambda kv: np.sum(kv[1]), reverse=True)
        print(f"  按币种累计净收益:")
        for sym, p in ranked:
            w = sum(1 for x in p if x > 0)
            print(f"    {sym:<16} {len(p):>4}笔 胜率{w/len(p)*100:5.1f}% "
                  f"累计{np.sum(p)*100:+7.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bar", type=str, default="1H")
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--lookback", type=int, default=48)
    ap.add_argument("--min-band", type=float, default=0.003)
    ap.add_argument("--rsi-period", type=int, default=14)
    ap.add_argument("--rsi-oversold", type=float, default=30.0)
    ap.add_argument("--rsi-overbought", type=float, default=70.0)
    ap.add_argument("--atr-sl", type=float, default=1.5)
    ap.add_argument("--max-hold", type=int, default=24)
    ap.add_argument("--symbols", type=str, default=",".join(TIER12_SYMBOLS))
    ap.add_argument("--sweep", action="store_true",
                    help="参数网格扫描（lookback/止盈比/区间宽度/ADX 阈值）")
    args = ap.parse_args()

    params = {
        "lookback_bars": args.lookback,
        "support_band_pct": 0.01,
        "min_band_width_pct": args.min_band,
        "rsi_period": args.rsi_period,
        "rsi_oversold": args.rsi_oversold,
        "rsi_overbought": args.rsi_overbought,
        "atr_sl_mult": args.atr_sl,
        "mid_tp_ratio": 1.0,
        "max_hold": args.max_hold,
    }

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    print(f"标的 {len(symbols)} 个 / bar={args.bar} / limit={args.limit} / "
          f"lookback={args.lookback} / max_hold={args.max_hold}")

    symbol_klines = {}
    for sym in symbols:
        try:
            ts, closes, highs, lows, volumes = fetch_klines(sym, args.bar, args.limit)
            if len(closes) < args.lookback + 60:
                print(f"[跳过] {sym}: 数据不足 ({len(closes)} 根)")
                continue
            symbol_klines[sym] = (ts, closes, highs, lows, volumes)
            print(f"[完成] {sym}: {len(closes)} 根 {args.bar} K线")
        except Exception as e:
            print(f"[失败] {sym}: {e}")

    if not symbol_klines:
        print("无可用数据")
        return

    if args.sweep:
        results = sweep_params(
            symbol_klines,
            lookbacks=[24, 48, 96],
            mid_tp_ratios=[0.5, 1.0, 1.5, 2.0],
            min_bands=[0.003, 0.005, 0.008],
            adx_thresholds=[None, 25, 20],
            rsi_oversold=args.rsi_oversold, rsi_overbought=args.rsi_overbought,
            rsi_period=args.rsi_period, atr_sl=args.atr_sl,
            max_hold=args.max_hold, support_band_pct=0.01,
        )
        print_sweep(results)
        return

    run_comparison(symbol_klines, params)


if __name__ == "__main__":
    main()
