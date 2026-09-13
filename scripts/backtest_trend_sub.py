"""
趋势子策略离线回测（企业级验证）

用 OKX 公开 K 线历史数据，对四类趋势子策略做信号级回测：
  1. 均线趋势（MA/EMA 金叉死叉 + 多均线排列 + ATR 震荡过滤）
  2. 唐奇安通道突破（Donchian + ATR 动态止盈止损）
  3. MACD 零轴过滤（零轴上方做多 / 下方做空，仅辅助）
  4. 动量策略（2h/4h 滚动涨跌幅，强者恒强）

判定方式：
  - 信号出现后，用未来 forward 根 K 线的涨跌方向判定信号是否正确。
  - long 命中 = 未来收益 > 0；short 命中 = 未来收益 < 0。

只读、不触碰实盘。用法：
  py -3 scripts/backtest_trend_sub.py [--symbols BTC,ETH,...] [--limit 400] [--forward 4,8,24]
"""
import argparse
import sys
from collections import defaultdict
from typing import List, Dict, Any, Tuple

import numpy as np
import requests

sys.path.insert(0, ".")

from strategies.trend_sub_strategies import (
    evaluate_ma_trend,
    evaluate_donchian,
    macd_zero_axis,
    rolling_return,
)

DEFAULT_SYMBOLS = [
    "BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "ADA-USDT-SWAP",
    "AVAX-USDT-SWAP", "NEAR-USDT-SWAP", "APT-USDT-SWAP", "SUI-USDT-SWAP",
    "ARB-USDT-SWAP", "OP-USDT-SWAP", "DOT-USDT-SWAP", "LINK-USDT-SWAP",
    "UNI-USDT-SWAP", "ATOM-USDT-SWAP",
]
PROXY = "http://127.0.0.1:7897"


# ------------------------------------------------------------
# 指标 helper（MACD）
# ------------------------------------------------------------
def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    k = 2.0 / (period + 1)
    out = [arr[0]]
    e = arr[0]
    for x in arr[1:]:
        e = x * k + e * (1.0 - k)
        out.append(e)
    return np.array(out)


def _macd(closes: np.ndarray) -> Tuple[float, float]:
    if len(closes) < 35:
        return 0.0, 0.0
    e12 = _ema(closes, 12)
    e26 = _ema(closes, 26)
    macd_line = e12 - e26
    signal = _ema(macd_line, 9)
    return float(macd_line[-1]), float(signal[-1])


# ------------------------------------------------------------
# 数据拉取
# ------------------------------------------------------------
def fetch_klines(symbol: str, bar: str, limit: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    url = f"https://www.okx.com/api/v5/market/history-candles?instId={symbol}&bar={bar}&limit={limit}"
    proxies = {"http": PROXY, "https": PROXY}
    try:
        r = requests.get(url, proxies=proxies, timeout=20)
    except Exception:
        r = requests.get(url, timeout=20)
    data = r.json()
    if data.get("code") != "0":
        raise RuntimeError(f"OKX error for {symbol}: {data}")
    rows = list(reversed(data["data"]))  # 倒序 → 正序
    closes = np.array([float(x[4]) for x in rows])
    highs = np.array([float(x[2]) for x in rows])
    lows = np.array([float(x[3]) for x in rows])
    return closes, highs, lows


# ------------------------------------------------------------
# 回测引擎
# ------------------------------------------------------------
class SubStrategyBacktester:
    def __init__(self, forwards: List[int]):
        self.forwards = forwards
        # stats[策略名][side][forward] = [命中数, 总数]
        self.stats: Dict[str, Dict[str, Dict[int, List[int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: [0, 0]))
        )

    def _record(self, name: str, side: str, correct: bool, forward: int):
        cell = self.stats[name][side][forward]
        cell[1] += 1
        if correct:
            cell[0] += 1

    def run_symbol(self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray):
        n = len(closes)
        max_fwd = max(self.forwards)
        # 至少需要 60 根（MA60）+ forward 余量
        for i in range(60, n - max_fwd):
            c = closes[: i + 1]
            h = highs[: i + 1]
            l = lows[: i + 1]
            cur = float(closes[i])

            # 1. 均线趋势
            ma = evaluate_ma_trend(c, h, l)
            if ma.get("signal") in ("long", "short"):
                for fwd in self.forwards:
                    fwd_ret = float(closes[i + fwd]) / cur - 1.0
                    correct = (ma["signal"] == "long" and fwd_ret > 0) or \
                              (ma["signal"] == "short" and fwd_ret < 0)
                    self._record("MA趋势", ma["signal"], correct, fwd)

            # 2. 唐奇安通道突破
            don = evaluate_donchian(h, l, c)
            if don.get("signal") in ("long", "short"):
                for fwd in self.forwards:
                    fwd_ret = float(closes[i + fwd]) / cur - 1.0
                    correct = (don["signal"] == "long" and fwd_ret > 0) or \
                              (don["signal"] == "short" and fwd_ret < 0)
                    self._record("唐奇安", don["signal"], correct, fwd)

            # 3. MACD 零轴过滤（bull → 做多，bear → 做空）
            mv, sv = _macd(c)
            axis = macd_zero_axis(mv, sv)
            if axis in ("bull", "bear"):
                for fwd in self.forwards:
                    fwd_ret = float(closes[i + fwd]) / cur - 1.0
                    correct = (axis == "bull" and fwd_ret > 0) or \
                              (axis == "bear" and fwd_ret < 0)
                    self._record("MACD零轴", "long" if axis == "bull" else "short", correct, fwd)

            # 4. 动量（4h 涨跌幅：强者恒强）
            mom4 = rolling_return(c, 4)
            if mom4 > 0.0005 or mom4 < -0.0005:
                side = "long" if mom4 > 0 else "short"
                for fwd in self.forwards:
                    fwd_ret = float(closes[i + fwd]) / cur - 1.0
                    correct = (side == "long" and fwd_ret > 0) or \
                              (side == "short" and fwd_ret < 0)
                    self._record("动量4h", side, correct, fwd)

    def report(self):
        print("\n" + "=" * 72)
        print("趋势子策略回测结果（方向命中率，long=未来涨/short=未来跌）")
        print("=" * 72)
        for name in ("MA趋势", "唐奇安", "MACD零轴", "动量4h"):
            print(f"\n【{name}】")
            for side in ("long", "short"):
                for fwd in self.forwards:
                    hit, tot = self.stats[name][side][fwd]
                    if tot == 0:
                        continue
                    wr = hit / tot * 100
                    bar = "#" * int(wr / 5)
                    print(f"  {side:<6} fwd={fwd:>2}根  {wr:5.1f}%  ({hit}/{tot})  {bar}")
        print("\n" + "=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--bar", type=str, default="1H")
    ap.add_argument("--forward", type=str, default="4,8,24")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    forwards = [int(x) for x in args.forward.split(",") if x.strip()]
    bt = SubStrategyBacktester(forwards)

    for sym in symbols:
        try:
            closes, highs, lows = fetch_klines(sym, args.bar, args.limit)
            if len(closes) < 90:
                print(f"[跳过] {sym}: 数据不足 ({len(closes)} 根)")
                continue
            bt.run_symbol(closes, highs, lows)
            print(f"[完成] {sym}: {len(closes)} 根 {args.bar} K线")
        except Exception as e:
            print(f"[失败] {sym}: {e}")

    bt.report()


if __name__ == "__main__":
    main()
