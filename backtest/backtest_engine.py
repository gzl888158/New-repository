"""
历史回测引擎
加载历史K线数据 → 模拟策略执行 → 统计绩效 → 参数优化

使用方式：
    from backtest.backtest_engine import BacktestEngine
    engine = BacktestEngine(config)
    result = engine.run(symbol="BTC-USDT-SWAP", strategy_name="trend", days=30)
    print(result.summary())
"""
import os
import json
import asyncio
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from loguru import logger
import yaml

import numpy as np

from utils.helpers import evaluate_signal_quality


@dataclass
class BacktestTrade:
    """回测交易记录"""
    timestamp: datetime
    symbol: str
    strategy: str
    direction: str  # long/short
    entry_price: float
    quantity: float
    exit_price: float = 0
    leverage: int = 1
    fee: float = 0
    maker_fee: float = 0
    taker_fee: float = 0
    funding_fee: float = 0
    funding_periods_settled: int = 0
    pnl: float = 0
    pnl_percent: float = 0
    hold_time_seconds: float = 0
    exit_reason: str = ""
    status: str = "open"  # open/closed


@dataclass
class BacktestResult:
    """回测结果"""
    symbol: str
    strategy: str
    start_time: datetime
    end_time: datetime
    initial_capital: float
    final_equity: float
    trades: List[BacktestTrade] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        closed_trades = [t for t in self.trades if t.status == "closed"]
        if not closed_trades:
            return {"error": "No closed trades"}

        wins = [t for t in closed_trades if t.pnl > 0]
        losses = [t for t in closed_trades if t.pnl < 0]
        total_pnl = sum(t.pnl for t in closed_trades)
        total_fee = sum(t.fee for t in closed_trades)
        total_maker_fee = sum(t.maker_fee for t in closed_trades)
        total_taker_fee = sum(t.taker_fee for t in closed_trades)
        total_funding_fee = sum(t.funding_fee for t in closed_trades)
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))

        win_rate = len(wins) / len(closed_trades) * 100
        # 无亏损时盈亏比无数学意义，用 None 表示「不适用」，
        # 避免 999.0 哨兵值污染 JSON 输出、误导下游分析
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
        avg_win = gross_profit / len(wins) if wins else 0
        avg_loss = gross_loss / len(losses) if losses else 0
        roi = (self.final_equity - self.initial_capital) / self.initial_capital * 100 if self.initial_capital > 0 else 0

        # 最大回撤
        equity_curve = self._calc_equity_curve()
        peak = equity_curve[0] if equity_curve else self.initial_capital
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        # 平均持仓时间
        avg_hold = sum(t.hold_time_seconds for t in closed_trades) / len(closed_trades) if closed_trades else 0

        # 夏普比率（简化版，假设无风险利率=0）
        if len(closed_trades) > 1:
            returns = [t.pnl / self.initial_capital for t in closed_trades]
            mean_ret = sum(returns) / len(returns)
            variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
            std = variance ** 0.5
            sharpe = mean_ret / std * (365 ** 0.5) if std > 0 else 0  # 年化
        else:
            sharpe = 0

        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "period": f"{self.start_time.date()} ~ {self.end_time.date()}",
            "initial_capital": self.initial_capital,
            "final_equity": self.final_equity,
            "roi_percent": roi,
            "total_trades": len(closed_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_percent": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_pnl": total_pnl,
            "total_fee": total_fee,
            "total_maker_fee": total_maker_fee,
            "total_taker_fee": total_taker_fee,
            "total_funding_fee": total_funding_fee,
            "net_pnl": total_pnl - total_fee - total_funding_fee,
            "max_drawdown_percent": max_dd * 100,
            "avg_hold_minutes": avg_hold / 60,
            "sharpe_ratio": sharpe
        }

    def _calc_equity_curve(self) -> List[float]:
        """计算权益曲线"""
        equity = self.initial_capital
        curve = [equity]
        for t in sorted(self.trades, key=lambda x: x.timestamp):
            if t.status == "closed":
                equity += t.pnl - t.fee - t.funding_fee
                curve.append(equity)
        return curve


class BacktestEngine:
    """历史回测引擎"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rest_url = config.get("okx", {}).get("rest_url", "https://www.okx.com")
        self.proxy = config.get("okx", {}).get("proxy")
        self.taker_fee = config.get("execution", {}).get("taker_fee", 0.0005)
        self.maker_fee = config.get("execution", {}).get("maker_fee", 0.0002)
        self.funding_rate = config.get("execution", {}).get("funding_rate", 0.0001)
        self.funding_interval_hours = config.get("execution", {}).get("funding_interval_hours", 8)
        # P1: 回测滑点模型 —— 平仓价格应用滑点，避免回测结果偏乐观
        # 默认 0.05% (5bps)，可通过 config.execution.backtest_slippage_rate 配置
        self.slippage_rate = config.get("execution", {}).get("backtest_slippage_rate", 0.0005)

    def fetch_historical_klines(self, symbol: str, bar: str = "1H", days: int = 30) -> List[Dict[str, Any]]:
        """从OKX获取历史K线数据
        bar: 1m/5m/15m/1H/4H/1D等
        """
        try:
            import requests
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=days)
            start_ts = int(start.timestamp() * 1000)

            all_candles = []
            after = int(end.timestamp() * 1000)
            session = requests.Session()
            if self.proxy:
                session.proxies = {"http": self.proxy, "https": self.proxy}

            # OKX history-candles 最多100根/请求，需要分页
            while after > start_ts:
                path = f"/api/v5/market/history-candles?instId={symbol}&bar={bar}&after={after}&limit=100"
                url = f"{self.rest_url}{path}"
                resp = session.get(url, timeout=15)
                data = resp.json()

                if data.get("code") != "0" or not data.get("data"):
                    break

                candles = data["data"]
                all_candles.extend(candles)
                # 下一页
                after = int(candles[-1][0])
                if len(candles) < 100:
                    break

            # 转换格式并按时间正序
            formatted = []
            for c in reversed(all_candles):
                formatted.append({
                    "timestamp": datetime.fromtimestamp(int(c[0]) / 1000),
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5])
                })

            logger.info(f"Fetched {len(formatted)} candles for {symbol} ({bar}, {days}d)")
            return formatted

        except Exception as e:
            logger.error(f"Failed to fetch klines for {symbol}: {type(e).__name__}: {e}")
            return []

    def run(self, symbol: str, strategy_name: str, days: int = 30,
            bar: str = "1H", initial_capital: float = 100.0,
            leverage: int = 6, fast_period: int = 5,
            slow_period: int = 20) -> BacktestResult:
        """运行回测（拉取历史K线后调用 run_with_candles）"""
        logger.info(f"Starting backtest: {symbol} {strategy_name} {days}d {bar} "
                    f"capital={initial_capital} fast={fast_period} slow={slow_period}")

        candles = self.fetch_historical_klines(symbol, bar, days)
        return self.run_with_candles(candles, symbol, strategy_name,
                                     initial_capital, leverage, fast_period, slow_period)

    def run_with_candles(self, candles: List[Dict[str, Any]], symbol: str,
                         strategy_name: str, initial_capital: float = 100.0,
                         leverage: int = 6, fast_period: int = 5,
                         slow_period: int = 20) -> BacktestResult:
        """基于已加载的K线数据运行回测（供参数优化复用，避免重复拉取）。

        真实策略模块依赖完整运行时上下文（redis_cache、okx_client、策略管理器等），
        无法在回测环境中直接复用，因此保留简化的均线交叉策略作为默认实现，
        通过 fast_period/slow_period 参数化使 A/B 测试可对比不同均线配置。
        """
        if len(candles) < slow_period + 1:
            logger.error(f"Insufficient candle data: {len(candles)}")
            return BacktestResult(symbol, strategy_name, datetime.now(), datetime.now(),
                                  initial_capital, initial_capital)

        start_time = candles[0]["timestamp"]
        end_time = candles[-1]["timestamp"]

        # 模拟策略执行
        trades: List[BacktestTrade] = []
        equity = initial_capital
        current_position: Optional[BacktestTrade] = None

        # 简化的均线交叉策略（实际应调用策略类的generate_signal方法）
        # 修复 look-ahead bias：
        #   - 信号在 candle[i] 收盘后生成（MA 计算含 candle[i]）
        #   - 入场在 candles[i+1]["open"]（下一根开盘价），i+1 不存在则跳过
        #   - 止盈止损用 candle["high"]/candle["low"] 检测触发，以触发价成交
        tp_pct = 0.02  # +2% 止盈
        sl_pct = 0.01  # -1% 止损
        max_hold_seconds = 3600 * 4  # 4小时超时
        funding_interval = self.funding_interval_hours * 3600

        for i in range(slow_period, len(candles)):
            candle = candles[i]

            if current_position:
                hold_time = (candle["timestamp"] - current_position.timestamp).total_seconds()
                # 资金费按 funding_interval 周期结算；用累计已结算次数做增量，
                # 消除原 -3600 魔数对 bar 间隔（隐含 bar=1H）的错误假设
                total_periods = int(hold_time / funding_interval)
                new_periods = total_periods - current_position.funding_periods_settled
                if new_periods > 0:
                    position_value = current_position.entry_price * current_position.quantity
                    funding_amount = position_value * self.funding_rate * new_periods
                    if current_position.direction == "short":
                        current_position.funding_fee -= funding_amount
                    else:
                        current_position.funding_fee += funding_amount
                    current_position.funding_periods_settled = total_periods

                if current_position.direction == "long":
                    tp_price = current_position.entry_price * (1 + tp_pct)
                    sl_price = current_position.entry_price * (1 - sl_pct)
                    if candle["high"] >= tp_price:
                        self._close_trade(current_position, tp_price, candle["timestamp"], "tp_sl", "taker")
                        trades.append(current_position)
                        equity += current_position.pnl - current_position.fee - current_position.funding_fee
                        current_position = None
                    elif candle["low"] <= sl_price:
                        self._close_trade(current_position, sl_price, candle["timestamp"], "tp_sl", "taker")
                        trades.append(current_position)
                        equity += current_position.pnl - current_position.fee - current_position.funding_fee
                        current_position = None
                    elif hold_time > max_hold_seconds:
                        self._close_trade(current_position, candle["close"], candle["timestamp"], "timeout", "taker")
                        trades.append(current_position)
                        equity += current_position.pnl - current_position.fee - current_position.funding_fee
                        current_position = None
                else:
                    tp_price = current_position.entry_price * (1 - tp_pct)
                    sl_price = current_position.entry_price * (1 + sl_pct)
                    if candle["low"] <= tp_price:
                        self._close_trade(current_position, tp_price, candle["timestamp"], "tp_sl", "taker")
                        trades.append(current_position)
                        equity += current_position.pnl - current_position.fee - current_position.funding_fee
                        current_position = None
                    elif candle["high"] >= sl_price:
                        self._close_trade(current_position, sl_price, candle["timestamp"], "tp_sl", "taker")
                        trades.append(current_position)
                        equity += current_position.pnl - current_position.fee - current_position.funding_fee
                        current_position = None
                    elif hold_time > max_hold_seconds:
                        self._close_trade(current_position, candle["close"], candle["timestamp"], "timeout", "taker")
                        trades.append(current_position)
                        equity += current_position.pnl - current_position.fee - current_position.funding_fee
                        current_position = None

            if not current_position and equity > 10 and i + 1 < len(candles):
                ma_short = sum(c["close"] for c in candles[i - fast_period + 1:i + 1]) / fast_period
                ma_long = sum(c["close"] for c in candles[i - slow_period + 1:i + 1]) / slow_period
                prev_ma_short = sum(c["close"] for c in candles[i - fast_period:i]) / fast_period
                prev_ma_long = sum(c["close"] for c in candles[i - slow_period:i]) / slow_period

                next_candle = candles[i + 1]
                entry_price = next_candle["open"]

                if ma_short > ma_long and prev_ma_short <= prev_ma_long:
                    qty = equity * 0.9 * leverage / entry_price
                    entry_fee = qty * entry_price * self.taker_fee
                    current_position = BacktestTrade(
                        timestamp=next_candle["timestamp"],
                        symbol=symbol,
                        strategy=strategy_name,
                        direction="long",
                        entry_price=entry_price,
                        quantity=qty,
                        leverage=leverage,
                        fee=entry_fee,
                        taker_fee=entry_fee,
                        status="open"
                    )
                elif ma_short < ma_long and prev_ma_short >= prev_ma_long:
                    qty = equity * 0.9 * leverage / entry_price
                    entry_fee = qty * entry_price * self.taker_fee
                    current_position = BacktestTrade(
                        timestamp=next_candle["timestamp"],
                        symbol=symbol,
                        strategy=strategy_name,
                        direction="short",
                        entry_price=entry_price,
                        quantity=qty,
                        leverage=leverage,
                        fee=entry_fee,
                        taker_fee=entry_fee,
                        status="open"
                    )

        # 平掉最后未平仓位
        if current_position:
            last_price = candles[-1]["close"]
            self._close_trade(current_position, last_price, candles[-1]["timestamp"], "end_of_backtest", "taker")
            trades.append(current_position)
            equity += current_position.pnl - current_position.fee - current_position.funding_fee

        result = BacktestResult(
            symbol=symbol,
            strategy=strategy_name,
            start_time=start_time,
            end_time=end_time,
            initial_capital=initial_capital,
            final_equity=equity,
            trades=trades
        )

        logger.info(f"Backtest completed: {len(trades)} trades, final equity={equity:.2f}")
        return result

    def _calc_rsi_series(self, closes: List[float], period: int) -> List[float]:
        """计算 Wilder RSI 序列，rsi[i] 为截至第 i 根收盘价的 RSI（回测专用，独立于生产策略）。"""
        n = len(closes)
        rsi = [50.0] * n
        if n <= period or period <= 0:
            return rsi

        gains = [0.0] * n
        losses = [0.0] * n
        for i in range(1, n):
            delta = closes[i] - closes[i - 1]
            gains[i] = max(delta, 0.0)
            losses[i] = max(-delta, 0.0)

        avg_gain = sum(gains[1:period + 1]) / period
        avg_loss = sum(losses[1:period + 1]) / period

        for i in range(period, n):
            if avg_loss > 0:
                rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
            else:
                rsi[i] = 100.0 if avg_gain > 0 else 50.0
            if i + 1 < n:
                avg_gain = (avg_gain * (period - 1) + gains[i + 1]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i + 1]) / period
        return rsi

    def _calc_volume_delta_series(self, closes: List[float], volumes: List[float],
                                  lookback: int = 10) -> List[float]:
        """成交量 delta 序列：最近 lookback 根K线的 (上涨量-下跌量)/(上涨量+下跌量)，对齐生产 _calculate_volume_delta。"""
        n = len(closes)
        delta = [0.0] * n
        for i in range(1, n):
            up = 0.0
            down = 0.0
            start = max(0, i - lookback)
            for j in range(start + 1, i + 1):
                if closes[j] > closes[j - 1]:
                    up += volumes[j]
                else:
                    down += volumes[j]
            if up + down > 0:
                delta[i] = (up - down) / (up + down)
        return delta

    def _calc_momentum_series(self, closes: List[float], periods: int = 5) -> List[float]:
        """动量序列：(close[i]-close[i-periods])/close[i-periods]。"""
        n = len(closes)
        mom = [0.0] * n
        for i in range(periods, n):
            if closes[i - periods] > 0:
                mom[i] = (closes[i] - closes[i - periods]) / closes[i - periods]
        return mom

    def _calc_atr_ratio_series(self, candles: List[Dict[str, Any]], period: int = 14) -> List[float]:
        """ATR/价格 序列（归一化波动率），对齐生产 evaluate_signal_quality 的 atr_ratio 口径。"""
        n = len(candles)
        ratio = [1.0] * n
        trs = [0.0] * n
        for i in range(1, n):
            h = float(candles[i]["high"])
            l = float(candles[i]["low"])
            pc = float(candles[i - 1]["close"])
            trs[i] = max(h - l, abs(h - pc), abs(l - pc))
        for i in range(period, n):
            atr = sum(trs[i - period + 1:i + 1]) / period
            c = float(candles[i]["close"])
            if c > 0 and atr > 0:
                ratio[i] = atr / c
        return ratio

    def _calc_ema(self, values: List[float], period: int) -> List[float]:
        n = len(values)
        ema = [0.0] * n
        if n == 0:
            return ema
        k = 2.0 / (period + 1)
        ema[0] = values[0]
        for i in range(1, n):
            ema[i] = values[i] * k + ema[i - 1] * (1 - k)
        return ema

    def _calc_vwap_distance_series(self, candles: List[Dict[str, Any]], lookback: int = 20) -> List[float]:
        """VWAP 偏离序列：(close - vwap)/vwap，对齐生产 evaluate_signal_quality 的 vwap_distance。"""
        n = len(candles)
        dist = [0.0] * n
        for i in range(n):
            start = max(0, i - lookback + 1)
            pv = 0.0
            vol = 0.0
            for j in range(start, i + 1):
                h = float(candles[j]["high"]); l = float(candles[j]["low"]); c = float(candles[j]["close"])
                typical = (h + l + c) / 3.0
                v = float(candles[j].get("volume", 0.0))
                pv += typical * v
                vol += v
            if vol > 0:
                vwap = pv / vol
                c = float(candles[i]["close"])
                if vwap > 0:
                    dist[i] = (c - vwap) / vwap
        return dist

    def _calc_macd_histogram_series(self, closes: List[float], fast: int = 12,
                                    slow: int = 26, signal: int = 9) -> List[float]:
        """MACD 柱序列，对齐生产 evaluate_signal_quality 的 macd_histogram。"""
        n = len(closes)
        ema_fast = self._calc_ema(closes, fast)
        ema_slow = self._calc_ema(closes, slow)
        macd_line = [ema_fast[i] - ema_slow[i] for i in range(n)]
        signal_line = self._calc_ema(macd_line, signal)
        return [macd_line[i] - signal_line[i] for i in range(n)]

    def _calc_bb_position_series(self, closes: List[float], period: int = 20,
                                 num_std: float = 2.0) -> List[float]:
        """布林带位置序列：(close - lower)/(upper - lower)，对齐生产 evaluate_signal_quality 的 bb_position。"""
        n = len(closes)
        pos = [0.5] * n
        for i in range(period - 1, n):
            window = closes[i - period + 1:i + 1]
            mean = sum(window) / period
            var = sum((x - mean) ** 2 for x in window) / period
            std = math.sqrt(var)
            upper = mean + num_std * std
            lower = mean - num_std * std
            if upper > lower:
                pos[i] = (closes[i] - lower) / (upper - lower)
        return pos

    def _calc_stoch_series(self, candles: List[Dict[str, Any]], k_period: int = 14,
                           d_period: int = 3) -> Tuple[List[float], List[float]]:
        """随机指标 %K / %D 序列，对齐生产 evaluate_signal_quality 的 stoch_k / stoch_d。"""
        n = len(candles)
        k = [50.0] * n
        d = [50.0] * n
        for i in range(k_period - 1, n):
            window = candles[i - k_period + 1:i + 1]
            hh = max(float(c["high"]) for c in window)
            ll = min(float(c["low"]) for c in window)
            close = float(candles[i]["close"])
            k[i] = (close - ll) / (hh - ll) * 100.0 if hh > ll else 50.0
        for i in range(n):
            d[i] = sum(k[max(0, i - d_period + 1):i + 1]) / min(i + 1, d_period)
        return k, d

    def _calc_adx_series(self, candles: List[Dict[str, Any]], period: int = 14) -> List[float]:
        """ADX 序列（简化 Wilder 平滑），用于市场状态趋势/震荡分类。"""
        n = len(candles)
        dx = [0.0] * n
        trs = [0.0] * n
        plus_dm = [0.0] * n
        minus_dm = [0.0] * n
        for i in range(1, n):
            h = float(candles[i]["high"]); l = float(candles[i]["low"])
            pc = float(candles[i - 1]["close"]); ph = float(candles[i - 1]["high"]); pl = float(candles[i - 1]["low"])
            trs[i] = max(h - l, abs(h - pc), abs(l - pc))
            up = h - ph
            down = pl - l
            plus_dm[i] = up if (up > down and up > 0) else 0.0
            minus_dm[i] = down if (down > up and down > 0) else 0.0
        for i in range(period, n):
            atr = sum(trs[i - period + 1:i + 1]) / period
            pdi = sum(plus_dm[i - period + 1:i + 1]) / period
            mdi = sum(minus_dm[i - period + 1:i + 1]) / period
            if atr > 0:
                pdi_val = pdi / atr * 100.0
                mdi_val = mdi / atr * 100.0
            else:
                pdi_val = mdi_val = 0.0
            dx[i] = abs(pdi_val - mdi_val) / (pdi_val + mdi_val) * 100.0 if (pdi_val + mdi_val) > 0 else 0.0
        return self._calc_ema(dx, period)

    def _scalping_signal_quality(self, indicators: Dict[str, float],
                                 market_state: Dict[str, Any], direction: str) -> float:
        """调用生产 evaluate_signal_quality，喂入回测 OHLCV 可计算的完整指标集 + 市场状态。"""
        return evaluate_signal_quality(indicators, market_state, direction)

    def _scalping_market_state(self, ema_fast: float, ema_slow: float, adx: float,
                               atr_ratio: float, volumes: List[float], i: int) -> Dict[str, Any]:
        """回测市场状态分类：趋势/波动率/量比，供生产 evaluate_signal_quality 的 state/volatility/volume_ratio 使用。"""
        slope = (ema_fast - ema_slow) / ema_slow if ema_slow > 0 else 0.0
        if adx >= 25 and slope > 0.001:
            state = "uptrend"
        elif adx >= 25 and slope < -0.001:
            state = "downtrend"
        else:
            state = "range"
        if atr_ratio > 0.015:
            volatility = "high"
        elif atr_ratio < 0.004:
            volatility = "low"
        else:
            volatility = "normal"
        lookback = volumes[max(0, i - 19):i + 1]
        avg_vol = sum(lookback) / len(lookback) if lookback else 0.0
        volume_ratio = volumes[i] / avg_vol if avg_vol > 0 else 1.0
        return {"state": state, "volatility": volatility, "volume_ratio": volume_ratio}

    def run_scalping_with_candles(self, candles: List[Dict[str, Any]], symbol: str,
                                  strategy_name: str = "scalping",
                                  initial_capital: float = 100.0, leverage: int = 6,
                                  rsi_period: int = 4, rsi_oversold: float = 30.0,
                                  rsi_overbought: float = 70.0,
                                  profit_target_min: float = 0.008, stop_loss: float = 0.005,
                                  max_hold_minutes: int = 10,
                                  min_signal_quality: Optional[float] = None,
                                  wear_fee_multiple: float = 3.0,
                                  max_daily_trades: Optional[int] = None) -> BacktestResult:
        """scalping 简化回测：RSI 均值回归入场 + 参数化止盈/止损/超时。

        修复历史退化问题：参数优化器此前对 scalping 用 MA 交叉回测评估，
        而 scalping 参数里没有 ema_fast/ema_slow 键，导致所有候选参数被压平成
        常量 MA(5,20)、fitness 恒等（-1.997）。本方法以 scalping 真实参数驱动回测，
        使 fitness 随 rsi_period/rsi_oversold/rsi_overbought/profit_target_min/
        stop_loss/max_hold_minutes 产生真实分化。

        复现口径（与 run_with_candles 的 MA 交叉回测并列）：
          - RSI < rsi_oversold → 超卖做多；RSI > rsi_overbought → 超买做空
          - 信号在 candle[i] 收盘生成，入场在 candle[i+1].open
          - 止盈 profit_target_min / 止损 stop_loss（价格比例），超时 max_hold_minutes

        生产过滤注入（解决「裸 RSI 信号 + taker 手续费」磨损、避免优化器被手续费噪声误导）：
          - 磨损过滤器：profit_target_min 须 > wear_fee_multiple × 往返 taker 手续费（3×手续费）
          - 信号质量门槛：min_signal_quality（默认 0.18）提纯裸 RSI 信号
          - 单日交易上限：max_daily_trades（默认 20）抑制高频
        """
        if rsi_period <= 0:
            rsi_period = 4
        if len(candles) < rsi_period + 1:
            return BacktestResult(symbol, strategy_name, datetime.now(), datetime.now(),
                                  initial_capital, initial_capital)

        # 从生产配置解析过滤阈值（参数未显式指定时回退到 config，缺省用安全默认值）
        scalp_cfg = (self.config or {}).get("strategies", {}).get("scalping", {}) or {}
        if min_signal_quality is None:
            min_signal_quality = float(scalp_cfg.get("min_signal_quality", 0.18))
        if max_daily_trades is None:
            max_daily_trades = int(scalp_cfg.get("max_daily_trades", 20) or 0)

        start_time = candles[0]["timestamp"]
        end_time = candles[-1]["timestamp"]
        trades: List[BacktestTrade] = []
        equity = initial_capital
        current_position: Optional[BacktestTrade] = None

        closes = [float(c["close"]) for c in candles]
        volumes = [float(c.get("volume", 0.0)) for c in candles]
        rsi_values = self._calc_rsi_series(closes, rsi_period)
        volume_delta_series = self._calc_volume_delta_series(closes, volumes)
        momentum_series = self._calc_momentum_series(closes, 5)
        atr_ratio_series = self._calc_atr_ratio_series(candles, 14)
        vwap_dist_series = self._calc_vwap_distance_series(candles, 20)
        macd_hist_series = self._calc_macd_histogram_series(closes)
        bb_pos_series = self._calc_bb_position_series(closes)
        stoch_k_series, stoch_d_series = self._calc_stoch_series(candles)
        adx_series = self._calc_adx_series(candles)
        ema_fast = self._calc_ema(closes, 20)
        ema_slow = self._calc_ema(closes, 50)
        max_hold_seconds = max_hold_minutes * 60
        funding_interval = self.funding_interval_hours * 3600

        # 磨损过滤器（3×手续费）：止盈目标须 > N× 往返 taker 手续费，否则结构上无法覆盖成本，弃单
        round_trip_fee_rate = self.taker_fee * 2
        if wear_fee_multiple > 0 and profit_target_min <= wear_fee_multiple * round_trip_fee_rate:
            logger.warning(
                f"Scalping backtest skipped: profit_target_min={profit_target_min:.4%} "
                f"<= {wear_fee_multiple:.0f}x round-trip fee={round_trip_fee_rate:.4%} (wear filter)"
            )
            return BacktestResult(symbol, strategy_name, start_time, end_time,
                                  initial_capital, initial_capital)

        # 单日交易上限（高频抑制）
        daily_trades = 0
        current_day = None

        for i in range(rsi_period, len(candles)):
            candle = candles[i]

            if current_position:
                hold_time = (candle["timestamp"] - current_position.timestamp).total_seconds()
                total_periods = int(hold_time / funding_interval)
                new_periods = total_periods - current_position.funding_periods_settled
                if new_periods > 0:
                    position_value = current_position.entry_price * current_position.quantity
                    funding_amount = position_value * self.funding_rate * new_periods
                    if current_position.direction == "short":
                        current_position.funding_fee -= funding_amount
                    else:
                        current_position.funding_fee += funding_amount
                    current_position.funding_periods_settled = total_periods

                if current_position.direction == "long":
                    tp_price = current_position.entry_price * (1 + profit_target_min)
                    sl_price = current_position.entry_price * (1 - stop_loss)
                    if candle["high"] >= tp_price:
                        self._close_trade(current_position, tp_price, candle["timestamp"], "tp", "taker")
                        trades.append(current_position)
                        equity += current_position.pnl - current_position.fee - current_position.funding_fee
                        current_position = None
                    elif candle["low"] <= sl_price:
                        self._close_trade(current_position, sl_price, candle["timestamp"], "sl", "taker")
                        trades.append(current_position)
                        equity += current_position.pnl - current_position.fee - current_position.funding_fee
                        current_position = None
                    elif hold_time > max_hold_seconds:
                        self._close_trade(current_position, candle["close"], candle["timestamp"], "timeout", "taker")
                        trades.append(current_position)
                        equity += current_position.pnl - current_position.fee - current_position.funding_fee
                        current_position = None
                else:
                    tp_price = current_position.entry_price * (1 - profit_target_min)
                    sl_price = current_position.entry_price * (1 + stop_loss)
                    if candle["low"] <= tp_price:
                        self._close_trade(current_position, tp_price, candle["timestamp"], "tp", "taker")
                        trades.append(current_position)
                        equity += current_position.pnl - current_position.fee - current_position.funding_fee
                        current_position = None
                    elif candle["high"] >= sl_price:
                        self._close_trade(current_position, sl_price, candle["timestamp"], "sl", "taker")
                        trades.append(current_position)
                        equity += current_position.pnl - current_position.fee - current_position.funding_fee
                        current_position = None
                    elif hold_time > max_hold_seconds:
                        self._close_trade(current_position, candle["close"], candle["timestamp"], "timeout", "taker")
                        trades.append(current_position)
                        equity += current_position.pnl - current_position.fee - current_position.funding_fee
                        current_position = None

            if not current_position and equity > 10 and i + 1 < len(candles):
                rsi = rsi_values[i]
                direction = "long" if rsi < rsi_oversold else ("short" if rsi > rsi_overbought else None)
                if direction is None:
                    continue

                # 生产信号质量门槛（完整 evaluate_signal_quality：8 指标 + 市场状态），
                # 对齐生产 mean_reversion 信号类型（RSI 超买超卖反向入场，不叠加 ADX 趋势门禁）
                if min_signal_quality > 0:
                    indicators = {
                        "rsi": rsi,
                        "volume_delta": volume_delta_series[i],
                        "vwap_distance": vwap_dist_series[i],
                        "momentum": momentum_series[i],
                        "macd_histogram": macd_hist_series[i],
                        "bb_position": bb_pos_series[i],
                        "stoch_k": stoch_k_series[i],
                        "stoch_d": stoch_d_series[i],
                        "atr_ratio": atr_ratio_series[i],
                    }
                    market_state = self._scalping_market_state(
                        ema_fast[i], ema_slow[i], adx_series[i], atr_ratio_series[i], volumes, i)
                    quality = self._scalping_signal_quality(indicators, market_state, direction)
                    if quality < min_signal_quality:
                        continue

                # 单日交易上限：按入场日归零计数
                entry_day = candles[i + 1]["timestamp"].date()
                if entry_day != current_day:
                    current_day = entry_day
                    daily_trades = 0
                if max_daily_trades > 0 and daily_trades >= max_daily_trades:
                    continue

                next_candle = candles[i + 1]
                entry_price = next_candle["open"]
                qty = equity * 0.9 * leverage / entry_price
                entry_fee = qty * entry_price * self.taker_fee
                current_position = BacktestTrade(
                    timestamp=next_candle["timestamp"], symbol=symbol, strategy=strategy_name,
                    direction=direction, entry_price=entry_price, quantity=qty, leverage=leverage,
                    fee=entry_fee, taker_fee=entry_fee, status="open",
                )
                daily_trades += 1

        if current_position:
            last_price = candles[-1]["close"]
            self._close_trade(current_position, last_price, candles[-1]["timestamp"], "end_of_backtest", "taker")
            trades.append(current_position)
            equity += current_position.pnl - current_position.fee - current_position.funding_fee

        result = BacktestResult(
            symbol=symbol, strategy=strategy_name, start_time=start_time, end_time=end_time,
            initial_capital=initial_capital, final_equity=equity, trades=trades,
        )
        logger.info(f"Scalping backtest completed: {len(trades)} trades, final equity={equity:.2f}")
        return result

    def run_grid_with_candles(self, candles: List[Dict[str, Any]], symbol: str,
                              strategy_name: str = "grid",
                              initial_capital: float = 100.0, leverage: int = 5,
                              grid_spacing: float = 0.03, tp_pct: float = 0.02,
                              sl_pct: float = 0.02, max_hold_hours: float = 72.0,
                              long_only: bool = False,
                              min_signal_quality: Optional[float] = None,
                              trend_filter_threshold: float = 0.05,
                              wear_fee_multiple: float = 0.0,
                              ref_period: int = 20) -> BacktestResult:
        """grid 简化回测：EMA 参考 + 网格间距均值回归入场 + 固定止盈/止损/超时。

        生产 GridStrategy 是带多层网格、5 维信号质量门槛、P33 趋势门禁、多级止盈、
        移动止损、趋势模式切换的异步状态机，无法在回测环境中直接复用。
        本方法量化的核心经济学是：
          参考价（EMA20）± grid_spacing 处买入/卖出（均值回归），
          以 tp_pct 止盈、sl_pct 止损、max_hold_hours 超时，
          每边 taker 手续费 —— 即「裸网格均值回归 + taker 手续费」的下限约束，
          与 run_scalping_with_candles 的「裸 RSI + taker 手续费」口径并列。

        关键参数对齐生产 config（strategies.grid）：
          - grid_spacing: min_grid_spacing(0.03) ~ max_grid_spacing(0.04)，入场带宽度
          - tp_pct: tp1_pct(0.02)（多级止盈 2%/4%/trailing 简化为单级 2%，忽略 tp2 上沿，偏保守）
          - sl_pct: 有效止损 = min(stop_loss_pct 0.02, max_stop_loss_pct 0.02) = 0.02
          - max_hold_hours: 72
          - min_signal_quality: 0.35（生产 5 维评分门槛，本回测用 8 指标 evaluate_signal_quality 近似）
          - trend_filter_threshold: 0.05（对齐生产 _confirm_grid_entry：|EMA20-EMA50|/EMA50 > 0.05 时禁止逆势开仓）
        """
        if len(candles) < ref_period + 1:
            return BacktestResult(symbol, strategy_name, datetime.now(), datetime.now(),
                                  initial_capital, initial_capital)

        grid_cfg = (self.config or {}).get("strategies", {}).get("grid", {}) or {}
        if min_signal_quality is None:
            min_signal_quality = float(grid_cfg.get("min_signal_quality", 0.35))

        start_time = candles[0]["timestamp"]
        end_time = candles[-1]["timestamp"]
        trades: List[BacktestTrade] = []
        equity = initial_capital
        current_position: Optional[BacktestTrade] = None

        closes = [float(c["close"]) for c in candles]
        volumes = [float(c.get("volume", 0.0)) for c in candles]
        ema_ref = self._calc_ema(closes, ref_period)
        # 信号质量门槛所需指标序列（复用 scalping 的辅助计算）
        volume_delta_series = self._calc_volume_delta_series(closes, volumes)
        momentum_series = self._calc_momentum_series(closes, 5)
        atr_ratio_series = self._calc_atr_ratio_series(candles, 14)
        vwap_dist_series = self._calc_vwap_distance_series(candles, 20)
        macd_hist_series = self._calc_macd_histogram_series(closes)
        bb_pos_series = self._calc_bb_position_series(closes)
        stoch_k_series, stoch_d_series = self._calc_stoch_series(candles)
        adx_series = self._calc_adx_series(candles)
        ema_fast = self._calc_ema(closes, 20)
        ema_slow = self._calc_ema(closes, 50)

        max_hold_seconds = max_hold_hours * 3600
        funding_interval = self.funding_interval_hours * 3600

        for i in range(ref_period, len(candles)):
            candle = candles[i]

            # ── 平仓检测（TP 优先，与 scalping 口径一致） ──
            if current_position:
                hold_time = (candle["timestamp"] - current_position.timestamp).total_seconds()
                total_periods = int(hold_time / funding_interval)
                new_periods = total_periods - current_position.funding_periods_settled
                if new_periods > 0:
                    position_value = current_position.entry_price * current_position.quantity
                    funding_amount = position_value * self.funding_rate * new_periods
                    if current_position.direction == "short":
                        current_position.funding_fee -= funding_amount
                    else:
                        current_position.funding_fee += funding_amount
                    current_position.funding_periods_settled = total_periods

                if current_position.direction == "long":
                    tp_price = current_position.entry_price * (1 + tp_pct)
                    sl_price = current_position.entry_price * (1 - sl_pct)
                    if candle["high"] >= tp_price:
                        self._close_trade(current_position, tp_price, candle["timestamp"], "tp", "taker")
                    elif candle["low"] <= sl_price:
                        self._close_trade(current_position, sl_price, candle["timestamp"], "sl", "taker")
                    elif hold_time > max_hold_seconds:
                        self._close_trade(current_position, candle["close"], candle["timestamp"], "timeout", "taker")
                    else:
                        continue
                    trades.append(current_position)
                    equity += current_position.pnl - current_position.fee - current_position.funding_fee
                    current_position = None
                else:
                    tp_price = current_position.entry_price * (1 - tp_pct)
                    sl_price = current_position.entry_price * (1 + sl_pct)
                    if candle["low"] <= tp_price:
                        self._close_trade(current_position, tp_price, candle["timestamp"], "tp", "taker")
                    elif candle["high"] >= sl_price:
                        self._close_trade(current_position, sl_price, candle["timestamp"], "sl", "taker")
                    elif hold_time > max_hold_seconds:
                        self._close_trade(current_position, candle["close"], candle["timestamp"], "timeout", "taker")
                    else:
                        continue
                    trades.append(current_position)
                    equity += current_position.pnl - current_position.fee - current_position.funding_fee
                    current_position = None

            # ── 入场检测（每根 K 线最多开 1 仓，单持仓简化） ──
            if current_position is None and equity > 10:
                ref = ema_ref[i]
                buy_band = ref * (1 - grid_spacing)
                sell_band = ref * (1 + grid_spacing)
                direction = None
                entry_price = None
                if candle["low"] <= buy_band:
                    direction = "long"
                    entry_price = buy_band  # 网格限价单在带价成交
                elif (not long_only) and candle["high"] >= sell_band:
                    direction = "short"
                    entry_price = sell_band

                if direction is None:
                    continue

                # 信号质量门槛（8 指标 evaluate_signal_quality 近似生产 5 维评分）
                if min_signal_quality > 0:
                    indicators = {
                        "rsi": 50.0,
                        "volume_delta": volume_delta_series[i],
                        "vwap_distance": vwap_dist_series[i],
                        "momentum": momentum_series[i],
                        "macd_histogram": macd_hist_series[i],
                        "bb_position": bb_pos_series[i],
                        "stoch_k": stoch_k_series[i],
                        "stoch_d": stoch_d_series[i],
                        "atr_ratio": atr_ratio_series[i],
                    }
                    market_state = self._scalping_market_state(
                        ema_fast[i], ema_slow[i], adx_series[i], atr_ratio_series[i], volumes, i)
                    quality = self._scalping_signal_quality(indicators, market_state, direction)
                    if quality < min_signal_quality:
                        continue

                # 趋势方向过滤（对齐生产 _confirm_grid_entry: strength>0.05 禁止逆势）
                # strength = |EMA20-EMA50|/EMA50；direction: 1=上涨, -1=下跌, 0=震荡
                if trend_filter_threshold > 0:
                    _trend_strength = abs(ema_fast[i] - ema_slow[i]) / ema_slow[i] if ema_slow[i] > 0 else 0.0
                    _trend_dir = 0 if _trend_strength < 0.005 else (1 if ema_fast[i] > ema_slow[i] else -1)
                    if _trend_strength > trend_filter_threshold:
                        if _trend_dir == 1 and direction == "short":
                            continue  # 强上涨趋势禁止逆势做空
                        if _trend_dir == -1 and direction == "long":
                            continue  # 强下跌趋势禁止逆势做多

                qty = equity * 0.9 * leverage / entry_price
                entry_fee = qty * entry_price * self.taker_fee
                current_position = BacktestTrade(
                    timestamp=candle["timestamp"], symbol=symbol, strategy=strategy_name,
                    direction=direction, entry_price=entry_price, quantity=qty, leverage=leverage,
                    fee=entry_fee, taker_fee=entry_fee, status="open",
                )

        if current_position:
            last_price = candles[-1]["close"]
            self._close_trade(current_position, last_price, candles[-1]["timestamp"], "end_of_backtest", "taker")
            trades.append(current_position)
            equity += current_position.pnl - current_position.fee - current_position.funding_fee

        result = BacktestResult(
            symbol=symbol, strategy=strategy_name, start_time=start_time, end_time=end_time,
            initial_capital=initial_capital, final_equity=equity, trades=trades,
        )
        logger.info(f"Grid backtest completed: {len(trades)} trades, final equity={equity:.2f}")
        return result

    def _close_trade(self, trade: BacktestTrade, exit_price: float, exit_time: datetime, reason: str, fee_type: str = "taker"):
        """平仓交易
        fee_type: "maker" 或 "taker"
        P1: 平仓价格应用滑点 —— 做多平仓（卖出）价格下移，做空平仓（买入）价格上移
        """
        # 应用滑点：做多平仓时卖出价格更低，做空平仓时买入价格更高
        if trade.direction == "long":
            # 平仓是卖出，滑点使实际卖出价更低
            actual_exit_price = exit_price * (1 - self.slippage_rate)
        else:
            # 平仓是买入，滑点使实际买入价更高
            actual_exit_price = exit_price * (1 + self.slippage_rate)

        trade.exit_price = actual_exit_price
        trade.status = "closed"
        trade.exit_reason = reason
        trade.hold_time_seconds = (exit_time - trade.timestamp).total_seconds()

        if trade.direction == "long":
            trade.pnl = (actual_exit_price - trade.entry_price) * trade.quantity
        else:
            trade.pnl = (trade.entry_price - actual_exit_price) * trade.quantity

        fee_rate = self.maker_fee if fee_type == "maker" else self.taker_fee
        exit_fee = trade.quantity * actual_exit_price * fee_rate
        trade.fee += exit_fee
        if fee_type == "maker":
            trade.maker_fee += exit_fee
        else:
            trade.taker_fee += exit_fee
        notional = trade.entry_price * trade.quantity
        trade.pnl_percent = trade.pnl / notional * 100 if notional > 0 else 0


class PortfolioBacktester:
    """策略相关性组合回测 —— 多策略联合回测并评估相关性风险。

    将多个策略在同一组 K 线上回测，按时间桶对齐收益序列，
    计算策略间相关性矩阵，并输出组合级别的 Sharpe / 最大回撤 / VaR，
    帮助识别策略同质化（高相关）和分散化收益。
    """

    def __init__(self, engine: BacktestEngine):
        self._engine = engine

    def run(
        self,
        candles: List[Dict[str, Any]],
        strategies: List[Dict[str, Any]],
        symbol: str = "BTC-USDT-SWAP",
        initial_capital_per_strategy: float = 100.0,
        leverage: int = 6,
        time_bucket_hours: int = 1,
    ) -> Dict[str, Any]:
        """对多个策略运行组合回测。

        Args:
            candles: 共享的 K 线数据。
            strategies: 策略列表，每项至少含 {"name": str}，可选
                        fast_period / slow_period 等参数。
            symbol: 交易对。
            initial_capital_per_strategy: 每个策略的初始资金。
            leverage: 杠杆倍数。
            time_bucket_hours: 收益对齐的时间桶（小时）。
        """
        if not strategies:
            return {"error": "no strategies provided"}

        individual_results: Dict[str, BacktestResult] = {}
        for strat_cfg in strategies:
            name = strat_cfg.get("name", "unknown")
            result = self._engine.run_with_candles(
                candles,
                symbol,
                name,
                initial_capital=initial_capital_per_strategy,
                leverage=leverage,
                fast_period=strat_cfg.get("fast_period", 5),
                slow_period=strat_cfg.get("slow_period", 20),
            )
            individual_results[name] = result

        returns_by_strategy = self._build_aligned_returns(
            individual_results, time_bucket_hours
        )
        strategy_names = sorted(returns_by_strategy.keys())
        if len(strategy_names) < 2:
            return {
                "error": "need at least 2 strategies with trades for correlation",
                "individual": {
                    n: r.summary() for n, r in individual_results.items()
                },
            }

        min_len = min(len(returns_by_strategy[n]) for n in strategy_names)
        returns_matrix = np.array(
            [returns_by_strategy[n][:min_len] for n in strategy_names]
        )
        corr_matrix = np.corrcoef(returns_matrix)

        portfolio_weights = self._equal_weights(len(strategy_names))
        portfolio_returns = portfolio_weights @ returns_matrix

        portfolio_metrics = self._compute_portfolio_metrics(
            portfolio_returns, initial_capital_per_strategy
        )
        individual_metrics = {
            n: individual_results[n].summary() for n in strategy_names
        }

        avg_corr = self._avg_pairwise_correlation(corr_matrix)
        high_corr_pairs = self._find_high_corr_pairs(
            strategy_names, corr_matrix, threshold=0.7
        )
        diversification_benefit = self._compute_diversification_benefit(
            individual_metrics, portfolio_metrics, strategy_names
        )

        return {
            "symbol": symbol,
            "strategies": strategy_names,
            "period": f"{candles[0].get('timestamp', '?')} ~ {candles[-1].get('timestamp', '?')}" if candles else "",
            "time_bucket_hours": time_bucket_hours,
            "aligned_periods": min_len,
            "individual": individual_metrics,
            "portfolio": portfolio_metrics,
            "correlation": {
                "strategies": strategy_names,
                "matrix": corr_matrix.round(4).tolist(),
                "avg_correlation": round(avg_corr, 4),
                "high_corr_pairs": high_corr_pairs,
            },
            "diversification_benefit": diversification_benefit,
        }

    def _build_aligned_returns(
        self,
        results: Dict[str, BacktestResult],
        bucket_hours: int,
    ) -> Dict[str, List[float]]:
        """将每个策略的交易按时间桶聚合为收益率序列。"""
        bucket_seconds = bucket_hours * 3600
        returns_by_strategy: Dict[str, List[float]] = {}

        all_timestamps: List[float] = []
        for name, result in results.items():
            for trade in result.trades:
                if trade.status == "closed":
                    all_timestamps.append(trade.timestamp.timestamp())

        if not all_timestamps:
            return {}

        t_min = min(all_timestamps)
        t_max = max(all_timestamps)
        n_buckets = max(1, int((t_max - t_min) / bucket_seconds) + 1)

        for name, result in results.items():
            capital = result.initial_capital
            bucket_pnl = [0.0] * n_buckets
            for trade in result.trades:
                if trade.status != "closed":
                    continue
                idx = int((trade.timestamp.timestamp() - t_min) / bucket_seconds)
                idx = min(idx, n_buckets - 1)
                bucket_pnl[idx] += trade.pnl - trade.fee - trade.funding_fee
            returns_by_strategy[name] = [
                pnl / capital if capital > 0 else 0.0 for pnl in bucket_pnl
            ]

        return returns_by_strategy

    @staticmethod
    def _equal_weights(n: int) -> np.ndarray:
        return np.ones(n) / n

    @staticmethod
    def _compute_portfolio_metrics(
        returns: np.ndarray, capital: float
    ) -> Dict[str, Any]:
        total_return = float(np.sum(returns))
        roi_pct = total_return / capital * 100 if capital > 0 else 0.0

        equity = capital
        peak = capital
        max_dd = 0.0
        for r in returns:
            equity += r
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        n = len(returns)
        mean_r = float(np.mean(returns)) if n > 0 else 0.0
        std_r = float(np.std(returns, ddof=1)) if n > 1 else 0.0
        sharpe = mean_r / std_r * (365 ** 0.5) if std_r > 0 else 0.0

        sorted_returns = np.sort(returns)
        var_95 = 0.0
        cvar_95 = 0.0
        if n >= 20:
            idx_95 = max(0, int(n * 0.05) - 1)
            var_95 = float(sorted_returns[idx_95])
            cvar_95 = float(np.mean(sorted_returns[: idx_95 + 1]))

        return {
            "total_pnl": round(total_return, 4),
            "roi_percent": round(roi_pct, 4),
            "max_drawdown_percent": round(max_dd * 100, 4),
            "sharpe_ratio": round(sharpe, 4),
            "var_95": round(var_95, 6),
            "cvar_95": round(cvar_95, 6),
            "total_periods": n,
            "positive_periods": int(np.sum(returns > 0)),
            "negative_periods": int(np.sum(returns < 0)),
        }

    @staticmethod
    def _avg_pairwise_correlation(corr: np.ndarray) -> float:
        n = corr.shape[0]
        if n < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += corr[i, j]
                count += 1
        return total / count if count > 0 else 0.0

    @staticmethod
    def _find_high_corr_pairs(
        names: List[str], corr: np.ndarray, threshold: float
    ) -> List[Dict[str, Any]]:
        pairs = []
        n = corr.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                r = float(corr[i, j])
                if abs(r) >= threshold:
                    pairs.append({
                        "strategy_a": names[i],
                        "strategy_b": names[j],
                        "correlation": round(r, 4),
                    })
        return sorted(pairs, key=lambda p: abs(p["correlation"]), reverse=True)

    @staticmethod
    def _compute_diversification_benefit(
        individual: Dict[str, Any],
        portfolio: Dict[str, Any],
        names: List[str],
    ) -> Dict[str, Any]:
        """计算分散化收益：组合回撤 vs 各策略平均回撤的改善幅度。"""
        dd_values = []
        sharpe_values = []
        for n in names:
            s = individual.get(n, {})
            dd_values.append(s.get("max_drawdown_percent", 0.0))
            sharpe_values.append(s.get("sharpe_ratio", 0.0))

        avg_dd = sum(dd_values) / len(dd_values) if dd_values else 0.0
        avg_sharpe = sum(sharpe_values) / len(sharpe_values) if sharpe_values else 0.0
        port_dd = portfolio.get("max_drawdown_percent", 0.0)
        port_sharpe = portfolio.get("sharpe_ratio", 0.0)

        dd_improvement = avg_dd - port_dd
        sharpe_improvement = port_sharpe - avg_sharpe

        return {
            "avg_individual_drawdown_pct": round(avg_dd, 4),
            "portfolio_drawdown_pct": round(port_dd, 4),
            "drawdown_improvement_pct": round(dd_improvement, 4),
            "avg_individual_sharpe": round(avg_sharpe, 4),
            "portfolio_sharpe": round(port_sharpe, 4),
            "sharpe_improvement": round(sharpe_improvement, 4),
        }


def run_backtest_cli():
    """命令行运行回测"""
    import sys
    config_path = "config.yaml"
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config.yaml, using empty config: {e}")
            config = {}

    engine = BacktestEngine(config)

    if len(sys.argv) > 1 and sys.argv[1] == "--portfolio":
        _run_portfolio_cli(engine, config, sys.argv[2:])
        return

    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC-USDT-SWAP"
    strategy = sys.argv[2] if len(sys.argv) > 2 else "trend"
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    result = engine.run(symbol, strategy, days)
    summary = result.summary()

    print("\n" + "=" * 60)
    print(f"  回测结果: {symbol} / {strategy} / {days}天")
    print("=" * 60)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")
    print("=" * 60)


def _run_portfolio_cli(engine: BacktestEngine, config: Dict, args: List[str]):
    """组合回测 CLI。"""
    import sys
    symbol = args[0] if len(args) > 0 else "BTC-USDT-SWAP"
    days = int(args[1]) if len(args) > 1 else 30

    portfolio_cfg = config.get("portfolio_backtest", {})
    strategy_names = portfolio_cfg.get("strategies", ["trend", "scalping", "grid"])
    strategies = []
    for name in strategy_names:
        strategies.append({"name": name})

    candles = engine.fetch_historical_klines(symbol, "1H", days)
    if not candles:
        print("ERROR: no candle data fetched")
        return

    pb = PortfolioBacktester(engine)
    result = pb.run(candles, strategies, symbol=symbol)

    print("\n" + "=" * 70)
    print(f"  组合回测: {symbol} / {days}天 / {len(strategies)}策略")
    print("=" * 70)

    port = result.get("portfolio", {})
    print("\n  [组合指标]")
    for k in ("total_pnl", "roi_percent", "max_drawdown_percent",
              "sharpe_ratio", "var_95", "cvar_95"):
        print(f"    {k:25s}: {port.get(k, 'N/A')}")

    corr = result.get("correlation", {})
    print(f"\n  [相关性] avg={corr.get('avg_correlation', 'N/A')}")
    for pair in corr.get("high_corr_pairs", []):
        print(f"    {pair['strategy_a']} <-> {pair['strategy_b']}: {pair['correlation']}")

    div = result.get("diversification_benefit", {})
    print("\n  [分散化收益]")
    print(f"    回撤改善: {div.get('drawdown_improvement_pct', 'N/A')}%")
    print(f"    夏普改善: {div.get('sharpe_improvement', 'N/A')}")
    print("=" * 70)


if __name__ == "__main__":
    run_backtest_cli()
