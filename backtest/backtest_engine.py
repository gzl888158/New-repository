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
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from loguru import logger
import yaml


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
        # 修复：profit_factor 为 inf 时 JSON 序列化失败，改用 999.0 表示
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0
        avg_win = gross_profit / len(wins) if wins else 0
        avg_loss = gross_loss / len(losses) if losses else 0
        roi = (self.final_equity - self.initial_capital) / self.initial_capital * 100

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
            logger.error(f"Failed to fetch klines for {symbol}: {e}")
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

    def _close_trade(self, trade: BacktestTrade, exit_price: float, exit_time: datetime, reason: str, fee_type: str = "taker"):
        """平仓交易
        fee_type: "maker" 或 "taker"
        """
        trade.exit_price = exit_price
        trade.status = "closed"
        trade.exit_reason = reason
        trade.hold_time_seconds = (exit_time - trade.timestamp).total_seconds()

        if trade.direction == "long":
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.quantity

        fee_rate = self.maker_fee if fee_type == "maker" else self.taker_fee
        exit_fee = trade.quantity * exit_price * fee_rate
        trade.fee += exit_fee
        if fee_type == "maker":
            trade.maker_fee += exit_fee
        else:
            trade.taker_fee += exit_fee
        trade.pnl_percent = trade.pnl / (trade.entry_price * trade.quantity) * 100 if trade.entry_price > 0 else 0


def run_backtest_cli():
    """命令行运行回测"""
    import sys
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    engine = BacktestEngine(config)

    # 默认参数
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


if __name__ == "__main__":
    run_backtest_cli()
