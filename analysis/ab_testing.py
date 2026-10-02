"""A/B测试框架

对同一策略用多组参数并行回测，对比统计指标，自动选出最优变体。
基于历史K线数据离线运行，不影响实盘。
"""
import asyncio
import json
import math
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from loguru import logger
import numpy as np


def _finite_values(values) -> List[float]:
    """过滤出有限的浮点数值，丢弃 None、NaN、Inf 及不可转换项。"""
    out: List[float] = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _safe_float(value, default: float = 0.0) -> float:
    """将值安全转换为有限浮点数，失败或非有限时返回默认值。"""
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


@dataclass
class TestVariant:
    """A/B测试变体定义"""
    name: str
    strategy_name: str  # grid / trend / scalping / arbitrage
    params: Dict[str, Any]  # 参数差异
    description: str = ""


@dataclass
class TestResult:
    """单变体回测结果"""
    variant_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float
    trade_pnls: List[float] = field(default_factory=list)


@dataclass
class StatisticalSignificance:
    """统计显著性检验结果"""
    variant_a: str
    variant_b: str
    metric: str
    mean_a: float
    mean_b: float
    difference: float
    difference_percent: float
    t_statistic: float
    p_value: float
    is_significant: bool
    ci_lower: float
    ci_upper: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float


class ABTestingFramework:
    def __init__(self, config: Dict[str, Any], okx_client=None):
        self.config = config
        self.okx_client = okx_client
        self._db_path = config.get("sqlite", {}).get("db_path", "./data/trading.db")
        self._results_dir = "./data/ab_tests"
        os.makedirs(self._results_dir, exist_ok=True)

    async def run_ab_test(
        self,
        symbol: str,
        variants: List[TestVariant],
        days: int = 30,
        kline_bar: str = "1H",
    ) -> Dict[str, Any]:
        """运行 A/B 测试
        Args:
            symbol: 测试的合约符号（如 BTC-USDT-SWAP）
            variants: 多组变体定义
            days: 历史数据天数
            kline_bar: K线周期
        Returns:
            对比报告，包含每变体的统计指标和最优变体推荐
        """
        logger.info(f"Starting A/B test: {symbol}, {len(variants)} variants, {days} days")

        # 1. 拉取历史K线
        klines = await self._fetch_historical_klines(symbol, days, kline_bar)
        if not klines or len(klines) < 50:
            return {"error": "Insufficient historical data", "klines_count": len(klines)}

        # 2. 对每个变体进行回测
        results: List[TestResult] = []
        for variant in variants:
            try:
                result = await self._backtest_variant(symbol, variant, klines, kline_bar)
                results.append(result)
                logger.info(f"Variant {variant.name}: pnl={result.total_pnl:.2f}, "
                            f"trades={result.total_trades}, win_rate={result.win_rate:.2%}")
            except Exception as e:
                logger.error(f"Failed to backtest variant {variant.name}: {e}")
                results.append(TestResult(
                    variant_name=variant.name, total_trades=0, winning_trades=0,
                    losing_trades=0, total_pnl=0, max_drawdown=0, sharpe_ratio=0,
                    win_rate=0, profit_factor=0, avg_win=0, avg_loss=0, max_win=0, max_loss=0
                ))

        # 3. 统计显著性检验
        significance = self.compute_statistical_significance(results)

        # 4. 选出最优变体
        winner = self._pick_winner(results)

        # 5. 生成报告
        report = {
            "symbol": symbol,
            "days": days,
            "kline_bar": kline_bar,
            "klines_count": len(klines),
            "timestamp": datetime.now().isoformat(),
            "variants": [self._result_to_dict(r) for r in results],
            "statistical_significance": significance,
            "winner": winner,
            "summary": self._generate_summary(results, winner, significance),
        }

        # 6. 持久化报告
        self._persist_report(symbol, report)

        return report

    async def _fetch_historical_klines(self, symbol: str, days: int, kline_bar: str) -> List[List]:
        """拉取历史K线"""
        if self.okx_client is None:
            return []
        try:
            # 每根1H K线1小时，30天=720根
            limit_map = {"1m": days * 1440, "5m": days * 288, "15m": days * 96,
                         "1H": days * 24, "4H": days * 6, "1d": days}
            limit = min(limit_map.get(kline_bar, days * 24), 300)  # OKX单次最多300根
            klines = self.okx_client.get_kline(symbol, bar=kline_bar, limit=limit)
            return klines or []
        except Exception as e:
            logger.error(f"Failed to fetch klines: {e}")
            return []

    async def _backtest_variant(
        self, symbol: str, variant: TestVariant,
        klines: List[List], kline_bar: str
    ) -> TestResult:
        """对单个变体进行回测
        优先调用 backtest_engine 完整模拟，失败或返回 None 则回退到简化回测
        """
        try:
            from backtest.backtest_engine import BacktestEngine
            engine = BacktestEngine(self.config)

            # 从 variant.params 读取均线参数，传给回测引擎
            params = variant.params or {}
            fast_period = int(params.get("fast_period", 5))
            slow_period = int(params.get("slow_period", 20))

            # 调用回测引擎（engine 内部会拉取数据，不传 klines）
            bt_result = engine.run(
                symbol=symbol,
                strategy_name=variant.strategy_name,
                days=30,
                bar=kline_bar,
                initial_capital=100.0,
                fast_period=fast_period,
                slow_period=slow_period,
            )

            if bt_result is None:
                raise RuntimeError("engine.run returned None")

            # 将 BacktestResult 转换为 TestResult（基于已平仓交易的 pnl）
            trades = [{"pnl": t.pnl} for t in bt_result.trades if t.status == "closed"]
            return self._compute_statistics_from_trades(variant.name, trades)
        except Exception as e:
            logger.error(f"Backtest failed for variant {variant.name}: {e}, using fallback")
            # 简化回退：直接基于K线生成模拟交易
            return self._simple_backtest(variant, klines)

    def _merge_variant_config(self, variant: TestVariant) -> Dict[str, Any]:
        """合并变体参数到config副本"""
        import copy
        cfg = copy.deepcopy(self.config)
        # 将 variant.params 应用到对应策略配置
        for key, value in variant.params.items():
            if variant.strategy_name in cfg.get("strategies", {}):
                cfg["strategies"][variant.strategy_name][key] = value
        return cfg

    def _compute_statistics_from_trades(self, variant_name: str, trades: List[Dict]) -> TestResult:
        """从交易列表计算统计指标"""
        if not trades:
            return TestResult(
                variant_name=variant_name, total_trades=0, winning_trades=0,
                losing_trades=0, total_pnl=0, max_drawdown=0, sharpe_ratio=0,
                win_rate=0, profit_factor=0, avg_win=0, avg_loss=0, max_win=0, max_loss=0
            )

        pnls = _finite_values([t.get("pnl") if isinstance(t, dict) else None for t in trades])
        if not pnls:
            return TestResult(
                variant_name=variant_name, total_trades=0, winning_trades=0,
                losing_trades=0, total_pnl=0, max_drawdown=0, sharpe_ratio=0,
                win_rate=0, profit_factor=0, avg_win=0, avg_loss=0, max_win=0, max_loss=0
            )

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        total_pnl = sum(pnls)
        total_trades = len(pnls)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0
        max_win = max(wins) if wins else 0
        max_loss = abs(min(losses)) if losses else 0

        # 无亏损时不使用 999.0 哨兵，统一返回 0.0（fail-closed，避免夸大无亏损策略）
        profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0.0

        # 最大回撤
        equity_curve = [0]
        for p in pnls:
            equity_curve.append(equity_curve[-1] + p)
        peak = equity_curve[0]
        max_dd = 0
        for v in equity_curve:
            if v > peak:
                peak = v
            dd = peak - v
            if dd > max_dd:
                max_dd = dd

        # 夏普比率（简化：用 pnl 序列的标准差）
        if len(pnls) > 1:
            std = float(np.std(pnls))
            sharpe = (sum(pnls) / len(pnls)) / std if std > 0 else 0
            # 年化（假设1H K线，年化系数=24*365=8760）
            sharpe_annualized = sharpe * (8760 ** 0.5)
        else:
            sharpe_annualized = 0

        return TestResult(
            variant_name=variant_name,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=round(total_pnl, 4),
            max_drawdown=round(max_dd, 4),
            sharpe_ratio=round(sharpe_annualized, 3),
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 3),
            avg_win=round(avg_win, 4),
            avg_loss=round(avg_loss, 4),
            max_win=round(max_win, 4),
            max_loss=round(max_loss, 4),
            trade_pnls=pnls,
        )

    def _simple_backtest(self, variant: TestVariant, klines: List[List]) -> TestResult:
        """简化回退：基于K线生成模拟交易（当backtest_engine不可用时）"""
        variant_name = variant.name
        params = variant.params or {}
        try:
            fast_period = int(params.get("fast_period", 5))
            slow_period = int(params.get("slow_period", 20))
        except (TypeError, ValueError):
            fast_period = 5
            slow_period = 20
        fast_period = max(1, fast_period)
        slow_period = max(2, slow_period)
        if fast_period >= slow_period:
            slow_period = fast_period + 1

        min_len = slow_period + 1
        if len(klines) < min_len:
            return TestResult(
                variant_name=variant_name, total_trades=0, winning_trades=0,
                losing_trades=0, total_pnl=0, max_drawdown=0, sharpe_ratio=0,
                win_rate=0, profit_factor=0, avg_win=0, avg_loss=0, max_win=0, max_loss=0
            )

        # 用简单的均线交叉策略生成模拟交易
        closes: List[float] = []
        for k in klines:
            if not isinstance(k, (list, tuple)) or len(k) < 5:
                continue
            try:
                c = float(k[4])
            except (TypeError, ValueError):
                continue
            if math.isfinite(c):
                closes.append(c)
        if len(closes) < min_len:
            return TestResult(
                variant_name=variant_name, total_trades=0, winning_trades=0,
                losing_trades=0, total_pnl=0, max_drawdown=0, sharpe_ratio=0,
                win_rate=0, profit_factor=0, avg_win=0, avg_loss=0, max_win=0, max_loss=0
            )

        ma_short = np.convolve(closes, np.ones(fast_period) / fast_period, mode='valid')
        ma_long = np.convolve(closes, np.ones(slow_period) / slow_period, mode='valid')

        trades = []
        position = None
        # ma_short 比 ma_long 长 (slow_period - fast_period) 个元素，需对齐到相同末尾索引
        offset = slow_period - fast_period
        for i in range(len(ma_long)):
            ma_short_idx = i + offset
            if ma_short_idx >= len(ma_short):
                break
            close_idx = i + slow_period - 1  # ma_long[i] 末尾对应的 closes 索引
            if ma_short[ma_short_idx] > ma_long[i] and position is None:
                position = {"entry": closes[close_idx], "side": "long"}
            elif ma_short[ma_short_idx] < ma_long[i] and position is not None:
                pnl = closes[close_idx] - position["entry"]
                trades.append({"pnl": pnl})
                position = None

        return self._compute_statistics_from_trades(variant_name, trades)

    def welch_t_test(self, a: List[float], b: List[float]) -> Dict[str, float]:
        """Welch's t检验（不假设等方差）
        Returns:
            t_statistic, p_value, degrees_of_freedom
        """
        a = _finite_values(a)
        b = _finite_values(b)
        if len(a) < 2 or len(b) < 2:
            return {"t_statistic": 0.0, "p_value": 1.0, "df": 0.0}

        mean_a = float(np.mean(a))
        mean_b = float(np.mean(b))
        var_a = float(np.var(a, ddof=1))
        var_b = float(np.var(b, ddof=1))
        n_a = len(a)
        n_b = len(b)

        se = (var_a / n_a + var_b / n_b) ** 0.5
        if se == 0:
            return {"t_statistic": 0.0, "p_value": 1.0, "df": 0.0}

        t_stat = (mean_b - mean_a) / se

        numerator = (var_a / n_a + var_b / n_b) ** 2
        denominator = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        df = numerator / denominator if denominator > 0 else 1.0

        p_value = self._two_tailed_p_value(t_stat, df)

        return {"t_statistic": round(t_stat, 4), "p_value": round(p_value, 4), "df": round(df, 2)}

    def _two_tailed_p_value(self, t_stat: float, df: float) -> float:
        """计算双尾p值（使用近似公式，避免scipy依赖）"""
        if df <= 0 or not math.isfinite(df):
            return 1.0
        if not math.isfinite(t_stat):
            return 0.0 if math.isinf(t_stat) else 1.0
        t_abs = abs(t_stat)
        x = df / (df + t_abs ** 2)

        if df % 2 == 0:
            k = df / 2
            p = x ** k
            term = p
            for i in range(1, int(k)):
                term *= (k - i) / i * (1 - x)
                p += term
            return float(p)
        else:
            from math import atan, sqrt, pi
            k = (df - 1) / 2
            p = 1 - 2 * atan(t_abs / sqrt(df)) / pi
            term = sqrt(x) * (1 - x)
            if k >= 1:
                p -= term * 2 / pi
                for i in range(1, int(k)):
                    term *= (2 * i - 1) / (2 * i) * (1 - x)
                    p -= term * 2 / pi
            return float(p)

    def bootstrap_confidence_interval(
        self, a: List[float], b: List[float], n_bootstrap: int = 1000,
        ci_level: float = 0.95, stat_func: Optional[Callable] = None
    ) -> Dict[str, float]:
        """Bootstrap置信区间（均值差）
        Args:
            a: 变体A的交易pnl
            b: 变体B的交易pnl
            n_bootstrap: bootstrap次数
            ci_level: 置信水平，默认0.95
            stat_func: 统计量函数，默认用均值差
        Returns:
            {"lower": ..., "upper": ..., "mean_diff": ...}
        """
        a = _finite_values(a)
        b = _finite_values(b)
        if len(a) < 2 or len(b) < 2:
            return {"lower": 0.0, "upper": 0.0, "mean_diff": 0.0}

        if stat_func is None:
            stat_func = lambda x, y: np.mean(y) - np.mean(x)

        np.random.seed(42)
        a_arr = np.array(a)
        b_arr = np.array(b)

        boot_diffs = []
        for _ in range(n_bootstrap):
            sample_a = np.random.choice(a_arr, size=len(a_arr), replace=True)
            sample_b = np.random.choice(b_arr, size=len(b_arr), replace=True)
            try:
                val = float(stat_func(sample_a, sample_b))
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            if math.isfinite(val):
                boot_diffs.append(val)

        if not boot_diffs:
            return {"lower": 0.0, "upper": 0.0, "mean_diff": 0.0}

        boot_diffs.sort()
        alpha = 1 - ci_level
        lower_idx = int(len(boot_diffs) * alpha / 2)
        upper_idx = int(len(boot_diffs) * (1 - alpha / 2)) - 1
        lower_idx = max(0, min(lower_idx, len(boot_diffs) - 1))
        upper_idx = max(0, min(upper_idx, len(boot_diffs) - 1))

        mean_diff = float(np.mean(b) - np.mean(a))

        return {
            "lower": round(boot_diffs[lower_idx], 6),
            "upper": round(boot_diffs[upper_idx], 6),
            "mean_diff": round(mean_diff, 6),
        }

    def compute_statistical_significance(
        self, results: List[TestResult], alpha: float = 0.05
    ) -> List[Dict[str, Any]]:
        """计算所有变体对之间的统计显著性
        Args:
            results: 各变体的测试结果
            alpha: 显著性水平，默认0.05
        Returns:
            显著性检验结果列表
        """
        significance_results = []

        valid_results = [r for r in results if r.total_trades >= 5 and len(_finite_values(r.trade_pnls)) >= 5]
        if len(valid_results) < 2:
            return significance_results

        for i in range(len(valid_results)):
            for j in range(i + 1, len(valid_results)):
                a = valid_results[i]
                b = valid_results[j]

                pnls_a = _finite_values(a.trade_pnls)
                pnls_b = _finite_values(b.trade_pnls)

                t_result = self.welch_t_test(pnls_a, pnls_b)
                boot_result = self.bootstrap_confidence_interval(pnls_a, pnls_b)

                mean_a = float(np.mean(pnls_a))
                mean_b = float(np.mean(pnls_b))
                diff = mean_b - mean_a
                diff_pct = (diff / abs(mean_a) * 100) if mean_a != 0 else 0.0

                se_a = float(np.std(pnls_a, ddof=1)) / (len(pnls_a) ** 0.5) if len(pnls_a) > 1 else 0
                se_b = float(np.std(pnls_b, ddof=1)) / (len(pnls_b) ** 0.5) if len(pnls_b) > 1 else 0
                se_diff = (se_a ** 2 + se_b ** 2) ** 0.5

                from math import sqrt
                ci_lower = diff - 1.96 * se_diff
                ci_upper = diff + 1.96 * se_diff

                is_significant = t_result["p_value"] < alpha and boot_result["lower"] * boot_result["upper"] > 0

                significance_results.append({
                    "variant_a": a.variant_name,
                    "variant_b": b.variant_name,
                    "metric": "avg_pnl_per_trade",
                    "mean_a": round(mean_a, 6),
                    "mean_b": round(mean_b, 6),
                    "difference": round(diff, 6),
                    "difference_percent": round(diff_pct, 2),
                    "t_statistic": t_result["t_statistic"],
                    "p_value": t_result["p_value"],
                    "degrees_of_freedom": t_result["df"],
                    "alpha": alpha,
                    "is_significant": is_significant,
                    "ci_95_lower": round(ci_lower, 6),
                    "ci_95_upper": round(ci_upper, 6),
                    "bootstrap_ci_lower": boot_result["lower"],
                    "bootstrap_ci_upper": boot_result["upper"],
                    "interpretation": (
                        f"差异{'显著' if is_significant else '不显著'} "
                        f"(p={t_result['p_value']:.4f}, {'p<' if t_result['p_value'] < alpha else 'p>='}{alpha})"
                    )
                })

        return significance_results

    def _pick_winner(self, results: List[TestResult]) -> Optional[Dict[str, Any]]:
        """选出最优变体：综合考虑 total_pnl, sharpe, profit_factor, win_rate"""
        if not results:
            return None

        def score(r: TestResult) -> float:
            # 综合评分：盈利能力 + 风险调整收益 + 胜率 + 交易次数（太少不靠谱）
            total_trades = int(_safe_float(r.total_trades))
            if total_trades < 3:
                return -999
            total_pnl = _safe_float(r.total_pnl)
            sharpe = _safe_float(r.sharpe_ratio)
            profit_factor = _safe_float(r.profit_factor)
            win_rate = _safe_float(r.win_rate)
            return (
                total_pnl * 0.4
                + sharpe * 5
                + (profit_factor if profit_factor < 10 else 10) * 10
                + win_rate * 50
            )

        winner = max(results, key=score)
        return {
            "variant_name": winner.variant_name,
            "score": round(score(winner), 3),
            "total_pnl": winner.total_pnl,
            "sharpe_ratio": winner.sharpe_ratio,
            "win_rate": winner.win_rate,
            "profit_factor": winner.profit_factor,
        }

    def _generate_summary(self, results: List[TestResult], winner: Optional[Dict],
                          significance: List[Dict] = None) -> str:
        if not winner:
            return "No valid variants tested"
        sig_part = ""
        if significance:
            sig_count = sum(1 for s in significance if s.get("is_significant"))
            sig_part = f" | {sig_count}/{len(significance)} pairs statistically significant"
        total_pnl = _safe_float(winner.get("total_pnl"))
        sharpe = _safe_float(winner.get("sharpe_ratio"))
        win_rate = _safe_float(winner.get("win_rate"))
        return (f"Tested {len(results)} variants. Winner: {winner['variant_name']} "
                f"(score={winner.get('score')}, pnl={total_pnl:.2f}, "
                f"sharpe={sharpe}, win_rate={win_rate:.2%}){sig_part}")

    def _result_to_dict(self, r: TestResult) -> Dict[str, Any]:
        return {
            "variant_name": r.variant_name,
            "total_trades": r.total_trades,
            "winning_trades": r.winning_trades,
            "losing_trades": r.losing_trades,
            "total_pnl": r.total_pnl,
            "max_drawdown": r.max_drawdown,
            "sharpe_ratio": r.sharpe_ratio,
            "win_rate": r.win_rate,
            "profit_factor": r.profit_factor,
            "avg_win": r.avg_win,
            "avg_loss": r.avg_loss,
            "max_win": r.max_win,
            "max_loss": r.max_loss,
        }

    def _persist_report(self, symbol: str, report: Dict[str, Any]) -> bool:
        """持久化测试报告（fail-closed：成功返回 True，失败返回 False）"""
        try:
            safe_symbol = symbol.replace("/", "_").replace("-", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"{self._results_dir}/ab_test_{safe_symbol}_{timestamp}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"A/B test report saved: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to persist A/B test report: {e}")
            return False

    def list_reports(self) -> List[Dict[str, Any]]:
        """列出所有历史测试报告"""
        reports = []
        try:
            for fname in sorted(os.listdir(self._results_dir), reverse=True):
                if fname.startswith("ab_test_") and fname.endswith(".json"):
                    fpath = os.path.join(self._results_dir, fname)
                    mtime = os.path.getmtime(fpath)
                    # 读取基本信息
                    try:
                        with open(fpath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                    except Exception:
                        continue
                    try:
                        ts = datetime.fromtimestamp(mtime).isoformat()
                    except (OSError, OverflowError, ValueError, TypeError):
                        ts = ""
                    reports.append({
                        "filename": fname,
                        "timestamp": ts,
                        "symbol": data.get("symbol", ""),
                        "days": data.get("days", 0),
                        "winner": data.get("winner", {}).get("variant_name", "") if data.get("winner") else "",
                        "variants_count": len(data.get("variants", [])),
                    })
        except Exception as e:
            logger.error(f"Error listing reports: {e}")
        return reports

    def get_report(self, filename: str) -> Optional[Dict[str, Any]]:
        """读取指定报告"""
        try:
            fpath = os.path.join(self._results_dir, filename)
            if not os.path.exists(fpath):
                return None
            with open(fpath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading report {filename}: {e}")
            return None
