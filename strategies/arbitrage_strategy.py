"""
生产级套利策略 (Production-Grade Arbitrage Strategy)
====================================================

三层套利体系：
  1. 资金费率套利 (Funding Rate Arbitrage) — 多空对冲赚取资金费率
  2. 基差套利 (Basis Arbitrage) — 期货/现货价差回归
  3. 统计套利 (Statistical/Correlation Arbitrage) — 高相关币种对价差偏离回归

生产级增强 (v2.0):
  - 凯利公式动态仓位管理
  - ATR动态止损止盈
  - 市场状态自适应检测
  - 成交量加权机会评分
  - 相关性矩阵缓存
  - AlertRegistry集成
  - 夏普比率/MDD性能追踪
  - 分批建仓/平仓
  - 五层风控拦截层联动
  - 原子对冲执行保障
"""

import asyncio
import numpy as np
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from loguru import logger

from core.models import Signal, FundingRate
from configs.settings import get_currency_tier
from utils.state_persistence import PersistentStrategy
from risk.dynamic_allocator import AdaptiveKelly

# TWAP 执行器（生产级大单拆分）
try:
    from execution.algo_orders.twap_executor import TWAPExecutor, TWAPConfig
    from execution.algo_orders.algo_execution_engine import AlgoOrderConfig
    _TWAP_AVAILABLE = True
except ImportError:
    _TWAP_AVAILABLE = False
    logger.warning("TWAP executor not available, batch execution disabled")


# ============================================================
# 市场状态枚举
# ============================================================

class MarketRegime:
    """市场状态分类"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


# ============================================================
# 生产级套利策略
# ============================================================

class ArbitrageStrategy(PersistentStrategy):
    """生产级套利策略 v2.0

    核心改进：
    - 凯利公式仓位：f = (bp - q) / b，其中 b = avg_win/avg_loss，p = win_rate，q = 1-p
    - ATR动态止损：止损 = 入仓价 ± N * ATR（N根据市场状态自适应）
    - 市场状态检测：基于趋势强度、波动率、成交量判断当前市场状态
    - 成交量加权评分：将成交量纳入机会评分，过滤低流动性假信号
    - 分批执行：大单自动拆分为多个小单，降低滑点
    """

    # ---- 凯利公式默认参数 ----
    KELLY_FRACTION = 0.25          # 使用1/4凯利降低回撤
    KELLY_MIN_WIN_RATE = 0.45      # 最低胜率阈值（低于此值禁用凯利）
    KELLY_MIN_SAMPLES = 10         # 最少样本数（不足时使用固定仓位）

    # ---- ATR动态止损默认参数 ----
    ATR_PERIOD = 14                # ATR计算周期
    ATR_STOP_MULTIPLIER = 2.0      # 止损ATR倍数
    ATR_TAKE_PROFIT_MULTIPLIER = 3.0  # 止盈ATR倍数
    ATR_MAX_MULTIPLIER = 4.0       # 高波动下最大ATR倍数
    ATR_MIN_MULTIPLIER = 1.5       # 低波动下最小ATR倍数

    # ---- 市场状态检测参数 ----
    REGIME_EMA_FAST = 20           # 快速EMA周期
    REGIME_EMA_SLOW = 50           # 慢速EMA周期
    REGIME_VOL_LOOKBACK = 24       # 波动率回看周期
    REGIME_HIGH_VOL_THRESHOLD = 2.0  # 高波动阈值（标准差倍数）
    REGIME_TREND_ADX_THRESHOLD = 25  # 趋势判定ADX阈值

    # ---- 分批执行参数 ----
    BATCH_MIN_NOTIONAL = 500.0     # 超过此名义价值自动拆分
    BATCH_COUNT = 3                # 默认拆分为3批
    BATCH_INTERVAL_SEC = 30        # 批次间隔30秒

    # ---- 相关性矩阵缓存 ----
    CORRELATION_CACHE_TTL = 3600   # 相关性矩阵缓存1小时
    CORRELATION_LOOKBACK = 48      # 相关性回看K线数
    CORRELATION_KLINE_BAR = "1H"   # 相关性计算用K线周期

    def __init__(self, config: Dict[str, Any], okx_client, redis_cache):
        super().__init__()
        self.config = config
        self.okx_client = okx_client
        self.redis_cache = redis_cache

        # ---- 配置加载 ----
        arb_cfg = config.get("strategies", {}).get("arbitrage", {})

        self._enabled = arb_cfg.get("enabled", False)
        self._funding_rate_threshold = arb_cfg.get("funding_rate_threshold", 0.0001)
        self._consecutive_periods = arb_cfg.get("consecutive_periods", 3)
        self._close_threshold = arb_cfg.get("close_threshold", 0.00005)
        self._leverage = arb_cfg.get("leverage", 3)
        self._hedge_leverage = arb_cfg.get("hedge_leverage", self._leverage)
        self._basis_threshold = arb_cfg.get("basis_threshold", 0.001)
        self._basis_reversal_threshold = arb_cfg.get("basis_reversal_threshold", 0.0005)
        self._rate_forecast_window = arb_cfg.get("rate_forecast_window", 8)
        self._stop_loss_pct = arb_cfg.get("stop_loss_pct", 0.02)
        self._take_profit_pct = arb_cfg.get("take_profit_pct", 0.015)
        self._max_hold_hours = arb_cfg.get("max_hold_hours", 16)
        self._min_net_profit_pct = arb_cfg.get("min_net_profit_pct", 0.0015)
        self._taker_fee = config.get("trading", {}).get("taker_fee_rate", 0.0005)
        self._max_slippage = config.get("trading", {}).get("max_slippage_pct", 0.0015)

        # ---- 生产级增强参数 ----
        self._kelly_enabled = arb_cfg.get("kelly_enabled", True)
        self._kelly_fraction = arb_cfg.get("kelly_fraction", self.KELLY_FRACTION)
        # P32: 企业级凯利引擎（贝叶斯胜率收缩 + 赔率收缩 + 半方差 + 负偏度惩罚）
        self._adaptive_kelly = AdaptiveKelly(config.get("adaptive_kelly", {}))
        self._atr_stop_enabled = arb_cfg.get("atr_stop_enabled", True)
        self._atr_stop_mult = arb_cfg.get("atr_stop_multiplier", self.ATR_STOP_MULTIPLIER)
        self._regime_adaptive = arb_cfg.get("regime_adaptive", True)
        self._batch_execution = arb_cfg.get("batch_execution", True)
        self._batch_min_notional = arb_cfg.get("batch_min_notional", self.BATCH_MIN_NOTIONAL)
        self._batch_count = arb_cfg.get("batch_count", self.BATCH_COUNT)
        self._batch_interval_sec = arb_cfg.get("batch_interval_sec", self.BATCH_INTERVAL_SEC)
        self._alert_integration = arb_cfg.get("alert_integration", True)
        self._max_net_exposure_pct = arb_cfg.get("max_net_exposure_pct", 0.05)

        # ---- TWAP 执行器 ----
        self._twap_executor: Optional[Any] = None
        if _TWAP_AVAILABLE and self._batch_execution:
            self._twap_executor = TWAPExecutor(config)
            logger.info("TWAP executor initialized for arbitrage batch execution")

        # ---- 状态存储 ----
        self._positions: Dict[str, Dict[str, Any]] = {}
        self._funding_history: Dict[str, List[float]] = {}
        self._basis_history: Dict[str, List[float]] = {}
        self._correlation_history: Dict[str, List[float]] = {}
        self._oi_history: Dict[str, List[float]] = {}
        self._volume_history: Dict[str, List[float]] = {}
        self._price_history: Dict[str, deque] = {}
        self._hedge_positions: Dict[str, Dict[str, Any]] = {}
        self._total_fee_earned: Dict[str, float] = {}
        self._rate_forecast_cache: Dict[str, float] = {}

        # ---- 市场状态 ----
        self._market_regimes: Dict[str, str] = {}
        self._atr_cache: Dict[str, float] = {}

        # ---- 相关性矩阵缓存 ----
        self._correlation_matrix: Dict[str, Dict[str, float]] = {}
        self._correlation_cache_time: float = 0

        # ---- 套利类型配置 ----
        self._arbitrage_types_enabled = arb_cfg.get("arbitrage_types", ["funding", "basis", "correlation"])
        self._check_interval = arb_cfg.get("check_interval", 900)
        self._forecast_interval = arb_cfg.get("forecast_interval", 7200)
        self._adaptive_params = arb_cfg.get("adaptive_params", True)

        # ---- 性能追踪 ----
        self._arbitrage_performance: Dict[str, Dict[str, Any]] = {
            "funding": {"wins": 0, "losses": 0, "total_pnl": 0, "count": 0,
                        "pnl_list": [], "max_drawdown": 0, "peak_equity": 0,
                        "consecutive_wins": 0, "consecutive_losses": 0},
            "basis": {"wins": 0, "losses": 0, "total_pnl": 0, "count": 0,
                      "pnl_list": [], "max_drawdown": 0, "peak_equity": 0,
                      "consecutive_wins": 0, "consecutive_losses": 0},
            "correlation": {"wins": 0, "losses": 0, "total_pnl": 0, "count": 0,
                            "pnl_list": [], "max_drawdown": 0, "peak_equity": 0,
                            "consecutive_wins": 0, "consecutive_losses": 0},
        }
        self._trade_history: deque = deque(maxlen=200)

        self._rate_prediction_model: Dict[str, Dict[str, Any]] = {}
        self._min_opportunity_score = arb_cfg.get("min_opportunity_score", 0.6)
        self._max_concurrent_arbitrage = arb_cfg.get("max_concurrent", 5)

        # 企业级增强：风险预算感知动态阈值（连续亏损/熔断时上浮机会分数阈值收紧开仓）
        self._risk_lock_quality_boost = arb_cfg.get("risk_lock_quality_boost", 0.10)
        # 企业级增强：过滤理由本地计数（get_health 暴露 + MetricsPipeline 埋点）
        self._filter_stats: Dict[str, int] = {}

        # ---- 品种列表 ----
        self._all_symbols = []
        for tier in ["tier1", "tier2", "tier3"]:
            for base in config["currencies"][f"{tier}_symbols"]:
                self._all_symbols.append(f"{base}-USDT-SWAP")

        # ---- 外部依赖注入 ----
        self._adaptive_controller = None
        self._coordinator = None
        self._risk_gate = None
        self._alert_registry = None
        self._position_provider = None

        # ---- 紧急状态 ----
        self._emergency_stop = False
        self._last_hedge_failure: Dict[str, float] = {}

        self._capital_cache_value = 0.0
        self._capital_cache_ts = 0.0
        self._capital_cache_ttl = 30.0

    # ============================================================
    # 依赖注入
    # ============================================================

    def set_adaptive_controller(self, controller):
        self._adaptive_controller = controller

    def set_coordinator(self, coordinator):
        self._coordinator = coordinator

    def set_risk_gate(self, risk_gate):
        """注入五层风控拦截器"""
        self._risk_gate = risk_gate

    def set_alert_registry(self, alert_registry):
        """注入告警注册中心"""
        self._alert_registry = alert_registry

    def set_position_provider(self, provider):
        """注入仓位提供函数，供 P22 状态漂移校验获取交易所真实仓位。"""
        self._position_provider = provider

    # ============================================================
    # 配置热更新
    # ============================================================

    async def update_config(self, updates: Dict[str, Any]):
        strategy_cfg = self.config.get("strategies", {}).get("arbitrage", {})
        strategy_cfg.update(updates)
        self.config.setdefault("strategies", {})["arbitrage"] = strategy_cfg

        attr_map = {
            "funding_rate_threshold": "funding_rate_threshold",
            "basis_threshold": "basis_threshold",
            "max_hold_hours": "max_hold_hours",
            "hedge_leverage": "hedge_leverage",
            "leverage": "leverage",
            "min_net_profit_pct": "min_net_profit_pct",
            "stop_loss_pct": "stop_loss_pct",
            "take_profit_pct": "take_profit_pct",
            "rate_forecast_window": "rate_forecast_window",
            "kelly_enabled": "kelly_enabled",
            "kelly_fraction": "kelly_fraction",
            "atr_stop_enabled": "atr_stop_enabled",
            "atr_stop_multiplier": "atr_stop_mult",
            "min_opportunity_score": "min_opportunity_score",
        }
        for cfg_key, attr_name in attr_map.items():
            if cfg_key in updates:
                setattr(self, f"_{attr_name}", updates[cfg_key])
                logger.info(f"Arbitrage config hot-updated: _{attr_name}={updates[cfg_key]}")

    def apply_param_update(self, params: Dict[str, Any]):
        applied = []
        for key, value in params.items():
            if key in self.config["strategies"].get("arbitrage", {}):
                self.config["strategies"]["arbitrage"][key] = value
                attr_name = f"_{key}"
                if hasattr(self, attr_name):
                    setattr(self, attr_name, value)
                applied.append(key)
        if applied:
            logger.info(f"Arbitrage strategy params hot-updated: {applied}")
        return applied

    # ============================================================
    # 资金管理
    # ============================================================

    def _dynamic_min_opportunity_score(self) -> float:
        """自适应机会分数阈值：连续亏损/风险熔断时上浮，收紧开仓。"""
        score = self._min_opportunity_score
        if not self._adaptive_params or self._adaptive_controller is None:
            return score
        try:
            status = self._adaptive_controller.get_risk_budget_status()
            if status.get("streak_lock_active"):
                score += self._risk_lock_quality_boost
        except Exception as e:
            # fail-closed: 风险预算状态查询失败时保守上浮阈值（收紧开仓），而非放行
            logger.warning(f"[arbitrage] 风险预算状态查询失败，保守上浮机会分数阈值: {e}")
            score += self._risk_lock_quality_boost
        return score

    def _record_filter(self, symbol: str, reason: str):
        """记录一次过滤/拒绝原因（本地计数 + MetricsPipeline 埋点）。"""
        self._filter_stats[reason] = self._filter_stats.get(reason, 0) + 1
        try:
            self._increment_metric("arbitrage_filter_total", 1.0,
                                   {"reason": reason, "symbol": symbol})
        except Exception:
            pass
        logger.debug(f"Arbitrage filter: {symbol} {reason}")

    def _get_allocation(self) -> float:
        if self._adaptive_controller:
            try:
                return self._adaptive_controller.get_allocation("arbitrage")
            except Exception as e:
                # fail-closed: 资金分配查询失败时返回 0，拒绝开仓，避免风险收缩失效
                logger.warning(f"[arbitrage] 资金分配查询失败，返回 0（fail-closed）: {e}")
                return 0.0
        return self.config["trading"].get("arbitrage_allocation", 0.15)

    def _get_effective_capital(self) -> float:
        import time
        now = time.time()
        if self._capital_cache_value > 0 and (now - self._capital_cache_ts) < self._capital_cache_ttl:
            return self._capital_cache_value
        try:
            account_info = self.okx_client.get_account_info()
            if account_info:
                details = account_info.get("details", [])
                for detail in details:
                    if detail.get("ccy") == "USDT":
                        eq = self._safe_float(detail.get("eq"), 0.0)
                        if eq > 0:
                            self._capital_cache_value = eq
                            self._capital_cache_ts = now
                            return eq
                total_eq = self._safe_float(account_info.get("totalEq"), 0.0)
                if total_eq > 0:
                    self._capital_cache_value = total_eq
                    self._capital_cache_ts = now
                    return total_eq
        except Exception as e:
            logger.warning(f"[arbitrage] get_account_info failed: {e}")
        logger.warning("[arbitrage] 账户权益查询失败，返回 0（fail-closed）")
        return 0.0

    def _calculate_kelly_position(self, arbitrage_type: str, base_position: float, symbol: str = "") -> float:
        """企业级凯利仓位计算（P32：接入 AdaptiveKelly）

        用企业级凯利引擎替代简单点估计：
        - 贝叶斯胜率收缩（向 50% 先验）+ 赔率收缩（向盈亏平衡 b=1 先验）
        - 市场状态调整 + 回撤惩罚 + 连续盈亏调整 + 样本量折扣
        - 下行半方差 / 负偏度惩罚（连续凯利，用于 PnL 序列稳健估计）

        Returns:
            调整后的仓位 = base_position × 企业级凯利乘数
        """
        if not self._kelly_enabled:
            return base_position

        perf = self._arbitrage_performance.get(arbitrage_type, {})
        total = perf.get("wins", 0) + perf.get("losses", 0)

        if total < self.KELLY_MIN_SAMPLES:
            return base_position

        win_rate = perf["wins"] / total if total > 0 else 0
        if win_rate < self.KELLY_MIN_WIN_RATE:
            return base_position * 0.5  # 胜率过低，减半仓位

        pnl_list = perf.get("pnl_list", [])
        if len(pnl_list) < 5:
            return base_position

        wins_pnl = [p for p in pnl_list if p > 0]
        losses_pnl = [abs(p) for p in pnl_list if p < 0]

        if not wins_pnl or not losses_pnl:
            return base_position

        avg_win = float(np.mean(wins_pnl))
        avg_loss = float(np.mean(losses_pnl))

        if avg_loss <= 0:
            return base_position

        # ── 市场状态（套利为市场中性的，regime 影响较弱，但作为风险因子传入）──
        regime = self._market_regimes.get(symbol, MarketRegime.UNKNOWN) if symbol else MarketRegime.UNKNOWN

        # ── 回撤（相对总权益的当前回撤比例）──
        peak_equity = float(perf.get("peak_equity", 0.0))
        cum_pnl = float(sum(pnl_list))
        total_equity = self._get_effective_capital()
        drawdown_pct = max(0.0, (peak_equity - cum_pnl) / total_equity) if total_equity > 0 else 0.0

        # ── 连续盈亏 ──
        consec_wins = int(perf.get("consecutive_wins", 0))
        consec_losses = int(perf.get("consecutive_losses", 0))

        # ── 企业级凯利引擎 ──
        result = self._adaptive_kelly.compute_kelly(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            regime=regime,
            drawdown=drawdown_pct,
            consecutive_wins=consec_wins,
            consecutive_losses=consec_losses,
            trade_count=total,
        )
        final_kelly = result["final_kelly"]
        shrunk_win_rate = result["wilson_win_rate"]
        shrunk_b = result["shrunk_odds"]

        # ── 仓位乘数：以中性凯利(5%)为基准映射 ──
        if final_kelly <= 0:
            adjusted = base_position * 0.5  # 无边际优势，减半
        else:
            adjusted = base_position * (1.0 + (final_kelly - 0.05) * 4.0)
            adjusted = max(base_position * 0.3, min(adjusted, base_position * 2.0))

        logger.debug(
            f"Enterprise Kelly sizing [{arbitrage_type}]: win_rate={win_rate:.2%} "
            f"→ shrunk={shrunk_win_rate:.2%}, b={shrunk_b:.2f}, kelly={final_kelly:.3f}, "
            f"regime={regime}, dd={drawdown_pct:.2%}, base={base_position:.2f}, adjusted={adjusted:.2f}"
        )
        return adjusted

    def _calculate_dynamic_stop_loss(self, symbol: str, entry_price: float, direction: str) -> Tuple[float, float]:
        """ATR动态止损止盈计算

        Returns:
            (stop_loss_price, take_profit_price)
        """
        if not self._atr_stop_enabled:
            # 回退到固定百分比
            if direction == "long":
                return (entry_price * (1 - self._stop_loss_pct),
                        entry_price * (1 + self._take_profit_pct))
            else:
                return (entry_price * (1 + self._stop_loss_pct),
                        entry_price * (1 - self._take_profit_pct))

        atr = self._atr_cache.get(symbol, entry_price * self._stop_loss_pct)
        regime = self._market_regimes.get(symbol, MarketRegime.UNKNOWN)

        # 根据市场状态调整ATR倍数
        if regime == MarketRegime.HIGH_VOLATILITY:
            stop_mult = min(self._atr_stop_mult * 1.5, self.ATR_MAX_MULTIPLIER)
            tp_mult = self.ATR_TAKE_PROFIT_MULTIPLIER * 1.3
        elif regime == MarketRegime.LOW_VOLATILITY:
            stop_mult = max(self._atr_stop_mult * 0.7, self.ATR_MIN_MULTIPLIER)
            tp_mult = self.ATR_TAKE_PROFIT_MULTIPLIER * 0.8
        else:
            stop_mult = self._atr_stop_mult
            tp_mult = self.ATR_TAKE_PROFIT_MULTIPLIER

        stop_distance = atr * stop_mult
        tp_distance = atr * tp_mult

        if direction == "long":
            return (entry_price - stop_distance, entry_price + tp_distance)
        else:
            return (entry_price + stop_distance, entry_price - tp_distance)

    # ============================================================
    # 市场状态检测
    # ============================================================

    def _detect_market_regime(self, symbol: str) -> str:
        """检测市场状态：趋势、震荡、高波动、低波动"""
        prices = list(self._price_history.get(symbol, []))
        volumes = self._volume_history.get(symbol, [])

        if len(prices) < self.REGIME_EMA_SLOW:
            return MarketRegime.UNKNOWN

        prices_arr = np.array(prices[-self.REGIME_EMA_SLOW:])

        # 趋势强度：EMA快慢线差值
        ema_fast = self._ema(prices_arr, self.REGIME_EMA_FAST)
        ema_slow = self._ema(prices_arr, self.REGIME_EMA_SLOW)
        trend_strength = abs(ema_fast - ema_slow) / ema_slow if ema_slow > 0 else 0

        # 波动率检测
        if len(prices) >= self.REGIME_VOL_LOOKBACK:
            recent = prices_arr[-self.REGIME_VOL_LOOKBACK:]
            with np.errstate(divide="ignore", invalid="ignore"):
                returns = np.diff(recent) / recent[:-1]
            returns = returns[np.isfinite(returns)]
            vol = float(np.std(returns)) if returns.size > 1 else 0

            if len(prices) >= self.REGIME_VOL_LOOKBACK * 2:
                window = prices_arr[-self.REGIME_VOL_LOOKBACK * 2:]
                with np.errstate(divide="ignore", invalid="ignore"):
                    hist_returns = np.diff(window) / window[:-1]
                hist_returns = hist_returns[np.isfinite(hist_returns)]
                historical_vol = float(np.std(hist_returns)) if hist_returns.size > 1 else 0
                vol_ratio = vol / historical_vol if historical_vol > 0 else 1
            else:
                vol_ratio = 1.0

            if vol_ratio > self.REGIME_HIGH_VOL_THRESHOLD:
                return MarketRegime.HIGH_VOLATILITY
            if vol_ratio < 0.5:
                return MarketRegime.LOW_VOLATILITY

        # 趋势判定
        if trend_strength > 0.02:
            if ema_fast > ema_slow:
                return MarketRegime.TRENDING_UP
            return MarketRegime.TRENDING_DOWN

        # 成交量确认
        if volumes and len(volumes) >= 10:
            recent_vol = np.mean(volumes[-5:])
            avg_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else recent_vol
            if avg_vol > 0 and recent_vol / avg_vol > 2.0:
                return MarketRegime.HIGH_VOLATILITY

        return MarketRegime.RANGING

    def _ema(self, data: np.ndarray, period: int) -> float:
        """计算EMA"""
        if len(data) < period:
            return float(np.mean(data))
        alpha = 2 / (period + 1)
        result = data[0]
        for val in data[1:]:
            result = alpha * val + (1 - alpha) * result
        return float(result)

    def _update_atr(self, symbol: str, price: float):
        """更新ATR缓存"""
        prices = self._price_history.get(symbol, deque(maxlen=self.ATR_PERIOD + 2))
        if len(prices) < 2:
            return

        # 使用简化ATR：最近N根K线的高低差均值
        if len(prices) >= self.ATR_PERIOD + 1:
            price_list = list(prices)
            true_ranges = []
            for i in range(1, min(len(price_list), self.ATR_PERIOD + 1)):
                tr = abs(price_list[i] - price_list[i - 1])
                true_ranges.append(tr)
            if true_ranges:
                atr = np.mean(true_ranges)
                self._atr_cache[symbol] = atr

    # ============================================================
    # 机会评分（增强版）
    # ============================================================

    def _calculate_opportunity_score(self, symbol: str, rate: float, arbitrage_type: str,
                                     volume_data: Dict[str, Any] = None) -> float:
        """增强版机会评分：包含成交量加权、O/I变化、市场状态"""
        history = self._funding_history.get(symbol, [])

        if not history:
            return 0.5

        avg_rate = np.mean(history)
        std_rate = np.std(history) if len(history) > 1 else abs(avg_rate) * 0.1

        score = 0.0

        # ---- 1. 费率偏离度 (40%) ----
        abs_rate = abs(rate)
        threshold = self._funding_rate_threshold

        if abs_rate >= threshold:
            score += 0.40
        elif abs_rate >= threshold * 0.7:
            score += 0.25
        elif abs_rate >= threshold * 0.5:
            score += 0.10

        # ---- 2. Z-score统计显著性 (20%) ----
        if std_rate > 0:
            z_score = abs(abs_rate - abs(avg_rate)) / std_rate
            score += min(0.20, z_score * 0.08)

        # ---- 3. 连续同向周期 (15%) ----
        consecutive_count = 0
        sign = 1 if rate > 0 else -1
        for r in reversed(history):
            if r * sign > 0:
                consecutive_count += 1
            else:
                break
        score += min(0.15, consecutive_count * 0.04)

        # ---- 4. 预测方向确认 (15%) ----
        forecast = self._rate_forecast_cache.get(symbol, rate)
        if abs(forecast) > abs(rate) * 1.1:
            score += 0.15
        elif abs(forecast) > abs(rate) * 0.9:
            score += 0.08

        # ---- 5. 成交量确认 (10%) ----
        if volume_data:
            try:
                vol_24h = float(volume_data.get("vol24h", 0))
                vol_usd_24h = float(volume_data.get("volCcy24h", 0))
                if vol_usd_24h > 500000:  # 24h成交额 > 50万USDT
                    score += 0.05
                if vol_usd_24h > 5000000:  # > 500万USDT
                    score += 0.05
            except (ValueError, TypeError):
                pass

        # ---- 6. 历史表现修正 (±10%) ----
        perf = self._arbitrage_performance.get(arbitrage_type, {})
        total = perf.get("wins", 0) + perf.get("losses", 0)
        if total >= 5:
            win_rate = perf["wins"] / total
            score += (win_rate - 0.5) * 0.10

        # ---- 7. 市场状态调整 ----
        regime = self._market_regimes.get(symbol, MarketRegime.UNKNOWN)
        if regime == MarketRegime.HIGH_VOLATILITY:
            score *= 0.7  # 高波动降低信心
        elif regime == MarketRegime.LOW_VOLATILITY:
            score *= 1.1  # 低波动略微增加信心
        elif regime == MarketRegime.RANGING:
            score *= 0.9

        return min(1.0, max(0.0, score))

    def _calculate_basis_score(self, symbol: str, basis: float) -> float:
        """基差套利机会评分"""
        history = self._basis_history.get(symbol, [])
        if len(history) < 5:
            return 0.5

        avg_basis = np.mean(history)
        std_basis = np.std(history) if len(history) >= 8 else abs(avg_basis) * 0.3

        score = 0.0

        # 偏离度
        if std_basis > 0:
            z_score = abs(basis - avg_basis) / std_basis
            score += min(0.5, z_score * 0.2)

        # 绝对值
        if abs(basis) >= self._basis_threshold:
            score += 0.3
        elif abs(basis) >= self._basis_threshold * 0.5:
            score += 0.15

        # 费率预测辅助
        forecast = self._rate_forecast_cache.get(symbol, 0)
        if basis > 0 and forecast < 0:
            score += 0.1
        elif basis < 0 and forecast > 0:
            score += 0.1

        regime = self._market_regimes.get(symbol, MarketRegime.UNKNOWN)
        if regime == MarketRegime.HIGH_VOLATILITY:
            score *= 0.6
        elif regime == MarketRegime.RANGING:
            score *= 1.15  # 震荡市基差套利更可靠

        return min(1.0, max(0.0, score))

    def _calculate_correlation_score(self, z_score: float, pair_key: str) -> float:
        """相关性套利机会评分"""
        score = 0.0

        abs_z = abs(z_score)
        if abs_z >= 3.0:
            score += 0.5
        elif abs_z >= 2.5:
            score += 0.4
        elif abs_z >= 2.0:
            score += 0.25

        # 相关性强度
        corr = self._get_pair_correlation(pair_key)
        if corr is not None and abs(corr) > 0.8:
            score += 0.3
        elif corr is not None and abs(corr) > 0.6:
            score += 0.15

        # 历史表现
        perf = self._arbitrage_performance.get("correlation", {})
        total = perf.get("wins", 0) + perf.get("losses", 0)
        if total >= 5:
            win_rate = perf["wins"] / total
            score += (win_rate - 0.5) * 0.15

        return min(1.0, max(0.0, score))

    # ============================================================
    # 相关性矩阵缓存
    # ============================================================

    def _get_pair_correlation(self, pair_key: str) -> Optional[float]:
        """获取缓存的相关性系数"""
        now = time.time()
        if now - self._correlation_cache_time > self.CORRELATION_CACHE_TTL:
            self._correlation_matrix = {}
            self._correlation_cache_time = 0

        sym_a, sym_b = pair_key.split(":")
        if sym_a in self._correlation_matrix and sym_b in self._correlation_matrix[sym_a]:
            return self._correlation_matrix[sym_a][sym_b]
        return None

    def _set_pair_correlation(self, sym_a: str, sym_b: str, corr: float):
        """设置相关性系数缓存"""
        if sym_a not in self._correlation_matrix:
            self._correlation_matrix[sym_a] = {}
        self._correlation_matrix[sym_a][sym_b] = corr
        self._correlation_cache_time = time.time()

    async def _update_correlation_matrix(self, pairs: List[Tuple[str, str]]):
        """更新相关性矩阵"""
        all_symbols = set()
        for a, b in pairs:
            all_symbols.add(a)
            all_symbols.add(b)

        # 收集价格数据
        price_data: Dict[str, List[float]] = {}
        for sym in all_symbols:
            try:
                klines = self.okx_client.get_klines(sym, bar=self.CORRELATION_KLINE_BAR,
                                                     limit=self.CORRELATION_LOOKBACK)
                if klines and len(klines) >= 10:
                    price_data[sym] = [float(k[4]) for k in klines]  # 收盘价
            except Exception:
                continue

        # 计算相关性
        for sym_a, sym_b in pairs:
            if sym_a not in price_data or sym_b not in price_data:
                continue
            prices_a = np.array(price_data[sym_a], dtype=float)
            prices_b = np.array(price_data[sym_b], dtype=float)

            # 过滤非正价格（脏数据防御），避免 np.log 产生 nan/-inf
            valid_mask = (prices_a > 0) & (prices_b > 0)
            prices_a = prices_a[valid_mask]
            prices_b = prices_b[valid_mask]

            min_len = min(len(prices_a), len(prices_b))
            if min_len < 10:
                continue

            # 使用对数收益率计算相关性
            with np.errstate(divide="ignore", invalid="ignore"):
                returns_a = np.diff(np.log(prices_a[:min_len]))
                returns_b = np.diff(np.log(prices_b[:min_len]))
            returns_a = returns_a[np.isfinite(returns_a)]
            returns_b = returns_b[np.isfinite(returns_b)]
            if returns_a.size < 2 or returns_b.size < 2:
                continue

            with np.errstate(divide="ignore", invalid="ignore"):
                corr_matrix = np.corrcoef(returns_a, returns_b)
            if corr_matrix.shape == (2, 2):
                corr = float(corr_matrix[0, 1])
                if not np.isfinite(corr):
                    continue
                self._set_pair_correlation(sym_a, sym_b, corr)
                logger.debug(f"Correlation {sym_a}-{sym_b}: {corr:.4f}")

    # ============================================================
    # 告警集成
    # ============================================================

    def _send_alert(self, severity: str, message: str, metrics: Dict[str, Any] = None):
        """通过AlertRegistry发送告警"""
        if not self._alert_integration or not self._alert_registry:
            return

        try:
            alert_metrics = {
                "arbitrage." + (metrics.get("metric", "status") if metrics else "status"):
                    metrics.get("value", 1) if metrics else 1
            }
            self._alert_registry.evaluate(alert_metrics)
            logger.warning(f"[Arbitrage Alert] [{severity}] {message}")
        except Exception as e:
            logger.debug(f"Alert send failed (non-critical): {e}")

    # ============================================================
    # 策略启动与主循环
    # ============================================================

    async def start(self):
        if not self._enabled:
            logger.info("Arbitrage strategy is disabled")
            return

        # 状态持久化初始化
        self.init_state_persistence("arbitrage", self.redis_cache)
        await self.load_state_async()
        asyncio.create_task(self.periodic_save_loop())

        # 对冲完整性检查
        asyncio.create_task(self._hedge_integrity_loop())

        # 市场状态更新
        asyncio.create_task(self._market_regime_loop())

        # 相关性矩阵更新
        asyncio.create_task(self._correlation_update_loop())

        logger.info("Starting Production-Grade Arbitrage Strategy v2.0")
        asyncio.create_task(self._monitor_loop())
        asyncio.create_task(self._forecast_loop())

    async def _market_regime_loop(self):
        """定期更新市场状态"""
        while True:
            try:
                for symbol in self._all_symbols:
                    regime = self._detect_market_regime(symbol)
                    self._market_regimes[symbol] = regime
                logger.debug(f"Market regimes updated: {len(self._market_regimes)} symbols")
            except Exception as e:
                logger.error(f"Market regime update error: {e}")
            await asyncio.sleep(300)  # 每5分钟更新

    async def _correlation_update_loop(self):
        """定期更新相关性矩阵"""
        correlation_pairs = [
            ("BTC-USDT-SWAP", "ETH-USDT-SWAP"),
            ("BTC-USDT-SWAP", "SOL-USDT-SWAP"),
            ("ETH-USDT-SWAP", "BNB-USDT-SWAP"),
            ("LTC-USDT-SWAP", "BTC-USDT-SWAP"),
            ("LINK-USDT-SWAP", "UNI-USDT-SWAP"),
            ("ADA-USDT-SWAP", "AVAX-USDT-SWAP"),
        ]
        while True:
            try:
                await self._update_correlation_matrix(correlation_pairs)
                logger.debug(f"Correlation matrix updated for {len(correlation_pairs)} pairs")
            except Exception as e:
                logger.error(f"Correlation update error: {e}")
            await asyncio.sleep(self.CORRELATION_CACHE_TTL)

    async def verify_exchange_positions(self) -> Dict[str, Any]:
        """P22: 状态无漂移 - 校验本地self._positions与交易所真实仓位一致性
        
        每tick/每K线前由StrategyContainer调用，确保套利策略不依赖内存变量记仓位。
        检测到漂移时自动清理本地幽灵仓位记录。
        """
        corrections = []
        try:
            positions = None
            if self._position_provider:
                try:
                    positions = self._position_provider()
                except Exception:
                    positions = None
            if positions is None:
                try:
                    positions = await self.okx_client.get_positions_async()
                except Exception:
                    positions = None
            if positions is None:
                # fail-closed: 无法获取交易所仓位时，不执行幽灵清理，避免误清真实持仓
                logger.warning("[arbitrage] P22 无法获取交易所仓位，跳过幽灵仓位校验（fail-closed）")
                return {"drifted": False, "corrections": [], "exchange_positions": []}
            if not positions:
                positions = []
            
            # 构建交易所仓位映射
            exchange_positions: Dict[str, Dict[str, float]] = {}
            for pos_data in positions:
                try:
                    pos = self.okx_client._parse_position(pos_data)
                    if pos and abs(pos.quantity) > 0:
                        exchange_positions.setdefault(pos.symbol, {})[pos.side] = abs(pos.quantity)
                except Exception:
                    continue
            
            # 校验本地self._positions中标记为"open"的仓位
            for symbol, local_pos in list(self._positions.items()):
                if local_pos.get("status") != "open":
                    continue
                # 相关性套利持仓 key 为 "A:B"，方向为 "long/short" 组合，无法映射交易所单一仓位结构，
                # 跳过幽灵仓位判定，避免误清理合法双腿持仓。
                if local_pos.get("arbitrage_type") == "correlation":
                    continue
                direction = local_pos.get("direction", "")
                exchange_qty = exchange_positions.get(symbol, {}).get(direction, 0)
                if exchange_qty <= 0:
                    logger.warning(
                        f"P22: Ghost position detected in arbitrage: {symbol} {direction}, "
                        f"local status=open but exchange has no position. Cleaning up."
                    )
                    self._positions[symbol]["status"] = "closed"
                    self._positions[symbol]["close_reason"] = "ghost_position_cleaned"
                    corrections.append({
                        "symbol": symbol,
                        "action": "ghost_cleaned",
                        "direction": direction,
                    })
            
            # 清理超时未更新的"open"仓位（超过24小时无活动视为幽灵）
            stale_threshold = time.time() - 86400
            for symbol, local_pos in list(self._positions.items()):
                if local_pos.get("status") == "open":
                    last_update = local_pos.get("last_update", 0)
                    if last_update < stale_threshold:
                        logger.warning(
                            f"P22: Stale position cleaned in arbitrage: {symbol}, "
                            f"age={time.time() - last_update:.0f}s"
                        )
                        self._positions[symbol]["status"] = "closed"
                        self._positions[symbol]["close_reason"] = "stale_position_cleaned"
                        corrections.append({
                            "symbol": symbol,
                            "action": "stale_cleaned",
                            "age_seconds": time.time() - last_update,
                        })
            
            return {
                "drifted": len(corrections) > 0,
                "corrections": corrections,
                "exchange_positions": list(exchange_positions.keys()),
            }
        except Exception as e:
            logger.debug(f"P22: Arbitrage position verification error: {e}")
            return {"drifted": False, "corrections": [], "exchange_positions": []}

    async def _monitor_loop(self):
        while True:
            if self._emergency_stop:
                logger.warning("Arbitrage emergency stop active, skipping monitor cycle")
                await asyncio.sleep(self._check_interval)
                continue

            try:
                # 净敞口硬限制检查（最高优先级）
                exposure_ok = await self._check_net_exposure_hard_limit()
                if not exposure_ok:
                    logger.warning("Net exposure hard limit triggered, skipping new opportunities this cycle")
                    await self._manage_positions()
                    await asyncio.sleep(self._check_interval)
                    continue

                open_count = sum(1 for p in self._positions.values() if p.get("status") == "open")
                if open_count < self._max_concurrent_arbitrage:
                    if "funding" in self._arbitrage_types_enabled:
                        await self._check_funding_rates()
                    if "basis" in self._arbitrage_types_enabled:
                        await self._check_basis_arbitrage()
                    if "correlation" in self._arbitrage_types_enabled:
                        await self._check_correlation_arbitrage()
                await self._manage_positions()
            except Exception as e:
                logger.error(f"Arbitrage monitor loop error: {e}")
                self._send_alert("warning", f"Monitor loop error: {e}", {"metric": "monitor_error", "value": 1})
            await asyncio.sleep(self._check_interval)

    # ============================================================
    # 对冲完整性检查
    # ============================================================

    async def _hedge_integrity_loop(self):
        """周期性对账任务，每5分钟检查净敞口"""
        while True:
            try:
                await asyncio.sleep(300)
                await self._check_hedge_integrity()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Arbitrage hedge integrity loop error: {e}")
                await asyncio.sleep(60)

    async def _check_hedge_integrity(self):
        """检查净敞口，非0则告警并自动修复"""
        for symbol, pos in list(self._positions.items()):
            if pos.get("status") != "open":
                continue
            arb_type = pos.get("arbitrage_type", "funding")
            if arb_type == "correlation":
                continue

            if symbol not in self._hedge_positions:
                logger.warning(f"[Hedge Integrity] {symbol} ({arb_type}) 主仓open但缺失对冲仓，净敞口={pos.get('quantity', 0)}，立即平仓主仓")
                self._send_alert("critical",
                    f"Hedge missing for {symbol}, closing main position",
                    {"metric": "hedge_missing", "value": 1})
                try:
                    await self._close_position(symbol, "hedge_missing")
                except Exception as e:
                    logger.error(f"[Hedge Integrity] 关闭 {symbol} 主仓失败: {e}")
                continue

            hedge = self._hedge_positions[symbol]
            main_qty = float(pos.get("quantity", 0))
            hedge_qty = float(hedge.get("quantity", 0))
            net_exposure = abs(main_qty - hedge_qty)

            if main_qty > 0 and net_exposure / main_qty > 0.05:
                logger.warning(
                    f"[Hedge Integrity] {symbol} ({arb_type}) 净敞口失衡: "
                    f"main={main_qty:.6f}, hedge={hedge_qty:.6f}, net={net_exposure:.6f} "
                    f"({net_exposure/main_qty:.2%})"
                )
                self._send_alert("warning",
                    f"Hedge imbalance for {symbol}: {net_exposure/main_qty:.2%}",
                    {"metric": "hedge_imbalance", "value": net_exposure / main_qty})

    # ============================================================
    # 费率预测
    # ============================================================

    async def _forecast_loop(self):
        while True:
            try:
                await self._update_rate_forecasts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Arbitrage forecast loop error: {e}")
            await asyncio.sleep(self._forecast_interval)

    async def _update_rate_forecasts(self):
        for symbol in self._all_symbols:
            try:
                history = self._funding_history.get(symbol, [])
                if len(history) >= self._rate_forecast_window:
                    oi_history = self._oi_history.get(symbol, None)
                    forecast = self._predict_next_rate(history, oi_history)
                    self._rate_forecast_cache[symbol] = forecast
            except Exception as e:
                logger.debug(f"[arbitrage] forecast update failed for {symbol}: {e}")

    def _predict_next_rate(self, history: List[float], oi_history: List[float] = None) -> float:
        if len(history) < 3:
            return history[-1] if history else 0

        recent = np.array(history[-self._rate_forecast_window:])

        avg_rate = np.mean(recent)
        std_rate = np.std(recent)

        weights = np.linspace(0.5, 1.5, len(recent))
        weights = weights / np.sum(weights)
        weighted_avg = np.sum(recent * weights)

        if len(recent) >= 5:
            x = np.arange(len(recent))
            slope, _ = np.polyfit(x, recent, 1)
        else:
            slope = 0

        recent_slope = 0
        if len(recent) >= 4:
            recent_slope = (recent[-1] - recent[-4]) / 3

        trend_weight = 0.6
        if oi_history and len(oi_history) >= 2:
            oi_change = (oi_history[-1] - oi_history[-2]) / max(abs(oi_history[-2]), 1)
            if oi_change > 0:
                trend_weight = min(1.0, trend_weight * 1.15)

        trend_component = slope * trend_weight + recent_slope * (1.0 - trend_weight)
        momentum = recent[-1] - avg_rate
        mean_reversion = -momentum * 0.3

        predicted = weighted_avg + trend_component + mean_reversion
        return max(min(predicted, avg_rate + std_rate * 2.5), avg_rate - std_rate * 2.5)

    # ============================================================
    # 资金费率套利
    # ============================================================

    async def _check_funding_rates(self):
        for symbol in self._all_symbols:
            if symbol in self._positions:
                continue

            funding_data = await self.okx_client.get_funding_rate_async(symbol)
            if not funding_data:
                continue

            rate = self._safe_float(funding_data.get("fundingRate"))

            if symbol not in self._funding_history:
                self._funding_history[symbol] = []

            self._funding_history[symbol].append(rate)
            if len(self._funding_history[symbol]) > self._consecutive_periods * 2:
                self._funding_history[symbol] = self._funding_history[symbol][-self._consecutive_periods * 2:]

            # 收集OI数据
            try:
                oi_data = funding_data.get("openInterest")
                if oi_data is not None:
                    if symbol not in self._oi_history:
                        self._oi_history[symbol] = []
                    self._oi_history[symbol].append(float(oi_data))
                    if len(self._oi_history[symbol]) > self._rate_forecast_window * 2:
                        self._oi_history[symbol] = self._oi_history[symbol][-self._rate_forecast_window * 2:]
            except Exception:
                pass

            if await self._check_arbitrage_condition(symbol, rate):
                # 获取成交量数据用于评分
                volume_data = None
                try:
                    ticker = await self.okx_client.get_ticker_async(symbol)
                    if ticker:
                        volume_data = ticker
                except Exception:
                    pass

                opportunity_score = self._calculate_opportunity_score(symbol, rate, "funding", volume_data)
                if opportunity_score >= self._dynamic_min_opportunity_score():
                    await self._generate_arbitrage_signal(symbol, rate, opportunity_score)

    async def _check_arbitrage_condition(self, symbol: str, rate: float) -> bool:
        history = self._funding_history.get(symbol, [])

        if len(history) < max(3, self._consecutive_periods // 2):
            return False

        forecast_rate = self._rate_forecast_cache.get(symbol, rate)

        recent = history[-min(self._consecutive_periods, len(history)):]
        avg_recent = np.mean(recent)

        same_direction_count = sum(1 for r in recent if r * rate > 0)
        direction_ratio = same_direction_count / len(recent) if recent else 0

        avg_above = abs(avg_recent) >= self._funding_rate_threshold * 0.6
        current_above = abs(rate) >= self._funding_rate_threshold * 0.8
        forecast_above = abs(forecast_rate) >= self._funding_rate_threshold * 0.4

        if symbol in self._positions:
            return False

        return direction_ratio >= 0.6 and (avg_above or current_above) and forecast_above

    async def _generate_arbitrage_signal(self, symbol: str, rate: float, opportunity_score: float = 0.0):
        # 企业级：参数前置校验，symbol 非法或 rate 非有限值直接拒绝
        if not self._validate_symbol(symbol):
            logger.warning(f"Arbitrage signal rejected: invalid symbol {symbol!r}")
            self._increment_metric("arbitrage_signal_rejected_total", 1.0, {"reason": "invalid_params", "symbol": symbol})
            return
        rate = self._safe_float(rate, default=0.0)

        if symbol in self._positions:
            self._record_filter(symbol, "position_exists")
            return

        # 检查币种是否被L5熔断冻结
        if self._risk_gate and self._risk_gate.is_symbol_frozen(symbol):
            logger.debug(f"Arbitrage skipped {symbol}: symbol frozen by L5 circuit breaker")
            self._record_filter(symbol, "symbol_frozen")
            return

        tier = get_currency_tier(symbol, self.config)
        tier_settings = self.config["currencies"][f"{tier}_settings"]

        total_capital = self._get_effective_capital()
        trading_capital = total_capital * self.config["trading"]["trading_capital_ratio"]
        allocation = self._get_allocation()
        position_limit = tier_settings["position_limit"]

        base_position = trading_capital * min(allocation, position_limit)

        # 凯利公式调整
        base_position = self._calculate_kelly_position("funding", base_position, symbol)

        # 空闲资金放大（受 adaptive_params 门控）
        if self._adaptive_params and self._adaptive_controller:
            try:
                boost = self._adaptive_controller.get_position_boost()
                if boost > 1.0:
                    base_position *= boost
            except Exception:
                pass

        # 降杠杆倍数调整
        if self._risk_gate:
            base_position *= self._risk_gate.get_leverage_multiplier()

        ticker = await self.okx_client.get_ticker_async(symbol)
        if not ticker:
            self._record_filter(symbol, "no_ticker")
            return

        price = self._safe_float(ticker.get("last"))
        if price <= 0:
            logger.debug(f"Arbitrage skipped {symbol}: invalid price {price!r}")
            self._record_filter(symbol, "invalid_price")
            return
        quantity = base_position / price

        # 更新价格历史和ATR
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=100)
        self._price_history[symbol].append(price)
        self._update_atr(symbol, price)

        # 更新成交量
        try:
            vol = float(ticker.get("vol24h", 0))
            if symbol not in self._volume_history:
                self._volume_history[symbol] = []
            self._volume_history[symbol].append(vol)
            if len(self._volume_history[symbol]) > 48:
                self._volume_history[symbol] = self._volume_history[symbol][-48:]
        except Exception:
            pass

        if rate > 0:
            direction = "short"
        else:
            direction = "long"

        forecast_rate = self._rate_forecast_cache.get(symbol, rate)

        # 净收益检查
        position_value = base_position * self._leverage
        total_cost_pct = self._taker_fee * 4
        total_cost = position_value * total_cost_pct

        expected_funding = abs(rate) * position_value * 1.0
        forecast_contribution = 0
        if forecast_rate * rate > 0:
            forecast_contribution = abs(forecast_rate) * position_value * 0.5
        gross_earnings = expected_funding + forecast_contribution
        net_earnings = gross_earnings - total_cost
        net_earnings_pct = net_earnings / position_value if position_value > 0 else 0

        if net_earnings <= 0:
            logger.debug(f"Arbitrage skipped {symbol}: net_earnings={net_earnings:.4f} <= 0")
            self._record_filter(symbol, "net_earnings_insufficient")
            return

        # ATR动态止损
        stop_loss, take_profit = self._calculate_dynamic_stop_loss(symbol, price, direction)

        signal = Signal(
            symbol=symbol,
            strategy_name="arbitrage",
            signal_type="arbitrage_entry",
            direction=direction,
            price=price,
            quantity=quantity,
            leverage=self._leverage,
            confidence=min(0.9, 0.7 + abs(rate) * 100),
            timestamp=datetime.now()
        )

        arb_id = f"arb_{int(datetime.now().timestamp() * 1000)}_{uuid.uuid4().hex[:8]}"

        self.redis_cache.publish_signal({
            "type": "signal",
            "data": {
                "symbol": signal.symbol,
                "strategy_name": signal.strategy_name,
                "signal_type": signal.signal_type,
                "direction": signal.direction,
                "price": signal.price,
                "quantity": signal.quantity,
                "leverage": signal.leverage,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "confidence": signal.confidence,
                "timestamp": signal.timestamp.isoformat(),
                "funding_rate": rate,
                "forecast_rate": forecast_rate,
                "arb_id": arb_id
            }
        })

        # 原子对冲
        hedge_ok = await self._place_hedge_position(symbol, direction, price, quantity, arb_id)
        if not hedge_ok:
            logger.error(f"Hedge failed for {symbol} (arb_id={arb_id}), closing main position to avoid single-side exposure")
            self._send_alert("critical",
                f"Hedge failed for {symbol}, emergency close main position",
                {"metric": "hedge_failed", "value": 1})
            # 记录对冲失败次数，防止反复重试
            now = time.time()
            last_fail = self._last_hedge_failure.get(symbol, 0)
            if now - last_fail < 300:  # 5分钟内连续失败则冻结该币种
                if self._risk_gate:
                    self._risk_gate.freeze_symbol(symbol, f"连续对冲失败: {symbol}")
            self._last_hedge_failure[symbol] = now
            await self._emergency_close_main(symbol, direction, price, quantity, arb_id)
            return

        self._positions[symbol] = {
            "direction": direction,
            "entry_price": price,
            "quantity": quantity,
            "entry_rate": rate,
            "forecast_rate": forecast_rate,
            "entry_time": datetime.now(),
            "status": "open",
            "expected_earnings": gross_earnings,
            "total_cost": total_cost,
            "net_earnings": net_earnings,
            "actual_earnings": 0,
            "arbitrage_type": "funding",
            "arb_id": arb_id,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr_at_entry": self._atr_cache.get(symbol, 0),
            "last_update": time.time(),
            "opportunity_score": opportunity_score,
        }

        self._total_fee_earned[symbol] = 0

        self._record_metric("arbitrage_signal_generated_total", 1.0, {"symbol": symbol, "direction": direction, "type": "funding"})
        self._record_metric("arbitrage_signal_confidence", signal.confidence, {"symbol": symbol, "direction": direction, "type": "funding"})

        logger.info(f"Arbitrage signal: {direction} {symbol} @ {price:.4f}, rate: {rate:.4%}, "
                    f"forecast: {forecast_rate:.4%}, gross: {gross_earnings:.2f}, cost: {total_cost:.2f}, "
                    f"net: {net_earnings:.2f} ({net_earnings_pct:.4%}), score: {opportunity_score:.2f}, arb_id={arb_id}")

    # ============================================================
    # 基差套利
    # ============================================================

    async def _check_basis_arbitrage(self):
        for symbol in self._all_symbols:
            if symbol in self._positions:
                continue

            futures_ticker = await self.okx_client.get_ticker_async(symbol)
            spot_symbol = symbol.replace("-SWAP", "") if symbol.endswith("-SWAP") else symbol
            spot_ticker = await self.okx_client.get_ticker_async(spot_symbol)

            if not futures_ticker or not spot_ticker:
                continue

            futures_price = self._safe_float(futures_ticker.get("last"))
            spot_price = self._safe_float(spot_ticker.get("last"))

            if futures_price <= 0 or spot_price <= 0:
                continue

            basis = (futures_price - spot_price) / spot_price

            if symbol not in self._basis_history:
                self._basis_history[symbol] = []

            self._basis_history[symbol].append(basis)
            if len(self._basis_history[symbol]) > 24:
                self._basis_history[symbol] = self._basis_history[symbol][-24:]

            # 更新市场数据
            if symbol not in self._price_history:
                self._price_history[symbol] = deque(maxlen=100)
            self._price_history[symbol].append(futures_price)
            self._update_atr(symbol, futures_price)

            opportunity_score = self._calculate_basis_score(symbol, basis)
            if opportunity_score >= self._dynamic_min_opportunity_score():
                await self._check_basis_condition(symbol, basis, futures_price, spot_price, opportunity_score)

    async def _check_basis_condition(self, symbol: str, basis: float, futures_price: float,
                                      spot_price: float, opportunity_score: float):
        history = self._basis_history.get(symbol, [])

        if len(history) < 5:
            return

        avg_basis = np.mean(history)
        std_basis = np.std(history) if len(history) >= 8 else abs(avg_basis) * 0.3

        basis_deviation = abs(basis - avg_basis)
        strong_signal = basis_deviation > std_basis * 1.5 and abs(basis) > self._basis_threshold
        moderate_signal = basis_deviation > std_basis and abs(basis) > self._basis_threshold * 0.7 and len(history) >= 10

        if strong_signal or moderate_signal:
            forecast_rate = self._rate_forecast_cache.get(symbol, 0)

            if basis > 0 and forecast_rate < 0:
                await self._generate_basis_signal(symbol, "long", futures_price, spot_price, basis, opportunity_score)
            elif basis < 0 and forecast_rate > 0:
                await self._generate_basis_signal(symbol, "short", futures_price, spot_price, basis, opportunity_score)

    async def _generate_basis_signal(self, symbol: str, direction: str, futures_price: float,
                                      spot_price: float, basis: float, opportunity_score: float = 0.0):
        # 企业级：参数前置校验
        if not self._validate_symbol(symbol) or not self._validate_direction(direction) \
                or not self._validate_price(futures_price):
            logger.warning(
                f"Basis signal rejected: invalid params "
                f"symbol={symbol!r} direction={direction!r} futures_price={futures_price!r}"
            )
            self._increment_metric("arbitrage_signal_rejected_total", 1.0, {"reason": "invalid_params", "symbol": symbol})
            return

        if symbol in self._positions:
            self._record_filter(symbol, "position_exists")
            return

        if self._risk_gate and self._risk_gate.is_symbol_frozen(symbol):
            self._record_filter(symbol, "symbol_frozen")
            return

        tier = get_currency_tier(symbol, self.config)
        tier_settings = self.config["currencies"][f"{tier}_settings"]

        total_capital = self._get_effective_capital()
        trading_capital = total_capital * self.config["trading"]["trading_capital_ratio"]
        allocation = self._get_allocation()
        position_limit = tier_settings["position_limit"]

        base_position = trading_capital * min(allocation, position_limit)
        base_position = self._calculate_kelly_position("basis", base_position, symbol)

        if self._risk_gate:
            base_position *= self._risk_gate.get_leverage_multiplier()

        quantity = base_position / futures_price

        basis_pnl = abs(basis) * base_position

        history = self._basis_history.get(symbol, [])
        avg_basis = float(np.mean(history)) if history else basis
        basis_std = float(np.std(history)) if len(history) >= 8 else abs(avg_basis) * 0.3
        if basis_std <= 0:
            basis_std = abs(basis) * 0.3 if abs(basis) > 0 else 0.001

        # ATR动态止损
        stop_loss, take_profit = self._calculate_dynamic_stop_loss(symbol, futures_price, direction)

        signal = Signal(
            symbol=symbol,
            strategy_name="arbitrage",
            signal_type="arbitrage_basis_entry",
            direction=direction,
            price=futures_price,
            quantity=quantity,
            leverage=self._leverage,
            confidence=min(0.85, 0.6 + abs(basis) * 200),
            timestamp=datetime.now()
        )

        arb_id = f"arb_{int(datetime.now().timestamp() * 1000)}_{uuid.uuid4().hex[:8]}"

        self.redis_cache.publish_signal({
            "type": "signal",
            "data": {
                "symbol": signal.symbol,
                "strategy_name": signal.strategy_name,
                "signal_type": signal.signal_type,
                "direction": signal.direction,
                "price": signal.price,
                "quantity": signal.quantity,
                "leverage": signal.leverage,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "confidence": signal.confidence,
                "timestamp": signal.timestamp.isoformat(),
                "basis": basis,
                "spot_price": spot_price,
                "arb_id": arb_id
            }
        })

        hedge_qty = await self._calculate_hedge_quantity(symbol, quantity, futures_price, "basis")
        hedge_ok = await self._place_hedge_position(symbol, direction, futures_price, hedge_qty, arb_id)
        if not hedge_ok:
            logger.error(f"Hedge failed for basis arb {symbol} (arb_id={arb_id}), closing main position")
            self._send_alert("critical", f"Basis hedge failed for {symbol}",
                            {"metric": "basis_hedge_failed", "value": 1})
            await self._emergency_close_main(symbol, direction, futures_price, quantity, arb_id)
            return

        self._positions[symbol] = {
            "direction": direction,
            "entry_price": futures_price,
            "entry_spot_price": spot_price,
            "quantity": quantity,
            "entry_rate": 0,
            "forecast_rate": 0,
            "entry_time": datetime.now(),
            "status": "open",
            "expected_earnings": basis_pnl,
            "actual_earnings": 0,
            "arbitrage_type": "basis",
            "entry_basis": basis,
            "basis_std": basis_std,
            "arb_id": arb_id,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr_at_entry": self._atr_cache.get(symbol, 0),
            "last_update": time.time(),
            "opportunity_score": opportunity_score,
        }

        self._total_fee_earned[symbol] = 0

        self._record_metric("arbitrage_signal_generated_total", 1.0, {"symbol": symbol, "direction": direction, "type": "basis"})
        self._record_metric("arbitrage_signal_confidence", signal.confidence, {"symbol": symbol, "direction": direction, "type": "basis"})

        logger.info(f"Basis arbitrage signal: {direction} {symbol} @ {futures_price:.4f}, spot: {spot_price:.4f}, "
                    f"basis: {basis:.4%}, basis_std: {basis_std:.4%}, expected: {basis_pnl:.2f} USDT, "
                    f"score: {opportunity_score:.2f}, arb_id={arb_id}")

    async def _calculate_hedge_quantity(self, symbol: str, main_qty: float, futures_price: float, arb_type: str) -> float:
        if arb_type == "basis":
            return main_qty

        try:
            ticker = await self.okx_client.get_ticker_async(symbol)
            if ticker and float(ticker.get("last", 0)) > 0:
                spot_price = float(ticker["last"])
                hedge_qty = (main_qty * futures_price) / spot_price
                hedge_qty = round(hedge_qty / 0.001) * 0.001
                return max(hedge_qty, 0.001)
        except Exception as e:
            logger.error(f"Failed to calculate hedge quantity: {e}")

        return main_qty

    async def _place_hedge_position(self, symbol: str, direction: str, price: float, quantity: float,
                                     arb_id: str = None) -> bool:
        hedge_direction = "buy" if direction == "short" else "sell"

        hedge_signal = Signal(
            symbol=symbol,
            strategy_name="arbitrage",
            signal_type="arbitrage_hedge",
            direction=hedge_direction,
            price=price,
            quantity=quantity,
            leverage=self._hedge_leverage,
            confidence=0.85,
            timestamp=datetime.now()
        )

        try:
            self.redis_cache.publish_signal({
                "type": "signal",
                "data": {
                    "symbol": hedge_signal.symbol,
                    "strategy_name": hedge_signal.strategy_name,
                    "signal_type": hedge_signal.signal_type,
                    "direction": hedge_signal.direction,
                    "price": hedge_signal.price,
                    "quantity": hedge_signal.quantity,
                    "leverage": hedge_signal.leverage,
                    "stop_loss": None,
                    "take_profit": None,
                    "confidence": hedge_signal.confidence,
                    "timestamp": hedge_signal.timestamp.isoformat(),
                    "arb_id": arb_id,
                    "reduce_only": False
                }
            })
        except Exception as e:
            logger.error(f"Hedge publish failed for {symbol} (arb_id={arb_id}): {e}")
            return False

        self._hedge_positions[symbol] = {
            "direction": hedge_direction,
            "symbol": symbol,
            "price": price,
            "quantity": quantity,
            "timestamp": datetime.now(),
            "arb_id": arb_id
        }

        logger.debug(f"Hedge position recorded: {hedge_direction} {symbol} @ {price:.4f}, qty: {quantity:.4f}, arb_id={arb_id}")
        return True

    async def _emergency_close_main(self, symbol: str, direction: str, price: float, quantity: float,
                                     arb_id: str = None):
        close_direction = "sell" if direction == "long" else "buy"
        try:
            self.redis_cache.publish_signal({
                "type": "signal",
                "data": {
                    "symbol": symbol,
                    "strategy_name": "arbitrage",
                    "signal_type": "arbitrage_close_hedge_failed",
                    "direction": close_direction,
                    "price": price,
                    "quantity": quantity,
                    "leverage": self._leverage,
                    "stop_loss": None,
                    "take_profit": None,
                    "confidence": 1.0,
                    "timestamp": datetime.now().isoformat(),
                    "arb_id": arb_id,
                    "reduce_only": True,
                    "close_position": True
                }
            })
            logger.warning(f"Emergency close main position: {close_direction} {symbol} @ {price:.4f}, "
                          f"qty: {quantity:.4f}, arb_id={arb_id} (hedge failed)")
        except Exception as e:
            logger.error(f"Emergency close publish failed for {symbol} (arb_id={arb_id}): {e}")

    # ============================================================
    # 相关性套利
    # ============================================================

    async def _check_correlation_arbitrage(self):
        correlation_pairs = [
            ("BTC-USDT-SWAP", "ETH-USDT-SWAP"),
            ("BTC-USDT-SWAP", "SOL-USDT-SWAP"),
            ("ETH-USDT-SWAP", "BNB-USDT-SWAP"),
            ("LTC-USDT-SWAP", "BTC-USDT-SWAP"),
            ("LINK-USDT-SWAP", "UNI-USDT-SWAP"),
            ("ADA-USDT-SWAP", "AVAX-USDT-SWAP"),
        ]

        for sym_a, sym_b in correlation_pairs:
            pair_key = f"{sym_a}:{sym_b}"
            if pair_key in self._positions:
                continue

            # 检查币种是否被冻结
            if self._risk_gate and (self._risk_gate.is_symbol_frozen(sym_a) or
                                     self._risk_gate.is_symbol_frozen(sym_b)):
                continue

            ticker_a = await self.okx_client.get_ticker_async(sym_a)
            ticker_b = await self.okx_client.get_ticker_async(sym_b)

            if not ticker_a or not ticker_b:
                continue

            price_a = self._safe_float(ticker_a.get("last"))
            price_b = self._safe_float(ticker_b.get("last"))

            if price_a <= 0 or price_b <= 0:
                continue

            ratio = price_a / price_b

            if pair_key not in self._correlation_history:
                self._correlation_history[pair_key] = []

            self._correlation_history[pair_key].append(ratio)
            if len(self._correlation_history[pair_key]) > 48:
                self._correlation_history[pair_key] = self._correlation_history[pair_key][-48:]

            history = self._correlation_history[pair_key]
            if len(history) < 24:
                continue

            avg_ratio = np.mean(history)
            std_ratio = np.std(history)

            if std_ratio == 0:
                continue

            z_score = (ratio - avg_ratio) / std_ratio

            opportunity_score = self._calculate_correlation_score(z_score, pair_key)
            if abs(z_score) > 2.0 and opportunity_score >= self._dynamic_min_opportunity_score():
                await self._generate_correlation_signal(sym_a, sym_b, z_score, price_a, price_b,
                                                         ratio, avg_ratio, opportunity_score)

    async def _generate_correlation_signal(self, sym_a: str, sym_b: str, z_score: float,
                                            price_a: float, price_b: float,
                                            current_ratio: float, avg_ratio: float,
                                            opportunity_score: float = 0.0):
        # 企业级：参数前置校验
        if not self._validate_symbol(sym_a) or not self._validate_symbol(sym_b) \
                or not self._validate_price(price_a) or not self._validate_price(price_b):
            logger.warning(
                f"Correlation signal rejected: invalid params "
                f"sym_a={sym_a!r} sym_b={sym_b!r} price_a={price_a!r} price_b={price_b!r}"
            )
            self._increment_metric("arbitrage_signal_rejected_total", 1.0, {"reason": "invalid_params", "symbol": f"{sym_a}:{sym_b}"})
            return

        pair_key = f"{sym_a}:{sym_b}"

        total_capital = self._get_effective_capital()
        trading_capital = total_capital * self.config["trading"]["trading_capital_ratio"]
        allocation = self._get_allocation()
        position_limit = self.config["currencies"]["tier1_settings"]["position_limit"] * 0.5
        base_position = trading_capital * min(allocation, position_limit)

        # 凯利公式调整
        base_position = self._calculate_kelly_position("correlation", base_position, sym_a)

        if self._risk_gate:
            base_position *= self._risk_gate.get_leverage_multiplier()

        # 净收益检查
        position_value = base_position * self._leverage
        total_cost = position_value * self._taker_fee * 4
        expected_reversion = abs(current_ratio - avg_ratio) / avg_ratio * position_value
        net_earnings = expected_reversion - total_cost

        if net_earnings <= 0:
            logger.debug(f"Correlation arbitrage skipped {pair_key}: net_earnings={net_earnings:.4f} <= 0")
            self._record_filter(pair_key, "net_earnings_insufficient")
            return

        if z_score > 2.0:
            direction_a = "short"
            direction_b = "long"
            confidence = min(0.85, 0.55 + abs(z_score) * 0.08)
        else:
            direction_a = "long"
            direction_b = "short"
            confidence = min(0.85, 0.55 + abs(z_score) * 0.08)

        qty_a = base_position / price_a
        qty_b = base_position / price_b

        signal_a = Signal(
            symbol=sym_a,
            strategy_name="arbitrage",
            signal_type="correlation_entry",
            direction=direction_a,
            price=price_a,
            quantity=qty_a,
            leverage=self._leverage,
            confidence=confidence,
            timestamp=datetime.now()
        )

        try:
            self.redis_cache.publish_signal({
                "type": "signal",
                "data": {
                    "symbol": signal_a.symbol,
                    "strategy_name": signal_a.strategy_name,
                    "signal_type": signal_a.signal_type,
                    "direction": signal_a.direction,
                    "price": signal_a.price,
                    "quantity": signal_a.quantity,
                    "leverage": signal_a.leverage,
                    "stop_loss": None,
                    "take_profit": None,
                    "confidence": signal_a.confidence,
                    "timestamp": signal_a.timestamp.isoformat(),
                    "z_score": z_score,
                    "pair": pair_key
                }
            })
        except Exception as e:
            logger.error(f"Correlation leg A publish failed for {pair_key} ({sym_a}): {e}")
            self._increment_metric("arbitrage_correlation_leg_failed_total", 1.0, {"pair": pair_key, "leg": "A"})
            return

        signal_b = Signal(
            symbol=sym_b,
            strategy_name="arbitrage",
            signal_type="correlation_entry",
            direction=direction_b,
            price=price_b,
            quantity=qty_b,
            leverage=self._leverage,
            confidence=confidence,
            timestamp=datetime.now()
        )

        try:
            self.redis_cache.publish_signal({
                "type": "signal",
                "data": {
                    "symbol": signal_b.symbol,
                    "strategy_name": signal_b.strategy_name,
                    "signal_type": signal_b.signal_type,
                    "direction": signal_b.direction,
                    "price": signal_b.price,
                    "quantity": signal_b.quantity,
                    "leverage": signal_b.leverage,
                    "stop_loss": None,
                    "take_profit": None,
                    "confidence": signal_b.confidence,
                    "timestamp": signal_b.timestamp.isoformat(),
                    "z_score": z_score,
                    "pair": pair_key
                }
            })
        except Exception as e:
            logger.error(f"Correlation leg B publish failed for {pair_key} ({sym_b}): {e}, rolling back leg A")
            self._increment_metric("arbitrage_correlation_leg_failed_total", 1.0, {"pair": pair_key, "leg": "B"})
            # 回滚已发布的A腿，避免单腿敞口
            close_dir_a = "sell" if direction_a == "long" else "buy"
            try:
                self.redis_cache.publish_signal({
                    "type": "signal",
                    "data": {
                        "symbol": sym_a,
                        "strategy_name": "arbitrage",
                        "signal_type": "arbitrage_close_correlation_rollback",
                        "direction": close_dir_a,
                        "price": price_a,
                        "quantity": qty_a,
                        "leverage": self._leverage,
                        "stop_loss": None,
                        "take_profit": None,
                        "confidence": 1.0,
                        "timestamp": datetime.now().isoformat(),
                        "reduce_only": True,
                        "close_position": True,
                        "pair": pair_key
                    }
                })
            except Exception as rollback_err:
                logger.error(f"Correlation rollback leg A failed for {pair_key}: {rollback_err}")
            # 冻结 pair，防止反复重入同一失败双腿
            if self._risk_gate:
                try:
                    self._risk_gate.freeze_symbol(sym_a, f"correlation leg B publish failed: {pair_key}")
                    self._risk_gate.freeze_symbol(sym_b, f"correlation leg B publish failed: {pair_key}")
                except Exception:
                    pass
            return

        self._positions[pair_key] = {
            "direction": f"{direction_a}/{direction_b}",
            "entry_price_a": price_a,
            "entry_price_b": price_b,
            "quantity_a": qty_a,
            "quantity_b": qty_b,
            "entry_ratio": current_ratio,
            "avg_ratio": avg_ratio,
            "entry_time": datetime.now(),
            "status": "open",
            "arbitrage_type": "correlation",
            "z_score": z_score,
            "opportunity_score": opportunity_score,
            "expected_reversion": expected_reversion,
            "net_earnings": net_earnings,
            "last_update": time.time(),
        }

        logger.info(f"Correlation arbitrage: {pair_key}, z={z_score:.2f}, {direction_a} {sym_a} + {direction_b} {sym_b}, "
                    f"ratio={current_ratio:.6f} vs avg={avg_ratio:.6f}, score={opportunity_score:.2f}")
        self._record_metric("arbitrage_signal_generated_total", 1.0, {"symbol": pair_key, "direction": f"{direction_a}/{direction_b}", "type": "correlation"})
        self._record_metric("arbitrage_signal_confidence", confidence, {"symbol": pair_key, "direction": f"{direction_a}/{direction_b}", "type": "correlation"})

    # ============================================================
    # 生产级分批执行（TWAP）
    # ============================================================

    async def _batch_execute_order(self, symbol: str, side: str, quantity: float,
                                    price: float, leverage: int, arb_id: str) -> bool:
        """通过TWAP执行器分批下单，降低滑点

        当名义价值超过 batch_min_notional 时自动拆分。
        返回 True 表示全部成交，False 表示部分成交或失败。
        """
        notional = quantity * price * leverage
        if not self._batch_execution or notional < self._batch_min_notional or not self._twap_executor:
            # 不拆分，直接下单
            return True  # 由调用方通过 publish_signal 处理

        try:
            algo_config = AlgoOrderConfig(
                order_id=f"arb_batch_{arb_id}",
                symbol=symbol,
                side=side,
                total_quantity=quantity,
                duration_seconds=self._batch_interval_sec * self._batch_count,
                num_slices=self._batch_count,
                adaptive=True,
                executor_fn=None,  # 由调用方注入
            )

            # 计算TWAP切片计划
            slices = self._twap_executor.compute_schedule(
                total_quantity=quantity,
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(seconds=algo_config.duration_seconds),
                num_slices=self._batch_count,
                jitter=True,
            )

            total_filled = 0.0
            for slc in slices:
                if slc.quantity <= 0:
                    continue

                # 发布切片信号
                self.redis_cache.publish_signal({
                    "type": "signal",
                    "data": {
                        "symbol": symbol,
                        "strategy_name": "arbitrage",
                        "signal_type": "arbitrage_batch_slice",
                        "direction": side,
                        "price": price,
                        "quantity": slc.quantity,
                        "leverage": leverage,
                        "stop_loss": None,
                        "take_profit": None,
                        "confidence": 0.85,
                        "timestamp": datetime.now().isoformat(),
                        "arb_id": arb_id,
                        "batch_slice": slc.sequence,
                        "batch_total": self._batch_count,
                    }
                })

                total_filled += slc.quantity

                # 批次间隔
                await asyncio.sleep(self._batch_interval_sec)

            fill_rate = total_filled / max(quantity, 1e-10)
            logger.info(f"Batch execution {arb_id}: {total_filled:.4f}/{quantity:.4f} "
                        f"({fill_rate:.1%}) in {self._batch_count} slices")
            return fill_rate >= 0.8  # 80%以上视为成功

        except Exception as e:
            logger.error(f"Batch execution failed for {arb_id}: {e}")
            return False

    # ============================================================
    # 净敞口硬限制
    # ============================================================

    async def _check_net_exposure_hard_limit(self) -> bool:
        """检查净敞口是否超过硬限制，超过则触发紧急平仓

        Returns:
            True 如果敞口安全，False 如果触发了紧急平仓
        """
        total_equity = self._get_effective_capital()
        if total_equity <= 0:
            # fail-closed: 无法获取账户权益时无法评估净敞口，禁止新开仓
            logger.warning("[arbitrage] 账户权益为 0，净敞口硬限制 fail-closed 拦截新开仓")
            return False

        total_net_exposure = 0.0
        exposure_details = []

        for symbol, pos in list(self._positions.items()):
            if pos.get("status") != "open":
                continue

            arb_type = pos.get("arbitrage_type", "funding")
            entry_price = pos.get("entry_price", pos.get("entry_price_a", 0))
            main_qty = float(pos.get("quantity", pos.get("quantity_a", 0)))

            if arb_type == "correlation":
                # 相关性套利：计算双边净敞口
                entry_price_b = pos.get("entry_price_b", 0)
                main_qty_b = float(pos.get("quantity_b", 0))
                main_notional = main_qty * entry_price
                hedge_notional = main_qty_b * entry_price_b
                net = abs(main_notional - hedge_notional)
                exposure_details.append((symbol, net, arb_type))
                total_net_exposure += net
            else:
                # 资金费率/基差套利：主仓与对冲仓
                main_notional = main_qty * entry_price * self._leverage
                hedge = self._hedge_positions.get(symbol, {})
                hedge_qty = float(hedge.get("quantity", 0))
                hedge_price = float(hedge.get("price", entry_price))
                hedge_notional = hedge_qty * hedge_price * self._hedge_leverage
                net = abs(main_notional - hedge_notional)
                exposure_details.append((symbol, net, arb_type))
                total_net_exposure += net

        net_exposure_pct = total_net_exposure / total_equity

        if net_exposure_pct > self._max_net_exposure_pct:
            logger.critical(
                f"NET EXPOSURE HARD LIMIT BREACHED: {net_exposure_pct:.2%} > {self._max_net_exposure_pct:.2%}, "
                f"total_net={total_net_exposure:.2f}, equity={total_equity:.2f}"
            )
            self._send_alert("emergency",
                f"Net exposure {net_exposure_pct:.2%} exceeds hard limit {self._max_net_exposure_pct:.2%}",
                {"metric": "net_exposure_breach", "value": net_exposure_pct})

            # 紧急关闭所有敞口最大的持仓
            exposure_details.sort(key=lambda x: x[1], reverse=True)
            for sym, net, arb_type in exposure_details:
                if net / max(total_equity, 1) > self._max_net_exposure_pct * 0.3:
                    logger.warning(f"Emergency closing {sym} ({arb_type}) due to net exposure breach, net={net:.2f}")
                    await self._close_position(sym, "net_exposure_breach")

            return False

        return True

    async def _manage_positions(self):
        for symbol in list(self._positions.keys()):
            state = self._positions[symbol]
            if state["status"] != "open":
                continue

            arb_type = state.get("arbitrage_type", "funding")

            if arb_type == "correlation":
                await self._check_correlation_stop_loss(symbol)
                await self._check_max_hold_time(symbol)
                continue

            await self._check_funding_close(symbol)
            await self._check_basis_close(symbol)
            await self._update_earnings(symbol)
            await self._check_extreme_movement(symbol)
            await self._check_stop_loss_take_profit(symbol)
            await self._check_max_hold_time(symbol)

    async def _check_funding_close(self, symbol: str):
        state = self._positions[symbol]

        if state["arbitrage_type"] != "funding":
            return

        funding_data = await self.okx_client.get_funding_rate_async(symbol)
        if not funding_data:
            return

        current_rate = self._safe_float(funding_data.get("fundingRate"), 0.0)

        # 费率反转应急
        entry_rate = state.get("entry_rate", 0)
        if entry_rate * current_rate < 0:
            logger.warning(f"Funding rate direction reversed for {symbol}: "
                          f"entry_rate={entry_rate:.6f}, current_rate={current_rate:.6f}, closing immediately")
            await self._close_position(symbol, "rate_reversed")
            return

        if abs(current_rate) <= self._close_threshold:
            await self._close_position(symbol, "rate_normalized")
            return

        forecast_rate = self._rate_forecast_cache.get(symbol, current_rate)
        if abs(forecast_rate) <= self._close_threshold * 0.5:
            await self._close_position(symbol, "rate_forecast_normalized")

    async def _check_basis_close(self, symbol: str):
        state = self._positions[symbol]

        if state["arbitrage_type"] != "basis":
            return

        futures_ticker = await self.okx_client.get_ticker_async(symbol)
        spot_symbol = symbol.replace("-SWAP", "") if symbol.endswith("-SWAP") else symbol
        spot_ticker = await self.okx_client.get_ticker_async(spot_symbol)

        if not futures_ticker or not spot_ticker:
            return

        futures_price = self._safe_float(futures_ticker.get("last"), 0.0)
        spot_price = self._safe_float(spot_ticker.get("last"), 0.0)

        if spot_price <= 0:
            return

        current_basis = (futures_price - spot_price) / spot_price
        entry_basis = state.get("entry_basis", 0)

        if entry_basis > 0 and current_basis < self._basis_reversal_threshold:
            await self._close_position(symbol, "basis_converged")
        elif entry_basis < 0 and current_basis > -self._basis_reversal_threshold:
            await self._close_position(symbol, "basis_converged")

    async def _update_earnings(self, symbol: str):
        state = self._positions[symbol]
        hours_passed = (datetime.now() - state["entry_time"]).total_seconds() / 3600

        if state["arbitrage_type"] == "funding":
            funding_data = await self.okx_client.get_funding_rate_async(symbol)
            if funding_data:
                current_rate = self._safe_float(funding_data.get("fundingRate"), 0.0)
                next_funding_ms = funding_data.get("nextFundingTime")
                settled_periods = 0
                if next_funding_ms:
                    try:
                        next_funding_ts = int(next_funding_ms) / 1000.0
                        total_secs = (datetime.now().timestamp() - state["entry_time"].timestamp())
                        settled_periods = max(0, int(total_secs / (8 * 3600)))
                    except (ValueError, TypeError):
                        settled_periods = hours_passed / 8
                else:
                    settled_periods = hours_passed / 8

                position_value = state["quantity"] * state["entry_price"]
                earnings = abs(current_rate) * position_value * settled_periods
                state["actual_earnings"] = earnings
                self._total_fee_earned[symbol] = earnings

        elif state["arbitrage_type"] == "basis":
            futures_ticker = await self.okx_client.get_ticker_async(symbol)
            spot_symbol = symbol.replace("-SWAP", "") if symbol.endswith("-SWAP") else symbol
            spot_ticker = await self.okx_client.get_ticker_async(spot_symbol)

            if futures_ticker and spot_ticker:
                futures_price = self._safe_float(futures_ticker.get("last"), 0.0)
                spot_price = self._safe_float(spot_ticker.get("last"), 0.0)

                if spot_price > 0:
                    current_basis = (futures_price - spot_price) / spot_price
                    basis_change = abs(state.get("entry_basis", 0)) - abs(current_basis)
                    earnings = max(0, basis_change * state["quantity"] * state["entry_price"])
                    state["actual_earnings"] = earnings

    async def _check_extreme_movement(self, symbol: str):
        state = self._positions[symbol]
        ticker = await self.okx_client.get_ticker_async(symbol)
        if not ticker:
            return

        current_price = self._safe_float(ticker.get("last"), 0.0)
        entry_price = self._safe_float(state.get("entry_price"), 0.0)
        if current_price <= 0 or entry_price <= 0:
            return
        movement = abs(current_price - entry_price) / entry_price

        if movement > 0.05:
            logger.warning(f"Extreme price movement detected for {symbol}: {movement:.2%}, force closing")
            self._send_alert("warning", f"Extreme movement {movement:.2%} for {symbol}, force closing",
                            {"metric": "extreme_movement", "value": movement})
            await self._close_position(symbol, "extreme_movement")

    async def _check_stop_loss_take_profit(self, symbol: str):
        state = self._positions[symbol]
        ticker = await self.okx_client.get_ticker_async(symbol)
        if not ticker:
            return

        current_price = self._safe_float(ticker.get("last"), 0.0)
        entry_price = self._safe_float(state.get("entry_price"), 0.0)
        direction = state["direction"]
        arb_type = state.get("arbitrage_type", "funding")
        if current_price <= 0 or entry_price <= 0:
            return

        # basis类型的止损使用basis deviation
        if arb_type == "basis":
            spot_symbol = symbol.replace("-SWAP", "") if symbol.endswith("-SWAP") else symbol
            spot_ticker = await self.okx_client.get_ticker_async(spot_symbol)
            if not spot_ticker:
                return
            spot_price = float(spot_ticker["last"])
            if spot_price <= 0:
                return

            current_basis = (current_price - spot_price) / spot_price
            entry_basis = state.get("entry_basis", 0)
            basis_std = state.get("basis_std", 0)

            basis_deviation = abs(current_basis - entry_basis)
            if basis_std > 0 and basis_deviation > 2 * basis_std:
                logger.warning(f"Basis arb stop loss: {symbol}, basis_deviation={basis_deviation:.4%} > "
                              f"2*basis_std={2*basis_std:.4%}")
                await self._close_position(symbol, "basis_stop_loss")
                return
            return

        # ATR动态止损 (优先于固定百分比)
        stop_loss_price = state.get("stop_loss")
        take_profit_price = state.get("take_profit")

        if stop_loss_price and take_profit_price:
            if direction == "long":
                if current_price <= stop_loss_price:
                    logger.warning(f"ATR stop loss: {symbol}, current={current_price:.4f}, sl={stop_loss_price:.4f}")
                    await self._close_position(symbol, "atr_stop_loss")
                    return
                if current_price >= take_profit_price:
                    logger.info(f"ATR take profit: {symbol}, current={current_price:.4f}, tp={take_profit_price:.4f}")
                    await self._close_position(symbol, "atr_take_profit")
                    return
            else:
                if current_price >= stop_loss_price:
                    logger.warning(f"ATR stop loss: {symbol}, current={current_price:.4f}, sl={stop_loss_price:.4f}")
                    await self._close_position(symbol, "atr_stop_loss")
                    return
                if current_price <= take_profit_price:
                    logger.info(f"ATR take profit: {symbol}, current={current_price:.4f}, tp={take_profit_price:.4f}")
                    await self._close_position(symbol, "atr_take_profit")
                    return

        # 固定百分比止损 (兜底)
        price_pnl = (current_price - entry_price) / entry_price if direction == "long" else (
                    entry_price - current_price) / entry_price
        total_pnl_pct = price_pnl + state.get("actual_earnings", 0) / (
                    entry_price * state["quantity"]) if state["quantity"] > 0 else price_pnl

        if total_pnl_pct <= -self._stop_loss_pct:
            logger.warning(f"Arbitrage stop loss: {symbol}, pnl={total_pnl_pct:.4%}")
            await self._close_position(symbol, "stop_loss")
            return

        if total_pnl_pct >= self._take_profit_pct:
            logger.info(f"Arbitrage take profit: {symbol}, pnl={total_pnl_pct:.4%}")
            await self._close_position(symbol, "take_profit")

    async def _check_correlation_stop_loss(self, pair_key: str):
        """相关性套利止损检查：z-score回归到0附近或进一步扩大"""
        state = self._positions[pair_key]
        sym_a, sym_b = pair_key.split(":")

        ticker_a = await self.okx_client.get_ticker_async(sym_a)
        ticker_b = await self.okx_client.get_ticker_async(sym_b)

        if not ticker_a or not ticker_b:
            return

        price_a = self._safe_float(ticker_a.get("last"), 0.0)
        price_b = self._safe_float(ticker_b.get("last"), 0.0)

        if price_a <= 0 or price_b <= 0:
            return

        current_ratio = price_a / price_b
        avg_ratio = state.get("avg_ratio", current_ratio)
        entry_z = state.get("z_score", 0)

        # 计算当前z-score
        history = self._correlation_history.get(pair_key, [])
        if len(history) >= 24:
            std_ratio = np.std(history)
            if std_ratio > 0:
                current_z = (current_ratio - avg_ratio) / std_ratio

                # 止盈：z-score回归到0附近
                if abs(current_z) < 0.5:
                    await self._close_position(pair_key, "correlation_converged")
                    return

                # 止损：z-score继续扩大（方向错误）
                if abs(current_z) > abs(entry_z) * 1.5:
                    logger.warning(f"Correlation z-score expanded: {pair_key}, entry_z={entry_z:.2f}, current_z={current_z:.2f}")
                    await self._close_position(pair_key, "correlation_expanded")
                    return

    async def _check_max_hold_time(self, symbol: str):
        state = self._positions[symbol]
        hours_passed = (datetime.now() - state["entry_time"]).total_seconds() / 3600

        if hours_passed >= self._max_hold_hours:
            logger.info(f"Arbitrage max hold time reached: {symbol}, {hours_passed:.1f}h")
            await self._close_position(symbol, "max_hold_time")

    # ============================================================
    # 平仓
    # ============================================================

    async def _close_position(self, symbol: str, reason: str):
        state = self._positions[symbol]
        arb_type = state.get("arbitrage_type", "funding")

        if arb_type == "correlation":
            await self._close_correlation_position(symbol, reason)
            return

        ticker = await self.okx_client.get_ticker_async(symbol)
        if not ticker:
            return

        price = self._safe_float(ticker.get("last"), 0.0)
        entry_price = self._safe_float(state.get("entry_price"), 0.0)
        direction = state["direction"]
        close_direction = "sell" if direction == "long" else "buy"

        if price <= 0 or entry_price <= 0:
            logger.warning(f"[arbitrage] 平仓失败: {symbol} 价格非法 price={price} entry={entry_price}")
            return

        pnl = (price - entry_price) / entry_price if direction == "long" else (
                    entry_price - price) / entry_price
        pnl_usdt = pnl * state["quantity"] * entry_price
        total_return = pnl_usdt + state["actual_earnings"]

        signal = Signal(
            symbol=symbol,
            strategy_name="arbitrage",
            signal_type=f"arbitrage_close_{reason}",
            direction=close_direction,
            price=price,
            quantity=state["quantity"],
            leverage=self._leverage,
            confidence=0.9,
            timestamp=datetime.now()
        )

        self.redis_cache.publish_signal({
            "type": "signal",
            "data": {
                "symbol": signal.symbol,
                "strategy_name": signal.strategy_name,
                "signal_type": signal.signal_type,
                "direction": signal.direction,
                "price": signal.price,
                "quantity": signal.quantity,
                "leverage": signal.leverage,
                "stop_loss": None,
                "take_profit": None,
                "confidence": signal.confidence,
                "timestamp": signal.timestamp.isoformat(),
                "reduce_only": True,
                "close_position": True
            }
        })

        await self._close_hedge_position(symbol)

        state["status"] = "closed"
        state["close_price"] = price
        state["close_time"] = datetime.now()
        state["pnl"] = pnl
        state["pnl_usdt"] = pnl_usdt
        state["total_return"] = total_return

        # 更新性能追踪
        self._update_performance(arb_type, total_return)

        # 企业级增强：出场埋点（本地统计 + MetricsPipeline）
        self._record_metric("arbitrage_position_closed_total", 1.0,
                            {"reason": reason, "symbol": symbol, "type": arb_type, "direction": direction})
        self._record_metric("arbitrage_position_pnl", total_return,
                            {"reason": reason, "symbol": symbol, "type": arb_type})

        # 记录交易历史
        self._trade_history.append({
            "symbol": symbol,
            "type": arb_type,
            "direction": direction,
            "entry_price": state["entry_price"],
            "close_price": price,
            "pnl_usdt": pnl_usdt,
            "total_return": total_return,
            "reason": reason,
            "time": datetime.now().isoformat(),
            "hold_hours": (datetime.now() - state["entry_time"]).total_seconds() / 3600,
        })

        logger.info(f"Arbitrage closed: {symbol}, type: {arb_type}, reason: {reason}, price: {price:.4f}, "
                    f"PnL: {pnl_usdt:.2f} USDT, earned: {state['actual_earnings']:.2f} USDT, total: {total_return:.2f} USDT")

        # 跟踪止损/异常关闭告警
        if reason in ("stop_loss", "atr_stop_loss", "extreme_movement", "rate_reversed"):
            self._send_alert("warning",
                f"Arbitrage {reason}: {symbol}, PnL={total_return:.2f} USDT",
                {"metric": f"arbitrage_{reason}", "value": 1})

    async def _close_hedge_position(self, symbol: str):
        if symbol in self._hedge_positions:
            hedge_info = self._hedge_positions[symbol]
            hedge_dir = hedge_info["direction"]
            # 对冲方向以 "buy"/"sell" 存储，统一归一后计算平仓反向
            close_direction = "sell" if hedge_dir in ("long", "buy") else "buy"

            self.redis_cache.publish_signal({
                "type": "signal",
                "data": {
                    "symbol": symbol,
                    "strategy_name": "arbitrage",
                    "signal_type": "close",
                    "direction": close_direction,
                    "price": hedge_info["price"],
                    "quantity": hedge_info["quantity"],
                    "leverage": self._leverage,
                    "stop_loss": None,
                    "take_profit": None,
                    "confidence": 0.9,
                    "timestamp": datetime.now().isoformat(),
                    "reduce_only": True,
                    "close_position": True
                }
            })

            del self._hedge_positions[symbol]
            logger.info(f"Hedge position closed for {symbol}: {close_direction} {hedge_info['quantity']:.4f}")

    async def _close_correlation_position(self, pair_key: str, reason: str):
        state = self._positions[pair_key]
        sym_a, sym_b = pair_key.split(":")

        ticker_a = await self.okx_client.get_ticker_async(sym_a)
        ticker_b = await self.okx_client.get_ticker_async(sym_b)

        if not ticker_a or not ticker_b:
            return

        price_a = self._safe_float(ticker_a.get("last"), 0.0)
        price_b = self._safe_float(ticker_b.get("last"), 0.0)

        direction_a, direction_b = state["direction"].split("/")
        close_dir_a = "sell" if direction_a == "long" else "buy"
        close_dir_b = "sell" if direction_b == "long" else "buy"

        # 平仓A
        self.redis_cache.publish_signal({
            "type": "signal",
            "data": {
                "symbol": sym_a,
                "strategy_name": "arbitrage",
                "signal_type": f"arbitrage_close_{reason}",
                "direction": close_dir_a,
                "price": price_a,
                "quantity": state["quantity_a"],
                "leverage": self._leverage,
                "stop_loss": None,
                "take_profit": None,
                "confidence": 0.9,
                "timestamp": datetime.now().isoformat(),
                "reduce_only": True,
                "close_position": True
            }
        })

        # 平仓B
        self.redis_cache.publish_signal({
            "type": "signal",
            "data": {
                "symbol": sym_b,
                "strategy_name": "arbitrage",
                "signal_type": f"arbitrage_close_{reason}",
                "direction": close_dir_b,
                "price": price_b,
                "quantity": state["quantity_b"],
                "leverage": self._leverage,
                "stop_loss": None,
                "take_profit": None,
                "confidence": 0.9,
                "timestamp": datetime.now().isoformat(),
                "reduce_only": True,
                "close_position": True
            }
        })

        entry_price_a = self._safe_float(state.get("entry_price_a"), 0.0)
        entry_price_b = self._safe_float(state.get("entry_price_b"), 0.0)
        if price_a <= 0 or price_b <= 0 or entry_price_a <= 0 or entry_price_b <= 0:
            logger.warning(f"[arbitrage] 相关性平仓失败: {pair_key} 价格非法")
            return

        pnl_a = (price_a - entry_price_a) / entry_price_a if direction_a == "long" else (
                    entry_price_a - price_a) / entry_price_a
        pnl_b = (price_b - entry_price_b) / entry_price_b if direction_b == "long" else (
                    entry_price_b - price_b) / entry_price_b
        pnl_usdt = pnl_a * state["quantity_a"] * entry_price_a + pnl_b * state["quantity_b"] * entry_price_b

        state["status"] = "closed"
        state["close_price_a"] = price_a
        state["close_price_b"] = price_b
        state["close_time"] = datetime.now()
        state["pnl"] = pnl_usdt

        self._update_performance("correlation", pnl_usdt)

        # 企业级增强：相关性套利出场埋点
        self._record_metric("arbitrage_position_closed_total", 1.0,
                            {"reason": reason, "symbol": pair_key, "type": "correlation",
                             "direction": state["direction"]})
        self._record_metric("arbitrage_position_pnl", pnl_usdt,
                            {"reason": reason, "symbol": pair_key, "type": "correlation"})

        self._trade_history.append({
            "symbol": pair_key,
            "type": "correlation",
            "direction": state["direction"],
            "entry_price_a": state["entry_price_a"],
            "entry_price_b": state["entry_price_b"],
            "close_price_a": price_a,
            "close_price_b": price_b,
            "pnl_usdt": pnl_usdt,
            "reason": reason,
            "time": datetime.now().isoformat(),
            "hold_hours": (datetime.now() - state["entry_time"]).total_seconds() / 3600,
        })

        logger.info(f"Correlation arbitrage closed: {pair_key}, reason: {reason}, PnL: {pnl_usdt:.2f} USDT")

    def _update_performance(self, arb_type: str, total_return: float):
        """更新性能追踪：包含夏普比率计算所需数据"""
        if arb_type not in self._arbitrage_performance:
            return

        perf = self._arbitrage_performance[arb_type]
        perf["count"] += 1
        perf["total_pnl"] += total_return

        if total_return > 0:
            perf["wins"] += 1
            perf["consecutive_wins"] = perf.get("consecutive_wins", 0) + 1
            perf["consecutive_losses"] = 0
        else:
            perf["losses"] += 1
            perf["consecutive_losses"] = perf.get("consecutive_losses", 0) + 1
            perf["consecutive_wins"] = 0

        # 追踪PnL序列用于计算夏普比率和最大回撤
        if "pnl_list" not in perf:
            perf["pnl_list"] = []
        perf["pnl_list"].append(total_return)
        if len(perf["pnl_list"]) > 200:
            perf["pnl_list"] = perf["pnl_list"][-200:]

        # 更新峰值权益和最大回撤
        if "peak_equity" not in perf:
            perf["peak_equity"] = 0
        cum_pnl = sum(perf["pnl_list"])
        if cum_pnl > perf["peak_equity"]:
            perf["peak_equity"] = cum_pnl
        drawdown = perf["peak_equity"] - cum_pnl
        if "max_drawdown" not in perf:
            perf["max_drawdown"] = 0
        if drawdown > perf["max_drawdown"]:
            perf["max_drawdown"] = drawdown

    # ============================================================
    # 统计与状态
    # ============================================================

    def get_stats(self) -> Dict[str, Any]:
        open_positions = sum(1 for p in self._positions.values() if p["status"] == "open")
        closed_positions = sum(1 for p in self._positions.values() if p["status"] == "closed")

        total_fee_earned = sum(self._total_fee_earned.values())
        total_return = sum(p.get("total_return", 0) for p in self._positions.values() if p["status"] == "closed")

        funding_positions = sum(1 for p in self._positions.values() if p.get("arbitrage_type") == "funding" and p[
            "status"] == "open")
        basis_positions = sum(1 for p in self._positions.values() if p.get("arbitrage_type") == "basis" and p[
            "status"] == "open")
        correlation_positions = sum(1 for p in self._positions.values() if p.get("arbitrage_type") == "correlation" and
                                    p["status"] == "open")

        # 计算净敞口
        total_equity = self._get_effective_capital()
        total_net_exposure = 0.0
        for symbol, pos in self._positions.items():
            if pos.get("status") != "open":
                continue
            arb_type = pos.get("arbitrage_type", "funding")
            if arb_type == "correlation":
                main_notional = float(pos.get("quantity_a", 0)) * pos.get("entry_price_a", 0)
                hedge_notional = float(pos.get("quantity_b", 0)) * pos.get("entry_price_b", 0)
            else:
                main_notional = float(pos.get("quantity", 0)) * pos.get("entry_price", 0) * self._leverage
                hedge = self._hedge_positions.get(symbol, {})
                hedge_notional = float(hedge.get("quantity", 0)) * float(hedge.get("price", 0)) * self._hedge_leverage
            total_net_exposure += abs(main_notional - hedge_notional)
        net_exposure_pct = total_net_exposure / total_equity if total_equity > 0 else 0

        # 计算夏普比率
        performance_with_sharpe = {}
        for arb_type, perf in self._arbitrage_performance.items():
            perf_copy = dict(perf)
            pnl_list = perf_copy.get("pnl_list", [])
            if len(pnl_list) >= 10:
                avg_pnl = np.mean(pnl_list)
                std_pnl = np.std(pnl_list) if len(pnl_list) > 1 else 0
                if std_pnl > 0:
                    # 年化夏普（假设平均每天1笔交易，252个交易日）
                    perf_copy["sharpe_ratio"] = round((avg_pnl / std_pnl) * np.sqrt(252), 3)
                else:
                    perf_copy["sharpe_ratio"] = 0
            else:
                perf_copy["sharpe_ratio"] = None
            performance_with_sharpe[arb_type] = perf_copy

        return {
            "strategy": "arbitrage",
            "version": "2.0",
            "is_enabled": self._enabled,
            "emergency_stop": self._emergency_stop,
            "open_positions": open_positions,
            "closed_positions": closed_positions,
            "funding_positions": funding_positions,
            "basis_positions": basis_positions,
            "correlation_positions": correlation_positions,
            "total_fee_earned": total_fee_earned,
            "total_return": total_return,
            "net_exposure": round(total_net_exposure, 2),
            "net_exposure_pct": round(net_exposure_pct, 4),
            "net_exposure_limit": self._max_net_exposure_pct,
            "arbitrage_performance": performance_with_sharpe,
            "enabled_types": self._arbitrage_types_enabled,
            "kelly_enabled": self._kelly_enabled,
            "atr_stop_enabled": self._atr_stop_enabled,
            "regime_adaptive": self._regime_adaptive,
            "batch_execution": self._batch_execution,
            "twap_available": _TWAP_AVAILABLE and self._twap_executor is not None,
            "market_regimes": self._market_regimes,
            "position_details": [{
                "symbol": s,
                "direction": p["direction"],
                "entry_price": p.get("entry_price", p.get("entry_price_a", 0)),
                "entry_rate": p.get("entry_rate", 0),
                "arbitrage_type": p.get("arbitrage_type", "funding"),
                "expected_earnings": p.get("expected_earnings", 0),
                "actual_earnings": p.get("actual_earnings", 0),
                "opportunity_score": p.get("opportunity_score", 0),
                "status": p["status"]
            } for s, p in self._positions.items() if p["status"] == "open"],
            "recent_trades": list(self._trade_history)[-10:],
        }

    def get_health(self) -> Dict[str, Any]:
        """企业级增强：暴露自适应调参状态与过滤埋点统计，供 Dashboard / 健康检查消费。"""
        return {
            "strategy": "arbitrage",
            "enabled": self._enabled,
            "emergency_stop": self._emergency_stop,
            "adaptive_params": self._adaptive_params,
            "min_opportunity_score": self._min_opportunity_score,
            "dynamic_min_opportunity_score": self._dynamic_min_opportunity_score(),
            "risk_lock_quality_boost": self._risk_lock_quality_boost,
            "adaptive_controller_injected": self._adaptive_controller is not None,
            "filter_stats": dict(self._filter_stats),
            "open_positions": sum(1 for p in self._positions.values() if p.get("status") == "open"),
        }

    # ============================================================
    # 状态持久化
    # ============================================================

    def collect_persistent_state(self) -> Dict[str, Any]:
        return {
            "positions": self._positions,
            "hedge_positions": self._hedge_positions,
            "basis_history": self._basis_history,
            "funding_history": self._funding_history,
            "correlation_history": self._correlation_history,
            "oi_history": self._oi_history,
            "volume_history": self._volume_history,
            "total_fee_earned": self._total_fee_earned,
            "arbitrage_performance": self._arbitrage_performance,
            "rate_forecast_cache": self._rate_forecast_cache,
            "market_regimes": self._market_regimes,
            "atr_cache": self._atr_cache,
            "trade_history": list(self._trade_history),
            # v2.0 新增持久化字段
            "price_history": {k: list(v) for k, v in self._price_history.items()},
            "correlation_matrix": self._correlation_matrix,
            "correlation_cache_time": self._correlation_cache_time,
            "last_hedge_failure": self._last_hedge_failure,
            "emergency_stop": self._emergency_stop,
        }

    def restore_persistent_state(self, state: Dict[str, Any]):
        try:
            if "positions" in state and isinstance(state["positions"], dict):
                restored = {}
                for sym, pos in state["positions"].items():
                    if isinstance(pos, dict) and pos.get("status") == "open":
                        for key in ("entry_time", "close_time"):
                            if key in pos and isinstance(pos[key], str):
                                try:
                                    pos[key] = datetime.fromisoformat(pos[key])
                                except (ValueError, TypeError):
                                    pass
                        restored[sym] = pos
                self._positions = restored
                logger.info(f"Arbitrage restored {len(restored)} open positions")

            if "hedge_positions" in state and isinstance(state["hedge_positions"], dict):
                restored_hedges = {}
                for sym, h in state["hedge_positions"].items():
                    if isinstance(h, dict):
                        for key in ("timestamp",):
                            if key in h and isinstance(h[key], str):
                                try:
                                    h[key] = datetime.fromisoformat(h[key])
                                except (ValueError, TypeError):
                                    pass
                        restored_hedges[sym] = h
                self._hedge_positions = restored_hedges
                logger.info(f"Arbitrage restored {len(restored_hedges)} hedge positions")

            for key in ("basis_history", "funding_history", "correlation_history", "oi_history",
                        "volume_history", "total_fee_earned", "arbitrage_performance",
                        "rate_forecast_cache", "market_regimes", "atr_cache"):
                if key in state and isinstance(state[key], dict):
                    setattr(self, f"_{key}", state[key])

            if "trade_history" in state and isinstance(state["trade_history"], list):
                self._trade_history = deque(state["trade_history"][-200:], maxlen=200)

            # v2.0 新增字段恢复
            if "price_history" in state and isinstance(state["price_history"], dict):
                for sym, prices in state["price_history"].items():
                    if isinstance(prices, list):
                        self._price_history[sym] = deque(prices, maxlen=100)

            if "correlation_matrix" in state and isinstance(state["correlation_matrix"], dict):
                self._correlation_matrix = state["correlation_matrix"]

            if "correlation_cache_time" in state:
                self._correlation_cache_time = float(state["correlation_cache_time"])

            if "last_hedge_failure" in state and isinstance(state["last_hedge_failure"], dict):
                self._last_hedge_failure = {k: float(v) for k, v in state["last_hedge_failure"].items()}

            if "emergency_stop" in state:
                self._emergency_stop = bool(state["emergency_stop"])
        except Exception as e:
            logger.error(f"Arbitrage restore_persistent_state failed: {e}")