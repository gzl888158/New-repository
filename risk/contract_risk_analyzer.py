"""
激进合约专属风险分析框架
=======================
三大维度全覆盖：
1. 市场行情风险：极端插针、夜间低流动性、大额资金砸盘、资金费率跳变
2. 程序技术风险：本地算力不足、网络波动、交易所API维护、内存溢出崩溃
3. 策略逻辑风险：震荡行情趋势策略持续亏损、参数过度拟合、高杠杆单边重仓

统一风险分析引擎（ContractRiskAnalyzer）：
- 周期性扫描三类风险
- 分级响应：INFO → WARNING → CRITICAL → EMERGENCY
- 联动已有风控：RiskGate L5熔断 / AdaptiveController / GlobalRiskControl
"""

import time
import psutil
import asyncio
import numpy as np
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from loguru import logger


# ============================================================
# 基础类型定义
# ============================================================

class RiskLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class RiskCategory(Enum):
    MARKET = "market"
    TECHNICAL = "technical"
    STRATEGY = "strategy"

class RiskAction(Enum):
    NONE = "none"
    LOG_ONLY = "log_only"
    REDUCE_SIZE = "reduce_size"          # 缩仓30%
    PAUSE_NEW_ORDERS = "pause_new"       # 暂停开新仓
    PAUSE_STRATEGY = "pause_strategy"    # 暂停策略
    EMERGENCY_CLOSE = "emergency_close"  # 紧急全平
    ADJUST_LEVERAGE = "adjust_leverage"  # 降低杠杆

@dataclass
class RiskAlert:
    """风险告警"""
    timestamp: datetime
    category: RiskCategory
    name: str
    level: RiskLevel
    action: RiskAction
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    symbol: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "name": self.name,
            "level": self.level.value,
            "action": self.action.value,
            "message": self.message,
            "details": self.details,
            "symbol": self.symbol,
        }


# ============================================================
# 维度1：市场行情风险检测器
# ============================================================

class MarketRiskDetector:
    """市场行情风险检测：插针/低流动性/砸盘/资金费率"""

    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("contract_risk", {}).get("market", {})
        # 插针检测
        self._spike_threshold_pct = cfg.get("spike_threshold_pct", 0.03)  # 3%瞬时变化
        self._spike_window_ms = cfg.get("spike_window_ms", 5000)           # 5秒窗口
        # 低流动性检测
        self._low_liq_hours = cfg.get("low_liquidity_hours", [0, 1, 2, 3, 4, 5, 6, 7])  # UTC
        self._min_volume_24h = cfg.get("min_volume_24h_usd", 200_000)    # P9: 24h最低成交量降至20万（100万对小币种过高）
        # 砸盘检测
        self._whale_drop_pct = cfg.get("whale_drop_pct", 0.02)            # 2%快速下跌
        self._whale_volume_mult = cfg.get("whale_volume_mult", 5.0)        # 成交量为均值5倍
        # 资金费率跳变
        self._funding_rate_threshold = cfg.get("funding_rate_threshold", 0.001)  # 0.1%
        self._funding_rate_spike_mult = cfg.get("funding_rate_spike_mult", 3.0)

        # 状态
        self._price_history: Dict[str, deque] = {}        # symbol → deque of (ts, price)
        self._funding_history: Dict[str, deque] = {}      # symbol → deque of rate
        self._volume_avg: Dict[str, float] = {}            # symbol → 平均成交量
        self._last_alerts: Dict[str, float] = {}           # 告警冷却

    def check_spike(self, symbol: str, current_price: float, prev_price: float = 0) -> Optional[RiskAlert]:
        """极端插针检测：短时间内价格剧烈变化"""
        now = time.time()
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=200)

        self._price_history[symbol].append((now, current_price))

        if prev_price <= 0:
            history = self._price_history[symbol]
            if len(history) < 2:
                return None
            prev_price = history[-2][1]

        if prev_price <= 0:
            return None

        change_pct = abs(current_price - prev_price) / prev_price

        if change_pct >= self._spike_threshold_pct:
            # 检查是否在短窗口内
            window_start = now - self._spike_window_ms / 1000
            window_prices = [p for ts, p in self._price_history[symbol] if ts >= window_start]
            if len(window_prices) >= 2:
                window_change = abs(window_prices[-1] - window_prices[0]) / window_prices[0]
                if window_change >= self._spike_threshold_pct:
                    alert_key = f"spike:{symbol}"
                    if self._check_cooldown(alert_key, now, 60):
                        return None
                    self._last_alerts[alert_key] = now
                    direction = "插针上拉" if current_price > prev_price else "插针下砸"
                    level = RiskLevel.EMERGENCY if window_change >= self._spike_threshold_pct * 3 else RiskLevel.CRITICAL
                    return RiskAlert(
                        timestamp=datetime.now(),
                        category=RiskCategory.MARKET,
                        name="price_spike",
                        level=level,
                        action=RiskAction.PAUSE_NEW_ORDERS if level == RiskLevel.CRITICAL else RiskAction.EMERGENCY_CLOSE,
                        message=f"{direction}: {symbol} 窗口变化{window_change:.2%} (阈值{self._spike_threshold_pct:.2%})",
                        details={"change_pct": window_change, "threshold": self._spike_threshold_pct},
                        symbol=symbol,
                    )
        return None

    def check_low_liquidity(self, symbol: str, volume_24h: float = 0) -> Optional[RiskAlert]:
        """夜间低流动性检测：特定时段成交量骤降"""
        now_utc = datetime.utcnow().hour
        if now_utc not in self._low_liq_hours:
            return None

        if volume_24h > 0 and volume_24h < self._min_volume_24h:
            alert_key = f"low_liq:{symbol}"
            now = time.time()
            if self._check_cooldown(alert_key, now, 300):
                return None
            self._last_alerts[alert_key] = now
            return RiskAlert(
                timestamp=datetime.now(),
                category=RiskCategory.MARKET,
                name="low_liquidity",
                level=RiskLevel.WARNING,
                action=RiskAction.REDUCE_SIZE,
                message=f"低流动性警告: {symbol} 24h成交量=${volume_24h:,.0f} < ${self._min_volume_24h:,.0f}",
                details={"volume_24h": volume_24h, "min_required": self._min_volume_24h, "utc_hour": now_utc},
                symbol=symbol,
            )
        return None

    def check_whale_activity(self, symbol: str, current_price: float, volume_24h: float = 0,
                              prev_price: float = 0) -> Optional[RiskAlert]:
        """大额资金砸盘检测：快速下跌+异常成交量

        Args:
            symbol: 交易对
            current_price: 当前价格
            volume_24h: 24小时成交量（原始数量，非USD）
            prev_price: 上一帧价格（可选，自动从历史获取）
        """
        now = time.time()

        # 从价格历史获取上一帧价格
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=200)
        self._price_history[symbol].append((now, current_price))

        if prev_price <= 0:
            history = self._price_history[symbol]
            if len(history) < 2:
                return None
            prev_price = history[-2][1]

        if prev_price <= 0:
            return None

        # 更新平均成交量（简单EMA）
        if volume_24h > 0:
            if symbol not in self._volume_avg:
                self._volume_avg[symbol] = volume_24h
            else:
                self._volume_avg[symbol] = 0.9 * self._volume_avg[symbol] + 0.1 * volume_24h
        avg_volume = self._volume_avg.get(symbol, 0)

        if avg_volume <= 0:
            return None

        price_drop = (prev_price - current_price) / prev_price
        volume_ratio = volume_24h / avg_volume if avg_volume > 0 else 1.0

        if price_drop >= self._whale_drop_pct and volume_ratio >= self._whale_volume_mult:
            alert_key = f"whale:{symbol}"
            if self._check_cooldown(alert_key, now, 120):
                return None
            self._last_alerts[alert_key] = now
            level = RiskLevel.EMERGENCY if price_drop >= self._whale_drop_pct * 2 else RiskLevel.CRITICAL
            return RiskAlert(
                timestamp=datetime.now(),
                category=RiskCategory.MARKET,
                name="whale_dump",
                level=level,
                action=RiskAction.PAUSE_NEW_ORDERS,
                message=f"疑似砸盘: {symbol} 下跌{price_drop:.2%}, 成交量{volume_ratio:.1f}x均值",
                details={"drop_pct": price_drop, "volume_ratio": volume_ratio},
                symbol=symbol,
            )
        return None

    def check_funding_rate(self, symbol: str, funding_rate: float,
                           position_direction: str = "") -> Optional[RiskAlert]:
        """资金费率跳变检测：高费率或异常跳变
        
        P0: 新增position_direction参数，区分仓位方向
        """
        if symbol not in self._funding_history:
            self._funding_history[symbol] = deque(maxlen=20)

        history = self._funding_history[symbol]
        history.append(funding_rate)

        # 高费率绝对值 —— P0: 仅当方向不利时才触发减仓
        if abs(funding_rate) >= self._funding_rate_threshold:
            is_paying = (position_direction == "long" and funding_rate > 0) or \
                        (position_direction == "short" and funding_rate < 0)
            if is_paying:
                alert_key = f"funding_abs:{symbol}"
                now = time.time()
                if self._check_cooldown(alert_key, now, 600):
                    return None
                self._last_alerts[alert_key] = now
                return RiskAlert(
                    timestamp=datetime.now(),
                    category=RiskCategory.MARKET,
                    name="funding_rate_high",
                    level=RiskLevel.WARNING,
                    action=RiskAction.REDUCE_SIZE,
                    message=f"资金费率不利: {symbol} {position_direction}方向支付{funding_rate:.6f} (阈值{self._funding_rate_threshold:.4f})",
                    details={"funding_rate": funding_rate, "threshold": self._funding_rate_threshold, "direction": position_direction},
                    symbol=symbol,
                )

        # 费率跳变（相对历史均值）—— P0: 方向反转单独告警
        if len(history) >= 5:
            avg_rate = np.mean(list(history)[:-1])
            if avg_rate != 0:
                spike_ratio = abs(funding_rate / avg_rate) if avg_rate != 0 else 1.0
                direction_reversed = (avg_rate > 0 and funding_rate < 0) or (avg_rate < 0 and funding_rate > 0)
                if spike_ratio >= self._funding_rate_spike_mult or direction_reversed:
                    alert_key = f"funding_spike:{symbol}"
                    now = time.time()
                    if self._check_cooldown(alert_key, now, 600):
                        return None
                    self._last_alerts[alert_key] = now
                    reversal_note = " [方向反转!]" if direction_reversed else ""
                    return RiskAlert(
                        timestamp=datetime.now(),
                        category=RiskCategory.MARKET,
                        name="funding_rate_reversal" if direction_reversed else "funding_rate_spike",
                        level=RiskLevel.WARNING,
                        action=RiskAction.LOG_ONLY,
                        message=f"资金费率跳变: {symbol} current={funding_rate:.6f}, avg={avg_rate:.6f}, {spike_ratio:.1f}x{reversal_note}",
                        details={"current": funding_rate, "avg": avg_rate, "ratio": spike_ratio, "reversed": direction_reversed},
                        symbol=symbol,
                    )
        return None

    def _check_cooldown(self, key: str, now: float, cooldown_s: float) -> bool:
        return key in self._last_alerts and (now - self._last_alerts[key]) < cooldown_s


# ============================================================
# 维度2：程序技术风险检测器
# ============================================================

class TechnicalRiskDetector:
    """程序技术风险检测：算力/网络/API/内存"""

    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("risk", {}).get("contract_risk", {}).get("technical", {})
        self._cpu_high_pct = cfg.get("cpu_high_pct", 85.0)         # CPU>85%告警
        self._cpu_critical_pct = cfg.get("cpu_critical_pct", 95.0)  # CPU>95%紧急
        self._mem_high_pct = cfg.get("mem_high_pct", 80.0)         # 内存>80%告警
        self._mem_critical_pct = cfg.get("mem_critical_pct", 92.0)  # 内存>92%紧急
        self._latency_warning_ms = cfg.get("latency_warning_ms", 2000)   # 2s延迟
        self._latency_critical_ms = cfg.get("latency_critical_ms", 5000) # 5s延迟
        self._error_rate_threshold = cfg.get("error_rate_threshold", 0.1) # API错误率>10%
        self._api_maintenance_codes = cfg.get("api_maintenance_codes", ["50013", "50014"])

        # 状态
        self._api_errors = deque(maxlen=100)
        self._api_total = deque(maxlen=100)
        self._latency_history = deque(maxlen=100)
        self._last_alerts: Dict[str, float] = {}

    def check_compute(self) -> Optional[RiskAlert]:
        """本地算力不足检测"""
        try:
            cpu_pct = psutil.cpu_percent(interval=0.5)
            mem = psutil.virtual_memory()
            mem_pct = mem.percent
            mem_avail_gb = mem.available / (1024 ** 3)

            # 内存即将溢出
            if mem_pct >= self._mem_critical_pct or mem_avail_gb < 0.5:
                return RiskAlert(
                    timestamp=datetime.now(),
                    category=RiskCategory.TECHNICAL,
                    name="memory_critical",
                    level=RiskLevel.EMERGENCY,
                    action=RiskAction.PAUSE_STRATEGY,
                    message=f"内存即将溢出: {mem_pct:.1f}% (可用{mem_avail_gb:.1f}GB)",
                    details={"mem_pct": mem_pct, "mem_avail_gb": mem_avail_gb},
                )

            if mem_pct >= self._mem_high_pct:
                return RiskAlert(
                    timestamp=datetime.now(),
                    category=RiskCategory.TECHNICAL,
                    name="memory_high",
                    level=RiskLevel.WARNING,
                    action=RiskAction.LOG_ONLY,
                    message=f"内存使用率偏高: {mem_pct:.1f}%",
                    details={"mem_pct": mem_pct},
                )

            # CPU过载
            if cpu_pct >= self._cpu_critical_pct:
                return RiskAlert(
                    timestamp=datetime.now(),
                    category=RiskCategory.TECHNICAL,
                    name="cpu_critical",
                    level=RiskLevel.CRITICAL,
                    action=RiskAction.PAUSE_NEW_ORDERS,
                    message=f"CPU严重过载: {cpu_pct:.1f}%",
                    details={"cpu_pct": cpu_pct},
                )

            if cpu_pct >= self._cpu_high_pct:
                return RiskAlert(
                    timestamp=datetime.now(),
                    category=RiskCategory.TECHNICAL,
                    name="cpu_high",
                    level=RiskLevel.WARNING,
                    action=RiskAction.LOG_ONLY,
                    message=f"CPU使用率偏高: {cpu_pct:.1f}%",
                    details={"cpu_pct": cpu_pct},
                )

        except Exception as e:
            logger.debug(f"Compute check error: {e}")
        return None

    def check_network(self, latency_ms: float = 0) -> Optional[RiskAlert]:
        """网络波动检测"""
        if latency_ms > 0:
            self._latency_history.append(latency_ms)

        if not self._latency_history:
            return None

        recent = list(self._latency_history)[-20:]
        avg_latency = np.mean(recent)
        p99_latency = np.percentile(recent, 99) if len(recent) >= 5 else max(recent)

        if p99_latency >= self._latency_critical_ms:
            return RiskAlert(
                timestamp=datetime.now(),
                category=RiskCategory.TECHNICAL,
                name="network_critical",
                level=RiskLevel.CRITICAL,
                action=RiskAction.PAUSE_NEW_ORDERS,
                message=f"网络延迟严重: P99={p99_latency:.0f}ms, avg={avg_latency:.0f}ms",
                details={"p99_ms": p99_latency, "avg_ms": avg_latency},
            )

        if avg_latency >= self._latency_warning_ms:
            return RiskAlert(
                timestamp=datetime.now(),
                category=RiskCategory.TECHNICAL,
                name="network_slow",
                level=RiskLevel.WARNING,
                action=RiskAction.LOG_ONLY,
                message=f"网络延迟偏高: avg={avg_latency:.0f}ms",
                details={"avg_ms": avg_latency},
            )
        return None

    def record_api_result(self, is_error: bool, error_code: str = ""):
        """记录API调用结果"""
        self._api_total.append(1)
        if is_error:
            self._api_errors.append(1)
        # 检查是否为维护代码
        if error_code in self._api_maintenance_codes:
            self._last_alerts["api_maintenance"] = time.time()

    def check_api_health(self) -> Optional[RiskAlert]:
        """交易所API维护/错误率检测"""
        # API维护中
        now = time.time()
        if "api_maintenance" in self._last_alerts and (now - self._last_alerts["api_maintenance"]) < 600:
            return RiskAlert(
                timestamp=datetime.now(),
                category=RiskCategory.TECHNICAL,
                name="api_maintenance",
                level=RiskLevel.CRITICAL,
                action=RiskAction.PAUSE_NEW_ORDERS,
                message="交易所API维护中，暂停开新仓",
                details={"status": "maintenance"},
            )

        # API错误率
        if len(self._api_total) >= 10:
            error_rate = sum(self._api_errors) / sum(self._api_total)
            if error_rate >= self._error_rate_threshold:
                return RiskAlert(
                    timestamp=datetime.now(),
                    category=RiskCategory.TECHNICAL,
                    name="api_error_rate",
                    level=RiskLevel.WARNING,
                    action=RiskAction.LOG_ONLY,
                    message=f"API错误率偏高: {error_rate:.1%} (阈值{self._error_rate_threshold:.0%})",
                    details={"error_rate": error_rate},
                )
        return None


# ============================================================
# 维度3：策略逻辑风险检测器
# ============================================================

class StrategyRiskDetector:
    """策略逻辑风险检测：震荡亏损/过拟合/高杠杆重仓"""

    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("contract_risk", {}).get("strategy", {})
        # 震荡行情趋势策略亏损
        self._trend_max_consecutive_loss = cfg.get("trend_max_consecutive_loss", 5)
        self._trend_max_drawdown_pct = cfg.get("trend_max_drawdown_pct", 0.05)  # 单策略5%回撤
        # 趋势策略类型列表（精确匹配，避免 "counter_trend" 等被误判为趋势策略）
        self._trend_strategy_types = cfg.get(
            "trend_strategy_types",
            ["trend_breakout", "trend_follow", "trend_ma"]
        )
        # 过拟合检测
        self._overfit_winrate_gap = cfg.get("overfit_winrate_gap", 0.15)   # 回测-实盘胜率差>15%
        self._overfit_min_trades = cfg.get("overfit_min_trades", 20)        # 最少20笔交易才评估
        # 高杠杆重仓
        self._high_leverage_threshold = cfg.get("high_leverage_threshold", 10)
        self._concentration_limit_pct = cfg.get("concentration_limit_pct", 0.4)  # 单币种不超过40%

        # 状态
        self._strategy_stats: Dict[str, Dict[str, Any]] = {}
        self._last_alerts: Dict[str, float] = {}

    def update_strategy_stats(self, strategy_name: str, symbol: str,
                               consecutive_losses: int = 0, drawdown_pct: float = 0,
                               live_win_rate: float = 0.5, backtest_win_rate: float = 0.5,
                               live_trades: int = 0, leverage: int = 1,
                               position_ratio: float = 0.0):
        """更新策略统计"""
        key = f"{strategy_name}:{symbol}"
        self._strategy_stats[key] = {
            "strategy_name": strategy_name,
            "symbol": symbol,
            "consecutive_losses": consecutive_losses,
            "drawdown_pct": drawdown_pct,
            "live_win_rate": live_win_rate,
            "backtest_win_rate": backtest_win_rate,
            "live_trades": live_trades,
            "leverage": leverage,
            "position_ratio": position_ratio,
            "updated_at": time.time(),
        }

    def check_trend_in_ranging(self, strategy_name: str, symbol: str,
                                 consecutive_losses: int, drawdown_pct: float) -> Optional[RiskAlert]:
        """震荡行情趋势策略持续亏损"""
        if not self._is_trend_strategy(strategy_name):
            return None

        if consecutive_losses >= self._trend_max_consecutive_loss:
            alert_key = f"trend_loss:{strategy_name}:{symbol}"
            now = time.time()
            if self._check_cooldown(alert_key, now, 300):
                return None
            self._last_alerts[alert_key] = now
            level = RiskLevel.CRITICAL if consecutive_losses >= self._trend_max_consecutive_loss * 2 else RiskLevel.WARNING
            return RiskAlert(
                timestamp=datetime.now(),
                category=RiskCategory.STRATEGY,
                name="trend_consecutive_loss",
                level=level,
                action=RiskAction.PAUSE_STRATEGY if level == RiskLevel.CRITICAL else RiskAction.REDUCE_SIZE,
                message=f"趋势策略持续亏损: {strategy_name} {symbol} 连亏{consecutive_losses}笔",
                details={"consecutive_losses": consecutive_losses, "drawdown_pct": drawdown_pct},
                symbol=symbol,
            )

        if drawdown_pct >= self._trend_max_drawdown_pct:
            alert_key = f"trend_dd:{strategy_name}:{symbol}"
            now = time.time()
            if self._check_cooldown(alert_key, now, 300):
                return None
            self._last_alerts[alert_key] = now
            return RiskAlert(
                timestamp=datetime.now(),
                category=RiskCategory.STRATEGY,
                name="trend_drawdown",
                level=RiskLevel.CRITICAL,
                action=RiskAction.PAUSE_STRATEGY,
                message=f"趋势策略回撤过大: {strategy_name} {symbol} 回撤{drawdown_pct:.2%}",
                details={"drawdown_pct": drawdown_pct, "threshold": self._trend_max_drawdown_pct},
                symbol=symbol,
            )
        return None

    def check_overfitting(self, strategy_name: str, symbol: str,
                           live_win_rate: float, backtest_win_rate: float,
                           live_trades: int) -> Optional[RiskAlert]:
        """参数过度拟合检测：回测胜率远高于实盘"""
        if live_trades < self._overfit_min_trades:
            return None

        gap = backtest_win_rate - live_win_rate
        if gap >= self._overfit_winrate_gap:
            alert_key = f"overfit:{strategy_name}:{symbol}"
            now = time.time()
            if self._check_cooldown(alert_key, now, 3600):
                return None
            self._last_alerts[alert_key] = now
            return RiskAlert(
                timestamp=datetime.now(),
                category=RiskCategory.STRATEGY,
                name="overfitting",
                level=RiskLevel.WARNING,
                action=RiskAction.ADJUST_LEVERAGE,
                message=f"疑似过拟合: {strategy_name} {symbol} 回测胜率{backtest_win_rate:.0%} vs 实盘{live_win_rate:.0%} (差{gap:.0%})",
                details={"backtest_wr": backtest_win_rate, "live_wr": live_win_rate, "gap": gap},
                symbol=symbol,
            )
        return None

    def check_high_leverage_concentration(self, symbol: str, leverage: int,
                                            position_ratio: float,
                                            strategy_name: str = "",
                                            position_count: int = 1) -> Optional[RiskAlert]:
        """高杠杆单边重仓检测

        Args:
            position_count: 当前持仓币种总数。集中度检测仅在多币种持仓时有意义；
                            单币种持仓时 position_ratio 恒为 1.0，应跳过集中度判定。
        """
        # 高杠杆+重仓双条件（仅多币种持仓时检测集中度，避免单持仓误报）
        if position_count > 1 and leverage >= self._high_leverage_threshold and position_ratio >= self._concentration_limit_pct:
            alert_key = f"hl_conc:{symbol}"
            now = time.time()
            if self._check_cooldown(alert_key, now, 300):
                return None
            self._last_alerts[alert_key] = now
            return RiskAlert(
                timestamp=datetime.now(),
                category=RiskCategory.STRATEGY,
                name="high_leverage_concentration",
                level=RiskLevel.CRITICAL,
                action=RiskAction.ADJUST_LEVERAGE,
                message=f"高杠杆重仓: {symbol} {leverage}x杠杆, 占仓位{position_ratio:.0%}",
                details={"leverage": leverage, "position_ratio": position_ratio, "position_count": position_count},
                symbol=symbol,
            )

        # 仅高杠杆（不依赖集中度，单/多持仓均检测）
        if leverage >= self._high_leverage_threshold * 1.5:
            alert_key = f"hl_only:{symbol}"
            now = time.time()
            if self._check_cooldown(alert_key, now, 600):
                return None
            self._last_alerts[alert_key] = now
            return RiskAlert(
                timestamp=datetime.now(),
                category=RiskCategory.STRATEGY,
                name="extreme_leverage",
                level=RiskLevel.WARNING,
                action=RiskAction.ADJUST_LEVERAGE,
                message=f"极高杠杆: {symbol} {leverage}x",
                details={"leverage": leverage},
                symbol=symbol,
            )
        return None

    def _check_cooldown(self, key: str, now: float, cooldown_s: float) -> bool:
        return key in self._last_alerts and (now - self._last_alerts[key]) < cooldown_s

    def _is_trend_strategy(self, strategy_name: str) -> bool:
        """判断是否为趋势策略（精确匹配配置的趋势策略类型列表）

        相比子串匹配 "trend" in name，可避免将 "counter_trend" 等反向策略误判为趋势策略。
        匹配规则：strategy_name 小写后等于列表中某项，或以 "列表项+" 开头（如 trend_breakout_v2）。
        """
        if not strategy_name:
            return False
        name_lower = strategy_name.lower()
        for trend_type in self._trend_strategy_types:
            t = trend_type.lower()
            if name_lower == t or name_lower.startswith(t + "_") or name_lower.startswith(t + "-"):
                return True
        return False


# ============================================================
# 统一风险分析引擎
# ============================================================

class ContractRiskAnalyzer:
    """
    激进合约风险分析引擎

    职责：
    1. 周期性执行三类风险扫描
    2. 汇聚告警，按级别排序
    3. 自动执行分级响应动作
    4. 联动已有风控（RiskGate/GlobalRisk/AdaptiveController）
    """

    def __init__(self, config: Dict[str, Any], okx_client=None, risk_gate=None):
        self._config = config
        self._okx_client = okx_client
        self._risk_gate = risk_gate

        self._market = MarketRiskDetector(config)
        self._technical = TechnicalRiskDetector(config)
        self._strategy = StrategyRiskDetector(config)

        # 告警历史
        self._alert_history: deque = deque(maxlen=500)
        self._active_alerts: Dict[str, RiskAlert] = {}

        # 响应回调（由scheduler注入）
        self._on_pause_new: Any = None
        self._on_pause_strategy: Any = None
        self._on_emergency_close: Any = None
        self._on_reduce_size: Any = None
        self._on_adjust_leverage: Any = None

        # 统计
        self._stats = {
            "total_scans": 0,
            "alerts_by_level": {"info": 0, "warning": 0, "critical": 0, "emergency": 0},
            "alerts_by_category": {"market": 0, "technical": 0, "strategy": 0},
            "actions_taken": {},
        }

    def set_callbacks(self, on_pause_new=None, on_pause_strategy=None,
                      on_emergency_close=None, on_reduce_size=None,
                      on_adjust_leverage=None):
        """注入响应回调"""
        self._on_pause_new = on_pause_new
        self._on_pause_strategy = on_pause_strategy
        self._on_emergency_close = on_emergency_close
        self._on_reduce_size = on_reduce_size
        self._on_adjust_leverage = on_adjust_leverage

    async def scan_market_risks(self, positions: List[Dict] = None) -> List[RiskAlert]:
        """扫描市场行情风险"""
        alerts = []

        # 遍历持仓进行市场风险检查
        if positions and self._okx_client:
            for pos in positions:
                symbol = pos.get("instId", "")
                try:
                    # 获取当前行情
                    ticker = self._okx_client.get_ticker(symbol)
                    if ticker:
                        last = float(ticker.get("last", 0))

                        # 插针检测（传 prev=0，由 check_spike 内部从 _price_history 取上一帧）
                        alert = self._market.check_spike(symbol, last, 0)
                        if alert:
                            alerts.append(alert)

                        # 低流动性检测
                        vol_24h = float(ticker.get("vol24h", 0)) * last  # 粗估USD
                        alert = self._market.check_low_liquidity(symbol, vol_24h)
                        if alert:
                            alerts.append(alert)

                        # 大额砸盘检测
                        vol_24h_raw = float(ticker.get("vol24h", 0))
                        alert = self._market.check_whale_activity(symbol, last, vol_24h_raw)
                        if alert:
                            alerts.append(alert)

                    # 资金费率检测
                    funding = self._okx_client.get_funding_rate(symbol)
                    if funding:
                        rate = float(funding.get("fundingRate", 0))
                        alert = self._market.check_funding_rate(symbol, rate)
                        if alert:
                            alerts.append(alert)

                except Exception as e:
                    logger.debug(f"Market risk scan error for {symbol}: {e}")

        self._record_alerts(alerts)
        return alerts

    async def scan_technical_risks(self) -> List[RiskAlert]:
        """扫描程序技术风险"""
        alerts = []

        # 算力检测
        alert = self._technical.check_compute()
        if alert:
            alerts.append(alert)

        # 网络检测（从RiskGate L2获取延迟数据）
        if self._risk_gate and hasattr(self._risk_gate, '_latency_history'):
            try:
                latencies = list(self._risk_gate._latency_history)
                if latencies:
                    avg_latency = np.mean(latencies)
                    alert = self._technical.check_network(avg_latency)
                    if alert:
                        alerts.append(alert)
            except Exception:
                pass

        # API健康检测
        alert = self._technical.check_api_health()
        if alert:
            alerts.append(alert)

        self._record_alerts(alerts)
        return alerts

    async def scan_strategy_risks(self, positions: List[Dict] = None,
                                    strategy_stats: Dict = None) -> List[RiskAlert]:
        """扫描策略逻辑风险"""
        alerts = []

        # 基于持仓的检查
        if positions:
            total_position_value = sum(abs(float(p.get("notionalUsd", 0))) for p in positions)
            position_count = len(positions)

            for pos in positions:
                symbol = pos.get("instId", "")
                leverage = int(float(pos.get("lever", 1)))
                notional = abs(float(pos.get("notionalUsd", 0)))
                position_ratio = notional / total_position_value if total_position_value > 0 else 0

                # 高杠杆重仓检测（传入持仓数，单持仓时跳过集中度判定）
                alert = self._strategy.check_high_leverage_concentration(
                    symbol, leverage, position_ratio,
                    position_count=position_count
                )
                if alert:
                    alerts.append(alert)

        # 基于策略统计的检查
        if strategy_stats:
            for key, stats in strategy_stats.items():
                strategy_name = stats.get("strategy_name", "")
                symbol = stats.get("symbol", "")

                # 震荡亏损检测
                alert = self._strategy.check_trend_in_ranging(
                    strategy_name, symbol,
                    stats.get("consecutive_losses", 0),
                    stats.get("drawdown_pct", 0),
                )
                if alert:
                    alerts.append(alert)

                # 过拟合检测
                alert = self._strategy.check_overfitting(
                    strategy_name, symbol,
                    stats.get("live_win_rate", 0.5),
                    stats.get("backtest_win_rate", 0.5),
                    stats.get("live_trades", 0),
                )
                if alert:
                    alerts.append(alert)

        self._record_alerts(alerts)
        return alerts

    async def full_scan(self, positions: List[Dict] = None,
                        strategy_stats: Dict = None) -> List[RiskAlert]:
        """执行全维度风险扫描"""
        all_alerts = []
        all_alerts.extend(await self.scan_market_risks(positions))
        all_alerts.extend(await self.scan_technical_risks())
        all_alerts.extend(await self.scan_strategy_risks(positions, strategy_stats))

        # 按级别排序：EMERGENCY > CRITICAL > WARNING > INFO
        level_order = {RiskLevel.EMERGENCY: 0, RiskLevel.CRITICAL: 1, RiskLevel.WARNING: 2, RiskLevel.INFO: 3}
        all_alerts.sort(key=lambda a: level_order.get(a.level, 99))

        # 自动执行响应
        for alert in all_alerts:
            await self._execute_action(alert)

        self._stats["total_scans"] += 1
        return all_alerts

    async def _execute_action(self, alert: RiskAlert):
        """执行风险响应动作"""
        action = alert.action
        if action == RiskAction.NONE or action == RiskAction.LOG_ONLY:
            if alert.level in (RiskLevel.CRITICAL, RiskLevel.EMERGENCY):
                logger.warning(f"[ContractRisk] {alert.level.value.upper()}: {alert.message}")
            else:
                logger.info(f"[ContractRisk] {alert.level.value}: {alert.message}")
            return

        action_name = action.value
        self._stats["actions_taken"][action_name] = self._stats["actions_taken"].get(action_name, 0) + 1

        try:
            if action == RiskAction.REDUCE_SIZE and self._on_reduce_size:
                await self._on_reduce_size(alert)
                logger.warning(f"[ContractRisk] 执行缩仓: {alert.message}")
            elif action == RiskAction.PAUSE_NEW_ORDERS and self._on_pause_new:
                await self._on_pause_new(alert)
                logger.warning(f"[ContractRisk] 暂停开仓: {alert.message}")
            elif action == RiskAction.PAUSE_STRATEGY and self._on_pause_strategy:
                await self._on_pause_strategy(alert)
                logger.warning(f"[ContractRisk] 暂停策略: {alert.message}")
            elif action == RiskAction.EMERGENCY_CLOSE and self._on_emergency_close:
                await self._on_emergency_close(alert)
                logger.critical(f"[ContractRisk] 紧急全平: {alert.message}")
            elif action == RiskAction.ADJUST_LEVERAGE and self._on_adjust_leverage:
                await self._on_adjust_leverage(alert)
                logger.warning(f"[ContractRisk] 调整杠杆: {alert.message}")
        except Exception as e:
            logger.error(f"Risk action execution error: {e}")

        # 联动RiskGate L5熔断
        if alert.level == RiskLevel.EMERGENCY and self._risk_gate:
            try:
                if hasattr(self._risk_gate, 'manual_trigger'):
                    self._risk_gate.manual_trigger(f"ContractRisk: {alert.message}")
            except Exception:
                pass

    def _record_alerts(self, alerts: List[RiskAlert]):
        for alert in alerts:
            self._alert_history.append(alert)
            self._active_alerts[f"{alert.category.value}:{alert.name}:{alert.symbol}"] = alert
            self._stats["alerts_by_level"][alert.level.value] += 1
            self._stats["alerts_by_category"][alert.category.value] += 1

    # ---- 公开接口 ----

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取当前活跃告警"""
        return [a.to_dict() for a in self._active_alerts.values()]

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取最近告警历史"""
        return [a.to_dict() for a in list(self._alert_history)[-limit:]]

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    def get_health_summary(self) -> Dict[str, Any]:
        """获取风险健康概要"""
        active = list(self._active_alerts.values())
        worst_level = RiskLevel.INFO
        for a in active:
            if a.level == RiskLevel.EMERGENCY:
                worst_level = RiskLevel.EMERGENCY
                break
            elif a.level == RiskLevel.CRITICAL and worst_level != RiskLevel.EMERGENCY:
                worst_level = RiskLevel.CRITICAL
            elif a.level == RiskLevel.WARNING and worst_level not in (RiskLevel.EMERGENCY, RiskLevel.CRITICAL):
                worst_level = RiskLevel.WARNING

        return {
            "status": worst_level.value,
            "active_alerts_count": len(active),
            "market_alerts": sum(1 for a in active if a.category == RiskCategory.MARKET),
            "technical_alerts": sum(1 for a in active if a.category == RiskCategory.TECHNICAL),
            "strategy_alerts": sum(1 for a in active if a.category == RiskCategory.STRATEGY),
            "total_scans": self._stats["total_scans"],
        }
