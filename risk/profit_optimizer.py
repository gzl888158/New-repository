"""
盈利优化器：复利机制、凯利仓位、手续费控制、资金费率计算
"""
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from loguru import logger
import numpy as np
import json
import sqlite3


class ProfitOptimizer:
    """综合盈利优化器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        trading_cfg = config.get("trading", {})

        # 复利配置
        self._compound_enabled = trading_cfg.get("compound_enabled", True)
        self._compound_reinvest_ratio = trading_cfg.get("compound_reinvest_ratio", 0.6)
        self._initial_capital = trading_cfg.get("total_capital", 100)

        # 仓位配置
        self._risk_per_trade = trading_cfg.get("risk_per_trade", 0.015)
        self._target_return = trading_cfg.get("target_return", 1.15)
        self._min_margin = trading_cfg.get("min_margin_per_trade", 0.5)

        # 手续费配置
        self._taker_fee = trading_cfg.get("taker_fee_rate", 0.0005)
        self._maker_fee = trading_cfg.get("maker_fee_rate", 0.0002)
        self._max_slippage = trading_cfg.get("max_slippage_pct", 0.002)

        # 资金费率
        self._funding_check = trading_cfg.get("funding_rate_check", True)
        self._funding_min_hold = trading_cfg.get("funding_rate_min_hold", 0.0003)

        # 历史交易统计（用于凯利公式）
        self._trade_history: List[Dict[str, float]] = []
        self._win_count = 0
        self._loss_count = 0
        self._total_pnl = 0.0
        self._peak_equity = self._initial_capital
        self._current_equity = self._initial_capital  # 启动后由scheduler立即更新为实际OKX权益

        # 从数据库加载历史峰值权益（跨重启持久化）
        # 注意：仅使用最近24小时内的峰值，避免不同账户状态的峰值误报
        try:
            import sqlite3
            db_path = config["sqlite"]["db_path"]
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
            c.execute("SELECT MAX(total_equity) FROM account_history WHERE timestamp > ?", (cutoff,))
            row = c.fetchone()
            if row and row[0] and row[0] > self._peak_equity:
                self._peak_equity = row[0]
            conn.close()
        except Exception:
            pass

        # 如果历史峰值远高于初始资金（可能来自不同账户状态），重置为初始资金
        # 回撤计算应基于当前会话的峰值，而非历史不可比数据
        if self._peak_equity > self._initial_capital * 1.5:
            logger.warning(
                f"Historical peak {self._peak_equity:.2f} far exceeds initial capital {self._initial_capital:.2f}, "
                f"resetting peak to initial capital (different account state detected)"
            )
            self._peak_equity = self._initial_capital

        # 复利增长因子
        self._compound_factor = 1.0

        # 回撤保护因子
        self._drawdown_factor = 1.0

    def update_equity(self, equity: float):
        """更新当前权益，计算复利因子"""
        self._current_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity

        if self._compound_enabled and self._initial_capital > 0:
            # 复利因子 = (当前权益/初始资金) ^ 再投资比例
            growth = equity / self._initial_capital
            self._compound_factor = max(0.5, min(3.0, growth ** self._compound_reinvest_ratio))

    def record_trade_result(self, pnl: float, strategy: str = ""):
        """记录交易结果，更新胜率统计"""
        self._total_pnl += pnl

        self._trade_history.append({
            "pnl": pnl,
            "strategy": strategy,
            "timestamp": datetime.now().isoformat()
        })

        # 保留最近100笔，同步更新累计计数以保持一致
        if len(self._trade_history) > 100:
            self._trade_history = self._trade_history[-100:]

        # 基于trade_history重新计算，保证统计一致
        self._win_count = sum(1 for t in self._trade_history if t["pnl"] > 0)
        self._loss_count = sum(1 for t in self._trade_history if t["pnl"] < 0)

    def get_win_rate(self) -> float:
        """获取胜率"""
        total = self._win_count + self._loss_count
        if total == 0:
            return 0.5
        return self._win_count / total

    def get_kelly_fraction(self) -> float:
        """贝叶斯凯利公式：用先验分布解决样本不足问题
        先验：假设胜率~Beta(α,β)，α=20, β=20（先验胜率50%，等效40次交易经验）
        后验：α' = α + 实际盈利数, β' = β + 实际亏损数
        最终胜率 = α' / (α' + β')，自然趋向先验，随样本增加趋向实际

        b = 赔率（平均盈利/平均亏损）
        p = 贝叶斯后验胜率
        q = 1 - p
        """
        # 贝叶斯先验参数：等效于"已观察到20胜20负"，避免小样本过拟合
        # 先验胜率=0.5，先验样本量=40，强度可调
        prior_alpha = 20.0  # 先验盈利数
        prior_beta = 20.0   # 先验亏损数

        actual_wins = self._win_count
        actual_losses = self._loss_count

        # 后验参数
        post_alpha = prior_alpha + actual_wins
        post_beta = prior_beta + actual_losses

        # 贝叶斯后验胜率（自然趋向先验0.5，随交易增加趋向实际胜率）
        p = post_alpha / (post_alpha + post_beta)

        # 计算赔率b：用先验+实际加权平均，避免小样本极端值
        wins = [t["pnl"] for t in self._trade_history if t["pnl"] > 0]
        losses = [abs(t["pnl"]) for t in self._trade_history if t["pnl"] < 0]

        # 先验赔率：1:1（盈亏平衡）
        prior_b = 1.0
        if wins and losses:
            actual_b = float(np.mean(wins)) / float(np.mean(losses)) if np.mean(losses) > 0 else 1.0
            # 加权融合：实际样本越多，越信任实际赔率
            total_actual = actual_wins + actual_losses
            weight = min(1.0, total_actual / 50.0)  # 50笔交易后完全用实际值
            b = prior_b * (1 - weight) + actual_b * weight
        else:
            b = prior_b

        q = 1 - p
        kelly = (b * p - q) / b if b > 0 else 0

        # 半凯利（更保守），限制在[0.2, 1.0]
        half_kelly = max(0.2, min(1.0, kelly * 0.5))
        return half_kelly

    def _calc_drawdown_factor(self, drawdown: float) -> float:
        """回撤保护因子：分段非线性曲线
        0-5%回撤：因子=1.0（不影响交易）
        5-10%回撤：因子从1.0线性降到0.85（轻微减仓）
        10-15%回撤：因子从0.85加速降到0.65（明显减仓）
        15-20%回撤：因子从0.65加速降到0.45（大幅减仓）
        20-25%回撤：因子从0.45降到0.30（接近暂停）
        >25%回撤：因子=0.30（下限保护）
        """
        if drawdown <= 0.05:
            return 1.0
        elif drawdown <= 0.10:
            # 5%-10%: 1.0 → 0.85
            return 1.0 - (drawdown - 0.05) / 0.05 * 0.15
        elif drawdown <= 0.15:
            # 10%-15%: 0.85 → 0.65
            return 0.85 - (drawdown - 0.10) / 0.05 * 0.20
        elif drawdown <= 0.20:
            # 15%-20%: 0.65 → 0.45
            return 0.65 - (drawdown - 0.15) / 0.05 * 0.20
        elif drawdown <= 0.25:
            # 20%-25%: 0.45 → 0.30
            return 0.45 - (drawdown - 0.20) / 0.05 * 0.15
        else:
            return 0.30

    def get_optimal_position_size(self, base_margin: float, leverage: int) -> float:
        """计算最优仓位大小（融合复利+凯利+回撤保护+最小仓位过滤）"""
        # 复利因子
        compound = self._compound_factor if self._compound_enabled else 1.0

        # 凯利因子
        kelly = self.get_kelly_fraction()

        # 回撤保护：使用分段非线性曲线
        drawdown = 1 - (self._current_equity / self._peak_equity) if self._peak_equity > 0 else 0
        self._drawdown_factor = self._calc_drawdown_factor(drawdown)

        # 综合仓位因子
        size_factor = compound * kelly * self._drawdown_factor
        
        # 小资金适配：设置合理的最小仓位因子，小资金更保守
        if self._current_equity > 0:
            if self._current_equity >= 500:
                min_factor = 0.5
            elif self._current_equity >= 200:
                min_factor = 0.4
            elif self._current_equity >= 100:
                min_factor = 0.3
            else:
                min_factor = 0.2
            if size_factor < min_factor:
                size_factor = min_factor

        size_factor = max(0.2, min(2.5, size_factor))

        adjusted_margin = base_margin * size_factor

        # 动态最小保证金：小资金降低门槛
        effective_min_margin = self._get_effective_min_margin()
        if adjusted_margin < effective_min_margin:
            return 0.0

        return adjusted_margin

    def _get_effective_min_margin(self) -> float:
        """获取有效的最小保证金：根据实际账户资金动态调整"""
        if self._current_equity >= 1000:
            return self._min_margin
        elif self._current_equity >= 500:
            return max(0.5, self._min_margin * 0.7)
        elif self._current_equity >= 200:
            return max(0.2, self._min_margin * 0.4)
        elif self._current_equity >= 100:
            return max(0.1, self._min_margin * 0.2)
        else:
            return max(0.05, self._min_margin * 0.1)

    def calculate_total_cost(self, position_value: float, leverage: int,
                             hold_hours: float = 0, funding_rate: float = 0,
                             direction: str = "long") -> Dict[str, float]:
        """计算交易总成本：手续费 + 滑点 + 资金费率
        Args:
            direction: 'long'或'short'，用于资金费率方向计算
        """
        # 开仓手续费（taker）
        open_fee = position_value * self._taker_fee
        # 平仓手续费（taker）
        close_fee = position_value * self._taker_fee
        # 滑点成本
        slippage_cost = position_value * self._max_slippage
        # 资金费率成本（每8小时结算一次）
        # P0: 根据持仓方向计算实际支付/收取 — 空头在正费率时收取，多头在负费率时收取
        funding_periods = hold_hours / 8.0
        is_paying = (direction == "long" and funding_rate > 0) or (direction == "short" and funding_rate < 0)
        if is_paying:
            funding_cost = position_value * abs(funding_rate) * funding_periods
        else:
            # 方向有利时资金费率是收入而非成本
            funding_cost = 0.0

        total_cost = open_fee + close_fee + slippage_cost + funding_cost

        return {
            "open_fee": open_fee,
            "close_fee": close_fee,
            "slippage_cost": slippage_cost,
            "funding_cost": funding_cost,
            "total_cost": total_cost,
            "cost_pct": total_cost / position_value if position_value > 0 else 0,
            "funding_direction": "paying" if is_paying else "receiving"
        }

    def is_trade_profitable(self, position_value: float, expected_profit: float,
                            leverage: int, hold_hours: float = 0,
                            funding_rate: float = 0) -> Tuple[bool, str]:
        """判断交易是否盈利（扣除所有成本后）"""
        costs = self.calculate_total_cost(position_value, leverage, hold_hours, funding_rate)

        net_profit = expected_profit - costs["total_cost"]

        if net_profit <= 0:
            return False, f"不盈利: 预期收益{expected_profit:.4f} < 总成本{costs['total_cost']:.4f} (费用{costs['cost_pct']:.4%})"

        # 收益成本比至少1.5:1
        if expected_profit / costs["total_cost"] < 1.5:
            return False, f"收益成本比过低: {expected_profit / costs['total_cost']:.2f}"

        return True, f"盈利: 净收益{net_profit:.4f}, 成本{costs['cost_pct']:.4%}"

    def should_close_for_funding(self, position_value: float, funding_rate: float,
                                  unrealized_pnl: float, hold_hours: float) -> Tuple[bool, str]:
        """判断是否因资金费率而平仓"""
        if not self._funding_check:
            return False, ""

        # 资金费率成本
        funding_periods = hold_hours / 8.0
        funding_cost = position_value * abs(funding_rate) * funding_periods

        # 如果资金费率成本超过未实现盈利的50%，平仓
        if funding_cost > abs(unrealized_pnl) * 0.5 and unrealized_pnl > 0:
            return True, f"资金费率成本{funding_cost:.4f}超过盈利的50%, 平仓保护"

        return False, ""

    def get_funding_rate_direction_bias(self, funding_rate: float) -> str:
        """根据资金费率获取方向偏好
        正资金费率：做空方收钱 -> 偏多
        负资金费率：做多方收钱 -> 偏空
        但反向持仓收益更高（收取资金费率）
        """
        if funding_rate > self._funding_min_hold:
            return "short"  # 做空收取资金费率
        elif funding_rate < -self._funding_min_hold:
            return "long"  # 做多收取资金费率
        return "neutral"

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self._win_count + self._loss_count
        avg_win = self._avg_win() if self._win_count > 0 else 0
        avg_loss = abs(self._avg_loss()) if self._loss_count > 0 else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 1.0
        return {
            "total_trades": total,
            "win_count": self._win_count,
            "loss_count": self._loss_count,
            "win_rate": self.get_win_rate(),
            "profit_factor": profit_factor,
            "total_pnl": self._total_pnl,
            "compound_factor": self._compound_factor,
            "kelly_fraction": self.get_kelly_fraction(),
            "kelly_factor": self.get_kelly_fraction(),
            "drawdown_factor": self._drawdown_factor,
            "current_equity": self._current_equity,
            "peak_equity": self._peak_equity,
            "drawdown": 1 - (self._current_equity / self._peak_equity) if self._peak_equity > 0 else 0
        }

    def _avg_win(self) -> float:
        wins = [t["pnl"] for t in self._trade_history if t["pnl"] > 0]
        return sum(wins) / len(wins) if wins else 0

    def _avg_loss(self) -> float:
        losses = [t["pnl"] for t in self._trade_history if t["pnl"] < 0]
        return sum(losses) / len(losses) if losses else 0


class EnhancedStopLoss:
    """强化止损管理器：动态止损 + 保本止损 + 移动止损 + 分级止盈 + 时间止盈 + 波动率保护"""

    def __init__(self, config: Dict[str, Any], strategy_name: str = ""):
        self.config = config
        self.strategy_name = strategy_name

        strategy_cfg = config.get("strategies", {}).get(strategy_name, {})
        trading_cfg = config.get("trading", {})

        self._stop_loss_pct = strategy_cfg.get("stop_loss_pct", 0.025)
        self._breakeven_trigger = strategy_cfg.get("breakeven_trigger_pct", 0.015)
        self._breakeven_stop = strategy_cfg.get("breakeven_stop_pct", 0.003)
        self._trailing_enabled = strategy_cfg.get("trailing_stop_enabled", True)
        self._trailing_pct = strategy_cfg.get("trailing_stop_pct", 0.015)

        # ── 分级止盈配置 ──
        self._tp_enabled = strategy_cfg.get("take_profit_enabled", True)
        self._take_profit_pct = strategy_cfg.get("take_profit_pct", 0.03)
        # TP1: 40%仓位在60%目标处，触发后止损移到保本
        # TP2: 50%仓位在100%目标处，触发后止损移到TP1
        # TP3: 10%仓位用移动止损追踪远端
        self._tp1_ratio = strategy_cfg.get("tp1_ratio", 0.4)
        self._tp1_pct = strategy_cfg.get("tp1_pct", 0.6)   # 60% of target
        self._tp2_ratio = strategy_cfg.get("tp2_ratio", 0.5)
        self._tp2_pct = strategy_cfg.get("tp2_pct", 1.0)   # 100% of target
        self._tp3_ratio = strategy_cfg.get("tp3_ratio", 0.1)
        self._tp_trailing_pct = strategy_cfg.get("tp3_trailing_pct", 0.02)

        # ── 时间止盈配置 ──
        self._time_exit_enabled = strategy_cfg.get("time_exit_enabled", True)
        self._max_hold_hours = strategy_cfg.get("max_hold_hours", 72)
        self._time_exit_partial_pct = strategy_cfg.get("time_exit_partial_pct", 0.5)
        self._time_exit_after_hours = strategy_cfg.get("time_exit_after_hours", 48)

        # ── 波动率紧急止损配置 ──
        self._volatility_stop_enabled = strategy_cfg.get("volatility_stop_enabled", True)
        self._vol_spike_threshold = strategy_cfg.get("vol_spike_threshold", 2.0)  # ATR突增倍数
        self._vol_stop_partial_pct = strategy_cfg.get("vol_stop_partial_pct", 0.3)
        self._volatility_lockout_minutes = strategy_cfg.get("volatility_lockout_minutes", 30)

        # ── 盈亏比动态调整 ──
        self._adaptive_rr_enabled = strategy_cfg.get("adaptive_rr_enabled", True)
        self._min_rr_ratio = strategy_cfg.get("min_rr_ratio", 1.5)

        # 每个持仓的止损状态
        self._position_stops: Dict[str, Dict[str, Any]] = {}

        # ── 企业级：止损状态跨重启持久化（watchdog 拉起后恢复保本/移动/分级止盈进度）──
        self._db_path = config.get("sqlite", {}).get("db_path", "")
        self._persistence_enabled = bool(self._db_path)
        if self._persistence_enabled:
            self._init_state_table()
            self._load_state()

    def init_position_stop(self, symbol: str, entry_price: float, direction: str,
                           quantity: float, entry_time: datetime = None):
        """初始化持仓止损（含分级止盈、时间止盈、波动率跟踪）
        
        Args:
            quantity: 持仓数量（必填），为0时止盈系统静默失效
        """
        # P0: quantity为0时记录严重警告，避免静默失效
        if quantity <= 0:
            logger.error(f"init_position_stop called with quantity=0 for {symbol}, stop-loss/take-profit system will be disabled!")
        if entry_time is None:
            entry_time = datetime.now()

        # 企业级：若该 symbol 已存在持久化状态（重启后由 _load_state 恢复），
        # 刷新 entry_price 并保留止损进度（保本/移动/分级止盈/时间/累计减仓），避免回退。
        if symbol in self._position_stops:
            existing = self._position_stops[symbol]
            if existing.get("direction") != direction:
                # 方向变化（罕见，防御）：视为新仓位，清除旧状态重新初始化
                logger.warning(f"Direction changed for {symbol} ({existing.get('direction')} -> {direction}), reinitializing stop state")
                self._position_stops.pop(symbol, None)
                self._delete(symbol)
            else:
                if entry_price > 0:
                    existing["entry_price"] = entry_price
                # 重算止盈目标/分级价格（基于最新 entry_price），保留 tp1/tp2 filled 进度
                self._recompute_tp_prices(existing)
                self._persist(symbol)
                return existing.get("current_stop", 0.0)

        if direction == "long":
            initial_sl = entry_price * (1 - self._stop_loss_pct)
            tp_target = entry_price * (1 + self._take_profit_pct)
            tp1_price = entry_price + (tp_target - entry_price) * self._tp1_pct
            tp2_price = entry_price + (tp_target - entry_price) * self._tp2_pct
        else:
            initial_sl = entry_price * (1 + self._stop_loss_pct)
            tp_target = entry_price * (1 - self._take_profit_pct)
            tp1_price = entry_price - (entry_price - tp_target) * self._tp1_pct
            tp2_price = entry_price - (entry_price - tp_target) * self._tp2_pct

        self._position_stops[symbol] = {
            "entry_price": entry_price,
            "direction": direction,
            "quantity": quantity,
            "entry_time": entry_time,
            "initial_stop": initial_sl,
            "current_stop": initial_sl,
            "peak_price": entry_price,
            "trough_price": entry_price,
            "breakeven_activated": False,
            "trailing_activated": False,
            # 分级止盈
            "tp_target": tp_target,
            "tp1_price": tp1_price,
            "tp2_price": tp2_price,
            "tp1_filled": False,
            "tp2_filled": False,
            "tp3_trailing_active": False,
            "tp1_quantity": quantity * self._tp1_ratio,
            "tp2_quantity": quantity * self._tp2_ratio,
            "tp3_quantity": quantity * self._tp3_ratio,
            # 波动率跟踪
            "atr_history": [],
            "last_atr": 0.0,
            "avg_atr": 0.0,
            "vol_spike_detected": False,
            "vol_stop_triggered": False,
            "vol_lockout_until": None,
            # 时间止盈
            "time_partial_done": False,
            "time_exit_done": False,
            # 累计已减仓量
            "total_closed_qty": 0.0,
        }
        self._persist(symbol)
        return initial_sl

    def update_stop(self, symbol: str, current_price: float, atr: float = 0.0) -> Tuple[float, str]:
        """更新止损价格，返回(新止损价, 止损类型)
        
        Args:
            symbol: 交易品种
            current_price: 当前价格
            atr: ATR 值，用于动态调整 trailing stop 百分比（0=不调整）
        """
        if symbol not in self._position_stops:
            return 0.0, "no_position"

        state = self._position_stops[symbol]
        entry = state["entry_price"]
        direction = state["direction"]
        current_stop = state["current_stop"]

        if direction == "long":
            profit_pct = (current_price - entry) / entry
            if current_price > state["peak_price"]:
                state["peak_price"] = current_price
        else:
            profit_pct = (entry - current_price) / entry
            if current_price < state["trough_price"]:
                state["trough_price"] = current_price

        # 更新 ATR 历史（用于波动率突增检测）
        if atr > 0:
            state["last_atr"] = atr
            state["atr_history"].append({"time": datetime.now(), "atr": atr})
            # 保留最近20个ATR样本（约100秒，5秒间隔）
            if len(state["atr_history"]) > 20:
                state["atr_history"] = state["atr_history"][-20:]
            # 计算平均ATR（排除最新值做对比）
            if len(state["atr_history"]) >= 5:
                historical_atrs = [h["atr"] for h in state["atr_history"][:-1]]
                state["avg_atr"] = sum(historical_atrs) / len(historical_atrs)

        new_stop = current_stop
        stop_type = "initial"

        # 1. TP1 触发后止损上移到保本（分级止盈联动）
        if state["tp1_filled"] and not state["breakeven_activated"]:
            if direction == "long":
                breakeven_price = entry * (1 + self._breakeven_stop)
            else:
                breakeven_price = entry * (1 - self._breakeven_stop)
            if direction == "long" and breakeven_price > new_stop:
                new_stop = breakeven_price
                state["breakeven_activated"] = True
                stop_type = "tp1_breakeven"
            elif direction == "short" and breakeven_price < new_stop:
                new_stop = breakeven_price
                state["breakeven_activated"] = True
                stop_type = "tp1_breakeven"

        # 2. TP2 触发后止损上移到 TP1 价格（锁定大部分利润）
        if state["tp2_filled"]:
            tp1_price = state["tp1_price"]
            if direction == "long" and tp1_price > new_stop:
                new_stop = tp1_price
                stop_type = "tp2_sl_uplift"
            elif direction == "short" and tp1_price < new_stop:
                new_stop = tp1_price
                stop_type = "tp2_sl_uplift"

        # 3. 保本止损：盈利达到触发阈值后，止损移到入场价附近
        if profit_pct >= self._breakeven_trigger and not state["breakeven_activated"]:
            if direction == "long":
                breakeven_price = entry * (1 + self._breakeven_stop)
            else:
                breakeven_price = entry * (1 - self._breakeven_stop)

            if direction == "long" and breakeven_price > new_stop:
                new_stop = breakeven_price
                state["breakeven_activated"] = True
                stop_type = "breakeven"
            elif direction == "short" and breakeven_price < new_stop:
                new_stop = breakeven_price
                state["breakeven_activated"] = True
                stop_type = "breakeven"

        # 4. 移动止损：盈利后跟踪价格移动止损（ATR 动态调整）
        if self._trailing_enabled and profit_pct > 0:
            if atr > 0 and current_price > 0:
                atr_ratio = atr / current_price
                if atr_ratio > 0.02:
                    trail_mult = 1.5
                elif atr_ratio > 0.01:
                    trail_mult = 1.2
                elif atr_ratio < 0.005:
                    trail_mult = 0.7
                else:
                    trail_mult = 1.0
            else:
                trail_mult = 1.0
            
            adjusted_trail = self._trailing_pct * trail_mult
            
            if direction == "long":
                trailing_stop = current_price * (1 - adjusted_trail)
                if trailing_stop > new_stop:
                    new_stop = trailing_stop
                    state["trailing_activated"] = True
                    stop_type = "trailing"
            else:
                trailing_stop = current_price * (1 + adjusted_trail)
                if trailing_stop < new_stop:
                    new_stop = trailing_stop
                    state["trailing_activated"] = True
                    stop_type = "trailing"

        # 5. TP3 远端移动止损（更紧的追踪）
        if state["tp2_filled"] and self._tp_trailing_pct > 0:
            if direction == "long":
                tp3_trailing = current_price * (1 - self._tp_trailing_pct)
                if tp3_trailing > new_stop:
                    new_stop = tp3_trailing
                    state["tp3_trailing_active"] = True
                    stop_type = "tp3_trailing"
            else:
                tp3_trailing = current_price * (1 + self._tp_trailing_pct)
                if tp3_trailing < new_stop:
                    new_stop = tp3_trailing
                    state["tp3_trailing_active"] = True
                    stop_type = "tp3_trailing"

        state["current_stop"] = new_stop
        self._persist(symbol)
        return new_stop, stop_type

    def check_stop_trigger(self, symbol: str, current_price: float) -> bool:
        """检查是否触发止损"""
        if symbol not in self._position_stops:
            return False

        state = self._position_stops[symbol]
        direction = state["direction"]
        stop = state["current_stop"]

        if direction == "long" and current_price <= stop:
            return True
        elif direction == "short" and current_price >= stop:
            return True
        return False

    def check_take_profit(self, symbol: str, current_price: float) -> List[Dict[str, Any]]:
        """
        检查分级止盈触发，返回需要执行的止盈操作列表。
        
        返回: [{action: "tp1"|"tp2"|"tp3", price, quantity, side}]
        """
        if symbol not in self._position_stops or not self._tp_enabled:
            return []

        state = self._position_stops[symbol]
        direction = state["direction"]
        actions = []

        # TP1: 近端止盈（40%仓位）
        if not state["tp1_filled"] and state["tp1_quantity"] > 0:
            tp1 = state["tp1_price"]
            if direction == "long" and current_price >= tp1:
                actions.append({
                    "action": "tp1",
                    "price": tp1,
                    "quantity": state["tp1_quantity"],
                    "side": "sell",
                    "reason": f"TP1 reached: {tp1:.4f}"
                })
                state["tp1_filled"] = True
            elif direction == "short" and current_price <= tp1:
                actions.append({
                    "action": "tp1",
                    "price": tp1,
                    "quantity": state["tp1_quantity"],
                    "side": "buy",
                    "reason": f"TP1 reached: {tp1:.4f}"
                })
                state["tp1_filled"] = True

        # TP2: 中端止盈（50%仓位）
        if not state["tp2_filled"] and state["tp2_quantity"] > 0:
            tp2 = state["tp2_price"]
            if direction == "long" and current_price >= tp2:
                actions.append({
                    "action": "tp2",
                    "price": tp2,
                    "quantity": state["tp2_quantity"],
                    "side": "sell",
                    "reason": f"TP2 reached: {tp2:.4f}"
                })
                state["tp2_filled"] = True
            elif direction == "short" and current_price <= tp2:
                actions.append({
                    "action": "tp2",
                    "price": tp2,
                    "quantity": state["tp2_quantity"],
                    "side": "buy",
                    "reason": f"TP2 reached: {tp2:.4f}"
                })
                state["tp2_filled"] = True

        if actions:
            self._persist(symbol)
        return actions

    def check_time_exit(self, symbol: str, current_price: float) -> Optional[Dict[str, Any]]:
        """
        检查时间止盈/止损。
        - 持仓超过 time_exit_after_hours 且盈利：减仓 time_exit_partial_pct
        - 持仓超过 max_hold_hours：强制全部平仓
        
        返回: None 或 {action: "time_partial"|"time_full", quantity, side, reason}
        """
        if symbol not in self._position_stops or not self._time_exit_enabled:
            return None

        state = self._position_stops[symbol]
        entry_time = state["entry_time"]
        now = datetime.now()
        hold_hours = (now - entry_time).total_seconds() / 3600.0

        direction = state["direction"]
        entry = state["entry_price"]
        remaining_qty = state["quantity"] - state["total_closed_qty"]
        if remaining_qty <= 0:
            return None

        is_profitable = (current_price > entry) if direction == "long" else (current_price < entry)

        # 最大持仓时间：强制全平
        if hold_hours >= self._max_hold_hours and not state["time_exit_done"]:
            state["time_exit_done"] = True
            self._persist(symbol)
            side = "sell" if direction == "long" else "buy"
            return {
                "action": "time_full",
                "quantity": remaining_qty,
                "side": side,
                "reason": f"Max hold time reached: {hold_hours:.1f}h > {self._max_hold_hours}h"
            }

        # 超时部分止盈：盈利且持仓超过阈值时减仓
        if (hold_hours >= self._time_exit_after_hours and not state["time_partial_done"]
                and is_profitable):
            partial_qty = remaining_qty * self._time_exit_partial_pct
            if partial_qty > 0:
                state["time_partial_done"] = True
                self._persist(symbol)
                side = "sell" if direction == "long" else "buy"
                return {
                    "action": "time_partial",
                    "quantity": partial_qty,
                    "side": side,
                    "reason": f"Time partial exit: {hold_hours:.1f}h, taking {self._time_exit_partial_pct:.0%} profit"
                }

        return None

    def check_volatility_spike(self, symbol: str, current_price: float) -> Optional[Dict[str, Any]]:
        """
        波动率突增检测：当 ATR 突然放大超过历史均值的 vol_spike_threshold 倍时，
        紧急减仓 vol_stop_partial_pct，避免黑天鹅行情下的大幅回撤。
        
        返回: None 或 {action: "vol_partial", quantity, side, reason}
        """
        if symbol not in self._position_stops or not self._volatility_stop_enabled:
            return None

        state = self._position_stops[symbol]

        # 冷却期内不重复触发
        if state.get("vol_lockout_until") and datetime.now() < state["vol_lockout_until"]:
            return None

        last_atr = state.get("last_atr", 0)
        avg_atr = state.get("avg_atr", 0)
        if last_atr <= 0 or avg_atr <= 0:
            return None

        spike_ratio = last_atr / avg_atr if avg_atr > 0 else 1.0

        if spike_ratio >= self._vol_spike_threshold and not state["vol_stop_triggered"]:
            state["vol_stop_triggered"] = True
            state["vol_spike_detected"] = True
            # 设置冷却期
            from datetime import timedelta
            state["vol_lockout_until"] = datetime.now() + timedelta(minutes=self._volatility_lockout_minutes)
            self._persist(symbol)

            direction = state["direction"]
            remaining_qty = state["quantity"] - state["total_closed_qty"]
            if remaining_qty <= 0:
                return None

            partial_qty = remaining_qty * self._vol_stop_partial_pct
            side = "sell" if direction == "long" else "buy"
            return {
                "action": "vol_partial",
                "quantity": partial_qty,
                "side": side,
                "reason": f"Volatility spike: ATR {spike_ratio:.1f}x avg, reducing {self._vol_stop_partial_pct:.0%} position"
            }

        return None

    def record_partial_close(self, symbol: str, quantity: float):
        """记录部分减仓，更新累计已平仓量"""
        if symbol in self._position_stops:
            self._position_stops[symbol]["total_closed_qty"] += quantity
            self._persist(symbol)

    def get_all_stops(self) -> Dict[str, Dict[str, Any]]:
        """获取所有持仓止损状态（用于监控）"""
        return dict(self._position_stops)

    def get_stats(self) -> Dict[str, Any]:
        """获取止损管理器统计信息"""
        total = len(self._position_stops)
        breakeven_count = sum(1 for s in self._position_stops.values() if s.get("breakeven_activated"))
        trailing_count = sum(1 for s in self._position_stops.values() if s.get("trailing_activated"))
        tp1_count = sum(1 for s in self._position_stops.values() if s.get("tp1_filled"))
        tp2_count = sum(1 for s in self._position_stops.values() if s.get("tp2_filled"))
        vol_spike_count = sum(1 for s in self._position_stops.values() if s.get("vol_spike_detected"))

        return {
            "total_positions": total,
            "breakeven_activated": breakeven_count,
            "trailing_activated": trailing_count,
            "tp1_filled": tp1_count,
            "tp2_filled": tp2_count,
            "volatility_spike_count": vol_spike_count,
            "strategy": self.strategy_name,
        }

    def remove_position(self, symbol: str):
        """移除持仓止损状态"""
        self._position_stops.pop(symbol, None)
        self._delete(symbol)

    def get_stop_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取止损信息"""
        return self._position_stops.get(symbol)

    # ─────────────────────────────────────────────────────────────
    # 企业级：止损状态跨重启持久化（watchdog 拉起后恢复进度，避免回退）
    # ─────────────────────────────────────────────────────────────

    def _init_state_table(self):
        """建 stop_loss_state 持久化表（strategy_name + symbol 为唯一键）。"""
        if not self._db_path:
            return
        conn = None
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stop_loss_state (
                    strategy_name VARCHAR(50),
                    symbol VARCHAR(50),
                    state_json TEXT,
                    updated_at DATETIME,
                    PRIMARY KEY (strategy_name, symbol)
                )
            """)
            conn.commit()
        except Exception as e:
            logger.warning(f"Failed to init stop_loss_state table: {e}")
        finally:
            if conn:
                conn.close()

    @staticmethod
    def _serialize_state(state: Dict[str, Any]) -> str:
        """把持仓止损状态序列化为 JSON（datetime 转 isoformat）。"""
        data: Dict[str, Any] = {}
        for key, value in state.items():
            if isinstance(value, datetime):
                data[key] = {"__dt__": value.isoformat()}
            elif key == "atr_history" and isinstance(value, list):
                data[key] = [
                    {"time": h["time"].isoformat() if isinstance(h.get("time"), datetime) else h.get("time"),
                     "atr": h.get("atr", 0.0)}
                    for h in value if isinstance(h, dict)
                ]
            else:
                data[key] = value
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _deserialize_state(raw: str) -> Dict[str, Any]:
        """从 JSON 还原持仓止损状态（isoformat 转 datetime）。"""
        data = json.loads(raw)
        for key, value in data.items():
            if isinstance(value, dict) and "__dt__" in value:
                try:
                    data[key] = datetime.fromisoformat(value["__dt__"])
                except (ValueError, TypeError):
                    data[key] = None
        hist = data.get("atr_history")
        if isinstance(hist, list):
            restored = []
            for h in hist:
                if isinstance(h, dict) and "time" in h:
                    t = h["time"]
                    if isinstance(t, str):
                        try:
                            t = datetime.fromisoformat(t)
                        except (ValueError, TypeError):
                            t = datetime.now()
                    restored.append({"time": t, "atr": h.get("atr", 0.0)})
            data["atr_history"] = restored
        return data

    def _persist(self, symbol: str):
        """把单持仓状态 upsert 到 SQLite（失败不阻断交易主流程）。"""
        if not self._persistence_enabled:
            return
        state = self._position_stops.get(symbol)
        if state is None:
            return
        conn = None
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "INSERT OR REPLACE INTO stop_loss_state "
                "(strategy_name, symbol, state_json, updated_at) VALUES (?, ?, ?, ?)",
                (self.strategy_name, symbol, self._serialize_state(state), datetime.now().isoformat()),
            )
            conn.commit()
        except Exception as e:
            logger.warning(f"Failed to persist stop-loss state for {symbol}: {e}")
        finally:
            if conn:
                conn.close()

    def _delete(self, symbol: str):
        """删除某持仓的持久化状态（平仓后调用）。"""
        if not self._persistence_enabled:
            return
        conn = None
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "DELETE FROM stop_loss_state WHERE strategy_name = ? AND symbol = ?",
                (self.strategy_name, symbol),
            )
            conn.commit()
        except Exception as e:
            logger.warning(f"Failed to delete stop-loss state for {symbol}: {e}")
        finally:
            if conn:
                conn.close()

    def _load_state(self):
        """从 SQLite 恢复本策略的所有持仓止损状态（__init__ 时调用）。"""
        if not self._persistence_enabled:
            return
        conn = None
        try:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT symbol, state_json FROM stop_loss_state WHERE strategy_name = ?",
                (self.strategy_name,),
            ).fetchall()
        except Exception as e:
            logger.debug(f"No persisted stop-loss state for strategy={self.strategy_name}: {e}")
            return
        finally:
            if conn:
                conn.close()

        loaded = 0
        for row in rows:
            try:
                state = self._deserialize_state(row["state_json"])
                self._position_stops[row["symbol"]] = state
                loaded += 1
            except Exception as e:
                logger.warning(f"Failed to restore stop-loss state for {row['symbol']}: {e}")
        if loaded > 0:
            logger.info(f"Loaded {loaded} persisted stop-loss state(s) for strategy={self.strategy_name}")

    def _recompute_tp_prices(self, state: Dict[str, Any]):
        """基于最新 entry_price 重算止盈目标/分级价格，保留 tp1/tp2 filled 进度。"""
        entry = state["entry_price"]
        direction = state["direction"]
        if direction == "long":
            tp_target = entry * (1 + self._take_profit_pct)
            state["tp_target"] = tp_target
            state["tp1_price"] = entry + (tp_target - entry) * self._tp1_pct
            state["tp2_price"] = entry + (tp_target - entry) * self._tp2_pct
        else:
            tp_target = entry * (1 - self._take_profit_pct)
            state["tp_target"] = tp_target
            state["tp1_price"] = entry - (entry - tp_target) * self._tp1_pct
            state["tp2_price"] = entry - (entry - tp_target) * self._tp2_pct
