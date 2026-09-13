"""负责订单的提交执行、限流重试、滑点控制与多策略止损管理。"""
import asyncio
import json
import os
import random
import time
from collections import deque
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

from .order_queue import TokenBucket
from .order_lifecycle_manager import OrderLifecycleManager, OrderStatus, OrderPhase
from .slippage_optimizer import SlippageOptimizer, SlippageTolerance
from configs.settings import get_currency_tier
from utils.helpers import validate_tp_sl_prices, calculate_take_profit, calculate_stop_loss, get_price_precision
from risk.profit_optimizer import EnhancedStopLoss, ProfitOptimizer
from core.mini_backtester import get_mini_backtester, TradeRecord
from core.capital_attrition_analyzer import CapitalAttritionAnalyzer
from core.direction_unifier import DirectionUnifier
from core.event_id import EventContext, event_id
from core.unified_layer import Event, EventType


class RetryableError(Enum):
    """可重试的错误类型"""
    RATE_LIMIT = "rate_limit"          # 429 / 请求频率限制
    SERVER_ERROR = "server_error"      # 5xx / 服务端错误
    NETWORK_ERROR = "network_error"    # 网络超时/连接失败
    TIMEOUT = "timeout"               # 请求超时
    UNKNOWN = "unknown"               # 未知错误


class PartialFillAction(Enum):
    """部分成交处理策略"""
    CANCEL_REMAINING = "cancel_remaining"   # 撤销剩余部分
    RESUBMIT_REMAINING = "resubmit"         # 重新提交剩余部分
    WAIT = "wait"                           # 等待自然成交
    MARKET_CLOSE = "market_close"           # 市价平掉剩余


class OrderExecutor:
    def __init__(self, config: Dict[str, Any], okx_client, redis_cache, sqlite_storage, account_manager=None, trade_journal=None):
        self.config = config
        self.okx_client = okx_client
        self.redis_cache = redis_cache
        self.sqlite_storage = sqlite_storage
        self._account_manager = account_manager
        self._trade_journal = trade_journal

        self._rate_limit = config["execution"]["rate_limit"]["rest_requests_per_second"]
        self._retry_attempts = config["execution"]["retry_attempts"]
        self._slippage_tolerance = config["execution"]["slippage_tolerance"]

        self._token_bucket = TokenBucket(self._rate_limit, capacity=20)

        self._last_request_time = 0
        self._active_orders: Dict[str, Dict[str, Any]] = {}

        self._order_queue = None

        # 强化止损管理器：按策略名分别管理
        self._stop_managers: Dict[str, EnhancedStopLoss] = {
            "grid": EnhancedStopLoss(config, "grid"),
            "trend": EnhancedStopLoss(config, "trend"),
            "scalping": EnhancedStopLoss(config, "scalping"),
            "arbitrage": EnhancedStopLoss(config, "arbitrage"),
            "spot_grid": EnhancedStopLoss(config, "spot_grid"),
            "spot_martingale": EnhancedStopLoss(config, "spot_martingale"),
        }
        self._position_strategy_map: Dict[str, str] = {}  # symbol -> strategy_name
        self._position_entry_time: Dict[str, datetime] = {}  # symbol -> entry_time
        self._min_hold_minutes = config.get("trading", {}).get("min_hold_minutes", 5)

        # 框架外持仓白名单（手动开单显式归属）：key = f"{symbol}:{side}"
        # 对账发现手动开单（strategy=unknown/sync）且 orphan_position_auto_close=False 时登记到此处，
        # 后续对账跳过重复告警/平仓，避免手动单被反复误判为失控仓位。
        self._manual_override_positions: Dict[str, Dict[str, Any]] = {}
        self._manual_override_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "manual_override_positions.json"
        )
        self._load_manual_override_positions()

        # 防重复触发：已发出止损信号但尚未成交的symbol集合
        self._stop_pending: set = set()

        # 最大并发持仓数
        self._max_concurrent_positions = config.get("trading", {}).get("max_concurrent_positions", 4)
        self._max_concurrent_positions_base = self._max_concurrent_positions  # P7: 保存基础值用于动态调整

        self._profit_optimizer: Optional[ProfitOptimizer] = None

        # 成交质量追踪器（由scheduler注入，可选）
        self._fill_quality_tracker = None

        # 执行质量监控器（由scheduler注入，可选）
        self._execution_quality_monitor = None

        # 滑点优化器（由scheduler注入，可选）
        self._slippage_optimizer: Optional[SlippageOptimizer] = None

        # P21: 撤单频率保护 - 防止频繁撤单触发交易所API封禁
        self._cancel_window_seconds = config.get("execution", {}).get("cancel_window_seconds", 60)
        self._cancel_max_per_window = config.get("execution", {}).get("cancel_max_per_window", 10)
        self._cancel_timestamps: deque = deque(maxlen=100)  # 撤单时间戳滑动窗口
        self._cancel_blocked_until: float = 0.0  # 撤单被封锁截止时间
        self._cancel_block_cooldown = config.get("execution", {}).get("cancel_block_cooldown", 300)  # 封锁5分钟

        # 五层风控拦截器（由scheduler注入，用于L4单日开仓次数/连续亏损跟踪）
        self._risk_gate = None

        # P2: 独立风控裁决器（由scheduler注入，下单前最终裁决 + traceID 全链路串联）
        self._risk_adjudicator = None

        # 告警管理器（由scheduler注入，用于框架外持仓兜底告警）
        self._alert_manager = None

        # 事件总线（由scheduler注入，P1 埋点用；None 时埋点静默跳过）
        self._event_bus = None

        # 条件单管理器（由scheduler注入，可选）
        self._conditional_manager = None

        # 统一自适应止损止盈引擎（由scheduler注入，可选；下单兜底重算时优先使用）
        self._adaptive_tp_sl_engine = None

        # 首次资金划转标记（确保下单前资金从资金账户划到交易账户）
        self._funds_transferred = False

        # 订单生命周期管理器
        self._lifecycle_manager = OrderLifecycleManager(config)

        # 策略状态清理回调：symbol -> list of callables
        # 当订单失败（如51169无持仓）时，通知策略清理内部状态
        self._position_cleanup_callbacks: list = []

        # 成交回执回调列表：订单完全成交时通知策略（ghost_close 专项 Phase 1 打通成交回执桥）
        self._fill_callbacks: list = []

        # 策略主动管理持仓 symbol 提供器（由 scheduler 注入，惰性求值）
        # 返回 set[str]：当前被策略（如 grid）主动管理的 symbol 集合。
        # 用于重启对账时区分「框架外 orphan」与「策略管理但 DB 记录被 sync 污染」的持仓。
        self._managed_position_symbols_provider = None

        # 方案A（fill 回执丢失根因修复）：活跃订单落盘 + 重启恢复 + 全量对账补回执。
        # persist_active_orders 配置开关：关闭即回退到纯内存态 + P22 兜底的现状。
        self._persist_active_orders = bool(
            config.get("execution", {}).get("persist_active_orders", False)
        )
        self._fill_receipt_recovered = 0   # 观测指标：全量对账补发的回执数
        self._last_fill_reconcile_ts = 0.0  # 全量对账低频节流时间戳
        self._fill_reconcile_interval = float(
            config.get("execution", {}).get("fill_reconcile_interval_sec", 30.0)
        )

        # 策略平仓盈亏回调列表：平仓 PnL 计算完成后通知策略（P1 日内亏损熔断等）
        self._strategy_pnl_callbacks: list = []

        # 同symbol连续失败退避：避免短期内反复尝试同一symbol造成API浪费
        # symbol -> {"fail_count": int, "last_fail_time": float, "blocked_until": float}
        self._symbol_fail_state: Dict[str, Dict[str, Any]] = {}
        self._max_fail_before_block = 3  # 连续失败3次后进入退避
        self._block_duration = 60  # 退避60秒
        self._fail_window = 300  # 失败计数窗口5分钟

        # P25: entry_price缓存，避免重复查询数据库中不存在的open记录
        # 格式: {symbol: {"entry_price": float, "cached_at": float, "source": str}}
        self._entry_price_cache: Dict[str, Dict[str, Any]] = {}
        self._entry_price_cache_ttl = 300  # 缓存5分钟，避免持仓期间反复查询
        self._entry_price_cache_ttl_degraded = 1800  # P26: 网络降级时延长到30分钟
        self._entry_price_cache_ttl_position = 3600  # P27: 从持仓获取的entry_price缓存1小时（avgPx不会变）
        self._entry_price_cache_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "entry_price_cache.json"
        )
        # P26: 启动时加载持久化缓存
        self._load_entry_price_cache()

        # 方案A：启动时恢复落盘的活跃订单（fill 回执丢失根因修复）
        self._restore_active_orders()

        # P2: 动态波动过滤替代固定时段禁交易
        # 不再使用硬编码禁交易时段，改为基于24h振幅的动态过滤
        # 当 symbol 的24h振幅超过阈值时才禁止开仓（避免高波动磨损）
        self._no_trade_volatility_threshold = config.get("execution", {}).get(
            "no_trade_volatility_threshold", 0.10  # 24h振幅超过10%时禁止开仓
        )
        # 保留旧兼容字段，但默认不再启用
        self._no_trade_hours: set = set(
            config.get("execution", {}).get("no_trade_hours", [])
        )

        # 资金磨损分析器（由scheduler注入，可选）
        self._attrition_analyzer: Optional[CapitalAttritionAnalyzer] = None

        # ═══════════════════════════════════════════════════════════════
        # 生产级执行引擎增强配置
        # ═══════════════════════════════════════════════════════════════

        # ── 幂等键 ──
        # 每次下单生成唯一 clOrdId，防止网络重试导致重复下单
        # OKX 支持 clOrdId 幂等：相同 clOrdId 的重复请求返回已有订单结果
        self._idempotency_prefix = config.get("execution", {}).get(
            "idempotency_prefix", "okxqt"
        )
        self._idempotency_enabled = config.get("execution", {}).get(
            "idempotency_enabled", True
        )

        # ── 指数退避重试 ──
        exec_cfg = config.get("execution", {})
        self._retry_base_delay = exec_cfg.get("retry_base_delay_sec", 1.0)
        self._retry_max_delay = exec_cfg.get("retry_max_delay_sec", 60.0)
        self._retry_jitter_pct = exec_cfg.get("retry_jitter_pct", 0.3)
        self._retry_multiplier = exec_cfg.get("retry_multiplier", 2.0)
        self._max_retry_attempts = exec_cfg.get("retry_attempts", 3)
        # 可重试错误码映射
        self._retryable_error_codes: Dict[str, RetryableError] = {
            "429": RetryableError.RATE_LIMIT,
            "500": RetryableError.SERVER_ERROR,
            "502": RetryableError.SERVER_ERROR,
            "503": RetryableError.SERVER_ERROR,
            "504": RetryableError.SERVER_ERROR,
            "51000": RetryableError.RATE_LIMIT,  # OKX 请求频率限制
            "51001": RetryableError.RATE_LIMIT,  # OKX 请求次数超限
            "51006": RetryableError.RATE_LIMIT,  # OKX 接口限频
        }

        # ── 部分成交处理 ──
        self._partial_fill_enabled = exec_cfg.get("partial_fill_enabled", True)
        self._partial_fill_action = PartialFillAction(
            exec_cfg.get("partial_fill_action", "cancel_remaining")
        )
        self._partial_fill_min_fill_ratio = exec_cfg.get("partial_fill_min_fill_ratio", 0.5)
        self._partial_fill_max_wait_sec = exec_cfg.get("partial_fill_max_wait_sec", 10.0)
        # 追踪部分成交订单: clOrdId -> {remaining_qty, filled_qty, action, start_time}
        self._partial_fill_tracker: Dict[str, Dict[str, Any]] = {}
        # 部分成交跟踪记录的过期清理阈值（秒）：避免长期挂单/孤儿跟踪记录内存泄漏
        self._partial_fill_tracker_ttl = exec_cfg.get("partial_fill_tracker_ttl_sec", 300.0)

        # ── 执行统计 ──
        self._execution_stats = {
            "total_orders": 0,
            "duplicate_prevented": 0,
            "retry_count": 0,
            "partial_fills": 0,
            "partial_fill_resolved": 0,
            "idempotency_hits": 0,
        }
        # 已发送的 clOrdId 缓存（用于幂等检测）: clOrdId -> (timestamp, exchange_order_id)
        self._sent_clordids: Dict[str, Tuple[float, str]] = {}
        self._clordid_cache_max = exec_cfg.get("clordid_cache_size", 1000)
        self._clordid_cache_ttl = exec_cfg.get("clordid_cache_ttl_sec", 3600)

    def set_order_queue(self, order_queue):
        self._order_queue = order_queue

    def _get_min_notional(self, symbol: str, price: float, strategy_name: str = "") -> float:
        """获取最低名义价值：确保每笔交易利润能覆盖手续费且有实际意义
        
        手续费来回 0.1%(taker*2)，要求利润 >= 手续费*3，按最低可接受利润率 0.5% 反算。
        网格/抢单/马丁策略：最低利润率 0.3%, 趋势/套利：最低利润率 0.5%
        """
        try:
            taker_fee = self.config.get("trading", {}).get("taker_fee_rate", 0.0005)
            round_trip_fee = taker_fee * 2  # 开平各一次
            
            # 按策略类型选择最低目标利润率
            low_margin_strategies = {"grid", "scalping", "spot_grid", "spot_martingale"}
            min_target_profit_pct = 0.003 if strategy_name in low_margin_strategies else 0.005
            
            # 安全利润 = 手续费 * 3 / 目标利润率 = 最低名义价值
            safe_profit_needed = round_trip_fee * 3
            min_notional = safe_profit_needed / min_target_profit_pct
            
            # 根据账户资金动态调整下限
            account_info = self.okx_client.get_account_info()
            if account_info:
                total_eq = float(account_info.get("totalEq", 0))
                if total_eq > 0:
                    # 每笔交易占用不超过总资金的 25%（小资金场景放宽）
                    max_per_trade = total_eq * 0.25
                    # 但不低于 2 USDT（适配6U小资金场景，OKX最低名义价值约2-3U）
                    return max(min(min_notional, max_per_trade), 2.0)
            
            return max(min_notional, 2.0)
        except Exception:
            return 2.0  # 异常时使用宽松下限

    def set_profit_optimizer(self, profit_optimizer: ProfitOptimizer):
        self._profit_optimizer = profit_optimizer

    def set_fill_quality_tracker(self, tracker):
        """注入成交质量追踪器"""
        self._fill_quality_tracker = tracker

    def set_execution_quality_monitor(self, monitor):
        """注入算法执行质量监控器"""
        self._execution_quality_monitor = monitor

    def set_slippage_optimizer(self, optimizer: SlippageOptimizer):
        """注入滑点优化器，用于动态限价偏移和成交率优化"""
        self._slippage_optimizer = optimizer

    def set_risk_gate(self, risk_gate):
        """注入五层风控拦截器（用于L4单日开仓次数/连续亏损跟踪）"""
        self._risk_gate = risk_gate

    def set_risk_adjudicator(self, adjudicator):
        """P2: 注入独立风控裁决器（下单前最终裁决 + traceID 全链路串联）。"""
        self._risk_adjudicator = adjudicator

    def set_conditional_manager(self, conditional_manager):
        """注入条件单管理器（用于止盈后止损上移等联动操作）"""
        self._conditional_manager = conditional_manager

    def set_trade_cost_analyzer(self, analyzer):
        """注入精准交易成本分析器 - 杜绝磨损型交易"""
        self._trade_cost_analyzer = analyzer

    def set_trade_auditor(self, auditor):
        """注入交易审核器 - 全链路审核"""
        self._trade_auditor = auditor

    def set_adaptive_tp_sl_engine(self, engine):
        """注入统一自适应止损止盈引擎，用于下单兜底重算时自适应计算 SL/TP"""
        self._adaptive_tp_sl_engine = engine

    def set_intelligent_agent(self, agent):
        """注入智能交易体 - 自适应判断 + 成长性学习"""
        self._intelligent_agent = agent

    def set_attrition_analyzer(self, analyzer: CapitalAttritionAnalyzer):
        """注入资金磨损分析器，用于记录交易手续费、滑点等磨损数据"""
        self._attrition_analyzer = analyzer

    def set_alert_manager(self, alert_manager):
        """注入告警管理器，用于框架外持仓(strategy=unknown)兜底告警"""
        self._alert_manager = alert_manager

    def set_managed_position_symbols_provider(self, provider):
        """注入策略主动管理持仓 symbol 提供器（返回 set[str]，惰性求值）。

        用于重启对账区分框架外 orphan 与策略管理持仓（DB 记录可能被 P7-4 sync 污染）。
        """
        self._managed_position_symbols_provider = provider

    def _get_managed_position_symbols(self) -> set:
        """获取策略主动管理的 symbol 集合，异常或未注入时返回空集合。"""
        try:
            if self._managed_position_symbols_provider is None:
                return set()
            return set(self._managed_position_symbols_provider() or [])
        except Exception:
            return set()


    def set_event_bus(self, event_bus):
        """P1 埋点：注入事件总线，下单/成交/平仓时发布事件（配合事件溯源落盘）。"""
        self._event_bus = event_bus

    def _publish_event(self, event_type, data: Dict[str, Any]):
        """P1 埋点：通过事件总线发布事件。失败不影响下单主流程（仅 DEBUG 记录）。"""
        try:
            if self._event_bus is None:
                return
            self._event_bus.publish_sync(Event(event_type, data))
        except Exception as e:
            logger.debug(f"Event publish skipped ({getattr(event_type, 'value', event_type)}): {e}")

    # ═══════════════════════════════════════════════════════════════
    # 生产级幂等键管理
    # ═══════════════════════════════════════════════════════════════

    def _generate_idempotency_key(self, symbol: str, strategy_name: str,
                                   signal_type: str, timestamp: float = None) -> str:
        """生成确定性、可复现的幂等键（clOrdId）

        OKX clOrdId 要求: 纯字母+数字，最长32位，不允许任何特殊字符（含下划线）
        格式: {prefix}{strategy}{symbol}{timestamp_ms}{signal_type}

        企业级强化：去掉随机后缀，改用 signal_type 确定性指纹，使同一
        (strategy, symbol, signal_type, 毫秒时间戳) 生成相同 clOrdId——
        网络重试或重复信号能命中同一幂等键，由交易所侧据此去重，避免重复下单。
        """
        import re
        if timestamp is None:
            timestamp = time.time()
        ts_ms = int(timestamp * 1000)
        # 取符号名（去掉-USDT后缀，并移除所有非字母数字字符）
        short_symbol = symbol.replace("-USDT", "").replace("-USDC", "")
        short_symbol = re.sub(r'[^a-zA-Z0-9]', '', short_symbol)[:6]
        # 截断策略名长度（同时移除特殊字符）
        short_strategy = re.sub(r'[^a-zA-Z0-9]', '', strategy_name)[:4]
        # 信号类型指纹（确定性，替换原随机后缀，保证同参数可复现）
        short_type = re.sub(r'[^a-zA-Z0-9]', '', str(signal_type or ""))[:4]
        # 构建 clOrdId（纯字母+数字，无任何分隔符，确保不超过32位）
        clordid = f"{self._idempotency_prefix}{short_strategy}{short_symbol}{ts_ms}{short_type}"
        # 截断到32位
        if len(clordid) > 32:
            clordid = clordid[:32]
        return clordid

    def _check_idempotency(self, clordid: str) -> Optional[str]:
        """检查幂等键是否已存在，返回已有的 exchange_order_id 或 None"""
        self._cleanup_clordid_cache()
        if clordid in self._sent_clordids:
            ts, existing_ord_id = self._sent_clordids[clordid]
            self._execution_stats["idempotency_hits"] += 1
            logger.info(
                f"Idempotency key hit: {clordid} -> existing order {existing_ord_id}"
            )
            return existing_ord_id
        return None

    def _record_idempotency_key(self, clordid: str, exchange_order_id: str):
        """记录已使用的幂等键"""
        self._sent_clordids[clordid] = (time.time(), exchange_order_id)
        # 缓存超限时清理最旧的一半
        if len(self._sent_clordids) > self._clordid_cache_max:
            sorted_keys = sorted(
                self._sent_clordids.items(), key=lambda x: x[1][0]
            )
            remove_count = len(sorted_keys) // 2
            for key, _ in sorted_keys[:remove_count]:
                del self._sent_clordids[key]

    def _cleanup_clordid_cache(self):
        """清理过期的 clOrdId 缓存"""
        now = time.time()
        expired = [
            k for k, (ts, _) in self._sent_clordids.items()
            if now - ts > self._clordid_cache_ttl
        ]
        for k in expired:
            del self._sent_clordids[k]

    def _cleanup_partial_fill_tracker(self):
        """清理超时的部分成交跟踪记录，防止长期挂单/孤儿记录内存泄漏。

        部分成交在 WAIT 分支若长时间未被再次回调（如订单被交易所取消、
        或跟踪循环异常中断），记录会永久残留。这里按 start_time 过期清理。
        """
        if not self._partial_fill_tracker:
            return
        now = time.time()
        stale = [
            k for k, tr in self._partial_fill_tracker.items()
            if now - tr.get("start_time", now) > self._partial_fill_tracker_ttl
        ]
        for k in stale:
            self._partial_fill_tracker.pop(k, None)
        if stale:
            logger.info(f"Cleaned {len(stale)} stale partial-fill tracker entries")

    # ═══════════════════════════════════════════════════════════════
    # 生产级指数退避重试
    # ═══════════════════════════════════════════════════════════════

    def _calculate_backoff_delay(self, attempt: int, error_type: RetryableError = None) -> float:
        """计算指数退避延迟（带随机抖动）
        
        公式: delay = min(max_delay, base_delay * multiplier^(attempt-1)) + jitter
        jitter = delay * jitter_pct * random(-1, 1)
        
        不同错误类型使用不同基础延迟:
        - RATE_LIMIT: base_delay * 2（更保守）
        - SERVER_ERROR: base_delay（标准）
        - NETWORK_ERROR: base_delay * 0.5（快速重试）
        """
        base = self._retry_base_delay
        if error_type == RetryableError.RATE_LIMIT:
            base *= 2.0
        elif error_type == RetryableError.NETWORK_ERROR:
            base *= 0.5

        backoff = base * (self._retry_multiplier ** max(attempt - 1, 0))

        # 添加随机抖动，避免惊群效应（在capping之前，确保最终值不超过max_delay）
        jitter = backoff * self._retry_jitter_pct * (random.random() * 2 - 1)
        backoff = max(0.1, backoff + jitter)

        # 抖动后再capping，确保不超过上限
        backoff = min(self._retry_max_delay, backoff)

        return backoff

    def _classify_error(self, error_code: str, error_msg: str = "") -> Optional[RetryableError]:
        """分类错误类型，判断是否可重试"""
        if not error_code:
            # 网络异常（无错误码）= 可重试
            return RetryableError.NETWORK_ERROR

        # 精确匹配可重试错误码
        if error_code in self._retryable_error_codes:
            return self._retryable_error_codes[error_code]

        # HTTP 状态码模式匹配（仅3位HTTP状态码，排除OKX 5位业务错误码）
        if len(error_code) == 3 and error_code.startswith("5"):
            return RetryableError.SERVER_ERROR
        if error_code == "429":
            return RetryableError.RATE_LIMIT

        # 消息中包含超时关键词
        msg_lower = error_msg.lower()
        if any(kw in msg_lower for kw in ["timeout", "timed out", "connection reset"]):
            return RetryableError.NETWORK_ERROR

        return None  # 不可重试

    # ═══════════════════════════════════════════════════════════════
    # 生产级部分成交处理
    # ═══════════════════════════════════════════════════════════════

    async def _handle_partial_fill(self, exchange_order_id: str,
                                    order_info: Dict[str, Any],
                                    filled_qty: float,
                                    total_qty: float,
                                    filled_price: float) -> bool:
        """处理部分成交：根据策略决定取消/重提/等待
        
        Returns:
            True: 部分成交已处理完毕
            False: 需要继续跟踪
        """
        if not self._partial_fill_enabled:
            return True  # 不处理部分成交

        remaining_qty = total_qty - filled_qty
        fill_ratio = filled_qty / total_qty if total_qty > 0 else 0

        symbol = order_info.get("symbol", "")
        clordid = order_info.get("clOrdId", "")

        self._execution_stats["partial_fills"] += 1

        logger.info(
            f"Partial fill detected: {symbol} ordId={exchange_order_id} "
            f"filled={filled_qty}/{total_qty} ({fill_ratio:.1%}) "
            f"remaining={remaining_qty}, price={filled_price:.4f}"
        )

        # 检查是否已有部分成交跟踪记录
        if clordid in self._partial_fill_tracker:
            tracker = self._partial_fill_tracker[clordid]
            tracker["filled_qty"] = filled_qty
            tracker["remaining_qty"] = remaining_qty
        else:
            self._partial_fill_tracker[clordid] = {
                "symbol": symbol,
                "exchange_order_id": exchange_order_id,
                "total_qty": total_qty,
                "filled_qty": filled_qty,
                "remaining_qty": remaining_qty,
                "filled_price": filled_price,
                "action": self._partial_fill_action,
                "start_time": time.time(),
                "strategy_name": order_info.get("strategy_name", ""),
            }

        # 判断处理策略
        action = self._partial_fill_action

        # 成交率太低：直接取消剩余
        if fill_ratio < self._partial_fill_min_fill_ratio:
            action = PartialFillAction.CANCEL_REMAINING
            logger.info(
                f"Partial fill ratio {fill_ratio:.1%} < min {self._partial_fill_min_fill_ratio:.1%}, "
                f"cancelling remaining"
            )

        if action == PartialFillAction.CANCEL_REMAINING:
            await self._cancel_remaining_order(symbol, exchange_order_id, clordid)
            self._partial_fill_tracker.pop(clordid, None)
            self._execution_stats["partial_fill_resolved"] += 1
            return True

        elif action == PartialFillAction.RESUBMIT_REMAINING:
            # 撤销剩余并重新提交
            await self._cancel_remaining_order(symbol, exchange_order_id, clordid)
            # 重新提交剩余数量
            if remaining_qty > 0:
                # 账本一致性：remaining_qty 为合约张数，_execute_order 期望币数，需先转币数
                remaining_coins = self.okx_client.contracts_to_coins(symbol, remaining_qty)
                new_order = dict(order_info)
                new_order["quantity"] = remaining_coins
                new_order["clOrdId"] = ""  # 新订单需要新ID
                await self._execute_order(new_order)
            self._partial_fill_tracker.pop(clordid, None)
            self._execution_stats["partial_fill_resolved"] += 1
            return True

        elif action == PartialFillAction.WAIT:
            # 等待超时后决定
            tracker = self._partial_fill_tracker.get(clordid, {})
            elapsed = time.time() - tracker.get("start_time", time.time())
            if elapsed > self._partial_fill_max_wait_sec:
                logger.info(
                    f"Partial fill wait timeout ({elapsed:.1f}s > {self._partial_fill_max_wait_sec}s), "
                    f"cancelling remaining"
                )
                await self._cancel_remaining_order(symbol, exchange_order_id, clordid)
                self._partial_fill_tracker.pop(clordid, None)
                self._execution_stats["partial_fill_resolved"] += 1
                return True
            return False  # 继续等待

        elif action == PartialFillAction.MARKET_CLOSE:
            # 市价平掉剩余（仅平仓信号适用）
            is_close = any(kw in str(order_info.get("signal_type", "")).lower()
                          for kw in ["close", "stop_loss", "exit", "reduce"])
            if is_close and remaining_qty > 0:
                logger.info(f"Market closing remaining {remaining_qty} for {symbol}")
                await self._cancel_remaining_order(symbol, exchange_order_id, clordid)
                # 市价单平剩余
                close_signal = {
                    "symbol": symbol,
                    "direction": DirectionUnifier.to_side(DirectionUnifier.opposite(self._resolve_track_direction(order_info))),
                    "price": filled_price,
                    "quantity": self.okx_client.contracts_to_coins(symbol, remaining_qty),
                    "leverage": order_info.get("leverage", 1),
                    "strategy_name": order_info.get("strategy_name", ""),
                    "signal_type": "partial_fill_market_close",
                    "reduce_only": True,
                }
                await self._execute_order(close_signal)
            self._partial_fill_tracker.pop(clordid, None)
            self._execution_stats["partial_fill_resolved"] += 1
            return True

        return True

    # ═══════════════════════════════════════════════════════════════
    # P21: 撤单频率保护
    # ═══════════════════════════════════════════════════════════════

    def _check_cancel_allowed(self, symbol: str) -> bool:
        """P21: 检查是否允许撤单，防止API封禁"""
        now = time.time()

        # 检查是否在封锁期内
        if now < self._cancel_blocked_until:
            return False

        # 清理过期时间戳
        cutoff = now - self._cancel_window_seconds
        while self._cancel_timestamps and self._cancel_timestamps[0] < cutoff:
            self._cancel_timestamps.popleft()

        # 检查是否超限
        if len(self._cancel_timestamps) >= self._cancel_max_per_window:
            self._cancel_blocked_until = now + self._cancel_block_cooldown
            logger.critical(
                f"P21: CANCEL RATE LIMIT EXCEEDED: {len(self._cancel_timestamps)} "
                f"cancels in {self._cancel_window_seconds}s (max={self._cancel_max_per_window}), "
                f"blocking all cancels for {self._cancel_block_cooldown}s"
            )
            return False

        return True

    def _record_cancel(self, symbol: str) -> None:
        """P21: 记录一次撤单操作"""
        self._cancel_timestamps.append(time.time())

    async def _cancel_remaining_order(self, symbol: str, exchange_order_id: str,
                                       clordid: str = ""):
        """取消订单剩余未成交部分"""
        # P21: 撤单频率保护检查
        if not self._check_cancel_allowed(symbol):
            logger.warning(
                f"P21: Cancel blocked for {symbol} - rate limit exceeded "
                f"({self._cancel_max_per_window}/{self._cancel_window_seconds}s)"
            )
            return
        try:
            result = self.okx_client.cancel_order(symbol, exchange_order_id)
            if result and not result.get("_failed", False):
                self._record_cancel(symbol)
                # P3 埋点：撤单成功 → 发布 ORDER_CANCELLED 事件（事件溯源落盘）
                self._publish_event(EventType.ORDER_CANCELLED, {
                    "symbol": symbol,
                    "exchange_order_id": exchange_order_id,
                    "reason": "cancel_remaining",
                    "clordid": clordid,
                })
                logger.info(
                    f"Remaining order cancelled: {symbol} ordId={exchange_order_id}"
                )
            else:
                logger.warning(
                    f"Failed to cancel remaining order {exchange_order_id}: "
                    f"{result.get('sMsg', '') if result else 'no response'}"
                )
        except Exception as e:
            logger.error(f"Error cancelling remaining order {exchange_order_id}: {e}")

    # ═══════════════════════════════════════════════════════════════
    # 执行统计
    # ═══════════════════════════════════════════════════════════════

    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行引擎统计信息"""
        stats = dict(self._execution_stats)
        stats.update({
            "active_orders": len(self._active_orders),
            "partial_fills_pending": len(self._partial_fill_tracker),
            "clordid_cache_size": len(self._sent_clordids),
            "symbol_fail_states": len(self._symbol_fail_state),
            # 方案A观测指标：活跃订单落盘开关 + 全量对账补发的回执数
            "persist_active_orders": self._persist_active_orders,
            "fill_receipt_recovered": self._fill_receipt_recovered,
            "timestamp": datetime.now().isoformat(),
        })
        return stats

    def register_position_cleanup_callback(self, callback):
        """注册持仓清理回调函数（策略注册自己的状态清理逻辑）"""
        self._position_cleanup_callbacks.append(callback)

    def register_fill_callback(self, callback):
        """注册成交回执回调函数（ghost_close 专项 Phase 1：订单完全成交时通知策略）"""
        self._fill_callbacks.append(callback)

    def _notify_fill_callbacks(self, fill_payload: Dict[str, Any]):
        """通知所有注册的成交回执回调"""
        for callback in self._fill_callbacks:
            try:
                callback(fill_payload)
            except Exception as e:
                logger.error(f"Fill callback error: {e}")

    # ═══════════════════════════════════════════════════════════════
    # 方案A：fill 回执丢失根因修复（活跃订单落盘 + 重启恢复 + 全量对账补回执）
    # ═══════════════════════════════════════════════════════════════

    def _restore_active_orders(self):
        """启动时从 SQLite 恢复落盘的活跃订单，根治「重启 _active_orders 内存态清空
        导致成交回执永久丢失」这一最严重根因（丢失点 #1）。

        恢复后由现有 `_order_tracking_loop` 每 1s 调用 `_track_active_orders` 自动接管，
        对恢复的 pending 订单逐个检测 filled 并走完整成交处理（含回执桥）。
        """
        if not self._persist_active_orders or not self.sqlite_storage:
            return
        try:
            restored = self.sqlite_storage.load_active_orders()
            if not restored:
                return
            # 仅恢复仍在 pending 的订单；已标记 filled/cancelled 的陈旧记录跳过并清理
            stale = []
            for exchange_order_id, order_info in restored.items():
                status = order_info.get("status", "pending")
                if status in ("filled", "cancelled", "failed"):
                    stale.append(exchange_order_id)
                    continue
                order_info.setdefault("status", "pending")
                self._active_orders[exchange_order_id] = order_info
            for sid in stale:
                self.sqlite_storage.delete_active_order(sid)
            if self._active_orders:
                logger.info(
                    f"[fill-receipt-A] restored {len(self._active_orders)} active orders "
                    f"from disk (persist_active_orders on)"
                )
        except Exception as e:
            logger.warning(f"_restore_active_orders failed: {e}")

    def _persist_active_order(self, exchange_order_id: str, order_info: Dict[str, Any]):
        """落盘活跃订单（下单成功进入 pending 时调用）。幂等、异步失败静默。"""
        if not self._persist_active_orders or not self.sqlite_storage:
            return
        try:
            self.sqlite_storage.save_active_order(exchange_order_id, order_info)
        except Exception as e:
            logger.warning(f"_persist_active_order failed for {exchange_order_id}: {e}")

    def _remove_active_order(self, exchange_order_id: str):
        """删除落盘的活跃订单（订单已成交/撤单/失败时调用）。"""
        if not self._persist_active_orders or not self.sqlite_storage:
            return
        try:
            self.sqlite_storage.delete_active_order(exchange_order_id)
        except Exception as e:
            logger.warning(f"_remove_active_order failed for {exchange_order_id}: {e}")

    async def _reconcile_fill_receipts(self):
        """方案A：REST 全量对账，低频补发因轮询间隙成交 / REST 失败 / 重启时间窗遗漏
        而丢失的成交回执（覆盖丢失点 #2/#3/#4，并加速重启后的回执恢复）。

        仅对 `_active_orders` 中仍 pending、但交易所已 filled 的订单补发回执；
        策略侧 `on_order_filled` 通过 `pending_entries.pop` 幂等消费，重复补发无害。
        补发后仅打「fill_reconciled」标记避免本方法重复补发，订单的完整成交记账
        （record_fill/update_status/资金费/写 trade_records）仍由 `_track_active_orders` 完成。
        """
        if not self._persist_active_orders or not self._active_orders:
            return
        try:
            if not hasattr(self.okx_client, "get_order_history"):
                return
            filled_orders = self.okx_client.get_order_history(limit=50)
            if not filled_orders:
                return
            for order in filled_orders:
                exchange_order_id = order.get("ordId", "")
                if not exchange_order_id or order.get("state", "") != "filled":
                    continue
                order_info = self._active_orders.get(exchange_order_id)
                if not order_info or order_info.get("fill_reconciled"):
                    continue
                avg_price = float(order.get("avgPx", "0") or 0)
                filled_qty = float(order.get("fillSz", "0") or 0) or order_info.get("quantity", 0)
                self._notify_fill_callbacks({
                    "symbol": order_info.get("symbol", ""),
                    "direction": order_info.get("direction", ""),
                    "pos_side": order_info.get("pos_side", ""),
                    "signal_type": order_info.get("signal_type", ""),
                    "strategy_name": order_info.get("strategy_name", "grid"),
                    "filled_qty": filled_qty,
                    "avg_price": avg_price,
                    "quantity": order_info.get("quantity", 0),
                    "clOrdId": order_info.get("clOrdId", ""),
                    "exchange_order_id": exchange_order_id,
                    "reduce_only": order_info.get("reduce_only", False),
                })
                order_info["fill_reconciled"] = True
                self._fill_receipt_recovered += 1
                logger.info(
                    f"[fill-receipt-A] recovered fill receipt: {exchange_order_id} "
                    f"{order_info.get('symbol')} {order_info.get('strategy_name')}"
                )
        except Exception as e:
            logger.warning(f"_reconcile_fill_receipts error: {e}")

    def register_strategy_pnl_callback(self, callback):
        """注册策略平仓盈亏回调（P1：策略日内亏损熔断等场景，平仓 PnL 计算完成后触发）"""
        self._strategy_pnl_callbacks.append(callback)

    def _notify_strategy_pnl_callbacks(self, strategy_name: str, symbol: str, pnl: float):
        """通知所有注册的策略平仓盈亏回调"""
        for callback in self._strategy_pnl_callbacks:
            try:
                callback(strategy_name, symbol, float(pnl))
            except Exception as e:
                logger.debug(f"Strategy PnL callback error: {e}")

    # ═══════════════════════════════════════════════════════════════
    # 生产级批量操作 (v2.0)
    # ═══════════════════════════════════════════════════════════════

    async def cancel_all_orders_by_strategy(self, strategy_name: str) -> int:
        """批量取消指定策略的所有活跃订单
        
        Returns: 取消的订单数量
        """
        cancelled = 0
        orders_to_cancel = []
        
        for ord_id, order_info in list(self._active_orders.items()):
            if order_info.get("strategy_name") == strategy_name:
                orders_to_cancel.append((ord_id, order_info))
        
        for ord_id, order_info in orders_to_cancel:
            try:
                symbol = order_info.get("symbol", "")
                # P21: 撤单频率保护
                if not self._check_cancel_allowed(symbol):
                    logger.warning(f"P21: Batch cancel blocked for {symbol} - rate limit")
                    break
                result = self.okx_client.cancel_order(symbol, ord_id)
                if result and not result.get("_failed", False):
                    self._record_cancel(symbol)
                    cancelled += 1
                    logger.info(f"Batch cancel: {ord_id} ({strategy_name}/{symbol})")
                else:
                    logger.warning(f"Batch cancel failed: {ord_id} ({strategy_name}/{symbol})")
            except Exception as e:
                logger.error(f"Batch cancel error for {ord_id}: {e}")
        
        if cancelled > 0:
            logger.info(f"Batch cancelled {cancelled} orders for strategy '{strategy_name}'")
        return cancelled

    async def close_all_positions_by_strategy(self, strategy_name: str) -> int:
        """批量平仓指定策略的所有持仓
        
        Returns: 提交的平仓订单数量
        """
        closed = 0
        try:
            positions = self.okx_client.get_positions()
            if not positions:
                return 0
            
            for pos_data in positions:
                try:
                    pos = self.okx_client._parse_position(pos_data)
                    if not pos or abs(float(pos.quantity)) == 0:
                        continue
                    
                    # 检查策略关联
                    symbol = pos.symbol
                    pos_side = pos.side
                    stored_strategy = self._position_strategy_map.get(symbol, "")
                    if stored_strategy != strategy_name:
                        continue
                    
                    # 构建平仓信号
                    close_side = DirectionUnifier.to_side(DirectionUnifier.opposite(pos_side))
                    pos_qty_coins = self.okx_client.contracts_to_coins(
                        symbol, abs(float(pos.quantity))
                    )
                    
                    close_signal = {
                        "symbol": symbol,
                        "direction": close_side,
                        "pos_side": pos_side,
                        "price": float(pos.mark_price) if pos.mark_price else 0,
                        "quantity": pos_qty_coins,
                        "leverage": int(pos.leverage) if pos.leverage else 1,
                        "strategy_name": strategy_name,
                        "signal_type": "batch_close_all",
                        "reduce_only": True,
                        "close_position": True,
                        "reason": f"batch_close_strategy_{strategy_name}",
                    }
                    
                    if self._order_queue:
                        await self._order_queue.add_order(close_signal)
                        closed += 1
                        logger.info(
                            f"Batch close: {symbol} {pos_side} "
                            f"qty={pos_qty_coins:.4f} ({strategy_name})"
                        )
                except Exception as e:
                    logger.error(f"Batch close error for position: {e}")
            
            if closed > 0:
                logger.info(f"Batch closed {closed} positions for strategy '{strategy_name}'")
        except Exception as e:
            logger.error(f"Batch close all error: {e}")
        
        return closed

    async def cancel_all_pending_orders(self) -> int:
        """取消所有活跃订单（紧急情况使用）- P21: 撤单频率保护
        
        Returns: 取消的订单数量
        """
        cancelled = 0
        pending_orders = list(self._active_orders.items())
        
        for ord_id, order_info in pending_orders:
            try:
                symbol = order_info.get("symbol", "")
                # P21: 撤单频率保护 - 紧急情况也遵守，但放宽阈值
                if not self._check_cancel_allowed(symbol):
                    logger.warning(f"P21: Emergency cancel blocked for {symbol} - rate limit")
                    break
                result = self.okx_client.cancel_order(symbol, ord_id)
                if result and not result.get("_failed", False):
                    self._record_cancel(symbol)
                    cancelled += 1
            except Exception as e:
                logger.error(f"Emergency cancel error for {ord_id}: {e}")
        
        if cancelled > 0:
            logger.warning(f"Emergency: cancelled {cancelled} pending orders")
        return cancelled

    def get_execution_quality_report(self) -> Dict[str, Any]:
        """获取执行质量综合报告"""
        stats = self.get_execution_stats()
        
        # 计算关键指标
        total = stats["total_orders"]
        dups = stats["duplicate_prevented"]
        retries = stats["retry_count"]
        partials = stats["partial_fills"]
        partial_resolved = stats["partial_fill_resolved"]
        
        report = {
            "summary": {
                "total_orders": total,
                "duplicate_prevented": dups,
                "duplicate_rate": round(dups / max(total, 1), 4),
                "retry_count": retries,
                "retry_rate": round(retries / max(total, 1), 4),
                "partial_fills": partials,
                "partial_resolve_rate": round(partial_resolved / max(partials, 1), 4),
                "idempotency_hits": stats["idempotency_hits"],
            },
            "active": {
                "active_orders": stats["active_orders"],
                "partial_fills_pending": stats["partial_fills_pending"],
                "clordid_cache_size": stats["clordid_cache_size"],
                "symbol_fail_states": stats["symbol_fail_states"],
            },
            "quality_score": self._calculate_execution_quality_score(stats),
            "timestamp": stats["timestamp"],
        }
        return report

    def _calculate_execution_quality_score(self, stats: Dict[str, Any]) -> float:
        """计算执行质量评分（0-1）"""
        score = 1.0
        
        total = max(stats["total_orders"], 1)
        
        # 重复率扣分（>5%扣0.1）
        dup_rate = stats["duplicate_prevented"] / total
        if dup_rate > 0.05:
            score -= 0.1
        
        # 重试率扣分（>10%扣0.15）
        retry_rate = stats["retry_count"] / total
        if retry_rate > 0.1:
            score -= 0.15
        
        # 部分成交未解决扣分
        pending = stats["partial_fills_pending"]
        if pending > 0:
            score -= 0.05 * min(pending, 5)
        
        return max(0.0, score)

    def _is_symbol_blocked(self, symbol: str) -> bool:
        """检查symbol是否在退避期内（连续失败后暂时停止下单）"""
        import time
        state = self._symbol_fail_state.get(symbol)
        if not state:
            return False
        now = time.time()
        # 退避期已过，清理状态
        if "blocked_until" in state and now >= state["blocked_until"]:
            self._symbol_fail_state.pop(symbol, None)
            return False
        if "blocked_until" in state:
            return True
        # 失败窗口外，重置计数
        if now - state.get("last_fail_time", 0) > self._fail_window:
            self._symbol_fail_state.pop(symbol, None)
            return False
        return False

    def _record_symbol_fail(self, symbol: str, reason: str = ""):
        """记录symbol下单失败，达到阈值后进入退避"""
        import time
        now = time.time()
        state = self._symbol_fail_state.get(symbol, {"fail_count": 0, "last_fail_time": 0})
        # 窗口外重置
        if now - state.get("last_fail_time", 0) > self._fail_window:
            state = {"fail_count": 0, "last_fail_time": 0}
        state["fail_count"] = state.get("fail_count", 0) + 1
        state["last_fail_time"] = now
        if state["fail_count"] >= self._max_fail_before_block:
            state["blocked_until"] = now + self._block_duration
            logger.warning(
                f"Symbol {symbol} blocked for {self._block_duration}s after "
                f"{state['fail_count']} consecutive failures (reason: {reason})"
            )
        self._symbol_fail_state[symbol] = state

    def _reset_symbol_fail(self, symbol: str):
        """symbol下单成功或持仓变化时重置失败计数"""
        self._symbol_fail_state.pop(symbol, None)

    def _check_margin_sufficient(self, symbol: str, quantity: float, price: float, leverage: int) -> bool:
        """预检保证金是否足够，避免直接发51008错误单"""
        try:
            if quantity <= 0 or price <= 0 or leverage <= 0:
                return False
            required_margin = quantity * price / leverage
            min_margin = self._get_effective_min_margin()
            if required_margin < min_margin:
                return False
            # 查询可用余额
            account_info = self.okx_client.get_account_info()
            if not account_info:
                logger.warning(f"Margin pre-check: account info unavailable for {symbol}, fail-closed (block order)")
                return False  # fail-closed：查询失败阻塞下单，避免保证金裸奔
            avail_bal = 0.0
            details = account_info.get("details", [])
            for detail in details:
                if detail.get("ccy") == "USDT":
                    avail_bal = float(detail.get("availBal", 0) or detail.get("availBal", "0"))
                    break
            if avail_bal <= 0:
                # 兜底用 totalEq - frozenBal
                total_eq = float(account_info.get("totalEq", 0))
                if total_eq > 0:
                    avail_bal = total_eq * 0.5  # 保守估计
            if avail_bal <= 0:
                return False
            # 可用余额必须大于所需保证金（留10%缓冲）
            return avail_bal >= required_margin * 1.1
        except Exception as e:
            logger.error(f"Margin pre-check exception for {symbol}: {e}, fail-closed (block order)")
            return False  # fail-closed：异常阻塞下单

    def _get_effective_min_margin(self) -> float:
        """获取有效的最小保证金：根据实际账户资金动态调整，小资金降低门槛"""
        base_min = self.config.get("trading", {}).get("min_margin_per_trade", 0.5)
        try:
            account_info = self.okx_client.get_account_info()
            if account_info:
                total_equity = 0.0
                details = account_info.get("details", [])
                for detail in details:
                    if detail.get("ccy") == "USDT":
                        total_equity = float(detail.get("eq", 0))
                        break
                if total_equity <= 0:
                    total_equity = float(account_info.get("totalEq", 0))
                if total_equity > 0:
                    if total_equity >= 1000:
                        return base_min
                    elif total_equity >= 500:
                        return max(0.5, base_min * 0.7)
                    elif total_equity >= 200:
                        return max(0.2, base_min * 0.4)
                    elif total_equity >= 100:
                        return max(0.1, base_min * 0.2)
                    else:
                        return max(0.05, base_min * 0.1)
        except Exception:
            pass
        return base_min

    async def _ensure_trading_balance(self, required_margin: float) -> bool:
        """确保交易账户有足够余额，不足时从资金账户划转"""
        try:
            # 首次下单前：尝试划转资金到交易账户
            # 注意：统一账户模式下不需要划转（资金池共享），但标准账户需要
            if not self._funds_transferred:
                self._funds_transferred = True
                account_info = self.okx_client.get_account_info()
                if account_info:
                    total_eq = float(account_info.get("totalEq", 0))
                    if total_eq > 0:
                        # 尝试划转，失败不阻塞（统一账户模式可能报58350）
                        transfer_amount = min(5.0, round(total_eq * 0.9, 2))
                        success = self.okx_client.transfer_to_trading("USDT", transfer_amount)
                        if success:
                            await asyncio.sleep(0.5)
                            logger.info(f"Funds transferred: {transfer_amount} USDT (total_eq={total_eq:.2f})")
                        else:
                            logger.debug(f"Fund transfer skipped (unified account or already in trading)")
                return True

            # 后续检查
            return True
        except Exception as e:
            logger.error(f"Error ensuring trading balance: {e}, fail-closed (block order)")
            return False  # fail-closed：确保余额异常时阻塞下单

    def _get_effective_profit_cost_ratio(self, strategy_name: str = "") -> float:
        """获取有效的收益成本比：小资金适当降低要求，网格策略专门降低"""
        base_ratio = self.config.get("trading", {}).get("min_profit_cost_ratio", 2.0)
        
        # 网格/马丁格尔策略：依赖大量微利交易，需要极低成本比门槛
        if strategy_name == "grid" or strategy_name == "spot_grid" or strategy_name == "spot_martingale":
            return max(0.2, base_ratio * 0.15)  # 网格/马丁格尔只需要覆盖手续费即可
        
        try:
            account_info = self.okx_client.get_account_info()
            if account_info:
                total_equity = 0.0
                details = account_info.get("details", [])
                for detail in details:
                    if detail.get("ccy") == "USDT":
                        total_equity = float(detail.get("eq", 0))
                        break
                if total_equity <= 0:
                    total_equity = float(account_info.get("totalEq", 0))
                if total_equity > 0:
                    if total_equity >= 1000:
                        return base_ratio
                    elif total_equity >= 200:
                        return max(1.2, base_ratio * 0.7)
                    else:
                        return max(1.0, base_ratio * 0.5)
        except Exception:
            pass
        return base_ratio
    
    async def start(self):
        asyncio.create_task(self._execution_loop())
        asyncio.create_task(self._reconciliation_loop())
        asyncio.create_task(self._order_tracking_loop())
        asyncio.create_task(self._dynamic_stop_loss_loop())
        await self._lifecycle_manager.start()
    
    async def execute_order(self, order_data: Dict[str, Any]):
        """公开下单接口 - 供调度器/策略直接调用
        
        将订单数据封装并加入执行队列，由 _execution_loop 异步处理。
        返回 True 表示已加入队列，False 表示失败。
        """
        try:
            if self._order_queue is None:
                logger.error("OrderQueue not initialized, cannot execute order")
                return False
            await self._order_queue.add_order(order_data)
            return True
        except Exception as e:
            logger.error(f"Failed to queue order: {e}")
            return False
    
    async def _execution_loop(self):
        while True:
            try:
                if self._order_queue is None:
                    await asyncio.sleep(0.5)
                    continue
                
                order_data = await self._order_queue.get_next_order()
                if order_data:
                    await self._execute_order(order_data)
                    await asyncio.sleep(0.05)  # 处理完订单后短暂休息
                else:
                    await asyncio.sleep(0.2)   # 队列空闲时降低轮询频率，减少CPU占用
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Loop error in _execution_loop: {e}")
                await asyncio.sleep(5)
    
    async def _execute_order(self, order_data: Dict[str, Any]):
        # P0: 全链路 traceID —— 用事件上下文包裹整笔订单执行，
        # 使该订单从队列出队到完成的所有日志共享同一 event_id，事后可串联复盘。
        # P2: 若上游（信号层独立风控裁决器）已回写 trace_id，则复用同源 traceID，实现信号→裁决→订单全链路串联。
        upstream_trace = order_data.get("trace_id") or None
        async with EventContext(f"execute_order {order_data.get('symbol', '')}", event_id=upstream_trace) as evt:
            order_data["trace_id"] = evt.id
            await self._execute_order_impl(order_data)

    async def _execute_order_impl(self, order_data: Dict[str, Any]):
        await self._token_bucket.acquire()

        symbol = order_data["symbol"]
        direction = order_data["direction"]
        price = order_data["price"]
        quantity = order_data["quantity"]
        leverage = order_data["leverage"]
        strategy_name = order_data["strategy_name"]
        signal_type = order_data["signal_type"]
        order_id = order_data.get("order_id", "")

        await self._lifecycle_manager.create_order(order_data)
        await self._lifecycle_manager.update_status(order_id, OrderStatus.VALIDATION, OrderPhase.VALIDATION)

        # 识别平仓信号：使用 reduce_only=True 避免误开反向仓
        # 同时检查 order_data 中显式传入的 reduce_only 标记（优先级最高）
        is_close_signal = order_data.get("reduce_only", False)
        if not is_close_signal:
            close_signal_types = ["stop_loss", "stop", "take_profit", "close", "reduce", "liquidation", "margin_call", "exit", "trailing", "tp"]
            sig_type_lower = str(signal_type).lower()
            is_close_signal = any(st in sig_type_lower for st in close_signal_types)

        # 平仓信号跳过退避检查（风控减仓必须执行）；开仓信号检查 symbol 是否在退避期
        if not is_close_signal and self._is_symbol_blocked(symbol):
            logger.debug(f"Symbol {symbol} in backoff period, skipping open order ({signal_type})")
            if order_id:
                await self._order_queue.update_order_status(order_id, "failed")
            await self._lifecycle_manager.update_status(order_id, OrderStatus.FAILED)
            return

        if order_id:
            await self._order_queue.increment_order_attempts(order_id)
        
        await self._lifecycle_manager.update_status(order_id, OrderStatus.EXECUTING, OrderPhase.PLACEMENT)

        # P8: 提前处理 'close' 方向 - 兼容策略直接传递的 direction='close'
        # 'close' 是上下文相关的方向，需要根据 explicit_pos_side 或现有持仓确定实际方向
        # 必须在所有 DirectionUnifier 调用之前处理
        explicit_pos_side = order_data.get("pos_side", "")
        if direction == "close":
            if explicit_pos_side and explicit_pos_side in ("long", "short"):
                # 从显式 pos_side 推导实际方向
                direction = explicit_pos_side
                logger.debug(f"Converted direction 'close' -> '{direction}' via explicit pos_side")
            else:
                # 从现有持仓推导方向
                positions = self.okx_client.get_positions()
                existing_pos = next((p for p in positions if p.get("instId", "") == symbol 
                                    and abs(float(p.get("pos", 0) or 0)) > 0), None)
                if existing_pos:
                    direction = existing_pos.get("posSide", "long")
                    logger.debug(f"Converted direction 'close' -> '{direction}' via existing position")
                else:
                    logger.error(f"Cannot resolve 'close' direction for {symbol}: no position found")
                    if order_id:
                        await self._order_queue.update_order_status(order_id, "failed")
                    await self._lifecycle_manager.update_status(order_id, OrderStatus.FAILED)
                    return

        # direction 可能是 long/short 或 buy/sell，转换为 pos_side（long/short）
        # P8: direction 此时已确保不是 'close'
        # 平仓信号无需设置杠杆（reduce-only 平仓不改变仓位保证金结构），
        # 且平仓 direction=buy/sell 映射 posSide 时会得到反向错误结果，跳过以消除无效 set-leverage 调用
        if not is_close_signal:
            pos_side_for_leverage = DirectionUnifier.to_pos_side(direction)
            await self._set_leverage(symbol, leverage, pos_side=pos_side_for_leverage)

        order_type = await self._determine_order_type(order_data)

        current_price = await self._get_current_price(symbol)

        # ── 滑点优化：使用 SlippageOptimizer 动态计算最优限价偏移 ──
        slippage_offset_applied = 0.0
        if self._slippage_optimizer and order_type == "limit":
            try:
                # 更新市场数据到滑点优化器
                try:
                    ticker = self.okx_client.get_ticker(symbol)
                    if ticker:
                        current_price_raw = float(ticker.get("last", current_price))
                        self._slippage_optimizer.update_market_data(
                            symbol, current_price_raw
                        )
                except Exception:
                    pass

                # 根据当前波动率确定紧急程度
                regime = self._slippage_optimizer.get_market_regime(symbol)
                if regime:
                    urgency_map = {"calm": 0.2, "normal": 0.4, "active": 0.6, "volatile": 0.85, "extreme": 1.0}
                    urgency = urgency_map.get(regime.value if hasattr(regime, 'value') else str(regime), 0.5)
                else:
                    urgency = 0.5
                
                side_for_optim = "buy" if pos_side_for_leverage == "long" else "sell"
                opt_price, opt_info = self._slippage_optimizer.calculate_optimal_price(
                    symbol, side_for_optim, current_price, order_type,
                    strategy=self._slippage_optimizer._default_strategy,
                    urgency=urgency
                )
                if opt_info.get("applied"):
                    slippage_offset_applied = opt_info.get("offset_ratio", 0.0)
                    logger.debug(
                        f"SlippageOptimizer: {symbol} {side_for_optim} "
                        f"offset={slippage_offset_applied:.4%} regime={opt_info.get('regime')} "
                        f"ref={current_price:.4f} -> opt={opt_price:.4f}"
                    )
                    price = opt_price  # 使用优化后的价格
            except Exception as e:
                logger.debug(f"SlippageOptimizer skipped for {symbol}: {e}")

        slippage = abs(current_price - price) / price if price > 0 else 0
        if slippage > self._slippage_tolerance:
            logger.warning(f"Slippage {slippage:.2%} exceeds tolerance {self._slippage_tolerance:.2%} for {symbol}")
            if "stop" not in signal_type.lower() and "loss" not in signal_type.lower():
                logger.info(f"Updating order price from {price:.4f} to {current_price:.4f}")
                price = current_price

        # 等位挂单：post_only 单必须挂在盘口 maker 侧（买一/卖一），否则会被 OKX 拒绝或立即吃单。
        # 开多挂买一(bid)、开空挂卖一(ask)，保证只做 maker，等待对手盘成交。
        if order_type == "post_only":
            try:
                ticker = self.okx_client.get_ticker(symbol)
                if ticker:
                    bid = float(ticker.get("bidPx", 0) or 0)
                    ask = float(ticker.get("askPx", 0) or 0)
                    if bid > 0 and ask > 0:
                        d = str(direction).lower()
                        if d in ("long", "buy"):
                            price = bid
                        elif d in ("short", "sell"):
                            price = ask
                        logger.debug(f"post_only maker-side price: {symbol} {direction} -> {price}")
            except Exception as e:
                logger.debug(f"post_only price adjustment skipped for {symbol}: {e}")

        # 按合约lot size向下取整，避免51121错误
        original_qty = quantity
        # 账本一致性（修复4）：round_quantity_to_lot 期望张数，quantity 为币数；仅用「币数→张数」取整结果
        # 做最小手数判断，quantity 本身保持币数（place_order 内部再统一转换，且后续 notional 计算需币数）
        contracts_check = self.okx_client.round_quantity_to_lot(
            symbol, self.okx_client.coin_to_contracts(symbol, quantity), round_up=is_close_signal
        )
        if contracts_check <= 0:
            logger.warning(f"Quantity {original_qty} too small for {symbol} lot size, skipping order")
            if order_id:
                await self._order_queue.update_order_status(order_id, "failed")
            return

        # P0: 最低名义价值检查 - 拒绝微利交易（开仓），确保每笔交易有实际意义
        # 手续费来回0.1%，盈利必须 > 手续费*3，且绝对收益要够覆盖成本
        if not is_close_signal:
            notional = quantity * price
            min_notional = self._get_min_notional(symbol, price, strategy_name)
            if notional < min_notional:
                logger.warning(
                    f"Rejected micro-trade: {symbol} notional={notional:.2f} < min={min_notional:.2f} "
                    f"(strategy={strategy_name}, signal={signal_type})"
                )
                if order_id:
                    await self._order_queue.update_order_status(order_id, "failed")
                return

        # 兼容策略传递的 direction 格式：可能是 long/short 或 buy/sell
        # 优先使用 order_data 中显式传入的 pos_side（如 _handle_take_profit_action 传入）
        explicit_pos_side = order_data.get("pos_side", "")
        if explicit_pos_side and explicit_pos_side in ("long", "short"):
            pos_side = explicit_pos_side
            if is_close_signal:
                # 平仓：订单方向 side 是持仓方向 pos_side 的反向（平多=sell，平空=buy）
                side = "sell" if pos_side == "long" else "buy"
            else:
                side = DirectionUnifier.to_side(pos_side)
        elif direction in ["long", "short"]:
            side = DirectionUnifier.to_side(direction)
            pos_side = direction
        else:
            # 策略直接传递 buy/sell
            side = direction
            pos_side = DirectionUnifier.to_pos_side(direction)

        # 平仓信号的 pos_side 应该等于被平仓持仓的方向
        # side=sell -> 平多 -> pos_side=long
        # side=buy -> 平空 -> pos_side=short
        if is_close_signal:
            pos_side = "long" if side == "sell" else "short"

        # 开仓前保证金预检，避免直接发51008错误单（平仓信号不需要保证金）
        if not is_close_signal and not self._check_margin_sufficient(symbol, quantity, price, leverage):
            logger.warning(f"Insufficient margin for {symbol} open order, skipping (pre-check)")
            self._record_symbol_fail(symbol, "margin_pre_check_failed")
            if order_id:
                await self._order_queue.update_order_status(order_id, "failed")
            return

        # P2: 开仓前动态波动过滤 - 替代固定时段禁交易
        # 当24h振幅超过阈值时禁止开仓，避免高波动磨损
        if not is_close_signal and self._no_trade_volatility_threshold > 0:
            try:
                ticker = await self.okx_client.get_ticker_async(symbol)
                if ticker:
                    high_24h = float(ticker.get("high24h", 0) or 0)
                    low_24h = float(ticker.get("low24h", 0) or 0)
                    if low_24h > 0:
                        volatility = (high_24h - low_24h) / low_24h
                        if volatility > self._no_trade_volatility_threshold:
                            logger.info(
                                f"Order blocked: {symbol} 24h volatility {volatility:.1%} "
                                f"> threshold {self._no_trade_volatility_threshold:.0%}, "
                                f"skipping {strategy_name} {signal_type}"
                            )
                            if order_id:
                                await self._order_queue.update_order_status(order_id, "failed")
                            return
            except Exception as e:
                logger.debug(f"Volatility check failed for {symbol}, allowing trade: {e}")

        # 开仓前重复持仓检查：同 symbol + 同方向只允许 1 笔 open 持仓
        # 避免策略连续同向信号导致保证金累积浪费（历史数据显示 ARB short 6 笔重复，浪费 5.5 USDT）
        # 网格策略豁免：网格策略的多层双向开仓是其设计核心，不应拦截
        if not is_close_signal and "grid" not in strategy_name.lower():
            existing_open = self.sqlite_storage.get_all_open_records(symbol)
            duplicate = False
            opposite_conflict = False
            for rec in existing_open:
                rec_side = str(rec.get("side", "")).lower()
                # 归一化方向：long/buy -> long，short/sell -> short
                rec_dir = DirectionUnifier.normalize(rec_side) if rec_side in ("long", "short", "buy", "sell") else ""
                if rec_dir and rec_dir == pos_side:
                    duplicate = True
                    break
                # P0: 反向持仓冲突检查 - 同symbol已有反向开仓记录时拦截
                if rec_dir and rec_dir != pos_side and rec_dir in ("long", "short"):
                    opposite_conflict = True
            if duplicate:
                logger.info(
                    f"Duplicate open position blocked: {symbol} {pos_side} already has open record, "
                    f"skipping new {strategy_name} {signal_type} order"
                )
                if order_id:
                    await self._order_queue.update_order_status(order_id, "failed")
                return
            # P0: 反向持仓冲突——已有相反方向持仓时禁止开仓（防止双向对冲浪费保证金）
            if opposite_conflict:
                opp_dir = DirectionUnifier.opposite(pos_side)
                logger.warning(
                    f"Opposite position conflict blocked: {symbol} already has {opp_dir} position, "
                    f"rejecting new {pos_side} {strategy_name} {signal_type} order"
                )
                if order_id:
                    await self._order_queue.update_order_status(order_id, "failed")
                return

        # 开仓前最大并发持仓数检查：动态调整+优先级驱逐
        if not is_close_signal:
            current_positions = self.okx_client.get_positions()
            active_count = sum(1 for p in current_positions
                             if abs(float(p.get("pos", 0) or p.get("position", 0) or 0)) > 0)
            
            # P7: 动态调整最大并发持仓数（基于账户权益）
            dynamic_max = self._get_dynamic_max_positions()
            
            if active_count >= dynamic_max:
                # P4-2: 智能优先级驱逐 - 降低置信度阈值，更积极驱逐低质量持仓
                signal_priority = order_data.get("priority", 2) if isinstance(order_data, dict) else 2
                signal_confidence = order_data.get("confidence", 0.5) if isinstance(order_data, dict) else 0.5
                
                if signal_priority <= 1 and signal_confidence >= 0.70:
                    # 高优先级+高置信度：允许超限开仓
                    logger.info(
                        f"Priority override: allowing {symbol} {signal_type} despite "
                        f"full positions ({active_count}/{dynamic_max}), "
                        f"priority={signal_priority}, confidence={signal_confidence:.2f}"
                    )
                elif signal_confidence >= 0.50:
                    # P4-2: 降低置信度阈值从0.6到0.5，更积极驱逐
                    evicted = await self._try_evict_low_quality_position(
                        current_positions, symbol, strategy_name, signal_confidence
                    )
                    if evicted:
                        logger.info(
                            f"P4-2: Position evicted to make room for {symbol} "
                            f"({active_count}/{dynamic_max}), confidence={signal_confidence:.2f}"
                        )
                    else:
                        # P4-2: 驱逐失败时，尝试强制驱逐最差持仓
                        force_evicted = await self._try_force_evict(
                            current_positions, symbol, strategy_name
                        )
                        if force_evicted:
                            logger.info(
                                f"P4-2: Force evicted to make room for {symbol} "
                                f"({active_count}/{dynamic_max})"
                            )
                        else:
                            logger.warning(
                                f"Max concurrent positions reached ({active_count}/{dynamic_max}), "
                                f"rejecting {symbol} {signal_type} from {strategy_name}"
                            )
                            if order_id:
                                await self._order_queue.update_order_status(order_id, "failed")
                            return
                else:
                    logger.warning(
                        f"Max concurrent positions reached ({active_count}/{dynamic_max}), "
                        f"rejecting {symbol} {signal_type} from {strategy_name}"
                    )
                    if order_id:
                        await self._order_queue.update_order_status(order_id, "failed")
                    return

        # ============ 风控前置串行闸门（下单前最后一道校验，不可绕过）============
        # 所有下单动作（含止损/减仓/平仓/条件单）在调用 place_order 前必须过 RiskGate
        # 平仓/减仓信号走 is_close=True 跳过 L1 保证金检查，但保留 L4/L5 全局风控
        if self._risk_adjudicator is not None or self._risk_gate is not None:
            try:
                risk_signal = {
                    "symbol": symbol,
                    "strategy_name": strategy_name,
                    "signal_type": signal_type,
                    "direction": direction,
                    "price": price,
                    "quantity": quantity,
                    "leverage": leverage,
                    "trace_id": order_data.get("trace_id", ""),  # P2: 全链路 traceID 串联
                }
                # P2: 优先独立风控裁决器（traceID 贯穿 + 裁决事件溯源），未注入回退 RiskGate
                if self._risk_adjudicator is not None:
                    risk_result = self._risk_adjudicator.adjudicate(risk_signal, is_close=is_close_signal)
                else:
                    risk_result = self._risk_gate.validate(risk_signal, is_close=is_close_signal)
                if not risk_result.passed:
                    # 风控拦截：直接丢弃，不下单，记录拦截原因
                    logger.warning(
                        f"⛔ OrderExecutor risk gate BLOCKED | "
                        f"layer={risk_result.blocked_layer.value if risk_result.blocked_layer else 'unknown'} | "
                        f"strategy={strategy_name} | symbol={symbol} | "
                        f"signal_type={signal_type} | is_close={is_close_signal} | "
                        f"reason={risk_result.summary}"
                    )
                    self._publish_event(EventType.ORDER_REJECTED, {
                        "trace_id": order_data.get("trace_id", ""),
                        "symbol": symbol,
                        "strategy": strategy_name,
                        "signal_type": signal_type,
                        "direction": direction,
                        "layer": "risk_gate",
                        "reason_code": risk_result.blocked_layer.value if risk_result.blocked_layer else "unknown",
                        "reason": risk_result.summary,
                        "is_close": is_close_signal,
                    })
                    if order_id:
                        await self._order_queue.update_order_status(order_id, "rejected_by_risk_gate")
                    await self._lifecycle_manager.update_status(order_id, OrderStatus.FAILED)
                    return
            except Exception as e:
                logger.error(f"RiskGate validate error in _execute_order: {e}")
                # 风控异常时保守拒绝（安全第一）
                if order_id:
                    await self._order_queue.update_order_status(order_id, "rejected_by_risk_gate")
                await self._lifecycle_manager.update_status(order_id, OrderStatus.FAILED)
                return

        # P0: 开仓前成本分析 - 杜绝磨损型交易
        cost_breakdown = None  # P0修复: 提前声明，后续TradeAuditor复用
        if not is_close_signal and hasattr(self, '_trade_cost_analyzer') and self._trade_cost_analyzer:
            try:
                should_open, cost_reason, cost_breakdown = self._trade_cost_analyzer.should_open_position(
                    symbol=symbol, side=side, price=price, quantity=quantity,
                    leverage=leverage, pos_side=pos_side,
                    expected_profit_pct=order_data.get("expected_profit_pct", 0.01),
                )
                if not should_open:
                    logger.warning(f"TradeCostAnalyzer BLOCKED {symbol}: {cost_reason}")
                    self._publish_event(EventType.ORDER_REJECTED, {
                        "trace_id": order_data.get("trace_id", ""),
                        "symbol": symbol,
                        "strategy": strategy_name,
                        "signal_type": signal_type,
                        "direction": direction,
                        "layer": "trade_cost",
                        "reason_code": "cost_blocked",
                        "reason": cost_reason,
                        "is_close": is_close_signal,
                    })
                    if order_id:
                        await self._order_queue.update_order_status(order_id, "rejected_by_cost")
                    await self._lifecycle_manager.update_status(order_id, OrderStatus.FAILED)
                    return
                logger.debug(f"TradeCostAnalyzer PASSED {symbol}: cost={cost_breakdown.total_cost_pct:.4%}, "
                           f"breakeven={cost_breakdown.breakeven_price_long or cost_breakdown.breakeven_price_short:.4f}")
            except Exception as e:
                # P1-1: fail-closed —— 成本分析器异常时保守拒绝，杜绝磨损型交易逃逸
                logger.error(f"TradeCostAnalyzer error for {symbol}: {e}")
                self._publish_event(EventType.ORDER_REJECTED, {
                    "trace_id": order_data.get("trace_id", ""),
                    "symbol": symbol,
                    "strategy": strategy_name,
                    "signal_type": signal_type,
                    "direction": direction,
                    "layer": "trade_cost",
                    "reason_code": "cost_analyzer_error",
                    "reason": f"TradeCostAnalyzer exception: {e}",
                    "is_close": is_close_signal,
                })
                if order_id:
                    await self._order_queue.update_order_status(order_id, "rejected_by_cost")
                await self._lifecycle_manager.update_status(order_id, OrderStatus.FAILED)
                return

        # P0: 开仓前智能体审核 - 多维度判断
        if not is_close_signal and hasattr(self, '_intelligent_agent') and self._intelligent_agent:
            try:
                agent_decision = self._intelligent_agent.audit_signal(
                    symbol=symbol, strategy_name=strategy_name, signal_type=signal_type,
                    direction=direction, price=price, quantity=quantity, confidence=order_data.get("confidence", 0.5),
                )
                if agent_decision.action == "reject":
                    logger.warning(f"IntelligentAgent REJECTED {symbol}: {agent_decision.reason}")
                    self._publish_event(EventType.ORDER_REJECTED, {
                        "trace_id": order_data.get("trace_id", ""),
                        "symbol": symbol,
                        "strategy": strategy_name,
                        "signal_type": signal_type,
                        "direction": direction,
                        "layer": "intelligent_agent",
                        "reason_code": "agent_rejected",
                        "reason": agent_decision.reason,
                        "is_close": is_close_signal,
                    })
                    if order_id:
                        await self._order_queue.update_order_status(order_id, "rejected_by_agent")
                    await self._lifecycle_manager.update_status(order_id, OrderStatus.FAILED)
                    return
                elif agent_decision.action == "reduce":
                    new_qty = agent_decision.details.get("reduced_quantity", quantity * 0.5)
                    logger.info(f"IntelligentAgent REDUCED {symbol}: {quantity:.4f} -> {new_qty:.4f} ({agent_decision.reason})")
                    quantity = new_qty
                elif agent_decision.action == "delay":
                    logger.info(f"IntelligentAgent DELAYED {symbol}: {agent_decision.reason}")
                    if order_id:
                        await self._order_queue.update_order_status(order_id, "delayed_by_agent")
                    await self._lifecycle_manager.update_status(order_id, OrderStatus.FAILED)
                    return
            except Exception as e:
                # P1-1: fail-closed —— 智能体审核异常时保守拒绝，避免异常逃逸导致无审核开仓
                logger.error(f"IntelligentAgent error for {symbol}: {e}")
                self._publish_event(EventType.ORDER_REJECTED, {
                    "trace_id": order_data.get("trace_id", ""),
                    "symbol": symbol,
                    "strategy": strategy_name,
                    "signal_type": signal_type,
                    "direction": direction,
                    "layer": "intelligent_agent",
                    "reason_code": "agent_error",
                    "reason": f"IntelligentAgent exception: {e}",
                    "is_close": is_close_signal,
                })
                if order_id:
                    await self._order_queue.update_order_status(order_id, "rejected_by_agent")
                await self._lifecycle_manager.update_status(order_id, OrderStatus.FAILED)
                return

        # P0: 开仓前交易审核 - 全链路成本/风险/信号质量审核
        if not is_close_signal and hasattr(self, '_trade_auditor') and self._trade_auditor:
            try:
                # P0修复: 复用上方TradeCostAnalyzer已缓存的结果，避免重复调用
                cost_analysis = cost_breakdown
                
                # 获取市场状态
                market_regime = None
                if hasattr(self, '_intelligent_agent') and self._intelligent_agent:
                    try:
                        regime = self._intelligent_agent.get_market_regime(symbol)
                        market_regime = regime.value if hasattr(regime, 'value') else str(regime)
                    except Exception:
                        pass
                
                audit_result, audit_reason, audit_details = self._trade_auditor.pre_open_audit(
                    symbol=symbol,
                    strategy_name=strategy_name,
                    signal_type=signal_type,
                    direction=direction,
                    price=price,
                    quantity=quantity,
                    leverage=leverage,
                    confidence=order_data.get("confidence", 0.5),
                    cost_analysis=cost_analysis,
                    market_regime=market_regime,
                )
                
                if audit_result.value == "block":
                    logger.warning(f"TradeAuditor BLOCKED {symbol} [{strategy_name}]: {audit_reason}")
                    self._publish_event(EventType.ORDER_REJECTED, {
                        "trace_id": order_data.get("trace_id", ""),
                        "symbol": symbol,
                        "strategy": strategy_name,
                        "signal_type": signal_type,
                        "direction": direction,
                        "layer": "trade_auditor",
                        "reason_code": "audit_blocked",
                        "reason": audit_reason,
                        "is_close": is_close_signal,
                    })
                    if order_id:
                        await self._order_queue.update_order_status(order_id, "rejected_by_auditor")
                    await self._lifecycle_manager.update_status(order_id, OrderStatus.FAILED)
                    return
                elif audit_result.value == "reduce":
                    new_qty = quantity * 0.5
                    logger.info(f"TradeAuditor REDUCED {symbol}: {quantity:.4f} -> {new_qty:.4f} ({audit_reason})")
                    quantity = new_qty
                elif audit_result.value == "warn":
                    logger.warning(f"TradeAuditor WARN {symbol}: {audit_reason}")
            except Exception as e:
                # P1-1: fail-closed —— 交易审计异常时保守拒绝，避免无审计开仓
                logger.error(f"TradeAuditor error for {symbol}: {e}")
                self._publish_event(EventType.ORDER_REJECTED, {
                    "trace_id": order_data.get("trace_id", ""),
                    "symbol": symbol,
                    "strategy": strategy_name,
                    "signal_type": signal_type,
                    "direction": direction,
                    "layer": "trade_auditor",
                    "reason_code": "auditor_error",
                    "reason": f"TradeAuditor exception: {e}",
                    "is_close": is_close_signal,
                })
                if order_id:
                    await self._order_queue.update_order_status(order_id, "rejected_by_auditor")
                await self._lifecycle_manager.update_status(order_id, OrderStatus.FAILED)
                return

        # 开仓前确保交易账户余额充足（从资金账户划转）
        if not is_close_signal:
            required_margin = quantity * price / max(leverage, 1)
            if not await self._ensure_trading_balance(required_margin):
                logger.error(f"Failed to ensure trading balance for {symbol}, required_margin={required_margin:.2f}")
                if order_id:
                    await self._order_queue.update_order_status(order_id, "failed")
                await self._lifecycle_manager.update_status(order_id, OrderStatus.FAILED)
                return

        # 诊断：下单前打印账户余额
        try:
            acct = self.okx_client.get_account_info()
            if acct:
                usdt_detail = None
                for d in acct.get("details", []):
                    if d.get("ccy") == "USDT":
                        usdt_detail = d
                        break
                logger.debug(
                    f"[ACCT_DIAG] totalEq={acct.get('totalEq')}, mgnRatio={acct.get('mgnRatio')}, "
                    f"acctLv={acct.get('acctLv')}, "
                    f"USDT: eq={usdt_detail.get('eq') if usdt_detail else 'N/A'}, "
                    f"availBal={usdt_detail.get('availBal') if usdt_detail else 'N/A'}, "
                    f"cashBal={usdt_detail.get('cashBal') if usdt_detail else 'N/A'}, "
                    f"frozenBal={usdt_detail.get('frozenBal') if usdt_detail else 'N/A'}"
                )
        except Exception:
            pass

        # ── 幂等键：生成唯一 clOrdId 防止重复下单 ──
        clordid = ""
        if self._idempotency_enabled:
            clordid = order_data.get("clOrdId", "")
            if not clordid:
                clordid = self._generate_idempotency_key(
                    symbol, strategy_name, signal_type
                )
                order_data["clOrdId"] = clordid

            # 检查幂等键是否已存在
            existing_ord_id = self._check_idempotency(clordid)
            if existing_ord_id:
                self._execution_stats["duplicate_prevented"] += 1
                logger.info(
                    f"Duplicate order prevented: {clordid} -> {existing_ord_id}"
                )
                if order_id:
                    await self._order_queue.update_order_status(order_id, "executed")
                await self._lifecycle_manager.update_status(order_id, OrderStatus.PENDING)
                return

        self._execution_stats["total_orders"] += 1

        # P10: 平仓前验证交易所实际持仓 - 防止51169幽灵平仓
        # 如果策略认为有持仓但交易所实际没有，直接跳过，避免无效的平仓请求
        if is_close_signal:
            exchange_positions = self.okx_client.get_positions()
            target_pos = None
            for p in exchange_positions:
                if p.get("instId", "") == symbol:
                    pos_qty = abs(float(p.get("pos", 0) or 0))
                    if pos_qty > 0:
                        target_pos = p
                        break
            if target_pos is None:
                logger.warning(
                    f"P10 ghost position skip: {symbol} close signal from {strategy_name} "
                    f"but no exchange position found, skipping order"
                )
                self._execution_stats["ghost_close_skipped"] = self._execution_stats.get("ghost_close_skipped", 0) + 1
                # 清理本地状态中可能残留的幽灵仓位记录
                try:
                    self.sqlite_storage.delete_open_record(symbol, pos_side)
                except Exception:
                    pass
                if order_id:
                    await self._order_queue.update_order_status(order_id, "failed")
                await self._lifecycle_manager.update_status(order_id, OrderStatus.FAILED)
                return

            # P10: 验证平仓方向与持仓方向一致
            exchange_pos_side = target_pos.get("posSide", "")
            if exchange_pos_side and exchange_pos_side != pos_side:
                logger.warning(
                    f"P10 pos_side mismatch: {symbol} close signal pos_side={pos_side} "
                    f"but exchange pos_side={exchange_pos_side}, correcting"
                )
                pos_side = exchange_pos_side
                side = "sell" if pos_side == "long" else "buy"

        # 等位挂单分档（文档 5.3）：post_only + pending_price>0 时，将仓位拆分为多档 maker 挂单
        # 在支撑/压力位附近逐档加深等回踩/反弹，避免单笔全量追行情。
        pending_price = float(order_data.get("pending_price", 0.0) or 0.0)
        if order_type == "post_only" and pending_price > 0 and not is_close_signal:
            await self._place_pending_tier_orders(
                order_data=order_data,
                symbol=symbol,
                side=side,
                pos_side=pos_side,
                quantity=quantity,
                leverage=leverage,
                order_type=order_type,
                strategy_name=strategy_name,
                signal_type=signal_type,
                order_id=order_id,
                pending_price=pending_price,
                clordid=clordid,
            )
            return

        try:
            result = self.okx_client.place_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                leverage=leverage,
                pos_side=pos_side,
                reduce_only=is_close_signal,
                clOrdId=clordid if self._idempotency_enabled else "",
            )

            # place_order 现在失败时返回 {"_failed": True, "sCode": "..."}，成功返回带 ordId 的 dict
            is_failed = isinstance(result, dict) and result.get("_failed", False)
            exchange_order_id = result.get("ordId", "") if isinstance(result, dict) and not is_failed else ""

            if not is_failed and exchange_order_id:
                logger.info(f"Order executed: {exchange_order_id}, {direction} {symbol} @ {price:.4f}")
                self._reset_symbol_fail(symbol)  # 成功后重置失败计数

                # P1 埋点：下单成功 → 发布 ORDER_PLACED 事件（事件溯源落盘）
                self._publish_event(EventType.ORDER_PLACED, {
                    "symbol": symbol,
                    "exchange_order_id": exchange_order_id,
                    "direction": direction,
                    "pos_side": pos_side,
                    "side": side,
                    "order_type": order_type,
                    "signal_type": signal_type,
                    "strategy_name": strategy_name,
                    "quantity": quantity,
                    "price": price,
                    "leverage": leverage,
                    "reduce_only": is_close_signal,
                    "trace_id": order_data.get("trace_id", ""),
                })

                # 记录幂等键
                if self._idempotency_enabled and clordid:
                    self._record_idempotency_key(clordid, exchange_order_id)

                self._active_orders[exchange_order_id] = {
                    **order_data,
                    "status": "pending",
                    "exchange_order_id": exchange_order_id,
                    "filled_price": 0,
                    "create_time": datetime.now(),
                    "slippage_offset_applied": slippage_offset_applied,  # 滑点优化记录
                    "clOrdId": clordid,  # 幂等键
                    "reduce_only": is_close_signal,  # P1 埋点修正：持久化正确平仓标记，供成交埋点发布平仓事件
                }

                # 方案A：活跃订单落盘（fill 回执丢失根因修复）
                self._persist_active_order(exchange_order_id, self._active_orders[exchange_order_id])

                await self._lifecycle_manager.set_exchange_order_id(order_id, exchange_order_id)
                await self._lifecycle_manager.update_status(order_id, OrderStatus.PENDING, OrderPhase.TRACKING)

                if self._account_manager:
                    margin_used = quantity * price / leverage
                    self._account_manager.notify_order_placed(strategy_name, margin_used)

                # 通过路径：记录交易到五层风控（L4单日开仓次数跟踪）
                if self._risk_gate is not None:
                    try:
                        self._risk_gate.record_trade()
                    except Exception as _rg_err:
                        logger.debug(f"RiskGate record_trade error: {_rg_err}")

                if order_id:
                    await self._order_queue.update_order_status(order_id, "executed")

                await self._place_conditional_orders(order_data, exchange_order_id)

                if not is_close_signal:
                    # P-复盘修复：开仓记录补记预估手续费（taker 单边），成交价从下单回报尽力回填，
                    # 真实成交价/手续费后续由 _track_fills 校正（避免历史数据 fees 全 0 污染）。
                    taker_fee_rate = self.config.get("trading", {}).get("taker_fee_rate", 0.0005)
                    est_open_fee = quantity * price * taker_fee_rate
                    fill_px = 0.0
                    if isinstance(result, dict):
                        try:
                            fill_px = float(result.get("avgPx") or result.get("fillPx") or 0.0)
                        except (TypeError, ValueError):
                            fill_px = 0.0
                    self.sqlite_storage.save_trade_record({
                        "id": exchange_order_id,
                        "symbol": symbol,
                        "strategy_name": strategy_name,
                        "side": direction,
                        "order_type": order_type,
                        "signal_type": signal_type,
                        "quantity": quantity,
                        "price": price,
                        "filled_price": fill_px if fill_px > 0 else 0,
                        "leverage": leverage,
                        "margin": quantity * price / leverage,
                        "pnl": 0,
                        "pnl_percent": 0,
                        "fees": est_open_fee,
                        "status": "open",
                        "create_time": datetime.now(),
                        "trace_id": order_data.get("trace_id", "")
                    })
                    
                    # P24: 开仓时不记录到智能交易体，平仓时根据实际PnL记录
                    # 之前的 is_win=False 导致每次开仓被计为亏损，触发连续亏损暂停
            else:
                # place_order 返回 None 的唯一场景：数量低于 lot size（round_quantity_to_lot 返回 <=0）
                # 该情况不可重试（重试数量不变），直接标记失败避免反复无效重试
                if result is None:
                    logger.warning(
                        f"Order rejected: {direction} {symbol} quantity below minimum lot size, "
                        f"skipping retry (insufficient position sizing)"
                    )
                    self._record_symbol_fail(symbol, "quantity_below_lot")
                    await self._lifecycle_manager.record_error(
                        order_id, "QUANTITY_BELOW_LOT", "quantity below minimum lot size"
                    )
                    await self._lifecycle_manager.update_status(order_id, OrderStatus.FAILED)
                    if order_id:
                        await self._order_queue.update_order_status(order_id, "failed")
                    return

                # 解析错误码（place_order 现在返回包含 sCode 的 dict）
                error_code = ""
                error_msg = ""
                if isinstance(result, dict):
                    error_code = result.get("sCode", "") or result.get("code", "")
                    error_msg = result.get("sMsg", "")

                logger.error(f"Order execution failed: {direction} {symbol} (sCode={error_code}, msg={error_msg})")

                # ── 增强错误分类：区分可重试/不可重试 ──
                retryable = self._classify_error(error_code, error_msg)

                # 平仓订单失败时，检查是否因为无持仓（51169），自动清理策略状态
                if is_close_signal:
                    # 51169明确表示交易所无对应持仓，无论本地验证结果如何，必须清理
                    if error_code == "51169":
                        logger.warning(f"Close signal 51169 for {symbol} pos_side={pos_side}: exchange has no position, cleaning up strategy state")
                        await self._cleanup_orphaned_position(symbol, strategy_name)
                        # 同时清除该symbol的防重复触发标记
                        self._stop_pending.discard(f"{symbol}:long")
                        self._stop_pending.discard(f"{symbol}:short")
                        await self._lifecycle_manager.record_error(order_id, error_code, error_msg)
                        await self._lifecycle_manager.update_status(order_id, OrderStatus.FAILED)
                        if order_id:
                            await self._order_queue.update_order_status(order_id, "failed")
                        return
                    # 其他平仓错误：检查持仓是否存在
                    has_position = await self._verify_position_exists(symbol, direction, pos_side)
                    if not has_position:
                        logger.warning(f"No position found for {symbol} pos_side={pos_side} dir={direction}, cleaning up strategy state")
                        await self._cleanup_orphaned_position(symbol, strategy_name)
                        if order_id:
                            await self._order_queue.update_order_status(order_id, "failed")
                        return  # 不重试无效平仓订单

                # 资金不足（51008）/ 数量不符合lot size（51121）/ 仓位限制（51100+）/ 无效平仓（51169）等不可重试错误
                non_retryable_codes = {"51008", "51121", "51100", "51101", "51102", "51103", "51104", "51105", "51106", "51169"}
                if error_code in non_retryable_codes:
                    logger.warning(f"Non-retryable error {error_code} for {symbol}, skipping retry")
                    self._record_symbol_fail(symbol, f"error_{error_code}")
                    await self._lifecycle_manager.record_error(order_id, error_code, error_msg)
                    await self._lifecycle_manager.update_status(order_id, OrderStatus.FAILED)
                    if order_id:
                        await self._order_queue.update_order_status(order_id, "failed")
                    return

                # ── 增强：可重试错误使用指数退避 ──
                if retryable:
                    self._execution_stats["retry_count"] += 1
                    self._record_symbol_fail(symbol, f"retryable_{error_code}")
                    await self._lifecycle_manager.record_error(order_id, error_code, error_msg)
                    if order_id:
                        await self._order_queue.update_order_status(order_id, "failed")
                    attempt = await self._order_queue.get_order_attempts(order_id)
                    await self._retry_order(order_data, attempt + 1, retryable)
                    return

                # 其他错误：记录失败但允许重试（使用默认退避）
                self._record_symbol_fail(symbol, f"error_{error_code}")
                await self._lifecycle_manager.record_error(order_id, error_code, error_msg)
                if order_id:
                    await self._order_queue.update_order_status(order_id, "failed")
                attempt = await self._order_queue.get_order_attempts(order_id)
                await self._retry_order(order_data, attempt + 1)

        except Exception as e:
            logger.error(f"Error executing order: {e}")
            self._record_symbol_fail(symbol, "exception")
            await self._lifecycle_manager.record_error(order_id, "EXCEPTION", str(e))
            await self._lifecycle_manager.update_status(order_id, OrderStatus.FAILED)
            if order_id:
                await self._order_queue.update_order_status(order_id, "failed")
            attempt = await self._order_queue.get_order_attempts(order_id)
            err_type = self._classify_error("", str(e))
            await self._retry_order(order_data, attempt + 1, err_type)
    
    async def _get_current_price(self, symbol: str) -> float:
        try:
            ticker = self.okx_client.get_ticker(symbol)
            if ticker:
                return float(ticker["last"])
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
        return 0.0

    async def _verify_position_exists(self, symbol: str, direction: str, pos_side: str = "") -> bool:
        """验证OKX是否真的有对应方向的持仓
        Args:
            symbol: 交易对
            direction: 订单方向 (long/short/buy/sell)
            pos_side: 持仓方向（平仓订单时必传，用于精确定位目标仓位）
        """
        try:
            positions = self.okx_client.get_positions()
            # 优先使用 pos_side 精确匹配（平仓订单）
            if pos_side:
                target_side = DirectionUnifier.to_pos_side(pos_side)
            else:
                target_side = DirectionUnifier.to_pos_side(direction)
            for pos_data in positions:
                pos = self.okx_client._parse_position(pos_data)
                if pos and pos.symbol == symbol and pos.side == target_side and abs(pos.quantity) > 0:
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to verify position for {symbol}: {e}")
            return False  # 查询失败时保守返回False，不冒51169风险（与scalping strategy保持一致）

    def _get_dynamic_max_positions(self) -> int:
        """P7: 基于账户权益动态调整最大并发持仓数
        
        权益越低，允许的并发持仓越少，降低风险敞口：
        - equity >= 500: 基础值
        - equity >= 200: 基础值 * 0.8
        - equity >= 100: 基础值 * 0.6
        - equity < 100:  max(8, 基础值 * 0.55)  # P14: 小账户进一步提升至8
        
        P9: 动态上限不低于当前实际持仓数，防止死锁（已有持仓>上限时无法开新仓）
        """
        base = self._max_concurrent_positions_base
        try:
            account = self.okx_client.get_account_info()
            if account:
                equity = float(account.get("totalEq", 0) or account.get("eq", 0))
                if equity <= 0:
                    return base
                if equity >= 500:
                    dynamic = base
                elif equity >= 200:
                    dynamic = max(4, int(base * 0.8))
                elif equity >= 100:
                    dynamic = max(5, int(base * 0.6))
                else:
                    # P4-5: 小账户(<100USDT)提升上限至15，适应微仓位高并发需求
                    # 每个仓位仅1-2 USDT保证金，需要更多仓位才能达到85%利用率
                    dynamic = max(15, int(base * 0.55))
                
                # P9: 防止死锁 - 动态上限不低于当前实际持仓数
                current_positions = self.okx_client.get_positions()
                active_count = sum(1 for p in current_positions
                                 if abs(float(p.get("pos", 0) or p.get("position", 0) or 0)) > 0)
                if active_count > dynamic:
                    logger.debug(
                        f"Dynamic max adjusted: {dynamic} -> {active_count} "
                        f"(active positions > dynamic limit, preventing deadlock)"
                    )
                    dynamic = active_count
                
                return dynamic
        except Exception as e:
            # fail-closed：异常时保留死锁保护（上限不低于当前实际持仓数），避免静默回退导致满载保护失效
            logger.warning(f"Dynamic max positions computation failed: {e}, preserving deadlock protection")
            try:
                current_positions = self.okx_client.get_positions()
                active_count = sum(1 for p in current_positions
                                   if abs(float(p.get("pos", 0) or p.get("position", 0) or 0)) > 0)
                return max(base, active_count)
            except Exception:
                logger.warning("Unable to count active positions in fallback, returning base")
        return base

    async def _try_evict_low_quality_position(
        self, current_positions: list, new_symbol: str, strategy_name: str, min_confidence: float
    ) -> bool:
        """P3-2: 智能驱逐 - 满仓时检查是否有低质量持仓可替换
        
        驱逐条件（满足任一即可）：
        1. 未实现亏损 > 2% 且持仓时间 > 30分钟
        2. 网格策略持仓且当前价格已远离网格范围
        3. 同一币种已有持仓（避免同币种重复）
        
        Returns:
            True if a position was evicted (closed) to make room
        """
        try:
            if not current_positions:
                return False
            
            # 获取当前价格用于判断
            ticker = None
            try:
                ticker = await self.okx_client.get_ticker_async(new_symbol)
            except Exception:
                pass
            
            candidates = []
            now = time.time()
            
            for pos in current_positions:
                pos_symbol = pos.get("instId", "")
                if not pos_symbol:
                    continue
                
                # 跳过同币种：强制驱逐同币种再开同币种同样是零和拉锯
                if pos_symbol == new_symbol:
                    continue
                
                pos_qty = abs(float(pos.get("pos", 0) or pos.get("position", 0) or 0))
                if pos_qty <= 0:
                    continue
                
                # 计算未实现PnL
                upl = float(pos.get("upl", 0) or pos.get("unrealizedPnl", 0) or 0)
                margin = float(pos.get("margin", 0) or pos.get("imr", 0) or 0.01)
                upl_pct = upl / margin if margin > 0 else 0
                
                # 获取持仓时间（从entry_price缓存获取）
                entry_time = self._entry_price_cache.get(pos_symbol, {}).get("timestamp", 0)
                hold_seconds = now - entry_time if entry_time > 0 else 99999
                
                # 驱逐评分（越低越该被驱逐）
                score = 100.0
                
                # 1. 亏损持仓 - 大幅扣分
                if upl_pct < -0.02:
                    score -= 40
                elif upl_pct < -0.01:
                    score -= 20
                elif upl_pct < 0:
                    score -= 5
                
                # 2. 持仓时间过长但无利润 - 扣分
                if hold_seconds > 1800 and upl_pct <= 0:  # 30分钟
                    score -= 30
                elif hold_seconds > 900 and upl_pct <= 0:  # 15分钟
                    score -= 15
                
                # 3. 同币种已有持仓 - 跳过：驱逐同币种再开同币种是零和拉锯，纯浪费手续费
                if pos_symbol == new_symbol:
                    continue
                
                # 4. 盈利持仓 - 加分（不驱逐）
                if upl_pct > 0.01:
                    score += 30
                if upl_pct > 0.03:
                    score += 20
                
                candidates.append({
                    "symbol": pos_symbol,
                    "score": score,
                    "upl_pct": upl_pct,
                    "hold_seconds": hold_seconds,
                    "qty": pos_qty
                })
            
            if not candidates:
                return False
            
            # 按评分排序，最低分优先驱逐
            candidates.sort(key=lambda x: x["score"])
            worst = candidates[0]
            
            # P4-2: 提高驱逐阈值，更积极驱逐低质量持仓
            if worst["score"] >= 70:
                logger.debug(
                    f"P4-2: No low-quality position to evict (best score={worst['score']:.0f})"
                )
                return False
            
            # 执行驱逐：平仓低质量持仓
            logger.info(
                f"P3-2: Evicting {worst['symbol']} (score={worst['score']:.0f}, "
                f"upl={worst['upl_pct']:.2%}, hold={worst['hold_seconds']:.0f}s) "
                f"to make room for {new_symbol}"
            )
            
            # 获取持仓方向、价格、杠杆
            pos_side = "long"
            mark_price = 0.0
            leverage = 1
            for p in current_positions:
                if p.get("instId") == worst["symbol"]:
                    pos_side = "long" if float(p.get("pos", 0) or 0) > 0 else "short"
                    mark_price = float(p.get("markPx", 0) or p.get("avgPx", 0) or 0)
                    leverage = int(p.get("lever", 0) or 1)
                    break

            # 发送平仓信号（quantity 为币数，由 contracts_to_coins 从张数转换）
            close_qty_coins = self.okx_client.contracts_to_coins(
                worst["symbol"], worst.get("qty", 0)
            )
            close_data = {
                "symbol": worst["symbol"],
                "signal_type": "close",
                "direction": "sell" if pos_side == "long" else "buy",
                "pos_side": pos_side,
                "price": mark_price,
                "quantity": close_qty_coins,
                "leverage": leverage,
                "strategy_name": strategy_name,
                "reduce_only": True,
                "priority": 1,
                "confidence": 0.9,
                "reason": f"P3-2 eviction for {new_symbol}"
            }
            
            await self.execute_order(close_data)
            return True
            
        except Exception as e:
            logger.debug(f"P3-2: Position eviction error: {e}")
            return False

    async def _try_force_evict(
        self, current_positions: list, new_symbol: str, strategy_name: str
    ) -> bool:
        """P4-2: 强制驱逐 - 当常规驱逐失败时，强制平仓最差持仓
        
        更激进的驱逐策略：只要持仓有微小亏损或长时间无利润，就驱逐
        """
        try:
            if not current_positions:
                return False
            
            candidates = []
            now = time.time()
            
            for pos in current_positions:
                pos_symbol = pos.get("instId", "")
                if not pos_symbol:
                    continue
                
                pos_qty = abs(float(pos.get("pos", 0) or pos.get("position", 0) or 0))
                if pos_qty <= 0:
                    continue
                
                upl = float(pos.get("upl", 0) or pos.get("unrealizedPnl", 0) or 0)
                margin = float(pos.get("margin", 0) or pos.get("imr", 0) or 0.01)
                upl_pct = upl / margin if margin > 0 else 0
                
                entry_time = self._entry_price_cache.get(pos_symbol, {}).get("timestamp", 0)
                hold_seconds = now - entry_time if entry_time > 0 else 99999
                
                # P4-2: 更激进的评分
                score = 100.0
                
                # 亏损即扣分
                if upl_pct < -0.005:  # 0.5%亏损
                    score -= 50
                elif upl_pct < 0:
                    score -= 30
                
                # 持仓超过30分钟且无利润
                if hold_seconds > 1800 and upl_pct <= 0:
                    score -= 40
                elif hold_seconds > 600 and upl_pct <= 0:  # 10分钟
                    score -= 20
                
                # 盈利加分
                if upl_pct > 0.005:
                    score += 40
                
                candidates.append({
                    "symbol": pos_symbol,
                    "score": score,
                    "upl_pct": upl_pct,
                    "hold_seconds": hold_seconds,
                    "qty": pos_qty
                })
            
            if not candidates:
                return False
            
            candidates.sort(key=lambda x: x["score"])
            worst = candidates[0]
            
            # P4-2: 只要评分低于75就驱逐（更激进）
            if worst["score"] >= 75:
                return False
            
            logger.info(
                f"P4-2: Force evicting {worst['symbol']} (score={worst['score']:.0f}, "
                f"upl={worst['upl_pct']:.2%}, hold={worst['hold_seconds']:.0f}s) "
                f"to make room for {new_symbol}"
            )
            
            pos_side = "long"
            mark_price = 0.0
            leverage = 1
            for p in current_positions:
                if p.get("instId") == worst["symbol"]:
                    pos_side = "long" if float(p.get("pos", 0) or 0) > 0 else "short"
                    mark_price = float(p.get("markPx", 0) or p.get("avgPx", 0) or 0)
                    leverage = int(p.get("lever", 0) or 1)
                    break

            close_qty_coins = self.okx_client.contracts_to_coins(
                worst["symbol"], worst.get("qty", 0)
            )
            close_data = {
                "symbol": worst["symbol"],
                "signal_type": "close",
                "direction": "sell" if pos_side == "long" else "buy",
                "pos_side": pos_side,
                "price": mark_price,
                "quantity": close_qty_coins,
                "leverage": leverage,
                "strategy_name": strategy_name,
                "reduce_only": True,
                "priority": 1,
                "confidence": 0.9,
                "reason": f"P4-2 force eviction for {new_symbol}"
            }
            
            await self.execute_order(close_data)
            return True
            
        except Exception as e:
            logger.debug(f"P4-2: Force eviction error: {e}")
            return False

    async def _cleanup_orphaned_position(self, symbol: str, strategy_name: str):
        """清理孤儿持仓：通知策略更新内部状态 + 清理数据库open记录 + 清除止损挂单标记"""
        try:
            # 0. 清除防重复触发标记
            self._stop_pending.discard(f"{symbol}:long")
            self._stop_pending.discard(f"{symbol}:short")

            # 1. 通知所有注册的策略清理回调
            for callback in self._position_cleanup_callbacks:
                try:
                    callback(symbol, strategy_name)
                except Exception as e:
                    logger.error(f"Position cleanup callback error: {e}")

            # 2. 清理数据库中的open记录（标记为closed，pnl 交由对账器校正，避免写 0 假数据）
            open_records = self.sqlite_storage.get_trade_records_by_status("open", limit=100)
            for rec in open_records:
                if rec.get("symbol") == symbol and rec.get("strategy_name") == strategy_name:
                    self.sqlite_storage.update_trade_record(
                        rec["id"],
                        {
                            "status": "closed",
                            "close_time": datetime.now().isoformat(),
                            "exit_reason": "orphaned_cleanup",
                        }
                    )
                    logger.info(f"Cleaned orphaned DB record: {symbol} {strategy_name} (id={rec['id'][:16]}...)")

            # 3. 清理内部止损状态
            for sm in self._stop_managers.values():
                sm.remove_position(symbol)
            self._position_strategy_map.pop(symbol, None)
            self._position_entry_time.pop(symbol, None)

            # 4. 清理trade_journal中的open持仓
            if self._trade_journal and symbol in self._trade_journal._open_positions:
                del self._trade_journal._open_positions[symbol]
                logger.info(f"Cleaned trade_journal position: {symbol}")

        except Exception as e:
            logger.error(f"Failed to cleanup orphaned position for {symbol}: {e}")

    async def _set_leverage(self, symbol: str, leverage: int, pos_side: str = None):
        try:
            self.okx_client.set_leverage(symbol, leverage, pos_side=pos_side)
        except Exception as e:
            logger.error(f"Failed to set leverage for {symbol}: {e}")
    
    async def _determine_order_type(self, order_data: Dict[str, Any]) -> str:
        """P22-7: 限价优先策略 - 智能选择订单类型
        
        核心逻辑：
        1. 止损/止损类信号必须使用市价单（紧急执行）
        2. 剥头皮/套利策略默认市价单，但价差有利时切换到限价单
        3. 其他策略默认限价单，但价差过大时切换到市价单
        4. 限价单可节省taker费率（0.05% vs 0.02%），在价差<0.1%时优先使用
        """
        signal_type = order_data.get("signal_type", "")
        strategy_name = order_data.get("strategy_name", "")
        symbol = order_data.get("symbol", "")

        # 显式指定的订单类型优先（风控减仓/全平仓明确要求市价单，
        # 避免自动判断把 price=0 的平仓单转成限价单导致 51000 Parameter px error）
        explicit_order_type = order_data.get("order_type", "")
        if explicit_order_type in ("market", "limit", "post_only"):
            return explicit_order_type

        # 平仓/减仓信号强制市价单：reduce_only=True 或 price<=0 时，
        # 避免把 price=0 的减仓单转成限价单（限价单缺 px 会触发 51000 Parameter px error）
        if order_data.get("reduce_only", False):
            return "market"
        try:
            if float(order_data.get("price", 0) or 0) <= 0:
                return "market"
        except (TypeError, ValueError):
            pass

        # 止损信号必须市价单
        if "stop" in signal_type.lower() or "loss" in signal_type.lower():
            return "market"
        
        # P22-7: 获取当前买卖价差
        spread_pct = await self._get_current_spread_pct(symbol)
        
        # 获取限价优先配置
        pt_config = self.config.get("paper_trading", {})
        limit_priority = pt_config.get("limit_order_priority", True)
        max_spread = pt_config.get("limit_order_max_spread_pct", 0.001)
        
        # 剥头皮/套利策略：默认市价，价差有利时切换限价
        if "scalping" in strategy_name or "arbitrage" in strategy_name:
            if limit_priority and spread_pct <= max_spread:
                logger.debug(
                    f"P22-7: {strategy_name} switching to limit order for {symbol}, "
                    f"spread={spread_pct:.4%} <= max={max_spread:.4%}"
                )
                return "limit"
            return "market"
        
        # 其他策略：默认限价，价差过大时切换市价
        if limit_priority and spread_pct > max_spread * 3:
            logger.debug(
                f"P22-7: {strategy_name} switching to market order for {symbol}, "
                f"spread={spread_pct:.4%} > 3x max={max_spread:.4%}"
            )
            return "market"
        
        return "limit"
    
    async def _get_current_spread_pct(self, symbol: str) -> float:
        """P22-7: 获取当前买卖价差百分比
        
        Returns:
            价差百分比，获取失败返回0（表示价差未知，保守使用限价单）
        """
        try:
            # 优先使用异步版本避免阻塞事件循环
            ticker = await self.okx_client.get_ticker_async(symbol)
            if ticker:
                bid = float(ticker.get("bidPx", 0))
                ask = float(ticker.get("askPx", 0))
                if bid > 0 and ask > 0:
                    return (ask - bid) / bid
        except Exception:
            pass
        return 0.0  # 价差未知，保守返回0

    def _build_pending_tier_prices(self, direction: str, pending_price: float, tier_count: int, spread: float) -> List[float]:
        """生成等位挂单分档价格列表（文档 5.3）。

        等位挂单铁则：趋势确认后不追行情，在支撑/压力位附近分档挂 maker 单等回踩/反弹。
        - 开多（long/buy）：锚点为支撑位，分档价自锚点向下逐档加深（更优价格、离市价更远）
        - 开空（short/sell）：锚点为压力位，分档价自锚点向上逐档抬高
        返回按「离市价由近到远」排序的挂单价，首档最接近锚点。
        """
        prices: List[float] = []
        d = str(direction).lower()
        for i in range(tier_count):
            if d in ("long", "buy"):
                p = pending_price * (1 - spread * i)
            else:
                p = pending_price * (1 + spread * i)
            prices.append(p)
        return prices

    async def _place_pending_tier_orders(self, order_data: Dict[str, Any], symbol: str, side: str,
                                         pos_side: str, quantity: float, leverage: int, order_type: str,
                                         strategy_name: str, signal_type: str, order_id: str,
                                         pending_price: float, clordid: str):
        """等位挂单分档：将仓位拆分为 2-3 档 post_only maker 单，挂在支撑/压力位附近。

        SL/TP 不在下单时立即挂出：分档单成交前仓位尚不存在，立即挂条件单会产生
        与未成交档错配的冗余止损单；成交后由 _track_active_orders 的成交回执桥 +
        止损管理器动态接管离场，故此处只挂 maker 限价单。
        """
        cfg = self.config.get("strategies", {}).get("trend", {})
        tier_count = max(2, int(cfg.get("pending_tier_count", 3)))
        spread = float(cfg.get("pending_tier_spread", 0.005))
        # 震荡期边缘（挂单间距翻倍）：策略层透传乘数，扩大每档间距以等更深回踩
        spread *= float(order_data.get("pending_spread_multiplier", 1.0) or 1.0)
        direction = str(order_data.get("direction", "long")).lower()

        prices = self._build_pending_tier_prices(direction, pending_price, tier_count, spread)

        # 均分数量：per_tier_qty 保持「币数」交给 place_order（place_order 内部统一
        # coin→contracts→lot 取整）。此处仅用张数做「是否够最小手数」的退化判断，
        # 避免币↔张往返转换引入浮点精度损失（3张↔0.03币往返会变成2.999…张）。
        per_tier_qty = quantity / tier_count
        per_tier_contracts_check = self.okx_client.round_quantity_to_lot(
            symbol, self.okx_client.coin_to_contracts(symbol, per_tier_qty))
        if per_tier_contracts_check <= 0:
            # 兜底：均分后不足最小手数，则退化为单档全量挂锚点
            per_tier_qty = quantity
            full_check = self.okx_client.round_quantity_to_lot(
                symbol, self.okx_client.coin_to_contracts(symbol, quantity))
            prices = [pending_price]
            if full_check <= 0:
                logger.warning(f"Pending tier quantity too small for {symbol}, skipping order")
                if order_id:
                    await self._order_queue.update_order_status(order_id, "failed")
                await self._lifecycle_manager.update_status(order_id, OrderStatus.FAILED)
                return

        placed = 0
        for i, tier_price in enumerate(prices):
            # 每档唯一幂等键（clordid + 档位序号）
            tier_clordid = ""
            if self._idempotency_enabled:
                base = clordid or self._generate_idempotency_key(symbol, strategy_name, signal_type)
                # 预留档位后缀空间：base 满 32 位时若直接拼接再截断会吃掉后缀，
                # 导致各档 clordid 相同而被幂等去重跳过，故先截 base 再拼后缀。
                suffix = f"T{i + 1}"
                tier_clordid = f"{base[:32 - len(suffix)]}{suffix}"
                if self._check_idempotency(tier_clordid):
                    logger.info(f"Pending tier {i + 1} duplicate skipped for {symbol}: {tier_clordid}")
                    continue

            # post_only 单价格精度对齐
            try:
                tier_price = round(tier_price, get_price_precision(symbol))
            except Exception:
                pass

            try:
                result = self.okx_client.place_order(
                    symbol=symbol,
                    side=side,
                    order_type="post_only",
                    quantity=per_tier_qty,
                    price=tier_price,
                    leverage=leverage,
                    pos_side=pos_side,
                    reduce_only=False,
                    clOrdId=tier_clordid if self._idempotency_enabled else "",
                )
            except Exception as e:
                logger.error(f"Pending tier {i + 1} place_order error for {symbol}: {e}")
                continue

            is_failed = isinstance(result, dict) and result.get("_failed", False)
            exchange_order_id = result.get("ordId", "") if isinstance(result, dict) and not is_failed else ""

            if is_failed or not exchange_order_id:
                msg = ""
                if isinstance(result, dict):
                    msg = result.get("sMsg", "")
                logger.warning(f"Pending tier {i + 1} rejected for {symbol}: {msg or result}")
                continue

            placed += 1
            self._active_orders[exchange_order_id] = {
                **order_data,
                "status": "pending",
                "exchange_order_id": exchange_order_id,
                "filled_price": 0,
                "create_time": datetime.now(),
                "slippage_offset_applied": 0.0,
                "clOrdId": tier_clordid,
                "price": tier_price,
                "quantity": per_tier_qty,
                "pending_tier": True,
                "pending_price": pending_price,
                "tier_index": i + 1,
            }

            # 方案A：活跃订单落盘（pending tier，fill 回执丢失根因修复）
            self._persist_active_order(exchange_order_id, self._active_orders[exchange_order_id])

            if self._idempotency_enabled and tier_clordid:
                self._record_idempotency_key(tier_clordid, exchange_order_id)

            await self._lifecycle_manager.set_exchange_order_id(order_id, exchange_order_id)

            if self._account_manager:
                self._account_manager.notify_order_placed(strategy_name, per_tier_qty * tier_price / max(leverage, 1))

            logger.info(
                f"Pending tier {i + 1}/{len(prices)}: {symbol} {direction} post_only @ {tier_price:.6f} "
                f"qty={per_tier_qty:.4f} (anchor={pending_price})"
            )

        if placed > 0:
            await self._lifecycle_manager.update_status(order_id, OrderStatus.PENDING, OrderPhase.TRACKING)
            if self._risk_gate is not None:
                try:
                    self._risk_gate.record_trade()
                except Exception as _rg_err:
                    logger.debug(f"RiskGate record_trade error in pending tier: {_rg_err}")
            if order_id:
                await self._order_queue.update_order_status(order_id, "executed")
            logger.info(f"Pending tier orders placed for {symbol}: {placed}/{len(prices)} tiers (anchor={pending_price})")
        else:
            logger.warning(f"All pending tiers failed for {symbol}, no order placed")
            if order_id:
                await self._order_queue.update_order_status(order_id, "failed")
            await self._lifecycle_manager.update_status(order_id, OrderStatus.FAILED)

    async def _place_conditional_orders(self, order_data: Dict[str, Any], order_id: str):
        stop_loss = order_data.get("stop_loss")
        take_profit = order_data.get("take_profit")
        entry_price = order_data.get("price")
        direction = order_data.get("direction")
        symbol = order_data.get("symbol")
        
        if stop_loss and take_profit and entry_price:
            ticker = self.okx_client.get_ticker(symbol)
            current_price = float(ticker["last"]) if ticker else entry_price

            # P26: 归一化方向，direction 可能是 long/short/buy/sell
            # TP/SL 必须与开仓方向一致，不能以交易所持仓为准：
            # 1) 新开仓在订单成交前查不到持仓；
            # 2) 双向持仓模式下同一 symbol 可能同时存在 long+short 两笔持仓，
            #    遍历取第一笔会拿到错误方向。
            # 因此直接使用订单自身的开仓方向（_execute_order 已解析 close/pos_side）。
            try:
                verified_direction = DirectionUnifier.normalize(direction)
            except Exception as e:
                logger.error(f"Direction normalization failed for {symbol} (dir={direction}): {e}, aborting TP/SL placement")
                return
            order_data["direction"] = verified_direction

            # P17: 使用实际市场价而非订单限价作为TP/SL验证基准
            # 订单限价可能与实际成交价有偏差（滑点），导致TP在错误一侧
            # 例如：long限价0.0789，成交价0.0799，TP=0.0795会低于成交价被拒
            ref_price = current_price if current_price > 0 else entry_price

            validation = validate_tp_sl_prices(ref_price, verified_direction, take_profit, stop_loss, current_price)
            
            if not validation["valid"]:
                logger.warning(f"TP/SL validation failed for {symbol} (dir={verified_direction}): {validation['errors']}")
                # P18-3: 传入current_price，确保调整后的TP/SL不低于/高于当前市价
                stop_loss, take_profit = self._adjust_tp_sl(symbol, entry_price, verified_direction, stop_loss, take_profit, current_price)
                if stop_loss is None or take_profit is None:
                    logger.error(f"TP/SL recalculation failed for {symbol} (invalid direction), skipping TP/SL placement")
                    stop_loss = None
                    take_profit = None
                else:
                    logger.info(f"Adjusted TP/SL: SL={stop_loss:.4f}, TP={take_profit:.4f}")
                
                # P25: 调整后重新验证，确保调整后的值有效
                if stop_loss and take_profit:
                    re_validation = validate_tp_sl_prices(ref_price, verified_direction, take_profit, stop_loss, current_price)
                    if not re_validation["valid"]:
                        logger.error(
                            f"P26: TP/SL still invalid after adjustment for {symbol}: "
                            f"{re_validation['errors']}, skipping conditional order placement"
                        )
                        # 跳过无效的TP/SL下单，避免发送必然失败的订单到交易所
                        stop_loss = None
                        take_profit = None
                    else:
                        for warning in re_validation["warnings"]:
                            logger.warning(f"TP/SL warning for {symbol}: {warning}")
            else:
                for warning in validation["warnings"]:
                    logger.warning(f"TP/SL warning for {symbol}: {warning}")

        if stop_loss:
            await self._place_stop_loss(order_data, stop_loss)
        
        if take_profit:
            await self._place_staged_take_profit(order_data, take_profit, entry_price)
    
    def _adjust_tp_sl(self, symbol: str, entry_price: float, direction: str, stop_loss: float, take_profit: float, current_price: float = None) -> tuple:
        """P27: 完全重新计算TP/SL，不再依赖传入的旧值（可能来自错误方向）"""
        # P28: 归一化方向，避免 direction='buy'/'sell' 被 else 分支误判为 short
        try:
            direction = DirectionUnifier.normalize(direction)
        except Exception as e:
            logger.error(f"Direction normalization failed in _adjust_tp_sl (dir={direction}): {e}, aborting TP/SL recalculation")
            return None, None

        # 企业级：统一自适应止损止盈引擎兜底（注入后优先使用，融合波动率/市场状态/策略表现/盈亏）
        if self._adaptive_tp_sl_engine is not None:
            try:
                ctx = self._adaptive_tp_sl_engine.build_context(symbol)
                adaptive = self._adaptive_tp_sl_engine.compute(
                    symbol=symbol,
                    entry_price=entry_price,
                    direction=direction,
                    current_price=current_price,
                    apply_smoothing=False,
                    **{k: v for k, v in ctx.items() if k != "symbol"},
                )
                if adaptive.get("stop_loss") and adaptive.get("take_profit"):
                    precision = get_price_precision(symbol)
                    logger.info(
                        f"AdaptiveTpSlEngine computed TP/SL for {symbol}: "
                        f"SL={adaptive['stop_loss']}, TP={adaptive['take_profit']} "
                        f"(mode={adaptive.get('protection_mode')})"
                    )
                    return round(adaptive["stop_loss"], precision), round(adaptive["take_profit"], precision)
            except Exception as e:
                logger.warning(f"AdaptiveTpSlEngine fallback failed for {symbol}, using fixed recalculation: {e}")

        tier = get_currency_tier(symbol, self.config)
        tier_settings = self.config["currencies"][f"{tier}_settings"]
        slippage = tier_settings.get("slippage", 0.001)
        precision = get_price_precision(symbol)
        
        # P27: 使用entry_price作为基准，确保TP/SL方向正确
        # 不再使用传入的旧TP/SL值，而是从配置重新计算
        # 对于long: TP必须 > entry_price, SL必须 < entry_price
        # 对于short: TP必须 < entry_price, SL必须 > entry_price
        
        # 使用更大的margin确保TP/SL不会被市场波动立刻触发
        tp_pct = max(tier_settings.get("grid_spacing_min", 0.006) * 4, 0.02)  # 至少2%
        # P28: 止损间距设下限（至少0.3%），避免 grid_spacing_min 过小导致 SL 距离过近被判定无效
        sl_pct = max(tier_settings.get("grid_spacing_min", 0.006), 0.003)
        
        if direction == "long":
            # Long: TP在entry之上, SL在entry之下
            take_profit = calculate_take_profit(entry_price, "long", tp_pct, slippage, precision)
            stop_loss = calculate_stop_loss(entry_price, "long", sl_pct, slippage, precision)
            # 确保SL < TP
            if stop_loss >= take_profit:
                stop_loss = entry_price * 0.98
                take_profit = entry_price * 1.02
        else:
            # Short: TP在entry之下, SL在entry之上
            take_profit = calculate_take_profit(entry_price, "short", tp_pct, slippage, precision)
            stop_loss = calculate_stop_loss(entry_price, "short", sl_pct, slippage, precision)
            # 确保SL > TP
            if stop_loss <= take_profit:
                stop_loss = entry_price * 1.02
                take_profit = entry_price * 0.98
        
        # 如果current_price可用，确保TP/SL不低于/高于当前市价
        if current_price and current_price > 0:
            if direction == "long" and take_profit <= current_price:
                take_profit = current_price * (1 + tp_pct * 0.5)
                logger.debug(f"P27: TP adjusted for current price: {take_profit:.{precision}f}")
            elif direction == "short" and take_profit >= current_price:
                take_profit = current_price * (1 - tp_pct * 0.5)
                logger.debug(f"P27: TP adjusted for current price: {take_profit:.{precision}f}")
        
        return round(stop_loss, precision), round(take_profit, precision)
    
    async def _place_stop_loss(self, order_data: Dict[str, Any], stop_price: float):
        symbol = order_data["symbol"]
        direction = order_data["direction"]
        quantity = order_data["quantity"]
        leverage = order_data["leverage"]

        # 账本一致性（修复4）：仅用「币数→张数」取整结果做最小手数判断，quantity 保持币数
        contracts_check = self.okx_client.round_quantity_to_lot(
            symbol, self.okx_client.coin_to_contracts(symbol, quantity)
        )
        if contracts_check <= 0:
            logger.warning(f"Stop loss quantity too small for {symbol}, skipping")
            return

        # 归一化方向：direction 可能是 long/short/buy/sell，统一转为 long/short
        if direction in ("buy", "sell"):
            pos_side = DirectionUnifier.to_pos_side(direction)
        else:
            pos_side = direction

        # P27: 预放置价格检查 - 获取最新市价验证SL方向
        try:
            ticker = self.okx_client.get_ticker(symbol)
            if ticker:
                current_price = float(ticker.get("last", 0))
                if current_price > 0:
                    if DirectionUnifier.is_long(pos_side) and stop_price >= current_price:
                        # Long SL必须低于市价，调整到市价以下
                        old_sl = stop_price
                        stop_price = round(current_price * 0.98, get_price_precision(symbol))
                        logger.info(f"P27: SL adjusted for long {symbol}: {old_sl:.4f} -> {stop_price:.4f} (market={current_price:.4f})")
                    elif DirectionUnifier.is_short(pos_side) and stop_price <= current_price:
                        # Short SL必须高于市价，调整到市价以上
                        old_sl = stop_price
                        stop_price = round(current_price * 1.02, get_price_precision(symbol))
                        logger.info(f"P27: SL adjusted for short {symbol}: {old_sl:.4f} -> {stop_price:.4f} (market={current_price:.4f})")
        except Exception:
            pass

        # 优先使用 ConditionOrderManager 统一管理，避免双轨问题
        if self._conditional_manager:
            try:
                result = await self._conditional_manager.place_stop_loss(
                    symbol=symbol, side=pos_side, quantity=quantity,
                    trigger_price=stop_price, leverage=leverage, is_new_position=True
                )
                if result:
                    logger.info(f"Stop loss placed via manager for {symbol}: {stop_price:.4f} (algoId={result})")
                    return
                else:
                    logger.warning(f"Stop loss via manager failed for {symbol}, falling back to direct API")
            except Exception as e:
                logger.error(f"Stop loss via manager error for {symbol}: {e}, falling back to direct API")

        close_direction = "sell" if pos_side == "long" else "buy"
        try:
            result = self.okx_client.place_order(
                symbol=symbol,
                side=close_direction,
                order_type="conditional",
                quantity=quantity,
                leverage=leverage,
                stop_price=stop_price,
                reduce_only=True,
                pos_side=pos_side,
                conditional_type="stop_loss"
            )

            if result and not result.get("_failed", False):
                logger.info(f"Stop loss placed for {symbol}: {stop_price:.4f}")
            elif result and result.get("_failed"):
                logger.warning(f"Stop loss failed for {symbol}: {result.get('sMsg', '')}")
        except Exception as e:
            logger.error(f"Failed to place stop loss: {e}")

    async def _place_take_profit(self, order_data: Dict[str, Any], take_profit_price: float):
        symbol = order_data["symbol"]
        direction = order_data["direction"]
        quantity = order_data["quantity"]
        leverage = order_data["leverage"]

        # 账本一致性（修复4）：仅用「币数→张数」取整结果做最小手数判断，quantity 保持币数
        contracts_check = self.okx_client.round_quantity_to_lot(
            symbol, self.okx_client.coin_to_contracts(symbol, quantity)
        )
        if contracts_check <= 0:
            logger.warning(f"Take profit quantity too small for {symbol}, skipping")
            return

        # 归一化方向：direction 可能是 long/short/buy/sell，统一转为 long/short
        if direction in ("buy", "sell"):
            pos_side = DirectionUnifier.to_pos_side(direction)
        else:
            pos_side = direction

        # 优先使用 ConditionOrderManager 统一管理，避免双轨问题
        if self._conditional_manager:
            try:
                result = await self._conditional_manager.place_take_profit(
                    symbol=symbol, side=pos_side, quantity=quantity,
                    trigger_price=take_profit_price, leverage=leverage
                )
                if result:
                    logger.info(f"Take profit placed via manager for {symbol}: {take_profit_price:.4f} (algoId={result})")
                    return
                else:
                    logger.warning(f"Take profit via manager failed for {symbol}, falling back to direct API")
            except Exception as e:
                logger.error(f"Take profit via manager error for {symbol}: {e}, falling back to direct API")

        close_direction = "sell" if pos_side == "long" else "buy"
        try:
            result = self.okx_client.place_order(
                symbol=symbol,
                side=close_direction,
                order_type="conditional",
                quantity=quantity,
                leverage=leverage,
                stop_price=take_profit_price,
                reduce_only=True,
                pos_side=pos_side,
                conditional_type="take_profit"
            )

            if result and not result.get("_failed", False):
                logger.info(f"Take profit placed for {symbol}: {take_profit_price:.4f}")
            elif result and result.get("_failed"):
                logger.warning(f"Take profit failed for {symbol}: {result.get('sMsg', '')}")
        except Exception as e:
            logger.error(f"Failed to place take profit: {e}")

    async def _place_staged_take_profit(self, order_data: Dict[str, Any], base_tp_price: float, entry_price: float):
        """分段止盈：25% 近端 + 50% 中端 + 25% 远端，锁定利润防止回吐"""
        symbol = order_data["symbol"]
        direction = order_data["direction"]
        total_quantity = order_data["quantity"]
        leverage = order_data["leverage"]

        # 归一化方向：direction 可能是 long/short/buy/sell，统一转为 long/short
        if direction in ("buy", "sell"):
            pos_side = DirectionUnifier.to_pos_side(direction)
        else:
            pos_side = direction

        # P0: 获取当前市价，确保TP价格方向正确（long的TP必须>市价，short的TP必须<市价）
        current_price = entry_price
        try:
            ticker = self.okx_client.get_ticker(symbol)
            if ticker:
                current_price = float(ticker.get("last", entry_price))
        except Exception:
            pass

        # 根据当前市价修正 base_tp_price，避免 OKX 51279 "TP trigger price cannot be lower/higher than last price"
        precision = get_price_precision(symbol)
        if DirectionUnifier.is_long(pos_side) and base_tp_price <= current_price:
            # 多头止盈应高于市价，基于市价重新计算
            base_tp_price = round(current_price * 1.003, precision)  # 市价 + 0.3%
            logger.info(f"TP adjusted for long {symbol}: base TP raised to {base_tp_price} (market={current_price})")
        elif DirectionUnifier.is_short(pos_side) and base_tp_price >= current_price:
            # 空头止盈应低于市价
            base_tp_price = round(current_price * 0.997, precision)  # 市价 - 0.3%
            logger.info(f"TP adjusted for short {symbol}: base TP lowered to {base_tp_price} (market={current_price})")

        # 优先使用 ConditionOrderManager 统一管理
        if self._conditional_manager:
            try:
                placed_ids = await self._conditional_manager.place_staged_take_profit(
                    symbol=symbol, side=pos_side, total_quantity=total_quantity,
                    base_tp_price=base_tp_price, entry_price=entry_price, leverage=leverage
                )
                if placed_ids:
                    logger.info(f"Staged TP placed via manager for {symbol}: {len(placed_ids)} stages")
                    return
                else:
                    logger.warning(f"Staged TP via manager returned no IDs for {symbol}, falling back to direct API")
            except Exception as e:
                logger.error(f"Staged TP via manager error for {symbol}: {e}, falling back to direct API")

        close_direction = DirectionUnifier.to_side(DirectionUnifier.opposite(pos_side))

        # 计算三级止盈价格
        if DirectionUnifier.is_long(pos_side):
            tp_range = base_tp_price - entry_price
            tp1_price = entry_price + tp_range * 0.6   # 近端 60% 距离
            tp2_price = base_tp_price                    # 中端 原始 TP
            tp3_price = entry_price + tp_range * 1.5    # 远端 150% 距离
            
            # P17: 验证每个止盈价是否高于当前市价（long方向），防止sCode 51279
            if tp1_price <= current_price:
                tp1_price = round(current_price * 1.002, precision)
                logger.debug(f"P17: tp1 adjusted to {tp1_price} (was below market {current_price})")
            if tp2_price <= current_price:
                tp2_price = round(current_price * 1.003, precision)
            if tp3_price <= current_price:
                tp3_price = round(max(tp2_price, current_price * 1.004), precision)
        else:
            tp_range = entry_price - base_tp_price
            tp1_price = entry_price - tp_range * 0.6
            tp2_price = base_tp_price
            tp3_price = entry_price - tp_range * 1.5
            
            # P17: 验证每个止盈价是否低于当前市价（short方向）
            if tp1_price >= current_price:
                tp1_price = round(current_price * 0.998, precision)
                logger.debug(f"P17: tp1 adjusted to {tp1_price} (was above market {current_price})")
            if tp2_price >= current_price:
                tp2_price = round(current_price * 0.997, precision)
            if tp3_price >= current_price:
                tp3_price = round(min(tp2_price, current_price * 0.996), precision)

        # 三级仓位分配: 25%, 50%, 25%
        stages = [
            (tp1_price, 0.25, "near"),
            (tp2_price, 0.50, "mid"),
            (tp3_price, 0.25, "far"),
        ]

        for tp_price, ratio, stage_name in stages:
            # 账本一致性（单位残留修复）：total_quantity 为币数，阶段仓位按币数计算，
            # 交由 place_order 内部统一 coin→contracts 转换；最小手数判断单独用张数取整结果。
            stage_qty = total_quantity * ratio
            contracts_check = self.okx_client.round_quantity_to_lot(
                symbol, self.okx_client.coin_to_contracts(symbol, stage_qty)
            )
            if contracts_check <= 0:
                logger.warning(f"Staged TP {stage_name} qty too small for {symbol}, skipping")
                continue

            try:
                result = self.okx_client.place_order(
                    symbol=symbol,
                    side=close_direction,
                    order_type="conditional",
                    quantity=stage_qty,
                    leverage=leverage,
                    stop_price=round(tp_price, 4),
                    reduce_only=True,
                    pos_side=pos_side,
                    conditional_type="take_profit"
                )
                if result and not result.get("_failed", False):
                    logger.info(f"Staged TP {stage_name} placed for {symbol}: {tp_price:.4f} ({ratio*100:.0f}%)")
                elif result and result.get("_failed"):
                    logger.warning(f"Staged TP {stage_name} failed for {symbol}: {result.get('sMsg', '')}")
            except Exception as e:
                logger.error(f"Failed to place staged TP {stage_name} for {symbol}: {e}")
    
    async def _retry_order(self, order_data: Dict[str, Any], attempt: int = 0,
                         error_type: RetryableError = None):
        """生产级重试：指数退避 + 随机抖动 + 错误分类"""
        order_id = order_data.get("order_id", "")
        
        if attempt >= self._max_retry_attempts:
            logger.error(
                f"Order failed after {self._max_retry_attempts} attempts "
                f"(symbol={order_data.get('symbol')}, type={order_data.get('signal_type')})"
            )
            if order_id:
                await self._order_queue.update_order_status(order_id, "failed")
            return

        # 计算指数退避延迟
        delay = self._calculate_backoff_delay(attempt, error_type)
        logger.info(
            f"Retrying order in {delay:.1f}s (attempt {attempt}/{self._max_retry_attempts}, "
            f"symbol={order_data.get('symbol')}, error_type={error_type.value if error_type else 'unknown'})"
        )
        await asyncio.sleep(delay)
        
        try:
            await self._execute_order(order_data)
        except Exception as e:
            err_type = self._classify_error("", str(e))
            await self._retry_order(order_data, attempt + 1, err_type)
    
    async def _reconciliation_loop(self):
        while True:
            try:
                await self._reconcile_positions()
                await asyncio.sleep(self.config["execution"]["reconciliation_interval"])
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Loop error in _reconciliation_loop: {e}")
                await asyncio.sleep(5)
    
    async def _reconcile_positions(self):
        try:
            positions = self.okx_client.get_positions()

            # 构建 OKX 实际持仓键集合，用于检测数据库中的幽灵持仓
            okx_pos_keys = set()
            okx_pos_map = {}
            for pos_data in positions:
                position = self.okx_client._parse_position(pos_data)
                if position and abs(position.quantity) > 0:
                    self.redis_cache.set_position(position)
                    self.sqlite_storage.save_position_history({
                        "id": f"{position.symbol}:{position.side}:{datetime.now().isoformat()}",
                        "symbol": position.symbol,
                        "side": position.side,
                        "quantity": position.quantity,
                        "avg_cost": position.avg_cost,
                        "mark_price": position.mark_price,
                        "unrealized_pnl": position.unrealized_pnl,
                        "margin": position.margin,
                        "leverage": position.leverage,
                        "timestamp": position.timestamp
                    })
                    key = f"{position.symbol}:{position.side}"
                    okx_pos_keys.add(key)
                    okx_pos_map[key] = position

            # 兜底规则：检测框架外持仓（交易所存在、本地无真实策略管理）
            # 历史事故：2026-08-22 TRUMP-USDT-SWAP 手动空单(strategy=unknown)绕过风控被强平亏损 832 USDT
            # 覆盖漏洞修复：P7-4 对账会把框架外持仓 sync 进 DB(strategy_name="sync"/signal_type="unknown")，
            # 原先"仅无 open 记录才算 orphan"的判断被绕过，导致 sync 占位记录逃过兜底告警。
            # 反向修复：grid 重启后其持仓的 open 记录也会被 sync 污染，若按上面规则一律判 orphan，
            # 会把 grid 自身持仓误判为框架外并登记 manual_override，导致 grid 失去持仓归属。
            # 因此先识别策略（如 grid）主动管理的 symbol，跳过 orphan 判定，交由后续恢复逻辑按 grid 归属。
            managed_symbols = self._get_managed_position_symbols()
            orphan_positions = []
            for key, position in okx_pos_map.items():
                symbol = position.symbol
                if key in self._manual_override_positions:
                    continue  # 手动开单白名单：已显式归属 manual_override，跳过 orphan 判定
                if symbol in self._position_strategy_map:
                    continue  # 已有策略管理，正常
                if symbol in managed_symbols:
                    continue  # 策略主动管理（如 grid），非框架外 orphan，交由恢复逻辑归属

                open_records = []
                try:
                    open_records = self.sqlite_storage.get_all_open_records(symbol) or []
                except Exception:
                    open_records = []

                # 是否存在"框架内真实策略"管理的 open 记录（sync/unknown/空 均为框架外占位）
                managed = False
                for rec in open_records:
                    strat = (rec.get("strategy_name") or "").strip().lower()
                    sig = (rec.get("signal_type") or "").strip().lower()
                    if strat not in ("", "sync", "unknown") and sig != "unknown":
                        managed = True
                        break

                if not managed:
                    orphan_positions.append(position)

            if orphan_positions:
                await self._handle_orphan_positions(orphan_positions)
                # 从后续恢复逻辑中排除框架外持仓，避免被 _recover_stop_losses_after_restart 默认归到 grid
                for position in orphan_positions:
                    okx_pos_map.pop(f"{position.symbol}:{position.side}", None)

            # 同步数据库：将 OKX 已不存在的 open 持仓标记为 closed
            try:
                db_open = self.sqlite_storage.get_trade_records_by_status("open")
                ghost_closed = 0
                # P-复盘修复：按 symbol:side 分组，避免同一 symbol 多条 open 记录被写入同一个 pnl（均摊/同值 bug）
                ghost_groups = {}
                for rec in db_open:
                    # P2: 跳过仍处于挂单中(active order)的 open 记录，避免把尚未成交的限价单误判为幽灵仓。
                    # _execute_order 在下单成功时即写入 status='open'，但限价单可能尚未成交，
                    # 此时 OKX 没有对应持仓，直接按持仓对账会把它们误标为 reconciled/ghost_close。
                    rec_id = rec.get("id", "")
                    if rec_id and rec_id in self._active_orders:
                        continue
                    symbol = rec.get("symbol", "")
                    side = (rec.get("side", "") or "").lower()
                    if side in ("buy", "long"):
                        norm_side = "long"
                    elif side in ("sell", "short"):
                        norm_side = "short"
                    else:
                        norm_side = side
                    key = f"{symbol}:{norm_side}"
                    if key not in okx_pos_keys:
                        ghost_groups.setdefault(key, []).append(rec)

                for key, recs in ghost_groups.items():
                    for rec in recs:
                        # 幽灵持仓的 pnl 无法从本地数据准确核算（entry/quantity 可能陈旧、量纲也可能不一致），
                        # 一律不写 pnl（留 NULL），交由 PnLReconciler 用 OKX 平仓账单的权威 pnl 精确对账，
                        # 避免用 mark_price 估算出假数据（历史事故：单笔被算成 +71% 或真实亏损 3.7 倍）。
                        updates = {
                            "status": "closed",
                            "close_time": datetime.now().isoformat(),
                            "exit_reason": "ghost_close",
                        }
                        self.sqlite_storage.update_trade_record(rec.get("id"), updates)
                        ghost_closed += 1
                if ghost_closed > 0:
                    logger.info(f"Reconciliation: cleaned {ghost_closed} ghost positions")
            except Exception as e:
                logger.debug(f"DB position sync failed: {e}")

            account_info = self.okx_client.get_account_info()
            if account_info:
                account = self.okx_client._parse_account_info(account_info)
                self.redis_cache.set_account_info(account)
                self.sqlite_storage.save_account_history({
                    "id": datetime.now().isoformat(),
                    "total_equity": account.total_equity,
                    "available_balance": account.available_balance,
                    "used_margin": account.used_margin,
                    "unrealized_pnl": account.unrealized_pnl,
                    "margin_rate": account.margin_rate,
                    "timestamp": account.timestamp
                })

            logger.debug("Position reconciliation completed")

            # P2: 重启后恢复止损单 - 为已有持仓补注册止损管理器
            await self._recover_stop_losses_after_restart(okx_pos_map)

        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")

    def _load_manual_override_positions(self):
        """加载手动开单白名单（显式归属 strategy=manual_override）。"""
        try:
            if os.path.exists(self._manual_override_file):
                with open(self._manual_override_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._manual_override_positions = data
        except Exception as e:
            logger.warning(f"加载手动开单白名单失败: {e}")
            self._manual_override_positions = {}

    def _save_manual_override_positions(self):
        """持久化手动开单白名单。"""
        try:
            os.makedirs(os.path.dirname(self._manual_override_file), exist_ok=True)
            tmp = self._manual_override_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._manual_override_positions, f, indent=2, ensure_ascii=False, default=str)
            os.replace(tmp, self._manual_override_file)
        except Exception as e:
            logger.warning(f"保存手动开单白名单失败: {e}")

    async def _handle_orphan_positions(self, orphan_positions):
        """兜底规则：检测到框架外持仓(strategy=unknown/sync)时立即告警 + 可选自动平仓。

        历史事故：2026-08-22 一笔 TRUMP-USDT-SWAP 手动空单(strategy=unknown)绕过了
        RiskGate，最终被强平亏损 832 USDT。此方法对交易所存在、但本地无真实策略管理
        （含被 P7-4 对账 sync 成 strategy="sync"/signal_type="unknown" 的占位记录）
        的持仓兜底拦截，避免再次失控。
        """
        auto_close = bool(self.config.get("execution", {}).get("orphan_position_auto_close", False))
        for position in orphan_positions:
            symbol = position.symbol
            side = position.side
            qty = position.quantity
            key = f"{symbol}:{side}"

            # 已登记的手动开单跳过，避免每个对账周期重复告警
            if key in self._manual_override_positions:
                continue

            msg = (
                f"检测到框架外持仓 {symbol} {side} qty={qty}，"
                f"不属于任何已注册策略(strategy=unknown)，存在失控风险"
            )
            logger.error(msg)

            # 持久化风险事件（修复 risk_events 审计缺口）
            try:
                self.sqlite_storage.save_risk_event({
                    "id": f"orphan_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                    "event_type": "orphan_position",
                    "severity": "CRITICAL",
                    "message": msg,
                    "symbol": symbol,
                    "timestamp": datetime.now(),
                })
            except Exception as e:
                logger.debug(f"save_risk_event failed for {symbol}: {e}")

            # 立即告警
            if self._alert_manager:
                try:
                    await self._alert_manager.send_alert(
                        "orphan_position", msg, severity="CRITICAL", symbol=symbol
                    )
                except Exception as e:
                    logger.debug(f"orphan alert failed for {symbol}: {e}")

            # 可选自动平仓
            if auto_close:
                try:
                    result = self.okx_client.close_position(symbol, side)
                    if result and result.get("success"):
                        logger.info(f"框架外持仓 {symbol} {side} 已自动平仓")
                    else:
                        err = (result or {}).get("error", "unknown")
                        logger.error(f"框架外持仓 {symbol} 自动平仓失败: {err}")
                except Exception as e:
                    logger.error(f"框架外持仓 {symbol} 自动平仓异常: {e}")
            else:
                # 不平仓：显式归属为手动开单（manual_override），避免后续对账重复告警/误判
                self._manual_override_positions[key] = {
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "strategy": "manual_override",
                    "registered_at": datetime.now().isoformat(),
                    "note": "框架外持仓，自动平仓已关闭，登记待人工确认",
                }
                self._save_manual_override_positions()
                logger.warning(
                    f"框架外持仓 {symbol} {side} 未自动平仓，已登记为手动开单白名单（manual_override），待人工确认"
                )

    async def _recover_stop_losses_after_restart(self, okx_pos_map: Dict[str, Any]):
        """P2: 重启后为已有持仓恢复止损单
        
        系统重启后，_position_strategy_map 和 _stop_managers 为空，
        但交易所仍有持仓。需要将持仓注册到止损管理器，让后续的
        _dynamic_stop_loss_loop 能为它们放置止损单。
        """
        if not okx_pos_map:
            return

        recovered_count = 0
        for key, position in okx_pos_map.items():
            symbol = position.symbol
            if symbol in self._position_strategy_map:
                continue  # 已有注册，跳过

            # 确定策略名称：优先从数据库查询，其次默认 grid。
            # 关键修复：DB open 记录可能被 P7-4 污染为 strategy_name="sync"/"unknown"（占位），
            # 直接采用会把 grid 持仓错误注册为 sync，导致 grid 失去持仓归属（重启孤儿化根因）。
            # 遍历全部 open 记录，取「最新的一条真实策略记录」；仅当全是占位/空时回退到 grid。
            strategy_name = "grid"
            try:
                open_records = self.sqlite_storage.get_all_open_records(symbol) or []
                for rec in open_records:
                    raw = (rec.get("strategy_name") or "").strip().lower()
                    if raw not in ("", "sync", "unknown"):
                        strategy_name = raw
                        break
            except Exception:
                pass

            # 注册到止损管理器
            sm = self._stop_managers.get(strategy_name) or self._stop_managers.get("grid")
            if sm and symbol:
                pos_side = position.side  # 'long' or 'short'
                entry_price = float(position.avg_cost) if position.avg_cost > 0 else float(position.mark_price)
                pos_qty_contracts = abs(float(position.quantity))
                pos_qty_coins = self.okx_client.contracts_to_coins(symbol, pos_qty_contracts)
                sm.init_position_stop(symbol, entry_price, pos_side, pos_qty_coins)
                self._position_strategy_map[symbol] = strategy_name
                self._position_entry_time[symbol] = datetime.now()
                recovered_count += 1
                logger.info(f"P2: Recovered stop loss registration for {symbol} {pos_side} "
                           f"qty={pos_qty_coins:.4f} entry={entry_price:.4f} strategy={strategy_name}")

        if recovered_count > 0:
            logger.info(f"P2: Recovered stop loss for {recovered_count} positions after restart")
    
    async def _order_tracking_loop(self):
        while True:
            try:
                await self._track_active_orders()
                self._cleanup_partial_fill_tracker()
                # 方案A：低频全量对账补回执（fill 回执丢失根因修复，节流避免 REST 限频）
                now = time.time()
                if (now - self._last_fill_reconcile_ts) >= self._fill_reconcile_interval:
                    self._last_fill_reconcile_ts = now
                    await self._reconcile_fill_receipts()
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Loop error in _order_tracking_loop: {e}")
                await asyncio.sleep(5)
    
    def _load_entry_price_cache(self):
        """P26: 从文件加载持久化的entry_price缓存"""
        try:
            if os.path.exists(self._entry_price_cache_file):
                with open(self._entry_price_cache_file, 'r') as f:
                    data = json.load(f)
                    # 只加载TTL内的缓存（网络降级时使用更长的TTL）
                    ttl = self._entry_price_cache_ttl_degraded  # 启动时用降级TTL
                    now = time.time()
                    for symbol, entry in data.items():
                        if now - entry.get("cached_at", 0) < ttl:
                            self._entry_price_cache[symbol] = entry
                    if self._entry_price_cache:
                        logger.info(f"P26: Loaded {len(self._entry_price_cache)} entry_price cache entries from file")
        except Exception as e:
            logger.debug(f"P26: Failed to load entry_price cache file: {e}")
    
    def _save_entry_price_cache(self):
        """P26: 持久化entry_price缓存到文件"""
        try:
            os.makedirs(os.path.dirname(self._entry_price_cache_file), exist_ok=True)
            with open(self._entry_price_cache_file, 'w') as f:
                json.dump(self._entry_price_cache, f)
        except Exception as e:
            logger.debug(f"P26: Failed to save entry_price cache: {e}")
    
    def _get_entry_price_cache_ttl(self) -> float:
        """P26: 根据网络状态返回合适的缓存TTL"""
        try:
            if hasattr(self.okx_client, 'is_network_degraded') and self.okx_client.is_network_degraded:
                return self._entry_price_cache_ttl_degraded
        except Exception:
            pass
        return self._entry_price_cache_ttl
    
    def _get_entry_price_cache_ttl_for_source(self, source: str = "") -> float:
        """P27: 根据entry_price来源返回合适的TTL
        - position: 从交易所持仓avgPx获取，1小时TTL（avgPx不会变）
        - 其他来源: 使用默认TTL
        """
        if source == "position":
            return self._entry_price_cache_ttl_position
        return self._get_entry_price_cache_ttl()

    def _resolve_track_direction(self, order_info: Dict[str, Any]) -> str:
        """将 order_info 中的 direction 解析为 'long' 或 'short'
        
        处理 'close' 方向：根据 reduce_only 标记和 pos_side 推断实际方向。
        平仓信号的方向是持仓方向的反方向。
        """
        direction = order_info.get("direction", "long")
        if direction in ("long", "short"):
            return direction
        if direction in ("buy",):
            return "long"
        if direction in ("sell",):
            return "short"
        # direction == "close" 或其他未知值
        pos_side = order_info.get("pos_side", "")
        if pos_side in ("long", "short"):
            # close 信号：平仓方向与持仓方向相反
            return "short" if pos_side == "long" else "long"
        # 最后兜底：从 reduce_only + side 推断
        if order_info.get("reduce_only", False):
            side = order_info.get("side", "")
            if side == "buy":
                return "short"  # 买入平仓 = 平空仓
            if side == "sell":
                return "long"   # 卖出平仓 = 平多仓
        # 完全无法推断，默认 long
        logger.debug(f"Cannot resolve track direction from order_info: direction={direction}, pos_side={pos_side}, defaulting to long")
        return "long"

    async def _track_active_orders(self):
        orders_to_remove = []
        
        for exchange_order_id, order_info in list(self._active_orders.items()):
            try:
                order_status = self.okx_client.get_order(order_info["symbol"], exchange_order_id)
                
                if order_status:
                    state = order_status.get("state", "")
                    strategy_name = order_info.get("strategy_name", "grid")
                    quantity = order_info.get("quantity", 0)
                    price = order_info.get("price", 0)
                    leverage = order_info.get("leverage", 1)
                    margin_used = quantity * price / leverage

                    # 等位挂单分档超时重评（文档 5.3）：挂单超过 pending_timeout_seconds 仍未成交 → 撤单
                    # 让行情重新触发，避免旧挂单在支撑/压力位失效后继续等待错失时机。
                    if order_info.get("pending_tier") and state in ("live", "partially_filled"):
                        timeout_sec = float(self.config.get("strategies", {}).get("trend", {}).get("pending_timeout_seconds", 1800))
                        create_time = order_info.get("create_time")
                        elapsed = (datetime.now() - create_time).total_seconds() if isinstance(create_time, datetime) else 0.0
                        if elapsed > timeout_sec:
                            logger.info(
                                f"Pending tier timeout: {exchange_order_id} {order_info.get('symbol')} "
                                f"live {elapsed:.0f}s > {timeout_sec:.0f}s, cancelling for re-evaluation"
                            )
                            cancelled = False
                            try:
                                if self._check_cancel_allowed(order_info.get("symbol", "")):
                                    cancel_res = self.okx_client.cancel_order(order_info.get("symbol", ""), exchange_order_id)
                                    if cancel_res and not cancel_res.get("_failed", False):
                                        self._record_cancel(order_info.get("symbol", ""))
                                        cancelled = True
                            except Exception as e:
                                logger.error(f"Pending tier timeout cancel error for {exchange_order_id}: {e}")
                            if cancelled:
                                if self._account_manager:
                                    self._account_manager.notify_order_canceled(strategy_name, margin_used)
                                # P3 埋点：等位挂单超时撤单 → 发布 ORDER_CANCELLED
                                self._publish_event(EventType.ORDER_CANCELLED, {
                                    "symbol": order_info.get("symbol", ""),
                                    "exchange_order_id": exchange_order_id,
                                    "strategy_name": strategy_name,
                                    "reason": "pending_tier_timeout",
                                })
                                await self._lifecycle_manager.update_status(order_info.get("order_id"), OrderStatus.CANCELLED)
                                orders_to_remove.append(exchange_order_id)
                                continue
                            # 取消失败：下一轮跟踪继续尝试

                    # ── 生产级：部分成交检测与处理 ──
                    if state == "partially_filled":
                        filled_qty = float(order_status.get("fillSz", "0"))
                        filled_price = float(order_status.get("avgPx", "0"))
                        total_qty = float(order_status.get("sz", quantity))
                        
                        logger.info(
                            f"Order partially filled: {exchange_order_id} "
                            f"{order_info.get('symbol')} filled={filled_qty}/{total_qty}"
                        )
                        
                        # 处理部分成交
                        resolved = await self._handle_partial_fill(
                            exchange_order_id, order_info,
                            filled_qty, total_qty, filled_price
                        )
                        if resolved:
                            # 部分成交已处理，如果已全部成交则继续处理
                            if filled_qty >= total_qty * 0.999:
                                # 接近全部成交，按filled处理
                                state = "filled"
                            else:
                                orders_to_remove.append(exchange_order_id)
                                continue
                    
                    if state == "filled":
                        filled_price = float(order_status.get("avgPx", "0"))
                        order_info["filled_price"] = filled_price
                        order_info["status"] = "filled"
                        
                        await self._lifecycle_manager.record_fill(order_info.get("order_id"), filled_price, order_info.get("quantity", 0))
                        await self._lifecycle_manager.update_status(order_info.get("order_id"), OrderStatus.FILLED, OrderPhase.SETTLEMENT)

                        # P1 埋点：成交 → 发布 ORDER_FILLED；平仓/减仓 → 再发布平仓事件
                        _fill_symbol = order_info.get("symbol", "")
                        _fill_reduce = bool(order_info.get("reduce_only", False))
                        _fill_sig = str(order_info.get("signal_type", "")).lower()
                        _fill_filled_qty = float(order_status.get("fillSz", "0") or 0)
                        _fill_payload = {
                            "symbol": _fill_symbol,
                            "exchange_order_id": exchange_order_id,
                            "direction": order_info.get("direction", ""),
                            "pos_side": order_info.get("pos_side", ""),
                            "signal_type": order_info.get("signal_type", ""),
                            "strategy_name": strategy_name,
                            "quantity": order_info.get("quantity", 0),
                            "filled_qty": _fill_filled_qty if _fill_filled_qty > 0 else order_info.get("quantity", 0),
                            "filled_price": filled_price,
                            "reduce_only": _fill_reduce,
                            "trace_id": order_info.get("trace_id", ""),
                        }
                        self._publish_event(EventType.ORDER_FILLED, _fill_payload)
                        if _fill_reduce:
                            if "stop_loss" in _fill_sig or "sl" in _fill_sig:
                                _close_evt = EventType.STOP_LOSS_TRIGGERED
                            elif "take_profit" in _fill_sig or "tp" in _fill_sig:
                                _close_evt = EventType.TAKE_PROFIT_TRIGGERED
                            else:
                                _close_evt = EventType.POSITION_CLOSED
                            self._publish_event(_close_evt, _fill_payload)

                        # Phase 1: 成交回执桥 — 通知策略（ghost_close 专项，仅观察不改仓）
                        filled_qty = float(order_status.get("fillSz", "0") or 0)
                        if filled_qty <= 0:
                            filled_qty = quantity
                        self._notify_fill_callbacks({
                            "symbol": order_info.get("symbol", ""),
                            "direction": order_info.get("direction", ""),
                            "pos_side": order_info.get("pos_side", ""),
                            "signal_type": order_info.get("signal_type", ""),
                            "strategy_name": strategy_name,
                            "filled_qty": filled_qty,
                            "avg_price": filled_price,
                            "quantity": quantity,
                            "clOrdId": order_info.get("clOrdId", ""),
                            "exchange_order_id": exchange_order_id,
                            "reduce_only": order_info.get("reduce_only", False),
                        })

                        # 使用 OKX 返回的真实手续费
                        actual_fee = abs(float(order_status.get("fee", "0"))) if order_status.get("fee") else filled_price * quantity * 0.0005

                        if self._account_manager:
                            self._account_manager.notify_order_filled(strategy_name, margin_used)

                        # 记录成交质量（滑点统计）
                        if self._fill_quality_tracker:
                            expected_px = order_info.get("price", price)
                            # side 和 order_type 从 order_info 获取，避免未定义错误
                            direction_norm = self._resolve_track_direction(order_info)
                            fill_side = DirectionUnifier.to_side(direction_norm)
                            order_type_norm = order_info.get("type", order_info.get("order_type", "market"))
                            self._fill_quality_tracker.record_fill(
                                order_id=exchange_order_id,
                                symbol=order_info["symbol"],
                                strategy_name=strategy_name,
                                side=fill_side,
                                expected_price=float(expected_px),
                                filled_price=filled_price,
                                quantity=quantity,
                                order_type=order_type_norm,
                                slippage_tolerance=self._slippage_tolerance,
                            )

                        # 滑点优化器自适应学习：记录实际成交滑点
                        if self._slippage_optimizer:
                            try:
                                expected_px = float(order_info.get("price", price))
                                direction_norm = self._resolve_track_direction(order_info)
                                fill_side = DirectionUnifier.to_side(direction_norm)
                                offset = order_info.get("slippage_offset_applied", 0.0)
                                symbol = order_info.get("symbol", "")
                                self._slippage_optimizer.record_fill_slippage(
                                    symbol=symbol,
                                    side=fill_side,
                                    order_type=order_info.get("order_type", "limit"),
                                    expected_price=expected_px,
                                    filled_price=filled_price,
                                    quantity=quantity,
                                    offset_applied=float(offset),
                                )
                            except Exception as e:
                                logger.debug(f"SlippageOptimizer fill recording skipped: {e}")

                        # 记录算法执行质量（到达价格滑点/实现缺口等）
                        if self._execution_quality_monitor:
                            try:
                                exp_px = float(order_info.get("price", price))
                                actual_commission = abs(float(order_status.get("fee", "0"))) if order_status.get("fee") else filled_price * quantity * 0.0005
                                self._execution_quality_monitor.record_execution(
                                    symbol=order_info["symbol"],
                                    side=fill_side,
                                    quantity=quantity,
                                    execution_price=filled_price,
                                    arrival_price=exp_px,
                                    decision_price=exp_px,
                                    commission=actual_commission,
                                    strategy=strategy_name,
                                    filled=True,
                                    fill_quantity=quantity,
                                )
                            except Exception as e:
                                logger.debug(f"Execution quality recording skipped: {e}")

                        # ── 资金磨损分析：记录手续费 + 滑点 ──
                        if self._attrition_analyzer:
                            try:
                                symbol_order = order_info.get("symbol", "")
                                direction_norm = self._resolve_track_direction(order_info)
                                fill_side = DirectionUnifier.to_side(direction_norm)
                                trade_value = quantity * filled_price
                                # 记录手续费
                                expected_px = float(order_info.get("price", price))
                                is_maker = order_info.get("order_type", "limit") == "limit"
                                self._attrition_analyzer.record_fee(
                                    symbol_order, strategy_name, actual_fee,
                                    trade_value, fill_side, is_maker=is_maker
                                )
                                # 记录滑点
                                if expected_px > 0 and filled_price > 0:
                                    self._attrition_analyzer.record_slippage(
                                        symbol_order, strategy_name,
                                        expected_px, filled_price, quantity, fill_side
                                    )
                                # 更新自适应费率
                                self._attrition_analyzer.update_adaptive_fee(
                                    actual_fee, trade_value, is_maker
                                )
                            except Exception as e:
                                logger.debug(f"Attrition recording skipped: {e}")

                        # 先判断是否为平仓信号（避免 close 信号被错误开仓）
                        close_signal_types = ["stop_loss", "take_profit", "close", "reduce", "liquidation", "margin_call", "exit", "trailing", "tp"]
                        sig_type_lower = str(order_info.get("signal_type", "")).lower()
                        is_close_order = any(st in sig_type_lower for st in close_signal_types)

                        symbol_order = order_info.get("symbol", "")

                        if is_close_order:
                            # 平仓订单：direction 应为「被平掉的持仓方向」(long/short)，而非下单 side。
                            # pos_side 显式指定了持仓方向（平多 pos_side=long，平空 pos_side=short）。
                            # 若依赖 _resolve_track_direction，direction="sell"(平多) 会被误解析为 "short"，
                            # 导致 PnL 计算反向、trade_journal 把平仓误记为加仓。
                            direction = order_info.get("pos_side", "")
                            if direction not in ("long", "short"):
                                # 兜底：_resolve_track_direction 返回下单 side（long/short 编码），
                                # 平仓时被平的持仓方向与之相反（平多=sell→short 编码 → 持仓 long）
                                direction = self._resolve_track_direction(order_info)
                                direction = "short" if direction == "long" else "long"
                            okx_pnl = float(order_status.get("pnl", "0")) if order_status.get("pnl") else 0

                            # 获取正确的入场价：优先从trade_journal获取，其次从数据库open记录精确查询
                            entry_price = None
                            open_trade_id = None
                            open_rec = None

                            # P25: 路径0 - 检查entry_price缓存，避免重复无效查询
                            cache_entry = self._entry_price_cache.get(symbol_order)
                            cache_source = cache_entry.get("source", "") if cache_entry else ""
                            cache_ttl = self._get_entry_price_cache_ttl_for_source(cache_source)
                            if cache_entry and (time.time() - cache_entry["cached_at"]) < cache_ttl:
                                entry_price = cache_entry["entry_price"]
                                logger.debug(f"P27: Using cached entry_price={entry_price:.4f} for {symbol_order} (ttl={cache_ttl}s, source={cache_source})")

                            # 路径1: 从trade_journal内存中的_open_positions获取
                            if not entry_price and self._trade_journal and symbol_order in self._trade_journal._open_positions:
                                pos = self._trade_journal._open_positions[symbol_order]
                                entry_price = pos.avg_cost

                            # 路径2: 从数据库精确查询该symbol+strategy的最近open记录。
                            # open_trade_id 始终查询（用于平仓回写关联），entry_price 仅在缺失时才从 open 记录回填。
                            try:
                                open_rec = self.sqlite_storage.get_latest_open_record(symbol_order, strategy_name)
                                if open_rec:
                                    open_trade_id = open_rec["id"]
                                    if not entry_price:
                                        # 优先用filled_price（开仓实际成交价），其次用price（下单价）
                                        entry_price = open_rec.get("filled_price") or open_rec.get("price")
                                        if entry_price and entry_price > 0:
                                            logger.debug(f"Recovered entry_price={entry_price:.4f} from DB open record {open_trade_id[:16]}...")
                            except Exception as e:
                                logger.warning(f"Failed to query open record for {symbol_order}/{strategy_name}: {e}")

                            # 路径3: 兜底从order_info获取（但区分开仓价和平仓价）
                            if not entry_price:
                                # order_info中的entry_price是开仓订单成交时记录的
                                entry_price = order_info.get("entry_price")
                                if entry_price:
                                    logger.debug(f"Using entry_price={entry_price:.4f} from order_info")

                            # 路径4: 实在找不到entry_price，记录警告，不要用filled_price凑（会导致pnl=0假数据）
                            if not entry_price or entry_price <= 0:
                                # P27: 网络降级时跳过bills查询，直接走路径5（持仓avgPx）
                                skip_bills = False
                                try:
                                    if hasattr(self.okx_client, 'is_network_degraded') and self.okx_client.is_network_degraded:
                                        skip_bills = True
                                except Exception:
                                    pass
                                
                                if not skip_bills:
                                    logger.warning(f"No entry_price found for {symbol_order}/{strategy_name}, querying OKX bills for actual entry")
                                    # 尝试从OKX账单获取最近的开仓成交记录
                                    try:
                                        bills = self.okx_client.get_bills(symbol_order, bill_type="2", limit=5)  # type=2是开仓
                                        if bills:
                                            for bill in bills:
                                                bill_pnl = float(bill.get("pnl", "0"))
                                                if bill_pnl == 0:  # 开仓记录pnl=0
                                                    entry_price = float(bill.get("fillPx", "0"))
                                                    if entry_price > 0:
                                                        logger.info(f"Recovered entry_price={entry_price:.4f} from OKX bills")
                                                        break
                                    except Exception as e:
                                        logger.warning(f"Failed to query OKX bills: {e}")
                                else:
                                    logger.debug(f"P27: Skipping bills query for {symbol_order} (network degraded)")

                            if not entry_price or entry_price <= 0:
                                # 路径5: 从交易所持仓数据获取avgPx（兜底，避免重复bill查询失败）
                                # 适用于系统重启后数据库丢失open记录，或手动开仓等场景
                                try:
                                    positions = await self.okx_client.get_positions_async()
                                    if positions:
                                        for pos in positions:
                                            if pos.get("instId") == symbol_order:
                                                pos_avg_px = float(pos.get("avgPx", 0))
                                                if pos_avg_px > 0:
                                                    entry_price = pos_avg_px
                                                    logger.info(
                                                        f"P25: Recovered entry_price={entry_price:.4f} for {symbol_order} "
                                                        f"from exchange position avgPx"
                                                    )
                                                    # 写入数据库缓存，避免下次重复查询
                                                    try:
                                                        self.sqlite_storage.update_trade_record(
                                                            exchange_order_id,
                                                            {"filled_price": entry_price}
                                                        )
                                                    except Exception:
                                                        pass
                                                    break
                                except Exception as e:
                                    logger.warning(f"P25: Failed to get entry_price from positions for {symbol_order}: {e}")

                            # P25/P26: 缓存成功获取的entry_price，避免重复查询，并持久化到文件
                            if entry_price and entry_price > 0:
                                self._entry_price_cache[symbol_order] = {
                                    "entry_price": entry_price,
                                    "cached_at": time.time(),
                                    "source": "position"  # P27: 标记来源为交易所持仓
                                }
                                # P26: 异步持久化到文件（不阻塞）
                                try:
                                    self._save_entry_price_cache()
                                except Exception:
                                    pass

                            if not entry_price or entry_price <= 0:
                                # 最终兜底：跳过PnL计算，记录错误，不要写pnl=0的假数据
                                logger.error(f"Cannot determine entry_price for {symbol_order}, skipping PnL calc to avoid fake pnl=0")
                                # 移除止损状态
                                for sm in self._stop_managers.values():
                                    sm.remove_position(symbol_order)
                                self._position_strategy_map.pop(symbol_order, None)
                                self._position_entry_time.pop(symbol_order, None)
                                # 标记订单为closed但pnl=None（数据库会存null而非0）
                                self.sqlite_storage.update_trade_record(
                                    exchange_order_id,
                                    {"filled_price": filled_price, "status": "closed", "close_time": datetime.now().isoformat(),
                                     "exit_reason": order_info.get("exit_reason") or sig_type_lower or "close"}
                                )
                                orders_to_remove.append(exchange_order_id)
                                continue

                            # 计算PnL：手动计算为基准，OKX返回值做校验
                            # 手动计算：(平仓价 - 开仓价) * 数量 - 手续费
                            if direction == "long":
                                gross_pnl = (filled_price - entry_price) * quantity
                            else:
                                gross_pnl = (entry_price - filled_price) * quantity
                            manual_pnl = gross_pnl - actual_fee

                            # 交易所返回的 okx_pnl 是权威值（已扣资金费/手续费），优先采用；
                            # manual_pnl 仅在 okx_pnl 缺失或为 0 时作为兜底。
                            # 方向解析正确后二者应接近；不再因比值差异弃用 OKX 值，
                            # 避免方向误判时反而采用错误的手动计算值。
                            if okx_pnl != 0:
                                pnl = okx_pnl
                            else:
                                pnl = manual_pnl

                            # pnl_percent 基于 margin 计算（而非 entry_price*qty）
                            # margin = entry_price * quantity / leverage
                            position_value = entry_price * quantity
                            margin_used = position_value / leverage if leverage > 0 else position_value
                            pnl_percent = (pnl / margin_used * 100) if margin_used > 0 else 0

                            # 移除止损状态
                            for sm in self._stop_managers.values():
                                sm.remove_position(symbol_order)
                            self._position_strategy_map.pop(symbol_order, None)
                            self._position_entry_time.pop(symbol_order, None)
                            # P0: 清理防重复触发标记，避免永久阻塞该symbol的止损检查
                            self._stop_pending.discard(f"{symbol_order}:long")
                            self._stop_pending.discard(f"{symbol_order}:short")

                            # 记录到 trade_journal
                            if self._trade_journal and symbol_order in self._trade_journal._open_positions:
                                if self._trade_journal._okx_client:
                                    try:
                                        ticker = self._trade_journal._okx_client.get_ticker(symbol_order)
                                        exit_price = float(ticker["last"]) if ticker else filled_price
                                    except Exception:
                                        exit_price = filled_price
                                else:
                                    exit_price = filled_price

                                fill_data = {
                                    "trade_id": exchange_order_id,
                                    "symbol": symbol_order,
                                    "strategy_name": strategy_name,
                                    "direction": "sell" if direction == "long" else "buy",
                                    "price": exit_price,
                                    "quantity": quantity,
                                    "leverage": leverage,
                                    "fees": actual_fee,
                                    "signal_type": order_info.get("signal_type", "unknown"),
                                    "exit_reason": order_info.get("exit_reason", sig_type_lower),
                                    "trace_id": order_info.get("trace_id", "")
                                }
                                await self._trade_journal.record_fill(fill_data)

                            # P1: 回传平仓盈亏到策略（grid 日内亏损熔断等）
                            self._notify_strategy_pnl_callbacks(strategy_name, symbol_order, float(pnl))

                            # 通知 ProfitOptimizer 记录交易结果（用于凯利公式）
                            if self._profit_optimizer:
                                self._profit_optimizer.record_trade_result(pnl, strategy_name)

                            # P24: 通知智能交易体记录交易结果（平仓时根据实际PnL记录）
                            if hasattr(self, '_intelligent_agent') and self._intelligent_agent:
                                try:
                                    self._intelligent_agent.record_trade_result(
                                        strategy_name=strategy_name,
                                        symbol=symbol_order,
                                        pnl=float(pnl),
                                        is_win=float(pnl) > 0,
                                    )
                                except Exception as e:
                                    logger.debug(f"IntelligentAgent record_trade error: {e}")

                            # 记录策略利润到磨损分析器（用于磨损率计算）
                            if self._attrition_analyzer:
                                try:
                                    self._attrition_analyzer.record_profit(strategy_name, float(pnl))
                                except Exception as e:
                                    logger.debug(f"Attrition profit recording skipped: {e}")

                            # 更新数据库中的开仓记录（区分全平 vs 部分平仓，避免部分平仓误关开仓记录）
                            if open_trade_id:
                                # 计算平仓后剩余持仓量：open 记录原始数量 - 本次平仓数量
                                open_qty = 0.0
                                try:
                                    if open_rec:
                                        open_qty = float(open_rec.get("quantity", 0) or 0)
                                except Exception:
                                    open_qty = 0.0
                                close_qty = float(quantity or 0)
                                remaining_qty = open_qty - close_qty
                                is_partial_close = open_qty > 0 and remaining_qty > 1e-8

                                if is_partial_close:
                                    # 部分平仓：保持 open，仅回写剩余数量，不标记 closed
                                    self.sqlite_storage.update_trade_record(
                                        open_trade_id,
                                        {"quantity": remaining_qty}
                                    )
                                    logger.info(f"Position partially closed: {symbol_order} closed {close_qty:.4f}, remaining {remaining_qty:.4f}, PnL: {pnl:.4f} (open record kept)")
                                else:
                                    # 全平：更新开仓记录为 closed 状态（保留entry_price在price字段）
                                    self.sqlite_storage.update_trade_record(
                                        open_trade_id,
                                        {
                                            "filled_price": filled_price,
                                            "status": "closed",
                                            "pnl": pnl,
                                            "pnl_percent": pnl_percent,
                                            "fees": actual_fee,
                                            "close_time": datetime.now().isoformat(),
                                            "exit_reason": order_info.get("exit_reason") or sig_type_lower or "close"
                                        }
                                    )
                                    logger.info(f"Position closed: {symbol_order} entry={entry_price:.4f} exit={filled_price:.4f}, PnL: {pnl:.4f}, fee: {actual_fee:.4f} (updated open record {open_trade_id[:16]}...)")
                                # 通过路径：记录平仓盈亏到五层风控（L4连续亏损降杠杆/暂停跟踪）
                                if self._risk_gate is not None:
                                    try:
                                        self._risk_gate.record_trade_result(float(pnl))
                                    except Exception as _rg_err:
                                        logger.debug(f"RiskGate record_trade_result error: {_rg_err}")
                                # 通过路径：回传交易记录到MiniBacktester（盘中动态回测校验）
                                try:
                                    mb = get_mini_backtester()
                                    strategy_id = f"{strategy_name}:{symbol_order}"
                                    trade_rec = TradeRecord(
                                        timestamp=datetime.now(),
                                        symbol=symbol_order,
                                        side=direction,
                                        entry_price=entry_price,
                                        exit_price=filled_price,
                                        quantity=quantity,
                                        pnl=float(pnl),
                                        pnl_pct=float(pnl_percent) if pnl_percent else 0.0,
                                        hold_bars=0,
                                        signal_source=strategy_name,
                                        entry_reason=order_info.get("signal_type", ""),
                                        exit_reason=order_info.get("exit_reason", "close")
                                    )
                                    mb.add_trade(strategy_id, trade_rec)
                                except Exception as _mb_err:
                                    logger.debug(f"MiniBacktester add_trade error: {_mb_err}")
                            else:
                                # 找不到开仓记录：用平仓订单ID创建closed记录（记录entry_price便于后续对账）
                                self.sqlite_storage.update_trade_record(
                                    exchange_order_id,
                                    {
                                        "filled_price": filled_price,
                                        "status": "closed",
                                        "pnl": pnl,
                                        "pnl_percent": pnl_percent,
                                        "fees": actual_fee,
                                        "close_time": datetime.now().isoformat(),
                                        "exit_reason": order_info.get("exit_reason") or sig_type_lower or "close"
                                    }
                                )
                                logger.warning(f"Position closed (no open record): {symbol_order} entry={entry_price:.4f} exit={filled_price:.4f}, PnL: {pnl:.4f}")
                                # P1: 回传平仓盈亏到策略（grid 日内亏损熔断等）
                                self._notify_strategy_pnl_callbacks(strategy_name, symbol_order, float(pnl))
                                # P24: 通知智能交易体记录交易结果
                                if hasattr(self, '_intelligent_agent') and self._intelligent_agent:
                                    try:
                                        self._intelligent_agent.record_trade_result(
                                            strategy_name=strategy_name,
                                            symbol=symbol_order,
                                            pnl=float(pnl),
                                            is_win=float(pnl) > 0,
                                        )
                                    except Exception as e:
                                        logger.debug(f"IntelligentAgent record_trade error: {e}")
                                # 通过路径：记录平仓盈亏到五层风控（L4连续亏损降杠杆/暂停跟踪）
                                if self._risk_gate is not None:
                                    try:
                                        self._risk_gate.record_trade_result(float(pnl))
                                    except Exception as _rg_err:
                                        logger.debug(f"RiskGate record_trade_result error: {_rg_err}")
                                # 通过路径：回传交易记录到MiniBacktester（盘中动态回测校验）
                                try:
                                    mb = get_mini_backtester()
                                    strategy_id = f"{strategy_name}:{symbol_order}"
                                    trade_rec = TradeRecord(
                                        timestamp=datetime.now(),
                                        symbol=symbol_order,
                                        side=direction,
                                        entry_price=entry_price,
                                        exit_price=filled_price,
                                        quantity=quantity,
                                        pnl=float(pnl),
                                        pnl_pct=float(pnl_percent) if pnl_percent else 0.0,
                                        hold_bars=0,
                                        signal_source=strategy_name,
                                        entry_reason=order_info.get("signal_type", ""),
                                        exit_reason=order_info.get("exit_reason", "close")
                                    )
                                    mb.add_trade(strategy_id, trade_rec)
                                except Exception as _mb_err:
                                    logger.debug(f"MiniBacktester add_trade error: {_mb_err}")
                        else:
                            # 开仓订单：记录entry_price到order_info，便于后续平仓时获取
                            order_info["entry_price"] = filled_price

                            # 记录到 trade_journal
                            if self._trade_journal:
                                fill_data = {
                                    "trade_id": exchange_order_id,
                                    "symbol": symbol_order,
                                    "strategy_name": strategy_name,
                                    "direction": self._resolve_track_direction(order_info),
                                    "price": filled_price,
                                    "quantity": quantity,
                                    "leverage": leverage,
                                    "fees": actual_fee,
                                    "signal_type": order_info.get("signal_type", "unknown"),
                                    "exit_reason": order_info.get("exit_reason", ""),
                                    "trace_id": order_info.get("trace_id", "")
                                }
                                await self._trade_journal.record_fill(fill_data)

                            self.sqlite_storage.update_trade_record(
                                exchange_order_id,
                                {"filled_price": filled_price, "fees": actual_fee, "status": "open"}
                            )

                            # 从OKX获取实际持仓avgPx，校正数据库中的entry_price
                            # 避免本地filled_price与交易所实际均价不同步
                            try:
                                await asyncio.sleep(0.3)  # 等待OKX持仓数据更新
                                okx_positions = self.okx_client.get_positions()
                                actual_avg_px = None
                                actual_pos_qty = None
                                for pos_data in okx_positions:
                                    if pos_data.get("instId") == symbol_order:
                                        pos_qty = float(pos_data.get("pos", 0))
                                        if abs(pos_qty) > 0:
                                            actual_avg_px = float(pos_data.get("avgPx", 0))
                                            actual_pos_qty = abs(pos_qty)
                                            break

                                if actual_avg_px and actual_avg_px > 0:
                                    # 用OKX实际均价校正数据库记录
                                    if abs(actual_avg_px - filled_price) / filled_price > 0.001:  # 差异>0.1%才更新
                                        logger.info(f"Syncing avgPx for {symbol_order}: local={filled_price:.4f} -> OKX={actual_avg_px:.4f}")
                                        self.sqlite_storage.update_trade_record(
                                            exchange_order_id,
                                            {"filled_price": actual_avg_px, "price": actual_avg_px}
                                        )
                                        # 同步更新order_info的entry_price
                                        order_info["entry_price"] = actual_avg_px
                                    else:
                                        # 差异很小，仅记录但不更新
                                        order_info["entry_price"] = actual_avg_px
                            except Exception as e:
                                logger.debug(f"Failed to sync avgPx for {symbol_order}: {e}")

                            # 初始化强化止损状态
                            direction_norm = self._resolve_track_direction(order_info)
                            dir_norm = direction_norm  # 已是 'long' 或 'short'
                            sm = self._stop_managers.get(strategy_name) or self._stop_managers.get("grid")
                            if sm and symbol_order:
                                actual_entry = order_info.get("entry_price", filled_price)
                                # actual_pos_qty 是 OKX API 返回的合约张数，需转为币数存入止损管理器
                                # 止损管理器后续生成的止盈/止损信号量会经 place_order 统一做 coin→contracts 转换
                                if actual_pos_qty and actual_pos_qty > 0:
                                    pos_quantity = self.okx_client.contracts_to_coins(symbol_order, actual_pos_qty)
                                else:
                                    # fallback：从 order_info 取（已在下单时由 place_order 转为合约张数，需反算）
                                    fallback_qty = float(order_info.get("quantity", quantity) or quantity)
                                    ct_val = float(self.okx_client.get_instrument_info(symbol_order).get("ctVal", "1") or "1")
                                    pos_quantity = fallback_qty * ct_val
                                sm.init_position_stop(symbol_order, actual_entry, dir_norm, pos_quantity)
                                self._position_strategy_map[symbol_order] = strategy_name
                                self._position_entry_time[symbol_order] = datetime.now()

                        orders_to_remove.append(exchange_order_id)
                        logger.info(f"Order filled: {exchange_order_id} @ {filled_price:.4f}")
                    
                    elif state in ("cancelled", "failed"):
                        if self._account_manager:
                            self._account_manager.notify_order_canceled(strategy_name, margin_used)
                        
                        status_enum = OrderStatus.CANCELLED if state == "cancelled" else OrderStatus.FAILED
                        await self._lifecycle_manager.update_status(order_info.get("order_id"), status_enum)

                        # P3 埋点：交易所回报 cancelled → 发布 ORDER_CANCELLED（事件溯源落盘）
                        if state == "cancelled":
                            self._publish_event(EventType.ORDER_CANCELLED, {
                                "symbol": order_info.get("symbol", ""),
                                "exchange_order_id": exchange_order_id,
                                "strategy_name": strategy_name,
                                "reason": "exchange_state_cancelled",
                            })
                        
                        # P2: 开仓订单取消/失败时，关闭下单时预写的 open 记录，
                        # 避免遗留为幽灵仓被 _reconcile_positions 误标为 reconciled/ghost_close。
                        # 平仓订单不会预写 open 记录（且持仓仍真实存在），跳过。
                        is_close_order = order_info.get("reduce_only", False) or any(
                            kw in str(order_info.get("signal_type", "")).lower()
                            for kw in ["stop_loss", "take_profit", "close", "reduce", "exit", "tp", "liquidation", "margin_call"]
                        )
                        if not is_close_order:
                            try:
                                self.sqlite_storage.update_trade_record(
                                    exchange_order_id,
                                    {
                                        "status": "closed",
                                        "close_time": datetime.now().isoformat(),
                                        "exit_reason": "cancelled" if state == "cancelled" else "failed",
                                        "pnl": 0.0,
                                        "pnl_percent": 0.0,
                                    }
                                )
                            except Exception as e:
                                logger.debug(f"P2: Failed to close cancelled open record {exchange_order_id}: {e}")
                        
                        orders_to_remove.append(exchange_order_id)
                        logger.info(f"Order cancelled/failed: {exchange_order_id}")
            except Exception as e:
                logger.error(f"Error tracking order {exchange_order_id}: {e}")
        
        for order_id in orders_to_remove:
            del self._active_orders[order_id]
            # 方案A：同步删除落盘记录（fill 回执丢失根因修复）
            self._remove_active_order(order_id)
    
    async def handle_signal(self, signal_data: Dict[str, Any]):
        if self._order_queue is None:
            logger.error("Order queue not set")
            return

        symbol = signal_data.get("symbol", "")
        confidence = signal_data.get("confidence", 0.0)
        strategy_name = signal_data.get("strategy_name", "")
        signal_type = signal_data.get("signal_type", "")

        # 平仓/止损信号不过滤，直接执行
        # 优先检查显式传入的 reduce_only 标记（最高优先级，最准确）
        is_close_signal = signal_data.get("reduce_only", False)
        if not is_close_signal:
            is_close_signal = any(kw in signal_type.lower() for kw in [
                "close", "stop_loss", "stop", "exit", "hedge", "reduce", "liquidation", "margin_call", "trailing", "tp", "take_profit"
            ])

        if not is_close_signal:
            # 信号质量阈值
            min_quality = self.config.get("strategies", {}).get(strategy_name, {}).get("min_signal_quality", 0.20)
            if confidence < min_quality:
                logger.debug(f"Signal filtered: {strategy_name} {signal_type} confidence {confidence:.2f} < {min_quality}")
                return

            # === 4层手续费+收益保护（杜绝手续费大于收益） ===
            price = signal_data.get("price", 0)
            quantity = signal_data.get("quantity", 0)
            leverage = signal_data.get("leverage", 1)

            if price > 0 and quantity > 0 and leverage > 0:
                position_value = quantity * price
                margin = position_value / leverage

                # ─ 第1层：最低保证金 ─
                min_margin = self._get_effective_min_margin()
                if margin < min_margin:
                    logger.debug(
                        f"Signal rejected: {symbol} margin={margin:.4f} < min={min_margin:.4f} "
                        f"(pos_value={position_value:.2f} USD)"
                    )
                    return

                # ─ 第2层：最低名义价值 ─
                min_notional = self.config.get("trading", {}).get("min_notional_usd", 15.0)
                if position_value < min_notional:
                    logger.debug(
                        f"Signal rejected: {symbol} notional={position_value:.2f} USD < min={min_notional} USD"
                    )
                    return

                # 成本计算：开仓taker + 平仓taker（保守采用taker费率）
                taker_fee = self.config.get("trading", {}).get("taker_fee_rate", 0.001)
                max_slippage = self.config.get("trading", {}).get("max_slippage_pct", 0.001)
                total_fee_cost = position_value * taker_fee * 2  # 开+平 taker 费
                slippage_cost = position_value * max_slippage   # 滑点成本
                total_cost = total_fee_cost + slippage_cost

                tp = signal_data.get("take_profit")
                sl = signal_data.get("stop_loss")

                # ─ 第3层：收益成本比（仅当信号有止盈止损时） ─
                if tp and sl:
                    direction = signal_data.get("direction", "")
                    if direction in ["long", "buy"]:
                        expected_profit_usd = (tp - price) * quantity
                    else:
                        expected_profit_usd = (price - tp) * quantity

                    if expected_profit_usd > 0 and total_cost > 0:
                        ratio = expected_profit_usd / total_cost
                        min_ratio = self._get_effective_profit_cost_ratio(strategy_name)
                        if ratio < min_ratio:
                            logger.warning(
                                f"Signal rejected: {symbol} {strategy_name} "
                                f"profit/cost={ratio:.2f} < {min_ratio} "
                                f"(expected_profit={expected_profit_usd:.4f} USD, total_cost={total_cost:.4f} USD)"
                            )
                            return

                # ─ 第4层：净收益硬门槛（扣费后利润必须为正） ─
                # 即使没有明确tp/sl，也需要确保有足够的价格空间覆盖费用
                if tp and price > 0:
                    direction = signal_data.get("direction", "")
                    if direction in ["long", "buy"]:
                        price_move_pct = (tp - price) / price
                    else:
                        price_move_pct = (price - tp) / price

                    # 最小所需价格变动 = 总费率 * 2（确保扣费后有正收益）
                    min_move_pct = taker_fee * 2 * 1.5  # taker开平 * 1.5倍安全边际（网格策略需要更小的盈利空间）
                    if price_move_pct < min_move_pct:
                        logger.warning(
                            f"Signal rejected: {symbol} {strategy_name} "
                            f"tp_move={price_move_pct:.4%} < min_required={min_move_pct:.4%} "
                            f"(fee would consume all profit)"
                        )
                        return

                # ProfitOptimizer 综合判断（凯利仓位 + 复利 + 回撤保护 + 最小保证金）
                # unified 信号已由统一仓位引擎完成仓位计算，跳过二次折扣，避免微单被拒
                if self._profit_optimizer and leverage > 0 and signal_data.get("position_sizing") != "unified":
                    base_margin = margin
                    adjusted_margin = self._profit_optimizer.get_optimal_position_size(base_margin, leverage)
                    if adjusted_margin <= 0:
                        logger.debug(f"Signal filtered by ProfitOptimizer: {strategy_name} {signal_data.get('symbol')} margin too small after optimization")
                        return
                    # 按比例调整 quantity（保持币数口径，交由 place_order 统一 coin→contracts 转换）
                    if base_margin > 0:
                        ratio = adjusted_margin / base_margin
                        new_qty = quantity * ratio
                        # 最小手数检查用张数取整结果，quantity 仍为币数，避免张数被当币数二次转换
                        contracts_check = self.okx_client.round_quantity_to_lot(
                            symbol, self.okx_client.coin_to_contracts(symbol, new_qty)
                        )
                        if contracts_check > 0:
                            signal_data = dict(signal_data)
                            signal_data["quantity"] = new_qty

        # 防御性规范化 direction：将 'close' 等非标准方向提前解析为 'long'/'short'
        raw_direction = signal_data.get("direction", "long")
        if raw_direction in ("long", "short", "buy", "sell"):
            # P-复盘修复：归一化 buy→long, sell→short，保证 trade_records.side 方向编码一致
            resolved_dir = DirectionUnifier.normalize(raw_direction)
        else:
            # 尝试从 pos_side 推断
            pos_side = signal_data.get("pos_side", "")
            if pos_side in ("long", "short"):
                # close 信号：平仓方向与持仓方向相反
                resolved_dir = "short" if pos_side == "long" else "long"
            elif signal_data.get("reduce_only", False):
                resolved_dir = "long"  # 兜底
            else:
                resolved_dir = "long"
            logger.debug(f"Normalized non-standard direction '{raw_direction}' -> '{resolved_dir}' via pos_side={pos_side}")

        order_data = {
            "symbol": signal_data["symbol"],
            "direction": resolved_dir,
            "price": signal_data["price"],
            "quantity": signal_data["quantity"],
            "leverage": signal_data["leverage"],
            "strategy_name": signal_data["strategy_name"],
            "signal_type": signal_data["signal_type"],
            # ghost_close 专项 Phase 2: 透传策略生成的 clOrdId 用于成交回执关联
            "clOrdId": signal_data.get("clOrdId", ""),
            # 透传显式指定的订单类型（风控减仓/全平仓明确要求市价单）
            "order_type": signal_data.get("order_type", ""),
            # 等位挂单价：>0 表示挂在支撑/压力位等回踩/反弹（post_only 时生效）
            "pending_price": signal_data.get("pending_price", 0.0),
            # 等位挂单间距乘数：震荡期边缘=2.0 翻倍间距
            "pending_spread_multiplier": signal_data.get("pending_spread_multiplier", 1.0),
            "stop_loss": signal_data.get("stop_loss"),
            "take_profit": signal_data.get("take_profit"),
            "confidence": signal_data.get("confidence", 0.0),
            "timestamp": datetime.fromisoformat(signal_data["timestamp"]),
            # 透传平仓标记和持仓方向（_handle_take_profit_action / _check_dynamic_stop_loss 会显式传入）
            "reduce_only": signal_data.get("reduce_only", False),
            "pos_side": signal_data.get("pos_side", ""),
            "close_position": signal_data.get("close_position", False),
            "reason": signal_data.get("reason", ""),
            # 透传离场原因，供平仓时回写到 trade_records.exit_reason（避免复盘时 exit_reason=None）
            "exit_reason": signal_data.get("exit_reason", ""),
        }

        await self._order_queue.add_order(order_data)

    async def _dynamic_stop_loss_loop(self):
        """动态止损循环：移动止损 + 保本止损 + 最小持仓周期检查"""
        while True:
            try:
                await self._check_dynamic_stop_loss()
            except Exception as e:
                logger.error(f"Dynamic stop loss loop error: {e}")
            await asyncio.sleep(5)

    async def _check_dynamic_stop_loss(self):
        """检查所有持仓的动态止损（集成分级止盈+时间止盈+波动率保护+止损联动）"""
        if not self._position_strategy_map:
            return

        positions = self.okx_client.get_positions()
        if not positions:
            return

        now = datetime.now()

        for pos_data in positions:
            try:
                position = self.okx_client._parse_position(pos_data)
                if not position or abs(float(position.quantity)) == 0:
                    continue

                symbol = position.symbol
                strategy_name = self._position_strategy_map.get(symbol)
                if not strategy_name:
                    continue

                sm = self._stop_managers.get(strategy_name)
                if not sm:
                    continue

                entry_time = self._position_entry_time.get(symbol)

                # 最小持仓周期检查（止盈不受限制）
                min_hold_ok = True
                if entry_time and (now - entry_time).total_seconds() < self._min_hold_minutes * 60:
                    min_hold_ok = False

                current_price = float(position.mark_price) if position.mark_price > 0 else 0
                if current_price <= 0:
                    continue

                # 获取 ATR（带缓存，5分钟刷新一次）
                atr = await self._get_cached_atr(symbol)

                # 更新止损（带 ATR 动态调整）
                new_stop, stop_type = sm.update_stop(symbol, current_price, atr)

                # ==================== 分级止盈检查 ====================
                if hasattr(sm, 'check_take_profit'):
                    tp_actions = sm.check_take_profit(symbol, current_price)
                    for tp_action in tp_actions:
                        await self._handle_take_profit_action(
                            symbol, strategy_name, position, tp_action
                        )

                # ==================== 时间止盈检查 ====================
                if hasattr(sm, 'check_time_exit') and entry_time:
                    time_action = sm.check_time_exit(symbol, current_price)
                    if time_action:
                        await self._handle_take_profit_action(
                            symbol, strategy_name, position, time_action
                        )

                # ==================== 波动率紧急止损检查 ====================
                if hasattr(sm, 'check_volatility_spike') and atr > 0:
                    vol_action = sm.check_volatility_spike(symbol, current_price)
                    if vol_action:
                        await self._handle_take_profit_action(
                            symbol, strategy_name, position, vol_action
                        )

                # ==================== 止损检查（受最小持仓周期限制） ====================
                if not min_hold_ok:
                    continue

                stop_key = f"{symbol}:{position.side}"
                if stop_key in self._stop_pending:
                    continue

                if sm.check_stop_trigger(symbol, current_price):
                    direction = "long" if position.side == "long" else "short"
                    close_side = "sell" if direction == "long" else "buy"

                    logger.info(f"Dynamic stop triggered: {symbol} {stop_type} stop={new_stop:.4f} price={current_price:.4f}")

                    # position.quantity 是合约张数，需转换为币数（place_order 会统一做 coin→contracts 转换）
                    pos_qty_contracts = abs(float(position.quantity))
                    pos_qty_coins = self.okx_client.contracts_to_coins(symbol, pos_qty_contracts)

                    close_signal = {
                        "type": "signal",
                        "data": {
                            "symbol": symbol,
                            "strategy_name": strategy_name,
                            "signal_type": f"dynamic_{stop_type}_stop",
                            "direction": close_side,
                            "pos_side": direction,
                            "price": current_price,
                            "quantity": pos_qty_coins,
                            "leverage": position.leverage,
                            "stop_loss": new_stop,
                            "take_profit": None,
                            "confidence": 0.95,
                            "timestamp": datetime.now().isoformat(),
                            # 显式标记平仓，避免执行层把动态止损误判为开仓（方向反转 bug）
                            "reduce_only": True,
                            "close_position": True,
                        }
                    }
                    if self._order_queue:
                        await self._order_queue.add_order(close_signal["data"])
                        self._stop_pending.add(stop_key)
            except Exception as e:
                logger.error(f"Error checking dynamic stop for position: {e}")

    async def _handle_take_profit_action(self, symbol: str, strategy_name: str,
                                         position, action: Dict[str, Any]):
        """处理止盈/减仓动作（TP1/TP2/TP3/时间止盈/波动率止损）"""
        try:
            action_type = action.get("action", "")
            side = action.get("side", "sell")
            quantity = float(action.get("quantity", 0))
            price = float(action.get("price", 0))
            reason = action.get("reason", action_type)

            if quantity <= 0:
                return

            # position.quantity 是合约张数，转换为币数后与止盈量（币数）比较
            pos_qty_contracts = abs(float(position.quantity))
            pos_qty_coins = self.okx_client.contracts_to_coins(symbol, pos_qty_contracts)
            quantity = min(quantity, pos_qty_coins)

            logger.info(f"Take-profit action: {symbol} {action_type} qty={quantity:.4f} price={price:.4f} reason={reason}")

            tp_signal = {
                "symbol": symbol,
                "strategy_name": strategy_name,
                "signal_type": f"tp_{action_type}",
                "direction": side,
                "pos_side": position.side,
                "price": price,
                "quantity": quantity,
                "leverage": position.leverage,
                "stop_loss": None,
                "take_profit": price,
                "confidence": 0.9,
                "timestamp": datetime.now().isoformat(),
                "reduce_only": True,
                "close_position": action.get("close_position", False),
                "reason": reason
            }

            if self._order_queue:
                await self._order_queue.add_order(tp_signal)

            # 止盈触发后联动止损上移
            if action_type in ("tp1", "tp2") and self._conditional_manager:
                await self._conditional_manager.update_stop_loss_after_tp(symbol, action_type)

            # 记录部分平仓到 EnhancedStopLoss
            sm = self._stop_managers.get(strategy_name)
            if sm and hasattr(sm, 'record_partial_close'):
                sm.record_partial_close(symbol, quantity)

        except Exception as e:
            logger.error(f"Error handling take-profit action for {symbol}: {e}")

    async def _get_cached_atr(self, symbol: str) -> float:
        """获取 ATR 值，带 5 分钟缓存"""
        now = time.time()
        if not hasattr(self, '_atr_cache'):
            self._atr_cache = {}
        
        cache_entry = self._atr_cache.get(symbol)
        if cache_entry and (now - cache_entry["ts"]) < 300:
            return cache_entry["value"]
        
        try:
            atr = await asyncio.to_thread(self.okx_client.get_atr, symbol)
            self._atr_cache[symbol] = {"value": atr, "ts": now}
            return atr
        except Exception as e:
            logger.debug(f"Failed to get ATR for {symbol}: {e}")
            return 0.0