"""OKX 交易所客户端封装，提供多 API 密钥轮询、REST 与 WebSocket 行情和交易接口。"""
import asyncio
import json
import hmac
import hashlib
import math
import base64
import os
import random
import time
import threading
from collections import deque
from datetime import datetime
from typing import Dict, Any, List, Optional
from loguru import logger
import websockets
import aiohttp
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

# websockets v13+ 用 .state 代替 .open，兼容性辅助
try:
    from websockets.protocol import State as _ws_State
    _WS_OPEN_STATE = _ws_State.OPEN
except ImportError:
    _WS_OPEN_STATE = None  # 旧版使用 .open 属性

from core.models import TickData, BarData, Order, Position, AccountInfo, FundingRate

class APIKeyPool:
    """API密钥池 - 轮询多个密钥解决限流问题"""

    def __init__(self, api_keys: List[Dict[str, str]]):
        self._keys = api_keys
        self._index = 0
        self._lock = threading.Lock()
        self._rate_limited = {}  # 记录每个密钥的限流恢复时间

    @property
    def count(self) -> int:
        return len(self._keys)

    def get_current(self) -> Dict[str, str]:
        """获取当前API密钥"""
        with self._lock:
            now = time.time()
            # 如果当前密钥被限流，跳过它
            current_key = self._keys[self._index]["api_key"]
            if current_key in self._rate_limited:
                if now < self._rate_limited[current_key]:
                    # 切换到下一个可用的
                    self._index = (self._index + 1) % len(self._keys)
                else:
                    # 限流已过，清除标记
                    del self._rate_limited[current_key]
            return self._keys[self._index].copy()

    def next(self) -> Dict[str, str]:
        """切换到下一个API密钥并返回"""
        with self._lock:
            self._index = (self._index + 1) % len(self._keys)
            return self._keys[self._index].copy()

    def mark_rate_limited(self, api_key: str, duration_seconds: int = 10):
        """标记某个API密钥为限流状态"""
        with self._lock:
            self._rate_limited[api_key] = time.time() + duration_seconds
            # 立即切换到下一个
            self._index = (self._index + 1) % len(self._keys)

    def reset(self):
        """重置所有限流状态"""
        with self._lock:
            self._rate_limited.clear()


class OKXClient:
    def __init__(self, config: Dict[str, Any]):
        # 支持多API密钥配置
        api_keys_config = config["okx"].get("api_keys", [])
        
        if api_keys_config and len(api_keys_config) > 0:
            # 使用API密钥池
            self._api_pool = APIKeyPool(api_keys_config)
            current = self._api_pool.get_current()
            self.api_key = current["api_key"]
            self.secret_key = current["secret_key"]
            self.passphrase = current["passphrase"]
            logger.info(f"API密钥池已初始化: {self._api_pool.count}个密钥")
        else:
            # 单密钥模式（兼容旧配置）
            self._api_pool = None
            self.api_key = config["okx"]["api_key"]
            self.secret_key = config["okx"]["secret_key"]
            self.passphrase = config["okx"]["passphrase"]

        self.is_testnet = config["okx"]["is_testnet"]
        self.rest_url = config["okx"]["rest_url"]
        self.ws_public_url = config["okx"]["websocket_url"]
        self.ws_private_url = config["okx"]["websocket_private_url"]
        self.proxy = config["okx"].get("proxy")
        # 境内直连被墙时关闭直连回退：代理失败仅切换/重试代理，不切直连、不临时禁用全部代理
        self.allow_direct_fallback = bool(config["okx"].get("allow_direct_fallback", False))

        # ── 多代理源自动切换 ──
        # proxy_list 优先级 > proxy 单值；为空时无代理（直连）
        _proxy_list_raw = config["okx"].get("proxy_list")
        if _proxy_list_raw and isinstance(_proxy_list_raw, list):
            self._proxy_list = [p.strip() for p in _proxy_list_raw
                                if p and isinstance(p, str) and p.strip()]
        elif self.proxy:
            self._proxy_list = [self.proxy]
        else:
            self._proxy_list = []
        # 兼容旧字段：self.proxy 始终指向当前活跃代理（供日志/外部读取）
        self.proxy = self._proxy_list[0] if self._proxy_list else None
        # 每个代理独立的失败计数 / 禁用截止时间 / 退避周期
        self._proxy_states: Dict[str, Dict[str, float]] = {
            p: {"fail_count": 0, "disabled_until": 0.0, "disable_cycle": 0}
            for p in self._proxy_list
        }
        self._current_proxy_idx = 0  # 当前活跃代理在 _proxy_list 中的索引
        self._proxy_fail_threshold = 3  # 单代理连续失败次数阈值，超过则禁用该代理并切换
        self._proxy_disable_duration = 60.0  # 单代理禁用基础时长（秒）
        self._proxy_max_disable = 600.0  # 单代理禁用上限（秒）

        self._session = requests.Session()
        self._async_session: Optional[aiohttp.ClientSession] = None
        self._async_session_direct: Optional[aiohttp.ClientSession] = None  # 直连备用 async session（代理故障降级）
        self._timeout = 15  # 全局请求超时15秒
        self._connect_timeout = 5  # 连接(含TLS握手)超时5秒：代理故障时快速失败并降级直连
        if self.proxy:
            self._session.proxies = {"http": self.proxy, "https": self.proxy}
            logger.info(f"Using proxy: {self.proxy}")
        # 直连备用session（代理故障时降级使用，显式清除代理防止环境变量干扰）
        self._session_direct = requests.Session()
        self._session_direct.proxies = {"http": None, "https": None}
        self._session_direct.trust_env = False  # 禁用系统代理环境变量
        # 代理恢复探测：代理被禁用后主动探测其是否恢复，避免半死代理被盲目重连
        self._proxy_probe_path = "/api/v5/public/time"  # 轻量探测端点（OKX服务器时间）
        self._proxy_probe_interval = 30.0  # 代理禁用期间每隔30秒探测一次
        self._proxy_probe_timeout = 3.0  # 探测超时（轻量探测，快速失败，不阻塞交易核心）
        self._proxy_last_probe_ts = 0.0  # 上次探测时间戳
        self._proxy_probe_in_progress = False  # 探测去重标志，防止并发重复探测

        # P5: API延迟韧性 - 自适应超时和延迟熔断
        self._latency_ema = 0.0  # 延迟指数移动平均（ms）
        self._latency_ema_alpha = 0.2  # EMA平滑系数
        self._latency_samples: deque = deque(maxlen=20)  # 最近20次延迟采样
        self._latency_circuit_open = False  # 延迟熔断开关
        self._latency_circuit_opened_at = 0.0  # 熔断开启时间
        self._latency_circuit_cooldown = 60.0  # 熔断冷却60秒
        self._latency_critical_threshold = 20000  # P7: 延迟>20秒触发熔断检查（10s太敏感，API延迟5-15s很常见）
        self._latency_critical_count = 0  # 连续高延迟计数
        self._latency_critical_count_max = 5  # P7: 连续5次高延迟触发熔断（3次太敏感）
        self._adaptive_timeout_min = 20  # P7: 自适应超时下限20s（15s太短，API延迟经常10-15s）
        self._adaptive_timeout_max = 60  # P7: 自适应超时上限60s（45s不足应对极端延迟）

        self._ws_public = None
        self._ws_private = None
        self._ws_running = False
        self._subscribed_public_channels: List[Dict[str, Any]] = []
        self._subscribed_private_channels: List[Dict[str, Any]] = []
        self._subscribed_symbols: List[str] = []
        self._books_last_sequence: Dict[str, int] = {}
        self._books_resync_pending: set = set()

        # 连接状态跟踪
        self._ws_public_connected = False
        self._ws_private_connected = False
        self._ws_public_last_pong = 0.0
        self._ws_private_last_pong = 0.0
        self._ws_public_last_data = 0.0
        self._ws_private_last_data = 0.0
        self._ws_heartbeat_interval = 30  # OKX要求30秒内发送ping
        self._ws_pong_timeout = 30  # P2: 30s (API延迟4000-7000ms，15s太激进)
        self._ws_data_timeout = 120  # P2: 120s (API延迟高时60s太短)
        self._ws_reconnect_lock_public = asyncio.Lock()
        self._ws_reconnect_lock_private = asyncio.Lock()
        self._ws_public_tasks: List[asyncio.Task] = []
        self._ws_private_tasks: List[asyncio.Task] = []
        self._ws_public_reconnecting = False
        self._ws_private_reconnecting = False
        self._ws_private_logged_in = False

        self.tick_callback = None
        self.order_callback = None

        # 风控埋点回调（由scheduler注入RiskGate的record_api_call/record_latency）
        self._on_api_call = None
        self._on_latency = None
        self.position_callback = None

        # instrument info 缓存（symbol -> info dict），避免每次下单都发HTTP请求
        self._instrument_cache: Dict[str, Dict[str, Any]] = {}
        self._instrument_cache_ts: Dict[str, float] = {}
        self._instrument_cache_ttl = 3600  # 缓存1小时

        # 全局REST请求频率控制：每个窗口最多N次请求，避免触发429限流
        self._rest_request_times: deque = deque(maxlen=50)
        self._rest_rate_limit_window = 2.0  # 2秒窗口
        self._rest_max_requests_per_window = 10  # 最多10次/2秒 = 5次/秒
        
        # P23: 异步请求并发信号量，防止同时发出过多请求触发429
        self._async_request_semaphore = asyncio.Semaphore(5)  # 最多5个并发请求
        
        # P26: 网络健康追踪 - 自适应降级
        self._network_fail_count = 0  # 连续网络失败计数
        self._network_healthy = True  # 网络健康状态
        self._kline_cache = {}  # P4-3: K线数据缓存 {key: {"data": [...], "ts": timestamp}}
        self._network_fail_threshold = 3  # 连续3次失败判定为网络降级
        self._network_recovery_threshold = 2  # 连续2次成功恢复网络
        self._network_success_count = 0  # 连续成功计数（用于恢复判定）
        self._positions_cache = None  # 持仓缓存
        self._positions_cache_ts = 0.0  # 持仓缓存时间戳
        # 网络降级时延长缓存TTL
        self._positions_cache_ttl_normal = 15  # 正常15秒
        self._positions_cache_ttl_degraded = 300  # 降级300秒

        # 网络断连告警：连续 N 次失败时写入醒目告警，避免"静默无单"不易察觉
        self._consecutive_net_fail = 0  # 跨请求类型统一的连续网络失败计数
        self._net_alert_threshold = 5   # 连续失败达到该阈值时发出明显告警
        self._net_alert_emitted = False  # 是否已发出断连告警（避免刷屏，恢复后重置）
        self._disconnect_cancel_needed = False  # 断连后需在恢复时撤掉挂单（等位挂单暴露保护）
        self._disconnect_cancel_in_progress = False  # 撤单进行中标志，避免撤单请求失败再次触发断连告警
        
        # P5-1: 429限流追踪 - 防止限流风暴
        self._rate_limited_until = 0.0  # 限流恢复时间戳
        self._rate_limit_429_count = 0  # 429计数
        self._rate_limit_429_reset_ts = 0.0  # 429计数重置时间
        
        # P27: 网络降级冷却 - 防止重试风暴导致CPU 100%
        self._network_degraded_since = 0.0  # 网络开始降级的时间戳
        self._network_degraded_cooldown = 60.0  # 降级后60秒内减少非关键请求
        self._last_retry_storm_warning = 0.0  # 上次重试风暴警告时间
        self._retry_storm_count = 0  # 重试风暴计数
        self._retry_storm_threshold = 5  # P30: 5秒内超过此数触发风暴保护(原10)
        
        # P28: Ticker缓存 - 减少网络不稳定时的重复API调用
        self._ticker_cache: Dict[str, Dict[str, Any]] = {}  # {symbol: {"data": ticker, "ts": timestamp}}
        self._ticker_cache_ttl_normal = 2.0  # 正常2秒
        self._ticker_cache_ttl_degraded = 30.0  # 降级30秒
        self._ticker_max_server_age_seconds = max(
            0.0,
            float(config.get("market_data", {}).get("ticker_max_server_age_seconds", 5.0)),
        )
        market_data_config = config.get("market_data", {})
        self._funding_rate_cache: Dict[str, Dict[str, Any]] = {}
        self._funding_rate_cache_ttl = max(
            0.0, float(market_data_config.get("funding_rate_cache_ttl_seconds", 60.0))
        )
        self._funding_rate_settlement_window = max(
            0.0, float(market_data_config.get("funding_rate_settlement_window_seconds", 120.0))
        )
        self._funding_rate_settlement_ttl = max(
            0.0, float(market_data_config.get("funding_rate_settlement_ttl_seconds", 10.0))
        )
        self._funding_rate_stale_fallback_ttl = max(
            self._funding_rate_cache_ttl,
            float(market_data_config.get("funding_rate_stale_fallback_ttl_seconds", 300.0)),
        )
        self._orderbook_min_depth = max(
            1,
            int(config.get("market_data", {}).get("orderbook_min_depth", 1)),
        )

    def _get_current_keys(self) -> Dict[str, str]:
        """获取当前使用的API密钥（支持密钥池）"""
        if self._api_pool:
            keys = self._api_pool.get_current()
            self.api_key = keys["api_key"]
            self.secret_key = keys["secret_key"]
            self.passphrase = keys["passphrase"]
        return {
            "api_key": self.api_key,
            "secret_key": self.secret_key,
            "passphrase": self.passphrase
        }

    def _rotate_key(self):
        """切换到下一个API密钥"""
        if self._api_pool and self._api_pool.count > 1:
            old_key = self.api_key
            keys = self._api_pool.next()
            self.api_key = keys["api_key"]
            self.secret_key = keys["secret_key"]
            self.passphrase = keys["passphrase"]
            logger.debug(f"API密钥切换: {old_key[:8]}... -> {self.api_key[:8]}...")

    @property
    def is_network_healthy(self) -> bool:
        """P26: 网络健康状态 - 供其他模块查询"""
        return self._network_healthy
    
    @property
    def is_network_degraded(self) -> bool:
        """P27: 网络是否处于降级状态（含冷却期和429限流）"""
        # P5-1: 429限流也视为网络降级
        if self._rate_limited_until > 0 and time.time() < self._rate_limited_until:
            return True
        if not self._network_healthy:
            return True
        # 刚恢复后也保持一段冷却期
        if self._network_degraded_since > 0:
            elapsed = time.time() - self._network_degraded_since
            if elapsed < self._network_degraded_cooldown:
                return True
        return False

    # ──────────────────────────────────────────────────────────
    # 多代理源自动切换：代理获取 / 失败标记 / 成功标记
    # 设计原则：代理获取、失败判定、回退策略解耦；回退（切换代理）只在编排层发生。
    # ──────────────────────────────────────────────────────────

    def _get_active_proxy(self) -> Optional[str]:
        """返回当前可用的代理 URL（跳过被禁用的代理，轮询切换）。

        规则：
        - 无代理列表 → 返回 None（直连）
        - 从当前索引开始轮询，返回第一个未被禁用的代理
        - 若全部被禁用：
          * allow_direct_fallback=True → 返回 None（直连兜底）
          * allow_direct_fallback=False → 返回列表第一个代理（强制走代理，让上层重试）
        """
        if not self._proxy_list:
            return None
        n = len(self._proxy_list)
        now = time.time()
        for offset in range(n):
            idx = (self._current_proxy_idx + offset) % n
            proxy_url = self._proxy_list[idx]
            state = self._proxy_states.get(proxy_url, {})
            disabled_until = state.get("disabled_until", 0.0)
            if now >= disabled_until:
                # 冷却期已过（且之前确实被冷却过）：重置失败计数和冷却时间，给代理重新评估的机会
                # disabled_until == 0 表示从未被冷却，不重置 fail_count（让失败计数正常累加）
                if disabled_until > 0 and state.get("fail_count", 0) > 0:
                    state["fail_count"] = 0
                    state["disabled_until"] = 0.0
                # 找到可用代理，更新当前索引
                if idx != self._current_proxy_idx:
                    self._current_proxy_idx = idx
                    self.proxy = proxy_url
                    logger.info(f"Proxy switched to: {proxy_url}")
                return proxy_url
        # 全部被禁用
        if self.allow_direct_fallback:
            return None
        # 不允许直连：返回第一个代理（上层会重试，等待禁用到期或探测恢复）
        first = self._proxy_list[0]
        self._current_proxy_idx = 0
        self.proxy = first
        return first

    @staticmethod
    def _is_connection_refused(exc) -> bool:
        """判断代理连接异常是否为「连接被拒」（目标端口无监听，永久死）而非「超时」（半死）。

        区分意义：无监听的代理（如 Clash 实例未启动）短时间内不会恢复，应直接拉满冷却，
        避免在死代理上反复空转并产生噪音；半死代理（TCP 可连但握手挂起）则维持指数退避，
        等待其自行恢复。异常可能被 urllib3/aiohttp/websockets 多层包装，需沿
        __cause__/__context__ 链下钻到最底层原因。
        """
        if exc is None:
            return False
        cur = exc
        seen = set()
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            if type(cur).__name__ == "ConnectionRefusedError":
                return True
            # 注意：urllib3 的 ProxyError 对超时与拒绝的 message 都是 "Cannot connect to proxy."，
            # 不能据此判断；必须下钻到底层（TimeoutError vs NewConnectionError/ConnectionRefusedError）。
            msg = (str(cur) + " " + repr(cur)).lower()
            if ("connection refused" in msg or "actively refused" in msg or "10061" in msg):
                return True
            # errno.ECONNREFUSED：Windows 10061 / Linux 111
            errno = getattr(cur, "errno", None)
            if errno in (10061, 111):
                return True
            cur = cur.__cause__ or cur.__context__
        return False

    def _mark_proxy_failed(self, proxy_url: Optional[str], hard_fail: bool = False) -> None:
        """标记某个代理失败。连续失败达阈值后禁用该代理并自动切换到下一个。

        仅禁用单个代理，不影响其他代理；不会触发全局代理禁用（消除死亡螺旋）。
        若该代理已处于禁用冷却期，则不再累加失败计数和退避周期（避免 allow_direct_fallback=False
        时强制复用被禁用代理导致退避周期在数秒内飙到上限）。

        hard_fail=True 表示「连接被拒」（目标无监听，永久死），冷却直接拉满 _proxy_max_disable，
        避免在死代理上反复重试；否则维持指数退避（适用于半死/超时类软故障）。
        """
        if not proxy_url or proxy_url not in self._proxy_states:
            return
        state = self._proxy_states[proxy_url]
        now = time.time()
        # 已在冷却期内：不再累加，避免退避周期指数飙升
        if now < state.get("disabled_until", 0.0):
            return
        state["fail_count"] = state.get("fail_count", 0) + 1
        if state["fail_count"] >= self._proxy_fail_threshold:
            state["disable_cycle"] = state.get("disable_cycle", 0) + 1
            if hard_fail:
                backoff = self._proxy_max_disable
            else:
                backoff = min(
                    self._proxy_max_disable,
                    self._proxy_disable_duration * (2 ** (state["disable_cycle"] - 1)),
                )
            state["disabled_until"] = now + backoff
            logger.warning(
                f"Proxy {proxy_url} disabled for {backoff}s "
                f"(cycle={state['disable_cycle']}, fail_count={state['fail_count']}, "
                f"hard_fail={hard_fail}), switching to next proxy"
            )
            # 切换到下一个代理索引
            n = len(self._proxy_list)
            if n > 1:
                self._current_proxy_idx = (self._current_proxy_idx + 1) % n
                self.proxy = self._proxy_list[self._current_proxy_idx]

    def _mark_proxy_success(self, proxy_url: Optional[str]) -> None:
        """标记某个代理成功：重置其失败计数和禁用状态。"""
        if not proxy_url or proxy_url not in self._proxy_states:
            return
        state = self._proxy_states[proxy_url]
        if state.get("fail_count", 0) > 0 or state.get("disabled_until", 0.0) > 0:
            state["fail_count"] = 0
            state["disable_cycle"] = 0
            state["disabled_until"] = 0.0
            logger.info(f"Proxy {proxy_url} health restored (fail_count reset)")

    def _on_network_failure(self):
        """网络断连告警：统一失败计数，连续 N 次失败时写入醒目告警。

        供同步 _make_request 与异步 _async_make_request_inner 的异常分支调用，
        跨请求类型聚合计数，避免每请求一条 ERROR 淹没在日志中而"静默无单"。
        """
        # 撤单过程中产生的网络失败不重复计数，避免撤单触发断连告警形成循环
        if self._disconnect_cancel_in_progress:
            return
        self._consecutive_net_fail += 1
        if self._consecutive_net_fail >= self._net_alert_threshold:
            self._network_healthy = False
            self._network_degraded_since = time.time()
            # 断连后标记：恢复时需撤掉所有挂单，避免旧限价/等位挂单在跳空后以不利价格成交
            self._disconnect_cancel_needed = True
            if not self._net_alert_emitted:
                self._net_alert_emitted = True
                logger.error(
                    f"[NETWORK-DOWN] 连续 {self._consecutive_net_fail} 次网络断连，"
                    f"交易系统已无法访问 OKX，将无法开单/平仓。"
                    f"请检查网络连接与代理 (proxy={self.proxy})。"
                )

    def _on_network_success(self):
        """网络恢复：重置失败计数，并在断连告警已发出时写恢复告警。"""
        if self._consecutive_net_fail > 0 or not self._network_healthy or self._net_alert_emitted:
            if self._net_alert_emitted:
                logger.warning("[NETWORK-RECOVERED] 网络已恢复，交易系统恢复正常开单。")
            self._consecutive_net_fail = 0
            self._net_alert_emitted = False
            self._network_healthy = True
            self._network_success_count = 0

        # 断连后恢复：撤销所有挂单，避免断连期间市场跳空导致旧限价/等位挂单以不利价格成交
        if self._disconnect_cancel_needed:
            self._disconnect_cancel_needed = False
            self._cancel_all_orders_after_disconnect()

    def _cancel_all_orders_after_disconnect(self):
        """断连恢复后撤销普通挂单（等位挂单暴露保护）。

        仅撤销普通限价/等位挂单，不撤销 algo 条件单（TP/SL 保护单）。
        断连恢复不应破坏持仓的保护单体系；TP/SL 由 heartbeat 自动维护。

        用 _disconnect_cancel_in_progress 标志隔离撤单过程中的网络失败，
        避免撤单请求再次触发 _on_network_failure 造成告警抖动/重复撤单。
        """
        if self._disconnect_cancel_in_progress:
            return
        self._disconnect_cancel_in_progress = True
        try:
            cancelled_orders = 0
            errors: list = []
            # 仅撤销普通挂单，不撤销 algo 条件单（TP/SL 保护单）
            try:
                pending = self.get_orders()
                for order in pending:
                    inst_id = order.get("instId", "")
                    ord_id = order.get("ordId", "")
                    if not inst_id or not ord_id:
                        continue
                    if self.cancel_order(inst_id, ord_id) is not None:
                        cancelled_orders += 1
                    else:
                        errors.append(f"撤单失败: {inst_id} {ord_id}")
            except Exception as e:
                errors.append(f"撤销普通挂单异常: {e}")

            logger.warning(
                f"[DISCONNECT-CANCEL] 网络恢复后撤销普通挂单 {cancelled_orders} 单"
                f"（algo 条件单已跳过，保护持仓 TP/SL 不受影响）"
                + (f"，错误: {errors}" if errors else "")
            )
        except Exception as e:
            logger.error(f"[DISCONNECT-CANCEL] 撤单异常: {e}")
        finally:
            self._disconnect_cancel_in_progress = False

    def _probe_proxy_recovery(self) -> bool:
        """代理恢复探测（同步）：代理被禁用期间主动探测当前活跃代理是否恢复。

        用极短超时通过代理请求 OKX 轻量端点 /api/v5/public/time，
        成功则判定代理恢复并重置该代理的禁用状态，失败保持禁用。
        仅在代理被禁用且到达探测间隔时调用，返回 True 表示可切回代理。
        """
        now = time.time()
        if now - self._proxy_last_probe_ts < self._proxy_probe_interval:
            return False
        if self._proxy_probe_in_progress:
            return False
        self._proxy_last_probe_ts = now
        self._proxy_probe_in_progress = True
        try:
            current_proxy = self._get_active_proxy()
            if not current_proxy:
                return False
            url = f"{self.rest_url}{self._proxy_probe_path}"
            resp = self._session.get(
                url,
                headers={"Content-Type": "application/json"},
                timeout=(self._proxy_probe_timeout, self._proxy_probe_timeout),
                proxies={"http": current_proxy, "https": current_proxy},
            )
            if resp.status_code < 400:
                try:
                    data = resp.json()
                except Exception:
                    data = {}
                if data.get("code") == "0" or resp.status_code == 200:
                    self._mark_proxy_success(current_proxy)
                    logger.info(f"[PROXY-RECOVERED] 代理 {current_proxy} 恢复探测成功。")
                    return True
            logger.debug(f"Proxy probe failed: HTTP {resp.status_code}")
            return False
        except Exception as e:
            logger.debug(f"Proxy probe exception: {type(e).__name__}: {e}")
            return False
        finally:
            self._proxy_probe_in_progress = False

    async def _probe_proxy_recovery_async(self) -> bool:
        """代理恢复探测（异步）：使用 aiohttp 临时会话，避免阻塞事件循环。"""
        now = time.time()
        if now - self._proxy_last_probe_ts < self._proxy_probe_interval:
            return False
        if self._proxy_probe_in_progress:
            return False
        self._proxy_last_probe_ts = now
        self._proxy_probe_in_progress = True
        try:
            current_proxy = self._get_active_proxy()
            if not current_proxy:
                return False
            url = f"{self.rest_url}{self._proxy_probe_path}"
            timeout = aiohttp.ClientTimeout(total=self._proxy_probe_timeout)
            async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as s:
                async with s.get(url, proxy=current_proxy) as resp:
                    if resp.status < 400:
                        try:
                            data = await resp.json()
                        except Exception:
                            data = {}
                        if data.get("code") == "0" or resp.status == 200:
                            self._mark_proxy_success(current_proxy)
                            logger.info(f"[PROXY-RECOVERED] 代理 {current_proxy} 恢复探测成功。")
                            return True
            return False
        except Exception as e:
            logger.debug(f"Async proxy probe exception: {type(e).__name__}: {e}")
            return False
        finally:
            self._proxy_probe_in_progress = False

    def _handle_429_rate_limit(self):
        """P5-1: 处理429限流，全局降级和动态退避"""
        now = time.time()
        # 重置窗口内的429计数
        if now - self._rate_limit_429_reset_ts > 60:
            self._rate_limit_429_count = 0
            self._rate_limit_429_reset_ts = now
        
        self._rate_limit_429_count += 1
        
        # 根据429频率动态设置恢复时间
        if self._rate_limit_429_count <= 2:
            backoff = 10
        elif self._rate_limit_429_count <= 5:
            backoff = 30
            # 减少并发数
            self._async_request_semaphore = asyncio.Semaphore(2)
        else:
            backoff = 60
            self._async_request_semaphore = asyncio.Semaphore(1)
            # 连续多次429，标记为网络降级
            self._network_healthy = False
            self._network_degraded_since = now
            logger.warning(f"P5-1: Network degraded due to {self._rate_limit_429_count} 429s in 60s")
        
        self._rate_limited_until = now + backoff
        logger.debug(f"P5-1: 429 rate limited, backoff={backoff}s, count={self._rate_limit_429_count}")
    
    def _is_rate_limited(self) -> bool:
        """P5-1: 检查是否处于429限流恢复期"""
        if self._rate_limited_until > 0 and time.time() < self._rate_limited_until:
            return True
        return False
    
    def _should_skip_non_critical_request(self) -> bool:
        """P27: 网络降级时是否应跳过非关键请求"""
        if self.is_network_degraded:
            # 随机跳过50%的非关键请求以减轻负载
            return random.random() < 0.5
        return False

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """安全地将值转换为float，处理空字符串和None
        
        OKX API返回的数值字段为字符串格式，当值为空时返回""而非"0"，
        直接float("")会抛出ValueError。
        """
        if value is None or value == "":
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _handle_rate_limit(self):
        """处理限流：标记当前密钥并切换，延长退避时间（异步安全）"""
        if self._api_pool and self._api_pool.count > 1:
            current_key = self.api_key
            backoff_seconds = self._get_rate_limit_backoff()
            self._api_pool.mark_rate_limited(current_key, duration_seconds=backoff_seconds)
            keys = self._api_pool.get_current()
            self.api_key = keys["api_key"]
            self.secret_key = keys["secret_key"]
            self.passphrase = keys["passphrase"]
            logger.warning(f"触发限流，切换API密钥: {current_key[:8]}... -> {self.api_key[:8]}... (退避{backoff_seconds}s)")
        else:
            backoff = self._get_rate_limit_backoff()
            logger.warning(f"单密钥模式触发限流，等待{min(backoff, 5)}秒后重试")
            # 尝试使用异步sleep，失败则降级为同步sleep
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    # 在异步上下文中无法直接await，由调用方使用_async_rate_limit_backoff处理
                    logger.warning("Rate limit in async context, caller should use _async_rate_limit_backoff")
                    return
            except RuntimeError:
                pass
            time.sleep(min(backoff, 5))

    def _get_rate_limit_backoff(self) -> int:
        """获取限流退避时间（指数退避，最大60秒）"""
        if not hasattr(self, '_rate_limit_count'):
            self._rate_limit_count = 0
            self._last_rate_limit_time = 0
        
        now = time.time()
        if now - self._last_rate_limit_time > 300:
            self._rate_limit_count = 0
        
        self._rate_limit_count += 1
        self._last_rate_limit_time = now
        
        base = 10
        backoff = min(60, base * (2 ** (self._rate_limit_count - 1)))
        return int(backoff)

    async def _async_rate_limit_backoff(self):
        """异步版本的限流退避（在异步上下文中使用）"""
        backoff = self._get_rate_limit_backoff()
        logger.warning(f"异步限流退避：等待{min(backoff, 5)}秒")
        await asyncio.sleep(min(backoff, 5))

    # P5: API延迟韧性方法
    def _track_latency(self, latency_ms: float):
        """追踪API延迟，维护EMA和采样窗口"""
        if latency_ms <= 0:
            return
        
        self._latency_samples.append(latency_ms)
        
        # EMA平滑
        if self._latency_ema == 0.0:
            self._latency_ema = latency_ms
        else:
            self._latency_ema = (self._latency_ema_alpha * latency_ms +
                                 (1 - self._latency_ema_alpha) * self._latency_ema)
        
        # 延迟熔断检查
        if latency_ms > self._latency_critical_threshold:
            self._latency_critical_count += 1
            if self._latency_critical_count >= self._latency_critical_count_max:
                self._latency_circuit_open = True
                self._latency_circuit_opened_at = time.time()
                logger.error(
                    f"API latency circuit breaker OPEN: "
                    f"EMA={self._latency_ema:.0f}ms, "
                    f"last={latency_ms:.0f}ms, "
                    f"consecutive_critical={self._latency_critical_count}"
                )
        else:
            # 延迟正常，递减计数
            self._latency_critical_count = max(0, self._latency_critical_count - 1)
    
    def _get_adaptive_timeout(self) -> float:
        """P5: 基于EMA延迟计算自适应超时
        
        超时 = max(min_timeout, min(ema * 2, max_timeout))
        确保超时始终 >= 延迟EMA的2倍，但不超过上限
        """
        if self._latency_ema <= 0:
            return float(self._timeout)
        
        # 基于EMA的2倍作为超时，确保有足够余量
        adaptive = max(self._adaptive_timeout_min,
                       min(self._latency_ema * 2 / 1000, self._adaptive_timeout_max))
        return adaptive
    
    def _is_latency_circuit_open(self) -> bool:
        """P5: 检查延迟熔断是否开启
        
        熔断自动恢复条件：
        - 熔断持续时间超过冷却时间
        - 且最近采样延迟低于阈值
        """
        if not self._latency_circuit_open:
            return False
        
        elapsed = time.time() - self._latency_circuit_opened_at
        
        # 冷却时间到期后自动恢复
        if elapsed >= self._latency_circuit_cooldown:
            # 检查最近延迟是否已恢复
            recent_samples = list(self._latency_samples)[-5:]
            if recent_samples:
                avg_recent = sum(recent_samples) / len(recent_samples)
                if avg_recent < self._latency_critical_threshold:
                    self._latency_circuit_open = False
                    self._latency_critical_count = 0
                    logger.info(
                        f"API latency circuit breaker CLOSED: "
                        f"recent_avg={avg_recent:.0f}ms, EMA={self._latency_ema:.0f}ms"
                    )
                    return False
            
            # 如果冷却时间超过2倍，即使延迟仍高也强制恢复（避免永久熔断）
            if elapsed >= self._latency_circuit_cooldown * 2:
                self._latency_circuit_open = False
                self._latency_critical_count = 0
                logger.warning(
                    f"API latency circuit breaker FORCE CLOSED after {elapsed:.0f}s "
                    f"(2x cooldown), EMA={self._latency_ema:.0f}ms"
                )
                return False
        
        return self._latency_circuit_open
    
    def get_latency_stats(self) -> Dict[str, Any]:
        """P5: 获取延迟统计信息"""
        return {
            "ema_ms": round(self._latency_ema, 1),
            "adaptive_timeout_s": round(self._get_adaptive_timeout(), 1),
            "circuit_open": self._latency_circuit_open,
            "critical_count": self._latency_critical_count,
            "recent_samples": list(self._latency_samples)[-5:],
            "sample_count": len(self._latency_samples)
        }

    def _request_with_retry(self, method: str, url: str, headers: dict, data: str = None) -> Optional[dict]:
        """带重试的HTTP请求，仅对网络错误和5xx重试，4xx不重试"""
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
            before_sleep=before_sleep_log(logger, "WARNING"),
            reraise=True
        )
        def _do_request():
            current_proxy = self._get_active_proxy()
            _proxies = {"http": current_proxy, "https": current_proxy} if current_proxy else None
            if method == "GET":
                resp = self._session.get(url, headers=headers, timeout=(self._connect_timeout, self._timeout), proxies=_proxies)
            else:
                resp = self._session.post(url, headers=headers, data=data, timeout=(self._connect_timeout, self._timeout), proxies=_proxies)
            # 5xx 服务端错误重试
            if resp.status_code >= 500:
                raise requests.ConnectionError(f"Server error {resp.status_code}")
            if current_proxy:
                self._mark_proxy_success(current_proxy)
            return resp.json()

        try:
            result = _do_request()
            # 检查是否是限流错误 (50011 = Too Many Requests)
            if result and isinstance(result, dict) and result.get("code") == "50011":
                self._handle_rate_limit()
            return result
        except (requests.ConnectionError, requests.Timeout) as e:
            logger.error(f"Request failed after retries: {url} - {e}")
            return None
        except Exception as e:
            logger.error(f"Request error: {url} - {e}")
            return None

    def _generate_signature(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        message = f"{timestamp}{method}{path}{body}"
        signature = hmac.new(self.secret_key.encode(), message.encode(), hashlib.sha256).digest()
        return base64.b64encode(signature).decode()

    def _get_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        # 确保使用当前密钥（密钥池模式下可能已切换）
        self._get_current_keys()
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        signature = self._generate_signature(timestamp, method, path, body)
        headers = {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json"
        }
        # OKX 模拟盘（is_testnet）：REST 请求必须携带该头，否则会打到真实主网账户
        if self.is_testnet:
            headers["x-simulated-trading"] = "1"
        return headers

    def set_risk_callbacks(self, on_api_call=None, on_latency=None):
        """注入风控埋点回调（由scheduler调用，连接RiskGate的L2事中风控）"""
        self._on_api_call = on_api_call
        self._on_latency = on_latency

    # P6: 延迟熔断绕过的关键API路径前缀
    _CRITICAL_API_PREFIXES = (
        "/api/v5/market/",           # 市场数据（策略决策必需）
        "/api/v5/account/positions", # 持仓查询
        "/api/v5/account/balance",   # 账户余额
        "/api/v5/trade/order",       # 交易操作（下单/撤单/改单）
        "/api/v5/public/funding-rate", # 资金费率
        "/api/v5/public/instruments",  # 合约信息
    )

    async def _async_make_request(self, method: str, path: str, body: str = "",
                                   bypass_circuit: bool = False) -> Optional[dict]:
        """异步统一请求方法 - 使用 aiohttp 避免阻塞事件循环
        
        P5: 自适应超时和延迟熔断
        P6: 关键API（市场数据、持仓、账户、交易操作）绕过延迟熔断，
            避免WS断开时连REST回退也被阻断
        """
        # P23: 并发请求信号量控制，防止同时发出过多请求触发429
        async with self._async_request_semaphore:
            return await self._async_make_request_inner(method, path, body, bypass_circuit)
    
    async def _async_make_request_inner(self, method: str, path: str, body: str = "",
                                   bypass_circuit: bool = False) -> Optional[dict]:
        """内部实际HTTP请求执行（信号量已获取）"""
        # P6: 检查是否为关键API路径，自动绕过熔断
        is_critical = bypass_circuit or any(path.startswith(prefix) for prefix in self._CRITICAL_API_PREFIXES)
        
        # P5: 延迟熔断检查（非关键API才受熔断影响）
        if not is_critical and self._is_latency_circuit_open():
            logger.debug(f"Latency circuit breaker open, skipping non-critical request to {path}")
            return {"code": "circuit_open", "data": {}, "msg": "Latency circuit breaker open"}
        
        # P14: WS断开期间降低非关键REST调用频率
        # 当公/私WS都断开时，网络可能不稳定，非关键API调用应降级
        if not is_critical and not self._ws_public_connected and not self._ws_private_connected:
            # 仅允许每30秒最多1次非关键调用，避免网络拥塞导致超时雪崩
            if not hasattr(self, '_ws_down_degrade_ts'):
                self._ws_down_degrade_ts = 0.0
            now = time.time()
            if now - self._ws_down_degrade_ts < 30:
                logger.debug(f"P14: WS disconnected, throttling non-critical request to {path}")
                return None
            self._ws_down_degrade_ts = now
        
        # P5-1: 429限流恢复期间跳过非关键请求
        if not is_critical and self._is_rate_limited():
            logger.debug(f"P5-1: 429 rate limited, skipping non-critical request to {path}")
            return None
        
        # P5: 自适应超时
        adaptive_timeout = self._get_adaptive_timeout()
        
        # 确保 async session 已初始化
        if self._async_session is None or self._async_session.closed:
            # P7: 优化连接池 - 增加连接数上限，添加keep-alive
            connector = aiohttp.TCPConnector(
                limit=50,           # P7: 总连接数50（原20）
                limit_per_host=25,  # P7: 单host连接数25（原10）
                ttl_dns_cache=300,  # P7: DNS缓存5分钟
                force_close=False,  # P7: 启用keep-alive连接复用
                enable_cleanup_closed=True,
            )
            timeout = aiohttp.ClientTimeout(total=adaptive_timeout)
            # P6: aiohttp需要代理配置，否则异步请求直连OKX会失败
            # aiohttp通过环境变量HTTP_PROXY/HTTPS_PROXY使用代理
            session_kwargs = {"connector": connector, "timeout": timeout, "trust_env": True}
            if self.proxy:
                os.environ.setdefault("HTTP_PROXY", self.proxy)
                os.environ.setdefault("HTTPS_PROXY", self.proxy)
                logger.debug(f"P6: aiohttp proxy set via env: {self.proxy}")
            self._async_session = aiohttp.ClientSession(**session_kwargs)
        else:
            # P5: 动态更新超时
            try:
                self._async_session._timeout = aiohttp.ClientTimeout(
                    connect=self._connect_timeout, total=adaptive_timeout
                )
            except Exception:
                pass

        # ── 直连备用 async session（trust_env=False，不走代理），供代理故障时降级 ──
        if self._async_session_direct is None or self._async_session_direct.closed:
            connector_direct = aiohttp.TCPConnector(
                limit=25, limit_per_host=10, ttl_dns_cache=300,
                force_close=False, enable_cleanup_closed=True,
            )
            timeout_direct = aiohttp.ClientTimeout(total=adaptive_timeout)
            self._async_session_direct = aiohttp.ClientSession(
                connector=connector_direct, timeout=timeout_direct, trust_env=False
            )

        # 全局请求频率控制：滑动窗口限流（异步版）
        now = time.time()
        while len(self._rest_request_times) >= self._rest_max_requests_per_window:
            oldest = self._rest_request_times[0]
            if now - oldest < self._rest_rate_limit_window:
                wait_s = min(0.5, self._rest_rate_limit_window - (now - oldest) + 0.05)
                await asyncio.sleep(wait_s)
                now = time.time()
                while self._rest_request_times and now - self._rest_request_times[0] >= self._rest_rate_limit_window:
                    self._rest_request_times.popleft()
            else:
                self._rest_request_times.popleft()
        self._rest_request_times.append(now)

        max_attempts = min(self._api_pool.count if self._api_pool else 1, 3)
        # P29: 网络降级时减少重试次数，避免CPU浪费
        if not self._network_healthy:
            max_attempts = 1  # 网络降级时不做重试，快速失败让缓存兜底
        # P7: 连接重试指数退避
        _consecutive_conn_failures = 0

        for attempt in range(max_attempts):
            # ── 多代理源自动切换：获取当前活跃代理（自动跳过被禁用的代理）──
            current_proxy = self._get_active_proxy()
            use_direct = current_proxy is None
            # ── 代理恢复探测：当前代理被禁用期间主动探测恢复 ──
            # 下单/撤单等延迟敏感操作跳过内联探测，避免阻塞交易核心
            if not use_direct and not path.startswith("/api/v5/trade/order"):
                cur_state = self._proxy_states.get(current_proxy, {})
                if time.time() < cur_state.get("disabled_until", 0.0):
                    if await self._probe_proxy_recovery_async():
                        current_proxy = self._get_active_proxy()
                        use_direct = current_proxy is None
            session = self._async_session_direct if use_direct else self._async_session

            try:
                headers = self._get_headers(method, path, body)
                url = f"{self.rest_url}{path}"

                _req_start = time.time()

                if method == "GET":
                    async with session.get(url, headers=headers, proxy=current_proxy) as resp:
                        _latency_ms = (time.time() - _req_start) * 1000
                        if resp.status == 429:
                            logger.warning(f"HTTP 429 rate limited on {path}, backing off")
                            self._handle_rate_limit()
                            self._handle_429_rate_limit()  # P5-1: 追踪429用于全局降级
                            if attempt < max_attempts - 1:
                                await self._async_rate_limit_backoff()
                                continue
                            return {"code": "429", "data": {}, "msg": "HTTP 429"}
                        if resp.status >= 400:
                            error_body = await resp.text()
                            # 403 且走代理：代理节点可能被 OKX 屏蔽，5 分钟去重告警（直连环境无意义降级）
                            if resp.status == 403 and current_proxy and not use_direct:
                                _now = time.time()
                                if _now - getattr(self, "_last_proxy_403_warn_ts", 0.0) > 300:
                                    self._last_proxy_403_warn_ts = _now
                                    logger.warning(
                                        f"HTTP 403 from {path}: 代理节点 {current_proxy} 可能被 OKX 屏蔽，"
                                        f"建议更换节点或检查代理配置"
                                    )
                            logger.debug(f"HTTP {resp.status} from {path}: {error_body[:200]}")
                            return {"code": str(resp.status), "data": {}, "msg": f"HTTP {resp.status}"}
                        data = await resp.json()
                else:
                    async with session.post(url, headers=headers, data=body, proxy=current_proxy) as resp:
                        _latency_ms = (time.time() - _req_start) * 1000
                        if resp.status == 429:
                            logger.warning(f"HTTP 429 rate limited on {path}, backing off")
                            self._handle_rate_limit()
                            self._handle_429_rate_limit()  # P5-1: 追踪429用于全局降级
                            if attempt < max_attempts - 1:
                                await self._async_rate_limit_backoff()
                                continue
                            return {"code": "429", "data": {}, "msg": "HTTP 429"}
                        if resp.status >= 400:
                            error_body = await resp.text()
                            # 403 且走代理：代理节点可能被 OKX 屏蔽，5 分钟去重告警（直连环境无意义降级）
                            if resp.status == 403 and current_proxy and not use_direct:
                                _now = time.time()
                                if _now - getattr(self, "_last_proxy_403_warn_ts", 0.0) > 300:
                                    self._last_proxy_403_warn_ts = _now
                                    logger.warning(
                                        f"HTTP 403 from {path}: 代理节点 {current_proxy} 可能被 OKX 屏蔽，"
                                        f"建议更换节点或检查代理配置"
                                    )
                            logger.debug(f"HTTP {resp.status} from {path}: {error_body[:200]}")
                            return {"code": str(resp.status), "data": {}, "msg": f"HTTP {resp.status}"}
                        data = await resp.json()

                # 风控埋点
                if self._on_latency:
                    try:
                        self._on_latency(_latency_ms)
                    except Exception:
                        pass
                if self._on_api_call:
                    try:
                        self._on_api_call()
                    except Exception:
                        pass

                # P5: 延迟追踪和自适应
                self._track_latency(_latency_ms)

                # 检查限流
                if data.get("code") == "50011":
                    if attempt < max_attempts - 1:
                        await self._async_rate_limit_backoff()
                        continue
                    else:
                        logger.warning(f"所有API密钥均触发限流: {path}")
                        return data

                # 代理成功：重置该代理的失败计数
                if current_proxy:
                    self._mark_proxy_success(current_proxy)
                self._on_network_success()
                return data
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                self._on_network_failure()
                # ── 代理故障：标记当前代理失败（达阈值后禁用该代理并自动切换下一个）──
                if current_proxy and not use_direct:
                    self._mark_proxy_failed(current_proxy, hard_fail=self._is_connection_refused(e))
                # allow_direct_fallback=True 时立即直连重试一次
                if current_proxy and not use_direct and self.allow_direct_fallback:
                    logger.warning(f"Proxy {current_proxy} failed for {path}: {e}, retrying direct...")
                    try:
                        _direct_session = self._async_session_direct
                        if _direct_session is None or _direct_session.closed:
                            _direct_session = aiohttp.ClientSession(
                                connector=aiohttp.TCPConnector(
                                    limit=25, limit_per_host=10, ttl_dns_cache=300,
                                    force_close=False, enable_cleanup_closed=True,
                                ),
                                timeout=aiohttp.ClientTimeout(total=adaptive_timeout),
                                trust_env=False,
                            )
                            self._async_session_direct = _direct_session
                        if method == "GET":
                            async with _direct_session.get(url, headers=headers) as resp:
                                if resp.status < 400:
                                    data = await resp.json()
                                    if data.get("code") == "0":
                                        self._on_network_success()
                                        return data
                        else:
                            async with _direct_session.post(url, headers=headers, data=body) as resp:
                                if resp.status < 400:
                                    data = await resp.json()
                                    if data.get("code") == "0":
                                        self._on_network_success()
                                        return data
                        logger.warning(f"Async direct connection also failed for {path}")
                    except Exception as direct_e:
                        logger.warning(f"Async direct connection failed for {path}: {direct_e}")
                _consecutive_conn_failures += 1
                # P27: 重试风暴检测 - 防止CPU 100%
                self._retry_storm_count += 1
                now_ts = time.time()
                if now_ts - self._last_retry_storm_warning < 10 and self._retry_storm_count > self._retry_storm_threshold:
                    if now_ts - self._last_retry_storm_warning > 5:
                        logger.warning(f"P27: Retry storm detected ({self._retry_storm_count} retries in <10s), "
                                      f"reducing concurrent requests")
                        self._last_retry_storm_warning = now_ts
                    # 风暴期间减少并发
                    self._async_request_semaphore = asyncio.Semaphore(2)
                    # 跳过非关键请求
                    if random.random() < 0.7:
                        return None
                elif self._retry_storm_count > self._retry_storm_threshold:
                    self._retry_storm_count = 0
                    self._async_request_semaphore = asyncio.Semaphore(5)  # 风暴过去后恢复
                
                logger.debug(f"Async request failed for {path}: {type(e).__name__}: {e}")
                # P5: 超时也计入延迟追踪（视为极端延迟）
                if isinstance(e, asyncio.TimeoutError):
                    self._track_latency(adaptive_timeout * 1000)
                if attempt < max_attempts - 1:
                    # P7: 指数退避重试，P27: 添加随机抖动防止惊群
                    backoff = 0.5 * (2 ** attempt) + random.uniform(0, 0.5)
                    await asyncio.sleep(backoff)
                    continue
                logger.warning(f"Async request exhausted retries for {path}: {type(e).__name__}")
                return None
            except Exception as e:
                logger.error(f"Async request error for {path}: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(0.5)
                    self._rotate_key()
                    continue
                return None

        return None

    def _make_request(self, method: str, path: str, body: str = "") -> Optional[dict]:
        """同步统一请求方法 - 兼容旧代码，新异步代码请使用 _async_make_request"""
        # 全局请求频率控制：滑动窗口限流
        now = time.time()
        while len(self._rest_request_times) >= self._rest_max_requests_per_window:
            oldest = self._rest_request_times[0]
            if now - oldest < self._rest_rate_limit_window:
                # 尝试异步sleep，失败则同步sleep
                try:
                    loop = asyncio.get_running_loop()
                    if loop.is_running():
                        pass  # 在异步上下文中调用了同步方法，已降级为同步sleep（频繁日志已抑制）
                except RuntimeError:
                    pass
                time.sleep(min(0.5, self._rest_rate_limit_window - (now - oldest) + 0.05))
                now = time.time()
                while self._rest_request_times and now - self._rest_request_times[0] >= self._rest_rate_limit_window:
                    self._rest_request_times.popleft()
            else:
                self._rest_request_times.popleft()
        self._rest_request_times.append(now)

        max_attempts = min(self._api_pool.count if self._api_pool else 1, 3)
        # P29: 网络降级时减少重试次数，避免CPU浪费
        if not self._network_healthy:
            max_attempts = 1
        # P5-1: 429限流期间强制max_attempts=1
        if self._is_rate_limited():
            max_attempts = 1
        for attempt in range(max_attempts):
            # ── 多代理源自动切换：获取当前活跃代理（自动跳过被禁用的代理）──
            current_proxy = self._get_active_proxy()
            use_direct = current_proxy is None
            # ── 代理恢复探测：当前代理被禁用期间主动探测恢复 ──
            # 下单/撤单等延迟敏感操作跳过内联探测，避免阻塞交易核心
            if not use_direct and not path.startswith("/api/v5/trade/order"):
                cur_state = self._proxy_states.get(current_proxy, {})
                if time.time() < cur_state.get("disabled_until", 0.0):
                    if self._probe_proxy_recovery():
                        current_proxy = self._get_active_proxy()
                        use_direct = current_proxy is None
            session = self._session_direct if use_direct else self._session
            _proxies = {"http": current_proxy, "https": current_proxy} if current_proxy else None

            try:
                headers = self._get_headers(method, path, body)
                url = f"{self.rest_url}{path}"

                _req_start = time.time()

                if method == "GET":
                    response = session.get(url, headers=headers, timeout=(self._connect_timeout, self._timeout), proxies=_proxies)
                else:
                    response = session.post(url, headers=headers, data=body, timeout=(self._connect_timeout, self._timeout), proxies=_proxies)

                # 代理成功：重置该代理的失败计数
                if current_proxy:
                    self._mark_proxy_success(current_proxy)

                # 风控埋点：记录API调用延迟和频次
                _latency_ms = (time.time() - _req_start) * 1000
                if self._on_latency:
                    try:
                        self._on_latency(_latency_ms)
                    except Exception:
                        pass
                if self._on_api_call:
                    try:
                        self._on_api_call()
                    except Exception:
                        pass

                # 检查 HTTP 状态码，4xx/5xx 直接返回错误信息
                if response.status_code >= 400:
                    error_body = None
                    try:
                        error_body = response.json()
                    except Exception:
                        error_body = response.text[:200]
                    # P5-1: 429限流处理
                    if response.status_code == 429:
                        logger.warning(f"HTTP 429 rate limited on {path} (sync)")
                        self._handle_rate_limit()
                        self._handle_429_rate_limit()
                        if attempt < max_attempts - 1:
                            time.sleep(self._get_rate_limit_backoff())
                            continue
                        return {"code": "429", "data": {}, "msg": "HTTP 429"}
                    logger.debug(f"HTTP {response.status_code} from {path}: {error_body}")
                    return {"code": str(response.status_code), "data": {}, "msg": f"HTTP {response.status_code}"}

                data = response.json()

                # 检查限流
                if data.get("code") == "50011":
                    if attempt < max_attempts - 1:
                        try:
                            loop = asyncio.get_running_loop()
                            if loop.is_running():
                                logger.warning("Rate limited in async context, using async backoff")
                                # Can't await here, fall through to no-sleep approach
                        except RuntimeError:
                            pass
                        self._handle_rate_limit()
                        continue
                    else:
                        logger.warning(f"所有API密钥均触发限流: {path}")
                        return data

                self._on_network_success()
                return data
            except (requests.exceptions.ProxyError, requests.exceptions.ConnectTimeout,
                    requests.exceptions.ConnectionError) as e:
                self._on_network_failure()
                # ── 代理故障：标记当前代理失败（达阈值后禁用该代理并自动切换下一个）──
                if current_proxy and not use_direct:
                    self._mark_proxy_failed(current_proxy, hard_fail=self._is_connection_refused(e))
                # allow_direct_fallback=True 时直连重试一次
                if current_proxy and not use_direct and self.allow_direct_fallback:
                    logger.warning(f"Proxy {current_proxy} failed for {path}: {e}, retrying without proxy...")
                    try:
                        headers = self._get_headers(method, path, body)
                        url = f"{self.rest_url}{path}"
                        if method == "GET":
                            response = self._session_direct.get(url, headers=headers, timeout=(self._connect_timeout, self._timeout))
                        else:
                            response = self._session_direct.post(url, headers=headers, data=body, timeout=(self._connect_timeout, self._timeout))
                        if response.status_code < 400:
                            data = response.json()
                            if data.get("code") == "0":
                                return data
                        logger.warning(f"Direct connection also failed for {path}")
                    except Exception as direct_e:
                        logger.warning(f"Direct connection failed for {path}: {direct_e}")
                if attempt < max_attempts - 1:
                    # 代理/连接错误不切换密钥，避免无意义轮换
                    continue
                raise e
            except Exception as e:
                if attempt < max_attempts - 1:
                    self._rotate_key()
                    continue
                raise e

        return None

    def _is_ticker_server_fresh(self, ticker: Any) -> bool:
        if not isinstance(ticker, dict):
            return False
        try:
            server_ts = float(ticker["ts"])
        except (KeyError, TypeError, ValueError):
            return False
        if server_ts > 1e11:
            server_ts /= 1000.0
        return abs(time.time() - server_ts) <= self._ticker_max_server_age_seconds

    def update_ticker_from_ws(self, symbol: str, data: Dict[str, Any]) -> None:
        """R4: WS tick 数据写入 ticker 缓存，供 get_ticker(_async) 命中缓存跳过 REST。"""
        if not data or not symbol:
            return
        self._ticker_cache[symbol] = {"data": data, "ts": time.time()}

    def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        # P28: Ticker缓存 - 减少网络不稳定时的重复API调用
        ttl = self._ticker_cache_ttl_degraded if not self._network_healthy else self._ticker_cache_ttl_normal
        cache_entry = self._ticker_cache.get(symbol)
        if (
            cache_entry
            and time.time() - cache_entry["ts"] < ttl
            and self._is_ticker_server_fresh(cache_entry.get("data"))
        ):
            return cache_entry["data"]
        
        try:
            path = f"/api/v5/market/ticker?instId={symbol}"
            data = self._make_request("GET", path)
            if data and data["code"] == "0":
                ticker = data["data"][0]
                if self._is_ticker_server_fresh(ticker):
                    self._ticker_cache[symbol] = {"data": ticker, "ts": time.time()}
                    return ticker
                logger.warning(f"Ignoring stale ticker response for {symbol}")
            logger.debug(f"Failed to get ticker for {symbol}: {data.get('msg', '') if data else ''}")
            if cache_entry and self._is_ticker_server_fresh(cache_entry.get("data")):
                return cache_entry["data"]
            return None
        except Exception as e:
            logger.debug(f"Error getting ticker for {symbol}: {e}")
            # P28: 网络异常时返回过期缓存
            if cache_entry and self._is_ticker_server_fresh(cache_entry.get("data")):
                return cache_entry["data"]
            return None

    async def get_ticker_async(self, symbol: str) -> Optional[Dict[str, Any]]:
        """异步获取ticker数据"""
        # P28: Ticker缓存 - 减少网络不稳定时的重复API调用
        ttl = self._ticker_cache_ttl_degraded if not self._network_healthy else self._ticker_cache_ttl_normal
        cache_entry = self._ticker_cache.get(symbol)
        if (
            cache_entry
            and time.time() - cache_entry["ts"] < ttl
            and self._is_ticker_server_fresh(cache_entry.get("data"))
        ):
            return cache_entry["data"]
        
        try:
            path = f"/api/v5/market/ticker?instId={symbol}"
            data = await self._async_make_request("GET", path)
            if data and data["code"] == "0":
                ticker = data["data"][0]
                if self._is_ticker_server_fresh(ticker):
                    self._ticker_cache[symbol] = {"data": ticker, "ts": time.time()}
                    return ticker
                logger.warning(f"Ignoring stale ticker response for {symbol}")
            logger.debug(f"Failed to get ticker for {symbol}: {data.get('msg', '') if data else ''}")
            if cache_entry and self._is_ticker_server_fresh(cache_entry.get("data")):
                return cache_entry["data"]
            return None
        except Exception as e:
            logger.debug(f"Error getting ticker for {symbol}: {e}")
            if cache_entry and self._is_ticker_server_fresh(cache_entry.get("data")):
                return cache_entry["data"]
            return None

    def get_liquidation_orders(self, uly: str = "BTC-USDT", inst_type: str = "SWAP",
                               state: str = "filled") -> Optional[Dict[str, Any]]:
        """获取合约爆仓单（public/liquidation-orders），用于风控爆仓量检测。

        复用 _make_request 的直连/代理降级、限流与超时处理，避免调用方裸
        requests.get 走系统代理（trust_env）导致 ProxyError。返回 OKX 完整响应
        {"code": "0", "data": [...]}；失败返回 None。

        注意：OKX 该接口路径为 /api/v5/public/liquidation-orders（非 market），
        参数用 uly（标的指数，如 BTC-USDT）+ state（filled/unfilled，必填）。
        data 中每项为按 instId 聚合的记录，真正的爆仓明细在 item["details"]。
        """
        path = f"/api/v5/public/liquidation-orders?instType={inst_type}&uly={uly}&state={state}"
        try:
            return self._make_request("GET", path)
        except Exception as e:
            logger.debug(f"Error getting liquidation orders for {uly}: {e}")
            return None

    def get_atr(self, symbol: str, period: int = 14, interval: str = "15m") -> float:
        """计算指定品种的 ATR（平均真实波幅）"""
        klines = self.get_kline(symbol, interval, limit=period + 2)
        if not klines or len(klines) < period:
            return 0.0
        
        tr_values = []
        for i in range(1, len(klines)):
            high = float(klines[i][2])
            low = float(klines[i][3])
            prev_close = float(klines[i - 1][4])
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)
        
        if not tr_values:
            return 0.0
        return sum(tr_values[-period:]) / period

    def get_kline(self, symbol: str, interval: str, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            bar_map = {
                "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                "1H": "1H", "2H": "2H", "4H": "4H", "6H": "6H",
                "8H": "8H", "12H": "12H", "1D": "1D", "1W": "1W",
                "1M": "1M", "1h": "1H", "4h": "4H", "1d": "1D", "5h": "5H", "15h": "15H"
            }
            bar = bar_map.get(interval, interval)
            
            path = f"/api/v5/market/history-candles?instId={symbol}&bar={bar}&limit={limit}"
            data = self._make_request("GET", path)
            if data and data["code"] == "0":
                return data["data"]
            logger.error(f"Failed to get kline for {symbol} (interval={interval}): {data}")
            return []
        except Exception as e:
            logger.error(f"Error getting kline for {symbol}: {e}")
            return []

    async def get_kline_async(self, symbol: str, interval: str, limit: int = 100) -> List[Dict[str, Any]]:
        """异步获取K线数据（失败时自动降级为同步请求）
        
        P4-3: 添加K线缓存，网络降级时优先使用缓存避免重复请求
        P0-118: 增强缓存 — 根据K线周期动态调整TTL，减少高频REST调用
        """
        try:
            bar_map = {
                "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                "1H": "1H", "2H": "2H", "4H": "4H", "6H": "6H",
                "8H": "8H", "12H": "12H", "1D": "1D", "1W": "1W",
                "1M": "1M", "1h": "1H", "4h": "4H", "1d": "1D", "5h": "5H", "15h": "15H"
            }
            bar = bar_map.get(interval, interval)
            cache_key = f"{symbol}:{bar}:{limit}"
            
            # P0-118: 根据K线周期动态调整缓存TTL
            # 短周期（1m-15m）：30秒 — 数据变化快，缓存窗口短
            # 中周期（30m-2H）：60秒 — 平衡新鲜度与API负载
            # 长周期（4H+）：120秒 — 数据变化慢，可长时间缓存
            bar_minutes = {
                "1m": 1, "5m": 5, "15m": 15, "30m": 30,
                "1H": 60, "2H": 120, "4H": 240, "6H": 360,
                "8H": 480, "12H": 720, "1D": 1440, "1W": 10080, "1M": 43200
            }
            minutes = bar_minutes.get(bar, 60)
            if minutes <= 15:
                cache_ttl = 30
            elif minutes <= 120:
                cache_ttl = 60
            else:
                cache_ttl = 120
            
            # P0-118: 优先检查缓存（即使网络正常也使用缓存，减少API调用）
            cached = self._kline_cache.get(cache_key)
            if cached and (time.time() - cached["ts"]) < cache_ttl:
                logger.debug(f"P0-118: Using cached kline for {symbol} {bar} (age={time.time()-cached['ts']:.0f}s, ttl={cache_ttl}s)")
                return cached["data"]
            
            # P4-3: 网络降级时优先使用缓存（更长TTL）
            if not self._network_healthy:
                if cached and (time.time() - cached["ts"]) < 300:  # 5分钟缓存
                    logger.debug(f"P4-3: Using cached kline for {symbol} (network degraded, age={time.time()-cached['ts']:.0f}s)")
                    return cached["data"]
            
            path = f"/api/v5/market/history-candles?instId={symbol}&bar={bar}&limit={limit}"
            data = await self._async_make_request("GET", path)
            if data and data["code"] == "0":
                # P4-3: 更新缓存
                self._kline_cache[cache_key] = {"data": data["data"], "ts": time.time()}
                # 清理过期缓存
                if len(self._kline_cache) > 200:
                    old_keys = sorted(self._kline_cache.keys(), key=lambda k: self._kline_cache[k]["ts"])[:50]
                    for k in old_keys:
                        del self._kline_cache[k]
                return data["data"]
            # 异步失败，检查缓存
            if data is None:
                if cached and (time.time() - cached["ts"]) < 120:  # 2分钟缓存
                    logger.debug(f"P4-3: Using cached kline for {symbol} after async failure (age={time.time()-cached['ts']:.0f}s)")
                    return cached["data"]
                # P4-3: 网络降级时不再降级到同步请求，直接返回缓存或空
                if not self._network_healthy:
                    logger.debug(f"P4-3: Skipping sync fallback for {symbol} (network degraded)")
                    return cached["data"] if cached else []
                logger.debug(f"Async kline failed for {symbol}, falling back to sync request")
                data = self._make_request("GET", path)
                if data and data.get("code") == "0":
                    self._kline_cache[cache_key] = {"data": data["data"], "ts": time.time()}
                    return data["data"]
            # P5-1: 429限流 / 403代理屏蔽 - 跳过同步回退，直接使用缓存
            elif data.get("code") in ("429", "403"):
                if cached:
                    age = time.time() - cached["ts"]
                    # 403（代理被屏蔽）恢复较慢，用更长缓存窗口（10分钟）
                    ttl = 600 if data.get("code") == "403" else 300
                    if age < ttl:
                        logger.debug(
                            f"Using cached kline for {symbol} after {data['code']} "
                            f"(age={age:.0f}s, ttl={ttl}s)"
                        )
                        return cached["data"]
                logger.debug(f"No valid cache for {symbol} after {data.get('code')}, returning empty")
                return cached["data"] if cached else []
            logger.error(f"Failed to get kline for {symbol} (interval={interval}): {data}")
            return []
        except Exception as e:
            logger.error(f"Error getting kline for {symbol}: {e}")
            return []

    def get_klines(self, symbol: str, interval: str, limit: int = 100,
                   after: int = None) -> List[Dict[str, Any]]:
        """获取K线数据（支持 after 分页参数，供 historical_loader 等批量拉取使用）"""
        try:
            bar_map = {
                "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                "1H": "1H", "2H": "2H", "4H": "4H", "6H": "6H",
                "8H": "8H", "12H": "12H", "1D": "1D", "1W": "1W",
                "1M": "1M", "1h": "1H", "4h": "4H", "1d": "1D", "5h": "5H", "15h": "15H"
            }
            bar = bar_map.get(interval, interval)
            path = f"/api/v5/market/history-candles?instId={symbol}&bar={bar}&limit={limit}"
            if after:
                path += f"&after={after}"
            data = self._make_request("GET", path)
            if data and data.get("code") == "0":
                return data["data"]
            logger.error(f"Failed to get klines for {symbol} (interval={interval}, after={after}): {data}")
            return []
        except Exception as e:
            logger.error(f"Error getting klines for {symbol}: {e}")
            return []

    def _funding_rate_ttl(self, funding_rate: Dict[str, Any]) -> float:
        try:
            next_funding_ts = float(funding_rate.get("nextFundingTime", 0))
        except (TypeError, ValueError):
            return self._funding_rate_cache_ttl
        if next_funding_ts > 1e11:
            next_funding_ts /= 1000.0
        seconds_to_settlement = next_funding_ts - time.time()
        if abs(seconds_to_settlement) <= self._funding_rate_settlement_window:
            return self._funding_rate_settlement_ttl
        return self._funding_rate_cache_ttl

    def _cached_funding_rate(self, symbol: str, max_age: float) -> Optional[Dict[str, Any]]:
        entry = self._funding_rate_cache.get(symbol)
        if not entry or time.time() - entry["ts"] > max_age:
            return None
        return entry["data"]

    def get_funding_rate(self, symbol: str) -> Optional[Dict[str, Any]]:
        cache_entry = self._funding_rate_cache.get(symbol)
        if cache_entry and time.time() - cache_entry["ts"] < self._funding_rate_ttl(cache_entry["data"]):
            return cache_entry["data"]
        try:
            path = f"/api/v5/public/funding-rate?instId={symbol}"
            data = self._make_request("GET", path)
            if data and data.get("code") == "0" and data.get("data"):
                funding_rate = data["data"][0]
                self._funding_rate_cache[symbol] = {"data": funding_rate, "ts": time.time()}
                return funding_rate
            logger.debug(f"Failed to get funding rate for {symbol}: {data.get('msg', '') if data else ''}")
        except Exception as e:
            logger.debug(f"Error getting funding rate for {symbol}: {e}")
        return self._cached_funding_rate(symbol, self._funding_rate_stale_fallback_ttl)

    async def get_funding_rate_async(self, symbol: str) -> Optional[Dict[str, Any]]:
        """异步获取资金费率"""
        cache_entry = self._funding_rate_cache.get(symbol)
        if cache_entry and time.time() - cache_entry["ts"] < self._funding_rate_ttl(cache_entry["data"]):
            return cache_entry["data"]
        try:
            path = f"/api/v5/public/funding-rate?instId={symbol}"
            data = await self._async_make_request("GET", path)
            if data and data.get("code") == "0" and data.get("data"):
                funding_rate = data["data"][0]
                self._funding_rate_cache[symbol] = {"data": funding_rate, "ts": time.time()}
                return funding_rate
            logger.debug(f"Failed to get funding rate for {symbol}: {data.get('msg', '') if data else ''}")
        except Exception as e:
            logger.debug(f"Error getting funding rate for {symbol}: {e}")
        return self._cached_funding_rate(symbol, self._funding_rate_stale_fallback_ttl)

    def get_order_book(self, symbol: str, depth: int = 5) -> Optional[Dict[str, Any]]:
        try:
            requested_depth = min(400, max(int(depth), self._orderbook_min_depth))
            path = f"/api/v5/market/books?instId={symbol}&sz={requested_depth}"
            data = self._make_request("GET", path)
            if data and data["code"] == "0":
                if not data.get("data") or not isinstance(data["data"][0], dict):
                    logger.warning(f"Empty order book response for {symbol}")
                    return None
                orderbook = data["data"][0]
                if not self._is_valid_orderbook(orderbook):
                    logger.warning(f"Rejecting incomplete or invalid order book for {symbol}")
                    return None
                return orderbook
            logger.error(f"Failed to get order book for {symbol}: {data}")
            return None
        except Exception as e:
            logger.error(f"Error getting order book: {e}")
            return None

    def _is_valid_orderbook(self, orderbook: Dict[str, Any]) -> bool:
        bids = orderbook.get("bids")
        asks = orderbook.get("asks")
        min_depth = self._orderbook_min_depth
        if not isinstance(bids, list) or not isinstance(asks, list):
            return False
        if len(bids) < min_depth or len(asks) < min_depth:
            return False

        for side in (bids[:min_depth], asks[:min_depth]):
            for level in side:
                if not isinstance(level, (list, tuple)) or len(level) < 2:
                    return False
                try:
                    price = float(level[0])
                    size = float(level[1])
                except (TypeError, ValueError, OverflowError):
                    return False
                if not math.isfinite(price) or not math.isfinite(size) or price <= 0 or size < 0:
                    return False
        return True

    def get_instrument_info(self, symbol: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        # 使用缓存避免每次下单都发HTTP请求（加剧API延迟）
        now = time.time()
        if use_cache and symbol in self._instrument_cache:
            if now - self._instrument_cache_ts.get(symbol, 0) < self._instrument_cache_ttl:
                return self._instrument_cache[symbol]
        try:
            inst_type = "SPOT" if "-" in symbol and "SWAP" not in symbol else "SWAP"
            path = f"/api/v5/public/instruments?instType={inst_type}&instId={symbol}"
            data = self._make_request("GET", path)
            if data and data["code"] == "0" and data["data"]:
                info = data["data"][0]
                self._instrument_cache[symbol] = info
                self._instrument_cache_ts[symbol] = now
                return info
            logger.error(f"Failed to get instrument info for {symbol}: {data}")
            return None
        except Exception as e:
            logger.error(f"Error getting instrument info: {e}")
            return None

    async def get_instrument_info_async(self, symbol: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """异步获取合约信息"""
        now = time.time()
        if use_cache and symbol in self._instrument_cache:
            if now - self._instrument_cache_ts.get(symbol, 0) < self._instrument_cache_ttl:
                return self._instrument_cache[symbol]
        try:
            inst_type = "SPOT" if "-" in symbol and "SWAP" not in symbol else "SWAP"
            path = f"/api/v5/public/instruments?instType={inst_type}&instId={symbol}"
            data = await self._async_make_request("GET", path)
            if data and data["code"] == "0" and data["data"]:
                info = data["data"][0]
                self._instrument_cache[symbol] = info
                self._instrument_cache_ts[symbol] = now
                return info
            logger.error(f"Failed to get instrument info for {symbol}: {data}")
            return None
        except Exception as e:
            logger.error(f"Error getting instrument info: {e}")
            return None

    def coin_to_contracts(self, symbol: str, coin_qty: float) -> float:
        """将策略的币数转换为合约张数（按 ctVal 合约面值换算）"""
        info = self.get_instrument_info(symbol)
        if not info:
            return coin_qty
        ct_val = float(info.get("ctVal", "1"))
        if ct_val <= 0:
            return coin_qty
        return coin_qty / ct_val

    def contracts_to_coins(self, symbol: str, contract_qty: float) -> float:
        """将合约张数转换为币数（ctVal 合约面值反向换算）
        
        用于从 OKX API 获取的持仓数量（contract_qty 为合约张数）转回币数，
        确保传入 place_order 前数量统一为币数，由 place_order 统一做 coin→contracts 转换。
        """
        info = self.get_instrument_info(symbol)
        if not info:
            return contract_qty
        ct_val = float(info.get("ctVal", "1"))
        if ct_val <= 0:
            return contract_qty
        return contract_qty * ct_val

    def round_quantity_to_lot(self, symbol: str, quantity: float, round_up: bool = False) -> float:
        """按合约 lot size 取整数量
        
        round_up=False (默认, 开仓): 向下取整到 lot_sz 倍数，返回 0 表示数量不足最小单位
        round_up=True (减仓/平仓): 向上取整到 lot_sz 倍数，至少返回 lot_sz（避免残留仓位）
        
        注意：quantity 应为合约张数（非币数），币数需先通过 coin_to_contracts 转换
        """
        info = self.get_instrument_info(symbol)
        if not info:
            return quantity
        try:
            lot_sz = float(info.get("lotSz", "1"))
            if lot_sz <= 0:
                return quantity

            from decimal import Decimal, ROUND_DOWN, ROUND_UP
            d_qty = Decimal(str(quantity))
            d_lot = Decimal(str(lot_sz))

            if round_up:
                # 向上取整到 lot_sz 倍数（减仓用，避免残留小于lot_sz的仓位）
                rounded = (d_qty / d_lot).quantize(Decimal("1"), rounding=ROUND_UP) * d_lot
            else:
                # 向下取整到 lot_sz 倍数（开仓用，避免下单量超过预期导致资金不足）
                rounded = (d_qty / d_lot).quantize(Decimal("1"), rounding=ROUND_DOWN) * d_lot
                # R71: 小账户 Kelly 计算量 < 1 lot 时，保底 1 lot（否则静默丢弃交易）
                if rounded == 0 and quantity > 0:
                    rounded = d_lot

            return float(rounded)
        except Exception as e:
            logger.debug(f"round_quantity_to_lot failed for {symbol}: {e}")
            return quantity

    def round_price_to_tick(self, symbol: str, price: float) -> float:
        """按合约 tick size 取整价格
        
        确保价格与交易所 tick size 对齐，避免 PRICE_PRECISION 验证失败。
        """
        info = self.get_instrument_info(symbol)
        if not info:
            return price
        try:
            tick_sz = float(info.get("tickSz", "0.01"))
            if tick_sz <= 0:
                return price
            
            from decimal import Decimal, ROUND_HALF_UP
            d_price = Decimal(str(price))
            d_tick = Decimal(str(tick_sz))
            rounded = (d_price / d_tick).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * d_tick
            return float(rounded)
        except Exception as e:
            logger.debug(f"round_price_to_tick failed for {symbol}: {e}")
            return price

    def get_positions_checked(self) -> Optional[List[Dict[str, Any]]]:
        """查询持仓并保留请求失败与合法空仓之间的区别。"""
        try:
            path = "/api/v5/account/positions"
            data = self._make_request("GET", path)
            if data and data["code"] == "0":
                positions = data.get("data")
                if isinstance(positions, list):
                    return positions
                logger.error(f"Invalid positions response: {data}")
                return None
            logger.error(f"Failed to get positions: {data}")
            return None
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return None

    def get_positions(self) -> List[Dict[str, Any]]:
        positions = self.get_positions_checked()
        return positions if positions is not None else []

    def _get_position_mgn_mode(self, symbol: str, pos_side: str = "") -> str:
        """查询交易所持仓的实际保证金模式（cross/isolated）。

        系统默认 isolated，但用户手动开的仓可能是 cross 模式；若 reduce-only 单的
        tdMode 与持仓实际 mgnMode 不一致，OKX 会在对应保证金桶内找不到持仓，返回
        51169（"该方向无持仓可平"）。返回 "" 表示未查询到持仓或查询失败。
        """
        try:
            positions = self.get_positions() or []
            for p in positions:
                if p.get("instId") != symbol:
                    continue
                if pos_side and p.get("posSide") != pos_side:
                    continue
                try:
                    if abs(float(p.get("pos", 0) or 0)) > 0:
                        return p.get("mgnMode", "")
                except (ValueError, TypeError):
                    continue
        except Exception:
            pass
        return ""

    async def get_positions_async(self) -> List[Dict[str, Any]]:
        """异步获取持仓，带重试机制和缓存回退"""
        max_retries = 3
        base_delay = 0.5
        for attempt in range(max_retries):
            try:
                path = "/api/v5/account/positions"
                data = await self._async_make_request("GET", path)
                if data and data["code"] == "0":
                    # P25: 缓存成功结果，用于网络异常时回退
                    self._positions_cache = data["data"]
                    self._positions_cache_ts = time.time()
                    # P26: 网络健康追踪 - 成功时恢复计数
                    self._network_success_count += 1
                    if self._network_success_count >= self._network_recovery_threshold and not self._network_healthy:
                        self._network_healthy = True
                        self._network_fail_count = 0
                        logger.info("P26: Network recovered, health restored")
                    return data["data"]
                logger.error(f"Failed to get positions: {data}")
                # 非业务错误才重试
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
            except Exception as e:
                logger.error(f"Error getting positions (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
        
        # P26: 网络健康追踪 - 失败时更新降级状态
        self._network_fail_count += 1
        self._network_success_count = 0
        if self._network_fail_count >= self._network_fail_threshold and self._network_healthy:
            self._network_healthy = False
            self._network_degraded_since = time.time()  # P27: 记录降级开始时间
            logger.warning(f"P26: Network degraded after {self._network_fail_count} consecutive failures")
        
        # P25/P26: 所有重试失败，使用缓存回退
        # 正常状态：15秒TTL；降级状态：300秒TTL
        cache_ttl = self._positions_cache_ttl_degraded if not self._network_healthy else self._positions_cache_ttl_normal
        if self._positions_cache is not None and self._positions_cache:
            cache_age = time.time() - self._positions_cache_ts
            if cache_age < cache_ttl:
                logger.warning(
                    f"P26: Using cached positions (age={cache_age:.1f}s, ttl={cache_ttl}s, "
                    f"network={'degraded' if not self._network_healthy else 'healthy'})"
                )
                return self._positions_cache
        
        return []

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        try:
            path = "/api/v5/account/balance"
            data = self._make_request("GET", path)
            if data and data["code"] == "0":
                return data["data"][0]
            logger.error(f"Failed to get account info: {data}")
            return None
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None

    async def get_account_info_async(self) -> Optional[Dict[str, Any]]:
        """异步获取账户信息"""
        try:
            path = "/api/v5/account/balance"
            data = await self._async_make_request("GET", path)
            if data and data["code"] == "0":
                return data["data"][0]
            logger.error(f"Failed to get account info: {data}")
            return None
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None

    def get_account_trade_fee(self, inst_type: str = "SWAP") -> Optional[Dict[str, Any]]:
        """查询账户当前交易费率等级与 maker/taker 费率（企业级资费）。

        接口: GET /api/v5/account/trade-fee?instType={inst_type}
        返回归一化结构:
            {
              "level": "Lv1",    # VIP等级（普通用户为 Lv1，VIP1~VIP9）
              "maker": 0.0002,   # 挂单费率
              "taker": 0.0005,   # 吃单费率
              "inst_type": "SWAP",
              "raw": {...}
            }
        查询失败返回 None。
        """
        try:
            path = f"/api/v5/account/trade-fee?instType={inst_type}"
            data = self._make_request("GET", path)
            if data and data.get("code") == "0":
                items = data.get("data") or []
                if not items:
                    return None
                item = items[0]
                maker = self._safe_float(item.get("maker", "0"))
                taker = self._safe_float(item.get("taker", "0"))
                # 合约费率优先取 swap 数组首项（分币种/分组费率）
                swap_list = item.get("swap") or []
                if inst_type.upper() == "SWAP" and swap_list:
                    maker = self._safe_float(swap_list[0].get("maker", str(maker)))
                    taker = self._safe_float(swap_list[0].get("taker", str(taker)))
                return {
                    "level": item.get("level", ""),
                    "maker": maker,
                    "taker": taker,
                    "inst_type": inst_type.upper(),
                    "raw": item,
                }
            logger.debug(f"Failed to get account trade fee: {data}")
            return None
        except Exception as e:
            logger.debug(f"Error getting account trade fee: {e}")
            return None

    def get_spot_balance(self, currency: str) -> Optional[Dict[str, Any]]:
        try:
            path = "/api/v5/account/balance"
            data = self._make_request("GET", path)
            if data and data["code"] == "0":
                for bal in data["data"]:
                    for detail in bal.get("details", []):
                        if detail.get("ccy") == currency:
                            return {
                                "currency": detail.get("ccy"),
                                "available": self._safe_float(detail.get("availBal", "0")),
                                "frozen": self._safe_float(detail.get("frozenBal", "0")),
                                "total": self._safe_float(detail.get("bal", "0"))
                            }
                return {"currency": currency, "available": 0, "frozen": 0, "total": 0}
            logger.error(f"Failed to get spot balance: {data}")
            return None
        except Exception as e:
            logger.error(f"Error getting spot balance for {currency}: {e}")
            return None

    def get_funding_balance(self, currency: str) -> Optional[Dict[str, Any]]:
        """查询资金账户（funding account）余额。"""
        try:
            path = f"/api/v5/asset/balances?ccy={currency}"
            data = self._make_request("GET", path)
            if data and data.get("code") == "0":
                for bal in data.get("data", []):
                    if bal.get("ccy") == currency:
                        return {
                            "currency": bal.get("ccy"),
                            "available": self._safe_float(bal.get("availBal", "0")),
                            "frozen": self._safe_float(bal.get("frozenBal", "0")),
                            "total": self._safe_float(bal.get("bal", "0"))
                        }
                return {"currency": currency, "available": 0, "frozen": 0, "total": 0}
            logger.debug(f"Failed to get funding balance: {data}")
            return None
        except Exception as e:
            logger.debug(f"Error getting funding balance for {currency}: {e}")
            return None

    def transfer_to_trading(self, currency: str, amount: float) -> bool:
        """Transfer funds from funding account to trading account"""
        try:
            # 划转前检查资金账户可用余额，避免统一账户模式（资金池共享）下无意义的 58350
            funding = self.get_funding_balance(currency)
            if funding is not None:
                avail = funding.get("available", 0)
                if avail <= 0:
                    logger.debug(
                        f"Skip transfer {amount} {currency}: funding account has no available balance"
                    )
                    return False
                # 可用余额不足时，仅划转实际可用部分，避免 58350
                if avail < amount:
                    amount = round(avail, 8)

            path = "/api/v5/asset/transfer"
            body = {
                "ccy": currency,
                "amt": str(amount),
                "from": "6",   # 6 = funding account
                "to": "18",    # 18 = trading account
                "type": "0"    # 0 = within account
            }
            data = self._make_request("POST", path, json.dumps(body))
            if data and data.get("code") == "0":
                logger.info(f"Transferred {amount} {currency} from funding to trading account")
                return True
            # 58350 余额不足属预期情况（统一账户/资金已划转），降级为 debug
            if data and str(data.get("code")) == "58350":
                logger.debug(f"Transfer skipped: insufficient funding balance for {currency}")
            else:
                logger.error(f"Transfer failed: {data}")
            return False
        except Exception as e:
            logger.error(f"Error transferring {currency}: {e}")
            return False

    def place_order(self, symbol: str, side: str, order_type: str, quantity: float,
                   price: float = None, leverage: int = 1, stop_price: float = None,
                   reduce_only: bool = False, pos_side: str = None,
                   conditional_type: str = "stop_loss",
                   clOrdId: str = "",
                   extra_params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        try:
            # 判断是否为现货交易（不包含-SWAP后缀）
            is_spot = "-SWAP" not in symbol

            if order_type in ("conditional", "move_stop"):
                return self._place_conditional_order(symbol, side, quantity, price,
                                                    leverage, stop_price, reduce_only, pos_side,
                                                    conditional_type, order_type, extra_params,
                                                    clOrdId)

            # 合约交易：币数→合约张数转换（策略信号默认传入币数）
            if not is_spot:
                quantity = self.coin_to_contracts(symbol, quantity)

            # 按合约 lot size 取整数量（开仓向下取整，减仓向上取整避免残留仓位）
            rounded_qty = self.round_quantity_to_lot(symbol, quantity, round_up=reduce_only)
            if rounded_qty <= 0:
                logger.warning(
                    f"Skipping order for {symbol}: quantity {quantity} below lot size "
                    f"(reduce_only={reduce_only})"
                )
                return None
            # 减仓时取整后的数量不能超过持仓量（reduce_only=True时OKX会自动截断，但提前检查避免误下单）
            quantity = rounded_qty

            path = "/api/v5/trade/order"
            body = {
                "instId": symbol,
                "side": side,
                "ordType": order_type,
                "sz": str(quantity)
            }

            # post_only 单必须有价格，否则 OKX 拒绝；前置校验避免发出无效单
            if order_type == "post_only" and not price:
                logger.error(f"post_only order for {symbol} requires price, refusing")
                return {"_failed": True, "sCode": "param_error", "sMsg": "post_only requires price"}

            if is_spot:
                body["tdMode"] = "cash"
            else:
                body["tdMode"] = "isolated"
                body["lever"] = str(leverage)

            if clOrdId:
                body["clOrdId"] = clOrdId
            if price:
                body["px"] = str(price)
            if reduce_only:
                body["reduceOnly"] = True
            if pos_side and not is_spot:
                body["posSide"] = pos_side
            elif not is_spot:
                # 合约交易必须指定 posSide（包括 reduce_only 平仓单）
                # reduce_only=True 时：sell=平多→posSide=long, buy=平空→posSide=short
                if reduce_only:
                    body["posSide"] = "long" if side == "sell" else "short"
                else:
                    body["posSide"] = "long" if side == "buy" else "short"

            # P-fix: 平仓/减仓单必须匹配持仓实际保证金模式（cross/isolated）。
            # 用户手动开的仓可能是 cross 模式，而系统默认 isolated；模式不一致会导致
            # OKX 在 isolated 桶内找不到持仓，返回 51169。
            if not is_spot and reduce_only:
                actual_mgn_mode = self._get_position_mgn_mode(symbol, body.get("posSide", ""))
                if actual_mgn_mode in ("cross", "isolated"):
                    body["tdMode"] = actual_mgn_mode

            body_str = json.dumps(body)
            logger.debug(f"place_order request body: {body_str}")
            data = self._make_request("POST", path, body_str)
            if data is None:
                return {"_failed": True, "sCode": "0", "sMsg": "Network error"}
            if data["code"] == "0":
                return data["data"][0]
            # 失败时返回包含 sCode 的字典，便于上层判断错误类型（51008资金不足等）
            logger.error(f"Failed to place order: {data}")
            err_item = (data.get("data") or [{}])[0] if isinstance(data.get("data"), list) else {}
            return {
                "sCode": err_item.get("sCode", data.get("code", "")),
                "sMsg": err_item.get("sMsg", data.get("msg", "")),
                "code": data.get("code", ""),
                "_failed": True
            }
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {"sCode": "exception", "sMsg": str(e), "_failed": True}

    def place_batch_orders(self, order_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量下单 — /api/v5/trade/batch-orders，最多 20 单/批。

        每个 order 必须是已预处理好的请求体（instId/side/ordType/sz/tdMode/posSide 等），
        与 place_order 内部构建的 body 格式一致。
        返回与 order_list 等长的结果列表，每项包含 ordId/sCode/sMsg 或 _failed 标记。
        """
        if not order_list:
            return []
        if len(order_list) > 20:
            logger.warning(f"place_batch_orders: {len(order_list)} orders exceeds max 20, truncating")
            order_list = order_list[:20]

        try:
            path = "/api/v5/trade/batch-orders"
            body_str = json.dumps({"orderList": order_list})
            logger.debug(f"place_batch_orders request: {len(order_list)} orders")
            data = self._make_request("POST", path, body_str)
            if data is None:
                return [{"_failed": True, "sCode": "0", "sMsg": "Network error"}] * len(order_list)

            if data["code"] == "0":
                results = data.get("data", [])
                # OKX 返回的 data 顺序与请求 orderList 一致
                return results if len(results) == len(order_list) else results + [
                    {"_failed": True, "sCode": "missing", "sMsg": "Result not returned"}
                ] * (len(order_list) - len(results))

            # 整批失败
            logger.error(f"Batch order failed: code={data.get('code')} msg={data.get('msg')}")
            return [{"_failed": True, "sCode": data.get("code", ""), "sMsg": data.get("msg", "")}] * len(order_list)
        except Exception as e:
            logger.error(f"Error in place_batch_orders: {e}")
            return [{"sCode": "exception", "sMsg": str(e), "_failed": True}] * len(order_list)

    def close_position(self, symbol: str, pos_side: str = "net") -> Dict[str, Any]:
        """平仓单个持仓（市价 reduce_only）

        根据 symbol 与 pos_side 定位持仓，用反向市价单 + reduce_only=True 平仓。
        pos_side 为空或 "net" 时，自动匹配该 symbol 下第一个非零持仓。
        """
        try:
            positions = self.get_positions() or []
            if not positions:
                return {"success": False, "error": "未获取到持仓数据"}

            target = None
            for p in positions:
                if p.get("instId") != symbol:
                    continue
                pos_qty = float(p.get("pos", 0) or 0)
                if pos_qty == 0:
                    continue
                p_side = p.get("posSide", "net")
                if pos_side and pos_side != "net" and p_side != pos_side:
                    continue
                target = p
                break

            if target is None:
                return {"success": False, "error": f"未找到 {symbol} {pos_side} 的持仓"}

            pos_qty = float(target.get("pos", 0) or 0)
            p_side = target.get("posSide", "net")

            # 平仓方向：多头卖、空头买；net 模式按持仓正负判断
            if p_side == "long":
                side, close_pos_side = "sell", "long"
            elif p_side == "short":
                side, close_pos_side = "buy", "short"
            else:
                side = "sell" if pos_qty > 0 else "buy"
                close_pos_side = "net"

            # 合约张数 → 币数（place_order 内部再统一转回合约张数）
            qty_coins = self.contracts_to_coins(symbol, abs(pos_qty))

            result = self.place_order(
                symbol=symbol,
                side=side,
                order_type="market",
                quantity=qty_coins,
                reduce_only=True,
                pos_side=close_pos_side,
            )

            if result is None:
                return {"success": False, "error": "下单失败：数量不足或网络错误"}
            if result.get("_failed"):
                return {
                    "success": False,
                    "error": f"{result.get('sCode', '')}: {result.get('sMsg', '下单失败')}",
                }
            return {
                "success": True,
                "symbol": symbol,
                "side": side,
                "pos_side": close_pos_side,
                "order": result,
            }
        except Exception as e:
            logger.error(f"close_position error for {symbol} {pos_side}: {e}")
            return {"success": False, "error": str(e)}

    def validate_conditional_price(self, symbol: str, order_type: str, trigger_price: float,
                                   pos_side: str = None, side: str = None) -> Optional[float]:
        """校验条件单触发价方向，返回触发价；方向错误时返回 None 拒绝下单。

        OKX 会以 sCode 51277 拒绝方向错误的止盈/止损价，例如：
        - long 持仓止盈价必须高于最新价，止损价必须低于最新价
        - short 持仓止盈价必须低于最新价，止损价必须高于最新价
        无法获取最新价或无法判定方向时不做拦截（返回原价），交由交易所决定。
        """
        if not trigger_price or float(trigger_price) <= 0:
            return None

        ticker = self.get_ticker(symbol)
        if not ticker:
            return trigger_price
        try:
            last = float(ticker.get("last", 0) or 0)
        except (ValueError, TypeError):
            return trigger_price
        if last <= 0:
            return trigger_price

        # 判定持仓方向：pos_side 优先；否则按平仓 side 推导（sell平多 / buy平空）
        if pos_side and pos_side in ("long", "short"):
            is_long_position = (pos_side == "long")
        elif side:
            is_long_position = (side == "sell")
        else:
            return trigger_price

        if order_type == "take_profit":
            valid = (trigger_price > last) if is_long_position else (trigger_price < last)
        else:  # stop_loss / move_stop 均按止损方向校验
            valid = (trigger_price < last) if is_long_position else (trigger_price > last)

        if not valid:
            logger.warning(
                f"Reject conditional {order_type} price {trigger_price} for {symbol} "
                f"(pos_side={pos_side}, side={side}, last={last}): would be rejected as sCode 51277"
            )
            return None
        return trigger_price

    def _place_conditional_order(self, symbol: str, side: str, quantity: float,
                                price: float = None, leverage: int = 1, stop_price: float = None,
                                reduce_only: bool = False, pos_side: str = None,
                                order_type: str = "stop_loss",
                                algo_type: str = "conditional",
                                extra_params: Dict[str, Any] = None,
                                clOrdId: str = "") -> Optional[Dict[str, Any]]:
        try:
            is_spot = "-SWAP" not in symbol

            if is_spot:
                logger.warning(f"Conditional orders are not supported for spot trading: {symbol}")
                return None

            # 账本一致性（修复4）：条件单与普通单统一「币数→合约张数」转换，再按 lot 取整
            quantity = self.coin_to_contracts(symbol, quantity)

            # 条件单同样需要按 lot size 取整。
            # 修复：条件单（TP/SL）为 reduce_only，数量必须 <= 实际持仓。开仓单向下取整（round_up=False），
            # 若条件单向上取整（round_up=reduce_only=True），当原始数量非 lot_sz 整数倍（如 0.755 张）时，
            # 开仓成交 0.75 张而 SL 挂 0.76 张，超挂 0.01 张。统一向下取整与开仓一致，避免超挂。
            rounded_qty = self.round_quantity_to_lot(symbol, quantity, round_up=False)
            if rounded_qty <= 0:
                logger.warning(
                    f"Skipping conditional order for {symbol}: quantity {quantity} below lot size"
                )
                return None
            quantity = rounded_qty

            path = "/api/v5/trade/order-algo"
            body = {
                "instId": symbol,
                "tdMode": "isolated",
                "side": side,
                "ordType": algo_type if algo_type else "conditional",
                "sz": str(quantity),
                "lever": str(leverage),
                "triggerType": "last"
            }

            # move_stop 类型：使用 callbackRatio 作为回调幅度
            if algo_type == "move_stop":
                trigger_price = stop_price if stop_price else price
                if trigger_price:
                    body["slTriggerPx"] = str(trigger_price)
                    body["slOrdPx"] = "-1"  # 市价
                if extra_params and "callbackRatio" in extra_params:
                    body["callbackRatio"] = extra_params["callbackRatio"]
                body["slTriggerPx"] = body.get("slTriggerPx", "0")
            else:
                trigger_price = stop_price if stop_price else price
                if trigger_price:
                    trigger_price = self.validate_conditional_price(
                        symbol, order_type, trigger_price, pos_side=pos_side, side=side
                    )
                    if trigger_price is None:
                        return None
                    if order_type == "take_profit":
                        body["tpTriggerPx"] = str(trigger_price)
                        # 触发后使用市价单（-1）确保一定成交，避免限价单在价格反弹时挂不上导致止盈失效
                        # 根因：tpOrdPx=tpTriggerPx（限价）时，触发后价格反弹会导致限价平仓单不成交，
                        # 最终过期被撤销，出现"止盈被撤销、止损一直挂着"的不对称失效
                        body["tpOrdPx"] = "-1"
                    else:
                        body["slTriggerPx"] = str(trigger_price)
                        # 止损同样使用市价单，避免触发后限价单不成交（比止盈更危险：止损不成交=裸仓暴露）
                        body["slOrdPx"] = "-1"

            if reduce_only:
                body["reduceOnly"] = True
            if pos_side:
                body["posSide"] = pos_side
            # P-fix: 平仓/减仓条件单同样匹配持仓实际保证金模式，避免 51169
            if reduce_only:
                effective_pos_side = pos_side or ("long" if side == "sell" else "short")
                actual_mgn_mode = self._get_position_mgn_mode(symbol, effective_pos_side)
                if actual_mgn_mode in ("cross", "isolated"):
                    body["tdMode"] = actual_mgn_mode
            if clOrdId:
                # 算法单幂等：客户端订单 ID（/api/v5/trade/order-algo 使用 algoClOrdId）
                body["algoClOrdId"] = clOrdId

            # 合并额外参数（允许调用方覆盖或扩展 body）
            if extra_params:
                for k, v in extra_params.items():
                    if k not in ("callbackRatio",):  # 已处理的字段跳过
                        body[k] = v

            body_str = json.dumps(body)
            logger.debug(f"place_conditional_order request body: {body_str}")
            data = self._make_request("POST", path, body_str)
            if data is None:
                return {"_failed": True, "sCode": "0", "sMsg": "Network error"}
            if data["code"] == "0":
                return data["data"][0]
            # 失败时返回包含 sCode/sMsg 的字典，便于上层区分错误类型（网络/限流可重试，参数错误不重试）
            logger.error(f"Failed to place conditional order: {data}")
            err_item = (data.get("data") or [{}])[0] if isinstance(data.get("data"), list) else {}
            return {
                "sCode": err_item.get("sCode", data.get("code", "")),
                "sMsg": err_item.get("sMsg", data.get("msg", "")),
                "code": data.get("code", ""),
                "_failed": True
            }
        except Exception as e:
            logger.error(f"Error placing conditional order: {e}")
            return {"sCode": "exception", "sMsg": str(e), "_failed": True}

    def cancel_order(self, symbol: str, order_id: str) -> Optional[Dict[str, Any]]:
        try:
            path = "/api/v5/trade/cancel-order"
            body = {"instId": symbol, "ordId": order_id}
            body_str = json.dumps(body)
            data = self._make_request("POST", path, body_str)
            if data and data["code"] == "0":
                return data["data"][0]
            logger.error(f"Failed to cancel order: {data}")
            return None
        except Exception as e:
            logger.error(f"Error canceling order: {e}")
            return None

    def cancel_all_orders(self, inst_type: str = "SWAP") -> Dict[str, Any]:
        """撤销所有挂单（普通挂单 + 算法/条件单），供 Dashboard 干预使用"""
        cancelled_orders = 0
        cancelled_algos = 0
        errors: List[str] = []

        # 1. 普通挂单
        try:
            pending = self.get_orders(inst_type=inst_type)
            for order in pending:
                inst_id = order.get("instId", "")
                ord_id = order.get("ordId", "")
                if not inst_id or not ord_id:
                    continue
                if self.cancel_order(inst_id, ord_id) is not None:
                    cancelled_orders += 1
                else:
                    errors.append(f"撤单失败: {inst_id} {ord_id}")
        except Exception as e:
            errors.append(f"撤销普通挂单异常: {e}")

        # 2. 算法/条件单（止盈止损/计划委托等）
        try:
            for ord_type in ("conditional", "oco", "trigger", "move_order_stop"):
                algos = self.get_algo_orders(ord_type=ord_type) or []
                for algo in algos:
                    inst_id = algo.get("instId", "")
                    algo_id = algo.get("algoId", "")
                    if not inst_id or not algo_id:
                        continue
                    if self.cancel_algo_order(inst_id, algo_id) is not None:
                        cancelled_algos += 1
                    else:
                        errors.append(f"撤销条件单失败: {inst_id} {algo_id}")
        except Exception as e:
            errors.append(f"撤销条件单异常: {e}")

        return {
            "cancelled_orders": cancelled_orders,
            "cancelled_algos": cancelled_algos,
            "errors": errors,
        }

    def get_order(self, symbol: str, order_id: str) -> Optional[Dict[str, Any]]:
        try:
            path = f"/api/v5/trade/order?instId={symbol}&ordId={order_id}"
            data = self._make_request("GET", path)
            if data and data["code"] == "0":
                return data["data"][0]
            logger.error(f"Failed to get order: {data}")
            return None
        except Exception as e:
            logger.error(f"Error getting order: {e}")
            return None

    def get_orders(self, inst_type: str = "SWAP") -> List[Dict[str, Any]]:
        try:
            path = f"/api/v5/trade/orders-pending?instType={inst_type}"
            data = self._make_request("GET", path)
            if data and data["code"] == "0":
                return data["data"]
            logger.error(f"Failed to get pending orders: {data}")
            return []
        except Exception as e:
            logger.error(f"Error getting pending orders: {e}")
            return []

    def get_order_history(self, limit: int = 50, state: str = "filled") -> List[Dict[str, Any]]:
        try:
            params = [f"instType=SWAP", f"limit={limit}"]
            # state 为空表示不过滤状态，返回全部（含 partially_filled/canceled 带部分成交），
            # 用于 fill 回执兜底覆盖部分平仓场景；默认仅拉已成交，保持既有调用方语义。
            if state:
                params.append(f"state={state}")
            path = "/api/v5/trade/orders-history?" + "&".join(params)
            data = self._make_request("GET", path)
            if data and data["code"] == "0":
                return data["data"]
            logger.error(f"Failed to get order history: {data}")
            return []
        except Exception as e:
            logger.error(f"Error getting order history: {e}")
            return []

    def get_bills(self, symbol: str = None, bill_type: str = None, limit: int = 50,
                  before: str = None, after: str = None) -> List[Dict[str, Any]]:
        """获取账户账单（成交记录），用于PnL对账
        bill_type: 1=所有, 2=开仓, 3=平仓, 4=强制平仓
        before: 此时间戳（毫秒）之前的账单（用于向前翻页）
        after: 此时间戳（毫秒）之后的账单（用于向后翻页）
        limit: 最大100
        """
        try:
            params = [f"instType=SWAP", f"limit={min(limit, 100)}"]
            if symbol:
                params.append(f"instId={symbol}")
            if bill_type:
                params.append(f"type={bill_type}")
            if before:
                params.append(f"before={before}")
            if after:
                params.append(f"after={after}")
            path = f"/api/v5/account/bills?" + "&".join(params)
            data = self._make_request("GET", path)
            if data and data["code"] == "0":
                return data["data"]
            logger.error(f"Failed to get bills: {data}")
            return []
        except Exception as e:
            logger.error(f"Error getting bills: {e}")
            return []

    def get_all_bills_paginated(self, symbol: str = None, bill_type: str = "2",
                                earliest_ts_ms: str = None, max_pages: int = 10) -> List[Dict[str, Any]]:
        """分页拉取所有账单（向前翻页直到没有更早的数据）
        earliest_ts_ms: 拉取到这个时间戳为止（毫秒），None表示拉取所有可访问的
        max_pages: 最大翻页次数，防止无限循环
        """
        all_bills = []
        before = None
        for page in range(max_pages):
            bills = self.get_bills(symbol=symbol, bill_type=bill_type, limit=100, before=before)
            if not bills:
                break
            all_bills.extend(bills)
            # 获取本页最早的时间戳作为下一次查询的before
            oldest_ts = min(b.get("ts", "0") for b in bills)
            # 检查是否到达指定的最早时间
            if earliest_ts_ms and oldest_ts <= earliest_ts_ms:
                break
            # 如果返回不足100条，说明没有更多数据
            if len(bills) < 100:
                break
            # 下一页从最早时间戳之前开始
            before = oldest_ts
        return all_bills

    def get_pnl_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """获取已实现盈亏历史，用于对账"""
        try:
            path = f"/api/v5/account/pnl?days={days}"
            data = self._make_request("GET", path)
            if data and data["code"] == "0":
                return data["data"]
            logger.error(f"Failed to get pnl history: {data}")
            return []
        except Exception as e:
            logger.error(f"Error getting pnl history: {e}")
            return []

    def cancel_algo_order(self, symbol: str, algo_id: str) -> Optional[Dict[str, Any]]:
        """取消算法/条件单（止盈止损单），使用 cancel-algos 接口"""
        try:
            path = "/api/v5/trade/cancel-algos"
            body = [{"instId": symbol, "algoId": algo_id}]
            body_str = json.dumps(body)
            data = self._make_request("POST", path, body_str)
            if data and data["code"] == "0":
                return data["data"][0] if data.get("data") else {"success": True}
            # P1: 51400 幂等识别——订单已成交/已撤销/不存在 = 撤销目标已达成，视为成功
            # 避免把「本地残留单 vs 交易所已自动撤销」的良性竞态误报为 ERROR，掩盖真实撤单失败
            if self._is_idempotent_cancel(data):
                logger.debug(f"Algo order already gone (idempotent success): {symbol} {algo_id}")
                return {"success": True, "idempotent": True}
            logger.error(f"Failed to cancel algo order: {data}")
            return None
        except Exception as e:
            logger.error(f"Error canceling algo order: {e}")
            return None

    @staticmethod
    def _is_idempotent_cancel(data: Dict[str, Any]) -> bool:
        """判断撤单返回是否为「订单已不存在」的幂等成功语义（sCode=51400）"""
        try:
            items = data.get("data") or []
            if not items:
                return False
            return str(items[0].get("sCode", "")) == "51400"
        except Exception:
            return False

    def get_algo_orders(self, symbol: str = None, ord_type: str = "conditional") -> Optional[List[Dict[str, Any]]]:
        """获取条件单列表"""
        try:
            path = "/api/v5/trade/orders-algo-pending"
            params = {"ordType": ord_type}
            if symbol:
                params["instId"] = symbol
            
            params_str = "&".join([f"{k}={v}" for k, v in params.items()])
            data = self._make_request("GET", f"{path}?{params_str}")
            
            if data and data["code"] == "0":
                return data.get("data", [])
            logger.error(f"Failed to get algo orders: {data}")
            return None
        except Exception as e:
            logger.error(f"Error getting algo orders: {e}")
            return None

    def get_algo_order_history(self, symbol: str = None, ord_type: str = "conditional",
                               state: str = None, limit: int = 100) -> Optional[List[Dict[str, Any]]]:
        """获取算法/条件单历史（含已触发 effective / 已取消 canceled）。

        与 get_algo_orders（orders-algo-pending）互补：条件单从 pending 消失后，需查
        orders-algo-history 判断是「已触发成交」还是「已取消」。state=effective 表示
        已触发（其 ordId/ordIdList 指向服务端自动生成的平仓单）。

        注意：该端点必须携带 ordType 参数，否则返回 51000。返回 None 表示查询失败
        （网络/API 错误），[] 表示成功但无记录，与 get_algo_orders 语义一致。
        """
        try:
            path = "/api/v5/trade/orders-algo-history"
            params = {"ordType": ord_type, "limit": str(min(limit, 100))}
            if symbol:
                params["instId"] = symbol
            if state:
                params["state"] = state
            params_str = "&".join([f"{k}={v}" for k, v in params.items()])
            data = self._make_request("GET", f"{path}?{params_str}")
            if data and data["code"] == "0":
                return data.get("data", [])
            logger.error(f"Failed to get algo order history: {data}")
            return None
        except Exception as e:
            logger.error(f"Error getting algo order history: {e}")
            return None

    def set_leverage(self, symbol: str, leverage: int, pos_side: str = None) -> Optional[Dict[str, Any]]:
        # 安全检查：过滤无效交易对名称
        if not symbol or symbol == "ALL" or "-" not in symbol:
            logger.warning(f"set_leverage rejected invalid symbol: '{symbol}'")
            return None
        try:
            path = "/api/v5/account/set-leverage"
            body = {
                "instId": symbol,
                "lever": str(leverage),
                "mgnMode": "isolated"
            }
            # long_short_mode 下 set-leverage 必须指定 posSide
            if pos_side:
                body["posSide"] = pos_side
            body_str = json.dumps(body)
            data = self._make_request("POST", path, body_str)
            if data and data["code"] == "0":
                logger.debug(f"Leverage set: {symbol} {leverage}x isolated (posSide={pos_side})")
                return data["data"][0]
            logger.error(f"Failed to set leverage for {symbol}: {data}")
            return None
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
            return None

    async def subscribe_market_data(self, symbols: List[str]):
        """订阅市场数据，记录订阅状态以便重连后自动恢复"""
        self._ws_running = True
        self._subscribed_symbols = symbols
        self._subscribed_public_channels = [
            {"channel": "books", "instId": symbol}
            for symbol in symbols
        ]

        await self._ensure_public_subscription()

    async def _ensure_public_subscription(self):
        """确保公共WebSocket已连接并完成订阅"""
        await self._connect_public_ws()

        if not self._ws_public or not self._is_ws_open(self._ws_public):
            logger.error("Cannot subscribe to market data: WebSocket not connected")
            return

        if not self._subscribed_public_channels:
            return

        subscribe_msg = {
            "op": "subscribe",
            "args": self._subscribed_public_channels
        }

        try:
            msg_str = json.dumps(subscribe_msg)
            logger.info(f"Sending public subscription: {msg_str[:200]}")
            await self._ws_public.send(msg_str)
            logger.info(f"Subscribed to {len(self._subscribed_public_channels)} public market data channels")
        except Exception as e:
            logger.error(f"Failed to send public subscription: {e}")

    async def subscribe_private_data(self):
        """订阅私有数据，记录订阅状态以便重连后自动恢复"""
        self._ws_running = True
        # OKX v5 私有频道必须指定 instType（SWAP=永续合约）
        self._subscribed_private_channels = [
            {"channel": "positions", "instType": "SWAP"},
            {"channel": "orders", "instType": "SWAP"},
            {"channel": "account"}
        ]

        await self._ensure_private_subscription()

    async def _ensure_private_subscription(self):
        """确保私有WebSocket已连接、登录并完成订阅"""
        await self._connect_private_ws()

        if not self._ws_private or not self._is_ws_open(self._ws_private):
            logger.error("Cannot subscribe to private data: WebSocket not connected")
            return

        # 等待login成功，最多等待5秒
        login_waited = 0
        while not self._ws_private_logged_in and login_waited < 5:
            await asyncio.sleep(0.2)
            login_waited += 0.2
        if not self._ws_private_logged_in:
            logger.warning("Private WebSocket login not confirmed, proceeding with subscription anyway")

        if not self._subscribed_private_channels:
            return

        subscribe_msg = {
            "op": "subscribe",
            "args": self._subscribed_private_channels
        }

        try:
            await self._ws_private.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to {len(self._subscribed_private_channels)} private data channels")
        except Exception as e:
            logger.error(f"Failed to send private subscription: {e}")

    async def _connect_public_ws(self):
        if self._ws_public_reconnecting:
            return
        self._ws_public_reconnecting = True
        logger.info(f"Attempting public WebSocket reconnection (attempt {self._ws_public_retry_count + 1 if hasattr(self, '_ws_public_retry_count') else 1})")
        try:
            async with self._ws_reconnect_lock_public:
                if self._ws_public and self._is_ws_open(self._ws_public):
                    self._ws_public_connected = True
                    return

                # 清理旧任务和连接，防止重连后任务泄漏
                self._cancel_ws_tasks(self._ws_public_tasks, "public")
                await self._close_ws(self._ws_public, "public")
                self._ws_public = None

                if not hasattr(self, '_ws_public_retry_count'):
                    self._ws_public_retry_count = 0
                    self._ws_public_last_retry = 0

                now = time.time()
                backoff = self._get_ws_backoff(self._ws_public_retry_count)
                if now - self._ws_public_last_retry < backoff:
                    await asyncio.sleep(backoff - (now - self._ws_public_last_retry))

                self._ws_public_retry_count += 1
                self._ws_public_last_retry = time.time()
                
                # P6: 重试超过20次后重置计数（避免退避一直停留在300秒）
                if self._ws_public_retry_count > 20:
                    self._ws_public_retry_count = 0
                    logger.info("Public WS retry count reset after 20 attempts, starting fresh cycle")

                # ── 多代理源自动切换：获取当前活跃代理（自动跳过被禁用的代理）──
                current_proxy = self._get_active_proxy()
                use_proxy_pub = current_proxy is not None

                try:
                    # P3-1-FIX: 境内环境必须优先代理（OKX 直连被墙）。配置了代理时先走代理，
                    # 代理失败才回退直连（境外环境 proxy=None 时直接走直连）。
                    if use_proxy_pub:
                        try:
                            from websockets_proxy import Proxy, proxy_connect
                            proxy = Proxy.from_url(current_proxy)
                            # P33: 代理连接必须显式超时，否则代理半死（TCP可连但握手挂起）时重连永久阻塞
                            self._ws_public = await proxy_connect(self.ws_public_url, proxy=proxy, ping_interval=25, proxy_conn_timeout=10)
                            self._mark_proxy_success(current_proxy)
                            logger.info(f"Connected to OKX public WebSocket (via proxy {current_proxy})")
                        except ImportError:
                            if self.allow_direct_fallback:
                                logger.error("websockets-proxy not installed, falling back to direct")
                                self._ws_public = await websockets.connect(self.ws_public_url, ping_interval=20,
                                    close_timeout=5, max_size=2**20)
                                logger.info("Connected to OKX public WebSocket (direct)")
                            else:
                                raise
                        except Exception as proxy_e:
                            # 标记当前代理失败（达阈值后禁用该代理并自动切换下一个）
                            self._mark_proxy_failed(current_proxy, hard_fail=self._is_connection_refused(proxy_e))
                            if self.allow_direct_fallback:
                                # P3-1: 代理失败时回退直连
                                logger.warning(f"P3-1: Proxy public WS failed ({proxy_e}), falling back to direct")
                                self._ws_public = await websockets.connect(self.ws_public_url, ping_interval=20,
                                    close_timeout=5, max_size=2**20)
                                logger.info("P3-1: Connected to OKX public WebSocket (direct after proxy fail)")
                            else:
                                logger.warning(f"Public WS proxy connect failed ({proxy_e}), will retry with next proxy")
                                raise
                    else:
                        self._ws_public = await websockets.connect(self.ws_public_url, ping_interval=20,
                            close_timeout=5, max_size=2**20)
                        logger.info("Connected to OKX public WebSocket (direct)")

                    self._ws_public_connected = True
                    self._ws_public_last_pong = time.time()
                    self._ws_public_last_data = time.time()
                    self._ws_public_retry_count = 0

                    # P14: 网络恢复后重置延迟状态，允许系统快速恢复正常
                    self._reset_latency_state()

                    # 启动消息处理、心跳和连接监控任务
                    self._ws_public_tasks.append(asyncio.create_task(self._process_public_ws()))
                    self._ws_public_tasks.append(asyncio.create_task(self._heartbeat_public()))
                    self._ws_public_tasks.append(asyncio.create_task(self._monitor_public_connection()))

                    # 自动恢复订阅
                    await self._ensure_public_subscription()
                except ImportError:
                    logger.error("websockets-proxy not installed, trying without proxy...")
                    try:
                        self._ws_public = await websockets.connect(self.ws_public_url, ping_interval=25, close_timeout=5)
                        self._ws_public_connected = True
                        self._ws_public_last_pong = time.time()
                        self._ws_public_last_data = time.time()
                        self._ws_public_retry_count = 0
                        logger.info("Connected to OKX public WebSocket (without proxy)")
                        self._ws_public_tasks.append(asyncio.create_task(self._process_public_ws()))
                        self._ws_public_tasks.append(asyncio.create_task(self._heartbeat_public()))
                        self._ws_public_tasks.append(asyncio.create_task(self._monitor_public_connection()))
                        await self._ensure_public_subscription()
                    except Exception as e:
                        self._ws_public_connected = False
                        logger.error(f"Failed to connect to public WebSocket: {e} (retry {self._ws_public_retry_count}, backoff {self._get_ws_backoff(self._ws_public_retry_count)}s)")
                except Exception as e:
                    self._ws_public_connected = False
                    logger.error(f"Failed to connect to public WebSocket: {e} (retry {self._ws_public_retry_count}, backoff {self._get_ws_backoff(self._ws_public_retry_count)}s)")
        finally:
            self._ws_public_reconnecting = False

    async def _connect_private_ws(self):
        if self._ws_private_reconnecting:
            return
        self._ws_private_reconnecting = True
        try:
            async with self._ws_reconnect_lock_private:
                if self._ws_private and self._is_ws_open(self._ws_private):
                    self._ws_private_connected = True
                    return

                # 清理旧任务和连接，防止重连后任务泄漏
                self._cancel_ws_tasks(self._ws_private_tasks, "private")
                await self._close_ws(self._ws_private, "private")
                self._ws_private = None
                self._ws_private_logged_in = False

                if not hasattr(self, '_ws_private_retry_count'):
                    self._ws_private_retry_count = 0
                    self._ws_private_last_retry = 0

                now = time.time()
                backoff = self._get_ws_backoff(self._ws_private_retry_count)
                if now - self._ws_private_last_retry < backoff:
                    await asyncio.sleep(backoff - (now - self._ws_private_last_retry))

                self._ws_private_retry_count += 1
                self._ws_private_last_retry = time.time()

                # ── 多代理源自动切换：获取当前活跃代理（自动跳过被禁用的代理）──
                current_proxy = self._get_active_proxy()
                use_proxy = current_proxy is not None

                try:
                    # P3-1-FIX: 境内环境必须优先代理（OKX 直连被墙）。配置了代理时先走代理，
                    # 代理失败才回退直连（境外环境 proxy=None 时直接走直连）。
                    if use_proxy:
                        try:
                            from websockets_proxy import Proxy, proxy_connect
                            proxy = Proxy.from_url(current_proxy)
                            # P33: 代理连接必须显式超时，否则代理半死时重连永久阻塞
                            self._ws_private = await proxy_connect(self.ws_private_url, proxy=proxy, ping_interval=25, proxy_conn_timeout=10)
                            self._mark_proxy_success(current_proxy)
                            logger.info(f"Connected to OKX private WebSocket (via proxy {current_proxy})")
                        except ImportError:
                            if self.allow_direct_fallback:
                                logger.error("websockets-proxy not installed, falling back to direct")
                                self._ws_private = await websockets.connect(self.ws_private_url, ping_interval=None,
                                    close_timeout=5, max_size=2**20)
                                logger.info("Connected to OKX private WebSocket (direct)")
                            else:
                                raise
                        except Exception as proxy_e:
                            # 标记当前代理失败（达阈值后禁用该代理并自动切换下一个）
                            self._mark_proxy_failed(current_proxy, hard_fail=self._is_connection_refused(proxy_e))
                            if self.allow_direct_fallback:
                                # P3-1: 代理失败时回退直连
                                logger.warning(f"P3-1: Proxy WS failed ({proxy_e}), falling back to direct")
                                self._ws_private = await websockets.connect(self.ws_private_url, ping_interval=None,
                                    close_timeout=5, max_size=2**20)
                                logger.info("P3-1: Connected to OKX private WebSocket (direct after proxy fail)")
                            else:
                                logger.warning(f"Private WS proxy connect failed ({proxy_e}), will retry with next proxy")
                                raise
                    else:
                        self._ws_private = await websockets.connect(self.ws_private_url, ping_interval=None,
                            close_timeout=5, max_size=2**20)
                        logger.info("Connected to OKX private WebSocket (direct)")

                    self._ws_private_connected = True
                    self._ws_private_last_pong = time.time()
                    self._ws_private_retry_count = 0
                    self._ws_private_logged_in = False

                    # P14: 网络恢复后重置延迟状态
                    self._reset_latency_state()

                    await self._send_ws_login()

                    self._ws_private_tasks.append(asyncio.create_task(self._process_private_ws()))
                    self._ws_private_tasks.append(asyncio.create_task(self._heartbeat_private()))
                    self._ws_private_tasks.append(asyncio.create_task(self._monitor_private_connection()))

                    await self._ensure_private_subscription()
                except Exception as e:
                    self._ws_private_connected = False
                    self._ws_private_logged_in = False
                    logger.error(f"Failed to connect to private WebSocket: {e} (retry {self._ws_private_retry_count}, backoff {self._get_ws_backoff(self._ws_private_retry_count)}s)")
        finally:
            self._ws_private_reconnecting = False

    def _get_ws_backoff(self, retry_count: int) -> int:
        """WebSocket重连退避时间
        
        P6: 扩展退避上限，快速重试→逐渐延长→最大5分钟
        0-3次: 2-8s, 4-6次: 16-64s, 7-10次: 128-300s
        P14: 添加随机jitter防止重连风暴
        """
        import random
        if retry_count <= 3:
            backoff = min(8, 2 * (2 ** max(0, retry_count - 1)))
        elif retry_count <= 6:
            backoff = min(60, 16 * (2 ** max(0, retry_count - 4)))
        else:
            backoff = min(300, 128 * (2 ** max(0, retry_count - 7)))
        # P14: 添加±30%的随机jitter，防止公/私WS同时重连
        jitter = random.uniform(0.7, 1.3)
        return int(backoff * jitter)
    
    def _reset_latency_state(self):
        """P14: WebSocket重连成功后重置延迟状态
        
        网络恢复后延迟EMA可能仍反映断开前的高延迟，需要重置
        以允许系统快速恢复正常运行。
        """
        if self._latency_ema > 5000:
            logger.info(
                f"P14: Resetting latency state after WS reconnect "
                f"(EMA was {self._latency_ema:.0f}ms, circuit_open={self._latency_circuit_open})"
            )
            self._latency_ema = 0.0
            self._latency_circuit_open = False
            self._latency_critical_count = 0
            self._latency_samples.clear()

    def _cancel_ws_tasks(self, tasks: List[asyncio.Task], ws_name: str = ""):
        """取消并清理旧的WebSocket任务，防止重连后任务泄漏
        
        P6: 跳过当前正在执行的任务（避免取消调用者自身导致竞态条件）
        """
        current_task = asyncio.current_task()
        for task in list(tasks):  # 复制列表避免迭代中修改
            if task is current_task:
                continue  # P6: 不取消当前任务
            if not task.done():
                try:
                    task.cancel()
                except Exception:
                    pass
        tasks.clear()
        logger.debug(f"Cancelled {ws_name} WebSocket tasks")

    async def _close_ws(self, ws, ws_name: str = ""):
        """安全关闭WebSocket连接"""
        if ws is None:
            return
        try:
            if self._is_ws_open(ws):
                await ws.close()
        except Exception:
            pass
        logger.debug(f"Closed {ws_name} WebSocket connection")

    async def _send_ws_login(self):
        if not self._ws_private or not self._is_ws_open(self._ws_private):
            return
        try:
            self._get_current_keys()
            timestamp = str(int(time.time()))
            method = "GET"
            path = "/users/self/verify"
            signature = self._generate_signature(timestamp, method, path)
            
            login_msg = {
                "op": "login",
                "args": [{
                    "apiKey": self.api_key,
                    "timestamp": timestamp,
                    "passphrase": self.passphrase,
                    "sign": signature
                }]
            }
            
            await self._ws_private.send(json.dumps(login_msg))
            logger.info("Sent WebSocket login message")
        except Exception as e:
            logger.error(f"Failed to send WebSocket login: {e}")

    async def _process_public_ws(self):
        """处理公共WebSocket消息循环，异常时由监控任务触发重连"""
        ws = self._ws_public
        try:
            async for message in ws:
                try:
                    # 处理纯文本 pong 响应（OKX pong 是纯文本 "pong"）
                    if message == "pong":
                        self._ws_public_last_pong = time.time()
                        continue

                    data = json.loads(message)

                    if data.get("event") == "subscribe":
                        continue

                    # 处理错误/取消订阅事件
                    if data.get("event") == "error":
                        logger.warning(f"Public WebSocket error event: {data}")
                        continue

                    channel = data.get("arg", {}).get("channel", "")
                    symbol = data.get("arg", {}).get("instId", "")
                    if channel == "books" and symbol:
                        tick_data = await self._consume_sequenced_orderbook(data, symbol)
                        if tick_data and self.tick_callback:
                            self._ws_public_last_data = time.time()
                            asyncio.create_task(self._dispatch_tick_callback(tick_data))
                except Exception as e:
                    logger.error(f"Error handling public WS message: {e}")
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Public WebSocket connection closed: {e}")
        except Exception as e:
            logger.error(f"Error processing public WebSocket: {e}")
        finally:
            # 只有当前处理的连接仍是当前连接时才更新状态，避免覆盖新连接状态
            if self._ws_public is ws:
                self._ws_public_connected = False

    async def _dispatch_tick_callback(self, tick_data):
        """独立分发tick回调，防止回调阻塞消息循环"""
        try:
            result = self.tick_callback(tick_data)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error(f"Error in tick callback: {e}")

    async def _consume_sequenced_orderbook(
        self,
        message: Dict[str, Any],
        symbol: str,
    ) -> Optional[TickData]:
        """Validate the OKX books snapshot/delta chain before deriving a quote tick."""
        rows = message.get("data") or []
        if not rows or not isinstance(rows[0], dict):
            return None
        book = rows[0]
        action = message.get("action")

        if action == "snapshot":
            try:
                self._books_last_sequence[symbol] = int(book["seqId"])
            except (KeyError, TypeError, ValueError):
                await self._resync_orderbook_stream(symbol, "snapshot_missing_seq_id")
                return None
            self._books_resync_pending.discard(symbol)
            return self._parse_tick_data(message, symbol)

        if action != "update" or symbol in self._books_resync_pending:
            return None

        try:
            previous_sequence = int(book["prevSeqId"])
            sequence = int(book["seqId"])
        except (KeyError, TypeError, ValueError):
            await self._resync_orderbook_stream(symbol, "update_missing_sequence")
            return None

        last_sequence = self._books_last_sequence.get(symbol)
        if last_sequence is None:
            await self._resync_orderbook_stream(symbol, "update_before_snapshot")
            return None
        if sequence == last_sequence and previous_sequence == last_sequence:
            return None
        if previous_sequence != last_sequence or sequence <= previous_sequence:
            await self._resync_orderbook_stream(symbol, "sequence_gap")
            return None

        self._books_last_sequence[symbol] = sequence
        return self._parse_tick_data(message, symbol)

    async def _resync_orderbook_stream(self, symbol: str, reason: str) -> None:
        self._books_last_sequence.pop(symbol, None)
        if symbol in self._books_resync_pending:
            return
        self._books_resync_pending.add(symbol)
        logger.warning(f"Orderbook sequence invalid for {symbol}: {reason}; resyncing")

        try:
            fallback_book = await asyncio.to_thread(self.get_order_book, symbol, 50)
            if fallback_book:
                fallback_tick = self._parse_tick_data({"data": [fallback_book]}, symbol)
                if fallback_tick and self.tick_callback:
                    asyncio.create_task(self._dispatch_tick_callback(fallback_tick))
        except Exception as exc:
            logger.warning(f"REST orderbook fallback failed for {symbol}: {exc}")

        if self._ws_public is None or not self._is_ws_open(self._ws_public):
            return
        subscription = {"channel": "books", "instId": symbol}
        try:
            await self._ws_public.send(json.dumps({"op": "unsubscribe", "args": [subscription]}))
            await self._ws_public.send(json.dumps({"op": "subscribe", "args": [subscription]}))
        except Exception as exc:
            logger.warning(f"Orderbook resubscribe failed for {symbol}: {exc}")

    async def _process_private_ws(self):
        """处理私有WebSocket消息循环，异常时由监控任务触发重连"""
        ws = self._ws_private
        try:
            async for message in ws:
                try:
                    # 处理纯文本 pong 响应（OKX pong 是纯文本 "pong"）
                    if message == "pong":
                        self._ws_private_last_pong = time.time()
                        continue

                    data = json.loads(message)

                    # 处理login成功事件
                    if data.get("event") == "login":
                        if data.get("code") == "0":
                            self._ws_private_logged_in = True
                            logger.info("Private WebSocket login successful")
                        else:
                            logger.error(f"Private WebSocket login failed: {data}")
                        continue

                    if data.get("event") == "subscribe":
                        continue

                    # 处理错误/取消订阅事件
                    if data.get("event") == "error":
                        logger.warning(f"Private WebSocket error event: {data}")
                        continue

                    channel = data.get("arg", {}).get("channel", "")
                    if channel in ("positions", "orders", "account"):
                        self._ws_private_last_data = time.time()
                    if channel == "positions" and self.position_callback:
                        for pos in data.get("data", []):
                            asyncio.create_task(self._dispatch_position_callback(pos))
                    elif channel == "orders" and self.order_callback:
                        for order in data.get("data", []):
                            asyncio.create_task(self._dispatch_order_callback(order))
                except Exception as e:
                    logger.error(f"Error handling private WS message: {e}")
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Private WebSocket connection closed: {e}")
        except Exception as e:
            logger.error(f"Error processing private WebSocket: {e}")
        finally:
            # 只有当前处理的连接仍是当前连接时才更新状态，避免覆盖新连接状态
            if self._ws_private is ws:
                self._ws_private_connected = False

    async def _dispatch_position_callback(self, position):
        try:
            result = self.position_callback(position)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error(f"Error in position callback: {e}")

    async def _dispatch_order_callback(self, order):
        try:
            result = self.order_callback(order)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error(f"Error in order callback: {e}")

    async def _heartbeat_public(self):
        """公共WebSocket心跳：定期发送 ping 保持连接活跃"""
        while self._ws_running and self._ws_public and self._is_ws_open(self._ws_public):
            try:
                await self._ws_public.send("ping")
                await asyncio.sleep(15)
            except Exception as e:
                logger.warning(f"Public WebSocket heartbeat error: {e}")
                break

    async def _heartbeat_private(self):
        """私有WebSocket心跳：定期发送 ping 保持连接活跃"""
        while self._ws_running and self._ws_private and self._is_ws_open(self._ws_private):
            try:
                await self._ws_private.send("ping")
                await asyncio.sleep(15)
            except Exception as e:
                logger.warning(f"Private WebSocket heartbeat error: {e}")
                break

    async def _monitor_public_connection(self):
        """监控公共WebSocket连接健康，必要时触发重连"""
        # 启动后短暂等待，让初始连接和订阅完成
        await asyncio.sleep(2)
        while self._ws_running:
            try:
                if self._ws_public_reconnecting:
                    await asyncio.sleep(1)
                    continue

                if not self._ws_public or not self._is_ws_open(self._ws_public):
                    self._ws_public_connected = False
                    if self._subscribed_public_channels:
                        logger.warning("Public WebSocket disconnected, triggering reconnect")
                        await self._connect_public_ws()
                else:
                    now = time.time()
                    data_age = now - self._ws_public_last_data if self._ws_public_last_data else None
                    # 数据断流检测（独立于 pong）：books 是高频推送，一旦断流即视为半死连接。
                    # 关键修复：旧逻辑用 max(pong, data) 合并判断，pong 正常时会掩盖数据断流，
                    # 导致「TCP 仍 OPEN 但服务器不再推送 books」的半死连接永不重连 → 交易停滞。
                    if data_age is not None and data_age > self._ws_data_timeout:
                        logger.warning(f"Public WebSocket data stall ({data_age:.0f}s no books data), closing and reconnecting")
                        try:
                            await self._ws_public.close()
                        except Exception:
                            pass
                        self._ws_public_connected = False
                        await self._connect_public_ws()
                    elif now - self._ws_public_last_pong > self._ws_heartbeat_interval + self._ws_pong_timeout:
                        # pong 超时但数据仍在流动：可能只是 pong 丢失，连接层仍健康，暂不关闭
                        logger.debug("Public WebSocket pong missed but data flowing, keeping connection")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error monitoring public WS: {e}")
                await asyncio.sleep(5)

    async def _monitor_private_connection(self):
        """监控私有WebSocket连接健康，必要时触发重连"""
        # 启动后短暂等待，让初始连接、登录和订阅完成
        await asyncio.sleep(2)
        while self._ws_running:
            try:
                if self._ws_private_reconnecting:
                    await asyncio.sleep(1)
                    continue

                if not self._ws_private or not self._is_ws_open(self._ws_private):
                    self._ws_private_connected = False
                    if self._subscribed_private_channels:
                        logger.warning("Private WebSocket disconnected, triggering reconnect")
                        await self._connect_private_ws()
                else:
                    now = time.time()
                    last_heartbeat = max(self._ws_private_last_pong, self._ws_private_last_data)
                    if now - last_heartbeat > self._ws_heartbeat_interval + self._ws_pong_timeout:
                        if now - self._ws_private_last_data <= self._ws_data_timeout:
                            logger.debug("Private WebSocket pong missed but data flowing, keeping connection")
                        else:
                            logger.warning("Private WebSocket heartbeat timeout, closing and reconnecting")
                            try:
                                await self._ws_private.close()
                            except Exception:
                                pass
                            self._ws_private_connected = False
                            await self._connect_private_ws()
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error monitoring private WS: {e}")
                await asyncio.sleep(5)

    def is_ws_public_connected(self) -> bool:
        return self._ws_public_connected and self._ws_public is not None and self._is_ws_open(self._ws_public)

    def is_ws_private_connected(self) -> bool:
        return self._ws_private_connected and self._ws_private is not None and self._is_ws_open(self._ws_private)

    @staticmethod
    def _is_ws_open(ws) -> bool:
        """兼容 websockets v13+ 的 .state 和新老版本的 .open 属性"""
        if ws is None:
            return False
        if _WS_OPEN_STATE is not None:
            # websockets v13+：使用 state 属性
            return ws.state == _WS_OPEN_STATE
        # 回退到旧版 .open 属性
        return getattr(ws, 'open', False)

    def get_ws_status(self) -> Dict[str, Any]:
        """获取WebSocket连接状态摘要"""
        now = time.time()
        return {
            "public_connected": self.is_ws_public_connected(),
            "private_connected": self.is_ws_private_connected(),
            "public_last_pong_age": round(now - self._ws_public_last_pong, 2) if self._ws_public_last_pong else None,
            "private_last_pong_age": round(now - self._ws_private_last_pong, 2) if self._ws_private_last_pong else None,
            "public_last_data_age": round(now - self._ws_public_last_data, 2) if self._ws_public_last_data else None,
            "private_last_data_age": round(now - self._ws_private_last_data, 2) if self._ws_private_last_data else None,
            "public_subscribed_channels": len(self._subscribed_public_channels),
            "private_subscribed_channels": len(self._subscribed_private_channels),
        }

    async def close_websocket(self):
        self._ws_running = False

        # 取消所有WebSocket相关任务
        self._cancel_ws_tasks(self._ws_public_tasks, "public")
        self._cancel_ws_tasks(self._ws_private_tasks, "private")

        if self._ws_public:
            try:
                await self._ws_public.close()
                logger.info("Public WebSocket closed")
            except Exception as e:
                logger.error(f"Error closing public WebSocket: {e}")

        if self._ws_private:
            try:
                await self._ws_private.close()
                logger.info("Private WebSocket closed")
            except Exception as e:
                logger.error(f"Error closing private WebSocket: {e}")

        self._ws_public_connected = False
        self._ws_private_connected = False

    def _parse_tick_data(self, raw_data: Dict[str, Any], symbol: str) -> Optional[TickData]:
        try:
            data = raw_data.get("data", [{}])[0]
            
            if "asks" in data and "bids" in data:
                asks = data.get("asks", [])
                bids = data.get("bids", [])
                bid_price = float(bids[0][0]) if bids else 0
                bid_volume = float(bids[0][1]) if bids else 0
                ask_price = float(asks[0][0]) if asks else 0
                ask_volume = float(asks[0][1]) if asks else 0
                price = (bid_price + ask_price) / 2 if bid_price > 0 and ask_price > 0 else ask_price or bid_price
                return TickData(
                    symbol=symbol,
                    price=price,
                    volume=0,
                    bid_price=bid_price,
                    bid_volume=bid_volume,
                    ask_price=ask_price,
                    ask_volume=ask_volume,
                    timestamp=datetime.fromtimestamp(int(data.get("ts", str(int(time.time() * 1000)))) / 1000)
                )
            else:
                return TickData(
                    symbol=symbol,
                    price=float(data.get("last", "0")),
                    volume=float(data.get("vol24h", "0")),
                    bid_price=float(data.get("bidPx", "0")),
                    bid_volume=float(data.get("bidSz", "0")),
                    ask_price=float(data.get("askPx", "0")),
                    ask_volume=float(data.get("askSz", "0")),
                    timestamp=datetime.fromtimestamp(int(data.get("ts", "0")) / 1000)
                )
        except Exception as e:
            logger.error(f"Error parsing tick data: {e}")
            return None

    def _parse_bar_data(self, raw_data: List[List[str]], symbol: str, interval: str) -> List[BarData]:
        bars = []
        for bar in raw_data:
            try:
                bars.append(BarData(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(int(bar[0]) / 1000),
                    open=float(bar[1]),
                    high=float(bar[2]),
                    low=float(bar[3]),
                    close=float(bar[4]),
                    volume=float(bar[5]),
                    interval=interval
                ))
            except Exception as e:
                logger.error(f"Error parsing bar data: {e}")
        return bars

    def _parse_position(self, raw_data: Dict[str, Any]) -> Optional[Position]:
        try:
            pos = raw_data.get("pos", "0")
            avg_px = raw_data.get("avgPx", "0")
            mark_px = raw_data.get("markPx", "0")
            upl = raw_data.get("upl", "0")
            margin = raw_data.get("margin", "0")
            lever = raw_data.get("lever", "1")
            mmr = raw_data.get("mmr", "0")
            notional_usd = raw_data.get("notionalUsd", "0")
            liq_px = raw_data.get("liqPx", "0")
            
            pos_qty = float(pos) if pos else 0.0
            avg_cost_val = float(avg_px) if avg_px else 0.0
            mark_price_val = float(mark_px) if mark_px else 0.0
            margin_val = float(margin) if margin else 0.0
            leverage_val = int(lever) if lever else 1
            notional_val = float(notional_usd) if notional_usd else 0.0
            liq_px_val = float(liq_px) if liq_px else 0.0
            
            # 降级方案：如果API返回的margin为0，用名义价值/杠杆估算
            if margin_val == 0 and pos_qty != 0 and leverage_val > 0:
                if notional_val > 0:
                    margin_val = notional_val / leverage_val
                elif avg_cost_val > 0:
                    # 最后降级：用数量×均价估算（不准确，未计入ctVal）
                    notional_val = abs(pos_qty) * avg_cost_val
                    margin_val = notional_val / leverage_val
                logger.debug(f"Position margin estimated for {raw_data.get('instId', '')}: {margin_val:.2f} (notional={notional_val:.2f}, leverage={leverage_val})")
            
            return Position(
                symbol=raw_data.get("instId", ""),
                side=raw_data.get("posSide", ""),
                quantity=pos_qty,
                avg_cost=avg_cost_val,
                mark_price=mark_price_val,
                unrealized_pnl=float(upl) if upl else 0.0,
                margin=margin_val,
                leverage=leverage_val,
                maintenance_margin_rate=float(mmr) if mmr else 0.0,
                notional_usd=notional_val,
                liquidation_price=liq_px_val,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error parsing position: {e}")
            return None

    def _parse_account_info(self, raw_data: Dict[str, Any]) -> Optional[AccountInfo]:
        try:
            total_equity = 0.0
            available_balance = 0.0
            used_margin = 0.0
            unrealized_pnl = 0.0

            total_eq = raw_data.get("totalEq", "0")
            try:
                total_equity = float(total_eq) if total_eq else 0.0
            except (ValueError, TypeError):
                total_equity = 0.0

            # USDT 本位合约账户：以 USDT 实际权益为准，避免 totalEq 混入非 USDT 资产导致失真
            usdt_equity = 0.0

            details = raw_data.get("details", [])
            for detail in details:
                ccy = detail.get("ccy", "")
                if ccy == "USDT":
                    avail_bal = detail.get("availBal", "0")
                    eq = detail.get("eq", "0")
                    frozen = detail.get("frozenBal", "0")
                    upl = detail.get("upl", "0")
                    avail_eq = detail.get("availEq", "0")
                    ord_frozen = detail.get("ordFrozen", "0")

                    try:
                        usdt_eq = float(eq) if eq else 0.0
                        usdt_equity = usdt_eq

                        avail_bal_val = float(avail_bal) if avail_bal else 0.0
                        available_balance += avail_bal_val

                        frozen_val = float(frozen) if frozen else 0.0
                        upl_val = float(upl) if upl else 0.0
                        unrealized_pnl += upl_val

                        avail_eq_val = float(avail_eq) if avail_eq else 0.0
                        ord_frozen_val = float(ord_frozen) if ord_frozen else 0.0

                        if usdt_eq > 0 and avail_eq_val > 0:
                            calc_used_margin = usdt_eq - avail_eq_val
                        elif frozen_val > 0:
                            calc_used_margin = frozen_val
                        else:
                            calc_used_margin = 0.0

                        used_margin += max(calc_used_margin, ord_frozen_val)
                    except (ValueError, TypeError):
                        pass

            if usdt_equity > 0:
                total_equity = usdt_equity

            if total_equity > 0 and used_margin == 0 and available_balance > 0:
                used_margin = max(0, total_equity - available_balance - unrealized_pnl)

            if used_margin < 0:
                used_margin = 0.0

            margin_rate = used_margin / total_equity if total_equity > 0 else 0.0

            return AccountInfo(
                total_equity=total_equity,
                available_balance=available_balance,
                used_margin=used_margin,
                unrealized_pnl=unrealized_pnl,
                margin_rate=margin_rate,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error parsing account info: {e}")
            return None

    def _parse_funding_rate(self, raw_data: Dict[str, Any], symbol: str) -> Optional[FundingRate]:
        try:
            return FundingRate(
                symbol=symbol,
                funding_rate=float(raw_data.get("fundingRate", "0")),
                next_funding_time=datetime.fromtimestamp(int(raw_data.get("nextFundingTime", "0")) / 1000),
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error parsing funding rate: {e}")
            return None

    def close(self):
        self._session.close()
        if self._async_session and not self._async_session.closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._async_session.close())
                else:
                    loop.run_until_complete(self._async_session.close())
            except RuntimeError:
                pass
        if self._ws_public:
            asyncio.create_task(self._ws_public.close())
        if self._ws_private:
            asyncio.create_task(self._ws_private.close())