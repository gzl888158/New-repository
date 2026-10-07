"""生产级仓位管理系统
==================
核心定位：实时追踪、风险联动、动态调整，确保仓位安全与资金效率

功能：
- 实时持仓追踪（REST + WebSocket 双通道同步）
- 风险联动（回撤减仓、保证金预警、单仓亏损限制）
- 动态仓位调整（再平衡、优化部署）
- 多策略仓位协调（跨策略持仓冲突检测）
- 状态持久化（快照保存/恢复）
- 紧急减仓（黑天鹅/流动性危机）
- 持仓健康监控（异常检测、幽灵仓位清理）
- 企业级同步增强（版本化追踪、健康监控、过期检测、自动恢复）

架构：
  PositionManager
  ├── PositionTracker（实时持仓追踪）
  │   ├── REST 同步（定时全量拉取）
  │   ├── WebSocket 增量更新
  │   └── 本地缓存（内存 + Redis）
  ├── RiskLinkage（风险联动）
  │   ├── 回撤减仓
  │   ├── 保证金率监控
  │   ├── 单仓亏损限制
  │   └── 全局风控协作
  ├── DynamicAdjuster（动态调整）
  │   ├── 再平衡引擎
  │   ├── 权重优化
  │   └── 闲置资金部署
  └── StatePersistence（状态持久化）
      ├── 快照保存
      ├── 快照恢复
      └── 历史记录
"""

import asyncio
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from loguru import logger


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

class PositionSide(Enum):
    """持仓方向"""
    LONG = "long"
    SHORT = "short"


class PositionStatus(Enum):
    """持仓状态"""
    ACTIVE = "active"          # 正常
    WARNING = "warning"        # 预警（接近止损/保证金不足）
    CRITICAL = "critical"      # 临界（即将触发强平）
    REDUCING = "reducing"      # 减仓中
    CLOSING = "closing"        # 平仓中
    CLOSED = "closed"          # 已平仓


class RiskLevel(Enum):
    """风险等级"""
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PositionSnapshot:
    """持仓快照"""
    symbol: str
    side: PositionSide
    quantity: float              # 持仓数量（币数）
    avg_cost: float              # 开仓均价
    mark_price: float            # 标记价格
    unrealized_pnl: float        # 未实现盈亏
    realized_pnl: float = 0.0    # 已实现盈亏
    margin: float = 0.0          # 占用保证金
    leverage: int = 1            # 杠杆倍数
    liquidation_price: float = 0.0  # 预估强平价
    margin_ratio: float = 0.0    # 保证金率
    strategy_name: str = ""      # 策略名称
    entry_time: float = 0.0      # 入场时间戳
    status: PositionStatus = PositionStatus.ACTIVE
    health_score: float = 1.0    # 健康评分（0-1）
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccountRiskSnapshot:
    """账户风险快照"""
    total_equity: float = 0.0
    available_balance: float = 0.0
    used_margin: float = 0.0
    margin_ratio: float = 0.0
    unrealized_pnl: float = 0.0
    drawdown_pct: float = 0.0
    risk_level: RiskLevel = RiskLevel.NORMAL
    position_count: int = 0
    total_leverage: float = 0.0
    margin_utilization: float = 0.0
    timestamp: float = 0.0


@dataclass
class RiskEvent:
    """风险事件"""
    event_type: str              # drawdown / margin_warning / liquidation_risk / position_loss
    severity: RiskLevel
    symbol: str = ""
    strategy_name: str = ""
    message: str = ""
    value: float = 0.0
    threshold: float = 0.0
    timestamp: float = field(default_factory=time.time)
    action_taken: str = ""
    resolved: bool = False


# 回调类型
RiskEventCallback = Callable[[RiskEvent], None]
PositionChangeCallback = Callable[[List[PositionSnapshot]], None]


# ═══════════════════════════════════════════════════════════════
# 生产级仓位管理器
# ═══════════════════════════════════════════════════════════════

class PositionManager:
    """
    生产级仓位管理系统

    使用示例:
        mgr = PositionManager(config, okx_client)
        mgr.on_risk_event = lambda event: print(f"Risk: {event}")
        await mgr.start()
        # ... 运行中 ...
        snapshot = mgr.get_position_snapshot("BTC-USDT")
        risk = mgr.get_account_risk()
        await mgr.stop()
    """

    def __init__(self, config: Dict[str, Any], okx_client=None,
                 sqlite_storage=None, redis_cache=None):
        self.config = config
        self._okx_client = okx_client
        self._sqlite_storage = sqlite_storage
        self._redis_cache = redis_cache
        self._alert_manager = None

        pm_cfg = config.get("position_manager", {})

        # ── 配置 ──
        self._enabled = pm_cfg.get("enabled", True)
        self._sync_interval = pm_cfg.get("sync_interval_sec", 3)
        try:
            self._sync_failure_threshold = max(
                1, int(pm_cfg.get("sync_failure_threshold", 3))
            )
        except (TypeError, ValueError):
            self._sync_failure_threshold = 3
        self._sync_fail_streak = 0
        self._sync_degraded = False
        self._risk_check_interval = pm_cfg.get("risk_check_interval_sec", 5)
        self._max_total_positions = pm_cfg.get("max_total_positions", 6)
        self._max_positions_per_symbol = pm_cfg.get("max_positions_per_symbol", 2)
        self._max_positions_per_strategy = pm_cfg.get("max_positions_per_strategy", 3)

        # 风险联动配置
        rl_cfg = pm_cfg.get("risk_linkage", {})
        self._drawdown_trigger = rl_cfg.get("drawdown_trigger", 0.15)
        self._drawdown_reduce_pct = rl_cfg.get("drawdown_reduce_pct", 0.3)
        self._margin_ratio_warning = rl_cfg.get("margin_ratio_warning", 0.5)
        self._margin_ratio_critical = rl_cfg.get("margin_ratio_critical", 0.3)
        self._position_loss_limit = rl_cfg.get("position_loss_limit", 0.1)
        # 峰值权益衰减：当回撤持续超过阈值达指定时长后，重置峰值到当前权益，
        # 避免历史高位永久卡死回撤计算导致持续 emergency 减仓死循环。
        self._peak_decay_drawdown_threshold = rl_cfg.get("peak_decay_drawdown_threshold", 0.50)
        self._peak_decay_after_hours = rl_cfg.get("peak_decay_after_hours", 24.0)

        # 动态调整配置
        da_cfg = pm_cfg.get("dynamic_adjust", {})
        self._dynamic_adjust_enabled = da_cfg.get("enabled", True)
        self._rebalance_interval = da_cfg.get("rebalance_interval", 300)
        self._max_adjust_per_iter = da_cfg.get("max_adjust_per_iter", 0.1)
        self._min_position_value = da_cfg.get("min_position_value", 5.0)

        # 持久化配置
        p_cfg = pm_cfg.get("persistence", {})
        self._persistence_enabled = p_cfg.get("enabled", True)
        self._save_interval = p_cfg.get("save_interval", 60)
        self._max_snapshots = p_cfg.get("max_snapshots", 100)

        # ── 持仓缓存 ──
        self._positions: Dict[str, PositionSnapshot] = {}  # symbol:side -> snapshot
        self._positions_by_strategy: Dict[str, List[str]] = defaultdict(list)  # strategy -> [keys]
        self._positions_by_symbol: Dict[str, List[str]] = defaultdict(list)  # symbol -> [keys]
        self._position_history: List[PositionSnapshot] = []
        self._data_stale: bool = False  # 持仓数据过期标记（解析失败时置True）

        # ── 账户风险 ──
        self._account_risk = AccountRiskSnapshot()
        self._peak_equity: float = 0.0
        self._drawdown_high_since: float = 0.0  # 回撤首次超过衰减阈值的时间戳
        self._risk_events: List[RiskEvent] = []
        self._max_risk_events = 200

        # ── 回调 ──
        self._risk_event_callbacks: List[RiskEventCallback] = []
        self._position_change_callbacks: List[PositionChangeCallback] = []
        self._position_removal_callbacks: List[callable] = []  # (symbol, side) -> None

        # ── 运行控制 ──
        self._running = False
        self._sync_task: Optional[asyncio.Task] = None
        self._risk_task: Optional[asyncio.Task] = None
        self._persist_task: Optional[asyncio.Task] = None
        self._last_rebalance_time: float = 0.0
        self._last_save_time: float = 0.0

        # ── 紧急减仓标记 ──
        self._emergency_reduce: bool = False
        self._emergency_reason: str = ""

        # ── 持久化路径 ──
        self._persist_dir = config.get("system", {}).get("data_dir", "data")
        self._persist_file = os.path.join(self._persist_dir, "position_state.json")

        # ── 企业级同步引擎 ──
        self._sync_engine = None
        try:
            from core.enterprise_sync import get_sync_engine
            self._sync_engine = get_sync_engine(config)
            self._sync_engine.register_channel(
                "position_rest", self,
                expected_interval=self._sync_interval,
                recovery_callback=lambda: asyncio.ensure_future(self._sync_positions())
            )
            self._sync_engine.register_channel(
                "position_ws", self,
                expected_interval=1.0  # WS 预期每秒推送
            )
            logger.info("PositionManager: EnterpriseSyncEngine integrated")
        except Exception as e:
            logger.warning(f"PositionManager: EnterpriseSyncEngine skipped: {e}")

        logger.info(
            f"PositionManager initialized: "
            f"max_positions={self._max_total_positions}, "
            f"sync_interval={self._sync_interval}s, "
            f"risk_check={self._risk_check_interval}s"
        )

    def set_alert_manager(self, alert_manager):
        """注入告警管理器"""
        self._alert_manager = alert_manager

    # ═══════════════════════════════════════════════════════════════
    # 回调注册
    # ═══════════════════════════════════════════════════════════════

    def on_risk_event(self, callback: RiskEventCallback):
        """注册风险事件回调"""
        self._risk_event_callbacks.append(callback)

    def on_position_change(self, callback: PositionChangeCallback):
        """注册持仓变化回调"""
        self._position_change_callbacks.append(callback)

    def on_position_removal(self, callback):
        """注册持仓移除回调 (symbol, side) -> None，用于手动平仓/同步删除时触发清理"""
        self._position_removal_callbacks.append(callback)

    def _notify_risk_event(self, event: RiskEvent):
        """通知风险事件"""
        self._risk_events.append(event)
        if len(self._risk_events) > self._max_risk_events:
            self._risk_events = self._risk_events[-self._max_risk_events:]

        for cb in self._risk_event_callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error(f"Risk event callback error: {e}")

    def _notify_position_change(self):
        """通知持仓变化"""
        positions = list(self._positions.values())
        for cb in self._position_change_callbacks:
            try:
                cb(positions)
            except Exception as e:
                logger.error(f"Position change callback error: {e}")

    # ═══════════════════════════════════════════════════════════════
    # 启动/停止
    # ═══════════════════════════════════════════════════════════════

    async def start(self):
        """启动仓位管理器"""
        if not self._enabled:
            logger.info("PositionManager disabled")
            return

        self._running = True
        logger.info("PositionManager starting...")

        # 恢复持久化状态
        if self._persistence_enabled:
            self._restore_state()

        # 初始同步
        await self._sync_positions()

        # 启动企业级同步引擎
        if self._sync_engine:
            try:
                self._sync_engine.start()
            except Exception as e:
                logger.warning(f"PositionManager: sync engine start failed: {e}")

        # 启动后台任务
        self._sync_task = asyncio.create_task(self._sync_loop())
        self._sync_task.set_name("pm_sync")
        self._risk_task = asyncio.create_task(self._risk_check_loop())
        self._risk_task.set_name("pm_risk")
        if self._persistence_enabled:
            self._persist_task = asyncio.create_task(self._persist_loop())
            self._persist_task.set_name("pm_persist")

        logger.info("PositionManager started")

    async def stop(self):
        """停止仓位管理器"""
        self._running = False
        for task in [self._sync_task, self._risk_task, self._persist_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if self._persistence_enabled:
            self._save_state()

        # 停止企业级同步引擎
        if self._sync_engine:
            try:
                self._sync_engine.stop()
            except Exception:
                pass

        logger.info("PositionManager stopped")

    # ═══════════════════════════════════════════════════════════════
    # 持仓同步
    # ═══════════════════════════════════════════════════════════════

    async def _sync_loop(self):
        """持仓同步循环"""
        while self._running:
            try:
                await self._sync_positions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Position sync error: {e}")
                self._record_sync_failure(str(e))
            await asyncio.sleep(self._sync_interval)

    @property
    def sync_degraded(self) -> bool:
        """Whether repeated REST position-sync failures require blocking new entries."""
        return self._sync_degraded

    @property
    def sync_fail_streak(self) -> int:
        """Number of consecutive failed REST position-sync attempts."""
        return self._sync_fail_streak

    def _record_sync_failure(self, error: str) -> None:
        self._sync_fail_streak += 1
        if (
            not self._sync_degraded
            and self._sync_fail_streak >= self._sync_failure_threshold
        ):
            self._sync_degraded = True
            logger.critical(
                "Position synchronization degraded after "
                f"{self._sync_fail_streak} consecutive failures; "
                "new entries must be blocked"
            )
        else:
            logger.warning(
                f"Position sync failure streak {self._sync_fail_streak}/"
                f"{self._sync_failure_threshold}: {error}"
            )

    def _record_sync_success(self) -> None:
        was_degraded = self._sync_degraded
        self._sync_fail_streak = 0
        self._sync_degraded = False
        if was_degraded:
            logger.info("Position synchronization recovered; new entries may resume")

    async def _sync_positions(self):
        """从OKX同步持仓数据"""
        if not self._okx_client:
            error = "OKX client is unavailable"
            logger.error(f"Position sync aborted: {error}")
            self._record_sync_event("position_rest", success=False, error=error)
            self._record_sync_failure(error)
            return False

        try:
            checked_query = getattr(self._okx_client, "get_positions_checked", None)
            positions = (
                checked_query()
                if callable(checked_query)
                else self._okx_client.get_positions()
            )
            if positions is None:
                error = "exchange position query failed; keeping local snapshot"
                logger.error(f"Position sync aborted: {error}")
                self._record_sync_event("position_rest", success=False, error=error)
                self._record_sync_failure(error)
                return False

            if not positions:
                # 空持仓列表：清除所有本地持仓记录
                if self._positions:
                    logger.info(f"Exchange returned empty positions, clearing {len(self._positions)} local records")
                    self._positions.clear()
                    self._positions_by_strategy.clear()
                    self._positions_by_symbol.clear()
                    self._notify_position_change()
                self._data_stale = False
                self._record_sync_success()
                self._record_sync_event("position_rest", success=True, entities=[])
                return True

            previous_keys = set(self._positions.keys())
            snapshots: Dict[str, PositionSnapshot] = {}
            changed = False

            for pos_data in positions:
                try:
                    pos = self._okx_client._parse_position(pos_data)
                    if not pos:
                        raise ValueError("position parser rejected exchange record")
                    if abs(pos.quantity) == 0:
                        continue

                    side = PositionSide.LONG if pos.side == "long" else PositionSide.SHORT
                    key = f"{pos.symbol}:{side.value}"

                    snapshot = PositionSnapshot(
                        symbol=pos.symbol,
                        side=side,
                        quantity=abs(float(pos.quantity)),
                        avg_cost=float(pos.avg_cost) if pos.avg_cost else 0.0,
                        mark_price=float(pos.mark_price) if pos.mark_price else 0.0,
                        unrealized_pnl=float(pos.unrealized_pnl) if pos.unrealized_pnl else 0.0,
                        margin=float(pos.margin) if pos.margin else 0.0,
                        leverage=int(pos.leverage) if pos.leverage else 1,
                        liquidation_price=float(pos.liq_price) if getattr(pos, 'liq_price', 0) else 0.0,
                        margin_ratio=float(pos.margin_ratio) if getattr(pos, 'margin_ratio', 0) else 0.0,
                        entry_time=time.time(),
                    )

                    snapshots[key] = snapshot
                except Exception as e:
                    error = f"invalid exchange position response: {e}"
                    logger.error(f"Position sync aborted: {error}; keeping local snapshot (marked stale)")
                    self._data_stale = True
                    self._record_sync_event("position_rest", success=False, error=error)
                    self._record_sync_failure(error)
                    return False

            current_keys = set(snapshots)

            for key, snapshot in snapshots.items():
                old = self._positions.get(key)
                if old:
                    snapshot.strategy_name = old.strategy_name
                    snapshot.entry_time = old.entry_time
                    snapshot.status = old.status
                    snapshot.tags = list(old.tags)
                    if self._positions_changed(old, snapshot):
                        changed = True
                else:
                    changed = True
                self._positions[key] = snapshot

                if snapshot.strategy_name and key not in self._positions_by_strategy[snapshot.strategy_name]:
                    self._positions_by_strategy[snapshot.strategy_name].append(key)
                if key not in self._positions_by_symbol[snapshot.symbol]:
                    self._positions_by_symbol[snapshot.symbol].append(key)

            # 清理已不存在的持仓
            removed_keys = previous_keys - current_keys
            for key in removed_keys:
                if key in self._positions:
                    removed = self._positions.pop(key)
                    # 清理索引
                    if removed.strategy_name:
                        self._positions_by_strategy[removed.strategy_name] = [
                            k for k in self._positions_by_strategy[removed.strategy_name]
                            if k != key
                        ]
                    self._positions_by_symbol[removed.symbol] = [
                        k for k in self._positions_by_symbol[removed.symbol]
                        if k != key
                    ]
                    changed = True
                    logger.info(f"Position removed: {removed.symbol} {removed.side.value}")
                    # P0-手动平仓清理链：通知监听者（OrderExecutor）触发止损/策略/状态清理
                    for cb in self._position_removal_callbacks:
                        try:
                            cb(removed.symbol, removed.side.value)
                        except Exception as cb_err:
                            logger.warning(f"Position removal callback error: {cb_err}")

            if changed:
                self._notify_position_change()

            # 更新 PnL 历史
            self._position_history.extend(self._positions.values())
            if len(self._position_history) > self._max_snapshots * 10:
                self._position_history = self._position_history[-self._max_snapshots * 10:]

            # ── 企业级同步：记录 REST 通道同步事件 ──
            self._data_stale = False
            self._record_sync_success()
            self._record_sync_event("position_rest", success=True,
                                    entities=[f"{k}" for k in current_keys])
            return True

        except Exception as e:
            logger.error(f"Failed to sync positions: {e}")
            self._record_sync_event("position_rest", success=False, error=str(e))
            self._record_sync_failure(str(e))
            return False

    def _positions_changed(self, old: PositionSnapshot, new: PositionSnapshot) -> bool:
        """检测持仓是否发生显著变化"""
        return (
            abs(old.quantity - new.quantity) > 0.0001
            or abs(old.mark_price - new.mark_price) / max(old.mark_price, 0.0001) > 0.001
            or abs(old.unrealized_pnl - new.unrealized_pnl) > 0.01
            or abs(old.margin_ratio - new.margin_ratio) > 0.001
        )

    def _record_sync_event(self, channel: str, success: bool = True,
                           entities: List[str] = None, error: str = ""):
        """记录同步事件到企业级同步引擎"""
        if self._sync_engine:
            try:
                self._sync_engine.record_sync(
                    channel, success=success, entities=entities, error=error
                )
                if entities and success:
                    for entity in entities:
                        self._sync_engine.update_entity_version(
                            entity, source=channel, exchange_ts=time.time()
                        )
            except Exception:
                pass

    def update_position_from_ws(self, ws_data: List[Dict[str, Any]]):
        """从WebSocket数据更新持仓（增量更新）"""
        if not ws_data:
            return

        changed = False
        failed_count = 0
        for data in ws_data:
            try:
                symbol = data.get("instId", "")
                pos_side = data.get("posSide", "")
                side = PositionSide.LONG if pos_side == "long" else PositionSide.SHORT
                key = f"{symbol}:{side.value}"

                pos_qty = abs(float(data.get("pos", 0)))
                if pos_qty == 0:
                    if key in self._positions:
                        self._positions.pop(key)
                        changed = True
                    continue

                avg_px = float(data.get("avgPx", 0))
                mark_px = float(data.get("markPx", 0))
                upl = float(data.get("upl", 0))
                margin = float(data.get("margin", 0))
                leverage = int(float(data.get("lever", 1)))
                liq_px = float(data.get("liqPx", 0))
                mgn_ratio = float(data.get("mgnRatio", 0))

                snapshot = PositionSnapshot(
                    symbol=symbol,
                    side=side,
                    quantity=pos_qty,
                    avg_cost=avg_px,
                    mark_price=mark_px,
                    unrealized_pnl=upl,
                    margin=margin,
                    leverage=leverage,
                    liquidation_price=liq_px,
                    margin_ratio=mgn_ratio,
                    entry_time=time.time(),
                )

                if key in self._positions:
                    old = self._positions[key]
                    snapshot.strategy_name = old.strategy_name
                    snapshot.entry_time = old.entry_time
                    snapshot.status = old.status

                if key not in self._positions or self._positions_changed(
                    self._positions.get(key, snapshot), snapshot
                ):
                    changed = True

                self._positions[key] = snapshot

            except Exception as e:
                failed_count += 1
                logger.debug(f"WS position update error: {e}")

        if changed:
            self._notify_position_change()

        # ── 企业级同步：记录 WS 通道同步事件（区分成功/失败） ──
        if failed_count > 0:
            self._record_sync_event("position_ws", success=False, error=f"{failed_count} position updates failed")
        else:
            self._record_sync_event("position_ws", success=True,
                                    entities=list(self._positions.keys()))

    # ═══════════════════════════════════════════════════════════════
    # 风险检查
    # ═══════════════════════════════════════════════════════════════

    async def _risk_check_loop(self):
        """风险检查循环"""
        while self._running:
            try:
                await self._check_account_risk()
                await self._check_position_risks()
                await self._check_risk_linkage()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Risk check error: {e}")
            await asyncio.sleep(self._risk_check_interval)

    async def _check_account_risk(self):
        """检查账户级别风险"""
        if not self._okx_client:
            return

        try:
            account_info = self._okx_client.get_account_info()
            if not account_info:
                return

            # 单币种 USDT 账户：account/balance 顶层 mgnRatio/availBal/totalMgn/upl
            # 字段为空（mgnRatio 仅跨币种/组合保证金模式启用），直接 float("") 会抛
            # ValueError 导致整个风险检查静默失败。改为从 per-currency details 解析，
            # 口径与 okx_client._parse_account_info 一致。
            total_eq = 0.0
            avail_bal = 0.0
            used_margin = 0.0
            upl = 0.0

            try:
                total_eq = float(account_info.get("totalEq") or 0)
            except (ValueError, TypeError):
                total_eq = 0.0

            for detail in account_info.get("details") or []:
                if not isinstance(detail, dict) or detail.get("ccy") != "USDT":
                    continue
                try:
                    eq = float(detail.get("eq") or 0)
                    avail_eq = float(detail.get("availEq") or 0)
                    frozen = float(detail.get("frozenBal") or 0)
                    ord_frozen = float(detail.get("ordFrozen") or 0)
                    avail_bal = float(detail.get("availBal") or 0)
                    upl = float(detail.get("upl") or 0)

                    if eq > 0:
                        total_eq = eq
                    if eq > 0 and avail_eq > 0:
                        used_margin = eq - avail_eq
                    elif frozen > 0:
                        used_margin = frozen
                    else:
                        used_margin = 0.0
                    used_margin = max(used_margin, ord_frozen)
                except (ValueError, TypeError):
                    continue
                break

            mgn_ratio = used_margin / total_eq if total_eq > 0 else 0.0

            self._account_risk = AccountRiskSnapshot(
                total_equity=total_eq,
                available_balance=avail_bal,
                used_margin=used_margin,
                margin_ratio=mgn_ratio,
                unrealized_pnl=upl,
                position_count=len(self._positions),
                margin_utilization=used_margin / max(total_eq, 0.01),
                timestamp=time.time(),
            )

            # 更新峰值权益
            if total_eq > self._peak_equity:
                self._peak_equity = total_eq

            # 计算回撤
            if self._peak_equity > 0:
                drawdown = (self._peak_equity - total_eq) / self._peak_equity
                self._account_risk.drawdown_pct = drawdown

                # 峰值权益衰减：回撤持续超阈值达指定时长后重置峰值，
                # 承认资金已进入新基准，避免永久 emergency 死循环。
                now_ts = time.time()
                if drawdown >= self._peak_decay_drawdown_threshold:
                    if self._drawdown_high_since <= 0:
                        self._drawdown_high_since = now_ts
                    elif (now_ts - self._drawdown_high_since) >= self._peak_decay_after_hours * 3600:
                        old_peak = self._peak_equity
                        self._peak_equity = total_eq
                        self._drawdown_high_since = 0.0
                        self._account_risk.drawdown_pct = 0.0
                        drawdown = 0.0
                        logger.warning(
                            f"Peak equity decayed: {old_peak:.2f} -> {total_eq:.2f} "
                            f"(drawdown sustained >{self._peak_decay_drawdown_threshold:.0%} "
                            f"for >{self._peak_decay_after_hours:.0f}h)"
                        )
                else:
                    self._drawdown_high_since = 0.0

                if drawdown >= self._drawdown_trigger:
                    self._account_risk.risk_level = RiskLevel.HIGH
                    event = RiskEvent(
                        event_type="drawdown",
                        severity=RiskLevel.HIGH,
                        message=f"Drawdown {drawdown:.2%} exceeded trigger {self._drawdown_trigger:.2%}",
                        value=drawdown,
                        threshold=self._drawdown_trigger,
                    )
                    self._notify_risk_event(event)
                    # 触发减仓
                    await self._trigger_drawdown_reduce(drawdown)

            # 保证金率检查
            if mgn_ratio > 0:
                self._account_risk.margin_ratio = mgn_ratio
                if mgn_ratio >= self._margin_ratio_critical:
                    self._account_risk.risk_level = RiskLevel.CRITICAL
                    event = RiskEvent(
                        event_type="margin_critical",
                        severity=RiskLevel.CRITICAL,
                        message=f"Margin ratio {mgn_ratio:.2%} >= critical {self._margin_ratio_critical:.2%}",
                        value=mgn_ratio,
                        threshold=self._margin_ratio_critical,
                    )
                    self._notify_risk_event(event)
                elif mgn_ratio >= self._margin_ratio_warning:
                    self._account_risk.risk_level = RiskLevel.ELEVATED
                    event = RiskEvent(
                        event_type="margin_warning",
                        severity=RiskLevel.ELEVATED,
                        message=f"Margin ratio {mgn_ratio:.2%} >= warning {self._margin_ratio_warning:.2%}",
                        value=mgn_ratio,
                        threshold=self._margin_ratio_warning,
                    )
                    self._notify_risk_event(event)

        except Exception as e:
            logger.error(f"Account risk check error: {e}")

    async def _check_position_risks(self):
        """检查单个持仓风险"""
        for key, pos in list(self._positions.items()):
            try:
                # 健康评分
                health = self._calculate_position_health(pos)
                pos.health_score = health

                # 单仓亏损限制
                if pos.avg_cost > 0 and pos.quantity > 0:
                    loss_pct = abs(pos.unrealized_pnl) / (pos.avg_cost * pos.quantity)
                    if loss_pct > self._position_loss_limit:
                        pos.status = PositionStatus.WARNING
                        event = RiskEvent(
                            event_type="position_loss",
                            severity=RiskLevel.ELEVATED,
                            symbol=pos.symbol,
                            strategy_name=pos.strategy_name,
                            message=f"Position loss {loss_pct:.2%} > limit {self._position_loss_limit:.2%}",
                            value=loss_pct,
                            threshold=self._position_loss_limit,
                        )
                        self._notify_risk_event(event)

                # 保证金率检查
                if pos.margin_ratio > 0 and pos.margin_ratio >= self._margin_ratio_critical:
                    pos.status = PositionStatus.CRITICAL

                # 清算价格接近
                if pos.liquidation_price > 0 and pos.mark_price > 0:
                    liq_distance = abs(pos.mark_price - pos.liquidation_price) / pos.mark_price
                    if liq_distance < 0.02:  # 2%以内
                        pos.status = PositionStatus.CRITICAL
                        event = RiskEvent(
                            event_type="liquidation_risk",
                            severity=RiskLevel.CRITICAL,
                            symbol=pos.symbol,
                            strategy_name=pos.strategy_name,
                            message=f"Liquidation distance {liq_distance:.2%} < 2%",
                            value=liq_distance,
                            threshold=0.02,
                        )
                        self._notify_risk_event(event)

            except Exception as e:
                logger.debug(f"Position risk check error for {key}: {e}")

    def _calculate_position_health(self, pos: PositionSnapshot) -> float:
        """计算持仓健康评分（0-1）"""
        score = 1.0

        # 亏损扣分
        if pos.avg_cost > 0 and pos.quantity > 0:
            loss_pct = abs(pos.unrealized_pnl) / (pos.avg_cost * pos.quantity)
            if loss_pct > 0.05:
                score -= 0.3
            elif loss_pct > 0.02:
                score -= 0.15

        # 保证金率扣分
        if pos.margin_ratio > 0:
            if pos.margin_ratio >= self._margin_ratio_critical:
                score -= 0.5
            elif pos.margin_ratio >= self._margin_ratio_warning:
                score -= 0.25

        # 清算距离扣分
        if pos.liquidation_price > 0 and pos.mark_price > 0:
            liq_distance = abs(pos.mark_price - pos.liquidation_price) / pos.mark_price
            if liq_distance < 0.01:
                score -= 0.4
            elif liq_distance < 0.03:
                score -= 0.2

        return max(0.0, score)

    async def _check_risk_linkage(self):
        """检查风险联动条件"""
        if self._emergency_reduce:
            return

        # 动态调整
        if self._dynamic_adjust_enabled:
            now = time.time()
            if now - self._last_rebalance_time > self._rebalance_interval:
                await self._dynamic_rebalance()
                self._last_rebalance_time = now

    async def _trigger_drawdown_reduce(self, drawdown: float):
        """回撤触发减仓 — 实际通过 API 平掉亏损最大的持仓"""
        if self._emergency_reduce:
            return

        self._emergency_reduce = True
        self._emergency_reason = f"drawdown_{drawdown:.2%}"

        logger.warning(
            f"EMERGENCY: Drawdown {drawdown:.2%} triggered position reduction "
            f"({self._drawdown_reduce_pct:.0%})"
        )

        # 按亏损程度排序持仓（亏损最大的在前）
        sorted_positions = sorted(
            self._positions.items(),
            key=lambda x: x[1].unrealized_pnl,
        )

        # 减仓亏损最大的持仓
        reduce_count = max(1, int(len(self._positions) * self._drawdown_reduce_pct))
        closed = 0
        for key, pos in sorted_positions[:reduce_count]:
            close_side = "sell" if pos.side == PositionSide.LONG else "buy"
            reduce_qty = abs(pos.quantity) * self._drawdown_reduce_pct
            if reduce_qty <= 0 or not self._okx_client:
                continue
            try:
                self._okx_client.place_order(
                    symbol=pos.symbol,
                    side=close_side,
                    order_type="market",
                    quantity=reduce_qty,
                    leverage=pos.leverage,
                    reduce_only=True,
                )
                closed += 1
                logger.warning(
                    f"Drawdown reduce: closed {pos.symbol} {close_side} "
                    f"qty={reduce_qty:.4f} (loss={pos.unrealized_pnl:.4f})"
                )
            except Exception as e:
                logger.error(f"Drawdown reduce order failed for {pos.symbol}: {e}")

            event = RiskEvent(
                event_type="drawdown_reduce",
                severity=RiskLevel.HIGH,
                symbol=pos.symbol,
                strategy_name=pos.strategy_name,
                message=f"Drawdown reduce: closed {pos.symbol} {close_side} qty={reduce_qty:.4f}",
                action_taken="close_position",
            )
            self._notify_risk_event(event)

        if closed > 0:
            logger.warning(f"Drawdown reduce completed: {closed}/{reduce_count} positions reduced")
        else:
            logger.error("Drawdown reduce FAILED: no positions were closed")

        self._emergency_reduce = False

    # ═══════════════════════════════════════════════════════════════
    # 动态调整
    # ═══════════════════════════════════════════════════════════════

    async def _dynamic_rebalance(self):
        """动态仓位再平衡"""
        if not self._dynamic_adjust_enabled:
            return

        logger.debug("Dynamic position rebalance check...")

        # 检查持仓数量限制
        if len(self._positions) > self._max_total_positions:
            logger.warning(
                f"Position count {len(self._positions)} > max {self._max_total_positions}"
            )

        # 检查单币种持仓数量
        for symbol, keys in self._positions_by_symbol.items():
            if len(keys) > self._max_positions_per_symbol:
                logger.warning(
                    f"Symbol {symbol} has {len(keys)} positions > max {self._max_positions_per_symbol}"
                )

        # 检查单策略持仓数量
        for strategy, keys in self._positions_by_strategy.items():
            if len(keys) > self._max_positions_per_strategy:
                logger.warning(
                    f"Strategy {strategy} has {len(keys)} positions > max {self._max_positions_per_strategy}"
                )

        # 清理无策略关联的持仓
        orphaned = []
        for key, pos in self._positions.items():
            if not pos.strategy_name:
                orphaned.append(key)
        if orphaned:
            logger.info(f"Found {len(orphaned)} orphaned positions without strategy tag")

    # ═══════════════════════════════════════════════════════════════
    # 状态持久化
    # ═══════════════════════════════════════════════════════════════

    async def _persist_loop(self):
        """持久化循环"""
        while self._running:
            try:
                await asyncio.sleep(self._save_interval)
                if time.time() - self._last_save_time > self._save_interval:
                    self._save_state()
                    self._last_save_time = time.time()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Persist loop error: {e}")

    def _save_state(self):
        """保存仓位状态到文件"""
        try:
            os.makedirs(self._persist_dir, exist_ok=True)

            state = {
                "positions": {
                    key: {
                        "symbol": pos.symbol,
                        "side": pos.side.value,
                        "quantity": pos.quantity,
                        "avg_cost": pos.avg_cost,
                        "mark_price": pos.mark_price,
                        "unrealized_pnl": pos.unrealized_pnl,
                        "realized_pnl": pos.realized_pnl,
                        "margin": pos.margin,
                        "leverage": pos.leverage,
                        "liquidation_price": pos.liquidation_price,
                        "margin_ratio": pos.margin_ratio,
                        "strategy_name": pos.strategy_name,
                        "entry_time": pos.entry_time,
                        "status": pos.status.value,
                        "health_score": pos.health_score,
                        "tags": pos.tags,
                    }
                    for key, pos in self._positions.items()
                },
                "account_risk": {
                    "total_equity": self._account_risk.total_equity,
                    "available_balance": self._account_risk.available_balance,
                    "used_margin": self._account_risk.used_margin,
                    "margin_ratio": self._account_risk.margin_ratio,
                    "unrealized_pnl": self._account_risk.unrealized_pnl,
                    "drawdown_pct": self._account_risk.drawdown_pct,
                    "risk_level": self._account_risk.risk_level.value,
                    "position_count": self._account_risk.position_count,
                },
                "peak_equity": self._peak_equity,
                "drawdown_high_since": self._drawdown_high_since,
                "timestamp": datetime.now().isoformat(),
            }

            with open(self._persist_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)

            logger.debug(f"Position state saved: {len(self._positions)} positions")

        except Exception as e:
            logger.error(f"Failed to save position state: {e}")
            if self._alert_manager:
                try:
                    import asyncio
                    asyncio.create_task(self._alert_manager.send_alert(
                        alert_type="system_error",
                        message=f"持仓状态持久化失败（可能导致重启后仓位丢失）: {e}",
                        severity="WARNING",
                        symbol="SYSTEM",
                        metadata={"error": str(e)},
                    ))
                except Exception:
                    pass

    def _restore_state(self):
        """从文件恢复仓位状态"""
        try:
            if not os.path.exists(self._persist_file):
                return

            with open(self._persist_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            positions_data = state.get("positions", {})
            account_data = state.get("account_risk", {})

            self._peak_equity = state.get("peak_equity", 0.0)
            self._drawdown_high_since = state.get("drawdown_high_since", 0.0)

            for key, data in positions_data.items():
                try:
                    pos = PositionSnapshot(
                        symbol=data["symbol"],
                        side=PositionSide(data["side"]),
                        quantity=data["quantity"],
                        avg_cost=data["avg_cost"],
                        mark_price=data["mark_price"],
                        unrealized_pnl=data["unrealized_pnl"],
                        realized_pnl=data.get("realized_pnl", 0.0),
                        margin=data.get("margin", 0.0),
                        leverage=data.get("leverage", 1),
                        liquidation_price=data.get("liquidation_price", 0.0),
                        margin_ratio=data.get("margin_ratio", 0.0),
                        strategy_name=data.get("strategy_name", ""),
                        entry_time=data.get("entry_time", 0.0),
                        status=PositionStatus(data.get("status", "active")),
                        health_score=data.get("health_score", 1.0),
                        tags=data.get("tags", []),
                    )
                    self._positions[key] = pos
                    if pos.strategy_name:
                        self._positions_by_strategy[pos.strategy_name].append(key)
                    self._positions_by_symbol[pos.symbol].append(key)
                except Exception as e:
                    logger.debug(f"Failed to restore position {key}: {e}")

            if account_data:
                self._account_risk = AccountRiskSnapshot(
                    total_equity=account_data.get("total_equity", 0.0),
                    available_balance=account_data.get("available_balance", 0.0),
                    used_margin=account_data.get("used_margin", 0.0),
                    margin_ratio=account_data.get("margin_ratio", 0.0),
                    unrealized_pnl=account_data.get("unrealized_pnl", 0.0),
                    drawdown_pct=account_data.get("drawdown_pct", 0.0),
                    risk_level=RiskLevel(account_data.get("risk_level", "normal")),
                    position_count=account_data.get("position_count", 0),
                )

            logger.info(
                f"Position state restored: {len(self._positions)} positions, "
                f"peak_equity={self._peak_equity:.2f}"
            )

        except Exception as e:
            logger.error(f"Failed to restore position state: {e}")

    # ═══════════════════════════════════════════════════════════════
    # 查询接口
    # ═══════════════════════════════════════════════════════════════

    def get_position(self, symbol: str, side: str = "long") -> Optional[PositionSnapshot]:
        """获取指定持仓"""
        key = f"{symbol}:{side}"
        return self._positions.get(key)

    def get_all_positions(self) -> List[PositionSnapshot]:
        """获取所有持仓"""
        return list(self._positions.values())

    def get_positions_by_symbol(self, symbol: str) -> List[PositionSnapshot]:
        """获取指定币种的所有持仓"""
        keys = self._positions_by_symbol.get(symbol, [])
        return [self._positions[k] for k in keys if k in self._positions]

    def get_positions_by_strategy(self, strategy_name: str) -> List[PositionSnapshot]:
        """获取指定策略的所有持仓"""
        keys = self._positions_by_strategy.get(strategy_name, [])
        return [self._positions[k] for k in keys if k in self._positions]

    def get_account_risk(self) -> AccountRiskSnapshot:
        """获取账户风险快照"""
        return self._account_risk

    def is_data_stale(self) -> bool:
        """持仓数据是否过期（解析失败后未成功同步）"""
        return self._data_stale

    def get_risk_events(self, limit: int = 50) -> List[RiskEvent]:
        """获取最近的风险事件"""
        return self._risk_events[-limit:]

    def get_total_exposure(self) -> float:
        """获取总风险敞口"""
        return sum(
            pos.quantity * pos.mark_price
            for pos in self._positions.values()
        )

    def get_position_count(self) -> int:
        """获取当前持仓数量"""
        return len(self._positions)

    def is_emergency(self) -> bool:
        """是否处于紧急状态"""
        return self._emergency_reduce

    def get_stats(self) -> Dict[str, Any]:
        """获取仓位管理统计"""
        positions = self.get_all_positions()
        total_pnl = sum(p.unrealized_pnl for p in positions)
        total_margin = sum(p.margin for p in positions)

        healthy = sum(1 for p in positions if p.health_score >= 0.7)
        warning = sum(1 for p in positions if 0.4 <= p.health_score < 0.7)
        critical = sum(1 for p in positions if p.health_score < 0.4)

        return {
            "total_positions": len(positions),
            "total_exposure": self.get_total_exposure(),
            "total_unrealized_pnl": round(total_pnl, 4),
            "total_margin": round(total_margin, 4),
            "account_risk": {
                "total_equity": round(self._account_risk.total_equity, 2),
                "margin_ratio": round(self._account_risk.margin_ratio, 4),
                "drawdown_pct": round(self._account_risk.drawdown_pct, 4),
                "risk_level": self._account_risk.risk_level.value,
            },
            "health": {
                "healthy": healthy,
                "warning": warning,
                "critical": critical,
            },
            "emergency_reduce": self._emergency_reduce,
            "risk_events_today": len([
                e for e in self._risk_events
                if e.timestamp > time.time() - 86400
            ]),
            "timestamp": datetime.now().isoformat(),
        }

    def set_position_strategy(self, symbol: str, side: str, strategy_name: str):
        """设置持仓的策略关联"""
        key = f"{symbol}:{side}"
        if key in self._positions:
            self._positions[key].strategy_name = strategy_name
            if key not in self._positions_by_strategy[strategy_name]:
                self._positions_by_strategy[strategy_name].append(key)

    def set_position_entry_time(self, symbol: str, side: str, entry_time: float):
        """设置持仓入场时间"""
        key = f"{symbol}:{side}"
        if key in self._positions:
            self._positions[key].entry_time = entry_time

    def remove_position(self, symbol: str, side: str):
        """移除持仓记录"""
        key = f"{symbol}:{side}"
        if key in self._positions:
            pos = self._positions.pop(key)
            if pos.strategy_name:
                self._positions_by_strategy[pos.strategy_name] = [
                    k for k in self._positions_by_strategy[pos.strategy_name]
                    if k != key
                ]
            self._positions_by_symbol[symbol] = [
                k for k in self._positions_by_symbol[symbol]
                if k != key
            ]
            self._notify_position_change()

    # ═══════════════════════════════════════════════════════════════
    # 生产级跨策略协调 (v2.0)
    # ═══════════════════════════════════════════════════════════════

    def detect_cross_strategy_conflicts(self) -> List[Dict[str, Any]]:
        """检测跨策略持仓冲突
        
        冲突类型：
        - 同币种多策略同向持仓（保证金浪费）
        - 同币种多策略反向持仓（对冲风险）
        - 单策略持仓超限
        """
        conflicts = []
        
        # 检查同币种多策略持仓
        for symbol, keys in self._positions_by_symbol.items():
            if len(keys) <= 1:
                continue
            
            positions = [self._positions[k] for k in keys if k in self._positions]
            if len(positions) <= 1:
                continue
            
            # 按方向分组
            long_positions = [p for p in positions if p.side == PositionSide.LONG]
            short_positions = [p for p in positions if p.side == PositionSide.SHORT]
            
            # 同方向多策略持仓 —— 保证金效率低
            if len(long_positions) > 1:
                strategies = list(set(p.strategy_name for p in long_positions))
                total_exposure = sum(p.quantity * p.mark_price for p in long_positions)
                conflicts.append({
                    "type": "same_direction_duplicate",
                    "severity": "warning",
                    "symbol": symbol,
                    "direction": "long",
                    "strategies": strategies,
                    "position_count": len(long_positions),
                    "total_exposure": round(total_exposure, 2),
                    "total_margin": round(sum(p.margin for p in long_positions), 2),
                    "suggestion": "Consider consolidating into single strategy or reducing total exposure",
                })
            
            if len(short_positions) > 1:
                strategies = list(set(p.strategy_name for p in short_positions))
                total_exposure = sum(p.quantity * p.mark_price for p in short_positions)
                conflicts.append({
                    "type": "same_direction_duplicate",
                    "severity": "warning",
                    "symbol": symbol,
                    "direction": "short",
                    "strategies": strategies,
                    "position_count": len(short_positions),
                    "total_exposure": round(total_exposure, 2),
                    "total_margin": round(sum(p.margin for p in short_positions), 2),
                    "suggestion": "Consider consolidating into single strategy or reducing total exposure",
                })
            
            # 反向持仓 —— 对冲浪费
            if long_positions and short_positions:
                long_strategies = list(set(p.strategy_name for p in long_positions))
                short_strategies = list(set(p.strategy_name for p in short_positions))
                long_exposure = sum(p.quantity * p.mark_price for p in long_positions)
                short_exposure = sum(p.quantity * p.mark_price for p in short_positions)
                net_exposure = long_exposure - short_exposure
                conflicts.append({
                    "type": "hedging_conflict",
                    "severity": "critical",
                    "symbol": symbol,
                    "long_strategies": long_strategies,
                    "short_strategies": short_strategies,
                    "long_exposure": round(long_exposure, 2),
                    "short_exposure": round(short_exposure, 2),
                    "net_exposure": round(net_exposure, 2),
                    "wasted_margin": round(
                        sum(p.margin for p in long_positions + short_positions), 2
                    ),
                    "suggestion": f"Net exposure {net_exposure:.2f} USDT — close offsetting positions to free margin",
                })
        
        # 检查单策略持仓超限
        for strategy, keys in self._positions_by_strategy.items():
            if len(keys) > self._max_positions_per_strategy:
                positions = [self._positions[k] for k in keys if k in self._positions]
                symbols = [p.symbol for p in positions]
                total_exposure = sum(p.quantity * p.mark_price for p in positions)
                conflicts.append({
                    "type": "strategy_limit_exceeded",
                    "severity": "warning",
                    "strategy": strategy,
                    "position_count": len(positions),
                    "max_allowed": self._max_positions_per_strategy,
                    "symbols": symbols,
                    "total_exposure": round(total_exposure, 2),
                    "suggestion": f"Reduce positions from {len(positions)} to {self._max_positions_per_strategy}",
                })
        
        return conflicts

    def get_position_concentration(self) -> Dict[str, Any]:
        """获取持仓集中度分析
        
        Returns:
            concentration: 按币种和策略的集中度指标
        """
        if not self._positions:
            return {
                "concentration": {},
                "hhi": 0.0,
                "top_symbols": [],
                "risk_level": "normal",
            }
        
        total_exposure = sum(
            p.quantity * p.mark_price for p in self._positions.values()
        )
        if total_exposure <= 0:
            return {
                "concentration": {},
                "hhi": 0.0,
                "top_symbols": [],
                "risk_level": "normal",
            }
        
        # 按币种计算集中度
        symbol_exposure = {}
        for symbol, keys in self._positions_by_symbol.items():
            exposure = sum(
                self._positions[k].quantity * self._positions[k].mark_price
                for k in keys if k in self._positions
            )
            symbol_exposure[symbol] = exposure
        
        # HHI (Herfindahl-Hirschman Index) 集中度指数
        hhi = sum(
            (exp / total_exposure) ** 2
            for exp in symbol_exposure.values()
        )
        
        # 按策略计算集中度
        strategy_exposure = {}
        for strategy, keys in self._positions_by_strategy.items():
            exposure = sum(
                self._positions[k].quantity * self._positions[k].mark_price
                for k in keys if k in self._positions
            )
            strategy_exposure[strategy] = exposure
        
        # 排序
        top_symbols = sorted(
            symbol_exposure.items(), key=lambda x: x[1], reverse=True
        )[:3]
        
        # 风险判断
        if hhi > 0.5:
            risk_level = "critical"
        elif hhi > 0.3:
            risk_level = "high"
        elif hhi > 0.15:
            risk_level = "elevated"
        else:
            risk_level = "normal"
        
        return {
            "concentration": {
                "by_symbol": {
                    s: round(e / total_exposure, 4)
                    for s, e in symbol_exposure.items()
                },
                "by_strategy": {
                    s: round(e / total_exposure, 4)
                    for s, e in strategy_exposure.items()
                },
            },
            "hhi": round(hhi, 4),
            "top_symbols": [
                {"symbol": s, "exposure": round(e, 2), "pct": round(e / total_exposure, 4)}
                for s, e in top_symbols
            ],
            "risk_level": risk_level,
            "total_exposure": round(total_exposure, 2),
            "symbol_count": len(symbol_exposure),
            "strategy_count": len(strategy_exposure),
        }

    def get_optimal_position_close_order(self, target_exposure_reduction: float) -> List[Dict[str, Any]]:
        """生成最优减仓顺序 —— 按亏损程度和风险评分排序
        
        Args:
            target_exposure_reduction: 目标减少的风险敞口（USDT）
        
        Returns:
            按优先级排序的减仓建议列表
        """
        if not self._positions:
            return []
        
        # 计算每个持仓的优先级评分
        scored_positions = []
        for key, pos in self._positions.items():
            exposure = pos.quantity * pos.mark_price
            if exposure <= 0:
                continue
            
            # 评分因素：
            # 1. 亏损越大越优先减仓（权重 0.4）
            # 2. 健康评分越低越优先（权重 0.3）
            # 3. 保证金占用越高越优先（权重 0.2）
            # 4. 清算距离越近越优先（权重 0.1）
            
            loss_score = 0.0
            if pos.avg_cost > 0 and pos.quantity > 0:
                loss_pct = abs(pos.unrealized_pnl) / (pos.avg_cost * pos.quantity)
                loss_score = min(loss_pct * 10, 1.0)  # 10%亏损 = 满分
            
            health_score = 1.0 - pos.health_score
            
            margin_score = pos.margin / max(self._account_risk.total_equity, 1.0)
            
            liq_score = 0.0
            if pos.liquidation_price > 0 and pos.mark_price > 0:
                liq_dist = abs(pos.mark_price - pos.liquidation_price) / pos.mark_price
                liq_score = max(0, 1.0 - liq_dist * 20)  # 5%距离 = 满分
            
            priority = (
                loss_score * 0.4
                + health_score * 0.3
                + margin_score * 0.2
                + liq_score * 0.1
            )
            
            scored_positions.append({
                "symbol": pos.symbol,
                "side": pos.side.value,
                "strategy": pos.strategy_name,
                "exposure": round(exposure, 2),
                "unrealized_pnl": round(pos.unrealized_pnl, 4),
                "health_score": round(pos.health_score, 2),
                "margin": round(pos.margin, 2),
                "liquidation_price": pos.liquidation_price,
                "priority": round(priority, 4),
            })
        
        # 按优先级降序排序
        scored_positions.sort(key=lambda x: x["priority"], reverse=True)
        
        # 选择足够的持仓来达到目标减仓量
        result = []
        cumulative = 0.0
        for pos in scored_positions:
            result.append(pos)
            cumulative += pos["exposure"]
            if cumulative >= target_exposure_reduction:
                break
        
        return result