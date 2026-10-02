"""负责全局风控：回撤、亏损限额、保证金预警与阶梯式减仓。"""
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from loguru import logger

from core.models import Position, AccountInfo, Signal
from configs.settings import get_currency_tier


class GlobalRiskControl:
    def __init__(self, config: Dict[str, Any], redis_cache, okx_client, alert_manager=None):
        self.config = config
        self.redis_cache = redis_cache
        self.okx_client = okx_client
        self._alert_manager = alert_manager  # 可选，通过 set_alert_manager 注入
        self._order_executor = None  # 风控前置：平仓/减仓信号走 order_executor 而非直接 place_order
        self._account_manager = None  # P1-⑤：单一权益基准源（通过 set_account_manager 注入）
        
        self._max_drawdown = config["trading"]["max_drawdown"]
        self._daily_max_loss = config["trading"]["daily_max_loss"]
        self._hourly_max_loss = config["trading"]["hourly_max_loss"]
        
        self._margin_call_threshold = config["risk"]["margin_call_threshold"]
        self._margin_warning_threshold = config["risk"]["margin_warning_threshold"]
        self._position_loss_threshold = config["risk"]["position_loss_threshold"]
        self._full_close_threshold = config["risk"]["full_close_threshold"]
        
        self._max_consecutive_losses = config["trading"]["max_consecutive_losses"]
        self._consecutive_loss_window = timedelta(hours=4)
        
        self._initial_equity = None
        self._peak_equity = 0  # 历史峰值权益（跨重启持久化）
        self._effective_peak = 0  # 阶梯风控有效峰值（取max(历史峰值, 当前权益)）
        self._current_equity = 0.0  # 当前权益（每次检查时更新，供导出使用）
        self._daily_start_equity = None
        self._hourly_start_equity = None
        self._last_daily_reset_date = None
        self._last_hourly_reset = None
        
        self._daily_pnl = 0.0
        self._hourly_pnl = 0.0
        
        self._consecutive_losses = 0
        self._last_loss_time = None
        self._is_paused = False
        self._pause_reason = None
        self._pause_time = None  # 暂停开始时间（用于自动恢复计时）
        self._last_resume_time = None  # 上次恢复时间

        # 阶梯式风控状态：记录已触发的阶梯，避免重复触发
        # 阶梯1: 回撤>=10% → 减仓20%
        # 阶梯2: 回撤>=15% → 减仓40%
        # 阶梯3: 回撤>=20% → 减仓60%
        # 阶梯4: 回撤>=25%(max_drawdown) → 全平+暂停
        # 紧急:  回撤>=50%(2倍阈值) → 强制全平
        self._tier1_threshold = 0.10
        self._tier2_threshold = 0.15
        self._tier3_threshold = 0.20
        self._tier_triggered = {1: False, 2: False, 3: False}
        # 阶梯重置：当回撤恢复到5%以内时，重置阶梯状态允许下次再次触发
        self._tier_reset_threshold = 0.05
        # TIER 冷却期：恢复后 30 分钟内不允许重新触发，给仓位重建时间
        self._tier_cooldown_seconds = 1800
        self._tier_last_recovered = {1: 0, 2: 0, 3: 0}
        
        self._circuit_breakers = CircuitBreakers(config, okx_client)
        self._breaker_linker = None  # P1-⑥：熔断联动器（通过 set_breaker_linker 注入）
        self._monitor_task = None  # 监控循环 task 引用（幂等启停 + 优雅取消）

    def set_order_executor(self, order_executor):
        """注入订单执行器（风控前置：平仓/减仓信号走 order_executor 经五层风控校验）"""
        self._order_executor = order_executor
        logger.info("OrderExecutor injected into GlobalRiskControl for pre-trade risk gate")

    def set_account_manager(self, account_manager):
        """注入 AccountManager 作为单一权益基准源（P1-⑤）。"""
        self._account_manager = account_manager
        if hasattr(self._circuit_breakers, "set_account_manager"):
            self._circuit_breakers.set_account_manager(account_manager)
        logger.info("AccountManager injected into GlobalRiskControl as single equity benchmark")

    def set_breaker_linker(self, breaker_linker):
        """注入 CircuitBreakerLinker，统一熔断写路径（P1-⑥）。

        当 GlobalRiskControl 的风控熔断（急速回撤/BTC波动/强平潮等）触发时，
        同步通知策略协调层的 CircuitBreakerLinker，使策略级阻断与全局暂停
        保持一致，消除「两套熔断实现并存、写路径不统一」的问题。
        """
        self._breaker_linker = breaker_linker
        logger.info("CircuitBreakerLinker injected into GlobalRiskControl for unified breaker write path")

    async def start(self):
        # 幂等启动：避免重复调用创建多套监控循环
        if self._monitor_task is not None and not self._monitor_task.done():
            return
        await self._initialize_equity()
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop(self):
        task = self._monitor_task
        self._monitor_task = None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _initialize_equity(self):
        account_info = self.okx_client.get_account_info()
        if account_info:
            account = self.okx_client._parse_account_info(account_info)
            self._initial_equity = account.total_equity
            self._daily_start_equity = account.total_equity
            self._hourly_start_equity = account.total_equity

            # 从数据库加载历史峰值权益
            self._peak_equity = account.total_equity
            try:
                import sqlite3
                conn = sqlite3.connect(self.config["sqlite"]["db_path"])
                c = conn.cursor()
                c.execute("SELECT MAX(total_equity) FROM account_history")
                row = c.fetchone()
                if row and row[0] and row[0] > self._peak_equity:
                    self._peak_equity = row[0]
                conn.close()
            except Exception as e:
                logger.warning(f"Failed to load peak equity from DB: {e}")

            # 历史峰值保留用于审计，effective_peak 用于阶梯风控
            # 如果启动时回撤已超过 tier2 阈值（15%），将 effective_peak 重置为当前权益
            # 避免历史峰值过高导致 TIER3 死亡螺旋（反复触发-恢复-再触发）
            current_drawdown = (self._peak_equity - account.total_equity) / self._peak_equity if self._peak_equity > 0 else 0
            if current_drawdown >= self._tier2_threshold:
                self._effective_peak = account.total_equity
                logger.warning(
                    f"Startup drawdown {current_drawdown:.2%} >= tier2 {self._tier2_threshold:.2%} "
                    f"(peak={self._peak_equity:.2f}, current={account.total_equity:.2f}). "
                    f"Effective peak reset to current equity to break drawdown death spiral. "
                    f"Historical peak preserved for audit."
                )
            else:
                self._effective_peak = self._peak_equity

            logger.info(f"Initial equity: {self._initial_equity:.2f}, Peak equity: {self._peak_equity:.2f}, Effective: {self._effective_peak:.2f}")

    async def _monitor_loop(self):
        # 每个检查独立兜底：单个检查异常不能杀死整个风控监控循环（fail-safe）
        while True:
            try:
                await self._check_account_risk()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in _check_account_risk: {e}")
            try:
                await self._check_position_risk()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in _check_position_risk: {e}")
            try:
                await self._check_consecutive_losses()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in _check_consecutive_losses: {e}")
            try:
                await self._check_circuit_breakers()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in _check_circuit_breakers: {e}")
            # 检查是否有手动重置指令
            await asyncio.to_thread(self._check_manual_controls)
            # 消费告警引擎/dashboard 下发的干预信号（告警 → 交易动作闭环）
            await asyncio.to_thread(self._check_alert_intervention)
            # 导出风控状态到JSON文件，供dashboard读取真实状态（放到线程池避免阻塞事件循环）
            await asyncio.to_thread(self._export_risk_status)
            await asyncio.sleep(5)

    def _check_manual_controls(self):
        """检查手动控制指令文件（如重置暂停、重置峰值）"""
        try:
            control_path = "./data/risk_control.json"
            import os
            import json
            if not os.path.exists(control_path):
                return
            with open(control_path, "r", encoding="utf-8") as f:
                control = json.load(f)
            action = control.get("action")
            if action == "reset_pause":
                self._is_paused = False
                self._pause_reason = None
                self._consecutive_losses = 0
                self._daily_pnl = 0
                self._hourly_pnl = 0
                self._tier_triggered = {1: False, 2: False, 3: False}
                logger.info("Manual reset: pause cleared via risk_control.json")
                try:
                    os.remove(control_path)
                except OSError:
                    pass
            elif action == "reset_peak":
                old_peak = self._peak_equity
                old_effective = self._effective_peak
                new_peak = control.get("new_peak", None)
                if new_peak and new_peak > 0:
                    self._peak_equity = float(new_peak)
                    self._effective_peak = float(new_peak)
                else:
                    self._peak_equity = self._current_equity
                    self._effective_peak = self._current_equity
                self._tier_triggered = {1: False, 2: False, 3: False}
                self._is_paused = False
                self._pause_reason = None
                self._consecutive_losses = 0
                logger.info(f"Manual peak reset: {old_peak:.2f} -> {self._peak_equity:.2f} "
                           f"(effective: {old_effective:.2f} -> {self._effective_peak:.2f})")
                try:
                    os.remove(control_path)
                except OSError:
                    pass
        except Exception as e:
            logger.debug(f"Failed to check manual controls: {e}")

    def _check_alert_intervention(self):
        """消费干预信号文件，形成「告警 → 交易动作」完整闭环。

        仅自动执行安全、可逆的动作：
        - global_pause：暂停开新仓（通过 global_resume / reset_pause 解除）
        - global_resume：仅当当前暂停确由干预信号引起时解除，避免突破熔断等硬暂停

        不可逆动作（emergency_close_all / cancel_all_orders 等）已在 dashboard
        端点内直接执行，此处仅记录并删除信号，避免 5s 监控循环重复处理。
        """
        try:
            import os
            import json
            sig_path = "./data/intervention_signal.json"
            if not os.path.exists(sig_path):
                return
            with open(sig_path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            if not raw:
                return
            sig = json.loads(raw)
            action = sig.get("action")
            source = sig.get("source", "")
            reason = sig.get("reason", "") or "干预信号"

            if action == "global_pause":
                if not self._is_paused:
                    self._is_paused = True
                    self._pause_reason = f"[干预·{source}] {reason}"
                    self._pause_time = datetime.now()
                    logger.warning(f"[告警闭环] 已消费干预信号 global_pause，暂停开新仓: {self._pause_reason}")
                else:
                    logger.debug(f"[告警闭环] global_pause 信号已消费，但系统已暂停（当前原因: {self._pause_reason}）")
            elif action == "global_resume":
                if self._is_paused and self._pause_reason and str(self._pause_reason).startswith("[干预·"):
                    self._is_paused = False
                    self._last_resume_time = datetime.now()
                    self._pause_reason = None
                    logger.info("[告警闭环] 已消费干预信号 global_resume，恢复交易")
                else:
                    logger.warning(
                        f"[告警闭环] global_resume 被拒绝：当前暂停原因非干预信号 "
                        f"(reason={self._pause_reason})"
                    )
            else:
                # emergency_close_all / cancel_all_orders 已由 dashboard 直接执行，不重复执行
                logger.info(f"[告警闭环] 忽略非自动执行干预动作: {action}（source={source}）")

            # 消费后删除信号文件，避免重复处理
            try:
                os.remove(sig_path)
            except OSError:
                pass
        except Exception as e:
            logger.debug(f"Failed to check alert intervention: {e}")

    def _export_risk_status(self):
        """导出当前风控状态到JSON文件，供dashboard实时读取"""
        try:
            import json
            import os
            # 从CircuitBreakers获取详细回撤状态
            cb_status = {}
            try:
                cb_status = self._circuit_breakers.get_drawdown_status()
            except Exception:
                pass
            
            status = {
                "current_equity": self._current_equity,
                "peak_equity": self._peak_equity,
                "effective_peak": self._effective_peak,
                "initial_capital": self.config.get("trading", {}).get("total_capital", 100),
                "max_drawdown": self._max_drawdown,
                "daily_max_loss": self._daily_max_loss,
                "hourly_max_loss": self._hourly_max_loss,
                "is_paused": self._is_paused,
                "pause_reason": self._pause_reason,
                "pause_time": self._pause_time.isoformat() if self._pause_time else None,
                "last_resume_time": self._last_resume_time.isoformat() if self._last_resume_time else None,
                "tier_triggered": self._tier_triggered,
                "tier_thresholds": {
                    "tier1": self._tier1_threshold,
                    "tier2": self._tier2_threshold,
                    "tier3": self._tier3_threshold,
                },
                "consecutive_losses": self._consecutive_losses,
                "daily_pnl": self._daily_pnl,
                "hourly_pnl": self._hourly_pnl,
                "current_drawdown": (self._effective_peak - self._current_equity) / self._effective_peak if self._effective_peak > 0 else 0,
                "last_update": datetime.now().isoformat(),
                "process_running": True,
                "circuit_breakers": cb_status,
            }
            status_path = "./data/risk_status.json"
            os.makedirs(os.path.dirname(status_path), exist_ok=True)
            with open(status_path, "w", encoding="utf-8") as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug(f"Failed to export risk status: {e}")

    async def _check_account_risk(self):
        # 单一权益基准源（P1-⑤）：优先从 AccountManager 读取当前权益，消除多源分歧。
        if self._account_manager is not None:
            equity = self._account_manager.get_total_equity()
            if equity <= 0:
                # 权益归零/爆仓：触发紧急平仓 + 暂停，而非静默短路放行
                logger.critical(f"Account equity {equity} <= 0, triggering emergency liquidation")
                await self._force_liquidation()
                self._is_paused = True
                self._pause_reason = f"Account equity {equity} <= 0 (bankruptcy protection)"
                if self._alert_manager:
                    asyncio.create_task(self._alert_manager.send_alert(
                        "EQUITY_BANKRUPT", f"Account equity {equity} <= 0, forced liquidation",
                        severity="CRITICAL", symbol="ALL"
                    ))
                return
        else:
            account_info = self.okx_client.get_account_info()
            if not account_info:
                return
            account = self.okx_client._parse_account_info(account_info)
            self.redis_cache.set_account_info(account)
            equity = account.total_equity
        
        if self._initial_equity is None:
            self._initial_equity = equity
        
        current_time = datetime.now()

        # 用日期比较判断日切换，避免5秒循环错过00:00
        current_date = current_time.date()
        if self._last_daily_reset_date != current_date:
            self._daily_start_equity = equity
            self._last_daily_reset_date = current_date

        current_hour = current_time.replace(minute=0, second=0, microsecond=0)
        if self._last_hourly_reset != current_hour:
            self._hourly_start_equity = equity
            self._last_hourly_reset = current_hour
        
        self._daily_pnl = equity - self._daily_start_equity
        self._hourly_pnl = equity - self._hourly_start_equity

        # 保存当前权益，供导出使用
        self._current_equity = equity

        if equity > self._peak_equity:
            self._peak_equity = equity
        
        # 更新effective_peak：用于阶梯风控，取max(历史峰值, 当前权益)
        # 当前权益>历史峰值时：真实新高，两者同步
        # 当前权益<历史峰值时：effective_peak保持不变（阶梯风控基于历史峰值）
        if equity > self._effective_peak and equity > self._peak_equity:
            self._effective_peak = equity
        
        # 回撤计算：从effective_peak计算（注入资金后不会误触发，亏损时正确反应）
        drawdown = (self._effective_peak - equity) / self._effective_peak if self._effective_peak > 0 else 0
        
        # ============ 阶梯式风控：分级减仓 + 分级恢复 ============
        # 分级恢复阈值：回撤回落时逐步恢复仓位（每次恢复一级）
        # tier1减20%，恢复阈值=8%；tier2减40%，恢复阈值=12%；tier3减60%，恢复阈值=17%
        tier_recover_thresholds = {
            1: self._tier1_threshold * 0.8,  # 8%
            2: self._tier2_threshold * 0.8,  # 12%
            3: self._tier3_threshold * 0.85, # 17%
        }
        
        # 分级恢复：从最高级开始检查，回撤恢复后逐级恢复
        if any(self._tier_triggered.values()):
            for tier in sorted(self._tier_triggered.keys(), reverse=True):
                if not self._tier_triggered[tier]:
                    continue
                recover_thresh = tier_recover_thresholds.get(tier, 0.05)
                if drawdown < recover_thresh:
                    # 恢复该阶梯：重新加仓该阶梯减掉的部分
                    reduce_ratio = {1: 0.20, 2: 0.40, 3: 0.60}.get(tier, 0.20)
                    logger.info(f"TIER {tier} RECOVERY: Drawdown {drawdown:.2%} < {recover_thresh:.2%}, "
                               f"restoring {reduce_ratio*100:.0f}% positions")
                    # 恢复仓位通过新信号自然建仓，这里只解除限制标记
                    self._tier_triggered[tier] = False
                    # 记录恢复时间，进入冷却期
                    self._tier_last_recovered[tier] = time.time()

        # 阶梯3: 回撤>=20% → 减仓60%（最高阶梯，触发后不再重复）
        # 冷却期内不重复触发，给仓位重建时间
        tier3_in_cooldown = (time.time() - self._tier_last_recovered[3]) < self._tier_cooldown_seconds
        if drawdown >= self._tier3_threshold and not self._tier_triggered[3] and not self._is_paused and not tier3_in_cooldown:
            logger.warning(f"TIER 3 RISK: Drawdown {drawdown:.2%} >= {self._tier3_threshold:.2%}, reducing 60% positions")
            await self._reduce_all_positions(0.60, "tier3_drawdown")
            self._tier_triggered[3] = True

        # 阶梯2: 回撤>=15% → 减仓40%
        elif drawdown >= self._tier2_threshold and not self._tier_triggered[2] and not self._is_paused:
            logger.warning(f"TIER 2 RISK: Drawdown {drawdown:.2%} >= {self._tier2_threshold:.2%}, reducing 40% positions")
            await self._reduce_all_positions(0.40, "tier2_drawdown")
            self._tier_triggered[2] = True

        # 阶梯1: 回撤>=10% → 减仓20%
        elif drawdown >= self._tier1_threshold and not self._tier_triggered[1] and not self._is_paused:
            logger.warning(f"TIER 1 RISK: Drawdown {drawdown:.2%} >= {self._tier1_threshold:.2%}, reducing 20% positions")
            await self._reduce_all_positions(0.20, "tier1_drawdown")
            self._tier_triggered[1] = True

        # ============ 原有阈值风控 ============
        # 紧急回撤保护：超过阈值的2倍立即停止所有交易
        emergency_drawdown = drawdown >= self._max_drawdown * 2
        
        daily_loss_ratio = (self._daily_pnl / self._daily_start_equity) if self._daily_start_equity else 0.0
        hourly_loss_ratio = (self._hourly_pnl / self._hourly_start_equity) if self._hourly_start_equity else 0.0
        checks = [
            ("max_drawdown", drawdown >= self._max_drawdown, 
             f"Max drawdown {drawdown:.2%} exceeded threshold {self._max_drawdown:.2%} (peak={self._peak_equity:.2f}, current={equity:.2f})", self._handle_max_drawdown),
            ("daily_loss", daily_loss_ratio <= -self._daily_max_loss,
             f"Daily loss {self._daily_pnl:.2f} exceeded threshold {self._daily_max_loss:.2%}", self._handle_daily_loss),
            ("hourly_loss", hourly_loss_ratio <= -self._hourly_max_loss,
             f"Hourly loss {self._hourly_pnl:.2f} exceeded threshold {self._hourly_max_loss:.2%}", self._handle_hourly_loss),
        ]
        
        for check_name, condition, message, handler in checks:
            if condition and not self._is_paused:
                logger.error(message)
                await handler()

        # 紧急回撤保护：立即强制平仓并暂停
        if emergency_drawdown and not self._is_paused:
            logger.critical(f"EMERGENCY DRAWDOWN PROTECTION: {drawdown:.2%} >= {self._max_drawdown*2:.2%}, force liquidating ALL positions")
            await self._force_liquidation()
            self._is_paused = True
            self._pause_reason = f"Emergency drawdown protection ({drawdown:.2%})"

    async def _check_position_risk(self):
        positions = self.okx_client.get_positions()
        if not positions:
            return

        # 获取账户总权益用于计算分品种仓位占比
        account_info = self.okx_client.get_account_info()
        try:
            total_equity = float(account_info.get("totalEq") or 0) if account_info else 0.0
        except (TypeError, ValueError):
            total_equity = 0.0

        # 按symbol聚合持仓保证金（同symbol可能有多策略）
        symbol_margins: Dict[str, float] = {}
        parsed_positions = []

        for pos_data in positions:
            position = self.okx_client._parse_position(pos_data)
            if not position:
                continue

            if position.margin <= 0:
                continue

            # 极小保证金仓位（<1 USDT）跳过亏损比率检查，仅依赖 liqPx 距离检测
            # 避免因 DB 恢复仓位或 cross-margin 模式下 margin 估算失真导致的误报
            margin_too_small = position.margin < 1.0

            parsed_positions.append(position)
            self.redis_cache.set_position(position)

            # 聚合同symbol保证金
            if position.symbol not in symbol_margins:
                symbol_margins[position.symbol] = 0
            symbol_margins[position.symbol] += position.margin

            # 单仓位亏损检查（原有逻辑）
            if position.unrealized_pnl < -position.margin * self._position_loss_threshold:
                logger.warning(f"Position loss {abs(position.unrealized_pnl)/position.margin:.2%} exceeds threshold for {position.symbol}")
                await self._handle_position_loss(position)

            if position.unrealized_pnl < -position.margin * self._full_close_threshold:
                logger.error(f"Position loss {abs(position.unrealized_pnl)/position.margin:.2%} exceeds full close threshold for {position.symbol}")
                await self._close_position(position)

            # 保证金监控：亏损占保证金比例，接近维持保证金率时预警（极小保证金仓位跳过）
            if not margin_too_small and position.margin > 0 and position.unrealized_pnl < 0:
                loss_ratio = abs(position.unrealized_pnl) / position.margin
                if loss_ratio > 0.7:
                    logger.critical(f"MARGIN WARNING: {position.symbol} loss_ratio={loss_ratio:.2%}, close to liquidation")
                    if loss_ratio > 0.8:
                        logger.error(f"Margin ratio critical for {position.symbol}, auto-closing")
                        await self._close_position(position)

            # 逐仓风险隔离：基于OKX强平价(liqPx)做距离监控，提前主动减仓避免强平费
            # isolated模式下，单仓强平不影响其他仓位，但仍要避免真的爆仓产生强平费
            try:
                liq_px_str = pos_data.get("liqPx", "") or pos_data.get("liqPx", "0")
                liq_px = float(liq_px_str) if liq_px_str else 0.0
                mark_px = position.mark_price
                if liq_px > 0 and mark_px > 0:
                    # 强平距离 = |mark - liq| / mark
                    liq_distance = abs(mark_px - liq_px) / mark_px
                    # 距离<3%：极危险，立即全平该仓位（隔离爆仓，不影响其他）
                    if liq_distance < 0.03:
                        logger.critical(
                            f"LIQUIDATION IMMINENT: {position.symbol} ({position.side}) "
                            f"mark={mark_px:.4f} liq={liq_px:.4f} distance={liq_distance:.2%}, force closing"
                        )
                        await self._close_position(position)
                    # 距离<6%：危险，减仓50%降低风险
                    elif liq_distance < 0.06:
                        logger.error(
                            f"LIQUIDATION WARNING: {position.symbol} ({position.side}) "
                            f"mark={mark_px:.4f} liq={liq_px:.4f} distance={liq_distance:.2%}, reducing 50%"
                        )
                        await self._reduce_position(position, 0.50)
                    # 距离<10%：警告
                    elif liq_distance < 0.10:
                        if self._alert_manager:
                            asyncio.create_task(
                                self._alert_manager.send_alert(
                                    "LIQUIDATION_WARNING",
                                    f"{position.symbol} ({position.side}) mark={mark_px:.4f} liq={liq_px:.4f} distance={liq_distance:.1%}",
                                    severity="WARNING",
                                    symbol=position.symbol
                                )
                            )
                        logger.warning(
                            f"LIQUIDATION ALERT: {position.symbol} ({position.side}) "
                            f"mark={mark_px:.4f} liq={liq_px:.4f} distance={liq_distance:.2%}"
                        )
            except (ValueError, TypeError):
                pass

        # 分品种仓位限制检查：单品种保证金不超过总权益的25%
        if total_equity > 0:
            max_single_symbol_ratio = 0.25  # 单品种最多占总权益25%
            for symbol, margin in symbol_margins.items():
                ratio = margin / total_equity
                if ratio > max_single_symbol_ratio:
                    logger.warning(f"Single symbol {symbol} margin ratio {ratio:.2%} > {max_single_symbol_ratio:.2%}, "
                                   f"margin={margin:.2f}, equity={total_equity:.2f}")
                    # 超限时减仓到阈值内
                    # 找到该symbol的持仓
                    for pos in parsed_positions:
                        if pos.symbol == symbol:
                            excess_ratio = (ratio - max_single_symbol_ratio) / ratio
                            if excess_ratio > 0.1:  # 超过10%才减
                                logger.warning(f"Reducing {symbol} by {excess_ratio*100:.0f}% due to single symbol limit")
                                await self._reduce_position(pos, excess_ratio)

    async def _check_consecutive_losses(self):
        try:
            orders = self.okx_client.get_order_history(limit=50)
        except Exception as e:
            logger.warning(f"Failed to fetch order history for consecutive loss check: {e}")
            return
        if not orders:
            return
        
        now = datetime.now()
        recent_orders = []
        for o in orders:
            try:
                update_ms = float(o.get("updateTime") or 0)
                if update_ms <= 0:
                    continue
                ts = datetime.fromtimestamp(update_ms / 1000)
            except (TypeError, ValueError, OSError, OverflowError):
                continue
            if (now - ts) <= self._consecutive_loss_window:
                recent_orders.append(o)
        
        closed_loss_orders = []
        for o in recent_orders:
            if o.get("state") != "filled":
                continue
            try:
                pnl = float(o.get("pnl") or 0)
            except (TypeError, ValueError):
                continue
            if pnl < 0:
                closed_loss_orders.append(o)
        
        self._consecutive_losses = len(closed_loss_orders)
        
        if self._consecutive_losses >= self._max_consecutive_losses:
            logger.error(f"Consecutive losses {self._consecutive_losses} exceeded threshold {self._max_consecutive_losses}")
            await self._handle_consecutive_losses()

    async def _handle_max_drawdown(self):
        if self._alert_manager:
            asyncio.create_task(self._alert_manager.send_alert(
                "MAX_DRAWDOWN", "Max drawdown exceeded, forcing full liquidation",
                severity="CRITICAL", symbol="ALL"
            ))
        logger.critical("Max drawdown exceeded, forcing full liquidation")
        await self._force_liquidation()
        self._is_paused = True
        self._pause_reason = "Max drawdown exceeded"

    async def _handle_daily_loss(self):
        if self._alert_manager:
            asyncio.create_task(self._alert_manager.send_alert(
                "DAILY_LOSS_LIMIT", "Daily max loss exceeded, stopping new positions",
                severity="CRITICAL", symbol="ALL"
            ))
        logger.critical("Daily max loss exceeded, stopping new positions")
        self._is_paused = True
        self._pause_reason = "Daily max loss exceeded"

    async def _handle_hourly_loss(self):
        if self._alert_manager:
            asyncio.create_task(self._alert_manager.send_alert(
                "HOURLY_LOSS_LIMIT", "Hourly max loss exceeded, pausing",
                severity="ERROR", symbol="ALL"
            ))
        logger.error("Hourly max loss exceeded, pausing aggressive strategies")
        self._is_paused = True
        self._pause_reason = "Hourly max loss exceeded"

    async def _handle_consecutive_losses(self):
        if self._alert_manager:
            asyncio.create_task(self._alert_manager.send_alert(
                "CONSECUTIVE_LOSSES", f"Consecutive losses ({self._consecutive_losses}) exceeded",
                severity="ERROR", symbol="ALL"
            ))
        logger.error("Consecutive losses exceeded, pausing trading")
        self._is_paused = True
        self._pause_reason = f"Consecutive losses ({self._consecutive_losses}) exceeded"

    async def _handle_margin_call(self, position: Position):
        logger.error(f"Margin call on {position.symbol}, closing position")
        await self._close_position(position)

    async def _handle_position_loss(self, position: Position):
        if position.unrealized_pnl < -position.margin * self._full_close_threshold:
            logger.error(f"Full close threshold reached for {position.symbol}")
            await self._close_position(position)
        else:
            logger.warning(f"Reducing position by 50% for {position.symbol}")
            await self._reduce_position(position, 0.5)

    async def _force_liquidation(self):
        """紧急强制平仓：分批次平仓避免砸盘，大仓位分2-3次"""
        try:
            positions = self.okx_client.get_positions()
            if not positions:
                logger.info("No positions to liquidate")
                return

            # 按仓位大小排序：先平小仓位（流动性好），再平大仓位
            parsed_positions = []
            for pos_data in positions:
                position = self.okx_client._parse_position(pos_data)
                if position and float(position.quantity) > 0:
                    parsed_positions.append(position)

            # 按保证金大小排序（小的先平）
            parsed_positions.sort(key=lambda p: p.margin)

            logger.critical(f"EMERGENCY LIQUIDATION: closing {len(parsed_positions)} positions in batches")

            for i, position in enumerate(parsed_positions):
                pos_qty = abs(float(position.quantity))
                pos_value = position.margin * position.leverage

                # 大仓位（保证金>5 USDT 或 价值>30 USDT）分批次平仓，避免滑点
                if position.margin > 5 or pos_value > 30:
                    # 分2次平仓：先平50%，等0.5秒再平剩余
                    logger.warning(f"Large position {position.symbol} (margin={position.margin:.2f}), batch closing")
                    await self._close_position(position, ratio=0.5)
                    await asyncio.sleep(0.5)
                    await self._close_position(position, ratio=1.0)  # 平剩余全部
                else:
                    # 小仓位直接平
                    await self._close_position(position)

                # 每次平仓间隔0.3秒，避免API限频
                await asyncio.sleep(0.3)

            logger.critical("Emergency liquidation completed")
        except Exception as e:
            logger.error(f"Error in _force_liquidation: {e}")

    async def _reduce_all_positions(self, ratio: float, reason: str = "tier_drawdown"):
        """阶梯式风控：按比例减仓所有持仓，避免一次性砸盘
        ratio: 减仓比例 (0.20=减20%, 0.40=减40%, 0.60=减60%)
        """
        try:
            positions = self.okx_client.get_positions()
            if not positions:
                logger.info(f"No positions to reduce for {reason}")
                return

            reduced_count = 0
            for pos_data in positions:
                position = self.okx_client._parse_position(pos_data)
                if not position or float(position.quantity) <= 0:
                    continue

                pos_qty = abs(float(position.quantity))
                reduce_qty = pos_qty * ratio
                if pos_qty > 0 and reduce_qty > 0:
                    await self._reduce_position(position, ratio)
                    reduced_count += 1
                    logger.info(f"{reason}: reduced {position.symbol} by {ratio*100:.0f}% (qty={reduce_qty:.4f})")
                    # 减仓间隔0.2秒，避免API限频
                    await asyncio.sleep(0.2)

            logger.warning(f"{reason} completed: reduced {reduced_count} positions by {ratio*100:.0f}%")
        except Exception as e:
            logger.error(f"Error in _reduce_all_positions ({reason}): {e}")

    async def _close_position(self, position: Position, ratio: float = 1.0):
        """平仓，支持部分平仓（ratio<1.0）
        ratio: 平仓比例，1.0=全部平仓，0.5=平50%

        风控前置架构：优先通过 order_executor 下单（经五层风控校验），
        order_executor 不可用时回退到直接 place_order（保留原有重试逻辑）。
        """
        side = "sell" if position.side == "long" else "buy"
        close_qty = abs(float(position.quantity)) * ratio
        if close_qty <= 0:
            return

        # 风控前置：优先走 order_executor（平仓模式，跳过L1保证金但保留L4/L5）
        if self._order_executor is not None:
            try:
                signal_type = "stop_loss" if ratio >= 1.0 else "reduce_position"
                close_signal = {
                    "signal_type": signal_type,
                    "symbol": position.symbol,
                    "direction": side,
                    "price": float(position.mark_price),
                    "quantity": close_qty,
                    "leverage": position.leverage,
                    "strategy_name": "global_risk",
                    "reduce_ratio": ratio,
                    "reason": f"global_risk close ratio={ratio:.2f}",
                    "priority": 999,
                    "timestamp": datetime.now().isoformat(),
                }
                await self._order_executor.handle_signal(close_signal)
                if ratio < 1.0:
                    logger.info(f"Partial close via order_executor: {position.symbol} {ratio*100:.0f}%")
                return
            except Exception as e:
                logger.error(f"order_executor close failed for {position.symbol}, fallback to direct place_order: {e}")

        # 回退：直接 place_order（order_executor 不可用时）
        for attempt in range(3):
            try:
                result = self.okx_client.place_order(
                    symbol=position.symbol,
                    side=side,
                    order_type="market",
                    quantity=close_qty,
                    leverage=position.leverage,
                    reduce_only=True,
                    pos_side=position.side
                )
                if result and result.get("_failed"):
                    error_code = result.get("sCode", "")
                    if error_code in {"50011", "50013", "50014"}:
                        logger.warning(f"Rate limited on {position.symbol}, switching API key")
                        self.okx_client._rotate_key()
                        continue
                    logger.error(f"Failed to close {position.symbol}: {result.get('sMsg', '')}")
                    break
                if ratio < 1.0:
                    logger.info(f"Partial close {position.symbol}: {ratio*100:.0f}% (qty={close_qty:.4f})")
                return
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed to close {position.symbol}: {e}")
                if attempt < 2:
                    await asyncio.sleep(0.5)
                    self.okx_client._rotate_key()
        logger.critical(f"Failed to close {position.symbol} after 3 attempts")

    async def _reduce_position(self, position: Position, ratio: float):
        """风控前置：减仓优先走 order_executor（经五层风控校验），回退到直接下单"""
        side = "sell" if position.side == "long" else "buy"
        reduce_quantity = abs(float(position.quantity)) * ratio

        if self._order_executor is not None:
            try:
                reduce_signal = {
                    "signal_type": "reduce_position",
                    "symbol": position.symbol,
                    "direction": side,
                    "price": float(position.mark_price),
                    "quantity": reduce_quantity,
                    "leverage": position.leverage,
                    "strategy_name": "global_risk",
                    "reduce_ratio": ratio,
                    "reason": f"global_risk reduce ratio={ratio:.2f}",
                    "priority": 500,
                    "timestamp": datetime.now().isoformat(),
                }
                await self._order_executor.handle_signal(reduce_signal)
                return
            except Exception as e:
                logger.error(f"order_executor reduce failed for {position.symbol}, fallback to direct: {e}")

        try:
            self.okx_client.place_order(
                symbol=position.symbol,
                side=side,
                order_type="market",
                quantity=reduce_quantity,
                leverage=position.leverage,
                reduce_only=True,
                pos_side=position.side
            )
        except Exception as e:
            logger.error(f"Failed to reduce {position.symbol}: {e}")

    async def _check_circuit_breakers(self):
        """分级响应黑天鹅事件：
        - minor: 仅暂停，60秒后自动恢复
        - major: 暂停+减仓30%，300秒后自动恢复
        - extreme: 暂停+强制全平，需手动reset恢复
        
        强化：恢复前确认市场已稳定（连续3次检查未触发），避免反复触发
        """
        result = await self._circuit_breakers.check_all()
        triggered = result.get("triggered", False)
        severity = result.get("severity")
        cb_name = result.get("name")

        if triggered:
            # 重置恢复计数器
            self._recovery_confirmed_count = 0
            
            if not self._is_paused:
                self._is_paused = True
                self._pause_reason = f"Circuit breaker: {cb_name} ({severity})"
                self._pause_time = datetime.now()
                logger.error(f"Trading paused: {self._pause_reason}")

                # P1-⑥：熔断统一 —— 同步通知策略协调层阻断策略
                self._notify_breaker_linker(severity, cb_name)

                if severity == "major":
                    # major级别：减仓30%防止损失扩大
                    logger.warning(f"MAJOR circuit breaker ({cb_name}), reducing all positions by 30%")
                    await self._reduce_all_positions(0.30, reason=f"major_circuit_breaker_{cb_name}")
                elif severity == "extreme":
                    # extreme级别：强制全平+持续暂停（需手动reset）
                    logger.critical(f"EXTREME circuit breaker ({cb_name}), force liquidating ALL positions")
                    await self._force_liquidation()
                    self._pause_reason = f"EXTREME Circuit breaker: {cb_name} (manual reset required)"
        else:
            # 仅minor/major级别的熔断暂停可自动恢复；extreme需手动reset
            # 干预信号（告警闭环）导致的暂停不在此自动恢复，需人工 global_resume / reset_pause
            if (self._is_paused and self._pause_reason
                    and "EXTREME" not in self._pause_reason
                    and not str(self._pause_reason).startswith("[干预·")):
                pause_duration = (datetime.now() - getattr(self, '_pause_time', datetime.now())).total_seconds()
                # minor: 60秒恢复，major: 300秒恢复
                is_major = "major" in (self._pause_reason or "")
                recover_threshold = 300 if is_major else 60
                
                if pause_duration > recover_threshold:
                    # 恢复确认：连续3次（约15秒）未触发才恢复，避免震荡市反复熔断
                    self._recovery_confirmed_count = getattr(self, '_recovery_confirmed_count', 0) + 1
                    required_confirmations = 3 if is_major else 2
                    if self._recovery_confirmed_count >= required_confirmations:
                        self._is_paused = False
                        self._pause_reason = None
                        self._last_resume_time = datetime.now()
                        self._recovery_confirmed_count = 0
                        logger.info(f"Circuit breaker cleared after {pause_duration:.0f}s "
                                   f"({required_confirmations} confirmations), resuming trading")
                        # P1-⑥：熔断恢复 —— 同步解除策略协调层阻断
                        await self._notify_breaker_linker_recovery()
                    else:
                        logger.debug(f"Circuit breaker recovery pending: "
                                    f"{self._recovery_confirmed_count}/{required_confirmations} confirmations")

    async def _notify_breaker_linker(self, severity: str, cb_name: str) -> None:
        """P1-⑥：将风控熔断事件同步到策略协调层 CircuitBreakerLinker。

        severity → level 映射：
        - minor  → 1（仅暂停，记录日志）
        - major  → 3（全局阻断，配合减仓30%）
        - extreme → 4（全局阻断 + 阻断所有策略，配合强制全平）
        """
        if self._breaker_linker is None:
            return
        level = {"minor": 1, "major": 3, "extreme": 4}.get(severity, 1)
        try:
            await self._breaker_linker.on_circuit_breaker(
                level, f"{cb_name} ({severity})"
            )
        except Exception as e:
            logger.error(f"Failed to notify CircuitBreakerLinker: {e}")

    async def _notify_breaker_linker_recovery(self) -> None:
        """P1-⑥：熔断恢复时同步解除策略协调层阻断。"""
        if self._breaker_linker is None:
            return
        try:
            await self._breaker_linker.reset()
        except Exception as e:
            logger.error(f"Failed to reset CircuitBreakerLinker: {e}")

    def can_trade(self) -> bool:
        return not self._is_paused

    def validate_signal(self, signal: Signal) -> bool:
        if not self.can_trade():
            logger.warning(f"Trading paused: {self._pause_reason}")
            return False
        
        return True

    def reset(self):
        self._is_paused = False
        self._pause_reason = None
        self._consecutive_losses = 0


class CircuitBreakers:
    def __init__(self, config: Dict[str, Any], okx_client, account_manager=None):
        self.config = config
        self.okx_client = okx_client
        self._account_manager = account_manager  # P1-⑤：单一权益基准源
        self._btc_threshold = config["risk"]["circuit_breakers"]["btc_movement_threshold"]
        self._btc_window = config["risk"]["circuit_breakers"]["btc_movement_window"]
        self._liquidation_threshold = config["risk"]["circuit_breakers"]["liquidation_volume_threshold"]
        self._liquidation_window = config["risk"]["circuit_breakers"]["liquidation_window"]
        self._api_timeout = config["risk"]["circuit_breakers"]["api_timeout"]
        self._network_timeout = config["risk"]["circuit_breakers"]["network_timeout"]
        
        self._rapid_drawdown_threshold = config["risk"]["circuit_breakers"].get("rapid_drawdown_threshold", 0.05)
        self._rapid_drawdown_window = config["risk"]["circuit_breakers"].get("rapid_drawdown_window", 3600)
        self._drawdown_acceleration_threshold = config["risk"]["circuit_breakers"].get("drawdown_acceleration_threshold", 0.02)
        
        # 多时间窗口快速回撤检测（秒: 阈值倍率）
        # 5分钟窗口更敏感（1.5x阈值），15分钟（1.2x），1小时（1.0x），4小时（0.8x）
        self._drawdown_windows = [
            (300, 1.5, "5min"),
            (900, 1.2, "15min"),
            (3600, 1.0, "1hour"),
            (14400, 0.8, "4hour"),
        ]
        
        # 波动率动态阈值：高波动时放宽阈值，低波动时收紧阈值
        self._volatility_adjust_enabled = config["risk"]["circuit_breakers"].get("volatility_adjust_enabled", True)
        self._volatility_lookback = 3600  # 1小时波动率回看
        self._volatility_high = 0.03  # 高波动率阈值（3%）
        self._volatility_low = 0.01  # 低波动率阈值（1%）
        
        # V形反弹检测：快速反弹后取消前一次熔断触发
        self._v_shape_recovery_enabled = True
        self._v_shape_recovery_pct = 0.015  # 反弹1.5%以上视为V形
        self._last_drawdown_level = 0.0  # 上次检测到的最大回撤
        self._last_drawdown_time = None
        
        # 熔断冷却期：触发后一段时间内不重复触发同级别
        self._cooldown_periods = {
            "minor": 600,   # minor冷却10分钟
            "major": 1800,  # major冷却30分钟
            "extreme": 3600,  # extreme冷却1小时
        }
        self._last_trigger_history: Dict[str, datetime] = {}  # 各级别上次触发时间
        
        self._btc_prices: List[float] = []
        self._btc_timestamps: List[datetime] = []
        self._liquidation_counts: List[int] = []
        self._liquidation_timestamps: List[datetime] = []
        self._equity_history: List[Dict[str, Any]] = []
        self._last_api_check = datetime.now()
        self._last_network_check = datetime.now()
        self._last_equity_check = datetime.now()
        # 熔断器预热期：启动后前5分钟不触发快速回撤熔断（避免启动初期数据不足导致误触发）
        self._circuit_breaker_warmup_seconds = config["risk"]["circuit_breakers"].get("warmup_seconds", 300)
        self._start_time = datetime.now()

    def set_account_manager(self, account_manager):
        """注入 AccountManager 作为单一权益基准源（P1-⑤）。"""
        self._account_manager = account_manager

    def _current_equity_source(self) -> float:
        """单一权益基准源（P1-⑤）：优先 AccountManager 当前权益，回退 okx_client。"""
        if self._account_manager is not None:
            eq = self._account_manager.get_total_equity()
            if eq and eq > 0:
                return float(eq)
        account_info = self.okx_client.get_account_info()
        if account_info:
            account = self.okx_client._parse_account_info(account_info)
            if account and account.total_equity > 0:
                return account.total_equity
        return 0.0

    async def check_all(self) -> Dict[str, Any]:
        """返回触发详情，包含 severity 级别：minor/major/extreme
        - minor: 仅暂停交易（如加速回撤）
        - major: 暂停+减仓30%（如急速回撤5%+、BTC大幅波动8%+、强平潮）
        - extreme: 暂停+强制全平（如急速回撤10%+、BTC剧烈波动15%+）
        """
        # 真正的风险事件检查；网络/API 检查因代理不稳定容易误触发，已禁用
        checks = [
            ("rapid_drawdown", await self._check_rapid_drawdown()),
            ("drawdown_acceleration", await self._check_drawdown_acceleration()),
            ("btc_movement", await self._check_btc_movement()),
            ("liquidation_volume", await self._check_liquidation_volume()),
        ]

        worst_severity = None
        worst_name = None
        severity_rank = {"minor": 1, "major": 2, "extreme": 3}

        for name, result in checks:
            # result 可能是 bool（旧格式）或 dict（新格式含severity）
            if isinstance(result, dict) and result.get("triggered"):
                sev = result.get("severity", "major")
                logger.error(f"Circuit breaker triggered: {name} (severity={sev}, detail={result.get('detail', '')})")
                if worst_severity is None or severity_rank.get(sev, 2) > severity_rank.get(worst_severity, 2):
                    worst_severity = sev
                    worst_name = name
            elif isinstance(result, bool) and result:
                logger.error(f"Circuit breaker triggered: {name}")
                if worst_severity is None:
                    worst_severity = "major"
                    worst_name = name

        if worst_severity:
            return {"triggered": True, "severity": worst_severity, "name": worst_name}
        return {"triggered": False, "severity": None, "name": None}
    
    async def _check_rapid_drawdown(self) -> Dict[str, Any]:
        try:
            now = datetime.now()
            # 预热期检查：启动后前warmup_seconds秒不触发
            uptime = (now - self._start_time).total_seconds()
            if uptime < self._circuit_breaker_warmup_seconds:
                return {"triggered": False, "detail": "warmup_period"}

            current_equity = self._current_equity_source()
            if current_equity <= 0:
                return {"triggered": False}

            self._equity_history.append({
                "equity": current_equity,
                "timestamp": now
            })

            # 保留最长窗口的数据（4小时）
            max_window = 14400
            cutoff = now - timedelta(seconds=max_window)
            while self._equity_history and self._equity_history[0]["timestamp"] < cutoff:
                self._equity_history.pop(0)

            # 至少需要10个数据点才开始检测（约50秒）
            if len(self._equity_history) < 10:
                return {"triggered": False}

            # 计算波动率（用于动态阈值调整）
            volatility = self._calculate_equity_volatility()
            vol_multiplier = 1.0
            if self._volatility_adjust_enabled and volatility > 0:
                if volatility >= self._volatility_high:
                    vol_multiplier = 1.5  # 高波动时放宽50%
                elif volatility >= self._volatility_low:
                    vol_multiplier = 1.0 + (volatility - self._volatility_low) / (self._volatility_high - self._volatility_low) * 0.5
                else:
                    vol_multiplier = 0.7  # 低波动时收紧30%

            # V形反弹检测：如果之前检测到较大回撤，现在快速反弹回来，更新状态
            if self._v_shape_recovery_enabled and self._last_drawdown_level > 0.01:
                recent_equities = [e["equity"] for e in self._equity_history[-5:]]
                if recent_equities:
                    recent_min = min(recent_equities)
                    recovery = (current_equity - recent_min) / recent_min if recent_min > 0 else 0
                    if recovery >= self._v_shape_recovery_pct:
                        logger.info(f"V-shape recovery detected: {recovery:.2%} rebound from trough, "
                                   f"previous drawdown={self._last_drawdown_level:.2%}")
                        self._last_drawdown_level = 0.0
                        self._last_drawdown_time = None

            # 多时间窗口检测：取最严重的结果
            worst_severity = None
            worst_detail = ""
            worst_window = ""

            for window_sec, threshold_mult, window_name in self._drawdown_windows:
                # 该窗口内的数据
                window_cutoff = now - timedelta(seconds=window_sec)
                window_data = [e for e in self._equity_history if e["timestamp"] >= window_cutoff]
                if len(window_data) < 3:
                    continue

                window_peak = max(e["equity"] for e in window_data)
                window_drawdown = (window_peak - current_equity) / window_peak if window_peak > 0 else 0

                # 应用波动率乘数和窗口倍率
                adjusted_threshold = self._rapid_drawdown_threshold * threshold_mult * vol_multiplier

                # 分级触发
                severity = None
                if window_drawdown >= adjusted_threshold * 2:
                    severity = "extreme"
                elif window_drawdown >= adjusted_threshold * 1.5:
                    severity = "major"
                elif window_drawdown >= adjusted_threshold:
                    severity = "minor"

                if severity:
                    # 检查冷却期
                    if self._is_in_cooldown(severity, now):
                        logger.debug(f"Rapid drawdown {window_drawdown:.2%} in {window_name}, "
                                    f"but {severity} in cooldown, skipping")
                        continue

                    severity_rank = {"minor": 1, "major": 2, "extreme": 3}
                    if worst_severity is None or severity_rank[severity] > severity_rank.get(worst_severity, 0):
                        worst_severity = severity
                        worst_detail = f"drawdown={window_drawdown:.2%} window={window_name} vol_mult={vol_multiplier:.2f}"
                        worst_window = window_name

                    # 记录最大回撤用于V形反弹检测
                    if window_drawdown > self._last_drawdown_level:
                        self._last_drawdown_level = window_drawdown
                        self._last_drawdown_time = now

            if worst_severity:
                self._last_trigger_history[worst_severity] = now
                log_method = logger.critical if worst_severity == "extreme" else logger.error if worst_severity == "major" else logger.warning
                log_method(f"{worst_severity.upper()} rapid drawdown: {worst_detail}")
                return {"triggered": True, "severity": worst_severity, "detail": worst_detail}

            return {"triggered": False}
        except Exception as e:
            logger.error(f"Failed to check rapid drawdown: {e}")
            return {"triggered": False}
    
    def _calculate_equity_volatility(self) -> float:
        """计算权益波动率（基于1小时内的收益率标准差）"""
        try:
            if len(self._equity_history) < 10:
                return 0.0
            now = datetime.now()
            vol_cutoff = now - timedelta(seconds=self._volatility_lookback)
            vol_data = [e for e in self._equity_history if e["timestamp"] >= vol_cutoff]
            if len(vol_data) < 5:
                return 0.0
            
            # 计算相邻收益率
            returns = []
            for i in range(1, len(vol_data)):
                prev = vol_data[i-1]["equity"]
                curr = vol_data[i]["equity"]
                if prev > 0:
                    returns.append(abs(curr - prev) / prev)
            
            if not returns:
                return 0.0
            
            avg_return = sum(returns) / len(returns)
            variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
            return variance ** 0.5
        except Exception:
            return 0.0

    def _is_in_cooldown(self, severity: str, now: datetime) -> bool:
        """检查是否处于冷却期"""
        if severity not in self._last_trigger_history:
            return False
        last_time = self._last_trigger_history[severity]
        cooldown = self._cooldown_periods.get(severity, 300)
        return (now - last_time).total_seconds() < cooldown

    def get_drawdown_status(self) -> Dict[str, Any]:
        """获取当前回撤状态（供外部查询和dashboard显示）"""
        now = datetime.now()
        if not self._equity_history:
            return {"has_data": False}
        
        current_equity = self._equity_history[-1]["equity"]
        
        # 各时间窗口回撤
        window_dd = {}
        for window_sec, _, window_name in self._drawdown_windows:
            cutoff = now - timedelta(seconds=window_sec)
            window_data = [e for e in self._equity_history if e["timestamp"] >= cutoff]
            if len(window_data) >= 2:
                peak = max(e["equity"] for e in window_data)
                dd = (peak - current_equity) / peak if peak > 0 else 0
                window_dd[window_name] = round(dd * 100, 2)
        
        # 波动率
        volatility = self._calculate_equity_volatility()
        
        # 冷却期状态
        cooldown_status = {}
        for sev in ["minor", "major", "extreme"]:
            if sev in self._last_trigger_history:
                elapsed = (now - self._last_trigger_history[sev]).total_seconds()
                total = self._cooldown_periods.get(sev, 300)
                cooldown_status[sev] = {
                    "active": elapsed < total,
                    "remaining_seconds": max(0, int(total - elapsed))
                }
            else:
                cooldown_status[sev] = {"active": False, "remaining_seconds": 0}
        
        return {
            "has_data": True,
            "current_equity": current_equity,
            "data_points": len(self._equity_history),
            "window_drawdowns_pct": window_dd,
            "volatility": round(volatility * 100, 3),
            "last_drawdown_level_pct": round(self._last_drawdown_level * 100, 2),
            "cooldown": cooldown_status,
            "warmup_remaining": max(0, int(self._circuit_breaker_warmup_seconds - (now - self._start_time).total_seconds()))
        }

    async def _check_drawdown_acceleration(self) -> Dict[str, Any]:
        try:
            if len(self._equity_history) < 12:
                return {"triggered": False}
            
            # 多时间尺度加速检测
            # 短期：最近3个点 vs 前3个点（15秒级）
            # 中期：最近6个点 vs 前6个点（30秒级）
            accelerations = []
            
            for short_n, med_n, label in [(3, 6, "short"), (6, 12, "medium")]:
                if len(self._equity_history) < med_n * 2:
                    continue
                recent = self._equity_history[-short_n:]
                previous = self._equity_history[-med_n - short_n:-med_n]
                
                if not recent or not previous:
                    continue
                
                recent_avg = sum(e["equity"] for e in recent) / len(recent)
                previous_avg = sum(e["equity"] for e in previous) / len(previous)
                
                if previous_avg > 0:
                    accel = (previous_avg - recent_avg) / previous_avg
                    accelerations.append((label, accel))
            
            if not accelerations:
                return {"triggered": False}
            
            # 取最严重的加速
            max_accel = max(a for _, a in accelerations)
            
            if max_accel >= self._drawdown_acceleration_threshold * 2:
                severity = "major"
            elif max_accel >= self._drawdown_acceleration_threshold:
                severity = "minor"
            else:
                return {"triggered": False}
            
            detail = f"acceleration={max_accel:.2%} levels={', '.join(f'{l}={a:.2%}' for l, a in accelerations)}"
            logger.warning(f"Drawdown acceleration detected ({severity}): {detail}")
            return {"triggered": True, "severity": severity, "detail": detail}
        except Exception as e:
            logger.error(f"Failed to check drawdown acceleration: {e}")
            return {"triggered": False}

    async def _check_btc_movement(self) -> Dict[str, Any]:
        try:
            # 用线程池包裹同步调用，避免阻塞事件循环；复用 okx_client 的代理
            # 直连降级逻辑，避免裸 requests.get 走系统代理（trust_env）导致 ProxyError
            ticker = await asyncio.to_thread(self.okx_client.get_ticker, "BTC-USDT")
            if not ticker or "last" not in ticker:
                return {"triggered": False}

            price = float(ticker["last"])
            now = datetime.now()

            self._btc_prices.append(price)
            self._btc_timestamps.append(now)

            cutoff = now - timedelta(seconds=self._btc_window)
            while self._btc_timestamps and self._btc_timestamps[0] < cutoff:
                self._btc_prices.pop(0)
                self._btc_timestamps.pop(0)

            if len(self._btc_prices) < 2:
                return {"triggered": False}

            max_price = max(self._btc_prices)
            min_price = min(self._btc_prices)
            movement = abs(max_price - min_price) / self._btc_prices[0]

            # 分级：≥2倍阈值=extreme(全平)，≥1.5倍=major(减仓30%)，≥阈值=minor(暂停)
            if movement >= self._btc_threshold * 2:
                logger.critical(f"EXTREME BTC movement: {movement:.2%} in {self._btc_window}s")
                return {"triggered": True, "severity": "extreme",
                        "detail": f"btc_movement={movement:.2%}"}
            if movement >= self._btc_threshold * 1.5:
                logger.error(f"MAJOR BTC movement: {movement:.2%} in {self._btc_window}s")
                return {"triggered": True, "severity": "major",
                        "detail": f"btc_movement={movement:.2%}"}
            if movement >= self._btc_threshold:
                logger.warning(f"MINOR BTC movement: {movement:.2%} in {self._btc_window}s")
                return {"triggered": True, "severity": "minor",
                        "detail": f"btc_movement={movement:.2%}"}
            return {"triggered": False}
        except Exception as e:
            logger.error(f"Failed to check BTC movement: {e}")
            return {"triggered": False}

    async def _check_liquidation_volume(self) -> Dict[str, Any]:
        try:
            # 使用 asyncio.to_thread 避免阻塞事件循环；复用 okx_client 的代理
            # 直连降级逻辑，避免裸 requests.get 走系统代理（trust_env）导致 ProxyError
            # 注意：get_liquidation_orders 首参为 uly（标的指数），非 instId
            liquidation_data = await asyncio.to_thread(
                self.okx_client.get_liquidation_orders, "BTC-USDT"
            )

            if not liquidation_data or liquidation_data.get("code") != "0":
                return {"triggered": False}

            # data 每项为按 instId 聚合的记录，真实爆仓明细在 item["details"]。
            # 逐条明细的 sz 为爆仓张数，累加得到本窗口爆仓总量，而非统计聚合记录条数
            # （旧逻辑 len(data) 恒为 1，导致爆仓量熔断实际永不触发）。
            total_liquidation_size = 0
            for item in liquidation_data.get("data", []) or []:
                if not isinstance(item, dict):
                    continue
                for detail in item.get("details", []) or []:
                    if not isinstance(detail, dict):
                        continue
                    try:
                        total_liquidation_size += float(detail.get("sz", 0) or 0)
                    except (TypeError, ValueError):
                        continue

            now = datetime.now()

            self._liquidation_counts.append(total_liquidation_size)
            self._liquidation_timestamps.append(now)

            cutoff = now - timedelta(seconds=self._liquidation_window)
            while self._liquidation_timestamps and self._liquidation_timestamps[0] < cutoff:
                self._liquidation_counts.pop(0)
                self._liquidation_timestamps.pop(0)

            if not self._liquidation_counts:
                return {"triggered": False}

            avg_liquidations = sum(self._liquidation_counts) / len(self._liquidation_counts)

            # 分级：≥3倍阈值=extreme，≥2倍=major，≥阈值=minor
            if avg_liquidations >= self._liquidation_threshold * 3:
                logger.critical(f"EXTREME liquidation volume: avg={avg_liquidations:.0f}")
                return {"triggered": True, "severity": "extreme",
                        "detail": f"avg_liq={avg_liquidations:.0f}"}
            if avg_liquidations >= self._liquidation_threshold * 2:
                logger.error(f"MAJOR liquidation volume: avg={avg_liquidations:.0f}")
                return {"triggered": True, "severity": "major",
                        "detail": f"avg_liq={avg_liquidations:.0f}"}
            if avg_liquidations >= self._liquidation_threshold:
                logger.warning(f"MINOR liquidation volume: avg={avg_liquidations:.0f}")
                return {"triggered": True, "severity": "minor",
                        "detail": f"avg_liq={avg_liquidations:.0f}"}
            return {"triggered": False}
        except Exception as e:
            logger.error(f"Failed to check liquidation volume: {e}")
            return {"triggered": False}