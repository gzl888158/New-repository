"""
P22-8: 生产级红线合规检查器

Usage:
    checker = ComplianceChecker(config, okx_client, trade_journal, account_manager, redis_cache)
    result = await checker.run_full_check()
    if not result["passed"]:
        logger.error(f"Compliance check FAILED: {result['issues']}")
"""

import asyncio
import json
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class ComplianceCheck:
    """单条合规检查项"""
    name: str
    category: str  # capital, risk, execution, data, strategy
    severity: str  # critical, error, warning
    passed: bool
    value: Any
    threshold: Any
    message: str
    suggestion: str = ""
    checked_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ComplianceChecker:
    """P22-8: 生产级红线合规检查器
    
    覆盖五大核心合规领域：
    1. 资金与杠杆 - 资金利用率、杠杆合规、保证金安全
    2. 风险控制 - 回撤、单日亏损、集中度、连续亏损
    3. 订单执行 - 撤单频率、订单超时、成交率、滑点
    4. 数据完整性 - 行情延迟、数据缺失、WebSocket状态
    5. 策略健康 - 策略运行状态、信号质量、胜率监控
    """
    
    # ── 合规红线阈值 ──────────────────────────────────────────────
    
    # 资金与杠杆
    MAX_CAPITAL_UTILIZATION = 0.80       # 最大资金利用率
    MIN_CAPITAL_UTILIZATION = 0.05       # 最低资金利用率（低于此值表示闲置）
    MAX_LEVERAGE_COMPLIANCE = 5          # 最大杠杆倍数合规上限
    MIN_MARGIN_RATIO = 0.05              # 最低保证金率
    
    # 风险控制
    MAX_DRAWDOWN_CRITICAL = 0.25         # 最大回撤 - 严重
    MAX_DRAWDOWN_WARNING = 0.15          # 最大回撤 - 警告
    MAX_DAILY_LOSS_CRITICAL = 0.10       # 单日最大亏损 - 严重
    MAX_DAILY_LOSS_WARNING = 0.05        # 单日最大亏损 - 警告
    MAX_POSITION_CONCENTRATION = 0.40    # 单币种最大持仓集中度
    MAX_CONSECUTIVE_LOSSES = 10          # 最大连续亏损次数
    MIN_WIN_RATE_CRITICAL = 0.05         # 最低胜率 - 严重（>50笔交易）
    MIN_WIN_RATE_WARNING = 0.20          # 最低胜率 - 警告（>20笔交易）
    MIN_PROFIT_FACTOR = 0.50             # 最低盈亏比
    
    # 订单执行
    MAX_CANCEL_RATE_PER_MINUTE = 10      # 每分钟最大撤单次数
    MAX_ORDER_TIMEOUT_SECONDS = 300      # 最大订单超时时间
    MIN_FILL_RATE = 0.50                 # 最低成交率
    MAX_SLIPPAGE_PCT = 0.01              # 最大滑点百分比
    
    # 数据完整性
    MAX_LATENCY_CRITICAL_MS = 5000       # 最大延迟 - 严重
    MAX_LATENCY_WARNING_MS = 2000        # 最大延迟 - 警告
    MAX_DATA_GAP_SECONDS = 120           # 最大数据中断时间
    MIN_WS_CONNECTION_HEALTH = 0.50      # 最低WebSocket连接健康度
    
    # 策略健康
    MAX_STRATEGY_DOWNTIME_SECONDS = 600  # 策略最大停机时间
    MIN_SIGNAL_QUALITY = 0.35            # 最低信号质量
    MAX_STRATEGY_PAUSE_DURATION = 86400  # 策略最大暂停时间（24h）
    
    def __init__(
        self,
        config: Dict[str, Any],
        okx_client=None,
        trade_journal=None,
        account_manager=None,
        redis_cache=None,
        scheduler=None,
    ):
        self.config = config
        self.okx_client = okx_client
        self.trade_journal = trade_journal
        self.account_manager = account_manager
        self.redis_cache = redis_cache
        self.scheduler = scheduler
        
        # 合规配置
        pt_config = config.get("paper_trading", {})
        self.enabled = pt_config.get("compliance_check_enabled", True)
        self.check_interval = pt_config.get("compliance_check_interval_sec", 3600)
        
        # 检查历史（用于趋势分析）
        self._check_history: List[Dict[str, Any]] = []
        self._last_check_ts: float = 0.0
        self._consecutive_failures = 0
        self._max_history = 100
        
        # 自动修复功能
        self._auto_fix_enabled = config.get("trading", {}).get("auto_fix_enabled", True)
        
        logger.info(
            f"P22-8: ComplianceChecker initialized - "
            f"enabled={self.enabled}, interval={self.check_interval}s, "
            f"auto_fix={self._auto_fix_enabled}"
        )
    
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
    
    async def run_full_check(self) -> Dict[str, Any]:
        """运行全量合规检查
        
        Returns:
            {
                "status": "PASS" | "FAIL" | "WARN",
                "passed": bool,
                "checks": [...],
                "issues": [...],
                "warnings": [...],
                "suggestions": [...],
                "checked_at": str,
                "auto_fix_actions": [...],
            }
        """
        checks = []
        issues = []
        warnings = []
        suggestions = []
        auto_fix_actions = []
        
        # 1. 资金与杠杆检查
        capital_checks = await self._check_capital_and_leverage()
        checks.extend(capital_checks)
        
        # 2. 风险控制检查
        risk_checks = await self._check_risk_controls()
        checks.extend(risk_checks)
        
        # 3. 订单执行检查
        execution_checks = await self._check_order_execution()
        checks.extend(execution_checks)
        
        # 4. 数据完整性检查
        data_checks = await self._check_data_integrity()
        checks.extend(data_checks)
        
        # 5. 策略健康检查
        strategy_checks = await self._check_strategy_health()
        checks.extend(strategy_checks)
        
        # 汇总结果
        for check in checks:
            if not check.passed:
                if check.severity == "critical":
                    issues.append(check)
                elif check.severity == "error":
                    issues.append(check)
                elif check.severity == "warning":
                    warnings.append(check)
                if check.suggestion:
                    suggestions.append(check.suggestion)
        
        # 自动修复
        if self._auto_fix_enabled and issues:
            auto_fix_actions = await self._attempt_auto_fix(issues)
        
        # 计算通过状态
        critical_failures = [c for c in checks if not c.passed and c.severity == "critical"]
        errors = [c for c in checks if not c.passed and c.severity == "error"]
        
        if critical_failures:
            status = "FAIL"
            passed = False
        elif errors:
            status = "FAIL"
            passed = False
        elif warnings:
            status = "WARN"
            passed = True
        else:
            status = "PASS"
            passed = True
        
        # 更新历史
        self._consecutive_failures = self._consecutive_failures + 1 if not passed else 0
        
        result = {
            "status": status,
            "passed": passed,
            "checks": [self._serialize_check(c) for c in checks],
            "issues": [self._serialize_check(c) for c in issues],
            "warnings": [self._serialize_check(c) for c in warnings],
            "suggestions": suggestions,
            "checked_at": datetime.now().isoformat(),
            "auto_fix_actions": auto_fix_actions,
            "consecutive_failures": self._consecutive_failures,
            "total_checks": len(checks),
            "passed_checks": sum(1 for c in checks if c.passed),
            "failed_checks": sum(1 for c in checks if not c.passed),
        }
        
        # 记录历史
        self._check_history.append({
            "status": status,
            "passed": passed,
            "timestamp": result["checked_at"],
            "issues_count": len(issues),
            "warnings_count": len(warnings),
        })
        if len(self._check_history) > self._max_history:
            self._check_history = self._check_history[-self._max_history:]
        
        self._last_check_ts = time.time()
        
        # 日志
        if not passed:
            issue_msgs = [f"{i.name}: {i.message}" for i in issues]
            logger.error(
                f"P22-8: Compliance check FAILED - {len(issues)} critical/error: "
                f"{'; '.join(issue_msgs[:5])}"
            )
        elif warnings:
            logger.warning(f"P22-8: Compliance check WARN - {len(warnings)} warnings")
        else:
            logger.debug(f"P22-8: Compliance check PASS - all {len(checks)} checks passed")
        
        return result
    
    # ── 1. 资金与杠杆检查 ────────────────────────────────────────
    
    async def _check_capital_and_leverage(self) -> List[ComplianceCheck]:
        """检查资金利用率、杠杆合规、保证金安全"""
        checks = []
        
        try:
            # 获取账户信息
            account_info = None
            if self.okx_client:
                try:
                    account_info = await asyncio.wait_for(
                        asyncio.to_thread(self.okx_client.get_account_info),
                        timeout=10,
                    )
                except (asyncio.TimeoutError, Exception):
                    pass
            
            if account_info:
                equity = self._safe_float(account_info.get("totalEq", 0))
                margin = self._safe_float(account_info.get("imr", 0))
                mmr = self._safe_float(account_info.get("mmr", 0))
                margin_ratio = self._safe_float(account_info.get("mgnRatio", 0))
                
                # 资金利用率
                if equity > 0:
                    utilization = margin / equity if margin > 0 else 0
                    
                    if utilization > self.MAX_CAPITAL_UTILIZATION:
                        checks.append(ComplianceCheck(
                            name="资本利用率过高",
                            category="capital",
                            severity="critical",
                            passed=False,
                            value=utilization,
                            threshold=self.MAX_CAPITAL_UTILIZATION,
                            message=f"资金利用率 {utilization:.1%} 超过上限 {self.MAX_CAPITAL_UTILIZATION:.0%}",
                            suggestion="立即降低仓位，优先平仓亏损仓位，暂停新开仓",
                        ))
                    elif utilization < self.MIN_CAPITAL_UTILIZATION and equity > 100:
                        checks.append(ComplianceCheck(
                            name="资金利用率过低",
                            category="capital",
                            severity="warning",
                            passed=False,
                            value=utilization,
                            threshold=self.MIN_CAPITAL_UTILIZATION,
                            message=f"资金利用率 {utilization:.1%} 低于下限 {self.MIN_CAPITAL_UTILIZATION:.0%}",
                            suggestion="考虑增加策略分配或提高单笔仓位以提升资金效率",
                        ))
                    else:
                        checks.append(ComplianceCheck(
                            name="资金利用率",
                            category="capital",
                            severity="warning",
                            passed=True,
                            value=utilization,
                            threshold=self.MAX_CAPITAL_UTILIZATION,
                            message=f"资金利用率 {utilization:.1%} (正常范围)",
                        ))
                
                # 保证金率
                if margin_ratio > 0:
                    if margin_ratio < self.MIN_MARGIN_RATIO:
                        checks.append(ComplianceCheck(
                            name="保证金率过低",
                            category="capital",
                            severity="critical",
                            passed=False,
                            value=margin_ratio,
                            threshold=self.MIN_MARGIN_RATIO,
                            message=f"保证金率 {margin_ratio:.1%} 低于安全线 {self.MIN_MARGIN_RATIO:.0%}，有爆仓风险",
                            suggestion="立即减仓或追加保证金，风险极高",
                        ))
                    else:
                        checks.append(ComplianceCheck(
                            name="保证金率",
                            category="capital",
                            severity="warning",
                            passed=True,
                            value=margin_ratio,
                            threshold=self.MIN_MARGIN_RATIO,
                            message=f"保证金率 {margin_ratio:.1%} (安全)",
                        ))
                
                # 杠杆合规
                if margin > 0 and equity > 0:
                    effective_leverage = margin / equity
                    if effective_leverage > self.MAX_LEVERAGE_COMPLIANCE:
                        checks.append(ComplianceCheck(
                            name="有效杠杆超限",
                            category="capital",
                            severity="error",
                            passed=False,
                            value=effective_leverage,
                            threshold=self.MAX_LEVERAGE_COMPLIANCE,
                            message=f"有效杠杆 {effective_leverage:.1f}x 超过上限 {self.MAX_LEVERAGE_COMPLIANCE}x",
                            suggestion="降低仓位或调整杠杆设置",
                        ))
            else:
                checks.append(ComplianceCheck(
                    name="账户信息获取失败",
                    category="capital",
                    severity="error",
                    passed=False,
                    value=None,
                    threshold="可用",
                    message="无法获取账户信息，可能API连接异常",
                    suggestion="检查OKX API连接和认证配置",
                ))
        
        except Exception as e:
            logger.exception(f"P24: Capital check exception: {e}")
            checks.append(ComplianceCheck(
                name="资本检查异常",
                category="capital",
                severity="error",
                passed=False,
                value=str(e),
                threshold="正常",
                message=f"资本检查执行异常: {e}",
            ))
        
        return checks
    
    # ── 2. 风险控制检查 ──────────────────────────────────────────
    
    async def _check_risk_controls(self) -> List[ComplianceCheck]:
        """检查回撤、单日亏损、集中度、连续亏损"""
        checks = []
        
        try:
            # 获取交易统计
            trades = []
            if self.trade_journal:
                try:
                    trades = await self.trade_journal.get_recent_trades(limit=200)
                except Exception:
                    pass
            
            # 获取持仓信息
            positions = []
            if self.okx_client:
                try:
                    positions = await asyncio.wait_for(
                        asyncio.to_thread(self.okx_client.get_positions),
                        timeout=10,
                    )
                except (asyncio.TimeoutError, Exception):
                    pass
            
            # 获取账户信息
            account_info = None
            if self.okx_client:
                try:
                    account_info = await asyncio.wait_for(
                        asyncio.to_thread(self.okx_client.get_account_info),
                        timeout=10,
                    )
                except (asyncio.TimeoutError, Exception):
                    pass
            
            equity = self._safe_float(account_info.get("totalEq", 0)) if account_info else 0
            unrealized_pnl = self._safe_float(account_info.get("upl", 0)) if account_info else 0
            
            # 回撤检查
            if self.trade_journal and equity > 0:
                try:
                    drawdown = await self.trade_journal.get_max_drawdown()
                    if drawdown is not None:
                        if drawdown > self.MAX_DRAWDOWN_CRITICAL:
                            checks.append(ComplianceCheck(
                                name="最大回撤超限",
                                category="risk",
                                severity="critical",
                                passed=False,
                                value=drawdown,
                                threshold=self.MAX_DRAWDOWN_CRITICAL,
                                message=f"最大回撤 {drawdown:.1%} 超过严重线 {self.MAX_DRAWDOWN_CRITICAL:.0%}",
                                suggestion="触发紧急熔断，暂停所有策略，检查持仓",
                            ))
                        elif drawdown > self.MAX_DRAWDOWN_WARNING:
                            checks.append(ComplianceCheck(
                                name="最大回撤偏高",
                                category="risk",
                                severity="warning",
                                passed=False,
                                value=drawdown,
                                threshold=self.MAX_DRAWDOWN_WARNING,
                                message=f"最大回撤 {drawdown:.1%} 超过警告线 {self.MAX_DRAWDOWN_WARNING:.0%}",
                                suggestion="缩减仓位至50%，暂停高风险策略",
                            ))
                        else:
                            checks.append(ComplianceCheck(
                                name="最大回撤",
                                category="risk",
                                severity="warning",
                                passed=True,
                                value=drawdown,
                                threshold=self.MAX_DRAWDOWN_CRITICAL,
                                message=f"最大回撤 {drawdown:.1%} (正常)",
                            ))
                except Exception:
                    pass
            
            # 单日亏损检查
            if self.trade_journal:
                try:
                    daily_pnl = await self.trade_journal.get_daily_pnl()
                    if daily_pnl is not None and equity > 0:
                        daily_loss_pct = abs(daily_pnl) / equity if daily_pnl < 0 else 0
                        if daily_loss_pct > self.MAX_DAILY_LOSS_CRITICAL:
                            checks.append(ComplianceCheck(
                                name="单日亏损超限",
                                category="risk",
                                severity="critical",
                                passed=False,
                                value=daily_loss_pct,
                                threshold=self.MAX_DAILY_LOSS_CRITICAL,
                                message=f"单日亏损 {daily_loss_pct:.1%} 超过严重线 {self.MAX_DAILY_LOSS_CRITICAL:.0%}",
                                suggestion="立即停止所有交易，等待次日评估",
                            ))
                        elif daily_loss_pct > self.MAX_DAILY_LOSS_WARNING:
                            checks.append(ComplianceCheck(
                                name="单日亏损偏高",
                                category="risk",
                                severity="warning",
                                passed=False,
                                value=daily_loss_pct,
                                threshold=self.MAX_DAILY_LOSS_WARNING,
                                message=f"单日亏损 {daily_loss_pct:.1%} 超过警告线 {self.MAX_DAILY_LOSS_WARNING:.0%}",
                                suggestion="缩减仓位，仅运行低风险策略",
                            ))
                except Exception:
                    pass
            
            # 持仓集中度检查
            # P30: 使用保证金(margin)而非名义价值(notionalUsd)衡量集中度。
            # 名义价值含杠杆放大（5x 杠杆下名义 = 5x 保证金），会高估小账户的
            # 真实资本集中度，导致"持仓集中度过高"误报。保证金才是实际占用资本，
            # 集中度 = 单币种保证金 / 总权益，与资金利用率、RiskGate 的仓位口径一致。
            # P29: 忽略微小仓位的集中度告警（最小保证金阈值，约 20 USDT 名义/5x）
            MIN_CONCENTRATION_MARGIN = 4.0  # 最小保证金阈值(USDT)
            if positions and equity > 0:
                for pos in positions:
                    # 与 _check_capital_utilization 相同的保证金提取链：margin -> imr -> notional/lever
                    margin = self._safe_float(pos.get("margin", 0))
                    if margin <= 0:
                        margin = self._safe_float(pos.get("imr", 0))
                    if margin <= 0:
                        lever = self._safe_float(pos.get("lever", 1)) or 1
                        notional = abs(self._safe_float(pos.get("notionalUsd", 0)))
                        if lever > 0 and notional > 0:
                            margin = notional / lever
                    if margin > 0:
                        # P29: 跳过微小仓位的集中度检查
                        if margin < MIN_CONCENTRATION_MARGIN:
                            continue
                        concentration = margin / equity
                        if concentration > self.MAX_POSITION_CONCENTRATION:
                            symbol = pos.get("instId", "unknown")
                            checks.append(ComplianceCheck(
                                name="持仓集中度过高",
                                category="risk",
                                severity="error",
                                passed=False,
                                value=concentration,
                                threshold=self.MAX_POSITION_CONCENTRATION,
                                message=f"{symbol} 持仓集中度 {concentration:.1%} 超过上限 {self.MAX_POSITION_CONCENTRATION:.0%}",
                                suggestion=f"降低 {symbol} 仓位，分散风险",
                            ))
            
            # 连续亏损检查
            if trades:
                consecutive_losses = 0
                for trade in reversed(trades):
                    pnl = self._safe_float(trade.get("pnl_usdt", 0))
                    if pnl < 0:
                        consecutive_losses += 1
                    else:
                        break
                
                if consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES:
                    checks.append(ComplianceCheck(
                        name="连续亏损超限",
                        category="risk",
                        severity="critical",
                        passed=False,
                        value=consecutive_losses,
                        threshold=self.MAX_CONSECUTIVE_LOSSES,
                        message=f"连续亏损 {consecutive_losses} 次达到上限 {self.MAX_CONSECUTIVE_LOSSES}",
                        suggestion="暂停所有开仓2小时，检查策略逻辑",
                    ))
                elif consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES * 0.7:
                    checks.append(ComplianceCheck(
                        name="连续亏损偏高",
                        category="risk",
                        severity="warning",
                        passed=False,
                        value=consecutive_losses,
                        threshold=self.MAX_CONSECUTIVE_LOSSES,
                        message=f"连续亏损 {consecutive_losses} 次，接近上限 {self.MAX_CONSECUTIVE_LOSSES}",
                        suggestion="缩减仓位，降低风险敞口",
                    ))
            
            # 胜率检查
            if len(trades) >= 20:
                wins = sum(1 for t in trades if self._safe_float(t.get("pnl_usdt", 0)) > 0)
                total = len(trades)
                win_rate = wins / total if total > 0 else 0
                
                if win_rate < self.MIN_WIN_RATE_CRITICAL and total >= 50:
                    checks.append(ComplianceCheck(
                        name="胜率极低",
                        category="risk",
                        severity="critical",
                        passed=False,
                        value=win_rate,
                        threshold=self.MIN_WIN_RATE_CRITICAL,
                        message=f"胜率 {win_rate:.1%} ({wins}/{total}) 低于严重线 {self.MIN_WIN_RATE_CRITICAL:.0%}",
                        suggestion="暂停策略24小时，分析失败原因，重新评估策略有效性",
                    ))
                elif win_rate < self.MIN_WIN_RATE_WARNING:
                    checks.append(ComplianceCheck(
                        name="胜率偏低",
                        category="risk",
                        severity="warning",
                        passed=False,
                        value=win_rate,
                        threshold=self.MIN_WIN_RATE_WARNING,
                        message=f"胜率 {win_rate:.1%} ({wins}/{total}) 低于警告线 {self.MIN_WIN_RATE_WARNING:.0%}",
                        suggestion="提高信号质量阈值，减少低质量交易",
                    ))
            
            # 盈亏比检查
            if len(trades) >= 10:
                gross_profit = sum(self._safe_float(t.get("pnl_usdt", 0)) for t in trades if self._safe_float(t.get("pnl_usdt", 0)) > 0)
                gross_loss = abs(sum(self._safe_float(t.get("pnl_usdt", 0)) for t in trades if self._safe_float(t.get("pnl_usdt", 0)) < 0))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
                
                if profit_factor < self.MIN_PROFIT_FACTOR:
                    checks.append(ComplianceCheck(
                        name="盈亏比过低",
                        category="risk",
                        severity="error",
                        passed=False,
                        value=profit_factor,
                        threshold=self.MIN_PROFIT_FACTOR,
                        message=f"盈亏比 {profit_factor:.2f} 低于下限 {self.MIN_PROFIT_FACTOR}",
                        suggestion="调整止盈止损比例，确保盈亏比 >= 1.5",
                    ))
        
        except Exception as e:
            logger.exception(f"P24: Risk check exception: {e}")
            checks.append(ComplianceCheck(
                name="风险检查异常",
                category="risk",
                severity="error",
                passed=False,
                value=str(e),
                threshold="正常",
                message=f"风险检查执行异常: {e}",
            ))
        
        return checks
    
    # ── 3. 订单执行检查 ──────────────────────────────────────────
    
    async def _check_order_execution(self) -> List[ComplianceCheck]:
        """检查撤单频率、订单超时、成交率、滑点"""
        checks = []
        
        try:
            # 获取最近的订单
            recent_orders = []
            if self.trade_journal:
                try:
                    recent_orders = await self.trade_journal.get_recent_orders(limit=100)
                except Exception:
                    pass
            
            if not recent_orders:
                return checks
            
            now = time.time()
            one_minute_ago = now - 60
            
            # 撤单频率检查
            canceled_orders = [
                o for o in recent_orders
                if o.get("status") == "canceled"
                and self._safe_float(o.get("update_time", 0)) > one_minute_ago
            ]
            cancel_rate = len(canceled_orders)
            if cancel_rate > self.MAX_CANCEL_RATE_PER_MINUTE:
                checks.append(ComplianceCheck(
                    name="撤单频率过高",
                    category="execution",
                    severity="error",
                    passed=False,
                    value=cancel_rate,
                    threshold=self.MAX_CANCEL_RATE_PER_MINUTE,
                    message=f"近1分钟撤单 {cancel_rate} 次，超过上限 {self.MAX_CANCEL_RATE_PER_MINUTE} 次",
                    suggestion="检查策略逻辑，减少不必要的撤单，可能触发API封禁",
                ))
            
            # 订单超时检查
            timed_out = [
                o for o in recent_orders
                if o.get("status") in ("pending", "open", "partially_filled")
                and (now - self._safe_float(o.get("create_time", now))) > self.MAX_ORDER_TIMEOUT_SECONDS
            ]
            if timed_out:
                symbols = set(o.get("symbol", "?") for o in timed_out)
                checks.append(ComplianceCheck(
                    name="存在超时订单",
                    category="execution",
                    severity="warning",
                    passed=False,
                    value=len(timed_out),
                    threshold=0,
                    message=f"有 {len(timed_out)} 个订单超时未成交: {', '.join(symbols)}",
                    suggestion="检查超时订单，考虑撤销并以更优价格重新挂单",
                ))
            
            # 成交率检查
            completed = [o for o in recent_orders if o.get("status") == "filled"]
            non_canceled = [o for o in recent_orders if o.get("status") != "canceled"]
            if len(non_canceled) > 10:
                fill_rate = len(completed) / len(non_canceled)
                if fill_rate < self.MIN_FILL_RATE:
                    checks.append(ComplianceCheck(
                        name="成交率过低",
                        category="execution",
                        severity="warning",
                        passed=False,
                        value=fill_rate,
                        threshold=self.MIN_FILL_RATE,
                        message=f"订单成交率 {fill_rate:.1%} 低于下限 {self.MIN_FILL_RATE:.0%}",
                        suggestion="检查限价单定价是否合理，考虑使用市价单或调整限价偏移",
                    ))
            
            # 滑点检查
            filled_orders = [o for o in recent_orders if o.get("status") == "filled"]
            slippage_violations = []
            for o in filled_orders:
                limit_price = self._safe_float(o.get("limit_price", 0))
                fill_price = self._safe_float(o.get("fill_price", 0))
                if limit_price > 0 and fill_price > 0:
                    slippage = abs(fill_price - limit_price) / limit_price
                    if slippage > self.MAX_SLIPPAGE_PCT:
                        slippage_violations.append((o.get("symbol", "?"), slippage))
            
            if len(slippage_violations) > 3:
                worst = max(slippage_violations, key=lambda x: x[1])
                checks.append(ComplianceCheck(
                    name="滑点异常",
                    category="execution",
                    severity="warning",
                    passed=False,
                    value=len(slippage_violations),
                    threshold=3,
                    message=f"有 {len(slippage_violations)} 笔订单滑点超标，最严重: {worst[0]} {worst[1]:.2%}",
                    suggestion="增大限价偏移，或在流动性差时使用市价单",
                ))
        
        except Exception as e:
            checks.append(ComplianceCheck(
                name="执行检查异常",
                category="execution",
                severity="warning",
                passed=False,
                value=str(e),
                threshold="正常",
                message=f"执行检查异常: {e}",
            ))
        
        return checks
    
    # ── 4. 数据完整性检查 ────────────────────────────────────────
    
    async def _check_data_integrity(self) -> List[ComplianceCheck]:
        """检查行情延迟、数据缺失、WebSocket状态"""
        checks = []
        
        try:
            # WebSocket状态检查
            if self.scheduler:
                ws_status = getattr(self.scheduler, "ws_connected", None)
                if ws_status is False:
                    checks.append(ComplianceCheck(
                        name="WebSocket断开",
                        category="data",
                        severity="critical",
                        passed=False,
                        value=False,
                        threshold=True,
                        message="WebSocket连接已断开，行情数据中断",
                        suggestion="触发自动重连，检查网络连接",
                    ))
            
            # 延迟检查
            if self.redis_cache:
                latency = await self._get_latency_stats()
                if latency:
                    avg_lat = latency.get("avg", 0)
                    if avg_lat > self.MAX_LATENCY_CRITICAL_MS:
                        checks.append(ComplianceCheck(
                            name="行情延迟严重",
                            category="data",
                            severity="critical",
                            passed=False,
                            value=avg_lat,
                            threshold=self.MAX_LATENCY_CRITICAL_MS,
                            message=f"平均行情延迟 {avg_lat:.0f}ms，超过严重线 {self.MAX_LATENCY_CRITICAL_MS}ms",
                            suggestion="暂停所有开仓操作，等待延迟恢复",
                        ))
                    elif avg_lat > self.MAX_LATENCY_WARNING_MS:
                        checks.append(ComplianceCheck(
                            name="行情延迟偏高",
                            category="data",
                            severity="warning",
                            passed=False,
                            value=avg_lat,
                            threshold=self.MAX_LATENCY_WARNING_MS,
                            message=f"平均行情延迟 {avg_lat:.0f}ms，超过警告线 {self.MAX_LATENCY_WARNING_MS}ms",
                            suggestion="暂停高风险策略开仓，关注网络状态",
                        ))
            
            # 数据中断检查
            if self.scheduler:
                last_bar_time = getattr(self.scheduler, "last_bar_timestamp", None)
                if last_bar_time:
                    gap = time.time() - last_bar_time
                    if gap > self.MAX_DATA_GAP_SECONDS:
                        checks.append(ComplianceCheck(
                            name="数据中断",
                            category="data",
                            severity="critical",
                            passed=False,
                            value=gap,
                            threshold=self.MAX_DATA_GAP_SECONDS,
                            message=f"数据中断 {gap:.0f}s，超过上限 {self.MAX_DATA_GAP_SECONDS}s",
                            suggestion="检查WebSocket连接和行情数据源",
                        ))
        
        except Exception as e:
            checks.append(ComplianceCheck(
                name="数据检查异常",
                category="data",
                severity="warning",
                passed=False,
                value=str(e),
                threshold="正常",
                message=f"数据检查异常: {e}",
            ))
        
        return checks
    
    # ── 5. 策略健康检查 ──────────────────────────────────────────
    
    async def _check_strategy_health(self) -> List[ComplianceCheck]:
        """检查策略运行状态、信号质量、胜率监控"""
        checks = []
        
        try:
            if not self.scheduler:
                return checks
            
            # 策略状态检查
            strategies = self._get_active_strategies()
            for strategy_name in strategies:
                strategy = getattr(self.scheduler, strategy_name, None)
                if strategy:
                    # 检查策略是否运行
                    is_running = getattr(strategy, "is_running", True)
                    if not is_running:
                        checks.append(ComplianceCheck(
                            name=f"策略已停止: {strategy_name}",
                            category="strategy",
                            severity="error",
                            passed=False,
                            value=False,
                            threshold=True,
                            message=f"策略 {strategy_name} 已停止运行",
                            suggestion=f"检查 {strategy_name} 日志，重启策略",
                        ))
                    
                    # 检查策略暂停时间
                    paused_at = getattr(strategy, "paused_at", None)
                    if paused_at:
                        pause_duration = time.time() - paused_at
                        if pause_duration > self.MAX_STRATEGY_PAUSE_DURATION:
                            checks.append(ComplianceCheck(
                                name=f"策略暂停过久: {strategy_name}",
                                category="strategy",
                                severity="warning",
                                passed=False,
                                value=pause_duration,
                                threshold=self.MAX_STRATEGY_PAUSE_DURATION,
                                message=f"策略 {strategy_name} 已暂停 {pause_duration/3600:.1f}h",
                                suggestion="检查暂停原因，考虑手动恢复或调整参数",
                            ))
                    
                    # 检查信号质量
                    signal_quality = getattr(strategy, "signal_quality", None)
                    if signal_quality is not None and signal_quality < self.MIN_SIGNAL_QUALITY:
                        checks.append(ComplianceCheck(
                            name=f"信号质量低: {strategy_name}",
                            category="strategy",
                            severity="warning",
                            passed=False,
                            value=signal_quality,
                            threshold=self.MIN_SIGNAL_QUALITY,
                            message=f"策略 {strategy_name} 信号质量 {signal_quality:.2f} 低于阈值 {self.MIN_SIGNAL_QUALITY}",
                            suggestion="调整策略参数，提高信号过滤标准",
                        ))
            
            # 策略最后活跃时间检查
            now = time.time()
            for strategy_name in strategies:
                strategy = getattr(self.scheduler, strategy_name, None)
                if strategy:
                    last_active = getattr(strategy, "last_signal_time", None)
                    if last_active and (now - last_active) > self.MAX_STRATEGY_DOWNTIME_SECONDS:
                        checks.append(ComplianceCheck(
                            name=f"策略无信号: {strategy_name}",
                            category="strategy",
                            severity="warning",
                            passed=False,
                            value=now - last_active,
                            threshold=self.MAX_STRATEGY_DOWNTIME_SECONDS,
                            message=f"策略 {strategy_name} {(now - last_active)/60:.0f}分钟无信号输出",
                            suggestion="检查市场条件是否满足策略触发条件",
                        ))
        
        except Exception as e:
            checks.append(ComplianceCheck(
                name="策略检查异常",
                category="strategy",
                severity="warning",
                passed=False,
                value=str(e),
                threshold="正常",
                message=f"策略检查异常: {e}",
            ))
        
        return checks
    
    # ── 自动修复 ──────────────────────────────────────────────────
    
    async def _attempt_auto_fix(self, issues: List[ComplianceCheck]) -> List[str]:
        """尝试自动修复合规问题"""
        actions = []
        
        for issue in issues:
            try:
                if "资金利用率过高" in issue.name:
                    if self.scheduler:
                        # 暂停新开仓
                        for strategy_name in self._get_active_strategies():
                            strategy = getattr(self.scheduler, strategy_name, None)
                            if strategy and hasattr(strategy, "pause_opening"):
                                strategy.pause_opening = True
                        actions.append(f"自动暂停所有策略新开仓 (触发: {issue.name})")
                
                elif "保证金率过低" in issue.name:
                    # 触发紧急熔断
                    if self.scheduler and hasattr(self.scheduler, "global_risk"):
                        self.scheduler.global_risk.emergency_stop = True
                        actions.append(f"触发紧急熔断 (触发: {issue.name})")
                
                elif "策略已停止" in issue.name:
                    # 尝试重启策略
                    strategy_name = issue.name.split(":")[-1].strip()
                    strategy = getattr(self.scheduler, strategy_name, None) if self.scheduler else None
                    if strategy and hasattr(strategy, "start"):
                        await strategy.start()
                        actions.append(f"自动重启策略: {strategy_name}")
            except Exception as e:
                logger.warning(f"Auto-fix for {issue.name} failed: {e}")
        
        if actions:
            logger.info(f"P22-8: Auto-fix actions: {actions}")
        
        return actions
    
    # ── 辅助方法 ──────────────────────────────────────────────────
    
    def _get_active_strategies(self) -> List[str]:
        """获取活跃策略列表"""
        strategies = []
        if self.scheduler:
            for attr in [
                "grid_strategy", "trend_strategy", "scalping_strategy",
                "arbitrage_strategy", "spot_grid_strategy", "spot_martingale_strategy",
            ]:
                if getattr(self.scheduler, attr, None) is not None:
                    strategies.append(attr)
        else:
            # 从配置读取
            strategies = self.config.get("trading", {}).get("strategies", [])
        return strategies
    
    async def _get_latency_stats(self) -> Dict[str, float]:
        """获取延迟统计"""
        stats = {"avg": 0, "max": 0, "min": 0, "count": 0}
        if self.redis_cache:
            try:
                cached = self.redis_cache.get("latency_stats")
                if cached:
                    stats = json.loads(cached) if isinstance(cached, str) else cached
            except Exception:
                pass
        return stats
    
    def _serialize_check(self, check: ComplianceCheck) -> Dict[str, Any]:
        """序列化检查结果"""
        return {
            "name": check.name,
            "category": check.category,
            "severity": check.severity,
            "passed": check.passed,
            "value": check.value,
            "threshold": check.threshold,
            "message": check.message,
            "suggestion": check.suggestion,
            "checked_at": check.checked_at,
        }
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取检查历史"""
        return self._check_history
    
    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        """获取最近一次检查结果"""
        if self._check_history:
            return self._check_history[-1]
        return None


# ── 全局单例 ────────────────────────────────────────────────────

_compliance_checker: Optional[ComplianceChecker] = None


def get_compliance_checker(
    config: Dict[str, Any] = None,
    **kwargs,
) -> Optional[ComplianceChecker]:
    """获取全局合规检查器"""
    global _compliance_checker
    if _compliance_checker is None and config:
        _compliance_checker = ComplianceChecker(config, **kwargs)
    return _compliance_checker


def reset_compliance_checker() -> None:
    """重置合规检查器（测试用）"""
    global _compliance_checker
    _compliance_checker = None