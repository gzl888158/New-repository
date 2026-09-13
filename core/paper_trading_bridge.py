"""
P22-5: 模拟盘交易桥接器 - Paper Trading Bridge
==============================================
将实时行情数据桥接到沙盒引擎，实现模拟盘交易环境。

核心功能：
- 从WebSocket实时行情流中提取K线/Tick数据
- 桥接到沙盒引擎进行策略信号生成和虚拟订单执行
- 独立于实盘交易系统运行，互不干扰
- 支持延迟告警、限价优先、合规检查等生产级特性
- 持久化模拟盘状态，支持暂停恢复
"""

import asyncio
import os
import json
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from collections import deque
from loguru import logger

from sandbox import (
    SandboxEngine, SandboxConfig, SandboxManager, SandboxState,
    SandboxRunMode, get_sandbox_manager
)


class LatencyMonitor:
    """P22-6: 行情延迟监控器
    
    监控WebSocket行情延迟，超阈值时自动暂停开仓。
    """
    
    def __init__(self, warning_ms: int = 3000, critical_ms: int = 8000,
                 pause_duration_sec: int = 300):
        self.warning_ms = warning_ms
        self.critical_ms = critical_ms
        self.pause_duration_sec = pause_duration_sec
        
        # 滑动窗口统计
        self._latency_window: deque = deque(maxlen=60)  # 60个样本
        self._window_lock = threading.Lock()
        
        # 暂停状态
        self._opening_paused = False
        self._pause_start_time: float = 0.0
        self._pause_reason = ""
        
        # 统计
        self._total_samples = 0
        self._warning_count = 0
        self._critical_count = 0
        self._last_warning_time: float = 0.0
        
    def record_latency(self, latency_ms: float) -> Dict[str, Any]:
        """记录一次延迟样本"""
        with self._window_lock:
            self._latency_window.append(latency_ms)
            self._total_samples += 1
        
        result = {"action": "none", "latency_ms": latency_ms}
        
        # 计算滑动平均
        avg_latency = self.get_average_latency()
        
        if latency_ms >= self.critical_ms:
            self._critical_count += 1
            if not self._opening_paused:
                self._opening_paused = True
                self._pause_start_time = time.time()
                self._pause_reason = f"Critical latency: {latency_ms:.0f}ms > {self.critical_ms}ms"
                result["action"] = "pause"
                result["reason"] = self._pause_reason
                logger.warning(
                    f"P22-6: Opening positions PAUSED - {self._pause_reason}, "
                    f"avg={avg_latency:.0f}ms, pause_duration={self.pause_duration_sec}s"
                )
        elif latency_ms >= self.warning_ms:
            self._warning_count += 1
            self._last_warning_time = time.time()
            if avg_latency >= self.warning_ms and not self._opening_paused:
                self._opening_paused = True
                self._pause_start_time = time.time()
                self._pause_reason = f"Sustained high latency: avg={avg_latency:.0f}ms > {self.warning_ms}ms"
                result["action"] = "pause"
                result["reason"] = self._pause_reason
                logger.warning(
                    f"P22-6: Opening positions PAUSED - {self._pause_reason}"
                )
        
        # 自动恢复检查
        if self._opening_paused and avg_latency < self.warning_ms * 0.7:
            elapsed = time.time() - self._pause_start_time
            if elapsed >= self.pause_duration_sec:
                self._opening_paused = False
                self._pause_reason = ""
                result["action"] = "resume"
                result["reason"] = f"Latency recovered: avg={avg_latency:.0f}ms"
                logger.info(
                    f"P22-6: Opening positions RESUMED - avg latency={avg_latency:.0f}ms"
                )
        
        return result
    
    def get_average_latency(self) -> float:
        """获取滑动窗口平均延迟"""
        with self._window_lock:
            if not self._latency_window:
                return 0.0
            return sum(self._latency_window) / len(self._latency_window)
    
    @property
    def is_opening_paused(self) -> bool:
        return self._opening_paused
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "avg_latency_ms": round(self.get_average_latency(), 1),
            "total_samples": self._total_samples,
            "warning_count": self._warning_count,
            "critical_count": self._critical_count,
            "opening_paused": self._opening_paused,
            "pause_reason": self._pause_reason,
            "pause_elapsed_sec": round(time.time() - self._pause_start_time, 1) if self._opening_paused else 0,
        }


class PaperTradingBridge:
    """P22-5: 模拟盘交易桥接器
    
    桥接实时行情与沙盒引擎，提供完整的模拟盘交易环境。
    
    使用方式：
        bridge = PaperTradingBridge(config)
        await bridge.start()
        # WebSocket行情通过 on_bar/on_tick 注入
        bridge.on_bar("BTC-USDT-SWAP", bar_data)
        # 查看绩效
        perf = bridge.get_performance_summary()
    """
    
    def __init__(self, config: Dict[str, Any]):
        pt_config = config.get("paper_trading", {})
        self.config = pt_config
        self.system_config = config
        
        # 延迟监控
        self.latency_monitor = LatencyMonitor(
            warning_ms=pt_config.get("latency_warning_ms", 3000),
            critical_ms=pt_config.get("latency_critical_ms", 8000),
            pause_duration_sec=pt_config.get("latency_pause_duration_sec", 300),
        )
        
        # 限价优先配置
        self.limit_order_priority = pt_config.get("limit_order_priority", True)
        self.limit_order_max_spread = pt_config.get("limit_order_max_spread_pct", 0.001)
        
        # 沙盒管理器
        self._manager = get_sandbox_manager(self.system_config)
        self._sandbox_id: Optional[str] = None
        self._running = False
        
        # 统计
        self._bar_count = 0
        self._tick_count = 0
        self._start_time: Optional[datetime] = None
        
        # 持久化
        self._persist_dir = pt_config.get("persist_dir", "./data/paper_trading")
        os.makedirs(self._persist_dir, exist_ok=True)
        
        logger.info(f"PaperTradingBridge initialized: "
                    f"mode={pt_config.get('mode', 'sandbox')}, "
                    f"limit_order_priority={self.limit_order_priority}, "
                    f"latency_warning={self.latency_monitor.warning_ms}ms")
    
    async def start(self) -> bool:
        """启动模拟盘交易桥接器"""
        pt_config = self.config
        
        if not pt_config.get("enabled", False):
            logger.info("Paper trading is disabled in config")
            return False
        
        try:
            # 创建沙盒配置
            sandbox_config = SandboxConfig(
                sandbox_id=pt_config.get("sandbox_id", "paper_main"),
                name=pt_config.get("sandbox_id", "Paper Trading Main"),
                initial_capital=pt_config.get("initial_capital", 10000.0),
                symbols=pt_config.get("symbols", []),
                strategies=pt_config.get("strategies", []),
                timeframes=pt_config.get("timeframes", ["5m", "15m"]),
                max_leverage=pt_config.get("max_leverage", 5),
                max_positions=pt_config.get("max_positions", 8),
                risk_per_trade_pct=pt_config.get("risk_per_trade_pct", 0.02),
                max_daily_loss_pct=pt_config.get("max_daily_loss_pct", 0.10),
                max_drawdown_pct=pt_config.get("max_drawdown_pct", 0.25),
                snapshot_interval_seconds=pt_config.get("snapshot_interval_seconds", 60),
                order_timeout_seconds=pt_config.get("order_timeout_seconds", 3600),
                enable_auto_trading=pt_config.get("enable_auto_trading", True),
                enable_stop_loss=pt_config.get("enable_stop_loss", True),
                enable_take_profit=pt_config.get("enable_take_profit", True),
                stop_loss_atr_mult=pt_config.get("stop_loss_atr_mult", 2.0),
                take_profit_atr_mult=pt_config.get("take_profit_atr_mult", 3.0),
                min_signal_strength=pt_config.get("min_signal_strength", 0.40),
                description="Production-grade paper trading sandbox",
            )
            
            # 创建沙盒
            self._sandbox_id = self._manager.create_custom(sandbox_config.to_dict())
            if not self._sandbox_id:
                logger.error("Failed to create paper trading sandbox")
                return False
            
            # 启动沙盒
            await self._manager.get_engine(self._sandbox_id).start()
            
            self._running = True
            self._start_time = datetime.now()
            self._loop = asyncio.get_running_loop()
            
            logger.info(
                f"Paper trading started: sandbox={self._sandbox_id}, "
                f"capital={sandbox_config.initial_capital} USDT, "
                f"symbols={len(sandbox_config.symbols)}, "
                f"strategies={sandbox_config.strategies}"
            )
            
            # 启动持久化定时器
            if pt_config.get("persist_state", True):
                asyncio.create_task(self._persist_loop())
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start paper trading bridge: {e}")
            return False
    
    async def stop(self) -> Dict[str, Any]:
        """停止模拟盘交易"""
        self._running = False
        summary = {}
        
        if self._sandbox_id:
            engine = self._manager.get_engine(self._sandbox_id)
            if engine:
                summary = await engine.stop()
            # 持久化最终状态
            self._persist_state()
        
        logger.info(f"Paper trading stopped: {summary.get('roi_pct', 0)}% ROI")
        return summary
    
    def on_bar(self, symbol: str, bar: Dict[str, Any]) -> None:
        """处理K线数据 - 注入沙盒引擎
        
        Args:
            symbol: 交易对
            bar: K线数据 {open, high, low, close, volume, timestamp}
        """
        if not self._running or not self._sandbox_id:
            return
        
        engine = self._manager.get_engine(self._sandbox_id)
        if not engine or engine.state != SandboxState.RUNNING:
            return
        
        # 延迟监控
        bar_ts = bar.get("timestamp", bar.get("ts", 0))
        if bar_ts:
            now_ms = time.time() * 1000
            latency_ms = now_ms - (int(bar_ts) if isinstance(bar_ts, (int, float)) else 0)
            if latency_ms > 0:
                self.latency_monitor.record_latency(latency_ms)
        
        # 注入沙盒（使用 start() 捕获的事件循环，避免跨线程无运行循环时抛 RuntimeError）
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                engine.on_bar(symbol, bar),
                self._loop
            )
        
        self._bar_count += 1
    
    def on_tick(self, symbol: str, tick: Dict[str, Any]) -> None:
        """处理Tick数据 - 注入沙盒引擎"""
        if not self._running or not self._sandbox_id:
            return
        
        engine = self._manager.get_engine(self._sandbox_id)
        if not engine or engine.state != SandboxState.RUNNING:
            return
        
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                engine.on_tick(symbol, tick),
                self._loop
            )
        
        self._tick_count += 1
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """批量更新价格"""
        if not self._running or not self._sandbox_id:
            return
        
        engine = self._manager.get_engine(self._sandbox_id)
        if engine:
            engine.update_prices(prices)
    
    def should_allow_opening(self) -> bool:
        """P22-6: 检查是否允许开仓（延迟熔断）"""
        if not self.config.get("latency_pause_enabled", True):
            return True
        return not self.latency_monitor.is_opening_paused
    
    def should_use_limit_order(self, spread_pct: float) -> bool:
        """P22-7: 判断是否应使用限价单
        
        Args:
            spread_pct: 当前买卖价差百分比
        
        Returns:
            True if limit order should be used
        """
        if not self.limit_order_priority:
            return False
        return spread_pct <= self.limit_order_max_spread
    
    # ── 绩效查询 ─────────────────────────────────────────────────
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取模拟盘绩效摘要"""
        if not self._sandbox_id:
            return {}
        
        engine = self._manager.get_engine(self._sandbox_id)
        if not engine:
            return {}
        
        perf = engine.get_performance_summary()
        
        # 添加桥接器统计
        run_seconds = 0
        if self._start_time:
            run_seconds = (datetime.now() - self._start_time).total_seconds()
        
        perf.update({
            "bridge_bar_count": self._bar_count,
            "bridge_tick_count": self._tick_count,
            "bridge_run_seconds": round(run_seconds, 0),
            "latency": self.latency_monitor.get_stats(),
            "opening_allowed": self.should_allow_opening(),
        })
        
        return perf
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """获取模拟盘持仓"""
        if not self._sandbox_id:
            return []
        
        engine = self._manager.get_engine(self._sandbox_id)
        if not engine:
            return []
        
        return engine.get_positions()
    
    def get_equity_curve(self) -> List[Dict[str, Any]]:
        """获取资金曲线"""
        if not self._sandbox_id:
            return []
        
        engine = self._manager.get_engine(self._sandbox_id)
        if not engine:
            return []
        
        return engine.get_equity_curve()
    
    def get_latency_stats(self) -> Dict[str, Any]:
        """P22-6: 获取延迟统计"""
        return self.latency_monitor.get_stats()
    
    # ── 持久化 ─────────────────────────────────────────────────
    
    def _persist_state(self) -> None:
        """持久化模拟盘状态"""
        if not self._sandbox_id:
            return
        
        try:
            state_file = os.path.join(self._persist_dir, "paper_trading_state.json")
            state = {
                "sandbox_id": self._sandbox_id,
                "saved_at": datetime.now().isoformat(),
                "performance": self.get_performance_summary(),
                "latency": self.latency_monitor.get_stats(),
            }
            
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            
            # 同时持久化沙盒元数据
            self._manager.persist_all()
            
        except Exception as e:
            logger.error(f"Failed to persist paper trading state: {e}")
    
    async def _persist_loop(self) -> None:
        """定期持久化循环"""
        interval = self.config.get("snapshot_interval_seconds", 60)
        while self._running:
            try:
                await asyncio.sleep(interval)
                self._persist_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Persist loop error: {e}")
                await asyncio.sleep(10)
    
    # ── 合规检查 ─────────────────────────────────────────────────
    
    def run_compliance_check(self) -> Dict[str, Any]:
        """P22-8: 生产级红线合规检查"""
        issues = []
        warnings = []
        
        # 1. 检查资金利用率
        perf = self.get_performance_summary()
        utilization = perf.get("margin_ratio", 0)
        if utilization < 0.10:
            warnings.append(f"Low capital utilization: {utilization:.1%}")
        if utilization > 0.80:
            issues.append(f"High capital utilization: {utilization:.1%}")
        
        # 2. 检查回撤
        drawdown = perf.get("max_drawdown_pct", 0)
        if drawdown > 0.20:
            issues.append(f"High drawdown: {drawdown:.1%}")
        elif drawdown > 0.10:
            warnings.append(f"Elevated drawdown: {drawdown:.1%}")
        
        # 3. 检查胜率
        win_rate = perf.get("win_rate_pct", 0)
        if win_rate < 20 and perf.get("total_trades_count", 0) > 20:
            issues.append(f"Very low win rate: {win_rate:.1f}%")
        elif win_rate < 35 and perf.get("total_trades_count", 0) > 20:
            warnings.append(f"Low win rate: {win_rate:.1f}%")
        
        # 4. 检查盈亏比
        profit_factor = perf.get("profit_factor", 0)
        if profit_factor < 0.5 and perf.get("total_trades_count", 0) > 10:
            issues.append(f"Poor profit factor: {profit_factor:.2f}")
        
        # 5. 检查延迟
        latency = self.latency_monitor.get_stats()
        avg_lat = latency.get("avg_latency_ms", 0)
        if avg_lat > 5000:
            issues.append(f"High average latency: {avg_lat:.0f}ms")
        elif avg_lat > 2000:
            warnings.append(f"Elevated latency: {avg_lat:.0f}ms")
        
        # 6. 检查持仓集中度
        positions = self.get_positions()
        if len(positions) > 8:
            warnings.append(f"High position count: {len(positions)}")
        
        passed = len(issues) == 0
        status = "PASS" if passed else "FAIL"
        
        result = {
            "status": status,
            "passed": passed,
            "issues": issues,
            "warnings": warnings,
            "checked_at": datetime.now().isoformat(),
            "performance": {
                "roi_pct": perf.get("roi_pct", 0),
                "win_rate_pct": perf.get("win_rate_pct", 0),
                "profit_factor": perf.get("profit_factor", 0),
                "max_drawdown_pct": perf.get("max_drawdown_pct", 0),
                "total_trades": perf.get("total_trades_count", 0),
            },
            "latency": latency,
        }
        
        return result


# ── 全局单例 ────────────────────────────────────────────────────

_bridge: Optional[PaperTradingBridge] = None
_bridge_lock = threading.Lock()


def get_paper_trading_bridge(config: Dict[str, Any] = None) -> Optional[PaperTradingBridge]:
    """获取全局模拟盘桥接器"""
    global _bridge
    with _bridge_lock:
        if _bridge is None and config:
            _bridge = PaperTradingBridge(config)
        return _bridge


def reset_paper_trading_bridge() -> None:
    """重置桥接器（测试用）"""
    global _bridge
    with _bridge_lock:
        _bridge = None