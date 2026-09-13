"""
企业级完善项（本轮）的针对性回归测试。

覆盖：
  1. black_swan_protection：_market_circuit_breaker_pct 死配置已接通，驱动 critical 熔断阈值
  2. pnl_reconciler：external_flow（出入金残差）持久化回写闭环
  3. order_executor：_partial_fill_tracker 超时清理，防内存泄漏
"""
import time
from datetime import datetime, timedelta

import pytest

from risk.black_swan_protection import BlackSwanProtection
from data.sqlite_storage import SQLiteStorage
from execution.order_executor import OrderExecutor


# ═══════════════════════════════════════════════════════════════
# 1. BlackSwanProtection._market_circuit_breaker_pct 已接入
# ═══════════════════════════════════════════════════════════════

def _make_black_swan(market_circuit_breaker_pct=0.08):
    config = {
        "risk": {
            "black_swan": {
                "enabled": True,
                "btc_flash_crash_pct": 0.03,
                "emergency_full_close_pct": 0.12,
                "market_circuit_breaker_pct": market_circuit_breaker_pct,
                "flash_crash_window_seconds": 300,
            }
        }
    }
    protection = BlackSwanProtection(config, okx_client=object())
    return protection


class TestBlackSwanCircuitBreakerThreshold:
    def _seed_history(self, protection, symbol, peak=100.0, current=92.0, n=10):
        now = datetime.now()
        history = []
        for i in range(n):
            t = now - timedelta(seconds=n - i)
            # 最后一个是当前价，其余都是峰值，保证窗口内 drawdown = (peak-current)/peak
            price = peak if i < n - 1 else current
            history.append((t, price))
        protection._price_history[symbol] = history
        return now

    async def test_drawdown_at_circuit_breaker_pct_is_critical(self):
        protection = _make_black_swan(market_circuit_breaker_pct=0.08)
        # drawdown = (100-92)/100 = 0.08 == market_circuit_breaker_pct → critical
        now = self._seed_history(protection, "BTC-USDT-SWAP", peak=100.0, current=92.0)

        await protection._check_flash_crash("BTC-USDT-SWAP", 92.0, now)

        assert protection._event_history
        assert protection._event_history[-1].severity == "critical"
        assert protection._event_history[-1].event_type == "flash_crash"

    async def test_drawdown_below_circuit_breaker_is_warning(self):
        protection = _make_black_swan(market_circuit_breaker_pct=0.08)
        # drawdown = (100-96)/100 = 0.04 → 超过 flash_crash(0.03) 但未达熔断(0.08)
        now = self._seed_history(protection, "BTC-USDT-SWAP", peak=100.0, current=96.0)

        await protection._check_flash_crash("BTC-USDT-SWAP", 96.0, now)

        assert protection._event_history[-1].severity == "warning"


# ═══════════════════════════════════════════════════════════════
# 2. PnLReconciliation external_flow 持久化回写
# ═══════════════════════════════════════════════════════════════

class TestExternalFlowPersistence:
    def test_external_flow_roundtrip(self):
        storage = SQLiteStorage({"sqlite": {"db_path": ":memory:"}})
        storage.save_pnl_reconciliation({
            "discrepancy": 15.0,
            "unattributed": 5.0,
            "external_flow": 135.0,
        })

        latest = storage.get_latest_pnl_reconciliation()
        assert latest is not None
        assert latest["external_flow"] == pytest.approx(135.0)
        assert latest["discrepancy"] == pytest.approx(15.0)

    def test_external_flow_defaults_to_zero(self):
        storage = SQLiteStorage({"sqlite": {"db_path": ":memory:"}})
        storage.save_pnl_reconciliation({"discrepancy": 1.0})

        latest = storage.get_latest_pnl_reconciliation()
        assert latest["external_flow"] == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════
# 3. OrderExecutor._partial_fill_tracker 超时清理
# ═══════════════════════════════════════════════════════════════

class TestPartialFillTrackerCleanup:
    def _executor(self, ttl=300.0):
        executor = object.__new__(OrderExecutor)
        executor._partial_fill_tracker_ttl = ttl
        executor._partial_fill_tracker = {
            "stale": {"start_time": time.time() - 1000.0},
            "fresh": {"start_time": time.time()},
        }
        return executor

    def test_stale_entries_removed(self):
        executor = self._executor(ttl=300.0)
        executor._cleanup_partial_fill_tracker()

        assert "stale" not in executor._partial_fill_tracker
        assert "fresh" in executor._partial_fill_tracker

    def test_empty_tracker_no_error(self):
        executor = self._executor(ttl=300.0)
        executor._partial_fill_tracker = {}
        executor._cleanup_partial_fill_tracker()  # 不应抛异常

        assert executor._partial_fill_tracker == {}
