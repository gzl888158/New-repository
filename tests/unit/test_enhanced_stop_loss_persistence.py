"""EnhancedStopLoss 企业级止损状态跨重启持久化 — 单元测试。

覆盖：
  P0  init_position_stop 持久化并能在新实例中恢复
  P0  移动/保本止损上移后的 current_stop / peak_price 跨重启保留
  P0  分级止盈 tp1_filled 进度跨重启保留
  P0  部分减仓累计量 total_closed_qty 跨重启保留
  P0  remove_position 删除持久化状态
  P0  init_position_stop 幂等：重启恢复后再次 init 不回退止损
  P1  无 db_path 时禁用持久化且不抛异常
  P1  atr_history 序列化/反序列化往返（datetime 正确还原）
"""

from datetime import datetime

import pytest

from risk.profit_optimizer import EnhancedStopLoss


def _make_config(db_path) -> dict:
    """构造含 sqlite 路径的最小配置（策略 grid 使用默认参数）。"""
    return {
        "sqlite": {"db_path": str(db_path)},
        "trading": {"taker_fee_rate": 0.0005},
        "strategies": {"grid": {}},
    }


class TestPersistence:
    def test_init_persists_and_restores(self, tmp_path):
        db = tmp_path / "trading.db"
        sm = EnhancedStopLoss(_make_config(db), "grid")
        sm.init_position_stop("BTC-USDT-SWAP", 100.0, "long", 1.0)

        sm2 = EnhancedStopLoss(_make_config(db), "grid")
        assert "BTC-USDT-SWAP" in sm2._position_stops
        state = sm2._position_stops["BTC-USDT-SWAP"]
        assert state["entry_price"] == pytest.approx(100.0)
        assert state["direction"] == "long"
        assert isinstance(state["entry_time"], datetime)

    def test_stop_uplift_survives_restart(self, tmp_path):
        db = tmp_path / "trading.db"
        sm = EnhancedStopLoss(_make_config(db), "grid")
        sm.init_position_stop("BTC-USDT-SWAP", 100.0, "long", 1.0)
        new_stop, stop_type = sm.update_stop("BTC-USDT-SWAP", 105.0, atr=1.0)
        assert new_stop > 100.0 * (1 - 0.025)

        sm2 = EnhancedStopLoss(_make_config(db), "grid")
        restored = sm2._position_stops["BTC-USDT-SWAP"]
        assert restored["current_stop"] == pytest.approx(new_stop)
        assert restored["peak_price"] == pytest.approx(105.0)
        assert restored["breakeven_activated"] is True

    def test_tp1_filled_survives_restart(self, tmp_path):
        db = tmp_path / "trading.db"
        sm = EnhancedStopLoss(_make_config(db), "grid")
        sm.init_position_stop("BTC-USDT-SWAP", 100.0, "long", 1.0)
        tp1 = sm._position_stops["BTC-USDT-SWAP"]["tp1_price"]
        actions = sm.check_take_profit("BTC-USDT-SWAP", tp1 + 0.1)
        assert any(a["action"] == "tp1" for a in actions)

        sm2 = EnhancedStopLoss(_make_config(db), "grid")
        assert sm2._position_stops["BTC-USDT-SWAP"]["tp1_filled"] is True

    def test_partial_close_survives_restart(self, tmp_path):
        db = tmp_path / "trading.db"
        sm = EnhancedStopLoss(_make_config(db), "grid")
        sm.init_position_stop("BTC-USDT-SWAP", 100.0, "long", 1.0)
        sm.record_partial_close("BTC-USDT-SWAP", 0.4)

        sm2 = EnhancedStopLoss(_make_config(db), "grid")
        assert sm2._position_stops["BTC-USDT-SWAP"]["total_closed_qty"] == pytest.approx(0.4)

    def test_remove_position_deletes_persisted_state(self, tmp_path):
        db = tmp_path / "trading.db"
        sm = EnhancedStopLoss(_make_config(db), "grid")
        sm.init_position_stop("BTC-USDT-SWAP", 100.0, "long", 1.0)
        sm.remove_position("BTC-USDT-SWAP")

        sm2 = EnhancedStopLoss(_make_config(db), "grid")
        assert "BTC-USDT-SWAP" not in sm2._position_stops

    def test_init_position_stop_idempotent_keeps_progress(self, tmp_path):
        db = tmp_path / "trading.db"
        sm = EnhancedStopLoss(_make_config(db), "grid")
        sm.init_position_stop("BTC-USDT-SWAP", 100.0, "long", 1.0)
        sm.update_stop("BTC-USDT-SWAP", 105.0, atr=1.0)
        uplifted = sm._position_stops["BTC-USDT-SWAP"]["current_stop"]
        assert uplifted > 97.5

        # 模拟重启：新实例加载持久化状态，随后 _recover_stop_losses_after_restart 再次 init
        sm2 = EnhancedStopLoss(_make_config(db), "grid")
        returned = sm2.init_position_stop("BTC-USDT-SWAP", 100.0, "long", 1.0)
        assert returned == pytest.approx(uplifted)
        assert sm2._position_stops["BTC-USDT-SWAP"]["current_stop"] == pytest.approx(uplifted)


class TestPersistenceDisabled:
    def test_no_db_path_disables_persistence(self, tmp_path):
        config = {"trading": {}, "strategies": {"grid": {}}}
        sm = EnhancedStopLoss(config, "grid")
        assert sm._persistence_enabled is False
        # 无 db_path 时不应抛异常
        sm.init_position_stop("BTC-USDT-SWAP", 100.0, "long", 1.0)
        sm.update_stop("BTC-USDT-SWAP", 101.0, atr=1.0)
        sm.record_partial_close("BTC-USDT-SWAP", 0.1)
        sm.remove_position("BTC-USDT-SWAP")


class TestSerializationRoundtrip:
    def test_atr_history_datetime_roundtrip(self, tmp_path):
        db = tmp_path / "trading.db"
        sm = EnhancedStopLoss(_make_config(db), "grid")
        sm.init_position_stop("BTC-USDT-SWAP", 100.0, "long", 1.0)
        sm.update_stop("BTC-USDT-SWAP", 100.0, atr=0.5)
        sm.update_stop("BTC-USDT-SWAP", 100.0, atr=0.6)

        sm2 = EnhancedStopLoss(_make_config(db), "grid")
        hist = sm2._position_stops["BTC-USDT-SWAP"]["atr_history"]
        assert len(hist) >= 2
        assert all(isinstance(h["time"], datetime) for h in hist)
        assert hist[-1]["atr"] == pytest.approx(0.6)

    def test_volatility_lockout_roundtrip(self, tmp_path):
        from datetime import timedelta

        db = tmp_path / "trading.db"
        sm = EnhancedStopLoss(_make_config(db), "grid")
        sm.init_position_stop("BTC-USDT-SWAP", 100.0, "long", 1.0)
        state = sm._position_stops["BTC-USDT-SWAP"]
        state["vol_lockout_until"] = datetime.now() + timedelta(minutes=30)
        sm._persist("BTC-USDT-SWAP")

        sm2 = EnhancedStopLoss(_make_config(db), "grid")
        restored = sm2._position_stops["BTC-USDT-SWAP"]["vol_lockout_until"]
        assert isinstance(restored, datetime)


class TestReconcile:
    """启动对账：按真实持仓清理幽灵止损/止盈状态（内存 + 持久化）。"""

    def test_reconcile_removes_stale_memory_and_db(self, tmp_path):
        db = tmp_path / "trading.db"
        sm = EnhancedStopLoss(_make_config(db), "grid")
        sm.init_position_stop("BTC-USDT-SWAP", 100.0, "long", 1.0)
        sm.init_position_stop("ETH-USDT-SWAP", 2000.0, "long", 1.0)

        # 只保留 BTC，ETH 是平仓未 remove 的幽灵残留
        cleaned = sm.reconcile_with_active_positions({"BTC-USDT-SWAP"})
        assert cleaned >= 1
        assert "BTC-USDT-SWAP" in sm._position_stops
        assert "ETH-USDT-SWAP" not in sm._position_stops

        # 持久化层面同样被清理（新实例加载不到 ETH）
        sm2 = EnhancedStopLoss(_make_config(db), "grid")
        assert "BTC-USDT-SWAP" in sm2._position_stops
        assert "ETH-USDT-SWAP" not in sm2._position_stops

    def test_reconcile_keeps_active_symbols(self, tmp_path):
        db = tmp_path / "trading.db"
        sm = EnhancedStopLoss(_make_config(db), "grid")
        sm.init_position_stop("BTC-USDT-SWAP", 100.0, "long", 1.0)
        sm.reconcile_with_active_positions({"BTC-USDT-SWAP"})
        assert "BTC-USDT-SWAP" in sm._position_stops

    def test_reconcile_empty_set_clears_all(self, tmp_path):
        db = tmp_path / "trading.db"
        sm = EnhancedStopLoss(_make_config(db), "grid")
        sm.init_position_stop("BTC-USDT-SWAP", 100.0, "long", 1.0)
        sm.reconcile_with_active_positions(set())
        assert sm._position_stops == {}
        sm2 = EnhancedStopLoss(_make_config(db), "grid")
        assert sm2._position_stops == {}

    def test_reconcile_no_db_path_no_crash(self, tmp_path):
        config = {"trading": {}, "strategies": {"grid": {}}}
        sm = EnhancedStopLoss(config, "grid")
        sm.init_position_stop("BTC-USDT-SWAP", 100.0, "long", 1.0)
        cleaned = sm.reconcile_with_active_positions({"ETH-USDT-SWAP"})
        assert cleaned >= 1
        assert sm._position_stops == {}
