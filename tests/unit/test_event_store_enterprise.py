"""P1 企业级强化：事件溯源内核（seq/幂等/校验和/损坏检测/断点重放）单元测试。

覆盖：
1. seq 全局单调递增 + 重启恢复（断点续写）
2. 幂等去重（相同 event_id 重复写入被拒绝）
3. sha256 校验和写入 + 校验（篡改被检出）
4. 损坏行检测与统计告警
5. from_seq 断点重放
6. stats 可观测性
"""

import json
import os
from datetime import datetime

import pytest

from core.event_store import EventStore


class TestEventStoreEnterprise:
    def _make_store(self, tmp_path):
        return EventStore(data_dir=str(tmp_path / "events"))

    # ── 序列号 ─────────────────────────────────────────
    def test_seq_monotonic(self, tmp_path):
        store = self._make_store(tmp_path)
        for i in range(3):
            store.append("tick", {"i": i})
        records = store.replay()
        seqs = [r["seq"] for r in records]
        assert seqs == [1, 2, 3]
        assert store.last_seq == 3

    def test_seq_restored_across_instances(self, tmp_path):
        path = str(tmp_path / "events")
        store1 = EventStore(data_dir=path)
        for i in range(3):
            store1.append("tick", {"i": i})

        # 模拟重启：新实例从磁盘恢复 last_seq，续写不跳号
        store2 = EventStore(data_dir=path)
        assert store2.last_seq == 3
        store2.append("tick", {"i": 3})
        seqs = [r["seq"] for r in store2.replay()]
        assert seqs == [1, 2, 3, 4]

    # ── 幂等去重 ─────────────────────────────────────────
    def test_idempotent_dedup(self, tmp_path):
        store = self._make_store(tmp_path)
        eid = "evt-fixed-00001"
        store.append("order_placed", {"symbol": "ETH"}, event_id=eid)
        store.append("order_placed", {"symbol": "ETH"}, event_id=eid)  # 重复

        assert store.count() == 1
        assert store.stats()["duplicates_rejected"] == 1

    # ── 校验和 ─────────────────────────────────────────
    def test_checksum_written_and_verified(self, tmp_path):
        store = self._make_store(tmp_path)
        store.append("order_placed", {"symbol": "ETH-USDT-SWAP"})
        records = store.replay(verify_checksum=True)
        assert len(records) == 1
        assert len(records[0]["checksum"]) == 64  # sha256 hex
        assert store.stats()["checksum_mismatches"] == 0

    def test_checksum_tamper_detected(self, tmp_path):
        path = str(tmp_path / "events")
        store = EventStore(data_dir=path)
        store.append("order_placed", {"symbol": "ETH-USDT-SWAP", "qty": 1})

        # 篡改磁盘上的 data（模拟日志被篡改/损坏）
        day = datetime.now().strftime("%Y-%m-%d")
        fpath = os.path.join(path, f"events_{day}.jsonl")
        with open(fpath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        rec["data"]["qty"] = 999  # 篡改成与 checksum 不一致
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # 校验时应检出并跳过该条
        store2 = EventStore(data_dir=path)
        records = store2.replay(verify_checksum=True)
        assert len(records) == 0
        assert store2.stats()["checksum_mismatches"] == 1

        # 不校验时仍可读（向后兼容）
        records_no_check = store2.replay()
        assert len(records_no_check) == 1

    # ── 损坏行检测 ─────────────────────────────────────────
    def test_corrupt_line_detected(self, tmp_path):
        path = str(tmp_path / "events")
        store = EventStore(data_dir=path)
        store.append("good", {"v": 1})
        day = datetime.now().strftime("%Y-%m-%d")
        fpath = os.path.join(path, f"events_{day}.jsonl")
        with open(fpath, "a", encoding="utf-8") as f:
            f.write("this is not valid json\n")

        records = store.replay()
        assert len(records) == 1  # 损坏行被跳过
        assert store.stats()["corrupt_lines"] == 1

    # ── 断点重放 ─────────────────────────────────────────
    def test_from_seq_replay(self, tmp_path):
        store = self._make_store(tmp_path)
        for i in range(5):
            store.append("tick", {"i": i})

        tail = store.replay(from_seq=2)
        assert [r["seq"] for r in tail] == [3, 4, 5]

    # ── stats ─────────────────────────────────────────
    def test_stats(self, tmp_path):
        store = self._make_store(tmp_path)
        store.append("a", {})
        store.append("b", {})
        s = store.stats()
        assert s["appended"] == 2
        assert s["last_seq"] == 2
        assert s["duplicates_rejected"] == 0
        assert s["corrupt_lines"] == 0
        assert s["checksum_mismatches"] == 0

    # ── 哈希链（不可篡改） ─────────────────────────────
    def test_checksum_chain_linked(self, tmp_path):
        """每条事件的 checksum 链入前一条，verify_chain 对无篡改链返回 valid。"""
        store = self._make_store(tmp_path)
        store.append("a", {"i": 1})
        store.append("b", {"i": 2})
        store.append("c", {"i": 3})

        records = store.replay()
        # 三条 checksum 应彼此不同（链入了不同的 prev_checksum）
        assert len({r["checksum"] for r in records}) == 3

        report = store.verify_chain()
        assert report["valid"] is True
        assert report["scanned"] == 3
        assert report["verified"] == 3
        assert report["mismatches"] == 0
        assert report["first_invalid_seq"] is None

    def test_verify_chain_detects_middle_tamper(self, tmp_path):
        """篡改中间记录的数据 → verify_chain 检出首个断裂点并判 invalid。"""
        path = str(tmp_path / "events")
        store = EventStore(data_dir=path)
        store.append("a", {"i": 1})
        store.append("b", {"i": 2})
        store.append("c", {"i": 3})

        day = datetime.now().strftime("%Y-%m-%d")
        fpath = os.path.join(path, f"events_{day}.jsonl")
        with open(fpath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 3
        recs = [json.loads(l) for l in lines]
        recs[1]["data"]["i"] = 999  # 篡改 seq=2 的数据，checksum 字段保持原值
        with open(fpath, "w", encoding="utf-8") as f:
            f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))

        store2 = EventStore(data_dir=path)
        report = store2.verify_chain()
        assert report["valid"] is False
        assert report["first_invalid_seq"] == 2
        assert report["mismatches"] >= 1

    def test_verify_chain_detects_deletion(self, tmp_path):
        """删除中间记录 → 后续记录链头错位，verify_chain 判 invalid。"""
        path = str(tmp_path / "events")
        store = EventStore(data_dir=path)
        store.append("a", {"i": 1})
        store.append("b", {"i": 2})
        store.append("c", {"i": 3})

        day = datetime.now().strftime("%Y-%m-%d")
        fpath = os.path.join(path, f"events_{day}.jsonl")
        with open(fpath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # 删除中间记录 (seq=2)
        del lines[1]
        with open(fpath, "w", encoding="utf-8") as f:
            f.writelines(lines)

        store2 = EventStore(data_dir=path)
        report = store2.verify_chain()
        assert report["valid"] is False
        assert report["first_invalid_seq"] == 3

    def test_legacy_checksum_accepted(self, tmp_path):
        """升级前的旧版（非链式）校验和事件应视为合法，而非误判为篡改。"""
        path = str(tmp_path / "events")
        os.makedirs(path, exist_ok=True)
        rec = {
            "seq": 1,
            "event_id": "legacy-1",
            "event_type": "order_placed",
            "timestamp": datetime.now().isoformat(),
            "source": "event_bus",
            "symbol": "ETH-USDT-SWAP",
            "version": 1,
            "data": {"symbol": "ETH-USDT-SWAP", "qty": 1},
        }
        rec["checksum"] = EventStore._compute_checksum_legacy(
            rec["event_type"], rec["timestamp"], rec["seq"], rec["data"]
        )
        day = datetime.now().strftime("%Y-%m-%d")
        with open(os.path.join(path, f"events_{day}.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        store = EventStore(data_dir=path)
        report = store.verify_chain()
        assert report["valid"] is True
        assert report["mismatches"] == 0
        # 旧版事件仍可被重放读出（链路复位，非篡改）
        records = store.replay(verify_checksum=True)
        assert len(records) == 1

    def test_restart_after_legacy_continues_clean_chain(self, tmp_path):
        """重启后续写：旧版尾不进入链头，新事件从空串续链，verify_chain 仍有效。"""
        path = str(tmp_path / "events")
        os.makedirs(path, exist_ok=True)
        legacy_rec = {
            "seq": 1,
            "event_id": "legacy-1",
            "event_type": "order_placed",
            "timestamp": datetime.now().isoformat(),
            "source": "event_bus",
            "symbol": "ETH-USDT-SWAP",
            "version": 1,
            "data": {"symbol": "ETH-USDT-SWAP", "qty": 1},
        }
        legacy_rec["checksum"] = EventStore._compute_checksum_legacy(
            legacy_rec["event_type"], legacy_rec["timestamp"],
            legacy_rec["seq"], legacy_rec["data"],
        )
        day = datetime.now().strftime("%Y-%m-%d")
        with open(os.path.join(path, f"events_{day}.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps(legacy_rec, ensure_ascii=False) + "\n")

        # 模拟重启：新实例恢复后立即续写一条新事件
        store = EventStore(data_dir=path)
        store.append("order_filled", {"symbol": "ETH-USDT-SWAP", "qty": 1})

        report = store.verify_chain()
        assert report["valid"] is True
        assert report["mismatches"] == 0
        # 新事件 seq 从 2 起，且与旧版尾正确断链（链首为空串）
        records = store.replay()
        assert [r["seq"] for r in records] == [1, 2]