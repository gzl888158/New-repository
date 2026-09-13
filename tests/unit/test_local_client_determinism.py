"""
LocalOKXClient 确定性回放测试（模块 6 (4)）
==========================================

验证 local_client 在给定 seed 下可复现行情路径，避免测试依赖真实网络：

- 同 seed 生成相同的 ticker / kline / funding / orderbook / tick_data
- 不同 seed 生成不同的价格路径
- reset() 重新播种，可重复回放同一序列
- 价格演化 + 下单后的持仓状态确定性

注意：`ts` / `nextFundingTime` 等时间戳字段由 time.time() / datetime.now()
生成，天然不可复现，因此断言聚焦在价格/数量等市场数据字段，而非时间戳。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.local_client import LocalOKXClient


def _client(seed=42):
    return LocalOKXClient(
        {"symbols": ["BTC-USDT-SWAP"], "initial_balance": 100000.0},
        seed=seed,
    )


def _ticker_fields(t):
    """仅取价格/数量等确定性字段，排除 ts。"""
    return {k: t[k] for k in ("last", "bidPx", "askPx", "bidSz", "askSz", "vol24h")}


def _bar_fields(bar):
    """去掉第 0 列时间戳，保留 OHLCV。"""
    return bar[1:]


class TestLocalClientDeterminism:
    def test_same_seed_same_ticker(self):
        a, b = _client(42), _client(42)
        assert _ticker_fields(a.get_ticker("BTC-USDT-SWAP")) == _ticker_fields(b.get_ticker("BTC-USDT-SWAP"))

    def test_same_seed_same_kline(self):
        a, b = _client(42), _client(42)
        ka = [_bar_fields(bar) for bar in a.get_kline("BTC-USDT-SWAP", "1h", 5)]
        kb = [_bar_fields(bar) for bar in b.get_kline("BTC-USDT-SWAP", "1h", 5)]
        assert ka == kb

    def test_same_seed_same_funding(self):
        a, b = _client(42), _client(42)
        assert a.get_funding_rate("BTC-USDT-SWAP")["fundingRate"] == b.get_funding_rate("BTC-USDT-SWAP")["fundingRate"]

    def test_same_seed_same_orderbook(self):
        a, b = _client(42), _client(42)
        oa = a.get_order_book("BTC-USDT-SWAP")
        ob = b.get_order_book("BTC-USDT-SWAP")
        assert oa["asks"] == ob["asks"]
        assert oa["bids"] == ob["bids"]

    def test_different_seed_diverges_price_path(self):
        a, b = _client(42), _client(43)
        a._update_prices()
        b._update_prices()
        assert a.get_current_price("BTC-USDT-SWAP") != b.get_current_price("BTC-USDT-SWAP")

    def test_reset_replays_same_price_sequence(self):
        a = _client(42)
        seq1 = [a.get_ticker("BTC-USDT-SWAP")["last"] for _ in range(3)]
        a.reset()
        seq2 = [a.get_ticker("BTC-USDT-SWAP")["last"] for _ in range(3)]
        assert seq1 == seq2

    def test_price_update_deterministic(self):
        a, b = _client(42), _client(42)
        a._update_prices()
        b._update_prices()
        assert a.get_current_price("BTC-USDT-SWAP") == b.get_current_price("BTC-USDT-SWAP")

    def test_position_state_deterministic(self):
        a, b = _client(42), _client(42)
        a._update_prices()
        b._update_prices()
        a.place_order("BTC-USDT-SWAP", "buy", "market", 0.1, leverage=10)
        b.place_order("BTC-USDT-SWAP", "buy", "market", 0.1, leverage=10)
        pa = a.get_positions_dict()["BTC-USDT-SWAP"]
        pb = b.get_positions_dict()["BTC-USDT-SWAP"]
        assert pa.avg_cost == pb.avg_cost
        assert pa.quantity == pb.quantity
        assert pa.margin == pb.margin
