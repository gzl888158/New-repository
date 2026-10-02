import asyncio

from core.okx_client import OKXClient
from core.ws_data_feed import WebSocketManager


class _FakeOKXClient:
    def __init__(self):
        self.book_requests = []

    def get_order_book(self, symbol, depth=5):
        self.book_requests.append((symbol, depth))
        return {"bids": [["100", "1"]], "asks": [["101", "1"]]}


def test_books_sequence_gap_uses_rest_fallback_and_waits_for_snapshot():
    okx = _FakeOKXClient()
    manager = WebSocketManager({}, okx)

    async def route(envelope):
        asyncio.run(manager._route_public_data(envelope))

    asyncio.run(manager._route_public_data({
        "arg": {"channel": "books", "instId": "BTC-USDT-SWAP"},
        "action": "snapshot",
        "data": [{"instId": "BTC-USDT-SWAP", "seqId": "100", "bids": [], "asks": []}],
    }))
    asyncio.run(manager._route_public_data({
        "arg": {"channel": "books", "instId": "BTC-USDT-SWAP"},
        "action": "update",
        "data": [{"instId": "BTC-USDT-SWAP", "prevSeqId": "100", "seqId": "101", "bids": [], "asks": []}],
    }))
    asyncio.run(manager._route_public_data({
        "arg": {"channel": "books", "instId": "BTC-USDT-SWAP"},
        "action": "update",
        "data": [{"instId": "BTC-USDT-SWAP", "prevSeqId": "99", "seqId": "102", "bids": [], "asks": []}],
    }))
    asyncio.run(manager._route_public_data({
        "arg": {"channel": "books", "instId": "BTC-USDT-SWAP"},
        "action": "update",
        "data": [{"instId": "BTC-USDT-SWAP", "prevSeqId": "102", "seqId": "103", "bids": [], "asks": []}],
    }))
    asyncio.run(manager._route_public_data({
        "arg": {"channel": "books", "instId": "BTC-USDT-SWAP"},
        "action": "snapshot",
        "data": [{"instId": "BTC-USDT-SWAP", "seqId": "200", "bids": [], "asks": []}],
    }))

    books = manager._book_buffer.pop_all()

    assert [book.get("seqId") for book in books if book.get("_source") == "websocket"] == ["100", "101", "200"]
    rest_snapshot = next(book for book in books if book.get("_source") == "rest_fallback")
    assert rest_snapshot["_sequence_valid"] is False
    assert rest_snapshot["_resync_reason"] == "sequence_gap"
    assert okx.book_requests == [("BTC-USDT-SWAP", 50)]
    assert manager._book_sequence_ids["BTC-USDT-SWAP"] == 200
    assert "BTC-USDT-SWAP" not in manager._book_resync_pending


def test_active_okx_client_rejects_gapped_book_deltas_until_snapshot():
    client = object.__new__(OKXClient)
    client._books_last_sequence = {}
    client._books_resync_pending = set()
    client._ws_public = None
    client.tick_callback = None
    rest_requests = []
    client.get_order_book = lambda symbol, depth: (
        rest_requests.append((symbol, depth))
        or {"bids": [["100", "1"]], "asks": [["101", "1"]]}
    )
    client._is_ws_open = lambda websocket: False
    client._parse_tick_data = lambda message, symbol: (symbol, message["data"][0].get("seqId"))

    async def route(action, book):
        return await client._consume_sequenced_orderbook(
            {"action": action, "data": [{"instId": "BTC-USDT-SWAP", **book}]},
            "BTC-USDT-SWAP",
        )

    async def verify():
        snapshot = await route("snapshot", {"seqId": "10"})
        contiguous = await route("update", {"prevSeqId": "10", "seqId": "11"})
        gap = await route("update", {"prevSeqId": "9", "seqId": "12"})
        dropped = await route("update", {"prevSeqId": "12", "seqId": "13"})
        recovered = await route("snapshot", {"seqId": "20"})
        return snapshot, contiguous, gap, dropped, recovered

    snapshot, contiguous, gap, dropped, recovered = asyncio.run(verify())

    assert snapshot == ("BTC-USDT-SWAP", "10")
    assert contiguous == ("BTC-USDT-SWAP", "11")
    assert gap is None
    assert dropped is None
    assert recovered == ("BTC-USDT-SWAP", "20")
    assert rest_requests == [("BTC-USDT-SWAP", 50)]