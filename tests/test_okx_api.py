import pytest
from src.okx_api import get_account_balance, get_position

def test_get_account_balance():
    """测试获取账户余额"""
    balance = get_account_balance("USDT")
    assert isinstance(balance, float)
    assert balance >= 0

def test_get_position():
    """测试获取持仓"""
    pos = get_position("BTC-USDT-SWAP")
    assert "instId" in pos
    assert "posAmt" in pos
    assert isinstance(pos["posAmt"], float)