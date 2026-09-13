"""
429 降级 / 限流重试回归测试（模块 6 高风险用例）
==============================================

补充现有 test_execution_core.py 未覆盖的精确行为：

_classify_error：
- OKX 业务限频码 51000 / 51001 / 51006 → RATE_LIMIT（真实限流码，比 HTTP 429 更重要）
- 3 位 5xx 兜底（520 / 599 等未显式列出的服务端错误）→ SERVER_ERROR
- 空错误码 / 超时关键词 → NETWORK_ERROR
- 不可重试业务错误 → None

_calculate_backoff_delay（关闭抖动 _retry_jitter_pct=0 后精确断言）：
- RATE_LIMIT 退避 = base * 2（更保守）
- NETWORK_ERROR 退避 = base * 0.5（快速重试）
- SERVER_ERROR / None = base
- 指数增长 multiplier^(attempt-1)
- 上限 max_delay 截断、下限 0.1s 兜底

通过 object.__new__ 注入配置属性，关闭随机抖动以做确定性断言。
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from execution.order_executor import OrderExecutor, RetryableError


# 与生产 __init__ 一致的可重试错误码映射
_RETRYABLE_CODES = {
    "429": RetryableError.RATE_LIMIT,
    "500": RetryableError.SERVER_ERROR,
    "502": RetryableError.SERVER_ERROR,
    "503": RetryableError.SERVER_ERROR,
    "504": RetryableError.SERVER_ERROR,
    "51000": RetryableError.RATE_LIMIT,
    "51001": RetryableError.RATE_LIMIT,
    "51006": RetryableError.RATE_LIMIT,
}


def _build(base_delay=1.0, max_delay=60.0, jitter_pct=0.0, multiplier=2.0):
    executor = OrderExecutor.__new__(OrderExecutor)
    executor._retryable_error_codes = dict(_RETRYABLE_CODES)
    executor._retry_base_delay = base_delay
    executor._retry_max_delay = max_delay
    executor._retry_jitter_pct = jitter_pct
    executor._retry_multiplier = multiplier
    return executor


class TestClassifyError:
    @pytest.mark.parametrize("code", ["429", "51000", "51001", "51006"])
    def test_rate_limit_codes(self, code):
        """HTTP 429 与 OKX 业务限频码均归为 RATE_LIMIT"""
        assert _build()._classify_error(code, "") == RetryableError.RATE_LIMIT

    @pytest.mark.parametrize("code", ["500", "502", "503", "504"])
    def test_known_server_error_codes(self, code):
        assert _build()._classify_error(code, "") == RetryableError.SERVER_ERROR

    @pytest.mark.parametrize("code", ["520", "599"])
    def test_three_digit_5xx_fallback(self, code):
        """未显式列出的 3 位 5xx 兜底为 SERVER_ERROR"""
        assert _build()._classify_error(code, "") == RetryableError.SERVER_ERROR

    def test_empty_code_is_network_error(self):
        assert _build()._classify_error("", "") == RetryableError.NETWORK_ERROR

    @pytest.mark.parametrize("msg", ["Connection timed out", "request timeout", "connection reset by peer"])
    def test_timeout_keyword_is_network_error(self, msg):
        assert _build()._classify_error("", msg) == RetryableError.NETWORK_ERROR

    @pytest.mark.parametrize("code", ["51008", "51121", "51169"])
    def test_non_retryable_business_errors(self, code):
        assert _build()._classify_error(code, "") is None


class TestBackoffDeterministic:
    def test_rate_limit_doubles_base(self):
        """RATE_LIMIT 退避 = base * 2"""
        delay = _build(base_delay=1.0)._calculate_backoff_delay(1, RetryableError.RATE_LIMIT)
        assert delay == pytest.approx(2.0)

    def test_network_halves_base(self):
        """NETWORK_ERROR 退避 = base * 0.5（快速重试）"""
        delay = _build(base_delay=1.0)._calculate_backoff_delay(1, RetryableError.NETWORK_ERROR)
        assert delay == pytest.approx(0.5)

    def test_server_error_uses_base(self):
        delay = _build(base_delay=1.0)._calculate_backoff_delay(1, RetryableError.SERVER_ERROR)
        assert delay == pytest.approx(1.0)

    def test_none_type_uses_base(self):
        delay = _build(base_delay=1.0)._calculate_backoff_delay(1, None)
        assert delay == pytest.approx(1.0)

    def test_exponential_growth(self):
        """第 3 次重试 = base * multiplier^2"""
        delay = _build(base_delay=1.0, multiplier=2.0)._calculate_backoff_delay(3, RetryableError.SERVER_ERROR)
        assert delay == pytest.approx(4.0)

    def test_max_delay_cap(self):
        """退避被 max_delay 截断"""
        delay = _build(base_delay=1.0, max_delay=10.0)._calculate_backoff_delay(10, RetryableError.SERVER_ERROR)
        assert delay == pytest.approx(10.0)

    def test_min_floor_0_1s(self):
        """极短退避被下限 0.1s 兜底"""
        delay = _build(base_delay=0.05)._calculate_backoff_delay(1, RetryableError.NETWORK_ERROR)
        assert delay == pytest.approx(0.1)
