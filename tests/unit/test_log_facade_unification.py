"""
日志门面收敛测试（模块 7 (1)）
============================

验证 loguru 单一日志门面：事件ID注入 + 全局脱敏在同一个 patcher 中同时生效，
避免多次 logger.configure(patcher=...) 后调覆盖先调导致脱敏失效。

覆盖：
- redact_text 的 key=value / Authorization Bearer / clOrdId / 注册精确值脱敏
- redact_log_record 同时脱敏 message 与 extra，且幂等
- configure_event_id_logging() 组合注入事件ID + 脱敏（集成验证）
"""

import io
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from loguru import logger
from core.event_id import (
    configure_event_id_logging,
    set_current_event_id,
    clear_current_event_id,
    inject_event_id,
)
from core.log_redactor import get_log_redactor, redact_text, redact_log_record


def test_redact_text_key_value_and_auth():
    redactor = get_log_redactor()
    assert redactor.redact("api_key=abc123xyz") == "api_key=***"
    assert redactor.redact("passphrase: secretpass") == "passphrase: ***"
    assert redactor.redact("Authorization: Bearer tok123") == "Authorization: Bearer ***"
    assert redactor.redact("Authorization=rawtok") == "Authorization=***"


def test_redact_text_clordid():
    redactor = get_log_redactor()
    masked = redactor.redact("clOrdId: abcdefghijklmnop")
    assert "abcdefghijklmnop" not in masked
    assert masked.startswith("clOrdId: ")
    assert "***" in masked


def test_redact_registered_exact_value():
    redactor = get_log_redactor()
    redactor.register("SUPERSECRETVALUE123", "test")
    assert redactor.redact("token is SUPERSECRETVALUE123") == "token is ***test***"
    assert "SUPERSECRETVALUE123" not in redactor.redact("x=SUPERSECRETVALUE123")


def test_redact_log_record_masks_message_and_extra():
    redactor = get_log_redactor()
    redactor.register("LEAKME123456", "secret")
    record = {"message": "hello LEAKME123456", "extra": {"k": "v=LEAKME123456"}}
    redact_log_record(record)
    assert "LEAKME123456" not in record["message"]
    assert "LEAKME123456" not in record["extra"]["k"]

    # 幂等：重复脱敏不改变结果
    msg_before = record["message"]
    extra_before = record["extra"]["k"]
    redact_log_record(record)
    assert record["message"] == msg_before
    assert record["extra"]["k"] == extra_before


def test_inject_event_id_fills_default():
    record = {}
    inject_event_id(record)
    assert record["extra"]["event_id"] == "-"


def test_configure_event_id_logging_composes_injection_and_redaction():
    redactor = get_log_redactor()
    redactor.register("INTEGSECRET999", "itest")

    sink = io.StringIO()
    handler_id = logger.add(sink, level="DEBUG", format="{extra[event_id]} | {message}")
    try:
        configure_event_id_logging()
        set_current_event_id("evt-unittest-1")
        logger.info("placing order with INTEGSECRET999")
        clear_current_event_id()

        output = sink.getvalue()
        assert "evt-unittest-1" in output          # 事件ID已注入
        assert "INTEGSECRET999" not in output       # 敏感值已脱敏
        assert "***" in output
    finally:
        logger.remove(handler_id)
        logger.configure(patcher=None)
