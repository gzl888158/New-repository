"""
端到端延迟追踪测试（模块 7 (4)）
================================

验证 core.apm_monitor 的调用链追踪能力：
- Span 父级自动传播：嵌套 start_span 共享同一 trace_id
- 慢请求标记：总耗时 > slow_threshold_ms 时 get_recent_traces 输出 is_slow=True
- 端到端交易阶段链 TradeFlowTrace：价格到达→计算→信号→风控→下单串成一条 trace
- trace_function 装饰器异常路径不再因缺 record_exception 而抛 AttributeError
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.apm_monitor import (
    Tracer,
    Span,
    SpanStatus,
    TradeFlowTrace,
    TRADE_FLOW_STAGES,
    get_apm_monitor,
)


def _make_tracer(slow_threshold_ms: float = 500.0) -> Tracer:
    return Tracer(service_name="test", slow_threshold_ms=slow_threshold_ms)


def test_start_span_inherits_parent_context():
    tracer = _make_tracer()
    root = tracer.start_span("root")
    child = tracer.start_span("child")  # 未显式传 parent，应自动继承 root
    tracer.finish_span(child)
    tracer.finish_span(root)

    assert child.trace_id == root.trace_id
    assert child.parent_span is root


def test_nested_spans_share_same_trace_id():
    tracer = _make_tracer()
    a = tracer.start_span("a")
    b = tracer.start_span("b")
    c = tracer.start_span("c")
    tracer.finish_span(c)
    tracer.finish_span(b)
    tracer.finish_span(a)

    assert a.trace_id == b.trace_id == c.trace_id


def test_independent_roots_have_distinct_trace_ids():
    tracer = _make_tracer()
    a = tracer.start_span("first")
    tracer.finish_span(a)
    b = tracer.start_span("second")
    tracer.finish_span(b)

    assert a.trace_id != b.trace_id


def test_get_recent_traces_marks_slow():
    tracer = _make_tracer(slow_threshold_ms=10.0)  # 极低阈值便于测试
    span = tracer.start_span("slow_span")
    time.sleep(0.03)  # 30ms > 10ms 阈值
    tracer.finish_span(span)

    traces = tracer.get_recent_traces()
    assert traces, "应存在已完成 trace"
    assert traces[0]["is_slow"] is True
    assert traces[0]["slow_threshold_ms"] == 10.0


def test_get_recent_traces_not_slow_under_threshold():
    tracer = _make_tracer(slow_threshold_ms=500.0)
    span = tracer.start_span("fast_span")
    tracer.finish_span(span)

    traces = tracer.get_recent_traces()
    assert traces
    assert traces[0]["is_slow"] is False


def test_trade_flow_trace_single_trace_with_stages():
    tracer = _make_tracer()
    flow = TradeFlowTrace(symbol="BTC-USDT-SWAP", tracer=tracer)

    span_ids = []
    for stage in TRADE_FLOW_STAGES:
        s = flow.start_stage(stage)
        span_ids.append(s.span_id)
        flow.finish_stage(s)

    flow.finish()

    traces = tracer.get_recent_traces()
    matching = [t for t in traces if t["trace_id"] == flow.trace_id]
    assert len(matching) == 1, "所有阶段应串成同一条 trace"
    # 根 span + 5 个阶段 span
    assert matching[0]["span_count"] == 1 + len(TRADE_FLOW_STAGES)


def test_trade_flow_trace_context_manager_auto_finishes():
    tracer = _make_tracer()
    with TradeFlowTrace(symbol="ETH-USDT-SWAP", tracer=tracer) as flow:
        s = flow.start_stage("signal_generation")
        flow.finish_stage(s)

    traces = tracer.get_recent_traces()
    matching = [t for t in traces if t["trace_id"] == flow.trace_id]
    assert len(matching) == 1
    assert matching[0]["span_count"] == 2  # 根 + 1 阶段


def test_trace_function_exception_records_error_without_attribute_error():
    """trace_function 异常路径调用 span.record_exception，此前该方法缺失会抛 AttributeError。"""
    from core.apm_monitor import trace_function

    @trace_function("boom")
    def boom():
        raise ValueError("boom!")

    try:
        boom()
    except ValueError:
        pass
    else:
        raise AssertionError("应抛出 ValueError")

    apm = get_apm_monitor()
    traces = apm.get_recent_traces()
    # 至少存在一条 error 状态的 span（trace_function 修复后，异常会正确记录并 finish）
    boom_spans = [
        s for t in traces for s in t["spans"]
        if s["name"] == "boom" and s["status"] == SpanStatus.ERROR.value
    ]
    assert boom_spans, "异常路径应记录 error 状态的 span"


def test_trace_function_success_finishes_span():
    from core.apm_monitor import trace_function

    @trace_function("ok_op")
    def ok_op():
        return 42

    assert ok_op() == 42

    apm = get_apm_monitor()
    traces = apm.get_recent_traces()
    ok_spans = [
        s for t in traces for s in t["spans"]
        if s["name"] == "ok_op"
    ]
    assert ok_spans, "成功路径应记录并 finish span"


def test_span_record_exception_sets_status_and_tags():
    span = Span("err_span")
    span.record_exception(ValueError("bad"))
    assert span.status == SpanStatus.ERROR
    assert span.tags["error"] is True
    assert span.tags["error_type"] == "ValueError"
