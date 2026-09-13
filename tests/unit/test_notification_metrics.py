"""
生产级通知分发与指标流水线集成测试
====================================
覆盖：NotificationDispatcher + MetricsPipeline
"""

import os
import sys
import json
import time
import asyncio
import unittest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.notification_dispatcher import (
    NotificationDispatcher,
    NotificationPriority,
    NotificationChannel,
    ChannelConfig,
    Notification,
    DeliveryStatus,
    DeliveryRecord,
    TemplateEngine,
    RateLimiter,
    BatchProcessor,
    BaseChannel,
    ConsoleChannel,
    TelegramChannel,
    WebhookChannel,
    EmailChannel,
    create_dispatcher_from_config,
    get_notification_dispatcher,
)
from core.metrics_pipeline import (
    MetricsPipeline,
    MetricRegistry,
    AggregationEngine,
    MetricExporter,
    DerivedMetricsEngine,
    MetricHealthChecker,
    MetricSnapshot,
    MetricType,
    MetricCategory,
    MetricDefinition,
    MetricSample,
    AggregatedMetric,
    create_pipeline_from_config,
    get_metrics_pipeline,
)


# ═══════════════════════════════════════════════════════════════
# 测试辅助
# ═══════════════════════════════════════════════════════════════

def load_config():
    import yaml
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestNotificationDispatcher(unittest.TestCase):
    """通知分发器测试"""

    def setUp(self):
        self.config = {"notification": {}, "system": {"data_dir": "data"}}
        self.dispatcher = NotificationDispatcher(self.config)

    def test_01_initialization(self):
        """初始化"""
        self.assertIsNotNone(self.dispatcher)
        self.assertEqual(self.dispatcher.get_queue_size(), 0)
        stats = self.dispatcher.get_stats()
        self.assertIn("channels", stats)
        self.assertIn("console", stats["channels"])
        print(f"  [PASS] test_01: dispatcher initialized with {len(stats['channels'])} channels")

    def test_02_configure_channel(self):
        """配置通道"""
        self.dispatcher.configure_channel(
            NotificationChannel.WEBHOOK,
            ChannelConfig(
                channel=NotificationChannel.WEBHOOK,
                enabled=True,
                endpoint="http://localhost:9999/webhook",
                auth_token="test_token",
            )
        )
        stats = self.dispatcher.get_channel_stats()
        self.assertIn("webhook", stats)
        self.assertTrue(stats["webhook"]["enabled"])
        print(f"  [PASS] test_02: webhook channel configured")

    def test_03_disable_enable_channel(self):
        """禁用/启用通道"""
        self.dispatcher.configure_channel(
            NotificationChannel.WEBHOOK,
            ChannelConfig(channel=NotificationChannel.WEBHOOK, enabled=True)
        )
        self.dispatcher.disable_channel(NotificationChannel.WEBHOOK)
        stats = self.dispatcher.get_channel_stats()
        self.assertFalse(stats["webhook"]["enabled"])

        self.dispatcher.enable_channel(NotificationChannel.WEBHOOK)
        stats = self.dispatcher.get_channel_stats()
        self.assertTrue(stats["webhook"]["enabled"])
        print(f"  [PASS] test_03: channel disable/enable works")

    def test_04_send_console(self):
        """发送控制台通知"""
        async def _test():
            msg_id = await self.dispatcher.send(
                channel=NotificationChannel.CONSOLE,
                title="Test",
                message="Hello World",
                priority=NotificationPriority.INFO,
                category="test",
            )
            self.assertTrue(msg_id.startswith("notif_"))
            return msg_id

        msg_id = asyncio.run(_test())
        print(f"  [PASS] test_04: console notification sent: {msg_id}")

    def test_05_send_emergency(self):
        """发送紧急通知（立即发送）"""
        async def _test():
            msg_id = await self.dispatcher.send(
                channel=NotificationChannel.CONSOLE,
                title="EMERGENCY",
                message="Critical alert!",
                priority=NotificationPriority.EMERGENCY,
                category="risk",
            )
            return msg_id

        msg_id = asyncio.run(_test())
        self.assertTrue(msg_id.startswith("notif_"))
        print(f"  [PASS] test_05: emergency notification sent: {msg_id}")

    def test_06_send_template(self):
        """使用模板发送通知"""
        async def _test():
            msg_id = await self.dispatcher.send_template(
                channel=NotificationChannel.CONSOLE,
                template="risk_alert",
                priority=NotificationPriority.CRITICAL,
                category="risk",
                severity_label="严重",
                dimension="回撤",
                value=0.25,
                threshold=0.20,
                level="CRITICAL",
                timestamp="2024-01-01 00:00:00",
                suggestion="立即减仓",
            )
            return msg_id

        msg_id = asyncio.run(_test())
        self.assertTrue(msg_id.startswith("notif_"))
        print(f"  [PASS] test_06: template notification sent: {msg_id}")

    def test_07_send_to_all(self):
        """向所有通道发送"""
        self.dispatcher.configure_channel(
            NotificationChannel.WEBHOOK,
            ChannelConfig(channel=NotificationChannel.WEBHOOK, enabled=True, endpoint="http://localhost:9999/webhook")
        )

        async def _test():
            ids = await self.dispatcher.send_to_all(
                title="Broadcast",
                message="All channels",
                priority=NotificationPriority.WARNING,
                category="system",
            )
            return ids

        ids = asyncio.run(_test())
        self.assertGreater(len(ids), 0)
        print(f"  [PASS] test_07: sent to {len(ids)} channels")

    def test_08_send_to_all_exclude(self):
        """排除特定通道"""
        async def _test():
            ids = await self.dispatcher.send_to_all(
                title="Broadcast",
                message="All except webhook",
                priority=NotificationPriority.WARNING,
                category="system",
                exclude=[NotificationChannel.WEBHOOK],
            )
            return ids

        ids = asyncio.run(_test())
        self.assertGreater(len(ids), 0)
        print(f"  [PASS] test_08: sent to {len(ids)} channels (webhook excluded)")

    def test_09_start_stop(self):
        """启动和停止"""
        async def _test():
            await self.dispatcher.start()
            self.assertTrue(self.dispatcher._running)
            await self.dispatcher.stop()
            self.assertFalse(self.dispatcher._running)

        asyncio.run(_test())
        print(f"  [PASS] test_09: start/stop works")

    def test_10_queue_enqueue(self):
        """队列入队"""
        async def _test():
            await self.dispatcher.start()
            # 发送多条消息
            for i in range(5):
                await self.dispatcher.send(
                    channel=NotificationChannel.CONSOLE,
                    title=f"Test {i}",
                    message=f"Message {i}",
                    priority=NotificationPriority.INFO,
                    category="test",
                )
            qsize = self.dispatcher.get_queue_size()
            await self.dispatcher.stop()
            return qsize

        qsize = asyncio.run(_test())
        print(f"  [PASS] test_10: queue size after 5 sends = {qsize}")

    def test_11_get_stats(self):
        """获取统计"""
        stats = self.dispatcher.get_stats()
        self.assertIn("queue_size", stats)
        self.assertIn("channels", stats)
        self.assertIn("silent_hours_active", stats)
        self.assertIn("total_templates", stats)
        self.assertGreater(stats["total_templates"], 0)
        print(f"  [PASS] test_11: stats = {stats['total_templates']} templates, {len(stats['channels'])} channels")

    def test_12_register_template(self):
        """注册自定义模板"""
        self.dispatcher.register_template("custom_test", "Custom: {message}")
        template = self.dispatcher.get_template("custom_test")
        self.assertIsNotNone(template)
        self.assertEqual(template, "Custom: {message}")
        print(f"  [PASS] test_12: custom template registered")

    def test_13_delivery_callback(self):
        """投递回调"""
        records = []

        def callback(record):
            records.append(record)

        self.dispatcher.on_delivery(callback)

        async def _test():
            await self.dispatcher.send(
                channel=NotificationChannel.CONSOLE,
                title="Callback Test",
                message="Testing callback",
                priority=NotificationPriority.EMERGENCY,
                category="test",
            )

        asyncio.run(_test())
        self.assertGreater(len(records), 0)
        self.assertIsInstance(records[0], DeliveryRecord)
        print(f"  [PASS] test_13: delivery callback received {len(records)} record(s)")

    def test_14_get_delivery_records(self):
        """获取投递记录"""
        async def _test():
            await self.dispatcher.start()
            await self.dispatcher.send(
                channel=NotificationChannel.CONSOLE,
                title="Record Test",
                message="Testing records",
                priority=NotificationPriority.EMERGENCY,
                category="test",
            )
            await asyncio.sleep(0.1)
            records = self.dispatcher.get_delivery_records(limit=10)
            await self.dispatcher.stop()
            return records

        records = asyncio.run(_test())
        self.assertGreater(len(records), 0)
        self.assertEqual(records[0]["channel"], "console")
        print(f"  [PASS] test_14: {len(records)} delivery record(s)")

    def test_15_create_from_config(self):
        """从配置创建分发器"""
        config = load_config()
        dispatcher = create_dispatcher_from_config(config)
        stats = dispatcher.get_stats()
        self.assertIn("channels", stats)
        print(f"  [PASS] test_15: created from config, channels={list(stats['channels'].keys())}")


class TestNotificationComponents(unittest.TestCase):
    """通知组件单元测试"""

    def test_16_priority_ordering(self):
        """优先级排序"""
        self.assertEqual(NotificationPriority.EMERGENCY.value, 0)
        self.assertEqual(NotificationPriority.CRITICAL.value, 1)
        self.assertEqual(NotificationPriority.WARNING.value, 2)
        self.assertEqual(NotificationPriority.INFO.value, 3)

        # 验证优先级队列排序
        import asyncio
        q = asyncio.PriorityQueue()
        n1 = Notification(priority=NotificationPriority.INFO.value, channel="console", title="Low", message="low")
        n2 = Notification(priority=NotificationPriority.EMERGENCY.value, channel="console", title="High", message="high")
        q.put_nowait(n1)
        q.put_nowait(n2)
        first = q.get_nowait()
        self.assertEqual(first.priority, NotificationPriority.EMERGENCY.value)
        print(f"  [PASS] test_16: priority ordering correct")

    def test_17_rate_limiter(self):
        """限流器测试"""
        limiter = RateLimiter(per_minute=3, per_hour=10)
        self.assertTrue(limiter.can_send())
        limiter.record()
        limiter.record()
        self.assertTrue(limiter.can_send())
        limiter.record()
        self.assertFalse(limiter.can_send())

        # EMERGENCY 不限流
        self.assertTrue(limiter.can_send(NotificationPriority.EMERGENCY))

        stats = limiter.get_stats()
        self.assertEqual(stats["minute_count"], 3)
        print(f"  [PASS] test_17: rate limiter works, stats={stats}")

    def test_18_template_engine(self):
        """模板引擎测试"""
        engine = TemplateEngine()
        result = engine.render("risk_alert",
            severity_label="严重",
            dimension="回撤",
            value=0.25,
            threshold=0.20,
            level="CRITICAL",
            timestamp="2024-01-01",
            suggestion="减仓",
        )
        self.assertIn("严重", result)
        self.assertIn("25.00%", result)
        self.assertIn("回撤", result)
        print(f"  [PASS] test_18: template rendered: {result[:50]}...")

        # 测试缺失键 - 返回原始模板
        result2 = engine.render("risk_alert", message="fallback")
        self.assertIn("{severity_label}", result2)
        print(f"  [PASS] test_18b: template missing key fallback")

    def test_19_batch_processor(self):
        """批处理器测试"""
        bp = BatchProcessor(window_sec=5.0, max_batch=3)

        # 紧急消息不批处理
        n1 = Notification(priority=NotificationPriority.EMERGENCY.value, channel="console", title="E", message="e")
        result = bp.add(n1)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].priority, NotificationPriority.EMERGENCY.value)

        # 低优先级消息批处理
        n2 = Notification(priority=NotificationPriority.INFO.value, channel="console", title="1", message="1", category="test")
        n3 = Notification(priority=NotificationPriority.INFO.value, channel="console", title="2", message="2", category="test")
        self.assertIsNone(bp.add(n2))
        self.assertIsNone(bp.add(n3))

        # 达到最大批处理数量
        n4 = Notification(priority=NotificationPriority.INFO.value, channel="console", title="3", message="3", category="test")
        result = bp.add(n4)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        print(f"  [PASS] test_19: batch processor works, batch size={len(result)}")

    def test_20_channel_circuit_breaker(self):
        """通道熔断测试"""
        config = ChannelConfig(
            channel=NotificationChannel.TELEGRAM,
            enabled=True,
            circuit_breaker_threshold=3,
            circuit_breaker_cooldown=60.0,
        )
        channel = TelegramChannel(config)

        # 初始可用
        self.assertTrue(channel.is_available)

        # 连续失败触发熔断
        channel.record_failure()
        channel.record_failure()
        self.assertTrue(channel.is_available)
        channel.record_failure()
        self.assertFalse(channel.is_available)
        print(f"  [PASS] test_20: circuit breaker opened after 3 failures")

    def test_21_console_channel(self):
        """控制台通道"""
        config = ChannelConfig(channel=NotificationChannel.CONSOLE, enabled=True)
        channel = ConsoleChannel(config)

        async def _test():
            n = Notification(priority=NotificationPriority.INFO.value, channel="console", title="Test", message="test")
            record = await channel.send(n)
            return record

        record = asyncio.run(_test())
        self.assertEqual(record.status, DeliveryStatus.DELIVERED)
        self.assertEqual(record.channel, "console")
        print(f"  [PASS] test_21: console channel delivered, latency={record.latency_ms:.2f}ms")

    def test_22_webhook_channel(self):
        """Webhook通道（无端点）"""
        config = ChannelConfig(channel=NotificationChannel.WEBHOOK, enabled=True, endpoint="")
        channel = WebhookChannel(config)

        async def _test():
            n = Notification(priority=NotificationPriority.INFO.value, channel="webhook", title="Test", message="test")
            record = await channel.send(n)
            return record

        record = asyncio.run(_test())
        self.assertEqual(record.status, DeliveryStatus.DROPPED)
        self.assertIn("not configured", record.error)
        print(f"  [PASS] test_22: webhook dropped when not configured")

    def test_23_telegram_channel(self):
        """Telegram通道（无配置）"""
        config = ChannelConfig(channel=NotificationChannel.TELEGRAM, enabled=True, auth_token="", chat_id="")
        channel = TelegramChannel(config)

        async def _test():
            n = Notification(priority=NotificationPriority.INFO.value, channel="telegram", title="Test", message="test")
            record = await channel.send(n)
            return record

        record = asyncio.run(_test())
        self.assertEqual(record.status, DeliveryStatus.DROPPED)
        self.assertIn("not configured", record.error)
        print(f"  [PASS] test_23: telegram dropped when not configured")


# ═══════════════════════════════════════════════════════════════
# MetricsPipeline 测试
# ═══════════════════════════════════════════════════════════════

class TestMetricsPipeline(unittest.TestCase):
    """指标流水线测试"""

    def setUp(self):
        self.config = {"metrics_pipeline": {}, "system": {"data_dir": "data"}}
        self.pipeline = MetricsPipeline(self.config)

    def test_24_initialization(self):
        """初始化流水线"""
        self.assertIsNotNone(self.pipeline)
        stats = self.pipeline.get_stats()
        self.assertIn("registry_size", stats)
        self.assertGreater(stats["registry_size"], 0)
        print(f"  [PASS] test_24: pipeline initialized with {stats['registry_size']} metrics")

    def test_25_record_metric(self):
        """记录指标"""
        self.pipeline.record("trading_pnl_total", 150.5, {"strategy": "grid"})
        self.pipeline.record("trading_pnl_total", 155.0, {"strategy": "grid"})
        self.pipeline.record("trading_pnl_total", 148.0, {"strategy": "grid"})

        agg = self.pipeline.get_metric("trading_pnl_total")
        self.assertIsNotNone(agg)
        self.assertEqual(agg.count, 3)
        self.assertAlmostEqual(agg.avg, 151.17, delta=0.1)
        self.assertAlmostEqual(agg.latest, 148.0)
        print(f"  [PASS] test_25: recorded 3 samples, avg={agg.avg:.2f}, latest={agg.latest}")

    def test_26_record_batch(self):
        """批量记录指标"""
        samples = [
            MetricSample("trading_trades_total", 10, {"strategy": "grid"}),
            MetricSample("trading_trades_total", 15, {"strategy": "grid"}),
            MetricSample("trading_trades_total", 20, {"strategy": "grid"}),
            MetricSample("trading_win_rate", 65.0, {"strategy": "grid"}),
            MetricSample("trading_win_rate", 70.0, {"strategy": "grid"}),
        ]
        self.pipeline.record_batch(samples)

        agg = self.pipeline.get_metric("trading_trades_total")
        self.assertIsNotNone(agg)
        self.assertEqual(agg.count, 3)

        agg2 = self.pipeline.get_metric("trading_win_rate")
        self.assertIsNotNone(agg2)
        self.assertEqual(agg2.count, 2)
        print(f"  [PASS] test_26: batch recorded trades={agg.count}, win_rate={agg2.count}")

    def test_27_increment_counter(self):
        """递增计数器"""
        self.pipeline.increment("trading_trades_total", 1.0)
        self.pipeline.increment("trading_trades_total", 1.0)
        self.pipeline.increment("trading_trades_total", 1.0)

        agg = self.pipeline.get_metric("trading_trades_total")
        self.assertIsNotNone(agg)
        self.assertEqual(agg.count, 3)
        self.assertAlmostEqual(agg.latest, 3.0)
        print(f"  [PASS] test_27: counter incremented to {agg.latest}")

    def test_28_record_latency(self):
        """记录延迟指标"""
        self.pipeline.record_latency("perf_order_place_ms", 150.0)
        self.pipeline.record_latency("perf_order_place_ms", 200.0)
        self.pipeline.record_latency("perf_order_place_ms", 180.0)

        agg = self.pipeline.get_metric("perf_order_place_ms")
        self.assertIsNotNone(agg)
        self.assertEqual(agg.count, 3)
        self.assertAlmostEqual(agg.max, 200.0)
        print(f"  [PASS] test_28: latency recorded, avg={agg.avg:.1f}ms, max={agg.max}ms")

    def test_29_aggregation_statistics(self):
        """聚合统计计算"""
        for v in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            self.pipeline.record("test_gauge", float(v))

        agg = self.pipeline.get_metric("test_gauge")
        self.assertEqual(agg.count, 10)
        self.assertAlmostEqual(agg.min, 10.0)
        self.assertAlmostEqual(agg.max, 100.0)
        self.assertAlmostEqual(agg.avg, 55.0)
        self.assertAlmostEqual(agg.p50, 55.0, delta=5.0)
        self.assertGreater(agg.p95, 0)
        self.assertGreater(agg.stddev, 0)
        print(f"  [PASS] test_29: aggregation stats: avg={agg.avg}, p50={agg.p50}, p95={agg.p95}, stddev={agg.stddev:.1f}")

    def test_30_get_metrics_by_category(self):
        """按类别获取指标"""
        self.pipeline.record("trading_pnl_total", 100.0)
        self.pipeline.record("risk_score", 0.35)
        self.pipeline.record("capital_total", 1000.0)

        trading = self.pipeline.get_metrics_by_category(MetricCategory.TRADING)
        risk = self.pipeline.get_metrics_by_category(MetricCategory.RISK)
        capital = self.pipeline.get_metrics_by_category(MetricCategory.CAPITAL)

        self.assertIn("trading_pnl_total", trading)
        self.assertIn("risk_score", risk)
        self.assertIn("capital_total", capital)
        print(f"  [PASS] test_30: categories - trading={len(trading)}, risk={len(risk)}, capital={len(capital)}")

    def test_31_get_metrics_by_labels(self):
        """按标签获取指标"""
        self.pipeline.record("strategy_pnl", 50.0, {"strategy": "grid"})
        self.pipeline.record("strategy_pnl", 80.0, {"strategy": "grid"})
        self.pipeline.record("strategy_pnl", 30.0, {"strategy": "trend"})
        self.pipeline.record("strategy_pnl", 60.0, {"strategy": "trend"})

        by_strategy = self.pipeline.get_metrics_by_labels("strategy_pnl", "strategy")
        self.assertIn("grid", by_strategy)
        self.assertIn("trend", by_strategy)
        self.assertAlmostEqual(by_strategy["grid"].avg, 65.0)
        self.assertAlmostEqual(by_strategy["trend"].avg, 45.0)
        print(f"  [PASS] test_31: labels - grid avg={by_strategy['grid'].avg}, trend avg={by_strategy['trend'].avg}")

    def test_32_derived_metrics(self):
        """派生指标计算"""
        daily_returns = [0.01, -0.005, 0.02, -0.01, 0.015, 0.008, -0.003, 0.012, -0.008, 0.018]
        equity_curve = [1000, 1010, 1005, 1025, 1015, 1030, 1038, 1035, 1047, 1039, 1058]

        derived = self.pipeline.get_derived_metrics(
            daily_returns=daily_returns,
            equity_curve=equity_curve,
            wins=6, total_trades=10,
            gross_profit=0.083, gross_loss=-0.026,
        )

        self.assertIn("sharpe_ratio", derived)
        self.assertIn("sortino_ratio", derived)
        self.assertIn("max_drawdown", derived)
        self.assertIn("win_rate", derived)
        self.assertIn("profit_factor", derived)

        self.assertAlmostEqual(derived["win_rate"], 0.6)
        self.assertGreater(derived["sharpe_ratio"], 0)
        self.assertGreater(derived["profit_factor"], 1.0)
        print(f"  [PASS] test_32: derived - sharpe={derived['sharpe_ratio']:.2f}, "
              f"sortino={derived['sortino_ratio']:.2f}, "
              f"max_dd={derived['max_drawdown']:.2%}, "
              f"win_rate={derived['win_rate']:.0%}, "
              f"profit_factor={derived['profit_factor']:.2f}")

    def test_33_export_prometheus(self):
        """导出Prometheus格式"""
        self.pipeline.record("trading_pnl_total", 100.0)
        self.pipeline.record("risk_score", 0.35)
        self.pipeline.record("capital_total", 1000.0)

        prom = self.pipeline.export("prometheus")
        self.assertIn("trading_pnl_total", prom)
        self.assertIn("HELP", prom)
        self.assertIn("TYPE", prom)
        print(f"  [PASS] test_33: prometheus export ({len(prom)} chars)")

    def test_34_export_json(self):
        """导出JSON格式"""
        self.pipeline.record("trading_pnl_total", 100.0)
        self.pipeline.record("risk_score", 0.35)

        json_str = self.pipeline.export("json")
        data = json.loads(json_str)
        self.assertIn("metrics", data)
        self.assertIn("definitions", data)
        self.assertIn("trading_pnl_total", data["metrics"])
        print(f"  [PASS] test_34: json export with {len(data['metrics'])} metrics and {len(data['definitions'])} definitions")

    def test_35_export_csv(self):
        """导出CSV格式"""
        self.pipeline.record("trading_pnl_total", 100.0)
        self.pipeline.record("risk_score", 0.35)

        csv_str = self.pipeline.export("csv")
        self.assertIn("metric,count", csv_str)
        self.assertIn("trading_pnl_total", csv_str)
        self.assertIn("risk_score", csv_str)
        print(f"  [PASS] test_35: csv export with {len(csv_str.splitlines())} lines")

    def test_36_export_influxdb(self):
        """导出InfluxDB格式"""
        self.pipeline.record("trading_pnl_total", 100.0)
        self.pipeline.record("risk_score", 0.35)

        influx = self.pipeline.export("influxdb")
        self.assertIn("trading_metrics", influx)
        self.assertIn("metric=trading_pnl_total", influx)
        print(f"  [PASS] test_36: influxdb export with {len(influx.splitlines())} lines")

    def test_37_health_check(self):
        """健康检查"""
        # 正常指标
        self.pipeline.record("risk_drawdown_pct", 0.05)
        self.pipeline.record("risk_score", 0.2)

        health = self.pipeline.check_health()
        self.assertTrue(health["healthy"])
        self.assertEqual(len(health["anomalies"]), 0)

        # 异常指标
        self.pipeline.record("risk_drawdown_pct", 0.25)
        self.pipeline.record("risk_score", 0.8)

        health2 = self.pipeline.check_health()
        self.assertFalse(health2["healthy"])
        self.assertGreater(len(health2["anomalies"]), 0)
        print(f"  [PASS] test_37: health check - healthy={health['healthy']}, anomalous={len(health2['anomalies'])} anomalies")

    def test_38_snapshot(self):
        """快照管理"""
        self.pipeline.record("trading_pnl_total", 100.0)
        self.pipeline.record("risk_score", 0.35)

        snapshot = self.pipeline.take_snapshot({"event": "test"})
        self.assertIn("timestamp", snapshot)
        self.assertIn("metrics", snapshot)
        self.assertIn("metadata", snapshot)

        latest = self.pipeline.get_latest_snapshot()
        self.assertIsNotNone(latest)
        self.assertEqual(latest["metadata"]["event"], "test")

        snapshots = self.pipeline.get_snapshots(limit=5)
        self.assertGreater(len(snapshots), 0)
        print(f"  [PASS] test_38: snapshot taken with {len(snapshot['metrics'])} metrics")

    def test_39_start_stop(self):
        """启动和停止流水线"""
        async def _test():
            await self.pipeline.start()
            self.assertTrue(self.pipeline._running)
            await asyncio.sleep(0.1)
            await self.pipeline.stop()
            self.assertFalse(self.pipeline._running)

        asyncio.run(_test())
        print(f"  [PASS] test_39: pipeline start/stop works")

    def test_40_register_custom_metric(self):
        """注册自定义指标"""
        custom = MetricDefinition(
            name="my_custom_counter",
            description="自定义计数器",
            unit="count",
            type=MetricType.COUNTER,
            category=MetricCategory.CUSTOM,
            labels=["tag"],
        )
        self.pipeline.register_metric(custom)

        definition = self.pipeline.get_metric_definition("my_custom_counter")
        self.assertIsNotNone(definition)
        self.assertEqual(definition.name, "my_custom_counter")
        self.assertEqual(definition.type, MetricType.COUNTER)
        print(f"  [PASS] test_40: custom metric registered: {definition.name}")

    def test_41_anomaly_callbacks(self):
        """异常回调"""
        anomalies_received = []

        def on_anomaly(anomaly):
            anomalies_received.append(anomaly)

        self.pipeline.on_anomaly(on_anomaly)

        # 触发异常
        self.pipeline.record("risk_drawdown_pct", 0.30)
        self.pipeline.check_health()

        self.assertGreater(len(anomalies_received), 0)
        self.assertEqual(anomalies_received[0]["metric"], "risk_drawdown_pct")
        self.assertEqual(anomalies_received[0]["severity"], "critical")
        print(f"  [PASS] test_41: anomaly callback received {len(anomalies_received)} anomaly(s)")

    def test_42_create_from_config(self):
        """从配置创建流水线"""
        config = load_config()
        pipeline = create_pipeline_from_config(config)
        stats = pipeline.get_stats()
        self.assertGreater(stats["registry_size"], 0)
        print(f"  [PASS] test_42: created from config, {stats['registry_size']} metrics")


# ═══════════════════════════════════════════════════════════════
# 组件单元测试
# ═══════════════════════════════════════════════════════════════

class TestMetricComponents(unittest.TestCase):
    """指标组件单元测试"""

    def test_43_metric_registry(self):
        """指标注册中心"""
        registry = MetricRegistry()
        all_defs = registry.get_all()
        self.assertGreater(len(all_defs), 0)

        # 按类别过滤
        system = registry.get_by_category(MetricCategory.SYSTEM)
        trading = registry.get_by_category(MetricCategory.TRADING)
        self.assertGreater(len(system), 0)
        self.assertGreater(len(trading), 0)

        # 标签管理
        registry.set_label("strategy_pnl", "strategy", "grid")
        labels = registry.get_labels("strategy_pnl")
        self.assertEqual(labels["strategy"], "grid")
        print(f"  [PASS] test_43: registry - {len(all_defs)} metrics, {len(system)} system, {len(trading)} trading")

    def test_44_aggregation_engine(self):
        """聚合引擎"""
        engine = AggregationEngine(window_sec=60.0, max_samples=100)
        self.assertEqual(engine.get_sample_count("test"), 0)

        engine.add_sample(MetricSample("test", 10.0))
        engine.add_sample(MetricSample("test", 20.0))
        engine.add_sample(MetricSample("test", 30.0))

        self.assertEqual(engine.get_sample_count("test"), 3)

        agg = engine.aggregate("test")
        self.assertEqual(agg.count, 3)
        self.assertAlmostEqual(agg.avg, 20.0)
        print(f"  [PASS] test_44: engine - count={agg.count}, avg={agg.avg}")

    def test_45_metric_exporter(self):
        """指标导出器"""
        registry = MetricRegistry()
        exporter = MetricExporter(registry)

        metrics = {
            "test_gauge": AggregatedMetric(name="test_gauge", count=3, avg=50.0, latest=55.0,
                                           min=40.0, max=60.0, p50=50.0, p95=58.0, p99=60.0, stddev=8.0),
        }

        prom = exporter.export_prometheus(metrics)
        self.assertIn("test_gauge", prom)

        json_str = exporter.export_json(metrics)
        self.assertIn("test_gauge", json_str)

        csv_str = exporter.export_csv(metrics)
        self.assertIn("test_gauge", csv_str)

        influx = exporter.export_influxdb(metrics)
        self.assertIn("test_gauge", influx)
        print(f"  [PASS] test_45: exporter - prometheus, json, csv, influxdb all work")

    def test_46_derived_metrics_engine(self):
        """派生指标计算引擎"""
        engine = DerivedMetricsEngine()

        # 夏普比率
        returns = [0.01, -0.005, 0.02, -0.01, 0.015]
        sharpe = engine.compute_sharpe_ratio(returns)
        self.assertGreater(sharpe, 0)

        # 最大回撤
        equity = [100, 110, 105, 95, 100, 115, 100, 90, 95]
        max_dd = engine.compute_max_drawdown(equity)
        self.assertGreater(max_dd, 0)
        self.assertLess(max_dd, 1.0)

        # 盈亏比
        pf = engine.compute_profit_factor(100.0, -50.0)
        self.assertEqual(pf, 2.0)

        # 期望值
        expectancy = engine.compute_expectancy(10.0, -5.0, 0.6)
        self.assertAlmostEqual(expectancy, 4.0)

        print(f"  [PASS] test_46: derived - sharpe={sharpe:.2f}, max_dd={max_dd:.2%}, pf={pf}, expect={expectancy}")

    def test_47_health_checker(self):
        """健康检查器"""
        checker = MetricHealthChecker()
        checker.set_threshold("test_metric", warning=50, critical=80, min=0)

        # 正常
        metrics = {"test_metric": AggregatedMetric(name="test_metric", latest=30.0)}
        anomalies = checker.check(metrics)
        self.assertEqual(len(anomalies), 0)

        # 警告
        metrics["test_metric"] = AggregatedMetric(name="test_metric", latest=60.0)
        anomalies = checker.check(metrics)
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["severity"], "warning")

        # 严重
        metrics["test_metric"] = AggregatedMetric(name="test_metric", latest=90.0)
        anomalies = checker.check(metrics)
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["severity"], "critical")

        summary = checker.get_anomaly_summary()
        self.assertGreater(summary["total_anomalies"], 0)
        print(f"  [PASS] test_47: health checker - {summary['total_anomalies']} total anomalies")

    def test_48_snapshot_persistence(self):
        """快照持久化"""
        snap = MetricSnapshot("data")

        metrics = {
            "test": AggregatedMetric(name="test", count=3, avg=50.0, latest=55.0),
        }

        snapshot = snap.take_snapshot(metrics, {"event": "test"})
        snap.save_snapshot(snapshot)

        latest = snap.get_latest_snapshot()
        self.assertIsNotNone(latest)
        self.assertIn("test", latest["metrics"])

        snapshots = snap.get_snapshots()
        self.assertGreater(len(snapshots), 0)
        print(f"  [PASS] test_48: snapshot persisted with {len(latest['metrics'])} metrics")


# ═══════════════════════════════════════════════════════════════
# 端到端集成测试
# ═══════════════════════════════════════════════════════════════

class TestE2EIntegration(unittest.TestCase):
    """端到端集成测试"""

    def test_49_e2e_notification_pipeline(self):
        """通知分发端到端流程"""
        async def _test():
            dispatcher = NotificationDispatcher({"notification": {}, "system": {"data_dir": "data"}})

            # 配置通道
            dispatcher.configure_channel(
                NotificationChannel.CONSOLE,
                ChannelConfig(channel=NotificationChannel.CONSOLE, enabled=True)
            )

            # 启动
            await dispatcher.start()

            # 发送多条不同类型的通知
            await dispatcher.send(
                channel=NotificationChannel.CONSOLE,
                title="Risk Alert",
                message="回撤超过20%",
                priority=NotificationPriority.EMERGENCY,
                category="risk",
            )

            await dispatcher.send_template(
                channel=NotificationChannel.CONSOLE,
                template="daily_report",
                priority=NotificationPriority.WARNING,
                category="report",
                date="2024-01-01",
                total_pnl=50.0,
                total_trades=10,
                win_rate=0.6,
                total_fees=2.5,
                max_drawdown=0.1,
                strategy_performance="grid: +30, trend: +20",
            )

            await dispatcher.send_to_all(
                title="System Status",
                message="All systems operational",
                priority=NotificationPriority.INFO,
                category="system",
            )

            # 等待投递完成
            await asyncio.sleep(0.2)

            # 验证
            stats = dispatcher.get_stats()
            records = dispatcher.get_delivery_records()
            self.assertGreater(len(records), 0)

            await dispatcher.stop()
            return stats, records

        stats, records = asyncio.run(_test())
        print(f"  [PASS] test_49: E2E notification - {len(records)} deliveries, "
              f"channels={len(stats['channels'])}")

    def test_50_e2e_metrics_pipeline(self):
        """指标流水线端到端流程"""
        async def _test():
            pipeline = MetricsPipeline({"metrics_pipeline": {}, "system": {"data_dir": "data"}})

            # 记录各类指标
            pipeline.record("trading_pnl_total", 150.0)
            pipeline.record("trading_pnl_daily", 25.0)
            pipeline.record("trading_trades_total", 12)
            pipeline.record("trading_win_rate", 65.0)
            pipeline.record("trading_fees_total", 3.5)
            pipeline.record("trading_slippage_total", 0.5)

            pipeline.record("risk_score", 0.35)
            pipeline.record("risk_drawdown_pct", 0.08)
            pipeline.record("risk_margin_ratio", 0.25)
            pipeline.record("risk_total_exposure", 500.0)

            pipeline.record("capital_total", 1000.0)
            pipeline.record("capital_available", 600.0)
            pipeline.record("capital_allocated", 400.0)
            pipeline.record("capital_efficiency", 75.0)

            pipeline.record("perf_signal_process_ms", 45.0)
            pipeline.record("perf_order_place_ms", 120.0)
            pipeline.record("perf_ws_latency_ms", 200.0)

            # 启动流水线
            await pipeline.start()

            # 等待一个聚合周期
            await asyncio.sleep(0.2)

            # 获取聚合指标
            metrics = pipeline.get_aggregated_metrics()
            self.assertGreater(len(metrics), 0)

            # 按类别获取
            trading = pipeline.get_metrics_by_category(MetricCategory.TRADING)
            risk = pipeline.get_metrics_by_category(MetricCategory.RISK)
            capital = pipeline.get_metrics_by_category(MetricCategory.CAPITAL)
            self.assertGreater(len(trading), 0)
            self.assertGreater(len(risk), 0)
            self.assertGreater(len(capital), 0)

            # 导出
            prom = pipeline.export("prometheus")
            json_str = pipeline.export("json")
            csv_str = pipeline.export("csv")
            self.assertGreater(len(prom), 0)
            self.assertGreater(len(json_str), 0)
            self.assertGreater(len(csv_str), 0)

            # 健康检查
            health = pipeline.check_health()

            # 快照
            snapshot = pipeline.take_snapshot({"event": "e2e_test"})
            self.assertIsNotNone(snapshot)

            # 统计
            stats = pipeline.get_stats()

            await pipeline.stop()
            return metrics, trading, risk, capital, health, stats

        metrics, trading, risk, capital, health, stats = asyncio.run(_test())
        print(f"  [PASS] test_50: E2E metrics - {len(metrics)} agg metrics, "
              f"trading={len(trading)}, risk={len(risk)}, capital={len(capital)}, "
              f"healthy={health['healthy']}, "
              f"registry={stats['registry_size']}")

    def test_51_e2e_full_integration(self):
        """通知+指标完整集成"""
        async def _test():
            config = {"notification": {}, "metrics_pipeline": {}, "system": {"data_dir": "data"}}

            # 创建两个模块
            dispatcher = NotificationDispatcher(config)
            pipeline = MetricsPipeline(config)

            # 配置通知通道
            dispatcher.configure_channel(
                NotificationChannel.CONSOLE,
                ChannelConfig(channel=NotificationChannel.CONSOLE, enabled=True)
            )

            # 启动
            await dispatcher.start()
            await pipeline.start()

            # 模拟交易场景：记录指标 + 发送通知
            # 1. 记录策略指标
            pipeline.record("strategy_pnl", 50.0, {"strategy": "grid"})
            pipeline.record("strategy_win_rate", 65.0, {"strategy": "grid"})

            # 2. 发送交易信号通知
            await dispatcher.send_template(
                channel=NotificationChannel.CONSOLE,
                template="trade_signal",
                priority=NotificationPriority.INFO,
                category="trading",
                symbol="BTC-USDT",
                direction="LONG",
                signal_type="trend_breakout",
                price=65000.0,
                quantity=0.01,
                confidence=0.75,
            )

            # 3. 记录风险指标
            pipeline.record("risk_drawdown_pct", 0.05)
            pipeline.record("risk_score", 0.25)

            # 4. 发送风险告警（如果触发）
            health = pipeline.check_health()
            if not health["healthy"]:
                for anomaly in health["anomalies"]:
                    await dispatcher.send(
                        channel=NotificationChannel.CONSOLE,
                        title=f"Metric Anomaly: {anomaly['metric']}",
                        message=anomaly["message"],
                        priority=NotificationPriority.WARNING,
                        category="risk",
                        metadata=anomaly,
                    )

            # 等待处理
            await asyncio.sleep(0.2)

            # 验证
            dispatcher_stats = dispatcher.get_stats()
            pipeline_stats = pipeline.get_stats()
            delivery_records = dispatcher.get_delivery_records()

            await dispatcher.stop()
            await pipeline.stop()

            return dispatcher_stats, pipeline_stats, delivery_records, health

        dispatcher_stats, pipeline_stats, records, health = asyncio.run(_test())
        print(f"  [PASS] test_51: full integration - "
              f"deliveries={len(records)}, "
              f"pipeline metrics={pipeline_stats['aggregated_metrics']}, "
              f"healthy={health['healthy']}")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)