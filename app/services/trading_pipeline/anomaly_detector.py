"""
交易系统异常检测模块，识别价格、成交量、延迟与信号频率等异常。

.. deprecated:: 实验性模块，未接入生产交易链路。
"""
from enum import Enum
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from loguru import logger
import statistics
import time

from app.services.enterprise import EnterpriseServiceMixin


class AnomalyType(Enum):
    PRICE_SPIKE = "price_spike"
    VOLUME_SURGE = "volume_surge"
    ORDER_FAILURE_RATE = "order_failure_rate"
    LATENCY_SPIKE = "latency_spike"
    SIGNAL_FREQUENCY = "signal_frequency"
    PNL_DROP = "pnl_drop"
    CONFIDENCE_ANOMALY = "confidence_anomaly"


class AnomalySeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Anomaly:
    def __init__(self, anomaly_type: AnomalyType, severity: AnomalySeverity, 
                 message: str, details: Dict[str, Any] = None):
        self.type = anomaly_type
        self.severity = severity
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


class AnomalyDetector(EnterpriseServiceMixin):
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._detectors = {}
        self._anomalies: List[Anomaly] = []
        self._max_anomalies = config.get("max_anomalies", 1000)
        self._enabled = True

        self._register_default_detectors()

    def _register_default_detectors(self):
        self._detectors["price_spike"] = PriceSpikeDetector(self._config)
        self._detectors["volume_surge"] = VolumeSurgeDetector(self._config)
        self._detectors["order_failure"] = OrderFailureRateDetector(self._config)
        self._detectors["latency_spike"] = LatencySpikeDetector(self._config)
        self._detectors["signal_frequency"] = SignalFrequencyDetector(self._config)
        self._detectors["pnl_drop"] = PNLDropDetector(self._config)
        logger.info("Default anomaly detectors registered")

    def _detect_correlated_anomalies(self, recent: List[Anomaly]) -> List[Anomaly]:
        """检测关联异常模式"""
        correlated = []
        types = {}
        for a in recent:
            key = a.type.value
            types[key] = types.get(key, 0) + 1
        
        # 价格尖刺 + 成交量激增 = 极端行情警告
        if types.get("price_spike", 0) >= 2 and types.get("volume_surge", 0) >= 2:
            correlated.append(Anomaly(
                AnomalyType.CONFIDENCE_ANOMALY,
                AnomalySeverity.CRITICAL,
                f"Correlated anomaly: price spike + volume surge detected together",
                {"correlated_types": ["price_spike", "volume_surge"], "counts": types}
            ))
        
        # 延迟尖刺 + 订单失败率 = 系统故障警告
        if types.get("latency_spike", 0) >= 2 and types.get("order_failure", 0) >= 2:
            correlated.append(Anomaly(
                AnomalyType.CONFIDENCE_ANOMALY,
                AnomalySeverity.HIGH,
                f"Correlated anomaly: latency spike + order failure detected together",
                {"correlated_types": ["latency_spike", "order_failure"], "counts": types}
            ))
        
        return correlated

    def add_detector(self, name: str, detector):
        self._detectors[name] = detector
        logger.info(f"Added custom anomaly detector: {name}")

    def register_detector(self, detector):
        """Register a detector (alias for add_detector, auto-naming)"""
        name = detector.__class__.__name__
        self._detectors[name] = detector
        logger.info(f"Registered anomaly detector: {name}")

    def enable(self):
        self._enabled = True
        logger.info("Anomaly detector enabled")

    def disable(self):
        self._enabled = False
        logger.info("Anomaly detector disabled")

    async def detect(self, data_type: str, data: Dict[str, Any]) -> List[Anomaly]:
        if not self._enabled:
            return []

        anomalies = []
        for name, detector in self._detectors.items():
            if detector.handles(data_type):
                try:
                    detected = await detector.detect(data)
                    anomalies.extend(detected)
                except Exception as e:
                    self._handle_exception(e, module="AnomalyDetector", function="detect", severity="high", category="anomaly_detection")

        for anomaly in anomalies:
            self._record_anomaly(anomaly)

        # 关联异常检测
        recent_anomalies = [a for a in self._anomalies[-20:] if datetime.now() - a.timestamp < timedelta(minutes=5)]
        correlated = self._detect_correlated_anomalies(recent_anomalies)
        anomalies.extend(correlated)
        for c in correlated:
            self._record_anomaly(c)

        return anomalies

    def _record_anomaly(self, anomaly: Anomaly):
        self._anomalies.append(anomaly)
        if len(self._anomalies) > self._max_anomalies:
            self._anomalies = self._anomalies[-self._max_anomalies:]

        logger.warning(f"Anomaly detected: {anomaly.type.value} - {anomaly.message}")

    def get_anomalies(self, limit: int = 100, severity: Optional[AnomalySeverity] = None) -> List[Anomaly]:
        filtered = self._anomalies
        if severity:
            filtered = [a for a in filtered if a.severity == severity]
        return filtered[-limit:]

    async def start(self):
        """Start anomaly detector"""
        self._enabled = True
        logger.info("AnomalyDetector started")

    async def stop(self):
        """Stop anomaly detector"""
        self._enabled = False
        logger.info("AnomalyDetector stopped")

    def get_anomaly_summary(self) -> Dict[str, Any]:
        severity_counts = {}
        type_counts = {}
        
        for anomaly in self._anomalies:
            severity_counts[anomaly.severity.value] = severity_counts.get(anomaly.severity.value, 0) + 1
            type_counts[anomaly.type.value] = type_counts.get(anomaly.type.value, 0) + 1

        recent_24h = [a for a in self._anomalies if datetime.now() - a.timestamp < timedelta(hours=24)]

        return {
            "total_anomalies": len(self._anomalies),
            "recent_24h": len(recent_24h),
            "severity_distribution": severity_counts,
            "type_distribution": type_counts,
        }

    def get_anomaly_score(self) -> float:
        """计算异常评分 [0-100]，分数越高异常越严重"""
        now = datetime.now()
        recent_5m = [a for a in self._anomalies if now - a.timestamp < timedelta(minutes=5)]
        if not recent_5m:
            return 0.0
        
        severity_weights = {
            AnomalySeverity.LOW.value: 1,
            AnomalySeverity.MEDIUM.value: 5,
            AnomalySeverity.HIGH.value: 15,
            AnomalySeverity.CRITICAL.value: 30,
        }
        
        score = sum(severity_weights.get(a.severity.value, 1) for a in recent_5m)
        # 关联异常加分
        correlated = sum(1 for a in recent_5m if "correlated" in a.message.lower())
        score += correlated * 10
        
        return min(100.0, score)


class BaseDetector:
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._history: List[Dict[str, Any]] = []
        self._max_history = config.get("max_history", 100)
        self._adaptive_enabled = config.get("adaptive_thresholds", True)
        self._baseline_stats: Dict[str, float] = {}
        self._alert_cooldown: Dict[str, float] = {}

    def handles(self, data_type: str) -> bool:
        return False

    async def detect(self, data: Dict[str, Any]) -> List[Anomaly]:
        return []

    def _add_to_history(self, data: Dict[str, Any]):
        self._history.append(data)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def _update_baseline(self, metric_name: str, value: float):
        """更新自适应基线"""
        if not self._adaptive_enabled:
            return
        if metric_name not in self._baseline_stats:
            self._baseline_stats[metric_name] = value
        else:
            alpha = 0.05  # EMA平滑
            self._baseline_stats[metric_name] = alpha * value + (1 - alpha) * self._baseline_stats[metric_name]

    def _get_adaptive_threshold(self, metric_name: str, default: float) -> float:
        """获取自适应阈值(基于2倍基线)"""
        baseline = self._baseline_stats.get(metric_name, default)
        if self._adaptive_enabled and self._baseline_stats:
            return max(default, baseline * 2.0)
        return default

    def _check_cooldown(self, alert_key: str, cooldown_seconds: float = 30) -> bool:
        """检查告警冷却"""
        now = time.time()
        last = self._alert_cooldown.get(alert_key, 0)
        if now - last < cooldown_seconds:
            return True  # 冷却中，不告警
        self._alert_cooldown[alert_key] = now
        return False


class PriceSpikeDetector(BaseDetector):
    def handles(self, data_type: str) -> bool:
        return data_type == "price"

    async def detect(self, data: Dict[str, Any]) -> List[Anomaly]:
        anomalies = []
        price = float(data.get("price", 0))
        symbol = data.get("symbol", "")

        if not self._history:
            self._add_to_history({"price": price, "timestamp": datetime.now()})
            return anomalies

        recent_prices = [h["price"] for h in self._history[-20:]]
        if len(recent_prices) >= 10:
            mean_price = statistics.mean(recent_prices)
            std_price = statistics.stdev(recent_prices)
            
            if std_price > 0:
                z_score = abs(price - mean_price) / std_price
                spike_threshold = self._get_adaptive_threshold("price_spike", self._config.get("price_spike_threshold", 3.0))
                
                if z_score > spike_threshold:
                    severity = AnomalySeverity.CRITICAL if z_score > 5.0 else \
                               AnomalySeverity.HIGH if z_score > 4.0 else \
                               AnomalySeverity.MEDIUM
                    
                    anomalies.append(Anomaly(
                        AnomalyType.PRICE_SPIKE,
                        severity,
                        f"Price spike detected for {symbol}: {z_score:.2f} sigma",
                        {"price": price, "mean_price": mean_price, "z_score": z_score}
                    ))
                    self._update_baseline("price_volatility", std_price)

        self._add_to_history({"price": price, "timestamp": datetime.now()})
        # 冷却检查过滤
        actual_anomalies = []
        for a in anomalies:
            alert_key = f"price_spike_{symbol}"
            if not self._check_cooldown(alert_key, 60):
                actual_anomalies.append(a)
        return actual_anomalies


class VolumeSurgeDetector(BaseDetector):
    def handles(self, data_type: str) -> bool:
        return data_type == "volume"

    async def detect(self, data: Dict[str, Any]) -> List[Anomaly]:
        anomalies = []
        volume = float(data.get("volume", 0))
        symbol = data.get("symbol", "")

        if not self._history:
            self._add_to_history({"volume": volume, "timestamp": datetime.now()})
            return anomalies

        recent_volumes = [h["volume"] for h in self._history[-30:]]
        if len(recent_volumes) >= 10:
            mean_volume = statistics.mean(recent_volumes)
            surge_ratio = volume / mean_volume if mean_volume > 0 else 0
            surge_threshold = self._get_adaptive_threshold("volume_surge", self._config.get("volume_surge_threshold", 3.0))

            if surge_ratio > surge_threshold:
                severity = AnomalySeverity.HIGH if surge_ratio > 5.0 else \
                           AnomalySeverity.MEDIUM if surge_ratio > 4.0 else \
                           AnomalySeverity.LOW
                
                anomalies.append(Anomaly(
                    AnomalyType.VOLUME_SURGE,
                    severity,
                    f"Volume surge detected for {symbol}: {surge_ratio:.2f}x normal",
                    {"volume": volume, "mean_volume": mean_volume, "surge_ratio": surge_ratio}
                ))
                self._update_baseline("mean_volume", mean_volume)

        self._add_to_history({"volume": volume, "timestamp": datetime.now()})
        # 冷却检查过滤
        actual_anomalies = []
        for a in anomalies:
            alert_key = f"volume_surge_{symbol}"
            if not self._check_cooldown(alert_key, 30):
                actual_anomalies.append(a)
        return actual_anomalies


class OrderFailureRateDetector(BaseDetector):
    def handles(self, data_type: str) -> bool:
        return data_type == "order"

    async def detect(self, data: Dict[str, Any]) -> List[Anomaly]:
        anomalies = []
        status = data.get("status", "")
        is_failure = status in ("failed", "rejected", "error")

        self._add_to_history({"is_failure": is_failure, "timestamp": datetime.now()})

        recent_orders = self._history[-50:]
        if len(recent_orders) >= 20:
            failure_rate = sum(1 for o in recent_orders if o["is_failure"]) / len(recent_orders)
            failure_threshold = self._config.get("order_failure_threshold", 0.3)

            if failure_rate > failure_threshold:
                severity = AnomalySeverity.CRITICAL if failure_rate > 0.5 else \
                           AnomalySeverity.HIGH if failure_rate > 0.4 else \
                           AnomalySeverity.MEDIUM
                
                anomalies.append(Anomaly(
                    AnomalyType.ORDER_FAILURE_RATE,
                    severity,
                    f"High order failure rate: {failure_rate:.1%}",
                    {"failure_rate": failure_rate, "sample_size": len(recent_orders)}
                ))
                self._update_baseline("failure_rate", failure_rate)

        # 冷却检查过滤
        actual_anomalies = []
        for a in anomalies:
            alert_key = "order_failure"
            if not self._check_cooldown(alert_key, 30):
                actual_anomalies.append(a)
        return actual_anomalies


class LatencySpikeDetector(BaseDetector):
    def handles(self, data_type: str) -> bool:
        return data_type == "latency"

    async def detect(self, data: Dict[str, Any]) -> List[Anomaly]:
        anomalies = []
        latency_ms = float(data.get("latency_ms", 0))

        self._add_to_history({"latency_ms": latency_ms, "timestamp": datetime.now()})

        recent_latencies = [h["latency_ms"] for h in self._history[-50:]]
        if len(recent_latencies) >= 20:
            mean_latency = statistics.mean(recent_latencies)
            std_latency = statistics.stdev(recent_latencies)

            if std_latency > 0:
                z_score = abs(latency_ms - mean_latency) / std_latency
                latency_threshold = self._config.get("latency_spike_threshold", 3.0)

                if z_score > latency_threshold:
                    severity = AnomalySeverity.HIGH if z_score > 5.0 else \
                               AnomalySeverity.MEDIUM if z_score > 4.0 else \
                               AnomalySeverity.LOW
                    
                    anomalies.append(Anomaly(
                        AnomalyType.LATENCY_SPIKE,
                        severity,
                        f"Latency spike: {latency_ms:.1f}ms ({z_score:.2f} sigma)",
                        {"latency_ms": latency_ms, "mean_latency": mean_latency, "z_score": z_score}
                    ))
                    self._update_baseline("mean_latency", mean_latency)

        # 冷却检查过滤
        actual_anomalies = []
        for a in anomalies:
            alert_key = "latency_spike"
            if not self._check_cooldown(alert_key, 30):
                actual_anomalies.append(a)
        return actual_anomalies


class SignalFrequencyDetector(BaseDetector):
    def handles(self, data_type: str) -> bool:
        return data_type == "signal"

    async def detect(self, data: Dict[str, Any]) -> List[Anomaly]:
        anomalies = []
        now = datetime.now()
        
        self._add_to_history({"timestamp": now})

        recent_signals = [h for h in self._history if now - h["timestamp"] < timedelta(minutes=5)]
        signal_rate = len(recent_signals) / 5.0

        min_rate = self._config.get("min_signal_rate", 0.1)
        max_rate = self._config.get("max_signal_rate", 10)

        if signal_rate < min_rate and len(self._history) > 100:
            anomalies.append(Anomaly(
                AnomalyType.SIGNAL_FREQUENCY,
                AnomalySeverity.LOW,
                f"Low signal frequency: {signal_rate:.2f}/min",
                {"signal_rate": signal_rate, "min_rate": min_rate}
            ))
        elif signal_rate > max_rate:
            anomalies.append(Anomaly(
                AnomalyType.SIGNAL_FREQUENCY,
                AnomalySeverity.MEDIUM,
                f"High signal frequency: {signal_rate:.2f}/min",
                {"signal_rate": signal_rate, "max_rate": max_rate}
            ))

        return anomalies


class PNLDropDetector(BaseDetector):
    def handles(self, data_type: str) -> bool:
        return data_type == "pnl"

    async def detect(self, data: Dict[str, Any]) -> List[Anomaly]:
        anomalies = []
        pnl = float(data.get("pnl", 0))
        equity = float(data.get("equity", 0))

        self._add_to_history({"pnl": pnl, "equity": equity, "timestamp": datetime.now()})

        if len(self._history) >= 10:
            max_equity = max(h["equity"] for h in self._history)
            drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0
            drawdown_threshold = self._config.get("drawdown_threshold", 0.1)

            if drawdown > drawdown_threshold:
                severity = AnomalySeverity.CRITICAL if drawdown > 0.2 else \
                           AnomalySeverity.HIGH if drawdown > 0.15 else \
                           AnomalySeverity.MEDIUM
                
                anomalies.append(Anomaly(
                    AnomalyType.PNL_DROP,
                    severity,
                    f"High drawdown: {drawdown:.1%}",
                    {"drawdown": drawdown, "equity": equity, "max_equity": max_equity}
                ))

        return anomalies
