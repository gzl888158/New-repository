"""
信号处理流水线，对交易信号进行去重、冷却与质量/市场状态过滤。

.. deprecated:: 实验性模块，未接入生产交易链路。
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from loguru import logger
import asyncio
import time

from app.services.enterprise import EnterpriseServiceMixin


class SignalProcessingPipeline(EnterpriseServiceMixin):
    def __init__(self, config: Dict[str, Any], quality_engine=None, regime_engine=None, strategy_coordinator=None):
        self._config = config
        self._quality_engine = quality_engine
        self._regime_engine = regime_engine
        self._strategy_coordinator = strategy_coordinator
        self._processors = []
        self._cooldown_map = config.get("cooldown_map", {})
        self._last_signal_time: Dict[str, datetime] = {}
        self._signal_history: List[Dict[str, Any]] = []
        self._max_history_size = config.get("max_history_size", 1000)
        self._stats = {
            "total_received": 0,
            "total_passed": 0,
            "total_rejected": 0,
            "rejection_reasons": {},
            "avg_processing_ms": 0,
            "processing_times": [],
        }
        self._dedup_window = config.get("dedup_window_seconds", 5)
        self._dedup_cache: Dict[str, float] = {}

    def add_processor(self, processor):
        self._processors.append(processor)
        logger.info(f"Added signal processor: {processor.__class__.__name__}")

    async def process(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        import time
        start_time = time.time()
        self._stats["total_received"] += 1
        if not self._validate_signal_data(signal_data):
            self._stats["total_rejected"] += 1
            return {"signal": signal_data, "valid": False, "errors": ["Invalid signal data"], "warnings": []}
        result = {"signal": signal_data, "valid": True, "errors": [], "warnings": []}

        for processor in self._processors:
            try:
                processor_result = await processor.process(signal_data, result)
                if processor_result is not None:
                    result.update(processor_result)
                    if not result.get("valid", True):
                        logger.warning(f"Signal rejected by {processor.__class__.__name__}")
                        break
            except Exception as e:
                self._handle_exception(e, module="SignalProcessingPipeline", function="process", severity="high", category="signal_processing")
                result["valid"] = False
                result["errors"].append(str(e))
                break

        elapsed = (time.time() - start_time) * 1000
        self._record_latency("signal_pipeline_process", elapsed, {"valid": str(result.get("valid"))})
        self._stats["processing_times"].append(elapsed)
        if len(self._stats["processing_times"]) > 500:
            self._stats["processing_times"] = self._stats["processing_times"][-500:]
        if self._stats["processing_times"]:
            self._stats["avg_processing_ms"] = sum(self._stats["processing_times"]) / len(self._stats["processing_times"])

        if result.get("valid"):
            self._stats["total_passed"] += 1
            self._record_signal(signal_data)
        else:
            self._stats["total_rejected"] += 1
            reason = result.get("errors", ["unknown"])[0] if result.get("errors") else "unknown"
            self._stats["rejection_reasons"][reason] = self._stats["rejection_reasons"].get(reason, 0) + 1

        return result

    def _record_signal(self, signal_data: Dict[str, Any]):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": signal_data.get("symbol", ""),
            "direction": signal_data.get("direction", ""),
            "strategy_name": signal_data.get("strategy_name", ""),
            "confidence": signal_data.get("confidence", 0.5),
        }
        self._signal_history.append(entry)
        if len(self._signal_history) > self._max_history_size:
            self._signal_history = self._signal_history[-self._max_history_size:]

    def get_signal_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self._signal_history[-limit:]

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """获取流水线统计"""
        return {
            "total_received": self._stats["total_received"],
            "total_passed": self._stats["total_passed"],
            "total_rejected": self._stats["total_rejected"],
            "pass_rate": self._stats["total_passed"] / max(self._stats["total_received"], 1),
            "avg_processing_ms": round(self._stats["avg_processing_ms"], 2),
            "rejection_reasons": dict(sorted(
                self._stats["rejection_reasons"].items(),
                key=lambda x: x[1], reverse=True
            )[:10]),
        }


class SignalValidator:
    def __init__(self, config: Dict[str, Any]):
        self._required_fields = config.get("required_fields", ["symbol", "direction", "strategy_name"])
        self._valid_directions = config.get("valid_directions", ["buy", "sell", "long", "short"])

    async def process(self, signal_data: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        errors = []
        
        for field in self._required_fields:
            if field not in signal_data:
                errors.append(f"Missing required field: {field}")
        
        direction = signal_data.get("direction", "").lower()
        if direction and direction not in self._valid_directions:
            errors.append(f"Invalid direction: {direction}")
        
        if errors:
            return {"valid": False, "errors": errors}
        
        return {"valid": True}


class SignalCooldownFilter:
    def __init__(self, config: Dict[str, Any]):
        self._cooldown_map = config.get("cooldown_map", {})
        self._default_cooldown = config.get("default_cooldown", 30)
        self._last_signal_time: Dict[str, datetime] = {}
        self._dedup_times: Dict[str, datetime] = {}
        self._dedup_window = config.get("dedup_window_seconds", 5)

    async def process(self, signal_data: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        symbol = signal_data.get("symbol", "")
        strategy_name = signal_data.get("strategy_name", "")
        direction = signal_data.get("direction", "")
        volatility = signal_data.get("volatility", 0)
        
        # 自适应冷却：高波动缩短冷却，低波动延长冷却
        cooldown = self._cooldown_map.get(strategy_name, self._default_cooldown)
        if volatility > 0.05:
            cooldown = int(cooldown * 0.5)  # 高波动缩短50%
        elif volatility < 0.01:
            cooldown = int(cooldown * 1.5)  # 低波动延长50%
        
        # 去重检查（同一symbol+strategy+direction在dedup窗口内）
        dedup_key = f"{symbol}:{strategy_name}:{direction}"
        now = datetime.now()
        
        if hasattr(self, '_dedup_times'):
            last_dedup = self._dedup_times.get(dedup_key)
            if last_dedup and (now - last_dedup).total_seconds() < getattr(self, '_dedup_window', 5):
                return {"valid": False, "errors": [f"Duplicate signal for {symbol} {strategy_name}"]}
            if not hasattr(self, '_dedup_times'):
                self._dedup_times = {}
            if not hasattr(self, '_dedup_window'):
                self._dedup_window = 5
            self._dedup_times[dedup_key] = now
        
        signal_key = f"{symbol}:{strategy_name}:{direction}"
        last_time = self._last_signal_time.get(signal_key)
        
        if last_time and (now - last_time).total_seconds() < cooldown:
            return {"valid": False, "errors": [f"Signal cooldown active for {strategy_name} {symbol} ({cooldown}s)"]}
        
        self._last_signal_time[signal_key] = now
        return None


class SignalQualityFilter:
    def __init__(self, config: Dict[str, Any]):
        self._min_confidence = config.get("min_confidence", 0.3)
        self._quality_engine = None

    def set_quality_engine(self, engine):
        self._quality_engine = engine

    async def process(self, signal_data: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        confidence = signal_data.get("confidence", 0.5)
        
        if confidence < self._min_confidence:
            return {"valid": False, "errors": [f"Confidence {confidence:.2f} below minimum {self._min_confidence}"]}
        
        if self._quality_engine:
            is_acceptable, breakdown = self._quality_engine.is_signal_acceptable(signal_data)
            if not is_acceptable:
                return {"valid": False, "errors": [f"Signal quality rejected"], "quality_breakdown": breakdown}
            return {"quality_breakdown": breakdown}
        
        return None


class MarketRegimeFilter:
    def __init__(self, config: Dict[str, Any]):
        self._regime_engine = None
        self._min_adjustment_factor = config.get("min_adjustment_factor", 0.7)

    def set_regime_engine(self, engine):
        self._regime_engine = engine

    async def process(self, signal_data: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self._regime_engine:
            return None
        
        strategy_name = signal_data.get("strategy_name", "")
        recommendation = self._regime_engine.get_strategy_recommendation(strategy_name)
        adjustment_factor = recommendation.get("adjustment_factor", 1.0)
        
        if adjustment_factor < self._min_adjustment_factor:
            return {"valid": False, "errors": [f"Strategy {strategy_name} incompatible with current market regime"]}
        
        return {"regime_adjustment": adjustment_factor}


class RiskFilter:
    def __init__(self, config: Dict[str, Any], global_risk=None, strategy_risk=None):
        self._global_risk = global_risk
        self._strategy_risk = strategy_risk

    async def process(self, signal_data: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self._global_risk and not self._global_risk.can_trade():
            return {"valid": False, "errors": ["Global trading is paused"]}
        
        if self._strategy_risk and not self._strategy_risk.validate_signal(signal_data):
            return {"valid": False, "errors": ["Strategy risk validation failed"]}
        
        return None


class ConflictFilter:
    def __init__(self, config: Dict[str, Any], coordinator=None, trade_journal=None):
        self._coordinator = coordinator
        self._trade_journal = trade_journal

    async def process(self, signal_data: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        strategy_name = signal_data.get("strategy_name", "")
        symbol = signal_data.get("symbol", "")
        direction = signal_data.get("direction", "")

        if self._trade_journal:
            open_positions = self._trade_journal._open_positions
            if symbol in open_positions:
                existing_direction = open_positions[symbol].direction.lower()
                opposite_direction = "short" if direction.lower() in ("buy", "long") else "long"
                if existing_direction == opposite_direction:
                    return {"valid": False, "errors": [f"Signal conflicts with existing {existing_direction} position"]}

        if self._coordinator:
            conflict_pass, conflicts = self._coordinator.check_signal_conflicts(strategy_name, signal_data)
            if not conflict_pass:
                resolved, adjusted = self._coordinator.resolve_conflict(strategy_name, signal_data, conflicts)
                if not resolved:
                    return {"valid": False, "errors": [f"Strategy conflict detected"]}
                return {"adjusted_signal": adjusted}

        return None
