"""
信号标准化器，将不同策略与格式的原始信号统一转换为标准信号结构。
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from loguru import logger
import uuid


class SignalType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    CLOSE = "close"
    SCALPING = "scalping"
    GRID = "grid"
    TREND = "trend"
    ARBITRAGE = "arbitrage"
    SPOT_GRID = "spot_grid"
    SPOT_MARTINGALE = "spot_martingale"


class SignalDirection(Enum):
    LONG = "long"
    SHORT = "short"
    BUY = "buy"
    SELL = "sell"


@dataclass
class StandardSignal:
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    strategy_name: str = ""
    signal_type: str = ""
    direction: str = ""
    price: float = 0.0
    quantity: float = 0.0
    leverage: int = 1
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    quality_breakdown: Dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""  # 全链路 traceID：信号生成阶段生成，贯穿裁决/订单/记账


class SignalNormalizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._direction_map = {
            "buy": "long",
            "sell": "short",
            "long": "long",
            "short": "short",
        }
        self._strategy_signal_types = {
            "grid": "grid",
            "trend": "trend",
            "scalping": "scalping",
            "arbitrage": "arbitrage",
            "spot_grid": "spot_grid",
            "spot_martingale": "spot_martingale",
        }

    @staticmethod
    def _safe_str(value: Any) -> str:
        """安全转字符串，容忍 None/枚举/非字符串"""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if hasattr(value, 'value'):  # 枚举类型
            return str(value.value)
        return str(value)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """安全转浮点数，非法值回退默认值"""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 1) -> int:
        """安全转整数，非法值回退默认值"""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def normalize(self, raw_signal: Any) -> StandardSignal:
        if hasattr(raw_signal, '__dataclass_fields__'):
            return self._normalize_dataclass(raw_signal)
        elif isinstance(raw_signal, dict):
            return self._normalize_dict(raw_signal)
        else:
            logger.error(f"Unsupported signal type: {type(raw_signal)}")
            return StandardSignal()

    def _normalize_dataclass(self, signal_obj) -> StandardSignal:
        direction = self._safe_str(getattr(signal_obj, 'direction', '')).lower()
        normalized_direction = self._direction_map.get(direction, direction)

        strategy_name = self._safe_str(getattr(signal_obj, 'strategy_name', ''))
        signal_type = self._safe_str(getattr(signal_obj, 'signal_type', ''))
        if not signal_type and strategy_name:
            signal_type = self._strategy_signal_types.get(strategy_name, signal_type)

        return StandardSignal(
            signal_id=str(uuid.uuid4()),
            symbol=self._safe_str(getattr(signal_obj, 'symbol', '')),
            strategy_name=strategy_name,
            signal_type=signal_type,
            direction=normalized_direction,
            price=self._safe_float(getattr(signal_obj, 'price', 0.0)),
            quantity=self._safe_float(getattr(signal_obj, 'quantity', 0.0)),
            leverage=self._safe_int(getattr(signal_obj, 'leverage', 1)),
            stop_loss=self._safe_float(getattr(signal_obj, 'stop_loss', 0.0)) if getattr(signal_obj, 'stop_loss', None) else None,
            take_profit=self._safe_float(getattr(signal_obj, 'take_profit', 0.0)) if getattr(signal_obj, 'take_profit', None) else None,
            confidence=self._safe_float(getattr(signal_obj, 'confidence', 0.5), 0.5),
            timestamp=getattr(signal_obj, 'timestamp', datetime.now()),
            source=strategy_name,
        )

    def _normalize_dict(self, signal_dict: Dict[str, Any]) -> StandardSignal:
        direction = self._safe_str(signal_dict.get('direction', '')).lower()
        normalized_direction = self._direction_map.get(direction, direction)

        strategy_name = self._safe_str(signal_dict.get('strategy_name', ''))
        signal_type = self._safe_str(signal_dict.get('signal_type', ''))
        if not signal_type and strategy_name:
            signal_type = self._strategy_signal_types.get(strategy_name, signal_type)

        timestamp = signal_dict.get('timestamp')
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except Exception:
                timestamp = datetime.now()
        elif not isinstance(timestamp, datetime):
            timestamp = datetime.now()

        return StandardSignal(
            signal_id=signal_dict.get('signal_id', str(uuid.uuid4())),
            symbol=self._safe_str(signal_dict.get('symbol', '')),
            strategy_name=strategy_name,
            signal_type=signal_type,
            direction=normalized_direction,
            price=self._safe_float(signal_dict.get('price', 0.0)),
            quantity=self._safe_float(signal_dict.get('quantity', 0.0)),
            leverage=self._safe_int(signal_dict.get('leverage', 1)),
            stop_loss=self._safe_float(signal_dict.get('stop_loss', 0.0)) if signal_dict.get('stop_loss') is not None else None,
            take_profit=self._safe_float(signal_dict.get('take_profit', 0.0)) if signal_dict.get('take_profit') is not None else None,
            # P0: confidence=0是合法值（完全不确定），or会错误覆盖为0.5
            confidence=self._safe_float(signal_dict.get('confidence', 0.5), 0.5),
            timestamp=timestamp,
            source=signal_dict.get('source', strategy_name),
            metadata=signal_dict.get('metadata', {}),
            quality_score=self._safe_float(signal_dict.get('quality_score', 0.0)),
            quality_breakdown=signal_dict.get('quality_breakdown', {}),
            trace_id=signal_dict.get('trace_id', ''),
        )

    def validate(self, signal: StandardSignal) -> tuple:
        errors = []
        warnings = []

        if not signal.symbol:
            errors.append("symbol is required")
        elif not signal.symbol.endswith("-SWAP") and not signal.symbol.endswith("-USDT"):
            warnings.append(f"Unsupported symbol format: {signal.symbol}")

        if not signal.strategy_name:
            errors.append("strategy_name is required")

        if signal.direction not in ("long", "short"):
            errors.append(f"Invalid direction: {signal.direction}")

        if signal.price <= 0:
            errors.append("price must be positive")

        if signal.quantity <= 0:
            errors.append("quantity must be positive")

        if signal.leverage < 1:
            errors.append("leverage must be >= 1")

        if signal.confidence < 0 or signal.confidence > 1:
            errors.append("confidence must be between 0 and 1")

        if signal.stop_loss and signal.take_profit:
            if signal.direction == "long":
                if signal.stop_loss >= signal.price:
                    warnings.append("stop_loss should be below entry price for long")
                if signal.take_profit <= signal.price:
                    warnings.append("take_profit should be above entry price for long")
            else:
                if signal.stop_loss <= signal.price:
                    warnings.append("stop_loss should be above entry price for short")
                if signal.take_profit >= signal.price:
                    warnings.append("take_profit should be below entry price for short")

        is_valid = len(errors) == 0
        return is_valid, errors, warnings

    def to_dict(self, signal: StandardSignal) -> Dict[str, Any]:
        return {
            "signal_id": signal.signal_id,
            "symbol": signal.symbol,
            "strategy_name": signal.strategy_name,
            "signal_type": signal.signal_type,
            "direction": signal.direction,
            "price": signal.price,
            "quantity": signal.quantity,
            "leverage": signal.leverage,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "confidence": signal.confidence,
            "timestamp": signal.timestamp.isoformat(),
            "source": signal.source,
            "metadata": signal.metadata,
            "quality_score": signal.quality_score,
            "quality_breakdown": signal.quality_breakdown,
            "trace_id": signal.trace_id,
        }

    def enrich_signal(self, signal: StandardSignal, market_data: Dict[str, Any] = None) -> StandardSignal:
        """增强信号市场上下文"""
        if market_data:
            signal.metadata["market_data"] = market_data
            current_price = market_data.get("price")
            if current_price and signal.price != current_price:
                signal.metadata["price_deviation"] = abs(signal.price - current_price) / current_price

            # 波动率上下文
            atr = market_data.get("atr")
            if atr and current_price:
                signal.metadata["atr"] = float(atr)
                signal.metadata["atr_ratio"] = float(atr) / float(current_price)

            # 布林带上下文
            bb_upper = market_data.get("bb_upper")
            bb_lower = market_data.get("bb_lower")
            bb_middle = market_data.get("bb_middle")
            if bb_upper and bb_lower:
                signal.metadata["bb_upper"] = float(bb_upper)
                signal.metadata["bb_lower"] = float(bb_lower)
                signal.metadata["bb_middle"] = float(bb_middle) if bb_middle else (float(bb_upper) + float(bb_lower)) / 2
                band_width = float(bb_upper) - float(bb_lower)
                if band_width > 0 and current_price:
                    signal.metadata["bb_position"] = (float(current_price) - float(bb_lower)) / band_width

            # 成交量上下文
            volume = market_data.get("volume")
            avg_volume = market_data.get("avg_volume")
            if volume:
                signal.metadata["volume"] = float(volume)
            if avg_volume:
                signal.metadata["avg_volume"] = float(avg_volume)
                if volume and float(avg_volume) > 0:
                    signal.metadata["volume_ratio"] = float(volume) / float(avg_volume)

            # RSI 上下文
            rsi = market_data.get("rsi")
            if rsi is not None:
                signal.metadata["rsi"] = float(rsi)

            # MACD 上下文
            macd_line = market_data.get("macd_line")
            signal_line = market_data.get("signal_line")
            if macd_line is not None:
                signal.metadata["macd_line"] = float(macd_line)
            if signal_line is not None:
                signal.metadata["signal_line"] = float(signal_line)

            # 资金费率上下文
            funding_rate = market_data.get("funding_rate")
            if funding_rate is not None:
                signal.metadata["funding_rate"] = float(funding_rate)

            # 买卖价差上下文
            bid = market_data.get("bid")
            ask = market_data.get("ask")
            if bid and ask:
                signal.metadata["bid"] = float(bid)
                signal.metadata["ask"] = float(ask)
                signal.metadata["spread"] = float(ask) - float(bid)
                if float(bid) > 0:
                    signal.metadata["spread_pct"] = (float(ask) - float(bid)) / float(bid)

        return signal

    def generate_fingerprint(self, signal: StandardSignal) -> str:
        """生成信号指纹用于去重"""
        import hashlib
        fingerprint_data = f"{signal.symbol}:{signal.strategy_name}:{signal.direction}:{signal.signal_type}:{round(signal.price, 6)}"
        return hashlib.md5(fingerprint_data.encode()).hexdigest()[:16]