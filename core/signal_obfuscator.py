"""
信号混淆器 (Signal Obfuscator)
==============================
生产级防针对量化数据模块 — 第一层防护

核心功能：
1. 信号延迟随机化 — 在信号生成到执行之间插入随机延迟，防止时序规律被识别
2. 信号权重抖动 — 对置信度/权重添加噪声，防止权重分布被反推
3. 虚拟信号注入 — 按概率生成假信号，增加外部观察者的信噪比
4. 信号类型旋转 — 偶尔将相似信号类型互换，防止信号类型频次被统计
5. 信号过期时间扰动 — 随机化信号TTL，防止生存周期被识别

设计原则：
- 所有混淆操作对实际交易执行的影响可控（最小化滑点成本）
- 安全信号（风控/止损）不参与混淆，保证执行确定性
- 混淆强度可根据市场状态自适应调整
"""

import random
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from core.signal_generator import TradingSignal, SignalType, SignalLevel, SignalSource


class ObfuscationMode(Enum):
    """混淆模式"""
    OFF = "off"               # 关闭混淆
    LIGHT = "light"           # 轻度混淆（低波动环境）
    STANDARD = "standard"     # 标准混淆（默认）
    AGGRESSIVE = "aggressive" # 激进混淆（检测到针对时）


@dataclass
class ObfuscationRecord:
    """混淆操作记录"""
    signal_id: str
    original_type: str
    obfuscated_type: str
    original_weight: float
    obfuscated_weight: float
    delay_ms: float
    timestamp: float
    mode: str


class SignalObfuscator:
    """
    生产级信号混淆器
    
    在信号流水线中插入混淆层，对信号进行随机化处理，
    隐藏真实交易意图，防止量化对手方通过信号模式进行针对。
    
    使用示例:
        obfuscator = SignalObfuscator(config)
        obfuscated_signals = obfuscator.obfuscate(signals, context)
    """
    
    # 安全信号类型（不参与混淆）
    SAFE_SIGNAL_TYPES = {
        SignalType.STOP_LOSS,
        SignalType.CLOSE_ALL,
    }
    
    # 可互换的信号类型对
    INTERCHANGEABLE_PAIRS = [
        (SignalType.OPEN_LONG, SignalType.ADD_POSITION),
        (SignalType.OPEN_SHORT, SignalType.ADD_POSITION),
        (SignalType.REDUCE_POSITION, SignalType.TAKE_PROFIT),
    ]
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        obf_cfg = config.get("anti_targeting", {}).get("signal_obfuscator", {}) if config else {}
        
        # ── 混淆模式 ──
        self._mode = ObfuscationMode(obf_cfg.get("mode", "standard"))
        self._enabled = obf_cfg.get("enabled", True)
        
        # ── 延迟随机化 ──
        self._delay_enabled = obf_cfg.get("delay_enabled", True)
        self._delay_min_ms = obf_cfg.get("delay_min_ms", 50)
        self._delay_max_ms = obf_cfg.get("delay_max_ms", 500)
        self._delay_distribution = obf_cfg.get("delay_distribution", "exponential")  # uniform, exponential, normal
        
        # ── 权重抖动 ──
        self._weight_jitter_enabled = obf_cfg.get("weight_jitter_enabled", True)
        self._weight_jitter_pct = obf_cfg.get("weight_jitter_pct", 0.05)  # ±5%
        
        # ── 虚拟信号注入 ──
        self._dummy_injection_enabled = obf_cfg.get("dummy_injection_enabled", True)
        self._dummy_probability = obf_cfg.get("dummy_probability", 0.03)  # 3%概率注入虚拟信号
        self._dummy_max_per_batch = obf_cfg.get("dummy_max_per_batch", 2)
        self._dummy_signals_injected = 0
        
        # ── 信号类型旋转 ──
        self._type_rotation_enabled = obf_cfg.get("type_rotation_enabled", True)
        self._rotation_probability = obf_cfg.get("rotation_probability", 0.08)  # 8%概率旋转
        
        # ── 信号过期时间扰动 ──
        self._expiry_jitter_enabled = obf_cfg.get("expiry_jitter_enabled", True)
        self._expiry_jitter_seconds = obf_cfg.get("expiry_jitter_seconds", 10)  # ±10秒
        
        # ── 自适应混淆 ──
        self._adaptive_enabled = obf_cfg.get("adaptive_enabled", True)
        self._pattern_detection_threshold = obf_cfg.get("pattern_detection_threshold", 0.7)
        self._adaptive_decay = obf_cfg.get("adaptive_decay", 0.95)
        
        # ── 状态追踪 ──
        self._lock = threading.RLock()
        self._obfuscation_history: deque = deque(maxlen=1000)
        self._signal_timing_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._dummy_signal_pool: List[TradingSignal] = []
        self._pattern_risk_score: float = 0.0  # 0-1，越高说明越可能被针对
        
        # ── 统计 ──
        self._total_obfuscated = 0
        self._total_delayed = 0
        self._total_jittered = 0
        self._total_rotated = 0
        self._total_dummies_injected = 0
        
        logger.info(
            f"SignalObfuscator initialized: mode={self._mode.value}, "
            f"delay={self._delay_min_ms}-{self._delay_max_ms}ms, "
            f"jitter={self._weight_jitter_pct:.1%}, "
            f"dummy_prob={self._dummy_probability:.1%}"
        )
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """热更新配置"""
        with self._lock:
            obf_cfg = new_config.get("anti_targeting", {}).get("signal_obfuscator", {})
            if not obf_cfg:
                return
            
            self._enabled = obf_cfg.get("enabled", self._enabled)
            if "mode" in obf_cfg:
                self._mode = ObfuscationMode(obf_cfg["mode"])
            
            self._delay_enabled = obf_cfg.get("delay_enabled", self._delay_enabled)
            self._delay_min_ms = obf_cfg.get("delay_min_ms", self._delay_min_ms)
            self._delay_max_ms = obf_cfg.get("delay_max_ms", self._delay_max_ms)
            
            self._weight_jitter_enabled = obf_cfg.get("weight_jitter_enabled", self._weight_jitter_enabled)
            self._weight_jitter_pct = obf_cfg.get("weight_jitter_pct", self._weight_jitter_pct)
            
            self._dummy_injection_enabled = obf_cfg.get("dummy_injection_enabled", self._dummy_injection_enabled)
            self._dummy_probability = obf_cfg.get("dummy_probability", self._dummy_probability)
            
            self._type_rotation_enabled = obf_cfg.get("type_rotation_enabled", self._type_rotation_enabled)
            self._rotation_probability = obf_cfg.get("rotation_probability", self._rotation_probability)
            
            self._adaptive_enabled = obf_cfg.get("adaptive_enabled", self._adaptive_enabled)
            
            logger.info(f"SignalObfuscator config updated: mode={self._mode.value}")
    
    def set_mode(self, mode: ObfuscationMode) -> None:
        """设置混淆模式"""
        self._mode = mode
        logger.info(f"SignalObfuscator mode set to: {mode.value}")
    
    def set_pattern_risk(self, score: float) -> None:
        """设置模式风险评分（由外部AntiPatternDetector提供）"""
        self._pattern_risk_score = max(0.0, min(1.0, score))
    
    def obfuscate(
        self,
        signals: List[TradingSignal],
        context: Dict[str, Any] = None,
    ) -> List[TradingSignal]:
        """
        对信号列表进行混淆处理
        
        Args:
            signals: 原始信号列表
            context: 市场上下文（用于自适应混淆）
        
        Returns:
            混淆后的信号列表（可能包含虚拟信号）
        """
        if not self._enabled or self._mode == ObfuscationMode.OFF:
            return signals
        
        if not signals:
            return signals
        
        with self._lock:
            obfuscated = []
            
            for signal in signals:
                # 安全信号不混淆
                if signal.signal_type in self.SAFE_SIGNAL_TYPES:
                    obfuscated.append(signal)
                    continue
                
                # 应用混淆
                modified = self._obfuscate_single(signal, context)
                if modified:
                    obfuscated.append(modified)
                    self._total_obfuscated += 1
            
            # 虚拟信号注入
            if self._dummy_injection_enabled and self._should_inject_dummy():
                dummies = self._generate_dummy_signals(signals, obfuscated)
                obfuscated.extend(dummies)
                self._total_dummies_injected += len(dummies)
            
            return obfuscated
    
    def _obfuscate_single(
        self,
        signal: TradingSignal,
        context: Dict[str, Any] = None,
    ) -> Optional[TradingSignal]:
        """对单个信号进行混淆"""
        modified = signal
        
        # 根据模式调整混淆强度
        intensity = self._get_intensity()
        
        # 1. 信号类型旋转
        if self._type_rotation_enabled and random.random() < self._rotation_probability * intensity:
            modified = self._rotate_signal_type(modified)
            if modified != signal:
                self._total_rotated += 1
        
        # 2. 权重抖动
        if self._weight_jitter_enabled:
            modified = self._jitter_weight(modified, intensity)
            if modified.weight != signal.weight:
                self._total_jittered += 1
        
        # 3. 延迟插入（记录到metadata，由执行层处理）
        if self._delay_enabled:
            delay_ms = self._generate_delay(intensity)
            modified.metadata["obfuscation_delay_ms"] = delay_ms
            self._total_delayed += 1
        
        # 4. 过期时间扰动
        if self._expiry_jitter_enabled and modified.expiry:
            jitter = random.uniform(-self._expiry_jitter_seconds, self._expiry_jitter_seconds)
            modified.expiry = modified.expiry + timedelta(seconds=jitter)
        
        # 记录混淆操作
        self._record_obfuscation(signal, modified)
        
        return modified
    
    def _get_intensity(self) -> float:
        """获取混淆强度因子（0.5-1.5）"""
        if self._mode == ObfuscationMode.LIGHT:
            base = 0.5
        elif self._mode == ObfuscationMode.AGGRESSIVE:
            base = 1.5
        else:
            base = 1.0
        
        # 自适应调整：检测到高风险时增强混淆
        if self._adaptive_enabled and self._pattern_risk_score > self._pattern_detection_threshold:
            boost = 1.0 + (self._pattern_risk_score - self._pattern_detection_threshold) * 2.0
            base = min(base * boost, 2.0)
        
        return base
    
    def _generate_delay(self, intensity: float) -> float:
        """生成随机延迟（毫秒）"""
        if self._delay_distribution == "exponential":
            # 指数分布：大部分延迟较短，偶尔较长
            scale = (self._delay_max_ms - self._delay_min_ms) / 3.0
            delay = random.expovariate(1.0 / max(scale, 1.0))
            delay = self._delay_min_ms + delay
        elif self._delay_distribution == "normal":
            # 正态分布：集中在中间
            mean = (self._delay_min_ms + self._delay_max_ms) / 2
            std = (self._delay_max_ms - self._delay_min_ms) / 4
            delay = random.normalvariate(mean, std)
        else:
            # 均匀分布
            delay = random.uniform(self._delay_min_ms, self._delay_max_ms)
        
        delay *= intensity
        return max(0, min(delay, self._delay_max_ms * 2))
    
    def _jitter_weight(self, signal: TradingSignal, intensity: float) -> TradingSignal:
        """对信号权重进行抖动"""
        jitter_range = self._weight_jitter_pct * intensity
        jitter = 1.0 + random.uniform(-jitter_range, jitter_range)
        
        original_weight = signal.weight
        signal.weight = min(1.0, max(0.1, original_weight * jitter))
        signal.metadata["original_weight"] = original_weight
        signal.metadata["weight_jitter"] = round(jitter - 1.0, 4)
        
        # 重新计算级别
        if signal.weight >= 0.8:
            signal.level = SignalLevel.STRONG
        elif signal.weight >= 0.5:
            signal.level = SignalLevel.MEDIUM
        else:
            signal.level = SignalLevel.WEAK
        
        return signal
    
    def _rotate_signal_type(self, signal: TradingSignal) -> TradingSignal:
        """旋转信号类型（在可互换对中切换）"""
        for pair in self.INTERCHANGEABLE_PAIRS:
            if signal.signal_type in pair:
                new_type = pair[0] if signal.signal_type == pair[1] else pair[1]
                signal.metadata["original_signal_type"] = signal.signal_type.value
                signal.signal_type = new_type
                signal.metadata["type_rotated"] = True
                break
        
        return signal
    
    def _should_inject_dummy(self) -> bool:
        """判断是否应该注入虚拟信号"""
        if self._dummy_signals_injected >= self._dummy_max_per_batch:
            return False
        
        # 根据模式调整概率
        intensity = self._get_intensity()
        adjusted_prob = self._dummy_probability * intensity
        
        return random.random() < adjusted_prob
    
    def _generate_dummy_signals(
        self,
        real_signals: List[TradingSignal],
        obfuscated: List[TradingSignal],
    ) -> List[TradingSignal]:
        """生成虚拟信号"""
        dummies = []
        symbols = list(set(s.symbol for s in real_signals))
        
        if not symbols:
            return dummies
        
        count = random.randint(1, min(self._dummy_max_per_batch, len(symbols)))
        
        for _ in range(count):
            symbol = random.choice(symbols)
            # 随机选择信号类型
            dummy_types = [
                SignalType.OPEN_LONG, SignalType.OPEN_SHORT,
                SignalType.ADD_POSITION, SignalType.REDUCE_POSITION,
            ]
            dummy_type = random.choice(dummy_types)
            
            # 使用随机权重
            dummy_weight = random.uniform(0.2, 0.6)
            level = SignalLevel.WEAK if dummy_weight < 0.5 else SignalLevel.MEDIUM
            
            dummy = TradingSignal(
                symbol=symbol,
                signal_type=dummy_type,
                source=SignalSource.MOMENTUM,
                level=level,
                weight=dummy_weight,
                price=0.0,  # 虚拟信号没有实际价格
                quantity=0.0,
                timestamp=datetime.now(),
                reason="[DUMMY] Virtual signal for obfuscation",
                metadata={
                    "is_dummy": True,
                    "obfuscation": True,
                    "quality_score": 0.0,
                },
                strategy_id="obfuscator_dummy",
            )
            dummies.append(dummy)
        
        self._dummy_signals_injected += len(dummies)
        return dummies
    
    def _record_obfuscation(self, original: TradingSignal, modified: TradingSignal) -> None:
        """记录混淆操作"""
        delay = modified.metadata.get("obfuscation_delay_ms", 0)
        self._obfuscation_history.append(ObfuscationRecord(
            signal_id=str(id(original)),
            original_type=original.signal_type.value,
            obfuscated_type=modified.signal_type.value,
            original_weight=original.weight,
            obfuscated_weight=modified.weight,
            delay_ms=delay,
            timestamp=time.time(),
            mode=self._mode.value,
        ))
    
    def is_dummy_signal(self, signal: TradingSignal) -> bool:
        """检查信号是否为虚拟信号"""
        return signal.metadata.get("is_dummy", False)
    
    def filter_dummy_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """过滤掉虚拟信号（在执行层使用）"""
        return [s for s in signals if not self.is_dummy_signal(s)]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取混淆器统计信息"""
        with self._lock:
            recent = list(self._obfuscation_history)[-100:]
            avg_delay = sum(r.delay_ms for r in recent) / max(len(recent), 1)
            
            return {
                "enabled": self._enabled,
                "mode": self._mode.value,
                "total_obfuscated": self._total_obfuscated,
                "total_delayed": self._total_delayed,
                "total_jittered": self._total_jittered,
                "total_rotated": self._total_rotated,
                "total_dummies_injected": self._total_dummies_injected,
                "avg_delay_ms": round(avg_delay, 1),
                "pattern_risk_score": round(self._pattern_risk_score, 4),
                "current_intensity": round(self._get_intensity(), 2),
                "delay_config": {
                    "min_ms": self._delay_min_ms,
                    "max_ms": self._delay_max_ms,
                    "distribution": self._delay_distribution,
                },
                "weight_jitter_pct": self._weight_jitter_pct,
                "dummy_probability": self._dummy_probability,
                "rotation_probability": self._rotation_probability,
            }
    
    def reset_stats(self) -> None:
        """重置统计计数器"""
        with self._lock:
            self._total_obfuscated = 0
            self._total_delayed = 0
            self._total_jittered = 0
            self._total_rotated = 0
            self._total_dummies_injected = 0
            self._dummy_signals_injected = 0
            self._obfuscation_history.clear()


__all__ = [
    "SignalObfuscator",
    "ObfuscationMode",
    "ObfuscationRecord",
]