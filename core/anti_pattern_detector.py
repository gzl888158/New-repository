"""
反模式检测器 (Anti-Pattern Detector)
=====================================
生产级防针对量化数据模块 — 第三层防护

核心功能：
1. 自相似性分析 — 检测自身交易模式是否可被预测
2. 时序熵监控 — 监控信号/订单时间间隔的熵值，低熵=可预测
3. 成交量分布分析 — 检测成交量是否呈现规律性（如固定大小）
4. 价位聚类检测 — 检测限价单价格是否形成可识别模式
5. 抢先交易检测 — 检测是否被市场参与者针对（滑点异常、价格反走）
6. 风险评分输出 — 综合评分驱动SignalObfuscator和OrderFingerprintMasker自适应

设计原则：
- 检测但不干预：检测到模式风险后通过评分输出给混淆层处理
- 多维度交叉验证：单一维度异常不触发升级，多维度共振才升级
- 滑动窗口统计：使用滚动窗口避免长期数据带来的噪声
"""

import time
import hashlib
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
import math


class RiskLevel(Enum):
    """模式风险等级"""
    LOW = "low"               # 低风险，正常交易
    ELEVATED = "elevated"     # 风险升高，增强混淆
    HIGH = "high"             # 高风险，激进混淆
    CRITICAL = "critical"     # 极高风险，暂停交易或切换策略


@dataclass
class SignalTimingRecord:
    """信号时序记录"""
    symbol: str
    signal_type: str
    source: str
    weight: float
    timestamp: float
    direction: str


@dataclass
class OrderPatternRecord:
    """订单模式记录"""
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: float
    is_split: bool = False


@dataclass
class PatternAlert:
    """模式告警"""
    alert_type: str  # "self_similarity", "timing_entropy", "volume_pattern", "price_clustering", "front_running"
    symbol: str
    severity: str  # "warning", "danger", "critical"
    score: float
    details: str
    timestamp: datetime


class AntiPatternDetector:
    """
    生产级反模式检测器
    
    监控自身交易行为，检测是否形成可被对手方利用的规律模式，
    并输出风险评分驱动混淆模块自适应调整。
    
    使用示例:
        detector = AntiPatternDetector(config)
        detector.record_signal(signal_record)
        detector.record_order(order_record)
        risk_score = detector.evaluate_risk()
        obfuscator.set_pattern_risk(risk_score)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        apd_cfg = config.get("anti_targeting", {}).get("anti_pattern_detector", {}) if config else {}
        
        # ── 基本配置 ──
        self._enabled = apd_cfg.get("enabled", True)
        self._check_interval_seconds = apd_cfg.get("check_interval_seconds", 60)
        self._last_check_time: float = 0
        
        # ── 滑动窗口大小 ──
        self._signal_window_size = apd_cfg.get("signal_window_size", 100)
        self._order_window_size = apd_cfg.get("order_window_size", 50)
        self._analysis_window_minutes = apd_cfg.get("analysis_window_minutes", 30)
        
        # ── 自相似性分析 ──
        self._self_similarity_enabled = apd_cfg.get("self_similarity_enabled", True)
        self._similarity_threshold = apd_cfg.get("similarity_threshold", 0.75)
        self._similarity_weight = apd_cfg.get("similarity_weight", 0.25)
        
        # ── 时序熵分析 ──
        self._timing_entropy_enabled = apd_cfg.get("timing_entropy_enabled", True)
        self._entropy_threshold_low = apd_cfg.get("entropy_threshold_low", 0.5)  # 低于此值=可能可预测
        self._entropy_threshold_warn = apd_cfg.get("entropy_threshold_warn", 0.3)
        self._timing_entropy_weight = apd_cfg.get("timing_entropy_weight", 0.20)
        
        # ── 成交量分布分析 ──
        self._volume_pattern_enabled = apd_cfg.get("volume_pattern_enabled", True)
        self._volume_cv_threshold = apd_cfg.get("volume_cv_threshold", 0.15)  # 变异系数<15%=规律性
        self._volume_pattern_weight = apd_cfg.get("volume_pattern_weight", 0.20)
        
        # ── 价位聚类检测 ──
        self._price_clustering_enabled = apd_cfg.get("price_clustering_enabled", True)
        self._price_cluster_radius_pct = apd_cfg.get("price_cluster_radius_pct", 0.002)  # 0.2%内算聚类
        self._price_cluster_ratio_threshold = apd_cfg.get("price_cluster_ratio_threshold", 0.6)  # 60%订单聚类=可识别
        self._price_clustering_weight = apd_cfg.get("price_clustering_weight", 0.20)
        
        # ── 抢先交易检测 ──
        self._front_running_enabled = apd_cfg.get("front_running_enabled", True)
        self._front_running_slippage_threshold = apd_cfg.get("front_running_slippage_threshold", 0.003)  # 0.3%滑点
        self._front_running_ratio_threshold = apd_cfg.get("front_running_ratio_threshold", 0.3)  # 30%订单被针对
        self._front_running_weight = apd_cfg.get("front_running_weight", 0.15)
        
        # ── 状态 ──
        self._lock = threading.RLock()
        self._signal_records: deque = deque(maxlen=self._signal_window_size)
        self._order_records: deque = deque(maxlen=self._order_window_size)
        self._slippage_records: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        
        # ── 风险评分 ──
        self._current_risk_score: float = 0.0
        self._risk_score_history: deque = deque(maxlen=30)  # 30个窗口的历史评分
        self._current_risk_level = RiskLevel.LOW
        
        # ── 告警 ──
        self._alerts: deque = deque(maxlen=50)
        self._alert_callbacks: List[callable] = []
        
        # ── 统计 ──
        self._total_checks = 0
        self._total_alerts = 0
        
        logger.info(
            f"AntiPatternDetector initialized: "
            f"signal_window={self._signal_window_size}, "
            f"order_window={self._order_window_size}, "
            f"check_interval={self._check_interval_seconds}s"
        )
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """热更新配置"""
        with self._lock:
            apd_cfg = new_config.get("anti_targeting", {}).get("anti_pattern_detector", {})
            if not apd_cfg:
                return
            
            self._enabled = apd_cfg.get("enabled", self._enabled)
            self._check_interval_seconds = apd_cfg.get("check_interval_seconds", self._check_interval_seconds)
            
            self._self_similarity_enabled = apd_cfg.get("self_similarity_enabled", self._self_similarity_enabled)
            self._timing_entropy_enabled = apd_cfg.get("timing_entropy_enabled", self._timing_entropy_enabled)
            self._volume_pattern_enabled = apd_cfg.get("volume_pattern_enabled", self._volume_pattern_enabled)
            self._price_clustering_enabled = apd_cfg.get("price_clustering_enabled", self._price_clustering_enabled)
            self._front_running_enabled = apd_cfg.get("front_running_enabled", self._front_running_enabled)
            
            logger.info("AntiPatternDetector config updated")
    
    def register_alert_callback(self, callback: callable) -> None:
        """注册告警回调"""
        self._alert_callbacks.append(callback)
    
    # ═══════════════════════════════════════════════════════════════
    # 数据记录
    # ═══════════════════════════════════════════════════════════════
    
    def record_signal(
        self,
        symbol: str,
        signal_type: str,
        source: str,
        weight: float,
        direction: str,
    ) -> None:
        """记录信号"""
        if not self._enabled:
            return
        
        with self._lock:
            self._signal_records.append(SignalTimingRecord(
                symbol=symbol,
                signal_type=signal_type,
                source=source,
                weight=weight,
                timestamp=time.time(),
                direction=direction,
            ))
    
    def record_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        is_split: bool = False,
    ) -> None:
        """记录订单"""
        if not self._enabled:
            return
        
        with self._lock:
            self._order_records.append(OrderPatternRecord(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                timestamp=time.time(),
                is_split=is_split,
            ))
    
    def record_slippage(
        self,
        symbol: str,
        expected_price: float,
        actual_price: float,
        side: str,
    ) -> None:
        """记录滑点（用于抢先交易检测）"""
        if not self._enabled or expected_price <= 0:
            return
        
        slippage = (actual_price - expected_price) / expected_price
        if side == "buy":
            slippage = slippage  # 买单正向滑点=价格更高
        else:
            slippage = -slippage  # 卖单正向滑点=价格更低
        
        with self._lock:
            self._slippage_records[symbol].append({
                "slippage": slippage,
                "timestamp": time.time(),
                "expected": expected_price,
                "actual": actual_price,
            })
    
    # ═══════════════════════════════════════════════════════════════
    # 风险评分
    # ═══════════════════════════════════════════════════════════════
    
    def evaluate_risk(self, force: bool = False) -> float:
        """
        综合评估模式风险
        
        Returns:
            风险评分 0.0-1.0，越高越可能被针对
        """
        if not self._enabled:
            return 0.0
        
        now = time.time()
        if not force and now - self._last_check_time < self._check_interval_seconds:
            return self._current_risk_score
        
        with self._lock:
            self._last_check_time = now
            self._total_checks += 1
            
            scores = {}
            alerts = []
            
            # 1. 自相似性分析
            if self._self_similarity_enabled:
                sim_score = self._analyze_self_similarity()
                scores["self_similarity"] = sim_score
                if sim_score > 0.6:
                    alerts.append(("self_similarity", sim_score))
            
            # 2. 时序熵分析
            if self._timing_entropy_enabled:
                entropy_score = self._analyze_timing_entropy()
                scores["timing_entropy"] = entropy_score
                if entropy_score > 0.6:
                    alerts.append(("timing_entropy", entropy_score))
            
            # 3. 成交量分布分析
            if self._volume_pattern_enabled:
                volume_score = self._analyze_volume_pattern()
                scores["volume_pattern"] = volume_score
                if volume_score > 0.6:
                    alerts.append(("volume_pattern", volume_score))
            
            # 4. 价位聚类检测
            if self._price_clustering_enabled:
                price_score = self._analyze_price_clustering()
                scores["price_clustering"] = price_score
                if price_score > 0.6:
                    alerts.append(("price_clustering", price_score))
            
            # 5. 抢先交易检测
            if self._front_running_enabled:
                front_score = self._analyze_front_running()
                scores["front_running"] = front_score
                if front_score > 0.6:
                    alerts.append(("front_running", front_score))
            
            # 加权综合评分
            weights = {
                "self_similarity": self._similarity_weight,
                "timing_entropy": self._timing_entropy_weight,
                "volume_pattern": self._volume_pattern_weight,
                "price_clustering": self._price_clustering_weight,
                "front_running": self._front_running_weight,
            }
            
            total_weight = sum(weights.values())
            if total_weight > 0:
                self._current_risk_score = sum(
                    scores.get(k, 0) * weights.get(k, 0) for k in weights
                ) / total_weight
            else:
                self._current_risk_score = 0.0
            
            # 多维度共振加成
            high_dimensions = sum(1 for s in scores.values() if s > 0.5)
            if high_dimensions >= 3:
                self._current_risk_score = min(1.0, self._current_risk_score * 1.3)
            elif high_dimensions >= 2:
                self._current_risk_score = min(1.0, self._current_risk_score * 1.1)
            
            # 更新风险等级
            self._risk_score_history.append(self._current_risk_score)
            self._current_risk_level = self._score_to_level(self._current_risk_score)
            
            # 生成告警
            if high_dimensions >= 2 or self._current_risk_score > 0.5:
                self._generate_alerts(alerts, scores)
            
            return self._current_risk_score
    
    def _score_to_level(self, score: float) -> RiskLevel:
        """评分转风险等级"""
        if score >= 0.8:
            return RiskLevel.CRITICAL
        elif score >= 0.6:
            return RiskLevel.HIGH
        elif score >= 0.4:
            return RiskLevel.ELEVATED
        return RiskLevel.LOW
    
    # ═══════════════════════════════════════════════════════════════
    # 维度分析
    # ═══════════════════════════════════════════════════════════════
    
    def _analyze_self_similarity(self) -> float:
        """分析信号自相似性
        
        检测信号之间的相似度，如果近期信号高度相似，说明模式可预测。
        使用信号特征向量（类型、方向、权重区间）的哈希比较。
        """
        records = list(self._signal_records)
        if len(records) < 10:
            return 0.0
        
        # 构建特征向量
        features = []
        for r in records[-30:]:  # 取最近30个
            # 简化的特征向量：类型+方向+权重区间
            weight_bucket = int(r.weight * 10)  # 0-10的桶
            features.append(f"{r.signal_type}:{r.direction}:{weight_bucket}")
        
        if len(features) < 2:
            return 0.0
        
        # 计算重复率
        unique_count = len(set(features))
        repetition_rate = 1.0 - (unique_count / len(features))
        
        # 归一化到0-1
        if repetition_rate > self._similarity_threshold:
            return min(1.0, (repetition_rate - self._similarity_threshold) / (1.0 - self._similarity_threshold))
        
        return 0.0
    
    def _analyze_timing_entropy(self) -> float:
        """分析时序熵
        
        计算信号时间间隔的信息熵。
        低熵=时间间隔规律=可预测。
        """
        records = list(self._signal_records)
        if len(records) < 10:
            return 0.0
        
        # 计算时间间隔
        intervals = []
        for i in range(1, len(records)):
            interval = records[i].timestamp - records[i - 1].timestamp
            if interval > 0:
                intervals.append(interval)
        
        if len(intervals) < 5:
            return 0.0
        
        # 计算熵
        entropy = self._compute_entropy(intervals, bins=10)
        
        # 归一化：低熵=高风险
        max_entropy = math.log2(10)  # 10个桶的最大熵
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        if normalized_entropy < self._entropy_threshold_warn:
            return 1.0
        elif normalized_entropy < self._entropy_threshold_low:
            return 0.7
        elif normalized_entropy < 0.7:
            return 0.3
        
        return 0.0
    
    def _compute_entropy(self, values: List[float], bins: int = 10) -> float:
        """计算数据分布的熵"""
        if not values:
            return 0.0
        
        min_val = min(values)
        max_val = max(values)
        if max_val == min_val:
            return 0.0
        
        bin_width = (max_val - min_val) / bins
        counts = [0] * bins
        
        for v in values:
            idx = min(bins - 1, int((v - min_val) / bin_width))
            counts[idx] += 1
        
        total = len(values)
        entropy = 0.0
        for c in counts:
            if c > 0:
                p = c / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _analyze_volume_pattern(self) -> float:
        """分析成交量分布模式
        
        计算订单量的变异系数(CV)。CV越低=量越规律=可识别。
        """
        records = list(self._order_records)
        if len(records) < 10:
            return 0.0
        
        volumes = [r.quantity for r in records if r.quantity > 0]
        if len(volumes) < 5:
            return 0.0
        
        mean = sum(volumes) / len(volumes)
        if mean == 0:
            return 0.0
        
        variance = sum((v - mean) ** 2 for v in volumes) / len(volumes)
        std = math.sqrt(variance)
        cv = std / mean  # 变异系数
        
        if cv < self._volume_cv_threshold * 0.5:
            return 1.0
        elif cv < self._volume_cv_threshold:
            return 0.7
        elif cv < self._volume_cv_threshold * 2:
            return 0.3
        
        return 0.0
    
    def _analyze_price_clustering(self) -> float:
        """分析价位聚类
        
        检测限价单价格是否在窄区间内聚类，聚类比例高=可识别。
        """
        records = list(self._order_records)
        if len(records) < 10:
            return 0.0
        
        # 按币种分组
        by_symbol: Dict[str, List[float]] = defaultdict(list)
        for r in records:
            if r.price > 0:
                by_symbol[r.symbol].append(r.price)
        
        max_clustering_score = 0.0
        
        for symbol, prices in by_symbol.items():
            if len(prices) < 5:
                continue
            
            # 计算聚类比例
            clustered = 0
            for i, p1 in enumerate(prices):
                for j, p2 in enumerate(prices):
                    if j <= i:
                        continue
                    if p1 > 0 and abs(p1 - p2) / p1 < self._price_cluster_radius_pct:
                        clustered += 1
                        break
            
            cluster_ratio = clustered / len(prices)
            
            if cluster_ratio > self._price_cluster_ratio_threshold:
                score = min(1.0, (cluster_ratio - self._price_cluster_ratio_threshold) / 
                           (1.0 - self._price_cluster_ratio_threshold))
                max_clustering_score = max(max_clustering_score, score)
        
        return max_clustering_score
    
    def _analyze_front_running(self) -> float:
        """分析抢先交易
        
        检测滑点是否异常，滑点大且频繁=可能被抢先交易。
        """
        all_slippages = []
        for symbol, records in self._slippage_records.items():
            for r in records:
                all_slippages.append(r["slippage"])
        
        if len(all_slippages) < 5:
            return 0.0
        
        # 计算异常滑点比例
        abnormal = sum(1 for s in all_slippages if abs(s) > self._front_running_slippage_threshold)
        abnormal_ratio = abnormal / len(all_slippages)
        
        if abnormal_ratio > self._front_running_ratio_threshold * 2:
            return 1.0
        elif abnormal_ratio > self._front_running_ratio_threshold:
            return 0.7
        elif abnormal_ratio > self._front_running_ratio_threshold * 0.5:
            return 0.3
        
        return 0.0
    
    def _generate_alerts(self, alerts: List[Tuple[str, float]], scores: Dict[str, float]) -> None:
        """生成告警"""
        for alert_type, score in alerts:
            severity = "warning" if score < 0.7 else ("danger" if score < 0.85 else "critical")
            
            alert = PatternAlert(
                alert_type=alert_type,
                symbol="*",
                severity=severity,
                score=score,
                details=f"Pattern detected: {alert_type} (score={score:.3f})",
                timestamp=datetime.now(),
            )
            self._alerts.append(alert)
            self._total_alerts += 1
            
            # 触发回调
            for cb in self._alert_callbacks:
                try:
                    cb(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
        
        if alerts:
            logger.warning(
                f"AntiPatternDetector: {len(alerts)} alerts, "
                f"risk_score={self._current_risk_score:.3f}, "
                f"level={self._current_risk_level.value}"
            )
    
    # ═══════════════════════════════════════════════════════════════
    # 查询接口
    # ═══════════════════════════════════════════════════════════════
    
    def get_risk_score(self) -> float:
        """获取当前风险评分"""
        return self._current_risk_score
    
    def get_risk_level(self) -> RiskLevel:
        """获取当前风险等级"""
        return self._current_risk_level
    
    def get_risk_trend(self) -> str:
        """获取风险趋势 (rising/falling/stable)"""
        if len(self._risk_score_history) < 5:
            return "stable"
        
        recent = list(self._risk_score_history)[-5:]
        if recent[-1] > recent[0] * 1.2:
            return "rising"
        elif recent[-1] < recent[0] * 0.8:
            return "falling"
        return "stable"
    
    def get_alerts(self, limit: int = 20) -> List[PatternAlert]:
        """获取最近告警"""
        return list(self._alerts)[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取检测器统计"""
        with self._lock:
            return {
                "enabled": self._enabled,
                "total_checks": self._total_checks,
                "total_alerts": self._total_alerts,
                "current_risk_score": round(self._current_risk_score, 4),
                "current_risk_level": self._current_risk_level.value,
                "risk_trend": self.get_risk_trend(),
                "signal_records_count": len(self._signal_records),
                "order_records_count": len(self._order_records),
                "slippage_records_count": sum(len(v) for v in self._slippage_records.values()),
                "risk_score_history": [round(s, 4) for s in list(self._risk_score_history)[-10:]],
                "recent_alerts": len(self._alerts),
                "check_interval_seconds": self._check_interval_seconds,
            }
    
    def reset_stats(self) -> None:
        """重置统计"""
        with self._lock:
            self._signal_records.clear()
            self._order_records.clear()
            self._slippage_records.clear()
            self._alerts.clear()
            self._risk_score_history.clear()
            self._current_risk_score = 0.0
            self._current_risk_level = RiskLevel.LOW
            self._total_checks = 0
            self._total_alerts = 0


__all__ = [
    "AntiPatternDetector",
    "RiskLevel",
    "PatternAlert",
    "SignalTimingRecord",
    "OrderPatternRecord",
]