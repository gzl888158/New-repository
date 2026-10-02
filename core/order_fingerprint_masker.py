"""
订单指纹掩盖器 (Order Fingerprint Masker)
==========================================
生产级防针对量化数据模块 — 第二层防护

核心功能：
1. 数量随机化 — 订单量±3-8%随机抖动，消除固定仓位特征
2. 价格随机化 — 限价单价格±0.01-0.05%偏移，消除固定价位特征
3. 订单拆分 — 大单自动拆分为随机数量的小单，隐藏真实意图
4. 时间切片 — 拆分订单执行间隔随机化，消除时序特征
5. 订单类型轮换 — 限价单/市价单按比例交替使用
6. 精度适配 — 自动适配各币种的最小交易精度

设计原则：
- 所有操作保证订单最终可成交（不因混淆导致滑点过大）
- 拆分后的总数量等于原始数量（不改变仓位敞口）
- 风控单（止损/紧急平仓）不参与拆分，保证执行速度
"""

import random
import math
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger


class MaskingMode(Enum):
    """掩盖模式"""
    OFF = "off"
    LIGHT = "light"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"


class OrderType(Enum):
    """订单类型"""
    LIMIT = "limit"
    MARKET = "market"
    POST_ONLY = "post_only"
    FOK = "fok"
    IOC = "ioc"


@dataclass
class MaskedOrder:
    """掩盖后的订单"""
    symbol: str
    side: str  # buy/sell
    order_type: str
    price: float
    quantity: float
    original_quantity: float
    original_price: float
    is_split: bool = False
    split_index: int = 0
    split_total: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MaskingRecord:
    """掩盖操作记录"""
    symbol: str
    original_qty: float
    masked_qty: float
    original_price: float
    masked_price: float
    split_count: int
    timestamp: float


class OrderFingerprintMasker:
    """
    生产级订单指纹掩盖器
    
    在订单执行层对订单进行随机化处理，隐藏订单的量化特征，
    防止市场参与者通过订单模式识别策略意图。
    
    使用示例:
        masker = OrderFingerprintMasker(config)
        masked_orders = masker.mask_order(order_params)
    """
    
    # 币种精度配置（最小数量和价格精度）
    SYMBOL_PRECISION = {
        "BTC-USDT-SWAP": {"qty_step": 0.001, "price_step": 0.1, "min_qty": 0.001},
        "ETH-USDT-SWAP": {"qty_step": 0.01, "price_step": 0.01, "min_qty": 0.01},
        "SOL-USDT-SWAP": {"qty_step": 0.1, "price_step": 0.01, "min_qty": 0.1},
        "XRP-USDT-SWAP": {"qty_step": 1.0, "price_step": 0.0001, "min_qty": 1.0},
        "DOGE-USDT-SWAP": {"qty_step": 1.0, "price_step": 0.00001, "min_qty": 1.0},
        "ADA-USDT-SWAP": {"qty_step": 0.1, "price_step": 0.0001, "min_qty": 0.1},
        "AVAX-USDT-SWAP": {"qty_step": 0.01, "price_step": 0.001, "min_qty": 0.01},
        "DOT-USDT-SWAP": {"qty_step": 0.1, "price_step": 0.001, "min_qty": 0.1},
        "LINK-USDT-SWAP": {"qty_step": 0.01, "price_step": 0.001, "min_qty": 0.01},
        "POL-USDT-SWAP": {"qty_step": 0.1, "price_step": 0.0001, "min_qty": 0.1},
        "UNI-USDT-SWAP": {"qty_step": 0.01, "price_step": 0.001, "min_qty": 0.01},
        "ATOM-USDT-SWAP": {"qty_step": 0.01, "price_step": 0.001, "min_qty": 0.01},
    }
    
    DEFAULT_PRECISION = {"qty_step": 0.01, "price_step": 0.01, "min_qty": 0.01}
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        msk_cfg = config.get("anti_targeting", {}).get("order_fingerprint_masker", {}) if config else {}
        
        # ── 掩盖模式 ──
        self._mode = MaskingMode(msk_cfg.get("mode", "standard"))
        self._enabled = msk_cfg.get("enabled", True)
        
        # ── 数量随机化 ──
        self._quantity_jitter_enabled = msk_cfg.get("quantity_jitter_enabled", True)
        self._quantity_jitter_min_pct = msk_cfg.get("quantity_jitter_min_pct", 0.03)  # ±3%
        self._quantity_jitter_max_pct = msk_cfg.get("quantity_jitter_max_pct", 0.08)  # ±8%
        
        # ── 价格随机化 ──
        self._price_jitter_enabled = msk_cfg.get("price_jitter_enabled", True)
        self._price_jitter_pct = msk_cfg.get("price_jitter_pct", 0.0005)  # ±0.05%
        
        # ── 订单拆分 ──
        self._split_enabled = msk_cfg.get("split_enabled", True)
        self._split_min_notional = msk_cfg.get("split_min_notional", 500.0)  # 超过500 USDT自动拆分
        self._split_min_parts = msk_cfg.get("split_min_parts", 2)
        self._split_max_parts = msk_cfg.get("split_max_parts", 5)
        self._split_size_ratio_min = msk_cfg.get("split_size_ratio_min", 0.15)  # 最小子单占比15%
        self._split_size_ratio_max = msk_cfg.get("split_size_ratio_max", 0.6)   # 最大子单占比60%
        
        # ── 时间切片 ──
        self._time_slice_enabled = msk_cfg.get("time_slice_enabled", True)
        self._time_slice_min_ms = msk_cfg.get("time_slice_min_ms", 100)
        self._time_slice_max_ms = msk_cfg.get("time_slice_max_ms", 2000)
        self._time_slice_distribution = msk_cfg.get("time_slice_distribution", "exponential")
        
        # ── 订单类型轮换 ──
        self._type_rotation_enabled = msk_cfg.get("type_rotation_enabled", True)
        self._limit_order_ratio = msk_cfg.get("limit_order_ratio", 0.7)  # 70%限价单
        self._post_only_ratio = msk_cfg.get("post_only_ratio", 0.15)     # 15%只挂单
        
        # ── 状态追踪 ──
        self._lock = threading.RLock()
        self._masking_history: deque = deque(maxlen=500)
        self._order_type_counter: Dict[str, int] = defaultdict(int)  # 订单类型轮换计数
        
        # ── 统计 ──
        self._total_masked = 0
        self._total_split = 0
        self._split_orders_created = 0
        
        logger.info(
            f"OrderFingerprintMasker initialized: mode={self._mode.value}, "
            f"qty_jitter={self._quantity_jitter_min_pct:.1%}-{self._quantity_jitter_max_pct:.1%}, "
            f"split_min={self._split_min_notional}USDT, "
            f"split_parts={self._split_min_parts}-{self._split_max_parts}"
        )

    @property
    def enabled(self) -> bool:
        return self._enabled and self._mode != MaskingMode.OFF

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """热更新配置"""
        with self._lock:
            msk_cfg = new_config.get("anti_targeting", {}).get("order_fingerprint_masker", {})
            if not msk_cfg:
                return
            
            self._enabled = msk_cfg.get("enabled", self._enabled)
            if "mode" in msk_cfg:
                self._mode = MaskingMode(msk_cfg["mode"])
            
            self._quantity_jitter_enabled = msk_cfg.get("quantity_jitter_enabled", self._quantity_jitter_enabled)
            self._quantity_jitter_min_pct = msk_cfg.get("quantity_jitter_min_pct", self._quantity_jitter_min_pct)
            self._quantity_jitter_max_pct = msk_cfg.get("quantity_jitter_max_pct", self._quantity_jitter_max_pct)
            
            self._price_jitter_enabled = msk_cfg.get("price_jitter_enabled", self._price_jitter_enabled)
            self._price_jitter_pct = msk_cfg.get("price_jitter_pct", self._price_jitter_pct)
            
            self._split_enabled = msk_cfg.get("split_enabled", self._split_enabled)
            self._split_min_notional = msk_cfg.get("split_min_notional", self._split_min_notional)
            self._split_min_parts = msk_cfg.get("split_min_parts", self._split_min_parts)
            self._split_max_parts = msk_cfg.get("split_max_parts", self._split_max_parts)
            
            self._time_slice_enabled = msk_cfg.get("time_slice_enabled", self._time_slice_enabled)
            self._time_slice_min_ms = msk_cfg.get("time_slice_min_ms", self._time_slice_min_ms)
            self._time_slice_max_ms = msk_cfg.get("time_slice_max_ms", self._time_slice_max_ms)
            
            logger.info(f"OrderFingerprintMasker config updated: mode={self._mode.value}")
    
    def mask_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        price: float,
        quantity: float,
        is_risk_order: bool = False,
        context: Dict[str, Any] = None,
    ) -> List[MaskedOrder]:
        """
        对订单进行指纹掩盖
        
        Args:
            symbol: 交易对
            side: 方向 (buy/sell)
            order_type: 订单类型 (limit/market)
            price: 价格
            quantity: 数量
            is_risk_order: 是否为风控订单（风控单不拆分）
            context: 市场上下文
        
        Returns:
            掩盖后的订单列表（可能被拆分为多个）
        """
        if not self._enabled or self._mode == MaskingMode.OFF:
            return [self._create_order(symbol, side, order_type, price, quantity)]
        
        with self._lock:
            # 获取精度
            precision = self.SYMBOL_PRECISION.get(symbol, self.DEFAULT_PRECISION)
            
            # 1. 价格随机化（仅限价单）
            masked_price = price
            if self._price_jitter_enabled and order_type.lower() == "limit":
                masked_price = self._jitter_price(price, side, precision)
            
            # 2. 订单类型轮换
            final_order_type = order_type
            if self._type_rotation_enabled:
                final_order_type = self._rotate_order_type(order_type, symbol)
            
            # 3. 订单拆分（使用原始数量判断，拆分后在各子单内做抖动）
            notional = masked_price * quantity
            if self._split_enabled and not is_risk_order and notional >= self._split_min_notional:
                split_orders = self._split_order(
                    symbol, side, final_order_type, masked_price, quantity, precision
                )
                self._total_split += 1
                self._split_orders_created += len(split_orders)
                return split_orders
            
            # 4. 数量随机化（非拆分订单）
            masked_qty = self._jitter_quantity(quantity, precision)
            
            self._total_masked += 1
            return [MaskedOrder(
                symbol=symbol,
                side=side,
                order_type=final_order_type,
                price=masked_price,
                quantity=masked_qty,
                original_quantity=quantity,
                original_price=price,
                is_split=False,
                metadata={
                    "masked": True,
                    "quantity_jitter": round(masked_qty / max(quantity, 0.0001) - 1.0, 4),
                    "price_jitter": round(masked_price / max(price, 0.0001) - 1.0, 4),
                }
            )]
    
    def _jitter_quantity(self, quantity: float, precision: Dict[str, float]) -> float:
        """对数量进行随机抖动"""
        if not self._quantity_jitter_enabled:
            return quantity
        
        intensity = self._get_intensity()
        jitter_pct = random.uniform(
            self._quantity_jitter_min_pct * intensity,
            self._quantity_jitter_max_pct * intensity
        )
        # 随机选择正负方向
        jitter_pct *= random.choice([-1, 1])
        
        new_qty = quantity * (1.0 + jitter_pct)
        
        # 精度对齐
        step = precision["qty_step"]
        min_qty = precision["min_qty"]
        new_qty = round(new_qty / step) * step
        new_qty = max(min_qty, new_qty)
        
        return new_qty
    
    def _jitter_price(self, price: float, side: str, precision: Dict[str, float]) -> float:
        """对价格进行随机抖动"""
        if not self._price_jitter_enabled:
            return price
        
        intensity = self._get_intensity()
        jitter_pct = self._price_jitter_pct * intensity
        
        # 方向偏好：买单略低，卖单略高（增加成交概率）
        if side == "buy":
            jitter = random.uniform(-jitter_pct * 0.5, jitter_pct * 0.3)
        else:
            jitter = random.uniform(-jitter_pct * 0.3, jitter_pct * 0.5)
        
        new_price = price * (1.0 + jitter)
        
        # 精度对齐
        step = precision["price_step"]
        new_price = round(new_price / step) * step
        new_price = max(step, new_price)
        
        return new_price
    
    def _rotate_order_type(self, order_type: str, symbol: str) -> str:
        """轮换订单类型"""
        if random.random() > 0.3:  # 70%保持原类型
            return order_type
        
        counter = self._order_type_counter[symbol]
        self._order_type_counter[symbol] = counter + 1
        
        # 按比例分配
        r = random.random()
        if r < self._limit_order_ratio:
            return "limit"
        elif r < self._limit_order_ratio + self._post_only_ratio:
            return "post_only"
        else:
            return "market"
    
    def _split_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        price: float,
        quantity: float,
        precision: Dict[str, float],
    ) -> List[MaskedOrder]:
        """将大单拆分为随机数量的小单"""
        intensity = self._get_intensity()
        max_parts = min(self._split_max_parts, int(self._split_max_parts * intensity))
        parts = random.randint(self._split_min_parts, max(2, max_parts))
        
        # 生成随机分配比例
        remaining = 1.0
        ratios = []
        for i in range(parts - 1):
            r = random.uniform(
                self._split_size_ratio_min,
                min(self._split_size_ratio_max, remaining - self._split_size_ratio_min * (parts - i - 1))
            )
            ratios.append(r)
            remaining -= r
        ratios.append(remaining)
        
        # 随机打乱分配顺序（避免大单总是第一个）
        random.shuffle(ratios)
        
        # 生成子订单
        step = precision["qty_step"]
        min_qty = precision["min_qty"]
        split_orders = []
        allocated = 0.0
        
        for i, ratio in enumerate(ratios):
            sub_qty = round(quantity * ratio / step) * step
            sub_qty = max(min_qty, sub_qty)
            
            # 调整最后一个子订单使总量精确
            if i == len(ratios) - 1:
                sub_qty = round((quantity - allocated) / step) * step
                sub_qty = max(min_qty, sub_qty)
            
            allocated += sub_qty
            
            # 每个子订单价格微调
            sub_price = price
            if self._price_jitter_enabled and order_type == "limit":
                price_jitter = random.uniform(-self._price_jitter_pct, self._price_jitter_pct)
                sub_price = round(price * (1.0 + price_jitter) / precision["price_step"]) * precision["price_step"]
                sub_price = max(precision["price_step"], sub_price)
            
            # 时间切片延迟
            time_slice_ms = 0
            if self._time_slice_enabled and i > 0:
                time_slice_ms = self._generate_time_slice()
            
            split_orders.append(MaskedOrder(
                symbol=symbol,
                side=side,
                order_type=order_type,
                price=sub_price,
                quantity=sub_qty,
                original_quantity=quantity,
                original_price=price,
                is_split=True,
                split_index=i + 1,
                split_total=parts,
                metadata={
                    "masked": True,
                    "split": True,
                    "split_ratio": round(ratio, 4),
                    "time_slice_ms": time_slice_ms,
                    "price_jitter": round(sub_price / max(price, 0.0001) - 1.0, 4),
                }
            ))
        
        # 记录
        self._masking_history.append(MaskingRecord(
            symbol=symbol,
            original_qty=quantity,
            masked_qty=sum(o.quantity for o in split_orders),
            original_price=price,
            masked_price=split_orders[0].price if split_orders else price,
            split_count=parts,
            timestamp=time.time(),
        ))
        
        self._total_masked += 1
        return split_orders
    
    def _generate_time_slice(self) -> float:
        """生成时间切片延迟（毫秒）"""
        if self._time_slice_distribution == "exponential":
            scale = (self._time_slice_max_ms - self._time_slice_min_ms) / 3.0
            delay = random.expovariate(1.0 / max(scale, 1.0))
            delay = self._time_slice_min_ms + delay
        elif self._time_slice_distribution == "normal":
            mean = (self._time_slice_min_ms + self._time_slice_max_ms) / 2
            std = (self._time_slice_max_ms - self._time_slice_min_ms) / 4
            delay = random.normalvariate(mean, std)
        else:
            delay = random.uniform(self._time_slice_min_ms, self._time_slice_max_ms)
        
        return max(self._time_slice_min_ms, min(delay, self._time_slice_max_ms))
    
    def _get_intensity(self) -> float:
        """获取掩盖强度因子"""
        if self._mode == MaskingMode.LIGHT:
            return 0.5
        elif self._mode == MaskingMode.AGGRESSIVE:
            return 1.5
        return 1.0
    
    def _create_order(
        self, symbol: str, side: str, order_type: str, price: float, quantity: float
    ) -> MaskedOrder:
        """创建基础订单（无掩盖）"""
        return MaskedOrder(
            symbol=symbol,
            side=side,
            order_type=order_type,
            price=price,
            quantity=quantity,
            original_quantity=quantity,
            original_price=price,
            is_split=False,
        )
    
    def get_symbol_precision(self, symbol: str) -> Dict[str, float]:
        """获取币种精度"""
        return self.SYMBOL_PRECISION.get(symbol, self.DEFAULT_PRECISION)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取掩盖器统计"""
        with self._lock:
            return {
                "enabled": self._enabled,
                "mode": self._mode.value,
                "total_masked": self._total_masked,
                "total_split": self._total_split,
                "split_orders_created": self._split_orders_created,
                "current_intensity": round(self._get_intensity(), 2),
                "quantity_jitter_range": f"{self._quantity_jitter_min_pct:.1%}-{self._quantity_jitter_max_pct:.1%}",
                "price_jitter_pct": self._price_jitter_pct,
                "split_config": {
                    "min_notional": self._split_min_notional,
                    "min_parts": self._split_min_parts,
                    "max_parts": self._split_max_parts,
                },
                "time_slice_range_ms": f"{self._time_slice_min_ms}-{self._time_slice_max_ms}",
                "order_type_rotation": {
                    "limit_ratio": self._limit_order_ratio,
                    "post_only_ratio": self._post_only_ratio,
                },
            }
    
    def reset_stats(self) -> None:
        """重置统计"""
        with self._lock:
            self._total_masked = 0
            self._total_split = 0
            self._split_orders_created = 0
            self._masking_history.clear()
            self._order_type_counter.clear()


__all__ = [
    "OrderFingerprintMasker",
    "MaskingMode",
    "MaskedOrder",
    "OrderType",
]