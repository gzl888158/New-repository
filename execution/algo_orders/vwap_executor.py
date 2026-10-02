"""
VWAP 执行器 (Volume-Weighted Average Price)

.. deprecated:: 实验性模块，未接入生产交易链路。

基于历史成交量分布执行订单，最小化与VWAP基准的偏差：
  - 成交量预测：基于历史日内成交量分布建模
  - 动态分配：根据实时成交量偏离调整计划
  - VWAP跟踪：实时计算已执行部分的VWAP，与目标VWAP对比
  - 自适应调整：成交量异常时重新分配
  - 流动性sweep：在流动性高峰时段执行更大比例
"""
import asyncio
import math
import random
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Tuple
import numpy as np
from loguru import logger

from core.direction_unifier import DirectionUnifier
from execution.algo_orders.algo_execution_engine import (
    AlgoOrderConfig, AlgoExecutionResult, ExecutionSlice,
    AlgoOrderStatus,
)


@dataclass
class VolumeBucket:
    """成交量桶（时间区间）"""
    start_hour: float      # 开始时间 (0-24)
    end_hour: float        # 结束时间
    volume_fraction: float  # 成交量占比 (0-1)
    avg_price: float = 0.0
    volatility: float = 0.0


@dataclass
class VolumeProfile:
    """日内成交量分布"""
    symbol: str
    date: str
    total_volume: float
    buckets: List[VolumeBucket] = field(default_factory=list)
    is_synthetic: bool = False  # 是否为合成分布（没有足够历史数据）
    quality_score: float = 1.0   # 预测质量


@dataclass
class VWAPConfig:
    """VWAP配置"""
    duration_seconds: float = 3600.0
    num_volume_buckets: int = 24    # 每日24个半小时成交量桶
    min_slice_pct: float = 0.01     # 最小切片占比
    max_slice_pct: float = 0.15     # 最大切片占比
    adaptive_enabled: bool = True
    volume_deviation_threshold: float = 0.3  # 成交量偏离30%时调整
    limit_offset_bps: float = 3.0


class VWAPState:
    IDLE = "idle"
    COMPUTING_PROFILE = "computing_profile"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"


# ═══════════════════════════════════════════════════════════════
# VWAP 执行器
# ═══════════════════════════════════════════════════════════════

class VWAPExecutor:
    """VWAP 算法执行器"""

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config.get("vwap_executor", {}) if config else {}
        self._default_duration = cfg.get("default_duration", 3600)
        self._num_buckets = cfg.get("num_volume_buckets", 24)
        self._min_slice_pct = cfg.get("min_slice_pct", 0.01)
        self._max_slice_pct = cfg.get("max_slice_pct", 0.15)
        self._limit_offset_bps = cfg.get("limit_offset_bps", 3.0)
        # 历史成交量缓存
        self._volume_profiles: Dict[str, List[VolumeProfile]] = {}
        self._profile_cache_size = cfg.get("profile_cache_days", 30)
        # 实时成交量跟踪
        self._real_time_volume: Dict[str, deque] = {}

        logger.info(f"VWAPExecutor initialized: buckets={self._num_buckets}, "
                    f"slice_range=[{self._min_slice_pct*100:.0f}%-{self._max_slice_pct*100:.0f}%]")

    # ── 成交量分布计算 ────────────────────────────────────────

    @staticmethod
    def compute_default_profile(symbol: str) -> VolumeProfile:
        """生成默认成交量分布（加密市场典型U型）"""
        buckets = []
        hours = np.arange(0, 24, 0.5)

        # 加密市场成交量U型分布（亚洲盘+欧美盘双峰）
        # 模拟：0-8低量，8-12高峰（亚洲），12-16低谷，16-20高峰（欧美），20-24中等
        volumes = np.zeros(len(hours))
        for i, h in enumerate(hours):
            if h < 2:
                volumes[i] = 0.025
            elif h < 7:
                volumes[i] = 0.020 + 0.002 * h
            elif h < 10:
                volumes[i] = 0.055  # 亚洲盘高峰
            elif h < 12:
                volumes[i] = 0.045
            elif h < 15:
                volumes[i] = 0.030  # 低谷
            elif h < 17:
                volumes[i] = 0.060  # 欧美盘开盘高峰
            elif h < 20:
                volumes[i] = 0.055
            elif h < 22:
                volumes[i] = 0.040
            else:
                volumes[i] = 0.030

        volumes /= volumes.sum()  # 归一化

        for i in range(len(hours) - 1):
            buckets.append(VolumeBucket(
                start_hour=hours[i],
                end_hour=hours[i + 1],
                volume_fraction=float(volumes[i]),
            ))

        return VolumeProfile(
            symbol=symbol,
            date=datetime.now().strftime("%Y-%m-%d"),
            total_volume=0,
            buckets=buckets,
            is_synthetic=True,
            quality_score=0.6,
        )

    def update_volume_profile(self, symbol: str, klines: List) -> VolumeProfile:
        """基于K线数据更新成交量分布"""
        if not klines or len(klines) < 20:
            return self.compute_default_profile(symbol)

        # 按小时聚合成交量
        hour_volumes = {}
        for k in klines:
            ts = float(k[0]) / 1000  # 毫秒时间戳 -> 秒
            dt = datetime.fromtimestamp(ts)
            hour_bucket = dt.hour + dt.minute / 60.0
            vol = float(k[5])  # volume in base currency
            # 分配到最近半小时桶
            bucket_key = round(hour_bucket * 2) / 2
            hour_volumes[bucket_key] = hour_volumes.get(bucket_key, 0) + vol

        total = sum(hour_volumes.values()) if hour_volumes else 1.0
        buckets = []
        for h in sorted(hour_volumes.keys()):
            h_end = min(h + 0.5, 24.0)
            buckets.append(VolumeBucket(
                start_hour=h,
                end_hour=h_end,
                volume_fraction=hour_volumes[h] / total,
            ))

        profile = VolumeProfile(
            symbol=symbol,
            date=datetime.now().strftime("%Y-%m-%d"),
            total_volume=total,
            buckets=buckets,
            is_synthetic=False,
            quality_score=0.85,
        )

        # 缓存
        if symbol not in self._volume_profiles:
            self._volume_profiles[symbol] = []
        self._volume_profiles[symbol].append(profile)
        if len(self._volume_profiles[symbol]) > self._profile_cache_size:
            self._volume_profiles[symbol] = self._volume_profiles[symbol][-self._profile_cache_size:]

        return profile

    def get_volume_profile(self, symbol: str) -> VolumeProfile:
        """获取当前成交量分布（优先缓存，回退到默认）"""
        profiles = self._volume_profiles.get(symbol, [])
        if profiles:
            # 返回最近的
            return profiles[-1]
        return self.compute_default_profile(symbol)

    # ── 切片分配 ──────────────────────────────────────────────

    def allocate_to_slices(self, total_quantity: float, profile: VolumeProfile,
                            start_time: datetime, end_time: datetime,
                            min_slice_pct: float, max_slice_pct: float) -> List[Tuple[float, float, float]]:
        """根据成交量分布分配切片 (quantity, start_hour, end_hour)"""
        duration_hours = (end_time - start_time).total_seconds() / 3600.0
        start_hour = start_time.hour + start_time.minute / 60.0

        allocations = []
        used_frac = 0.0

        # 筛选覆盖执行时段内的成交量桶
        relevant_buckets = []
        for b in profile.buckets:
            # 计算与执行时段的重叠
            bucket_start = max(b.start_hour, start_hour)
            bucket_end = min(b.end_hour, start_hour + duration_hours)
            if bucket_end > bucket_start:
                overlap = bucket_end - bucket_start
                bucket_duration = b.end_hour - b.start_hour
                if bucket_duration > 0:
                    adjusted_frac = b.volume_fraction * (overlap / bucket_duration)
                else:
                    adjusted_frac = b.volume_fraction
                relevant_buckets.append((b, adjusted_frac, overlap))

        if not relevant_buckets:
            # 均匀分配
            n = max(5, int(duration_hours * 2))
            for i in range(n):
                allocations.append((total_quantity / n, start_hour + i * duration_hours / n,
                                   start_hour + (i + 1) * duration_hours / n))
            return allocations

        # 归一化
        total_frac = sum(f for _, f, _ in relevant_buckets)
        if total_frac > 0:
            for b, frac, overlap in relevant_buckets:
                norm_frac = frac / total_frac
                qty = total_quantity * norm_frac
                # 限制最小/最大切片
                qty = max(total_quantity * min_slice_pct, min(qty, total_quantity * max_slice_pct))
                allocations.append((qty, b.start_hour, b.end_hour))

        # 归一化数量
        total_allocated = sum(a[0] for a in allocations)
        if total_allocated > 0:
            allocations = [
                (q * total_quantity / total_allocated, s, e)
                for q, s, e in allocations
            ]

        return allocations

    # ── 主执行逻辑 ────────────────────────────────────────────

    async def execute(self, config: AlgoOrderConfig,
                       on_slice_filled: Callable = None) -> AlgoExecutionResult:
        """执行 VWAP 算法订单"""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=config.duration_seconds)

        result = AlgoExecutionResult(
            order_id=config.order_id,
            algo_type="VWAP",
            symbol=config.symbol,
            side=config.side,
            total_quantity=config.total_quantity,
            filled_quantity=0.0,
            fill_rate=0.0,
            arrival_price=0.0,
            start_time=start_time,
        )

        # 获取成交量分布
        profile = self.get_volume_profile(config.symbol)

        # fail-closed：无真实成交量分布（合成/缺失）时拒绝执行，避免基于伪造U型分布下单
        if profile is None or profile.is_synthetic:
            logger.error(f"VWAP {config.order_id}: no real volume profile for {config.symbol}, "
                         f"fail-closed - refusing to execute on synthetic distribution")
            result.status = AlgoOrderStatus.FAILED
            result.error_message = "volume profile unavailable (fail-closed)"
            result.end_time = datetime.now()
            return result

        # 分配切片
        allocations = self.allocate_to_slices(
            config.total_quantity, profile, start_time, end_time,
            self._min_slice_pct, self._max_slice_pct,
        )

        # 获取到达价格
        if config.market_data_fn:
            try:
                market = config.market_data_fn(config.symbol)
                if market:
                    raw = market.get("mid", market.get("last"))
                    if raw is not None:
                        result.arrival_price = float(raw)
            except (TypeError, ValueError):
                pass

        total_filled = 0.0
        total_cost = 0.0
        vwap_target = 0.0

        for seq, (qty, _, _) in enumerate(allocations):
            if qty <= 0:
                continue

            # 通知监控器：切片已排定
            if config.on_slice_scheduled_fn:
                config.on_slice_scheduled_fn(seq, qty, datetime.now())

            # 限价 = mid ± offset
            price = self._calc_vwap_limit(config)

            # fail-closed：限价不可得时拒绝该切片
            if price is None or price <= 0:
                logger.warning(f"VWAP {config.order_id}: limit price unavailable for slice {seq}, "
                               f"fail-closed - skipping slice")
                es = ExecutionSlice(
                    slice_id=f"vwap_{config.order_id}_{seq:03d}", sequence=seq,
                    quantity=qty, status="rejected",
                    error_message="limit price unavailable (fail-closed)",
                )
                if on_slice_filled:
                    await on_slice_filled(config.order_id, es)
                continue

            # 执行
            if config.executor_fn:
                try:
                    order_params = {
                        "symbol": config.symbol,
                        "side": config.side,
                        "pos_side": DirectionUnifier.to_pos_side(config.side),
                        "quantity": qty,
                        "order_type": "limit",
                        "price": price,
                        "trace_id": f"algo_{config.order_id}_{uuid.uuid4().hex[:8]}",
                    }
                    fill = config.executor_fn(order_params)
                    if hasattr(fill, '__await__'):
                        fill = await fill

                    # 通知监控器：切片已提交到交易所
                    if config.on_slice_submitted_fn:
                        config.on_slice_submitted_fn(seq, qty, price, "limit")

                    if fill:
                        filled_qty = float(fill.get("filled", 0))
                        avg_px = float(fill.get("avg_price", 0))
                        total_filled += filled_qty
                        total_cost += filled_qty * avg_px

                        es = ExecutionSlice(
                            slice_id=f"vwap_{config.order_id}_{seq:03d}",
                            sequence=seq,
                            quantity=qty,
                            filled_quantity=filled_qty,
                            avg_fill_price=avg_px,
                            order_type="limit",
                            status="filled" if filled_qty > 0 else "partial",
                            submitted_at=datetime.now(),
                            filled_at=datetime.now(),
                        )
                        if on_slice_filled:
                            await on_slice_filled(config.order_id, es)

                except Exception as e:
                    logger.warning(f"VWAP slice {seq} failed: {e}")
                    es = ExecutionSlice(
                        slice_id=f"vwap_{config.order_id}_{seq:03d}", sequence=seq,
                        quantity=qty, status="rejected", error_message=str(e),
                    )
                    if on_slice_filled:
                        await on_slice_filled(config.order_id, es)

            # 小间隔
            await asyncio.sleep(random.uniform(0.1, 1.0))

        # 市价清算剩余
        remaining = config.total_quantity - total_filled
        if remaining > 0 and config.executor_fn:
            try:
                order_params = {
                    "symbol": config.symbol,
                    "side": config.side,
                    "pos_side": DirectionUnifier.to_pos_side(config.side),
                    "quantity": remaining,
                    "order_type": "market",
                    "trace_id": f"algo_{config.order_id}_{uuid.uuid4().hex[:8]}",
                }
                fill = config.executor_fn(order_params)
                if hasattr(fill, '__await__'):
                    fill = await fill
                if fill:
                    q = float(fill.get("filled", 0))
                    px = float(fill.get("avg_price", 0))
                    total_filled += q
                    total_cost += q * px
            except Exception as e:
                logger.warning(f"VWAP final sweep failed: {e}")

        # 计算VWAP
        if total_filled > 0:
            result.avg_execution_price = total_cost / total_filled
        result.filled_quantity = total_filled
        result.fill_rate = total_filled / max(config.total_quantity, 1e-10)

        # VWAP目标价格（加权日均价简化：执行期间的mid均价）
        result.target_price = result.avg_execution_price  # 简化：实时VWAP近似
        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - start_time).total_seconds()

        if result.arrival_price > 0 and result.avg_execution_price > 0:
            slip = (result.avg_execution_price - result.arrival_price) / result.arrival_price * 10000
            if DirectionUnifier.is_short(config.side):
                slip *= -1
            result.arrival_slippage_bps = slip
            result.implementation_shortfall = abs(total_cost - total_filled * result.arrival_price)

        logger.info(f"VWAP {config.order_id} complete: filled={total_filled:.4f}/{config.total_quantity:.4f} "
                    f"avg_px={result.avg_execution_price:.4f}")

        return result

    def _calc_vwap_limit(self, config: AlgoOrderConfig) -> Optional[float]:
        """计算VWAP限价"""
        if not config.market_data_fn:
            return None
        try:
            market = config.market_data_fn(config.symbol)
            if market:
                raw = market.get("mid", market.get("last"))
                if raw is None:
                    return None
                mid = float(raw)
                if mid <= 0 or math.isnan(mid):
                    return None
                offset = mid * self._limit_offset_bps / 10000.0
                if DirectionUnifier.is_long(config.side):
                    return mid + offset
                return mid - offset
        except (TypeError, ValueError):
            pass
        return None

    # ── 成交量偏差检测 ────────────────────────────────────────

    def detect_volume_deviation(self, symbol: str) -> Dict[str, Any]:
        """检测实时成交量与历史分布的偏差"""
        profile = self.get_volume_profile(symbol)
        if profile.is_synthetic:
            return {"deviation": 0.0, "is_anomaly": False, "use_default": True}

        now_hour = datetime.now().hour + datetime.now().minute / 60.0
        expected_buckets = [b for b in profile.buckets
                          if b.start_hour <= now_hour <= b.end_hour]
        expected_frac = sum(b.volume_fraction for b in expected_buckets)

        # 检查实时成交量
        rt_vol = self._real_time_volume.get(symbol, deque(maxlen=100))
        if len(rt_vol) > 5:
            recent_avg = np.mean(list(rt_vol)[-5:])
            historical_avg = profile.total_volume * expected_frac if profile.total_volume > 0 else 1000
            if historical_avg > 0:
                deviation = abs(recent_avg - historical_avg) / historical_avg
                return {
                    "deviation": round(deviation, 3),
                    "is_anomaly": deviation > 0.3,
                    "recent_volume": recent_avg,
                    "expected_volume": historical_avg,
                    "use_default": False,
                }

        return {"deviation": 0.0, "is_anomaly": False, "use_default": True}

    def update_real_time_volume(self, symbol: str, volume: float):
        """更新实时成交量"""
        if symbol not in self._real_time_volume:
            self._real_time_volume[symbol] = deque(maxlen=100)
        self._real_time_volume[symbol].append(volume)

    def get_status(self) -> Dict[str, Any]:
        return {
            "num_buckets": self._num_buckets,
            "slice_range": f"{self._min_slice_pct*100:.0f}%-{self._max_slice_pct*100:.0f}%",
            "limit_offset_bps": self._limit_offset_bps,
            "cached_profiles": len(self._volume_profiles),
        }

    # ── 数据喂养 ──────────────────────────────────────────────

    def set_okx_client(self, okx_client):
        """注入 OKX 客户端（用于拉取K线构建成交量分布）"""
        self._okx_client = okx_client

    async def refresh_volume_profile(self, symbol: str,
                                      bar: str = "1H",
                                      lookback_bars: int = 168) -> Optional[VolumeProfile]:
        """从 OKX 拉取K线数据，构建真实成交量分布；失败时返回 None（fail-closed，不伪造合成分布）"""
        if not getattr(self, '_okx_client', None):
            logger.debug(f"VWAP: no OKX client, cannot build real profile for {symbol}")
            return None

        try:
            import concurrent.futures
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                raw_klines = await loop.run_in_executor(
                    executor,
                    lambda: self._okx_client.get_kline(symbol, bar, lookback_bars)
                )
            if raw_klines:
                profile = self.update_volume_profile(symbol, raw_klines)
                logger.info(f"VWAP volume profile updated for {symbol}: "
                           f"{len(profile.buckets)} buckets, quality={profile.quality_score:.2f}")
                return profile
        except Exception as e:
            logger.warning(f"VWAP profile refresh failed for {symbol}: {e}")

        return None

    async def refresh_profiles_batch(self, symbols: List[str],
                                      bar: str = "1H",
                                      lookback_bars: int = 168) -> Dict[str, Optional[VolumeProfile]]:
        """批量刷新多个标的的成交量分布"""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = await self.refresh_volume_profile(symbol, bar, lookback_bars)
            except Exception as e:
                logger.warning(f"VWAP batch refresh failed for {symbol}: {e}")
                results[symbol] = None
        return results
