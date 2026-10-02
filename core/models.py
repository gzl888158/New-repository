"""定义交易系统核心数据模型，包括行情、K线、订单、持仓、账户和信号等 dataclass 结构。"""
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime
import math

@dataclass
class TickData:
    symbol: str
    price: float
    volume: float
    bid_price: float
    bid_volume: float
    ask_price: float
    ask_volume: float
    timestamp: datetime
    instrument_type: str = "SWAP"

@dataclass
class BarData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: str

@dataclass
class Order:
    order_id: str
    symbol: str
    side: str
    type: str
    quantity: float
    price: float
    filled_quantity: float
    status: str
    create_time: datetime
    update_time: Optional[datetime] = None
    stop_price: Optional[float] = None
    leverage: Optional[int] = None
    strategy_name: Optional[str] = None
    priority: int = 3

@dataclass
class Position:
    symbol: str
    side: str
    quantity: float
    avg_cost: float
    mark_price: float
    unrealized_pnl: float
    margin: float
    leverage: int
    maintenance_margin_rate: float
    notional_usd: float = 0.0  # OKX API返回的名义价值(USD)，已正确计入合约乘数ctVal
    liquidation_price: float = 0.0  # OKX API返回的强平价(liqPx)；0=未提供，下游风控需回退估算
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AccountInfo:
    total_equity: float
    available_balance: float
    used_margin: float
    unrealized_pnl: float
    margin_rate: float
    timestamp: datetime

@dataclass
class Signal:
    symbol: str
    strategy_name: str
    signal_type: str
    direction: str
    price: float
    quantity: float
    leverage: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.0
    order_type: str = ""  # 显式订单类型（"market"/"limit"/"post_only"），空则由执行层自动判断
    pending_price: float = 0.0  # 等位挂单价：>0 表示挂在支撑/压力位等回踩/反弹（post_only 时生效），0 表示正常下单
    pending_spread_multiplier: float = 1.0  # 等位挂单间距乘数：震荡期边缘=2.0 翻倍间距，正常=1.0
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        # 防御显式传入 None 的历史调用，避免下游 .isoformat() 崩溃
        if self.timestamp is None:
            self.timestamp = datetime.now()

        if self.direction not in {"long", "short", "buy", "sell"}:
            raise ValueError("direction must be one of: long, short, buy, sell")

        for field_name in ("price", "quantity", "confidence"):
            value = getattr(self, field_name)
            try:
                finite = math.isfinite(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field_name} must be a finite number") from exc
            if not finite:
                raise ValueError(f"{field_name} must be a finite number")

        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")

@dataclass
class FundingRate:
    symbol: str
    funding_rate: float
    next_funding_time: datetime
    timestamp: datetime

@dataclass
class RiskMetric:
    symbol: str
    position_size: float
    margin: float
    unrealized_pnl: float
    margin_rate: float
    max_drawdown: float
    daily_pnl: float
    hourly_pnl: float

@dataclass
class StrategyState:
    strategy_name: str
    symbol: str
    is_running: bool
    current_position: Optional[Position] = None
    statistics: Dict[str, float] = field(default_factory=dict)
    last_trade_time: Optional[datetime] = None
    consecutive_losses: int = 0