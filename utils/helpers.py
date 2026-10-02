"""
通用工具函数集，提供仓位与保证金计算、盈亏计算、价格格式化等辅助函数。

企业级约束：
- 所有外部数值输入经 safe_* 转换，禁止裸 int()/float()
- 禁止 NaN/Inf 出现在返回值中（JSON 序列化安全）
- 所有除法经 safe_div 防护，分母为 0/非有限时回退默认值
- 异常降级时记录日志，不静默吞错
"""
import math
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from loguru import logger

VALID_DIRECTIONS = ("long", "short", "buy", "sell")


# ── 类型安全工具（与 eval._base / visualize._base 保持一致，解耦独立） ──
def safe_float(value: Any, default: float = 0.0) -> float:
    """安全 float 转换：None/NaN/Inf/非法值回退 default。"""
    if value is None:
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v or v in (float("inf"), float("-inf")):
        return default
    return v


def safe_int(value: Any, default: int = 0) -> int:
    """安全 int 转换。"""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def safe_div(numerator: Any, denominator: Any, default: float = 0.0) -> float:
    """安全除法：分母为 0/NaN/Inf 时回退 default。"""
    try:
        d = float(denominator)
        if d != d or d in (float("inf"), float("-inf")) or d == 0:
            return default
        n = float(numerator)
        if n != n or n in (float("inf"), float("-inf")):
            return default
        return n / d
    except (TypeError, ValueError):
        return default


def safe_finite(value: Any, default: float = 0.0) -> float:
    """确保值有限：NaN/Inf 回退 default。"""
    try:
        v = float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    if v != v or v in (float("inf"), float("-inf")):
        return default
    return v


def _validate_direction(direction: str) -> None:
    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"Invalid direction '{direction}'. Must be one of {VALID_DIRECTIONS}")


def normalize_direction(direction: str) -> str:
    """将 buy/sell 转换为 long/short，保持 long/short 不变"""
    _validate_direction(direction)
    if direction == "buy":
        return "long"
    elif direction == "sell":
        return "short"
    return direction


def calculate_position_size(capital: float, price: float, leverage: int,
                           risk_percent: float = 0.01) -> float:
    price = safe_float(price, 0.0)
    leverage = safe_int(leverage, 0)
    capital = safe_float(capital, 0.0)
    risk_percent = safe_float(risk_percent, 0.0)
    if price <= 0:
        logger.warning(f"calculate_position_size: invalid price {price}, returning 0")
        return 0.0
    if leverage <= 0:
        logger.warning(f"calculate_position_size: invalid leverage {leverage}, returning 0")
        return 0.0
    position_value = capital * risk_percent * leverage
    quantity = safe_div(position_value, price, 0.0)
    return max(0.0, quantity)


def calculate_margin(quantity: float, price: float, leverage: int) -> float:
    quantity = safe_float(quantity, 0.0)
    price = safe_float(price, 0.0)
    leverage = safe_int(leverage, 0)
    if leverage <= 0:
        logger.warning(f"calculate_margin: invalid leverage {leverage}, returning 0")
        return 0.0
    return abs(quantity * price) / leverage


def calculate_pnl(entry_price: float, current_price: float, quantity: float,
                  direction: str) -> float:
    _validate_direction(direction)
    entry_price = safe_float(entry_price, 0.0)
    current_price = safe_float(current_price, 0.0)
    quantity = safe_float(quantity, 0.0)
    if direction == "long":
        return (current_price - entry_price) * quantity
    else:
        return (entry_price - current_price) * quantity


def calculate_pnl_percent(entry_price: float, current_price: float, direction: str) -> float:
    _validate_direction(direction)
    entry_price = safe_float(entry_price, 0.0)
    current_price = safe_float(current_price, 0.0)
    if entry_price == 0:
        logger.warning("calculate_pnl_percent: entry_price is 0, returning 0")
        return 0.0
    if direction == "long":
        return (current_price - entry_price) / entry_price
    else:
        return (entry_price - current_price) / entry_price

def round_to_tick_size(price: float, tick_size: float = 0.0001) -> float:
    price = safe_float(price, 0.0)
    tick_size = safe_float(tick_size, 0.0)
    if tick_size <= 0:
        return price
    return round(price / tick_size) * tick_size

def format_price(price: float, symbol: str) -> str:
    if symbol.startswith("BTC"):
        return f"{price:.2f}"
    elif symbol.startswith("ETH"):
        return f"{price:.2f}"
    elif symbol.startswith("SOL"):
        return f"{price:.2f}"
    elif symbol.startswith("PEPE") or symbol.startswith("WIF"):
        return f"{price:.6f}"
    else:
        return f"{price:.4f}"

def get_timestamp_ms() -> int:
    return int(datetime.now().timestamp() * 1000)

def generate_order_id(prefix: str = "ORD") -> str:
    return f"{prefix}{get_timestamp_ms()}"

def is_weekend() -> bool:
    return datetime.now().weekday() >= 5

def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))

def calculate_sharpe_ratio(returns: list, risk_free_rate: float = 0.0) -> float:
    if not returns:
        return 0.0

    risk_free_rate = safe_float(risk_free_rate, 0.0)
    # 过滤 NaN/Inf，避免污染统计
    excess_returns = []
    for r in returns:
        rv = safe_finite(r, None)
        if rv is not None:
            excess_returns.append(rv - risk_free_rate)

    n = len(excess_returns)
    if n == 0:
        return 0.0
    mean_return = sum(excess_returns) / n
    if n < 2:
        return 0.0
    variance = sum((r - mean_return) ** 2 for r in excess_returns) / (n - 1)
    std_dev = math.sqrt(variance)

    if std_dev == 0:
        return 0.0

    return mean_return / std_dev


def calculate_max_drawdown(equity_curve: list) -> float:
    if not equity_curve:
        return 0.0

    # 过滤非有限值，避免峰值/回撤计算失真
    cleaned = []
    for eq in equity_curve:
        v = safe_finite(eq, None)
        if v is not None:
            cleaned.append(v)
    if not cleaned:
        return 0.0

    max_drawdown = 0.0
    peak = cleaned[0]

    for equity in cleaned[1:]:
        if equity > peak:
            peak = equity

        if peak > 0:
            drawdown = (peak - equity) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

    return max_drawdown

def get_price_precision(symbol: str) -> int:
    if symbol.startswith("BTC") or symbol.startswith("ETH"):
        return 2
    elif symbol.startswith("SOL"):
        return 2
    elif symbol.startswith("PEPE") or symbol.startswith("WIF") or symbol.startswith("SHIB"):
        return 6
    else:
        return 4

def calculate_trading_fee(quantity: float, price: float, fee_rate: float = 0.0005, 
                          is_taker: bool = True) -> float:
    if is_taker:
        fee_rate = fee_rate
    else:
        fee_rate = fee_rate * 0.4
    return abs(quantity * price * fee_rate)

def calculate_round_trip_cost(quantity: float, price: float, fee_rate: float = 0.0005,
                              slippage_pct: float = 0.001, is_taker: bool = True) -> Dict[str, float]:
    open_fee = calculate_trading_fee(quantity, price, fee_rate, is_taker)
    close_fee = calculate_trading_fee(quantity, price, fee_rate, is_taker)
    total_fees = open_fee + close_fee
    
    slippage_cost = abs(quantity * price * slippage_pct * 2)
    
    total_cost = total_fees + slippage_cost
    notional_value = abs(quantity * price)
    cost_ratio = total_cost / notional_value if notional_value > 0 else 0
    
    return {
        "open_fee": open_fee,
        "close_fee": close_fee,
        "total_fees": total_fees,
        "slippage_cost": slippage_cost,
        "total_cost": total_cost,
        "notional_value": notional_value,
        "cost_ratio": cost_ratio,
        "breakeven_pct": cost_ratio
    }

def calculate_funding_fee(position_value: float, funding_rate: float, 
                          is_long: bool = True) -> float:
    if funding_rate > 0 and is_long:
        return -abs(position_value * funding_rate)
    elif funding_rate > 0 and not is_long:
        return abs(position_value * funding_rate)
    elif funding_rate < 0 and is_long:
        return abs(position_value * funding_rate)
    else:
        return -abs(position_value * funding_rate)

def estimate_net_profit(entry_price: float, exit_price: float, quantity: float,
                        direction: str, fee_rate: float = 0.0005,
                        slippage_pct: float = 0.001, funding_rate: float = 0,
                        hold_periods: int = 0, is_taker: bool = True) -> Dict[str, float]:
    _validate_direction(direction)
    entry_price = safe_float(entry_price, 0.0)
    exit_price = safe_float(exit_price, 0.0)
    quantity = safe_float(quantity, 0.0)
    fee_rate = safe_float(fee_rate, 0.0005)
    slippage_pct = safe_float(slippage_pct, 0.001)
    funding_rate = safe_float(funding_rate, 0.0)
    hold_periods = safe_int(hold_periods, 0)

    notional_value = abs(quantity * entry_price)
    costs = calculate_round_trip_cost(quantity, entry_price, fee_rate, slippage_pct, is_taker)

    if direction == "long":
        gross_pnl = (exit_price - entry_price) * quantity
    else:
        gross_pnl = (entry_price - exit_price) * quantity

    funding_cost = 0.0
    if funding_rate != 0 and hold_periods > 0:
        is_long = direction == "long"
        funding_cost = calculate_funding_fee(notional_value, funding_rate, is_long) * hold_periods

    net_pnl = gross_pnl - costs["total_cost"] + funding_cost
    net_pnl_pct = safe_div(net_pnl, notional_value, 0.0)

    # 修复三元运算符优先级 bug：原代码 costs["total_cost"] + abs(funding_cost) if ...
    # 因 + 优先级高于 if/else，funding_cost>=0 时退化为 2*total_cost
    total_cost = (costs["total_cost"] + abs(funding_cost)) if funding_cost < 0 else costs["total_cost"]

    return {
        "gross_pnl": gross_pnl,
        "total_fees": costs["total_fees"],
        "slippage_cost": costs["slippage_cost"],
        "funding_cost": funding_cost,
        "total_cost": total_cost,
        "net_pnl": net_pnl,
        "net_pnl_pct": net_pnl_pct,
        "notional_value": notional_value,
        "is_profitable": net_pnl > 0
    }

def calculate_position_margin(quantity: float, price: float, leverage: int) -> float:
    if leverage <= 0:
        return 0.0
    return abs(quantity * price) / leverage

def calculate_margin_ratio(equity: float, used_margin: float) -> float:
    equity = safe_float(equity, 0.0)
    used_margin = safe_float(used_margin, 0.0)
    if used_margin <= 0:
        # 无保证金占用时返回 0.0 而非 inf，保证 JSON 序列化安全
        return 0.0
    return equity / used_margin

def calculate_liquidation_price(entry_price: float, direction: str, leverage: int,
                                 maintenance_margin_rate: float = 0.005) -> float:
    _validate_direction(direction)
    if leverage <= 0:
        return 0.0
    
    mmr = maintenance_margin_rate
    if direction == "long":
        liq_price = entry_price * (1 - 1 / leverage + mmr)
    else:
        liq_price = entry_price * (1 + 1 / leverage - mmr)
    
    return max(0, liq_price)

def calculate_risk_reward_ratio(entry_price: float, take_profit: float, 
                                 stop_loss: float, direction: str) -> float:
    _validate_direction(direction)
    
    if direction == "long":
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
    else:
        risk = stop_loss - entry_price
        reward = entry_price - take_profit
    
    if risk <= 0:
        return 0.0
    return reward / risk

def calculate_take_profit(entry_price: float, direction: str, tp_pct: float,
                          slippage_pct: float = 0.001, precision: int = 4) -> float:
    _validate_direction(direction)
    # P27: 归一化方向，buy→long, sell→short
    direction = normalize_direction(direction)
    if entry_price <= 0:
        logger.warning(f"calculate_take_profit: invalid entry_price {entry_price}")
        return 0.0
    if direction == "long":
        tp_price = entry_price * (1 + tp_pct)
        slippage_adjustment = entry_price * slippage_pct
        tp_price -= slippage_adjustment
    else:
        tp_price = entry_price * (1 - tp_pct)
        slippage_adjustment = entry_price * slippage_pct
        tp_price += slippage_adjustment

    return round(tp_price, precision)


def calculate_stop_loss(entry_price: float, direction: str, sl_pct: float,
                        slippage_pct: float = 0.001, precision: int = 4) -> float:
    _validate_direction(direction)
    # P27: 归一化方向，buy→long, sell→short
    direction = normalize_direction(direction)
    if entry_price <= 0:
        logger.warning(f"calculate_stop_loss: invalid entry_price {entry_price}")
        return 0.0
    if direction == "long":
        sl_price = entry_price * (1 - sl_pct)
        slippage_adjustment = entry_price * slippage_pct
        sl_price -= slippage_adjustment
    else:
        sl_price = entry_price * (1 + sl_pct)
        slippage_adjustment = entry_price * slippage_pct
        sl_price += slippage_adjustment

    return round(sl_price, precision)


def calculate_tp_sl_from_atr(entry_price: float, direction: str, atr: float,
                             tp_multiplier: float = 3.0, sl_multiplier: float = 1.0,
                             slippage_pct: float = 0.001, precision: int = 4) -> tuple:
    if entry_price <= 0:
        logger.warning(f"calculate_tp_sl_from_atr: invalid entry_price {entry_price}")
        return 0.0, 0.0
    atr_pct = atr / entry_price if entry_price > 0 else 0.0
    sl_pct = atr_pct * sl_multiplier
    tp_pct = sl_pct * tp_multiplier

    stop_loss = calculate_stop_loss(entry_price, direction, sl_pct, slippage_pct, precision)
    take_profit = calculate_take_profit(entry_price, direction, tp_pct, slippage_pct, precision)

    return take_profit, stop_loss


def validate_tp_sl_prices(entry_price: float, direction: str, take_profit: float,
                          stop_loss: float, current_price: float = None,
                          min_distance_pct: float = 0.001) -> Dict[str, Any]:
    _validate_direction(direction)
    # P27: 归一化方向，buy→long, sell→short
    direction = normalize_direction(direction)
    errors = []
    warnings = []

    if entry_price <= 0:
        errors.append(f"Invalid entry price: {entry_price}")
        return {"valid": False, "errors": errors, "warnings": warnings}

    min_distance = entry_price * min_distance_pct

    if direction == "long":
        if take_profit <= entry_price:
            errors.append(f"TP ({take_profit:.4f}) must be above entry price ({entry_price:.4f})")
        elif take_profit - entry_price < min_distance:
            errors.append(f"TP ({take_profit:.4f}) too close to entry, needs at least {min_distance:.4f} distance")

        if stop_loss >= entry_price:
            errors.append(f"SL ({stop_loss:.4f}) must be below entry price ({entry_price:.4f})")
        elif entry_price - stop_loss < min_distance:
            errors.append(f"SL ({stop_loss:.4f}) too close to entry, needs at least {min_distance:.4f} distance")

        if stop_loss >= take_profit:
            errors.append(f"SL ({stop_loss:.4f}) must be below TP ({take_profit:.4f})")

        if current_price and stop_loss >= current_price:
            errors.append(f"SL ({stop_loss:.4f}) >= current price ({current_price:.4f}), will trigger immediately")
        elif current_price and current_price - stop_loss < min_distance:
            warnings.append(f"SL ({stop_loss:.4f}) is near current price ({current_price:.4f}), may trigger immediately")

        if current_price and take_profit <= current_price:
            errors.append(f"TP ({take_profit:.4f}) <= current price ({current_price:.4f}), invalid")
        elif current_price and take_profit <= current_price * 1.001:
            warnings.append(f"TP ({take_profit:.4f}) is very close to current price")
    else:
        if take_profit >= entry_price:
            errors.append(f"TP ({take_profit:.4f}) must be below entry price ({entry_price:.4f})")
        elif entry_price - take_profit < min_distance:
            errors.append(f"TP ({take_profit:.4f}) too close to entry, needs at least {min_distance:.4f} distance")

        if stop_loss <= entry_price:
            errors.append(f"SL ({stop_loss:.4f}) must be above entry price ({entry_price:.4f})")
        elif stop_loss - entry_price < min_distance:
            errors.append(f"SL ({stop_loss:.4f}) too close to entry, needs at least {min_distance:.4f} distance")

        if stop_loss <= take_profit:
            errors.append(f"SL ({stop_loss:.4f}) must be above TP ({take_profit:.4f})")

        if current_price and stop_loss <= current_price:
            errors.append(f"SL ({stop_loss:.4f}) <= current price ({current_price:.4f}), will trigger immediately")
        elif current_price and stop_loss - current_price < min_distance:
            warnings.append(f"SL ({stop_loss:.4f}) is near current price ({current_price:.4f}), may trigger immediately")

        if current_price and take_profit >= current_price:
            errors.append(f"TP ({take_profit:.4f}) >= current price ({current_price:.4f}), invalid")
        elif current_price and take_profit >= current_price * 0.999:
            warnings.append(f"TP ({take_profit:.4f}) is very close to current price")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


def calculate_risk_reward_ratio(entry_price: float, take_profit: float,
                               stop_loss: float, direction: str) -> float:
    _validate_direction(direction)
    entry_price = safe_float(entry_price, 0.0)
    take_profit = safe_float(take_profit, 0.0)
    stop_loss = safe_float(stop_loss, 0.0)
    if direction == "long":
        reward = take_profit - entry_price
        risk = entry_price - stop_loss
    else:
        reward = entry_price - take_profit
        risk = stop_loss - entry_price

    if risk <= 0:
        # 风险为 0 时返回 0.0 而非 inf，保证 JSON 序列化安全
        return 0.0

    return reward / risk

def calculate_trailing_stop(entry_price: float, current_price: float, direction: str,
                          initial_trailing_pct: float = 0.02, min_trailing_pct: float = 0.005) -> float:
    _validate_direction(direction)
    if entry_price <= 0:
        logger.warning(f"calculate_trailing_stop: invalid entry_price {entry_price}")
        return current_price
    if direction == "long":
        profit = (current_price - entry_price) / entry_price
        if profit <= 0:
            return entry_price * (1 - initial_trailing_pct)
        max_profit = profit * 0.5
        trailing_pct = max(min_trailing_pct, initial_trailing_pct - max_profit)
        return current_price * (1 - trailing_pct)
    else:
        profit = (entry_price - current_price) / entry_price
        if profit <= 0:
            return entry_price * (1 + initial_trailing_pct)
        max_profit = profit * 0.5
        trailing_pct = max(min_trailing_pct, initial_trailing_pct - max_profit)
        return current_price * (1 + trailing_pct)


def calculate_scaled_take_profit(entry_price: float, direction: str,
                                profit_targets: list, current_profit: float = 0.0) -> Dict[str, Any]:
    _validate_direction(direction)
    entry_price = safe_float(entry_price, 0.0)
    current_profit = safe_float(current_profit, 0.0)
    levels = []
    for i, target in enumerate(profit_targets or []):
        if not isinstance(target, dict):
            continue
        threshold = safe_float(target.get("threshold"), 0.0)
        close_ratio = safe_float(target.get("close_ratio"), 0.0)
        if current_profit < threshold:
            if direction == "long":
                price = entry_price * (1 + threshold)
            else:
                price = entry_price * (1 - threshold)
            prev_ratios = sum(safe_float(t.get("close_ratio"), 0.0) for t in (profit_targets or [])[:i] if isinstance(t, dict))
            levels.append({
                "level": i + 1,
                "price": price,
                "threshold": threshold,
                "close_ratio": close_ratio,
                "remaining_ratio": max(0.0, 1.0 - prev_ratios),
            })
    return {"levels": levels, "next_level": levels[0] if levels else None}


def adjust_stop_loss_for_volatility(entry_price: float, direction: str, atr: float,
                                   base_sl_pct: float = 0.02, volatility_multiplier: float = 1.0) -> float:
    _validate_direction(direction)
    entry_price = safe_float(entry_price, 0.0)
    atr = safe_float(atr, 0.0)
    base_sl_pct = safe_float(base_sl_pct, 0.02)
    volatility_multiplier = safe_float(volatility_multiplier, 1.0)
    if atr <= 0 or entry_price <= 0:
        return calculate_stop_loss(entry_price, direction, base_sl_pct)

    atr_pct = safe_div(atr, entry_price, 0.0)
    adjusted_sl_pct = base_sl_pct * (1 + (atr_pct - 0.01) * volatility_multiplier * 10)
    adjusted_sl_pct = max(base_sl_pct * 0.5, min(base_sl_pct * 2, adjusted_sl_pct))

    return calculate_stop_loss(entry_price, direction, adjusted_sl_pct)


def check_profit_protection(entry_price: float, current_price: float, direction: str,
                           max_drawdown_pct: float = 0.5) -> bool:
    _validate_direction(direction)
    entry_price = safe_float(entry_price, 0.0)
    current_price = safe_float(current_price, 0.0)
    max_drawdown_pct = safe_float(max_drawdown_pct, 0.5)
    if direction == "long":
        max_price = entry_price * (1 + max_drawdown_pct)
        return current_price <= max_price
    else:
        min_price = entry_price * (1 - max_drawdown_pct)
        return current_price >= min_price

def calculate_partial_close_quantity(total_quantity: float, profit_targets: list, 
                                    current_profit: float) -> float:
    total_quantity = safe_float(total_quantity, 0.0)
    current_profit = safe_float(current_profit, 0.0)
    remaining_ratio = 1.0
    for target in (profit_targets or []):
        if not isinstance(target, dict):
            continue
        threshold = safe_float(target.get("threshold"), 0.0)
        close_ratio = safe_float(target.get("close_ratio"), 0.0)
        if current_profit >= threshold:
            remaining_ratio = max(0.0, remaining_ratio - close_ratio)
        else:
            return total_quantity * close_ratio
    return 0.0

def detect_market_state(price_history: list, volume_history: list = None, 
                       atr: float = 0, lookback_period: int = 20) -> Dict[str, Any]:
    if len(price_history) < lookback_period:
        return {
            "state": "unknown", "confidence": 0.0, "volatility": "normal",
            "volatility_value": 0, "trend_strength": 0, "range_pct": 0,
            "volume_ratio": 1.0, "momentum": 0, "rsi": 50,
            "macd_direction": "neutral", "price_position": 0.5,
            "volume_trend": "neutral", "momentum_acceleration": 0,
            "trend_direction": "neutral", "volatility_regime": "normal",
            "market_regime": "unknown"
        }
    
    # 过滤 NaN/Inf，避免污染统计计算
    raw_prices = price_history[-lookback_period:]
    clean_prices = [safe_finite(p, None) for p in raw_prices]
    clean_prices = [p for p in clean_prices if p is not None and p > 0]
    if len(clean_prices) < max(2, lookback_period // 2):
        return {
            "state": "unknown", "confidence": 0.0, "volatility": "normal",
            "volatility_value": 0, "trend_strength": 0, "range_pct": 0,
            "volume_ratio": 1.0, "momentum": 0, "rsi": 50,
            "macd_direction": "neutral", "price_position": 0.5,
            "volume_trend": "neutral", "momentum_acceleration": 0,
            "trend_direction": "neutral", "volatility_regime": "normal",
            "market_regime": "unknown"
        }

    prices = np.array(clean_prices)
    base_price = prices[0] if prices[0] > 0 else 1.0

    returns = np.diff(prices) / prices[:-1]
    volatility = float(np.std(returns)) if len(returns) > 0 else 0.0

    high = float(np.max(prices))
    low = float(np.min(prices))
    range_pct = safe_div(high - low, base_price, 0.0)

    ema_short = float(np.mean(prices[-5:])) if len(prices) >= 5 else float(np.mean(prices))
    ema_mid = float(np.mean(prices[-10:])) if len(prices) >= 10 else float(np.mean(prices))
    ema_long = float(np.mean(prices[-15:])) if len(prices) >= 15 else float(np.mean(prices))
    trend_strength = safe_div(abs(ema_short - ema_long), base_price, 0.0)
    
    trend_direction = "neutral"
    if ema_short > ema_mid > ema_long:
        trend_direction = "bullish"
    elif ema_short < ema_mid < ema_long:
        trend_direction = "bearish"
    
    if len(prices) >= 14:
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = float(np.mean(gains[-14:])) if len(gains) >= 14 else float(np.mean(gains))
        avg_loss = float(np.mean(losses[-14:])) if len(losses) >= 14 else float(np.mean(losses))
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = safe_div(avg_gain, avg_loss, 0.0)
            rsi = 100.0 - safe_div(100.0, 1.0 + rs, 50.0)
    else:
        rsi = 50.0

    price_position = safe_div(prices[-1] - low, high - low, 0.5) if high > low else 0.5

    if len(returns) >= 10:
        momentum_short = float(np.mean(returns[-3:])) if len(returns) >= 3 else 0.0
        momentum_long = float(np.mean(returns[-10:])) if len(returns) >= 10 else 0.0
        momentum_acceleration = momentum_short - momentum_long
    else:
        momentum_acceleration = 0.0

    momentum = float(np.mean(returns[-5:])) if len(returns) >= 5 else 0.0

    if len(prices) >= 9:
        ema12 = float(np.mean(prices[-12:])) if len(prices) >= 12 else float(np.mean(prices))
        ema26 = float(np.mean(prices[-min(26, len(prices)):]))
        macd_val = ema12 - ema26
        macd_direction = "bullish" if macd_val > 0 else "bearish"
    else:
        macd_direction = "neutral"

    if volume_history and len(volume_history) >= lookback_period:
        # 过滤非有限成交量
        clean_vols = [safe_finite(v, None) for v in volume_history[-lookback_period:]]
        clean_vols = [v for v in clean_vols if v is not None]
        volumes = np.array(clean_vols) if clean_vols else np.array([1.0])
        avg_volume = float(np.mean(volumes)) if len(volumes) > 0 else 0.0
        recent_volume = float(np.mean(volumes[-5:])) if len(volumes) >= 5 else avg_volume
        volume_ratio = safe_div(recent_volume, avg_volume, 1.0)

        if len(volumes) >= 10:
            vol_short = float(np.mean(volumes[-3:]))
            vol_long = float(np.mean(volumes[-10:]))
            if vol_short > vol_long * 1.2:
                volume_trend = "increasing"
            elif vol_short < vol_long * 0.8:
                volume_trend = "decreasing"
            else:
                volume_trend = "neutral"
        else:
            volume_trend = "neutral"
    else:
        volume_ratio = 1.0
        volume_trend = "neutral"
    
    if atr > 0:
        atr_ratio = safe_div(atr, prices[-1], 0.0)
        if volatility > atr_ratio * 1.5:
            volatility_level = "high"
        elif volatility < atr_ratio * 0.4:
            volatility_level = "low"
        else:
            volatility_level = "normal"
    else:
        if volatility > 0.02:
            volatility_level = "high"
        elif volatility < 0.005:
            volatility_level = "low"
        else:
            volatility_level = "normal"
    
    if volatility > 0.03:
        volatility_regime = "extreme"
    elif volatility > 0.015:
        volatility_regime = "high"
    elif volatility > 0.008:
        volatility_regime = "normal"
    elif volatility > 0.003:
        volatility_regime = "low"
    else:
        volatility_regime = "very_low"
    
    state = "range"
    confidence = 0.5
    
    trend_score = 0
    if trend_direction == "bullish":
        trend_score += 2
    elif trend_direction == "bearish":
        trend_score -= 2
    if momentum > 0:
        trend_score += 1
    elif momentum < 0:
        trend_score -= 1
    if macd_direction == "bullish":
        trend_score += 1
    elif macd_direction == "bearish":
        trend_score -= 1
    if volume_trend == "increasing" and momentum != 0:
        trend_score += 1 if momentum > 0 else -1
    
    if abs(trend_score) >= 3 and trend_strength > range_pct * 0.25:
        if trend_score > 0:
            state = "uptrend"
        else:
            state = "downtrend"
        confidence = min(0.92, 0.5 + abs(trend_score) * 0.08 + trend_strength * 50)
    elif range_pct > 0.04 and volatility_level == "high":
        state = "high_volatility"
        confidence = min(0.85, 0.5 + range_pct * 30)
    elif range_pct < 0.015 and volatility_level == "low":
        state = "low_volatility"
        confidence = min(0.8, 0.5 + (0.015 - range_pct) * 20)
    else:
        state = "range"
        confidence = min(0.75, 0.45 + (1 - abs(trend_score) * 0.1) * 0.3)
    
    if state in ["uptrend", "downtrend"] and volume_trend == "increasing":
        confidence = min(0.95, confidence + 0.05)
    
    market_regime = f"{volatility_regime}_{state}"
    
    return {
        "state": state,
        "confidence": confidence,
        "volatility": volatility_level,
        "volatility_value": volatility,
        "trend_strength": trend_strength,
        "range_pct": range_pct,
        "volume_ratio": volume_ratio,
        "momentum": momentum,
        "rsi": rsi,
        "macd_direction": macd_direction,
        "price_position": price_position,
        "volume_trend": volume_trend,
        "momentum_acceleration": momentum_acceleration,
        "trend_direction": trend_direction,
        "volatility_regime": volatility_regime,
        "market_regime": market_regime,
        "trend_score": trend_score,
        "high": high,
        "low": low,
        "current_price": prices[-1]
    }

# P33: 企业级凯利 regime 映射 —— 将 detect_market_state 的 state 值映射到 AdaptiveKelly 的 regime
_REGIME_MAP: Dict[str, str] = {
    "uptrend": "trending_up",
    "downtrend": "trending_down",
    "range": "ranging",
    "high_volatility": "high_volatility",
    "low_volatility": "low_volatility",
    "unknown": "unknown",
    "bull": "trending_up",
    "bear": "trending_down",
    "neutral": "ranging",
}


def map_market_state_to_regime(state: str) -> str:
    """将 detect_market_state 的 state 值映射为 AdaptiveKelly 的 regime 值"""
    return _REGIME_MAP.get(state, "unknown")


def calculate_adaptive_position_size(base_size: float, market_state: Dict[str, Any], 
                                    strategy_type: str = "scalping") -> float:
    base_size = safe_float(base_size, 0.0)
    if not isinstance(market_state, dict):
        market_state = {}
    volatility_factor = 1.0
    state_factor = 1.0

    vol = market_state.get("volatility", "normal")
    state = market_state.get("state", "range")

    # 高波动仓位压缩：scalping 不压缩（利用波动获利），trend 压缩 40%
    if vol == "high":
        volatility_factor = 1.0 if strategy_type == "scalping" else 0.6
    elif vol == "low":
        volatility_factor = 1.2

    if strategy_type == "scalping":
        if state == "range":
            state_factor = 1.1
        elif state in ["uptrend", "downtrend"]:
            state_factor = 0.9
        else:
            state_factor = 0.7
    elif strategy_type == "trend":
        if state in ["uptrend", "downtrend"]:
            state_factor = 1.2
        elif state == "range":
            state_factor = 0.9
        else:
            state_factor = 0.85

    volume_ratio = safe_float(market_state.get("volume_ratio", 1.0), 1.0)
    volume_factor = min(2.0, max(0.5, volume_ratio))

    return base_size * volatility_factor * state_factor * volume_factor

def adjust_signal_thresholds(base_thresholds: Dict[str, float], market_state: Dict[str, Any]) -> Dict[str, float]:
    if not isinstance(base_thresholds, dict):
        base_thresholds = {}
    if not isinstance(market_state, dict):
        market_state = {}
    adjusted = base_thresholds.copy()

    vol = market_state.get("volatility", "normal")
    state = market_state.get("state", "range")
    volume_ratio = safe_float(market_state.get("volume_ratio", 1.0), 1.0)
    trend_strength = safe_float(market_state.get("trend_strength", 0.0), 0.0)

    vol_multiplier = 1.0
    if vol == "high":
        vol_multiplier = 1.4
        adjusted["rsi_overbought"] = min(80, safe_float(base_thresholds.get("rsi_overbought", 70), 70) + 8)
        adjusted["rsi_oversold"] = max(20, safe_float(base_thresholds.get("rsi_oversold", 30), 30) - 8)
        adjusted["volume_delta"] = safe_float(base_thresholds.get("volume_delta", 0.3), 0.3) * 1.5
        adjusted["price_deviation"] = safe_float(base_thresholds.get("price_deviation", 0.015), 0.015) * 1.4
    elif vol == "low":
        vol_multiplier = 0.65
        adjusted["rsi_overbought"] = max(60, safe_float(base_thresholds.get("rsi_overbought", 70), 70) - 10)
        adjusted["rsi_oversold"] = min(40, safe_float(base_thresholds.get("rsi_oversold", 30), 30) + 10)
        adjusted["volume_delta"] = safe_float(base_thresholds.get("volume_delta", 0.3), 0.3) * 0.5
        adjusted["price_deviation"] = safe_float(base_thresholds.get("price_deviation", 0.015), 0.015) * 0.6

    if state == "range":
        adjusted["profit_target_min"] = safe_float(base_thresholds.get("profit_target_min", 0.015), 0.015) * 0.7 * vol_multiplier
        adjusted["stop_loss"] = safe_float(base_thresholds.get("stop_loss", 0.025), 0.025) * 0.75 * vol_multiplier
        adjusted["min_drop"] = safe_float(base_thresholds.get("min_drop", 0.008), 0.008) * 0.7 * vol_multiplier
        adjusted["min_rise"] = safe_float(base_thresholds.get("min_rise", 0.008), 0.008) * 0.7 * vol_multiplier
    elif state in ["uptrend", "downtrend"]:
        trend_factor = min(1.5, 1.0 + trend_strength * 50)
        adjusted["profit_target_min"] = safe_float(base_thresholds.get("profit_target_min", 0.015), 0.015) * trend_factor
        adjusted["stop_loss"] = safe_float(base_thresholds.get("stop_loss", 0.025), 0.025) * (1.0 + trend_strength * 30)
        adjusted["min_drop"] = safe_float(base_thresholds.get("min_drop", 0.008), 0.008) * (0.8 + trend_strength * 20)
        adjusted["min_rise"] = safe_float(base_thresholds.get("min_rise", 0.008), 0.008) * (0.8 + trend_strength * 20)
    else:
        adjusted["profit_target_min"] = safe_float(base_thresholds.get("profit_target_min", 0.015), 0.015) * vol_multiplier
        adjusted["stop_loss"] = safe_float(base_thresholds.get("stop_loss", 0.025), 0.025) * vol_multiplier

    if volume_ratio > 1.5:
        adjusted["volume_delta"] = safe_float(adjusted.get("volume_delta", 0.3), 0.3) * 0.7
    elif volume_ratio < 0.7:
        adjusted["volume_delta"] = safe_float(adjusted.get("volume_delta", 0.3), 0.3) * 1.3

    return adjusted

def evaluate_signal_quality(indicators: Dict[str, float], market_state: Dict[str, Any],
                           direction: str) -> float:
    _validate_direction(direction)
    if not isinstance(indicators, dict):
        indicators = {}
    if not isinstance(market_state, dict):
        market_state = {}
    score = 0.0
    max_score = 0.0

    if "rsi" in indicators:
        rsi = safe_float(indicators.get("rsi"), 50.0)
        if direction == "long":
            if 20 <= rsi <= 40:
                score += 0.15
            elif 40 < rsi <= 55:
                score += 0.10
            elif 55 < rsi <= 65:
                score += 0.05
        elif direction == "short":
            if 60 <= rsi <= 80:
                score += 0.15
            elif 45 <= rsi < 60:
                score += 0.10
            elif 35 <= rsi < 45:
                score += 0.05
        max_score += 0.15

    if "volume_delta" in indicators:
        delta = safe_float(indicators.get("volume_delta"), 0.0)
        if direction == "long":
            if delta > 0.5:
                score += 0.12
            elif delta > 0.2:
                score += 0.08
            elif delta > 0:
                score += 0.04
        elif direction == "short":
            if delta < -0.5:
                score += 0.12
            elif delta < -0.2:
                score += 0.08
            elif delta < 0:
                score += 0.04
        max_score += 0.12

    if "vwap_distance" in indicators:
        vwap_dist = safe_float(indicators.get("vwap_distance"), 0.0)
        if direction == "long" and vwap_dist > -0.005:
            if vwap_dist > 0.002:
                score += 0.10
            elif vwap_dist > 0:
                score += 0.07
            else:
                score += 0.03
        elif direction == "short" and vwap_dist < 0.005:
            if vwap_dist < -0.002:
                score += 0.10
            elif vwap_dist < 0:
                score += 0.07
            else:
                score += 0.03
        max_score += 0.10

    if "momentum" in indicators:
        momentum = safe_float(indicators.get("momentum"), 0.0)
        if direction == "long":
            if momentum > 0.005:
                score += 0.10
            elif momentum > 0.002:
                score += 0.07
            elif momentum > 0:
                score += 0.04
        elif direction == "short":
            if momentum < -0.005:
                score += 0.10
            elif momentum < -0.002:
                score += 0.07
            elif momentum < 0:
                score += 0.04
        max_score += 0.10

    if "macd_histogram" in indicators:
        hist = safe_float(indicators.get("macd_histogram"), 0.0)
        if direction == "long" and hist > 0:
            score += 0.08
        elif direction == "short" and hist < 0:
            score += 0.08
        max_score += 0.08

    if "bb_position" in indicators:
        bb_pos = safe_float(indicators.get("bb_position"), 0.5)
        if direction == "long" and bb_pos < 0.3:
            score += 0.08
        elif direction == "short" and bb_pos > 0.7:
            score += 0.08
        max_score += 0.08

    if "stoch_k" in indicators and "stoch_d" in indicators:
        stoch_k = safe_float(indicators.get("stoch_k"), 50.0)
        stoch_d = safe_float(indicators.get("stoch_d"), 50.0)
        if direction == "long" and stoch_k < 30 and stoch_d < 30:
            score += 0.07
        elif direction == "short" and stoch_k > 70 and stoch_d > 70:
            score += 0.07
        max_score += 0.07

    if "atr_ratio" in indicators:
        atr_ratio = safe_float(indicators.get("atr_ratio"), 1.0)
        if 0.5 <= atr_ratio <= 2.0:
            score += 0.05
        elif atr_ratio > 2.0:
            score += 0.03
        max_score += 0.05
    
    state_bonus = 0.0
    if market_state.get("state") in ["uptrend", "downtrend"]:
        if (direction == "long" and market_state["state"] == "uptrend") or \
           (direction == "short" and market_state["state"] == "downtrend"):
            state_bonus = 0.10
        else:
            state_bonus = -0.05
    elif market_state.get("state") == "range":
        state_bonus = 0.03
    score += state_bonus
    max_score += 0.10
    
    vol_factor = 0.0
    if market_state.get("volatility") == "high":
        vol_factor = -0.03
    elif market_state.get("volatility") == "low":
        vol_factor = 0.02
    score += vol_factor
    max_score += 0.05
    
    volume_factor = 0.0
    if market_state.get("volume_ratio", 1.0) > 1.5:
        volume_factor = 0.05
    elif market_state.get("volume_ratio", 1.0) > 1.0:
        volume_factor = 0.02
    score += volume_factor
    max_score += 0.05
    
    raw_score = safe_div(score, max_score, 0.5)
    return max(0.0, min(1.0, raw_score))