"""
输入验证和边界检查 - Input Validation and Boundary Checking

功能：
1. 参数类型验证
2. 数值范围检查
3. 必填字段检查
4. 业务规则验证
"""

from typing import Any, Dict, List, Optional, Callable, Union
from loguru import logger
from datetime import datetime


class ValidationError(Exception):
    """验证错误"""
    
    def __init__(self, field: str, message: str, value: Any = None):
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"Validation error for '{field}': {message}")


class Validator:
    """验证器基类"""
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> bool:
        """验证方法"""
        raise NotImplementedError


class TypeValidator(Validator):
    """类型验证器"""
    
    def __init__(self, expected_type: Union[type, tuple]):
        self.expected_type = expected_type
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> bool:
        if not isinstance(value, self.expected_type):
            raise ValidationError(
                "value",
                f"Expected type {self.expected_type}, got {type(value).__name__}",
                value
            )
        return True


class RangeValidator(Validator):
    """范围验证器"""
    
    def __init__(self, min_val: Any = None, max_val: Any = None, inclusive: bool = True):
        self.min_val = min_val
        self.max_val = max_val
        self.inclusive = inclusive
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> bool:
        if self.min_val is not None:
            if self.inclusive:
                if value < self.min_val:
                    raise ValidationError("value", f"Value {value} is less than minimum {self.min_val}", value)
            else:
                if value <= self.min_val:
                    raise ValidationError("value", f"Value {value} must be greater than {self.min_val}", value)
        
        if self.max_val is not None:
            if self.inclusive:
                if value > self.max_val:
                    raise ValidationError("value", f"Value {value} is greater than maximum {self.max_val}", value)
            else:
                if value >= self.max_val:
                    raise ValidationError("value", f"Value {value} must be less than {self.max_val}", value)
        
        return True


class RequiredValidator(Validator):
    """必填验证器"""
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> bool:
        if value is None:
            raise ValidationError("value", "Value is required but was None")
        if isinstance(value, str) and not value.strip():
            raise ValidationError("value", "Value is required but was empty string")
        return True


class EnumValidator(Validator):
    """枚举验证器"""
    
    def __init__(self, allowed_values: List[Any]):
        self.allowed_values = allowed_values
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> bool:
        if value not in self.allowed_values:
            raise ValidationError(
                "value",
                f"Value {value} not in allowed values: {self.allowed_values}",
                value
            )
        return True


class SymbolValidator(Validator):
    """交易对符号验证器"""
    
    def __init__(self, allowed_prefixes: List[str] = None):
        self.allowed_prefixes = allowed_prefixes or ["BTC", "ETH", "SOL", "ADA", "AVAX", "ARB", "UNI", "LINK", "DOT", "LTC", "XRP", "BNB"]
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> bool:
        if not isinstance(value, str):
            raise ValidationError("symbol", f"Symbol must be string, got {type(value).__name__}", value)
        
        # 检查格式：XXX-USDT-SWAP 或 XXX-USDT
        if not value.endswith("-USDT") and not value.endswith("-USDT-SWAP"):
            raise ValidationError("symbol", f"Invalid symbol format: {value}", value)
        
        # 提取基础货币
        base = value.split("-")[0]
        if base not in self.allowed_prefixes:
            logger.warning(f"Symbol {value} uses non-standard base currency: {base}")
        
        return True


class OrderValidator:
    """订单验证器"""
    
    @staticmethod
    def validate_order_params(
        symbol: str,
        side: str,
        quantity: float,
        price: float = None,
        leverage: int = 1,
        stop_loss: float = None,
        take_profit: float = None
    ) -> Dict[str, Any]:
        """
        验证订单参数
        
        返回：验证后的参数字典
        """
        errors = []
        
        # 验证symbol
        try:
            SymbolValidator().validate(symbol)
        except ValidationError as e:
            errors.append(str(e))
        
        # 验证side
        try:
            EnumValidator(["buy", "sell"]).validate(side.lower())
        except ValidationError as e:
            errors.append(str(e))
        
        # 验证quantity
        try:
            RequiredValidator().validate(quantity)
            TypeValidator((int, float)).validate(quantity)
            RangeValidator(0.001, None).validate(quantity)  # 最小数量
        except ValidationError as e:
            errors.append(str(e))
        
        # 验证price
        if price is not None:
            try:
                TypeValidator((int, float)).validate(price)
                RangeValidator(0.0001, None).validate(price)  # 最小价格
            except ValidationError as e:
                errors.append(str(e))
        
        # 验证leverage
        try:
            TypeValidator(int).validate(leverage)
            RangeValidator(1, 125).validate(leverage)  # OKX最大杠杆
        except ValidationError as e:
            errors.append(str(e))
        
        # 验证stop_loss和take_profit逻辑
        if stop_loss is not None and price is not None:
            if side.lower() == "buy" and stop_loss >= price:
                errors.append(f"Stop loss {stop_loss} must be less than entry price {price} for long position")
            elif side.lower() == "sell" and stop_loss <= price:
                errors.append(f"Stop loss {stop_loss} must be greater than entry price {price} for short position")
        
        if take_profit is not None and price is not None:
            if side.lower() == "buy" and take_profit <= price:
                errors.append(f"Take profit {take_profit} must be greater than entry price {price} for long position")
            elif side.lower() == "sell" and take_profit >= price:
                errors.append(f"Take profit {take_profit} must be less than entry price {price} for short position")
        
        if errors:
            raise ValidationError("order_params", f"Multiple validation errors: {', '.join(errors)}")
        
        return {
            "symbol": symbol,
            "side": side.lower(),
            "quantity": quantity,
            "price": price,
            "leverage": leverage,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }
    
    @staticmethod
    def validate_quantity_against_balance(quantity: float, price: float, leverage: int, available_balance: float) -> bool:
        """验证订单数量是否超过可用余额"""
        required_margin = (quantity * price) / leverage
        
        if required_margin > available_balance:
            raise ValidationError(
                "quantity",
                f"Required margin {required_margin:.2f} USDT exceeds available balance {available_balance:.2f} USDT",
                quantity
            )
        
        return True


class PositionValidator:
    """持仓验证器"""
    
    @staticmethod
    def validate_position_size(position_value: float, total_equity: float, max_position_ratio: float = 0.3) -> bool:
        """验证持仓大小是否超过最大比例"""
        position_ratio = position_value / total_equity if total_equity > 0 else 0
        
        if position_ratio > max_position_ratio:
            logger.warning(f"Position ratio {position_ratio:.2%} exceeds maximum {max_position_ratio:.2%}")
            return False
        
        return True
    
    @staticmethod
    def validate_leverage(leverage: int, max_leverage: int = 20) -> bool:
        """验证杠杆是否超过最大值"""
        if leverage > max_leverage:
            raise ValidationError("leverage", f"Leverage {leverage} exceeds maximum {max_leverage}")
        return True


class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_trading_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """验证交易配置"""
        errors = []
        
        # 必填字段
        required_fields = ["total_capital", "risk_per_trade", "max_total_leverage", "max_drawdown"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # 数值范围检查
        if "risk_per_trade" in config:
            if not (0 < config["risk_per_trade"] <= 0.1):
                errors.append(f"risk_per_trade {config['risk_per_trade']} should be between 0 and 0.1")
        
        if "max_total_leverage" in config:
            if not (1 <= config["max_total_leverage"] <= 125):
                errors.append(f"max_total_leverage {config['max_total_leverage']} should be between 1 and 125")
        
        if "max_drawdown" in config:
            if not (0 < config["max_drawdown"] <= 0.5):
                errors.append(f"max_drawdown {config['max_drawdown']} should be between 0 and 0.5")
        
        # 策略分配总和检查
        allocation_fields = ["scalping_allocation", "trend_allocation", "grid_allocation", "arbitrage_allocation", "spot_grid_allocation", "spot_martingale_allocation"]
        total_allocation = sum(config.get(f, 0) for f in allocation_fields)
        if abs(total_allocation - 1.0) > 0.01:
            errors.append(f"Allocation sum {total_allocation:.4f} should equal 1.0")
        
        if errors:
            raise ValidationError("config", f"Config validation errors: {', '.join(errors)}")
        
        return config


def validate_inputs(**validators):
    """
    输入验证装饰器
    
    用法：
    @validate_inputs(symbol=SymbolValidator(), quantity=RangeValidator(0.001, None))
    def place_order(self, symbol, quantity, ...):
        ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 获取函数签名
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # 验证每个参数
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    try:
                        validator.validate(value)
                    except ValidationError as e:
                        logger.error(f"Validation failed for {param_name}: {e}")
                        raise
            
            return func(*args, **kwargs)
        return wrapper
    return decorator