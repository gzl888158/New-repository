"""决策校验器：基于风控限额、合约规格与凯利准则校验交易决策的合法性。

.. deprecated:: 实验性模块，未接入生产交易链路。
"""
import time
from collections import deque
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from loguru import logger


class ValidationResult(Enum):
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"


class ValidationError:
    def __init__(self, code: str, message: str, severity: str = "error", field: str = ""):
        self.code = code
        self.message = message
        self.severity = severity
        self.field = field

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "field": self.field,
        }


class DecisionValidator:
    def __init__(self, config=None, okx_client=None):
        self.config = config or {}
        self._okx_client = okx_client
        self._validators = []
        self._risk_limits = {
            "max_position_size": 1000.0,
            "max_leverage": 20,
            "max_exposure": 1.0,
            "min_confidence": 0.15,
            "max_daily_loss": 0.05,
            "kelly_fraction_cap": 0.25,       # Kelly仓位上限（资本占比）
            "max_streak_count": 8,            # R129: 4→8 连续同向信号限制放宽
            "max_churn_per_hour": 30,         # P7: 每小时最大换手次数（10→30，网格策略19个币种10次太低）
        }
        # 硬编码合约规格作为回退（当OKX API不可用时使用）
        self._contract_specs = {
            "BTC": {"min_qty": 0.001, "tick_size": 0.1, "lot_size": 0.001, "min_notional": 5.0},  # R66: 10→5
            "ETH": {"min_qty": 0.01, "tick_size": 0.01, "lot_size": 0.01, "min_notional": 5.0},  # R66: 10→5
            "SOL": {"min_qty": 0.1, "tick_size": 0.01, "lot_size": 0.1, "min_notional": 2.0},  # R66: 5→2
            "BNB": {"min_qty": 0.01, "tick_size": 0.1, "lot_size": 0.01, "min_notional": 5.0},  # R66: 10→5
            "XRP": {"min_qty": 1.0, "tick_size": 0.0001, "lot_size": 1.0, "min_notional": 2.0},
            "ADA": {"min_qty": 1.0, "tick_size": 0.0001, "lot_size": 1.0, "min_notional": 2.0},
            "DOGE": {"min_qty": 10.0, "tick_size": 0.00001, "lot_size": 10.0, "min_notional": 2.0},
            "LTC": {"min_qty": 0.1, "tick_size": 0.01, "lot_size": 0.1, "min_notional": 2.0},  # R66: 5→2
            "DOT": {"min_qty": 0.1, "tick_size": 0.001, "lot_size": 0.1, "min_notional": 2.0},  # R66: 5→2
            "LINK": {"min_qty": 0.1, "tick_size": 0.001, "lot_size": 0.1, "min_notional": 2.0},  # R66: 5→2
            "UNI": {"min_qty": 0.1, "tick_size": 0.001, "lot_size": 0.1, "min_notional": 2.0},  # R66: 5→2
            "AVAX": {"min_qty": 0.1, "tick_size": 0.01, "lot_size": 0.1, "min_notional": 2.0},  # R66: 5→2
            "ARB": {"min_qty": 1.0, "tick_size": 0.0001, "lot_size": 1.0, "min_notional": 2.0},
        }
        # 动态合约规格缓存（从OKX API获取，优先于硬编码值）
        self._dynamic_specs: Dict[str, Dict[str, float]] = {}
        self._dynamic_specs_ts: Dict[str, float] = {}
        self._dynamic_specs_ttl = 1800  # 缓存30分钟
        # Circuit breaker state
        self.circuit_breaker_triggered = False
        self._circuit_breaker_reason = ""
        # Pre-validation hooks
        self._pre_validation_hooks: List[Callable] = []
        # Validation latency tracking
        self._validation_latency_history: deque = deque(maxlen=100)
        # ── Kelly准则状态 ──
        self._win_history: deque = deque(maxlen=50)       # 历史胜负 (True/False) 
        self._pnl_history: deque = deque(maxlen=50)        # 历史盈亏值
        self._last_kelly_fraction: Optional[float] = None   # 最近计算的Kelly最优仓位
        # ── 连续同向/换手率检测 ──
        self._streak_direction: Optional[str] = None        # 当前连续方向
        self._streak_count: int = 0                          # 连续同向计数
        self._position_changes: deque = deque(maxlen=100)   # 仓位变更时间戳 (per-symbol)

    def add_validator(self, validator):
        self._validators.append(validator)
        logger.info(f"Added validator: {validator.__name__}")

    def set_risk_limits(self, limits: Dict[str, float]):
        self._risk_limits.update(limits)
        logger.info(f"Updated risk limits: {limits}")

    async def validate(self, decision: Any) -> Tuple[ValidationResult, List[ValidationError]]:
        errors = []
        t0 = time.perf_counter()

        # Execute pre-validation hooks
        for hook in self._pre_validation_hooks:
            try:
                hook_result = hook(decision)
                if hook_result is not None:
                    decision = hook_result
            except Exception as e:
                logger.error(f"Pre-validation hook failed: {e}")

        errors.extend(await self._validate_circuit_breaker(decision))
        if any(e.severity == "error" for e in errors):
            t1 = time.perf_counter()
            self._validation_latency_history.append(t1 - t0)
            return ValidationResult.INVALID, errors

        errors.extend(await self._validate_structure(decision))
        errors.extend(await self._validate_risk(decision))
        errors.extend(await self._validate_market_risk(decision))
        errors.extend(await self._validate_slippage(decision))
        errors.extend(await self._validate_business_rules(decision))
        errors.extend(await self._validate_kelly_sizing(decision))
        errors.extend(await self._validate_streak_and_churn(decision))

        for validator in self._validators:
            try:
                result = await validator(decision)
                if result:
                    errors.extend(result)
            except Exception as e:
                logger.error(f"Custom validator failed: {e}")

        t1 = time.perf_counter()
        self._validation_latency_history.append(t1 - t0)

        if any(e.severity == "error" for e in errors):
            return ValidationResult.INVALID, errors
        elif errors:
            return ValidationResult.WARNING, errors
        else:
            return ValidationResult.VALID, []

    async def _validate_structure(self, decision: Any) -> List[ValidationError]:
        errors = []

        if not hasattr(decision, "decision_type"):
            errors.append(ValidationError(
                code="MISSING_TYPE",
                message="Decision type is required",
                severity="error",
                field="decision_type"
            ))

        if not hasattr(decision, "data") or not isinstance(decision.data, dict):
            errors.append(ValidationError(
                code="INVALID_DATA",
                message="Decision data must be a dictionary",
                severity="error",
                field="data"
            ))
        else:
            data = decision.data

            if decision.decision_type.value == "signal":
                required_fields = ["symbol", "direction", "quantity"]
                for field in required_fields:
                    if field not in data:
                        errors.append(ValidationError(
                            code=f"MISSING_FIELD_{field.upper()}",
                            message=f"Signal must contain '{field}'",
                            severity="error",
                            field=f"data.{field}"
                        ))

                direction = data.get("direction", "").lower()
                if direction not in ("buy", "sell", "long", "short"):
                    errors.append(ValidationError(
                        code="INVALID_DIRECTION",
                        message=f"Invalid direction: {direction}",
                        severity="error",
                        field="data.direction"
                    ))

                if "quantity" in data and float(data["quantity"]) <= 0:
                    errors.append(ValidationError(
                        code="INVALID_QUANTITY",
                        message="Quantity must be positive",
                        severity="error",
                        field="data.quantity"
                    ))

            elif decision.decision_type.value == "order":
                required_fields = ["symbol", "side", "order_type"]
                for field in required_fields:
                    if field not in data:
                        errors.append(ValidationError(
                            code=f"MISSING_FIELD_{field.upper()}",
                            message=f"Order must contain '{field}'",
                            severity="error",
                            field=f"data.{field}"
                        ))

        if not hasattr(decision, "confidence") or decision.confidence < 0 or decision.confidence > 1:
            errors.append(ValidationError(
                code="INVALID_CONFIDENCE",
                message="Confidence must be between 0 and 1",
                severity="error",
                field="confidence"
            ))

        return errors

    async def _validate_risk(self, decision: Any) -> List[ValidationError]:
        errors = []
        data = getattr(decision, "data", {})

        if isinstance(data, dict):
            position_size = float(data.get("position_size", 0))
            if position_size > self._risk_limits["max_position_size"]:
                errors.append(ValidationError(
                    code="POSITION_SIZE_EXCEEDED",
                    message=f"Position size {position_size:.2f} exceeds maximum {self._risk_limits['max_position_size']}",
                    severity="error",
                    field="data.position_size"
                ))

            leverage = float(data.get("leverage", 1))
            if leverage > self._risk_limits["max_leverage"]:
                errors.append(ValidationError(
                    code="LEVERAGE_EXCEEDED",
                    message=f"Leverage {leverage}x exceeds maximum {self._risk_limits['max_leverage']}x",
                    severity="error",
                    field="data.leverage"
                ))

            exposure = float(data.get("exposure", 0))
            if exposure > self._risk_limits["max_exposure"]:
                errors.append(ValidationError(
                    code="EXPOSURE_EXCEEDED",
                    message=f"Exposure {exposure:.0%} exceeds maximum {self._risk_limits['max_exposure']:.0%}",
                    severity="error",
                    field="data.exposure"
                ))

            confidence = getattr(decision, "confidence", 0)
            if confidence < self._risk_limits["min_confidence"]:
                errors.append(ValidationError(
                    code="LOW_CONFIDENCE",
                    message=f"Confidence {confidence:.2f} below minimum {self._risk_limits['min_confidence']}",
                    severity="warning",
                    field="confidence"
                ))

        return errors

    def _resolve_contract_spec(self, base_symbol: str, symbol: str) -> Dict[str, float]:
        """解析合约规格：优先使用动态获取的规格，回退到硬编码值"""
        # 检查动态缓存
        now = time.time()
        if base_symbol in self._dynamic_specs:
            if now - self._dynamic_specs_ts.get(base_symbol, 0) < self._dynamic_specs_ttl:
                return self._dynamic_specs[base_symbol]
        return self._contract_specs.get(base_symbol, {})

    async def _fetch_contract_specs_dynamic(self, symbol: str) -> Optional[Dict[str, float]]:
        """从OKX API动态获取合约规格并缓存"""
        if not self._okx_client:
            return None

        base_symbol = symbol.split("-")[0] if "-" in symbol else symbol
        full_symbol = f"{base_symbol}-USDT-SWAP"

        try:
            info = await self._okx_client.get_instrument_info_async(full_symbol)
            if not info:
                return None

            ct_val = float(info.get("ctVal", "1"))
            min_sz = float(info.get("minSz", "0.001"))
            tick_sz = float(info.get("tickSz", "0.1"))
            lot_sz = float(info.get("lotSz", "0.001"))
            min_notional = max(min_sz * ct_val, 1.0)

            specs = {
                "min_qty": min_sz,
                "tick_size": tick_sz,
                "lot_size": lot_sz,
                "min_notional": min_notional,
                "ct_val": ct_val,
            }

            self._dynamic_specs[base_symbol] = specs
            self._dynamic_specs_ts[base_symbol] = time.time()
            logger.debug(f"Dynamic contract specs for {base_symbol}: min_qty={min_sz}, tick={tick_sz}, lot={lot_sz}")
            return specs
        except Exception as e:
            logger.debug(f"Failed to fetch dynamic specs for {full_symbol}: {e}")
            return None

    async def _validate_business_rules(self, decision: Any) -> List[ValidationError]:
        errors = []
        data = getattr(decision, "data", {})

        if isinstance(data, dict):
            symbol = data.get("symbol", "")
            if symbol and not symbol.endswith("-SWAP") and not symbol.endswith("-USDT"):
                errors.append(ValidationError(
                    code="UNSUPPORTED_SYMBOL",
                    message=f"Unsupported symbol format: {symbol}",
                    severity="warning",
                    field="data.symbol"
                ))

            price = float(data.get("price", 0))
            if price <= 0:
                errors.append(ValidationError(
                    code="INVALID_PRICE",
                    message="Price must be positive",
                    severity="error",
                    field="data.price"
                ))

            # 动态获取合约规格：优先OKX API，回退硬编码
            base_symbol = symbol.replace("-USDT-SWAP", "").replace("-USDT", "")
            
            # 尝试从OKX API动态获取（异步非阻塞，失败时使用缓存或回退）
            if self._okx_client and base_symbol not in self._dynamic_specs:
                try:
                    await self._fetch_contract_specs_dynamic(symbol)
                except Exception:
                    pass
            
            spec = self._resolve_contract_spec(base_symbol, symbol)
            if spec:
                qty = float(data.get("quantity", 0))
                if spec.get("min_qty") and qty < spec["min_qty"]:
                    errors.append(ValidationError(
                        code="QUANTITY_TOO_SMALL",
                        message=f"Quantity {qty} below minimum {spec['min_qty']} for {symbol}",
                        severity="error",
                        field="data.quantity"
                    ))

                tick_size = spec.get("tick_size", 0)
                # P15: 使用Decimal精确计算，避免浮点数取模精度问题
                # 与okx_client.round_price_to_tick保持一致
                if tick_size > 0:
                    from decimal import Decimal, ROUND_HALF_UP
                    d_price = Decimal(str(price))
                    d_tick = Decimal(str(tick_size))
                    rounded = (d_price / d_tick).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * d_tick
                    aligned_price = float(rounded)
                    if aligned_price != price:
                        # 价格未对齐，自动修复而非报错
                        logger.info(
                            f"P15: Auto-correcting price {price} to tick-aligned {aligned_price} "
                            f"(tick_size={tick_size}) for {symbol}"
                        )
                        # 修改data中的价格为对齐后的值
                        data["price"] = aligned_price
                        price = aligned_price  # 更新局部变量供后续检查使用

                notional = qty * price
                min_notional = spec.get("min_notional", 1.0)
                # P4-4: 小账户动态降低最低名义价值要求
                # 所有策略在小账户中需要更灵活的名义价值
                strategy_name = data.get("strategy_name", "")
                # 趋势/剥头皮允许到0.5 USDT，网格允许到0.8 USDT
                if strategy_name in ("trend", "scalping"):
                    min_notional = min(min_notional, 0.5)
                elif strategy_name == "grid":
                    min_notional = min(min_notional, 0.8)
                if notional < min_notional:
                    errors.append(ValidationError(
                        code="NOTIONAL_TOO_SMALL",
                        message=f"Notional value {notional:.4f} below minimum {min_notional} for {symbol}",
                        severity="error",
                        field="data.quantity"
                    ))

        return errors

    # ═══════════════════════════════════════════════════════
    # Kelly准则仓位校验
    # ═══════════════════════════════════════════════════════

    def record_trade_outcome(self, was_win: bool, pnl: float = 0.0):
        """记录交易结果，用于Kelly准则计算"""
        self._win_history.append(was_win)
        self._pnl_history.append(pnl)

    def _compute_kelly_fraction(self) -> Optional[float]:
        """
        计算Kelly最优仓位比例

        f* = (p * b - q) / b
        p = 胜率, q = 1-p, b = 平均盈利/平均亏损

        Returns:
            Kelly分数（资本占比），或None（数据不足时）
        """
        if len(self._win_history) < 10:
            return None

        wins = sum(1 for w in self._win_history if w)
        losses = len(self._win_history) - wins
        p = wins / len(self._win_history)
        q = losses / len(self._win_history)

        if p == 0:
            return 0.0

        # 计算盈亏比 b = avg_win / abs(avg_loss)
        win_pnls = []
        loss_pnls = []
        for i, was_win in enumerate(self._win_history):
            pnl = self._pnl_history[i] if i < len(self._pnl_history) else 0.0
            if was_win:
                win_pnls.append(pnl)
            elif pnl < 0:
                loss_pnls.append(abs(pnl))

        avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 1.0
        avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 1.0

        if avg_loss < 1e-10:
            return None

        b = avg_win / avg_loss
        if b <= 0:
            return 0.0

        # Kelly公式
        kelly = (p * b - q) / b
        # 半Kelly更安全
        half_kelly = kelly * 0.5
        self._last_kelly_fraction = max(0.0, half_kelly)
        return self._last_kelly_fraction

    async def _validate_kelly_sizing(self, decision: Any) -> List[ValidationError]:
        """
        Kelly准则仓位校验

        验证当前仓位是否超出Kelly最优分数，超出时发出警告。
        适用于 SIGNAL/ORDER 类型决策。
        """
        errors = []
        data = getattr(decision, "data", {})
        if not isinstance(data, dict):
            return errors

        # 仅对开仓信号校验
        direction = data.get("direction", "").lower()
        if direction in ("close",):
            return errors

        kelly_fraction = self._compute_kelly_fraction()
        if kelly_fraction is None:
            return errors  # 数据不足，不校验

        quantity = float(data.get("quantity", 0))
        price = float(data.get("price", 0))
        leverage = float(data.get("leverage", 5.0))
        notional = quantity * price
        margin = notional / leverage if leverage > 0 else notional

        available = self._get_available_capital_simple()
        if available <= 0:
            return errors

        actual_fraction = margin / available

        kelly_cap = self._risk_limits.get("kelly_fraction_cap", 0.25)

        if actual_fraction > kelly_fraction * 1.5:
            errors.append(ValidationError(
                code="KELLY_OVERSIZE",
                message=(
                    f"Position fraction {actual_fraction:.2%} exceeds {kelly_fraction*1.5:.2%} "
                    f"(1.5x Kelly={kelly_fraction:.2%}); consider reducing size"
                ),
                severity="warning",
                field="data.quantity",
            ))
        elif actual_fraction > kelly_cap:
            errors.append(ValidationError(
                code="KELLY_CAP_EXCEEDED",
                message=(
                    f"Position fraction {actual_fraction:.2%} exceeds hard cap {kelly_cap:.2%}"
                ),
                severity="error",
                field="data.quantity",
            ))

        return errors

    # ═══════════════════════════════════════════════════════
    # 连续同向/换手率检测
    # ═══════════════════════════════════════════════════════

    async def _validate_streak_and_churn(self, decision: Any) -> List[ValidationError]:
        """
        检测连续同向信号和换手率异常

        - 连续同向：连续N次相同方向信号 → 可能过拟合/趋势衰竭
        - 换手率：单位时间内仓位变更次数过高 → 过度交易
        """
        errors = []
        data = getattr(decision, "data", {})
        if not isinstance(data, dict):
            return errors

        direction = data.get("direction", "").lower()
        if direction not in ("buy", "sell", "long", "short"):
            return errors

        normalized_dir = "buy" if direction in ("buy", "long") else "sell"
        symbol = data.get("symbol", "")

        # ── 连续同向检测 ──
        if self._streak_direction == normalized_dir:
            self._streak_count += 1
        else:
            self._streak_direction = normalized_dir
            self._streak_count = 1

        max_streak = self._risk_limits.get("max_streak_count", 4)
        if self._streak_count > max_streak:
            errors.append(ValidationError(
                code="CONSECUTIVE_DIRECTION",
                message=(
                    f"Consecutive {normalized_dir} signals: {self._streak_count} "
                    f"(max={max_streak}); possible overfitting or trend exhaustion"
                ),
                severity="warning",
                field="data.direction",
            ))

        # ── 换手率检测 ──
        # P10: 网格策略豁免换手率检查 - 网格策略的多层双向调仓是其设计核心，
        # 频繁调仓不是过度交易而是正常操作，不应计入换手率统计
        strategy_name = getattr(decision, "source", "")
        if "grid" not in strategy_name.lower():
            now = time.time()
            self._position_changes.append(now)

            # 清理1小时前的记录
            cutoff = now - 3600
            recent_changes = [t for t in self._position_changes if t >= cutoff]
            churn_count = len(recent_changes)

            # P7: 自适应换手率阈值 - 基于活跃symbol数动态调整
            active_symbols = getattr(self, '_active_symbols_count', 0)
            adaptive_max_churn = self._risk_limits.get("max_churn_per_hour", 30)
            if active_symbols > 0:
                # 每个活跃symbol允许3次换手/小时（P9: 2→3，网格策略频繁调仓时仍有噪音）
                adaptive_max_churn = max(adaptive_max_churn, active_symbols * 3)
            
            if churn_count > adaptive_max_churn:
                errors.append(ValidationError(
                    code="HIGH_CHURN_RATE",
                    message=(
                        f"Position churn rate {churn_count}/hour exceeds max {adaptive_max_churn}"
                        f"{f' (active_symbols={active_symbols})' if active_symbols > 0 else ''}; "
                        f"possible overtrading"
                    ),
                    severity="warning",
                    field="data.direction",
                ))

        return errors

    def get_kelly_stats(self) -> Dict[str, Any]:
        """获取Kelly准则统计"""
        kelly = self._compute_kelly_fraction()
        win_count = sum(1 for w in self._win_history if w)
        total = len(self._win_history)
        return {
            "kelly_fraction": round(kelly, 4) if kelly is not None else None,
            "win_rate": round(win_count / total, 4) if total > 0 else 0,
            "total_trades": total,
            "streak_direction": self._streak_direction,
            "streak_count": self._streak_count,
            "churn_1h": sum(1 for t in self._position_changes if t >= time.time() - 3600),
        }

    def _get_available_capital_simple(self) -> float:
        """简单获取可用资金（用于Kelly计算）"""
        if self._okx_client:
            try:
                return float(self.config.get("trading", {}).get("total_capital", 1000.0))
            except Exception:
                pass
        return float(self.config.get("trading", {}).get("total_capital", 1000.0))

    async def prefetch_contract_specs(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """预取多个币种的合约规格到缓存"""
        results = {}
        if not self._okx_client:
            return results

        for symbol in symbols:
            try:
                specs = await self._fetch_contract_specs_dynamic(symbol)
                if specs:
                    base = symbol.split("-")[0]
                    results[base] = specs
            except Exception as e:
                logger.debug(f"Prefetch spec failed for {symbol}: {e}")
        
        return results

    def invalidate_spec_cache(self, base_symbol: str = None):
        """清除合约规格缓存（支持全部清除或单个币种）"""
        if base_symbol is None:
            self._dynamic_specs.clear()
            self._dynamic_specs_ts.clear()
            logger.info("All dynamic contract specs cache invalidated")
        else:
            self._dynamic_specs.pop(base_symbol, None)
            self._dynamic_specs_ts.pop(base_symbol, None)
            logger.debug(f"Contract specs cache invalidated for {base_symbol}")

    def set_circuit_breaker_state(self, triggered: bool, reason: str = ""):
        """Set the global circuit breaker state. When triggered, all non-close decisions are rejected."""
        self.circuit_breaker_triggered = triggered
        self._circuit_breaker_reason = reason
        if triggered:
            logger.warning(f"Circuit breaker TRIGGERED: {reason}")
        else:
            logger.info("Circuit breaker RESET")

    def add_pre_validation_hook(self, hook: Callable):
        """Add a pre-validation hook that can modify decision data before validation.

        Hook signature: hook(decision) -> Optional[Any]
        If the hook returns a non-None value, it replaces the decision object.
        """
        self._pre_validation_hooks.append(hook)
        logger.info(f"Pre-validation hook registered: {getattr(hook, '__name__', str(hook))}")

    def get_validation_stats(self) -> Dict[str, Optional[float]]:
        """Return validation latency statistics: avg, p95, max (in seconds)."""
        if not self._validation_latency_history:
            return {"avg": None, "p95": None, "max": None}
        latencies = sorted(self._validation_latency_history)
        n = len(latencies)
        avg = sum(latencies) / n
        p95_idx = max(0, int(n * 0.95) - 1)
        p95 = latencies[p95_idx]
        max_val = latencies[-1]
        return {"avg": round(avg, 6), "p95": round(p95, 6), "max": round(max_val, 6)}

    async def _validate_circuit_breaker(self, decision: Any) -> List[ValidationError]:
        """Reject non-close decisions when the circuit breaker is triggered."""
        errors = []
        if not self.circuit_breaker_triggered:
            return errors

        data = getattr(decision, "data", {})
        if not isinstance(data, dict):
            return errors

        direction = data.get("direction", "").lower()
        if direction == "close":
            return errors

        errors.append(ValidationError(
            code="CIRCUIT_BREAKER_TRIPPED",
            message=f"Circuit breaker active: {self._circuit_breaker_reason}",
            severity="error",
            field="circuit_breaker"
        ))
        return errors

    async def _validate_market_risk(self, decision: Any, correlation_scores: Optional[Dict[str, float]] = None) -> List[ValidationError]:
        """Multi-dimensional market risk validation.

        Checks:
          - Volatility-adjusted position sizing
          - Liquidity (notional vs 24h volume)
          - Correlation risk (high-correlation entry flagging)
        """
        errors = []
        data = getattr(decision, "data", {})
        if not isinstance(data, dict):
            return errors

        # Volatility-adjusted position sizing
        volatility = float(data.get("volatility", 0))
        quantity = float(data.get("quantity", 0))
        symbol = data.get("symbol", "")
        if volatility > 0 and quantity > 0:
            vol_threshold_high = float(data.get("vol_threshold_high", 0.05))
            if vol_threshold_high > 0 and volatility > vol_threshold_high:
                suggested_qty = quantity * (vol_threshold_high / volatility)
                errors.append(ValidationError(
                    code="HIGH_VOLATILITY_REDUCE_SIZE",
                    message=f"Volatility {volatility:.4f} exceeds threshold {vol_threshold_high:.4f}; consider reducing qty from {quantity} to ~{suggested_qty:.4f}",
                    severity="warning",
                    field="data.quantity"
                ))

        # Liquidity check
        volume_24h = float(data.get("volume_24h", 0))
        price = float(data.get("price", 0))
        if volume_24h > 0 and quantity > 0 and price > 0:
            notional = quantity * price
            ratio = notional / volume_24h
            if ratio > 0.01:
                errors.append(ValidationError(
                    code="LOW_LIQUIDITY",
                    message=f"Notional {notional:.2f} is {ratio:.2%} of 24h volume for {symbol}; may cause excessive impact",
                    severity="warning",
                    field="data.quantity"
                ))

        # Correlation risk
        corr_scores = correlation_scores or data.get("correlation_scores", {})
        if corr_scores and isinstance(corr_scores, dict):
            for other_symbol, score in corr_scores.items():
                if abs(score) > 0.85:
                    errors.append(ValidationError(
                        code="HIGH_CORRELATION",
                        message=f"{symbol} has high correlation ({score:.2f}) with {other_symbol}; diversify check recommended",
                        severity="warning",
                        field="data.correlation"
                    ))

        return errors

    async def _validate_slippage(self, decision: Any) -> List[ValidationError]:
        """Estimate pre-execution slippage and warn if expected slippage exceeds 0.3%.

        Uses order size relative to average trade size and current spread.
        """
        errors = []
        data = getattr(decision, "data", {})
        if not isinstance(data, dict):
            return errors

        quantity = float(data.get("quantity", 0))
        price = float(data.get("price", 0))
        if quantity <= 0 or price <= 0:
            return errors

        avg_trade_size = float(data.get("avg_trade_size", 0))
        spread = float(data.get("spread", 0))

        estimated_slippage_pct = 0.0

        # Size-based slippage: larger orders relative to avg trade size slip more
        if avg_trade_size > 0 and quantity > avg_trade_size:
            size_ratio = quantity / avg_trade_size
            estimated_slippage_pct += 0.0005 * size_ratio

        # Spread-based slippage: half-spread as baseline
        if spread > 0 and price > 0:
            estimated_slippage_pct += (spread / price) * 0.5

        if estimated_slippage_pct > 0.003:
            errors.append(ValidationError(
                code="HIGH_SLIPPAGE_RISK",
                message=f"Estimated slippage {estimated_slippage_pct:.4%} exceeds 0.3% threshold",
                severity="warning",
                field="data.quantity"
            ))

        return errors

    def get_validation_summary(self, results: List[Tuple[ValidationResult, List[ValidationError]]]) -> Dict[str, Any]:
        total = len(results)
        valid = sum(1 for r, _ in results if r == ValidationResult.VALID)
        invalid = sum(1 for r, _ in results if r == ValidationResult.INVALID)
        warnings = sum(1 for r, _ in results if r == ValidationResult.WARNING)

        error_counts = {}
        for _, errors in results:
            for error in errors:
                error_counts[error.code] = error_counts.get(error.code, 0) + 1

        return {
            "total_decisions": total,
            "valid": valid,
            "invalid": invalid,
            "warnings": warnings,
            "error_distribution": error_counts,
        }
