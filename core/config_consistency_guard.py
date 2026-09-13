"""
配置一致性 + 防回归校验 (ConfigConsistencyGuard)
================================================
配置在启动 / 热更新 / 持久化前，先做两类检查：

1. 一致性（consistency）：跨字段硬约束，例如资金分配总和、资本比例总和、
   告警延迟阈值大小关系、tier min/max 关系等。违反即视为 error（无效配置）。

2. 防回归（anti-regression）：将新配置与一个"已知良好"的基线快照对比，
   若安全关键参数朝更危险方向移动（最大回撤放大、风险敞口/杠杆上升、
   止损放宽、信号质量门槛下调等）则给出 warning，提示人工复核。

设计原则：
- 纯内存、无数据库/网络依赖，便于独立单元测试。
- 只产出 `ConfigGuardReport`，不做阻断（由调用方决定是否拒绝加载）。
- 与 Pydantic 的绝对值边界校验互补：Pydantic 管"是否越界"，本守卫管
  "相对基线是否退化"与"跨字段是否自洽"。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


# 资金分配字段（策略级分配，总和应 ≈ 1.0）
_ALLOCATION_FIELDS = [
    "scalping_allocation", "trend_allocation", "grid_allocation",
    "arbitrage_allocation", "spot_grid_allocation", "spot_martingale_allocation",
]

# 安全关键参数防回归规则：(路径, 更危险方向, 相对容差)
#   direction="increase" 表示数值增大更危险；"decrease" 表示数值减小更危险。
#   相对变化超过容差即给出 warning。
_REGRESSION_RULES: Tuple[Tuple[str, str, float], ...] = (
    ("trading.max_drawdown", "increase", 0.05),
    ("trading.daily_max_loss", "increase", 0.05),
    ("trading.hourly_max_loss", "increase", 0.05),
    ("trading.daily_risk_limit", "increase", 0.05),
    ("trading.risk_per_trade", "increase", 0.10),
    ("trading.max_total_leverage", "increase", 0.05),
    ("trading.max_stop_loss_pct", "increase", 0.05),
    ("trading.max_slippage_pct", "increase", 0.10),
)

# 策略级：止损/止盈放宽 = 更危险；信号质量门槛下调 = 更危险
_STRATEGY_STOP_KEYS = ("stop_loss", "stop_loss_pct", "max_stop_loss_pct")


@dataclass
class ConfigGuardReport:
    """配置校验报告"""
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def merge(self, other: "ConfigGuardReport") -> "ConfigGuardReport":
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.valid = self.valid and other.valid
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


class ConfigConsistencyGuard:
    """配置一致性 + 防回归校验器"""

    def __init__(self, baseline: Optional[Dict[str, Any]] = None):
        self._baseline = baseline or {}

    # ============================================================
    # 对外接口
    # ============================================================

    def validate(self, config: Dict[str, Any], baseline: Optional[Dict[str, Any]] = None) -> ConfigGuardReport:
        """一致性 + 防回归 组合校验"""
        report = self.check_consistency(config)
        report.merge(self.check_regression(config, baseline=baseline))
        return report

    def check_consistency(self, config: Dict[str, Any]) -> ConfigGuardReport:
        """校验跨字段一致性（硬约束）"""
        report = ConfigGuardReport()
        self._check_consistency(config, report)
        return report

    def check_regression(self, new_config: Dict[str, Any], baseline: Optional[Dict[str, Any]] = None) -> ConfigGuardReport:
        """校验相对基线的安全关键参数是否退化"""
        report = ConfigGuardReport()
        base = baseline if baseline is not None else self._baseline
        if base:
            self._check_regression(new_config, base, report)
        return report

    def extract_baseline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """提取防回归比对所需的精简基线快照（仅含安全关键数值字段）。

        用于持久化到磁盘，使重启后仍能做防回归比对；不包含 API 密钥等敏感字段，
        也不含与防回归无关的字段，避免将机密写入基线文件。
        """
        snapshot: Dict[str, Any] = {}
        if not isinstance(config, dict):
            return snapshot

        trading_src = config.get("trading", {}) or {}
        trading_snapshot: Dict[str, Any] = {}
        for path, _direction, _tol in _REGRESSION_RULES:
            if path.startswith("trading."):
                key = path.split(".", 1)[1]
                if key in trading_src and trading_src[key] is not None:
                    trading_snapshot[key] = trading_src[key]
        if trading_snapshot:
            snapshot["trading"] = trading_snapshot

        strategies_src = config.get("strategies", {}) or {}
        strategies_snapshot: Dict[str, Any] = {}
        if isinstance(strategies_src, dict):
            for name, s in strategies_src.items():
                if not isinstance(s, dict):
                    continue
                fields: Dict[str, Any] = {}
                if s.get("min_signal_quality") is not None:
                    fields["min_signal_quality"] = s["min_signal_quality"]
                for k in _STRATEGY_STOP_KEYS:
                    if k in s and s[k] is not None:
                        fields[k] = s[k]
                if fields:
                    strategies_snapshot[name] = fields
        if strategies_snapshot:
            snapshot["strategies"] = strategies_snapshot

        return snapshot

    # ============================================================
    # 一致性检查
    # ============================================================

    def _check_consistency(self, config: Dict[str, Any], report: ConfigGuardReport) -> None:
        if not isinstance(config, dict):
            report.add_error("config 必须是 dict")
            return

        trading = config.get("trading", {}) or {}

        # 1) 策略分配总和 ≈ 1.0
        total_alloc = sum(_to_float(trading.get(k)) for k in _ALLOCATION_FIELDS)
        if abs(total_alloc - 1.0) > 0.01:
            report.add_error(f"策略分配总和 {total_alloc:.4f} 偏离 1.0")

        # 2) 资本比例总和 ≈ 1.0
        capital_sum = (
            _to_float(trading.get("trading_capital_ratio", 0.95))
            + _to_float(trading.get("risk_reserve_ratio", 0.05))
            + _to_float(trading.get("profit_reserve_ratio", 0.0))
        )
        if abs(capital_sum - 1.0) > 0.01:
            report.add_error(f"资本比例总和 {capital_sum:.4f} 偏离 1.0")

        # 3) daily_max_loss 不应超过 max_drawdown（日损限额应在最大回撤容忍之内）
        max_dd = _to_float(trading.get("max_drawdown"))
        daily_loss = _to_float(trading.get("daily_max_loss"))
        if max_dd > 0 and daily_loss > max_dd:
            report.add_error(f"daily_max_loss({daily_loss}) 超过 max_drawdown({max_dd})")

        # 4) 告警延迟 warning < critical
        monitoring = config.get("monitoring", {}) or {}
        warn_ms = _to_float(monitoring.get("api_latency_warning_ms"))
        crit_ms = _to_float(monitoring.get("api_latency_critical_ms"))
        if warn_ms > 0 and crit_ms > 0 and warn_ms >= crit_ms:
            report.add_error(
                f"api_latency_warning_ms({warn_ms}) 应小于 api_latency_critical_ms({crit_ms})"
            )

        # 5) tier min <= max
        currencies = config.get("currencies", {}) or {}
        for tier_key in ("tier1_settings", "tier2_settings", "tier3_settings"):
            tier = currencies.get(tier_key) or {}
            if not isinstance(tier, dict):
                continue
            for prefix in ("grid_spacing", "leverage", "position_size", "stop_loss", "take_profit"):
                key_min = f"{prefix}_min"
                key_max = f"{prefix}_max"
                if key_min in tier and key_max in tier:
                    mn = _to_float(tier.get(key_min))
                    mx = _to_float(tier.get(key_max))
                    if mn > mx:
                        report.add_error(f"{tier_key}.{key_min}({mn}) 大于 {key_max}({mx})")

    # ============================================================
    # 防回归检查
    # ============================================================

    def _check_regression(self, new_config: Dict[str, Any], baseline: Dict[str, Any], report: ConfigGuardReport) -> None:
        if not isinstance(new_config, dict) or not isinstance(baseline, dict):
            return

        # 交易级安全参数
        for path, direction, tolerance in _REGRESSION_RULES:
            new_val = _get_path(new_config, path)
            base_val = _get_path(baseline, path)
            self._check_numeric_regression(report, path, direction, tolerance, new_val, base_val)

        # 策略级：信号质量门槛 + 止损
        base_strategies = baseline.get("strategies", {}) or {}
        new_strategies = new_config.get("strategies", {}) or {}
        if isinstance(base_strategies, dict) and isinstance(new_strategies, dict):
            for name, base_s in base_strategies.items():
                if not isinstance(base_s, dict):
                    continue
                new_s = new_strategies.get(name, {}) or {}
                if not isinstance(new_s, dict):
                    continue

                self._check_numeric_regression(
                    report,
                    f"strategies.{name}.min_signal_quality",
                    "decrease", 0.10,
                    new_s.get("min_signal_quality"),
                    base_s.get("min_signal_quality"),
                )
                for stop_key in _STRATEGY_STOP_KEYS:
                    self._check_numeric_regression(
                        report,
                        f"strategies.{name}.{stop_key}",
                        "increase", 0.10,
                        new_s.get(stop_key),
                        base_s.get(stop_key),
                    )

    def _check_numeric_regression(
        self,
        report: ConfigGuardReport,
        path: str,
        direction: str,
        tolerance: float,
        new_val: Any,
        base_val: Any,
    ) -> None:
        """比较单个数值参数是否相对基线朝更危险方向退化。"""
        if new_val is None or base_val is None:
            return
        base_f = _to_float(base_val)
        new_f = _to_float(new_val)
        if base_f == 0:
            return  # 基线为 0，无法计算相对变化，跳过

        change = (new_f - base_f) / abs(base_f)
        if direction == "increase" and change > tolerance:
            report.add_warning(
                f"{path} 较基线上升 {change:.1%}（{base_f} → {new_f}），可能放大风险敞口"
            )
        elif direction == "decrease" and change < -tolerance:
            report.add_warning(
                f"{path} 较基线下降 {abs(change):.1%}（{base_f} → {new_f}），可能降低风控/信号门槛"
            )


# ============================================================
# 工具函数
# ============================================================

def _get_path(config: Dict[str, Any], path: str) -> Any:
    """按点号路径读取嵌套配置值，缺省返回 None。"""
    value: Any = config
    for key in path.split("."):
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


__all__ = ["ConfigConsistencyGuard", "ConfigGuardReport"]
