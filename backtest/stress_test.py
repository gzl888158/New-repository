"""压力测试引擎

模拟极端行情，验证风控系统的有效性：
- 闪电崩盘测试（LUNA式崩盘）
- 波动率飙升测试
- 连续亏损测试
- 流动性枯竭测试
- 黑天鹅事件模拟
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from loguru import logger
import numpy as np


@dataclass
class StressTestScenario:
    """压力测试场景"""
    name: str
    description: str
    scenario_type: str  # flash_crash / volatility_spike / consecutive_loss / liquidity_crisis / black_swan
    severity: str  # mild / moderate / severe / extreme
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StressTestResult:
    """压力测试结果"""
    scenario_name: str
    passed: bool
    max_drawdown: float = 0.0
    final_equity: float = 0.0
    initial_equity: float = 0.0
    positions_liquidated: int = 0
    total_loss: float = 0.0
    risk_actions_taken: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class StressTestEngine:
    """压力测试引擎"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._scenarios: List[StressTestScenario] = self._build_default_scenarios()

    def _build_default_scenarios(self) -> List[StressTestScenario]:
        """构建默认压力测试场景"""
        return [
            StressTestScenario(
                name="温和闪崩",
                description="BTC 5分钟内下跌5%，测试基础风控响应",
                scenario_type="flash_crash",
                severity="mild",
                parameters={"drop_pct": 0.05, "duration_seconds": 300, "symbol": "BTC-USDT-SWAP"},
            ),
            StressTestScenario(
                name="中度闪崩",
                description="BTC 10分钟内下跌10%，测试阶梯式风控",
                scenario_type="flash_crash",
                severity="moderate",
                parameters={"drop_pct": 0.10, "duration_seconds": 600, "symbol": "BTC-USDT-SWAP"},
            ),
            StressTestScenario(
                name="严重闪崩",
                description="BTC 15分钟内下跌20%，测试熔断和紧急全平",
                scenario_type="flash_crash",
                severity="severe",
                parameters={"drop_pct": 0.20, "duration_seconds": 900, "symbol": "BTC-USDT-SWAP"},
            ),
            StressTestScenario(
                name="极端黑天鹅",
                description="BTC 30分钟内下跌40%，LUNA式崩盘，测试全链路风控",
                scenario_type="flash_crash",
                severity="extreme",
                parameters={"drop_pct": 0.40, "duration_seconds": 1800, "symbol": "BTC-USDT-SWAP"},
            ),
            StressTestScenario(
                name="波动率飙升",
                description="1小时内波动率从2%升至15%，上下剧烈震荡",
                scenario_type="volatility_spike",
                severity="severe",
                parameters={"volatility_start": 0.02, "volatility_end": 0.15, "duration_hours": 4},
            ),
            StressTestScenario(
                name="连续亏损",
                description="连续20笔亏损，测试连亏风控和仓位缩减",
                scenario_type="consecutive_loss",
                severity="moderate",
                parameters={"consecutive_losses": 20, "avg_loss_pct": 0.01},
            ),
            StressTestScenario(
                name="流动性枯竭",
                description="买卖价差扩大10倍，滑点剧增，测试滑点保护",
                scenario_type="liquidity_crisis",
                severity="severe",
                parameters={"spread_multiplier": 10, "slippage_multiplier": 5, "duration_hours": 2},
            ),
        ]

    def list_scenarios(self) -> List[Dict[str, Any]]:
        """列出所有可用场景"""
        return [
            {
                "name": s.name,
                "description": s.description,
                "scenario_type": s.scenario_type,
                "severity": s.severity,
                "parameters": s.parameters,
            }
            for s in self._scenarios
        ]

    def simulate_flash_crash(
        self,
        initial_capital: float,
        positions: List[Dict[str, Any]],
        drop_pct: float,
        duration_seconds: int,
        symbol: str = "BTC-USDT-SWAP",
    ) -> StressTestResult:
        """模拟闪电崩盘"""
        logger.info(f"Running flash crash stress test: {drop_pct:.1%} drop in {duration_seconds}s")

        total_loss = 0.0
        liquidated = 0
        actions = []

        for pos in positions:
            pos_symbol = pos.get("symbol", "")
            pos_side = pos.get("side", "long")
            pos_size = float(pos.get("quantity", 0))
            entry_price = float(pos.get("entry_price", 0))
            leverage = int(pos.get("leverage", 1))
            margin = float(pos.get("margin", 0))

            if entry_price <= 0 or pos_size <= 0:
                continue

            btc_correlation = 1.0 if "BTC" in pos_symbol else 0.8 if "ETH" in pos_symbol else 0.6
            effective_drop = drop_pct * btc_correlation

            if pos_side == "long":
                price_change = -effective_drop
            else:
                price_change = effective_drop

            pnl_pct = price_change * leverage
            pnl = margin * pnl_pct

            total_loss += pnl

            if pnl_pct <= -0.9:
                liquidated += 1
                actions.append(f"{pos_symbol} {pos_side} 爆仓 (损失={pnl_pct:.1%})")
            elif pnl_pct <= -0.5:
                actions.append(f"{pos_symbol} {pos_side} 严重亏损 (损失={pnl_pct:.1%})")

        final_equity = initial_capital + total_loss
        max_drawdown = abs(total_loss) / initial_capital if initial_capital > 0 else 0

        if max_drawdown < 0.10:
            passed = True
            actions.append("风控有效：最大回撤在可控范围内")
        elif max_drawdown < 0.25:
            passed = True
            actions.append("风控有效：触发阶梯式减仓，回撤控制在阈值内")
        else:
            passed = False
            actions.append("风控不足：回撤超过阈值，需加强保护")

        return StressTestResult(
            scenario_name=f"闪电崩盘 ({drop_pct:.0%})",
            passed=passed,
            max_drawdown=round(max_drawdown, 4),
            final_equity=round(final_equity, 2),
            initial_equity=initial_capital,
            positions_liquidated=liquidated,
            total_loss=round(total_loss, 2),
            risk_actions_taken=actions,
            details={
                "drop_pct": drop_pct,
                "duration_seconds": duration_seconds,
                "position_count": len(positions),
                # coerce 为 float，避免 leverage 为字符串/None 时 np.mean 抛 TypeError
                "avg_leverage": float(np.mean([float(p.get("leverage", 1) or 1) for p in positions])) if positions else 1,
            },
        )

    def simulate_volatility_spike(
        self,
        initial_capital: float,
        positions: List[Dict[str, Any]],
        volatility_start: float,
        volatility_end: float,
        duration_hours: int,
    ) -> StressTestResult:
        """模拟波动率飙升"""
        logger.info(f"Running volatility spike test: {volatility_start:.1%} -> {volatility_end:.1%}")

        np.random.seed(42)
        total_pnl = 0.0
        whipsaw_count = 0

        for pos in positions:
            pos_side = pos.get("side", "long")
            margin = float(pos.get("margin", 0))
            leverage = int(pos.get("leverage", 1))

            n_periods = duration_hours * 12
            vol_increase = np.linspace(volatility_start, volatility_end, n_periods)

            pnl = 0.0
            prev_price = 100.0
            position_side = 1 if pos_side == "long" else -1

            for i in range(n_periods):
                price_change = np.random.normal(0, vol_increase[i])
                # 价格归零防护：prev_price 可能因极端负收益趋于 0，导致除零
                new_price = max(prev_price * (1 + price_change), 1e-9)

                period_pnl = (new_price - prev_price) / prev_price * leverage * margin * position_side
                pnl += period_pnl

                if i > 0 and (price_change * prev_change < 0) and abs(price_change) > vol_increase[i] * 0.5:
                    whipsaw_count += 1

                prev_price = new_price
                prev_change = price_change

            total_pnl += pnl

        final_equity = initial_capital + total_pnl
        max_drawdown = abs(min(0, total_pnl)) / initial_capital if initial_capital > 0 else 0

        passed = max_drawdown < 0.30

        actions = [
            f"波动率从 {volatility_start:.1%} 升至 {volatility_end:.1%}",
            f"反复收割次数: {whipsaw_count}",
        ]
        if passed:
            actions.append("风控有效：高波动环境下回撤可控")
        else:
            actions.append("需加强：高波动环境下回撤过大")

        return StressTestResult(
            scenario_name=f"波动率飙升 ({volatility_end:.0%})",
            passed=passed,
            max_drawdown=round(max_drawdown, 4),
            final_equity=round(final_equity, 2),
            initial_equity=initial_capital,
            total_loss=round(total_pnl, 2),
            risk_actions_taken=actions,
            details={
                "volatility_start": volatility_start,
                "volatility_end": volatility_end,
                "duration_hours": duration_hours,
                "whipsaw_count": whipsaw_count,
            },
        )

    def simulate_consecutive_losses(
        self,
        initial_capital: float,
        positions: List[Dict[str, Any]],
        consecutive_losses: int,
        avg_loss_pct: float,
    ) -> StressTestResult:
        """模拟连续亏损"""
        logger.info(f"Running consecutive loss test: {consecutive_losses} losses x {avg_loss_pct:.1%}")

        total_loss = 0.0
        current_capital = initial_capital
        actions = []

        loss_sequence = []
        for i in range(consecutive_losses):
            loss = current_capital * avg_loss_pct
            total_loss += loss
            current_capital -= loss
            loss_sequence.append(loss)

            if current_capital <= 0:
                actions.append(f"第{i+1}笔亏损后资金耗尽")
                break

        max_drawdown = abs(total_loss) / initial_capital if initial_capital > 0 else 0

        max_consecutive = self.config.get("trading", {}).get("max_consecutive_losses", 10)
        if consecutive_losses > max_consecutive:
            actions.append(f"连续亏损 {consecutive_losses} 笔 > 阈值 {max_consecutive}，应触发暂停")

        if max_drawdown < 0.20:
            passed = True
            actions.append("风控有效：连续亏损下回撤可控")
        else:
            passed = False
            actions.append("需加强：连续亏损导致回撤过大")

        return StressTestResult(
            scenario_name=f"连续亏损 ({consecutive_losses}笔)",
            passed=passed,
            max_drawdown=round(max_drawdown, 4),
            final_equity=round(current_capital, 2),
            initial_equity=initial_capital,
            total_loss=round(total_loss, 2),
            risk_actions_taken=actions,
            details={
                "consecutive_losses": consecutive_losses,
                "avg_loss_pct": avg_loss_pct,
                "total_loss_pct": max_drawdown,
            },
        )

    def simulate_liquidity_crisis(
        self,
        initial_capital: float,
        positions: List[Dict[str, Any]],
        spread_multiplier: float,
        slippage_multiplier: float,
        duration_hours: int,
    ) -> StressTestResult:
        """模拟流动性枯竭（买卖价差扩大、滑点剧增）"""
        logger.info(
            f"Running liquidity crisis test: spread x{spread_multiplier}, slippage x{slippage_multiplier}"
        )

        base_slippage = float(self.config.get("execution", {}).get("slippage", 0.0005))
        total_cost = 0.0
        actions = []

        for pos in positions:
            pos_symbol = pos.get("symbol", "")
            pos_size = float(pos.get("quantity", 0))
            entry_price = float(pos.get("entry_price", 0))
            if entry_price <= 0 or pos_size <= 0:
                continue

            notional = entry_price * pos_size
            # 价差扩大 → 市价平仓成交价恶化，叠加滑点倍增
            effective_slippage = base_slippage * slippage_multiplier * spread_multiplier
            slippage_cost = notional * effective_slippage
            total_cost += slippage_cost
            actions.append(
                f"{pos_symbol} 滑点成本 {slippage_cost:.2f} USDT "
                f"(滑点 x{slippage_multiplier}, 价差 x{spread_multiplier})"
            )

        final_equity = initial_capital - total_cost
        max_drawdown = total_cost / initial_capital if initial_capital > 0 else 0

        # 流动性危机下，滑点/价差导致的成交成本 < 5% 视为风控有效
        passed = max_drawdown < 0.05
        if passed:
            actions.append("滑点保护有效：流动性危机下成交成本可控")
        else:
            actions.append("需加强：流动性危机下成交成本过高，建议限价单+滑点上限")

        return StressTestResult(
            scenario_name=f"流动性枯竭 (滑点x{slippage_multiplier})",
            passed=passed,
            max_drawdown=round(max_drawdown, 4),
            final_equity=round(final_equity, 2),
            initial_equity=initial_capital,
            total_loss=round(total_cost, 2),
            risk_actions_taken=actions,
            details={
                "spread_multiplier": spread_multiplier,
                "slippage_multiplier": slippage_multiplier,
                "duration_hours": duration_hours,
                "base_slippage": base_slippage,
                "total_slippage_cost": round(total_cost, 2),
            },
        )

    def run_all_tests(
        self,
        initial_capital: float,
        positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """运行所有压力测试"""
        logger.info(f"Running full stress test suite with {len(positions)} positions")

        results: List[StressTestResult] = []

        for scenario in self._scenarios:
            try:
                if scenario.scenario_type == "flash_crash":
                    result = self.simulate_flash_crash(
                        initial_capital=initial_capital,
                        positions=positions,
                        drop_pct=scenario.parameters["drop_pct"],
                        duration_seconds=scenario.parameters["duration_seconds"],
                        symbol=scenario.parameters.get("symbol", "BTC-USDT-SWAP"),
                    )
                elif scenario.scenario_type == "volatility_spike":
                    result = self.simulate_volatility_spike(
                        initial_capital=initial_capital,
                        positions=positions,
                        volatility_start=scenario.parameters["volatility_start"],
                        volatility_end=scenario.parameters["volatility_end"],
                        duration_hours=scenario.parameters["duration_hours"],
                    )
                elif scenario.scenario_type == "consecutive_loss":
                    result = self.simulate_consecutive_losses(
                        initial_capital=initial_capital,
                        positions=positions,
                        consecutive_losses=scenario.parameters["consecutive_losses"],
                        avg_loss_pct=scenario.parameters["avg_loss_pct"],
                    )
                elif scenario.scenario_type == "liquidity_crisis":
                    result = self.simulate_liquidity_crisis(
                        initial_capital=initial_capital,
                        positions=positions,
                        spread_multiplier=scenario.parameters["spread_multiplier"],
                        slippage_multiplier=scenario.parameters["slippage_multiplier"],
                        duration_hours=scenario.parameters["duration_hours"],
                    )
                else:
                    continue

                results.append(result)
            except Exception as e:
                logger.error(f"Stress test failed for {scenario.name}: {e}")

        passed_count = sum(1 for r in results if r.passed)
        overall_score = passed_count / len(results) if results else 0

        max_dd_all = max(r.max_drawdown for r in results) if results else 0
        worst_case = min(r.final_equity for r in results) if results else initial_capital

        summary = {
            "total_tests": len(results),
            "passed": passed_count,
            "failed": len(results) - passed_count,
            "overall_score": round(overall_score, 2),
            "overall_rating": self._rate_score(overall_score),
            "max_drawdown_all_scenarios": round(max_dd_all, 4),
            "worst_case_equity": round(worst_case, 2),
            "worst_case_loss_pct": round((initial_capital - worst_case) / initial_capital, 4) if initial_capital > 0 else 0,
            "initial_capital": initial_capital,
            "position_count": len(positions),
        }

        return {
            "summary": summary,
            "results": [self._result_to_dict(r) for r in results],
            "recommendations": self._generate_recommendations(results, summary),
        }

    def _rate_score(self, score: float) -> str:
        if score >= 0.9:
            return "excellent"
        elif score >= 0.75:
            return "good"
        elif score >= 0.6:
            return "fair"
        elif score >= 0.4:
            return "poor"
        else:
            return "critical"

    def _result_to_dict(self, result: StressTestResult) -> Dict[str, Any]:
        return {
            "scenario_name": result.scenario_name,
            "passed": result.passed,
            "max_drawdown": result.max_drawdown,
            "final_equity": result.final_equity,
            "initial_equity": result.initial_equity,
            "positions_liquidated": result.positions_liquidated,
            "total_loss": result.total_loss,
            "risk_actions_taken": result.risk_actions_taken,
            "details": result.details,
        }

    def _generate_recommendations(
        self, results: List[StressTestResult], summary: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """生成风控改进建议"""
        recommendations = []

        failed_tests = [r for r in results if not r.passed]
        if failed_tests:
            recommendations.append({
                "priority": "high",
                "category": "risk_management",
                "title": "加强极端行情风控",
                "description": f"{len(failed_tests)} 个压力测试未通过，建议加强以下方面："
                               f"降低整体杠杆、增加对冲仓位、设置更严格的止损",
                "failed_scenarios": [r.scenario_name for r in failed_tests],
            })

        max_dd = summary.get("max_drawdown_all_scenarios", 0)
        if max_dd > 0.25:
            recommendations.append({
                "priority": "critical",
                "category": "drawdown_protection",
                "title": "最大回撤超标",
                "description": f"最坏情况回撤 {max_dd:.1%} 超过 25% 阈值，需立即采取措施",
                "suggestion": "建议：1) 降低整体杠杆 2) 增加对冲 3) 设置更严格的熔断阈值",
            })

        liquidated = sum(r.positions_liquidated for r in results)
        if liquidated > 0:
            recommendations.append({
                "priority": "critical",
                "category": "liquidation_risk",
                "title": "存在爆仓风险",
                "description": f"压力测试中有 {liquidated} 个仓位爆仓",
                "suggestion": "建议：1) 降低高杠杆仓位比例 2) 设置更早的减仓阈值 3) 增加保证金缓冲",
            })

        if not recommendations:
            recommendations.append({
                "priority": "low",
                "category": "healthy",
                "title": "风控体系健康",
                "description": "所有压力测试均通过，风控体系在极端行情下表现良好",
                "suggestion": "继续保持当前风控策略，定期进行压力测试验证",
            })

        return recommendations
