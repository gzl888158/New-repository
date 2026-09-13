"""决策质量评估器：记录决策结果并计算胜率、夏普、DQS 等绩效指标。"""
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
import statistics


class DecisionQualityEvaluator:
    def __init__(self, config=None):
        self.config = config or {}
        self._decisions: List[Dict[str, Any]] = []
        self._performance_metrics: Dict[str, Any] = {}
        self._rolling_window = 100
        self._latency_history: deque = deque(maxlen=200)
        self._dqs_history: deque = deque(maxlen=500)
        self._max_pnl_seen: float = self.config.get("max_expected_pnl", 0.0)
        self._max_duration_seen: float = self.config.get("max_expected_duration", 0.0)

    async def record_decision(self, decision_id: str, decision: Dict[str, Any], outcome: str, pnl: float,
                              duration: float, latency_ms: Optional[float] = None,
                              market_regime: str = ""):
        record = {
            "decision_id": decision_id,
            "decision": decision,
            "outcome": outcome,
            "pnl": pnl,
            "duration": duration,
            "timestamp": datetime.now(),
            "latency_ms": latency_ms,
            "market_regime": market_regime or decision.get("market_regime", "unknown"),
        }
        self._decisions.append(record)

        if len(self._decisions) > self._rolling_window:
            self._decisions = self._decisions[-self._rolling_window:]

        if latency_ms is not None:
            self._latency_history.append(latency_ms)

        abs_pnl = abs(pnl)
        if abs_pnl > self._max_pnl_seen:
            self._max_pnl_seen = abs_pnl
        if duration > self._max_duration_seen:
            self._max_duration_seen = duration

        dqs = self.compute_dqs(record)
        self._dqs_history.append(dqs)

        self._update_metrics()
        logger.info(f"Recorded decision outcome: {decision_id}, outcome: {outcome}, PnL: {pnl:.2f}, DQS: {dqs:.1f}")

    def _update_metrics(self):
        if not self._decisions:
            self._performance_metrics = {
                "total_decisions": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "avg_duration": 0.0,
            }
            return

        wins = [d for d in self._decisions if d["outcome"] == "win"]
        losses = [d for d in self._decisions if d["outcome"] == "loss"]
        total = len(self._decisions)

        win_rate = len(wins) / total if total > 0 else 0.0

        avg_win = statistics.mean(d["pnl"] for d in wins) if wins else 0.0
        avg_loss = statistics.mean(d["pnl"] for d in losses) if losses else 0.0

        total_win = sum(d["pnl"] for d in wins) if wins else 0.0
        total_loss = abs(sum(d["pnl"] for d in losses)) if losses else 1.0
        profit_factor = total_win / total_loss if total_loss > 0 else 0.0

        pnl_values = [d["pnl"] for d in self._decisions]
        mean_pnl = statistics.mean(pnl_values) if pnl_values else 0.0
        std_pnl = statistics.stdev(pnl_values) if len(pnl_values) > 1 else 1.0
        sharpe_ratio = mean_pnl / std_pnl if std_pnl > 0 else 0.0

        cumulative = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for d in self._decisions:
            cumulative += d["pnl"]
            peak = max(peak, cumulative)
            drawdown = (peak - cumulative) / max(peak, 1)
            max_drawdown = max(max_drawdown, drawdown)

        avg_duration = statistics.mean(d["duration"] for d in self._decisions) if self._decisions else 0.0

        self._performance_metrics = {
            "total_decisions": total,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "avg_duration": avg_duration,
            "total_pnl": sum(d["pnl"] for d in self._decisions),
        }

    def get_metrics(self) -> Dict[str, Any]:
        return self._performance_metrics

    def get_confidence_calibration(self) -> Dict[str, Any]:
        if not self._decisions:
            return {}

        confidence_buckets = {}
        for d in self._decisions:
            confidence = round(d["decision"].get("confidence", 0.5) * 10) / 10
            bucket = f"{confidence:.1f}-{(confidence + 0.1):.1f}"
            if bucket not in confidence_buckets:
                confidence_buckets[bucket] = {"total": 0, "wins": 0}
            confidence_buckets[bucket]["total"] += 1
            if d["outcome"] == "win":
                confidence_buckets[bucket]["wins"] += 1

        calibration = {}
        for bucket, stats in confidence_buckets.items():
            calibration[bucket] = {
                "confidence": float(bucket.split("-")[0]),
                "actual_win_rate": stats["wins"] / stats["total"],
                "sample_size": stats["total"],
            }

        return calibration

    def get_decisions_by_source(self, source: str) -> List[Dict[str, Any]]:
        return [d for d in self._decisions if d["decision"].get("source") == source]

    def evaluate_source_performance(self, source: str) -> Dict[str, Any]:
        source_decisions = self.get_decisions_by_source(source)
        if not source_decisions:
            return {"source": source, "error": "No decisions found"}

        wins = [d for d in source_decisions if d["outcome"] == "win"]
        losses = [d for d in source_decisions if d["outcome"] == "loss"]
        total = len(source_decisions)

        return {
            "source": source,
            "total_decisions": total,
            "win_rate": len(wins) / total if total > 0 else 0.0,
            "avg_win": statistics.mean(d["pnl"] for d in wins) if wins else 0.0,
            "avg_loss": statistics.mean(d["pnl"] for d in losses) if losses else 0.0,
            "total_pnl": sum(d["pnl"] for d in source_decisions),
            "avg_duration": statistics.mean(d["duration"] for d in source_decisions) if source_decisions else 0.0,
        }

    def get_time_based_metrics(self, hours: int = 24) -> Dict[str, Any]:
        cutoff_time = datetime.now() - datetime.timedelta(hours=hours)
        recent_decisions = [d for d in self._decisions if d["timestamp"] >= cutoff_time]

        if not recent_decisions:
            return {"error": f"No decisions in the last {hours} hours"}

        wins = [d for d in recent_decisions if d["outcome"] == "win"]
        losses = [d for d in recent_decisions if d["outcome"] == "loss"]

        return {
            "period": f"last_{hours}_hours",
            "total_decisions": len(recent_decisions),
            "win_rate": len(wins) / len(recent_decisions) if recent_decisions else 0.0,
            "total_pnl": sum(d["pnl"] for d in recent_decisions),
            "avg_win": statistics.mean(d["pnl"] for d in wins) if wins else 0.0,
            "avg_loss": statistics.mean(d["pnl"] for d in losses) if losses else 0.0,
        }

    def get_summary(self) -> Dict[str, Any]:
        return {
            "metrics": self.get_metrics(),
            "confidence_calibration": self.get_confidence_calibration(),
            "sources": {
                source: self.evaluate_source_performance(source)
                for source in set(d["decision"].get("source") for d in self._decisions if "source" in d["decision"])
            },
            "last_24h": self.get_time_based_metrics(24),
        }

    # ── Decision Quality Score (DQS) ──────────────────────────────

    def compute_dqs(self, decision_record: Dict[str, Any]) -> float:
        """计算复合质量评分 [0-100]

        评分构成：
          - 正确性 (40%): 正确=1, 错误=0
          - PnL 效率 (30%): pnl / max_expected_pnl
          - 持续时间效率 (10%): 1 - duration / max_expected_duration
          - 置信度准确性 (20%): 1 - |confidence - correctness|
        """
        is_win = 1.0 if decision_record["outcome"] == "win" else 0.0

        # 正确性
        correctness_score = is_win

        # PnL 效率：实际 PnL 相对于已观测最大绝对 PnL
        if self._max_pnl_seen > 0:
            pnl_efficiency = max(-1.0, min(1.0, decision_record["pnl"] / self._max_pnl_seen))
        else:
            pnl_efficiency = 1.0 if decision_record["pnl"] > 0 else (0.0 if decision_record["pnl"] == 0 else -1.0)

        # 持续时间效率
        if self._max_duration_seen > 0:
            duration_efficiency = max(0.0, 1.0 - decision_record["duration"] / self._max_duration_seen)
        else:
            duration_efficiency = 1.0

        # 置信度准确性
        confidence = decision_record["decision"].get("confidence", 0.5)
        confidence_accuracy = 1.0 - abs(confidence - is_win)

        dqs = (
            correctness_score * 40 +
            max(0.0, pnl_efficiency) * 30 +
            duration_efficiency * 10 +
            confidence_accuracy * 20
        )

        return round(max(0.0, min(100.0, dqs)), 1)

    def get_avg_dqs(self) -> float:
        """返回平均 DQS"""
        if not self._dqs_history:
            return 0.0
        return round(sum(self._dqs_history) / len(self._dqs_history), 2)

    # ── Rolling Performance Windows ───────────────────────────────

    _WINDOW_PRESETS: Dict[str, int] = {
        "1h": 60,
        "4h": 240,
        "6h": 360,
        "12h": 720,
        "24h": 1440,
        "7d": 10080,
    }

    def get_rolling_metrics(self, window_minutes: int) -> Dict[str, Any]:
        """计算最近 N 分钟的滚动指标"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=window_minutes)
        window_decisions = [d for d in self._decisions if d["timestamp"] >= cutoff]
        window_dqs = [
            self.compute_dqs(d) for d in window_decisions
        ]

        if not window_decisions:
            return {
                "window_minutes": window_minutes,
                "total_decisions": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_pnl": 0.0,
                "avg_dqs": 0.0,
            }

        wins = [d for d in window_decisions if d["outcome"] == "win"]
        losses = [d for d in window_decisions if d["outcome"] == "loss"]

        win_rate = len(wins) / len(window_decisions)

        total_win = sum(d["pnl"] for d in wins) if wins else 0.0
        total_loss = abs(sum(d["pnl"] for d in losses)) if losses else 1.0
        profit_factor = total_win / total_loss if total_loss > 0 else 0.0

        total_pnl = sum(d["pnl"] for d in window_decisions)
        avg_dqs = round(sum(window_dqs) / len(window_dqs), 2) if window_dqs else 0.0

        return {
            "window_minutes": window_minutes,
            "total_decisions": len(window_decisions),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "total_pnl": round(total_pnl, 2),
            "avg_dqs": avg_dqs,
        }

    def get_all_rolling_metrics(self) -> Dict[str, Dict[str, Any]]:
        """返回所有预设窗口（1h/4h/24h/7d）的滚动指标"""
        result = {}
        for label, minutes in self._WINDOW_PRESETS.items():
            result[label] = self.get_rolling_metrics(minutes)
        return result

    # ── Expectancy Ratio ──────────────────────────────────────────

    def get_expectancy(self) -> float:
        """计算每笔交易的期望值: avg_win * win_rate - avg_loss * loss_rate"""
        if not self._decisions:
            return 0.0

        wins = [d for d in self._decisions if d["outcome"] == "win"]
        losses = [d for d in self._decisions if d["outcome"] == "loss"]
        total = len(self._decisions)

        win_rate = len(wins) / total if total > 0 else 0.0
        loss_rate = len(losses) / total if total > 0 else 0.0

        avg_win = statistics.mean(d["pnl"] for d in wins) if wins else 0.0
        avg_loss = abs(statistics.mean(d["pnl"] for d in losses)) if losses else 0.0

        expectancy = avg_win * win_rate - avg_loss * loss_rate
        return round(expectancy, 4)

    # ── PnL Attribution ───────────────────────────────────────────

    def get_pnl_attribution(self) -> Dict[str, Any]:
        """按来源和小时分解 PnL"""
        if not self._decisions:
            return {"by_source": {}, "by_hour": {}}

        # 按 source 归因
        source_pnl: Dict[str, float] = {}
        for d in self._decisions:
            source = d["decision"].get("source", "unknown")
            source_pnl[source] = source_pnl.get(source, 0.0) + d["pnl"]

        # 按小时归因
        hour_pnl: Dict[str, float] = {}
        for d in self._decisions:
            hour = d["timestamp"].strftime("%H")
            hour_pnl[hour] = hour_pnl.get(hour, 0.0) + d["pnl"]

        # 按小时归因同时计算笔数
        hour_detail: Dict[str, Dict[str, float]] = {}
        for d in self._decisions:
            hour = d["timestamp"].strftime("%H")
            if hour not in hour_detail:
                hour_detail[hour] = {"total_pnl": 0.0, "count": 0}
            hour_detail[hour]["total_pnl"] += d["pnl"]
            hour_detail[hour]["count"] += 1

        return {
            "by_source": {k: round(v, 2) for k, v in sorted(source_pnl.items(), key=lambda x: x[1], reverse=True)},
            "by_hour": {k: v for k, v in sorted(hour_detail.items())},
            "total_pnl": round(sum(d["pnl"] for d in self._decisions), 2),
        }

    def get_regime_attribution(self) -> Dict[str, Any]:
        """按市场状态归因分析PnL表现
        
        分析不同市场状态（趋势/震荡/极端波动）下的策略表现，
        识别策略在不同市场环境中的强弱项。
        """
        if not self._decisions:
            return {"regimes": {}, "total_pnl": 0}

        regime_stats: Dict[str, Dict[str, Any]] = {}

        for d in self._decisions:
            regime = d.get("market_regime", "unknown")
            if regime not in regime_stats:
                regime_stats[regime] = {
                    "total_pnl": 0.0,
                    "count": 0,
                    "wins": 0,
                    "losses": 0,
                    "max_win": 0.0,
                    "max_loss": 0.0,
                    "total_duration": 0.0,
                }
            stats = regime_stats[regime]
            pnl = d["pnl"]
            stats["total_pnl"] += pnl
            stats["count"] += 1
            stats["total_duration"] += d.get("duration", 0)

            if pnl > 0:
                stats["wins"] += 1
                stats["max_win"] = max(stats["max_win"], pnl)
            else:
                stats["losses"] += 1
                stats["max_loss"] = min(stats["max_loss"], pnl)

        # 计算各状态的聚合指标
        for regime, stats in regime_stats.items():
            cnt = stats["count"]
            stats["win_rate"] = round(stats["wins"] / cnt, 3) if cnt > 0 else 0
            stats["avg_pnl"] = round(stats["total_pnl"] / cnt, 4) if cnt > 0 else 0
            stats["avg_duration_s"] = round(stats["total_duration"] / cnt, 1) if cnt > 0 else 0

            # 盈亏比：平均盈利 / 平均亏损的绝对值
            avg_win = stats["max_win"] if stats["wins"] > 0 else 0
            avg_loss = abs(stats["max_loss"]) if stats["losses"] > 0 else 1
            stats["profit_loss_ratio"] = round(avg_win / max(avg_loss, 1e-6), 2)

            # 每笔夏普近似
            if cnt >= 3 and "total_pnl" in stats:
                regime_pnls = [d["pnl"] for d in self._decisions if d.get("market_regime") == regime]
                if len(regime_pnls) >= 3:
                    mean_pnl = statistics.mean(regime_pnls)
                    std_pnl = statistics.stdev(regime_pnls) if len(regime_pnls) >= 3 else 0
                    stats["sharpe_approx"] = round(mean_pnl / max(std_pnl, 1e-6), 3)
                else:
                    stats["sharpe_approx"] = 0

        # 按PnL排序
        sorted_regimes = sorted(regime_stats.items(), key=lambda x: x[1]["total_pnl"], reverse=True)

        return {
            "regimes": {r: s for r, s in sorted_regimes},
            "total_pnl": round(sum(s["total_pnl"] for s in regime_stats.values()), 2),
            "best_regime": sorted_regimes[0][0] if sorted_regimes else "unknown",
            "worst_regime": sorted_regimes[-1][0] if sorted_regimes else "unknown",
        }

    # ── Decision Latency Tracking ─────────────────────────────────

    def get_latency_stats(self) -> Dict[str, float]:
        """返回 avg / p95 / max 延迟（毫秒）"""
        if not self._latency_history:
            return {"avg_latency_ms": 0.0, "p95_latency_ms": 0.0, "max_latency_ms": 0.0, "sample_count": 0}

        latencies = sorted(self._latency_history)
        avg = statistics.mean(latencies)
        p95_index = int(len(latencies) * 0.95)
        p95 = latencies[p95_index] if p95_index < len(latencies) else latencies[-1]

        return {
            "avg_latency_ms": round(avg, 2),
            "p95_latency_ms": round(p95, 2),
            "max_latency_ms": round(max(latencies), 2),
            "sample_count": len(latencies),
        }

    # ── Quality Alerts ────────────────────────────────────────────

    def check_quality_alerts(self) -> List[str]:
        """检查质量告警条件并返回告警消息列表"""
        alerts: List[str] = []
        if not self._decisions:
            return alerts

        # 1. 最近 20 笔胜率低于 35%
        recent_20 = self._decisions[-min(20, len(self._decisions)):]
        if len(recent_20) >= 5:
            recent_wins = sum(1 for d in recent_20 if d["outcome"] == "win")
            recent_win_rate = recent_wins / len(recent_20)
            if recent_win_rate < 0.35:
                alerts.append(
                    f"⚠ Win rate dropped to {recent_win_rate:.1%} (last {len(recent_20)} decisions, threshold: 35%)"
                )

        # 2. 平均 DQS 低于 40
        avg_dqs = self.get_avg_dqs()
        if avg_dqs > 0 and avg_dqs < 40:
            alerts.append(f"⚠ Average DQS is {avg_dqs:.1f} (threshold: 40)")

        # 3. 利润因子低于 0.8
        metrics = self.get_metrics()
        profit_factor = metrics.get("profit_factor", 1.0)
        if profit_factor < 0.8 and metrics.get("total_decisions", 0) >= 3:
            alerts.append(f"⚠ Profit factor is {profit_factor:.2f} (threshold: 0.8)")

        # 4. 连续亏损超过 5 笔
        consecutive_losses = 0
        for d in reversed(self._decisions):
            if d["outcome"] == "loss":
                consecutive_losses += 1
            else:
                break
        if consecutive_losses > 5:
            alerts.append(f"⚠ {consecutive_losses} consecutive losses detected")

        return alerts
