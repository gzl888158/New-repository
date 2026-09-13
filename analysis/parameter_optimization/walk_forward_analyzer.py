"""
前向行走分析器 (Walk-Forward Analyzer)

前向行走 (Walk-Forward) 分析用于评估策略参数的稳健性：
  - 滚动窗口优化：在训练窗口上优化参数，在验证窗口上验证
  - 锚定/非锚定窗口模式
  - 多指标评估：收益率、夏普、最大回撤、胜率
  - 过拟合检测：IS/OOS 绩效比、参数稳定性
  - 最优窗口大小建议
  - 优化器集成：支持 GA、贝叶斯或网格搜索
"""
import asyncio
import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Callable
import numpy as np
from loguru import logger

from analysis.parameter_optimization.genetic_optimizer import ParameterDef


class WindowMode(Enum):
    """窗口模式"""
    ANCHORED = "anchored"        # 锚定：训练窗口固定起点，逐渐扩展
    ROLLING = "rolling"          # 滚动：训练窗口大小固定，向前滑动
    EXPANDING = "expanding"      # 扩展：训练窗口不断增大


@dataclass
class WindowResult:
    """单窗口结果"""
    window_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    # 最优参数
    best_params: Dict[str, float]
    # 训练集绩效
    train_return: float = 0.0
    train_sharpe: float = 0.0
    train_max_dd: float = 0.0
    train_win_rate: float = 0.0
    # 验证集绩效 (OOS)
    test_return: float = 0.0
    test_sharpe: float = 0.0
    test_max_dd: float = 0.0
    test_win_rate: float = 0.0
    # 指标
    is_oos_ratio: float = 0.0    # IS/OOS Sharpe比 (<1.5 为健康)
    param_stability: float = 0.0  # 参数稳定性 (与前一窗口的欧氏距离)
    optimization_time: float = 0.0


@dataclass
class WalkForwardResult:
    """前向行走分析结果"""
    window_results: List[WindowResult] = field(default_factory=list)
    # 聚合统计
    avg_test_sharpe: float = 0.0
    std_test_sharpe: float = 0.0
    avg_test_return: float = 0.0
    avg_is_oos_ratio: float = 0.0
    # 稳健性指标
    robustness_score: float = 0.0        # 0-1 综合稳健性评分
    overfit_risk: str = "unknown"         # low / moderate / high
    param_stability_mean: float = 0.0
    # 最优参数（OOS表现最好的窗口）
    best_window_params: Dict[str, float] = field(default_factory=dict)
    best_window_index: int = 0
    # 参数分布
    param_distributions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # 元信息
    total_windows: int = 0
    total_time_seconds: float = 0.0
    window_mode: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aggregates": {
                "avg_test_sharpe": round(self.avg_test_sharpe, 4),
                "std_test_sharpe": round(self.std_test_sharpe, 4),
                "avg_test_return": round(self.avg_test_return, 4),
                "avg_is_oos_ratio": round(self.avg_is_oos_ratio, 4),
                "param_stability_mean": round(self.param_stability_mean, 4),
            },
            "robustness": {
                "score": round(self.robustness_score, 4),
                "overfit_risk": self.overfit_risk,
            },
            "best_window_params": self.best_window_params,
            "best_window_index": self.best_window_index,
            "param_distributions": {
                k: {kk: round(vv, 4) for kk, vv in v.items()}
                for k, v in self.param_distributions.items()
            },
            "windows": [
                {
                    "index": w.window_index,
                    "train_period": f"{w.train_start}~{w.train_end}",
                    "test_period": f"{w.test_start}~{w.test_end}",
                    "train_sharpe": round(w.train_sharpe, 4),
                    "test_sharpe": round(w.test_sharpe, 4),
                    "is_oos_ratio": round(w.is_oos_ratio, 4),
                    "param_stability": round(w.param_stability, 4),
                }
                for w in self.window_results
            ],
            "total_windows": self.total_windows,
            "total_time_seconds": round(self.total_time_seconds, 2),
            "window_mode": self.window_mode,
            "timestamp": self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════
# 前向行走分析器
# ═══════════════════════════════════════════════════════════════

class WalkForwardAnalyzer:
    """前向行走分析器"""

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config.get("walk_forward", {}) if config else {}
        self._train_window = cfg.get("train_window_days", 60)
        self._test_window = cfg.get("test_window_days", 20)
        self._step_size = cfg.get("step_size_days", 10)
        self._min_windows = cfg.get("min_windows", 5)
        self._window_mode = WindowMode(cfg.get("window_mode", "rolling"))
        # 过拟合检测阈值
        self._overfit_is_oos_threshold = cfg.get("overfit_is_oos_threshold", 2.0)
        self._overfit_param_stability_threshold = cfg.get("overfit_param_stability_threshold", 0.3)
        self._seed = cfg.get("seed", None)
        # 优化器引用
        self._optimizer = None  # 外部注入（GA 或 Bayesian）
        self._optimize_fn: Optional[Callable] = None  # optimize(params_defs, train_data) -> best_params
        self._eval_fn: Optional[Callable] = None        # eval(params, data) -> performance_dict
        self._label_fn: Optional[Callable] = None       # label(data, index) -> (start_label, end_label)
        # 数据
        self._price_data: Optional[np.ndarray] = None
        self._date_labels: Optional[List[str]] = None
        self._param_defs: List[ParameterDef] = []

        logger.info(f"WalkForwardAnalyzer initialized: mode={self._window_mode.value}, "
                    f"train={self._train_window}d, test={self._test_window}d, step={self._step_size}d")

    def set_param_defs(self, param_defs: List[ParameterDef]):
        self._param_defs = param_defs

    def set_optimize_fn(self, fn: Callable):
        """设置优化函数 fn(param_defs, train_data) -> Dict[str, float]"""
        self._optimize_fn = fn

    def set_eval_fn(self, fn: Callable):
        """设置评估函数 fn(params, data) -> Dict[str, float]"""
        self._eval_fn = fn

    def set_data(self, price_data: np.ndarray, date_labels: List[str] = None):
        """设置价格数据"""
        self._price_data = price_data
        self._date_labels = date_labels or [str(i) for i in range(len(price_data))]

    # ── 窗口生成 ──────────────────────────────────────────────

    def _generate_windows(self) -> List[Tuple[int, int, int, int]]:
        """生成训练/验证窗口索引
        返回: [(train_start, train_end, test_start, test_end), ...]
        """
        n = len(self._price_data)
        windows = []

        if self._window_mode == WindowMode.ANCHORED:
            # 锚定模式：训练起点固定，终点逐渐后移
            train_start = 0
            train_end = self._train_window
            while train_end + self._test_window <= n and len(windows) < 30:
                test_start = train_end
                test_end = train_end + self._test_window
                windows.append((train_start, train_end, test_start, test_end))
                train_end += self._step_size
                # 扩展训练窗口
                # (anchored keeps start, moves end)

        elif self._window_mode == WindowMode.ROLLING:
            train_start = 0
            while train_start + self._train_window + self._test_window <= n:
                train_end = train_start + self._train_window
                test_start = train_end
                test_end = min(test_start + self._test_window, n)
                if test_end - test_start < self._test_window // 2:
                    break
                windows.append((train_start, train_end, test_start, test_end))
                train_start += self._step_size

        else:  # EXPANDING
            train_start = 0
            train_end = self._train_window
            while train_end + self._test_window <= n:
                test_start = train_end
                test_end = min(test_start + self._test_window, n)
                windows.append((train_start, train_end, test_start, test_end))
                train_end += self._step_size

        logger.info(f"Generated {len(windows)} walk-forward windows")
        return windows[:max(len(windows), self._min_windows)]

    def _get_window_data(self, start: int, end: int) -> np.ndarray:
        """提取窗口数据"""
        return self._price_data[start:end]

    # ── 绩效计算 ──────────────────────────────────────────────

    @staticmethod
    def _compute_returns(prices: np.ndarray) -> np.ndarray:
        """计算收益率序列"""
        if len(prices) < 2:
            return np.zeros(1)
        return np.diff(prices) / prices[:-1]

    @staticmethod
    def _compute_sharpe(returns: np.ndarray, risk_free: float = 0.0) -> float:
        """计算夏普比率"""
        if len(returns) < 2:
            return 0.0
        excess = returns - risk_free / 365
        mean = float(np.mean(excess))
        std = float(np.std(excess, ddof=1))
        if std <= 0:
            return 0.0
        return mean / std * math.sqrt(365)

    @staticmethod
    def _compute_max_drawdown(returns: np.ndarray) -> float:
        """计算最大回撤"""
        if len(returns) < 2:
            return 0.0
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        dd = (cumulative - running_max) / np.maximum(running_max, 1e-10)
        return float(abs(np.min(dd)))

    @staticmethod
    def _compute_win_rate(returns: np.ndarray) -> float:
        """计算胜率"""
        if len(returns) == 0:
            return 0.0
        return float(np.sum(returns > 0) / len(returns))

    def _eval_performance(self, params: Dict[str, float],
                          data: np.ndarray) -> Optional[Dict[str, float]]:
        """调用评估函数并归一化为指标字典。

        兼容两种返回值：
          - Dict：直接提取 return / sharpe / max_drawdown / win_rate
          - float：视为统一绩效指标（如夏普或收益），作为 return 与 sharpe
        """
        if not self._eval_fn:
            return None
        try:
            result = self._eval_fn(params, data)
        except Exception as e:
            logger.warning(f"eval_fn failed: {e}")
            return None

        if isinstance(result, dict):
            return {
                "return": float(result.get("return", result.get("total_return", 0.0) or 0.0)),
                "sharpe": float(result.get("sharpe", result.get("sharpe_ratio", 0.0) or 0.0)),
                "max_drawdown": float(result.get("max_drawdown", result.get("max_dd", 0.0) or 0.0)),
                "win_rate": float(result.get("win_rate", 0.0) or 0.0),
            }

        try:
            val = float(result)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(val):
            return None
        return {"return": val, "sharpe": val, "max_drawdown": 0.0, "win_rate": 0.0}

    # ── 参数稳定性 ────────────────────────────────────────────

    def _compute_param_stability(self, params1: Dict[str, float],
                                  params2: Dict[str, float]) -> float:
        """计算两组参数之间的归一化欧氏距离"""
        if not self._param_defs:
            return 0.0
        total_dist = 0.0
        for pd in self._param_defs:
            v1 = params1.get(pd.name, 0)
            v2 = params2.get(pd.name, 0)
            range_val = max(pd.high - pd.low, 1e-10)
            total_dist += ((v1 - v2) / range_val) ** 2
        return math.sqrt(total_dist / len(self._param_defs))

    # ── 稳健性评分 ────────────────────────────────────────────

    def _compute_robustness(self, result: WalkForwardResult):
        """计算综合稳健性评分"""
        # 1. OOS Sharpe 稳定性 (权重 0.35)
        sharpe_stability = 1.0 - min(result.std_test_sharpe / max(abs(result.avg_test_sharpe), 0.01), 1.0)

        # 2. IS/OOS 比率 (权重 0.25)
        is_oos_score = max(0, 1.0 - result.avg_is_oos_ratio / self._overfit_is_oos_threshold)

        # 3. 参数稳定性 (权重 0.25)
        param_score = max(0, 1.0 - result.param_stability_mean / self._overfit_param_stability_threshold)

        # 4. 平均 OOS Sharpe (权重 0.15)
        sharpe_score = min(result.avg_test_sharpe / 1.0, 1.0) if result.avg_test_sharpe > 0 else 0

        score = sharpe_stability * 0.35 + is_oos_score * 0.25 + param_score * 0.25 + sharpe_score * 0.15

        # 过拟合风险评估
        if result.avg_is_oos_ratio > self._overfit_is_oos_threshold:
            result.overfit_risk = "high"
        elif result.avg_is_oos_ratio > self._overfit_is_oos_threshold * 0.7:
            result.overfit_risk = "moderate"
        else:
            result.overfit_risk = "low"

        result.robustness_score = score

    def _analyze_param_distributions(self, result: WalkForwardResult):
        """分析参数分布统计"""
        if not result.window_results or not self._param_defs:
            return
        for pd in self._param_defs:
            values = [w.best_params.get(pd.name, 0) for w in result.window_results]
            arr = np.array(values)
            result.param_distributions[pd.name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "median": float(np.median(arr)),
            }

    # ── 主分析流程 ────────────────────────────────────────────

    async def analyze(self) -> WalkForwardResult:
        """执行前向行走分析"""
        if self._price_data is None or len(self._price_data) < self._train_window:
            raise ValueError("Insufficient price data")
        if not self._optimize_fn:
            raise ValueError("No optimize function set")
        if not self._eval_fn:
            raise ValueError("No eval function set")

        start_time = time.time()

        # 确定性：设置随机种子（若配置），保证窗口扰动可复现
        if self._seed is not None:
            random.seed(self._seed)
            np.random.seed(self._seed)

        windows = self._generate_windows()
        result = WalkForwardResult(total_windows=len(windows), window_mode=self._window_mode.value)

        prev_params = None

        for wi, (tr_start, tr_end, ts_start, ts_end) in enumerate(windows):
            # 提取训练和验证数据
            train_data = self._get_window_data(tr_start, tr_end)
            test_data = self._get_window_data(ts_start, ts_end)

            win_start = time.time()

            # 优化参数
            try:
                params = self._optimize_fn(self._param_defs, train_data)
                if hasattr(params, '__await__'):
                    params = await params
            except Exception as e:
                logger.warning(f"WF window {wi} optimization failed: {e}")
                continue

            # 训练集评估
            train_perf = self._eval_performance(params, train_data)
            if train_perf is not None:
                train_return = train_perf["return"]
                train_sharpe = train_perf["sharpe"]
                train_dd = train_perf["max_drawdown"]
                train_wr = train_perf["win_rate"]
            else:
                train_ret = self._compute_returns(train_data)
                train_return = float(np.sum(train_ret))
                train_sharpe = self._compute_sharpe(train_ret)
                train_dd = self._compute_max_drawdown(train_ret)
                train_wr = self._compute_win_rate(train_ret)

            # 验证集评估 (OOS)
            test_perf = self._eval_performance(params, test_data)
            if test_perf is not None:
                test_return = test_perf["return"]
                test_sharpe = test_perf["sharpe"]
                test_dd = test_perf["max_drawdown"]
                test_wr = test_perf["win_rate"]
            else:
                test_ret = self._compute_returns(test_data)
                test_return = float(np.sum(test_ret))
                test_sharpe = self._compute_sharpe(test_ret)
                test_dd = self._compute_max_drawdown(test_ret)
                test_wr = self._compute_win_rate(test_ret)

            # IS/OOS 比率
            is_oos = train_sharpe / max(abs(test_sharpe), 0.01) if test_sharpe != 0 else 0

            # 参数稳定性
            stability = self._compute_param_stability(prev_params, params) if prev_params else 0
            prev_params = dict(params)

            date_labels = self._date_labels or [str(i) for i in range(len(self._price_data))]

            wr = WindowResult(
                window_index=wi,
                train_start=date_labels[tr_start] if tr_start < len(date_labels) else str(tr_start),
                train_end=date_labels[tr_end - 1] if tr_end - 1 < len(date_labels) else str(tr_end - 1),
                test_start=date_labels[ts_start] if ts_start < len(date_labels) else str(ts_start),
                test_end=date_labels[min(ts_end - 1, len(date_labels) - 1)],
                best_params=params,
                train_return=train_return,
                train_sharpe=train_sharpe,
                train_max_dd=train_dd,
                train_win_rate=train_wr,
                test_return=test_return,
                test_sharpe=test_sharpe,
                test_max_dd=test_dd,
                test_win_rate=test_wr,
                is_oos_ratio=is_oos,
                param_stability=stability,
                optimization_time=time.time() - win_start,
            )
            result.window_results.append(wr)

            logger.debug(f"WF window {wi}: train_sharpe={train_sharpe:.3f}, "
                        f"test_sharpe={test_sharpe:.3f}, is/oos={is_oos:.2f}")

        # 聚合统计
        valid_windows = [w for w in result.window_results]
        if valid_windows:
            test_sharpes = [w.test_sharpe for w in valid_windows]
            test_returns = [w.test_return for w in valid_windows]
            is_oos_ratios = [w.is_oos_ratio for w in valid_windows]
            stabilities = [w.param_stability for w in valid_windows]

            result.avg_test_sharpe = float(np.mean(test_sharpes))
            result.std_test_sharpe = float(np.std(test_sharpes, ddof=1))
            result.avg_test_return = float(np.mean(test_returns))
            result.avg_is_oos_ratio = float(np.mean(is_oos_ratios))
            result.param_stability_mean = float(np.mean(stabilities))

            # 最优窗口
            best_idx = int(np.argmax(test_sharpes))
            result.best_window_index = best_idx
            result.best_window_params = dict(valid_windows[best_idx].best_params)

        # 稳健性评分
        self._compute_robustness(result)
        self._analyze_param_distributions(result)
        result.total_time_seconds = time.time() - start_time

        logger.info(f"Walk-Forward complete: {len(valid_windows)} windows, "
                    f"avg_test_sharpe={result.avg_test_sharpe:.3f}, "
                    f"robustness={result.robustness_score:.2f}, "
                    f"overfit_risk={result.overfit_risk}")

        return result

    def get_status(self) -> Dict[str, Any]:
        return {
            "train_window_days": self._train_window,
            "test_window_days": self._test_window,
            "step_size_days": self._step_size,
            "window_mode": self._window_mode.value,
            "data_length": len(self._price_data) if self._price_data is not None else 0,
            "parameters": len(self._param_defs),
        }
