"""置信度校准器：通过 Platt 缩放与保序回归校准模型预测置信度。"""
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
import math
import statistics


class ConfidenceCalibrator:
    def __init__(self, config=None):
        self.config = config or {}
        self._calibration_data: List[Dict[str, Any]] = []
        self._calibration_map: Dict[float, float] = {}
        self._is_calibrated = False
        self._min_samples = 50
        self._decay_factor = 0.95

        # Brier score tracking
        self._brier_scores: deque = deque(maxlen=100)

        # Online Platt scaling
        self._platt_params: Dict[str, float] = {"A": 0.0, "B": 0.0}
        self._platt_fitted: bool = False

        # Isotonic regression (PAVA)
        self._isotonic_enabled = (config or {}).get("confidence_calibrator", {}).get("isotonic_enabled", True)
        self._isotonic_fitted: bool = False
        self._isotonic_thresholds: List[float] = []
        self._isotonic_values: List[float] = []
        self._isotonic_min_samples = (config or {}).get("confidence_calibrator", {}).get("isotonic_min_samples", 100)

    async def add_sample(self, predicted_confidence: float, actual_outcome: bool):
        sample = {
            "predicted": predicted_confidence,
            "actual": 1.0 if actual_outcome else 0.0,
            "timestamp": datetime.now(),
            "_weight": 1.0,
        }
        self._calibration_data.append(sample)

        self._apply_decay()

        # Track Brier score for calibration quality monitoring
        brier = (predicted_confidence - sample["actual"]) ** 2
        self._brier_scores.append(brier)

        if len(self._calibration_data) >= self._min_samples:
            self._recalibrate()
            if len(self._calibration_data) >= self._min_samples * 2:
                self._fit_platt()
            # 更多样本时拟合Isotonic回归
            if len(self._calibration_data) >= self._isotonic_min_samples and self._isotonic_enabled:
                self._fit_isotonic()

        logger.debug(f"Added calibration sample: predicted={predicted_confidence:.2f}, actual={actual_outcome}")

    def _apply_decay(self):
        # Apply multiplicative decay to all existing sample weights
        for sample in self._calibration_data:
            sample["_weight"] = sample.get("_weight", 1.0) * self._decay_factor

        # Trim excessively old / low-weight samples to bound memory
        if len(self._calibration_data) > 500:
            self._calibration_data = self._calibration_data[-500:]

    def _recalibrate(self):
        n_samples = len(self._calibration_data)

        # Adaptive bucket sizing based on sample count
        if n_samples < 50:
            n_buckets = 5
        elif n_samples <= 200:
            n_buckets = 10
        else:
            n_buckets = 20

        buckets = {}
        for sample in self._calibration_data:
            bucket_key = round(sample["predicted"] * n_buckets) / n_buckets
            if bucket_key not in buckets:
                buckets[bucket_key] = {"sum": 0.0, "count": 0.0}
            weight = sample.get("_weight", 1.0)
            buckets[bucket_key]["sum"] += sample["actual"] * weight
            buckets[bucket_key]["count"] += weight

        sorted_keys = sorted(buckets.keys())
        self._calibration_map = {}

        for key in sorted_keys:
            actual_rate = buckets[key]["sum"] / buckets[key]["count"]
            self._calibration_map[key] = {"value": actual_rate, "count": buckets[key]["count"]}

        for i in range(len(sorted_keys) - 1):
            k1, k2 = sorted_keys[i], sorted_keys[i + 1]
            v1, v2 = self._calibration_map[k1]["value"], self._calibration_map[k2]["value"]
            step = (k2 - k1) / 10
            for j in range(1, 10):
                interp_key = k1 + j * step
                weight = j / 10
                self._calibration_map[interp_key] = {"value": v1 * (1 - weight) + v2 * weight, "count": 0.0}

        self._is_calibrated = True
        logger.info(f"Recalibrated confidence map with {len(buckets)} buckets")

    def calibrate(self, confidence: float, use_platt: Optional[bool] = None,
                  use_isotonic: bool = False) -> Tuple[float, float]:
        # Auto-detect: Isotonic > Platt > bucket, each with minimum sample requirements
        if use_isotonic and self._isotonic_fitted:
            return self.calibrate_isotonic(confidence), 1.0

        if use_platt is None:
            use_platt = self._platt_fitted

        if use_platt and self._platt_fitted:
            platt_value = self.calibrate_platt(confidence)
            if self._is_calibrated and self._calibration_map:
                _, reliability = self._bucket_calibrate(confidence)
                return platt_value, reliability
            return platt_value, 1.0

        return self._bucket_calibrate(confidence)

    def _bucket_calibrate(self, confidence: float) -> Tuple[float, float]:
        """Original bucket-based calibration logic."""
        if not self._is_calibrated or not self._calibration_map:
            return confidence, 1.0

        if confidence in self._calibration_map:
            entry = self._calibration_map[confidence]
            reliability = min(1.0, entry["count"] / 20.0)
            return entry["value"], reliability

        sorted_keys = sorted(self._calibration_map.keys())

        if confidence <= sorted_keys[0]:
            entry = self._calibration_map[sorted_keys[0]]
            reliability = min(1.0, entry["count"] / 20.0)
            return entry["value"], reliability
        if confidence >= sorted_keys[-1]:
            entry = self._calibration_map[sorted_keys[-1]]
            reliability = min(1.0, entry["count"] / 20.0)
            return entry["value"], reliability

        for i in range(len(sorted_keys) - 1):
            if sorted_keys[i] <= confidence <= sorted_keys[i + 1]:
                k1, k2 = sorted_keys[i], sorted_keys[i + 1]
                v1, v2 = self._calibration_map[k1]["value"], self._calibration_map[k2]["value"]
                weight = (confidence - k1) / (k2 - k1)
                calibrated_value = v1 * (1 - weight) + v2 * weight
                c1, c2 = self._calibration_map[k1]["count"], self._calibration_map[k2]["count"]
                count = c1 * (1 - weight) + c2 * weight
                reliability = min(1.0, count / 20.0)
                return calibrated_value, reliability

        return confidence, 1.0

    def calibrate_batch(self, confidences: List[float], use_platt: Optional[bool] = None) -> List[Tuple[float, float]]:
        return [self.calibrate(c, use_platt=use_platt) for c in confidences]

    # ------------------------------------------------------------------
    # Brier Score – calibration quality metric
    # ------------------------------------------------------------------

    def get_brier_score(self) -> float:
        """Return mean Brier score over tracked samples (lower = better calibration)."""
        if not self._brier_scores:
            return 0.0
        return statistics.mean(self._brier_scores)

    # ------------------------------------------------------------------
    # Online Platt Scaling (sigmoid calibration)
    # ------------------------------------------------------------------

    def _fit_platt(self) -> None:
        """Fit sigmoid calibration P(y=1|f) = 1/(1+exp(A*f+B)) via Newton-Raphson.

        Uses the most recent samples (up to 200) for fitting.  Weights from
        the sample buffer are honoured so that older data has less influence.
        """
        samples = self._calibration_data[-200:]
        if len(samples) < 20:
            return

        # Extract features and targets, weighted
        fs = []
        ts = []
        ws = []
        for s in samples:
            fs.append(s["predicted"])
            ts.append(s["actual"])
            ws.append(s.get("_weight", 1.0))

        A = self._platt_params["A"]
        B = self._platt_params["B"]

        for _ in range(10):  # Newton-Raphson iterations
            grad_A, grad_B = 0.0, 0.0
            hess_AA, hess_AB, hess_BB = 0.0, 0.0, 0.0

            for f_i, t_i, w_i in zip(fs, ts, ws):
                z = A * f_i + B
                # Clamp to avoid overflow in exp
                z = max(-50.0, min(50.0, z))
                exp_z = math.exp(-z)
                p_i = 1.0 / (1.0 + exp_z)
                diff = p_i - t_i
                grad_A += w_i * diff * f_i
                grad_B += w_i * diff
                hess_AA += w_i * p_i * (1.0 - p_i) * f_i * f_i
                hess_AB += w_i * p_i * (1.0 - p_i) * f_i
                hess_BB += w_i * p_i * (1.0 - p_i)

            # Solve 2x2 Hessian system
            det = hess_AA * hess_BB - hess_AB * hess_AB
            if abs(det) < 1e-12:
                break
            delta_A = (hess_BB * grad_A - hess_AB * grad_B) / det
            delta_B = (-hess_AB * grad_A + hess_AA * grad_B) / det

            A -= delta_A
            B -= delta_B

            if abs(delta_A) < 1e-6 and abs(delta_B) < 1e-6:
                break

        self._platt_params = {"A": A, "B": B}
        self._platt_fitted = True
        logger.debug(f"Platt scaling fitted: A={A:.4f}, B={B:.4f}")

    def _fit_isotonic(self) -> None:
        """
        Isotonic回归校准 - PAVA (Pool Adjacent Violators Algorithm)

        保序回归保证校准后置信度单调递增，不会出现"高置信度校准后反而变低"的反直觉情况。
        适合小样本场景（≥100个样本即可使用）。
        """
        samples = self._calibration_data[-min(500, len(self._calibration_data)):]
        if len(samples) < self._isotonic_min_samples:
            return

        # 按预测置信度排序
        sorted_samples = sorted(samples, key=lambda s: s["predicted"])
        n = len(sorted_samples)
        predictions = [s["predicted"] for s in sorted_samples]
        actuals = [s["actual"] for s in sorted_samples]
        weights = [s.get("_weight", 1.0) for s in sorted_samples]

        # PAVA: 将违反单调性的相邻桶合并
        bucket_predictions: List[float] = []
        bucket_actuals: List[float] = []
        bucket_weights: List[float] = []
        bucket_counts: List[int] = []

        for p, a, w in zip(predictions, actuals, weights):
            bucket_predictions.append(p)
            bucket_actuals.append(a)
            bucket_weights.append(w)
            bucket_counts.append(1)

            # 回溯合并违反单调性的桶
            while len(bucket_actuals) >= 2 and bucket_actuals[-1] < bucket_actuals[-2]:
                # 合并最后两个桶
                w1, w2 = bucket_weights[-2], bucket_weights[-1]
                if w1 + w2 > 0:
                    merged_actual = (bucket_actuals[-2] * w1 + bucket_actuals[-1] * w2) / (w1 + w2)
                else:
                    merged_actual = (bucket_actuals[-2] + bucket_actuals[-1]) / 2
                merged_pred = (bucket_predictions[-2] + bucket_predictions[-1]) / 2
                merged_weight = w1 + w2
                merged_count = bucket_counts[-2] + bucket_counts[-1]

                bucket_actuals.pop()
                bucket_predictions.pop()
                bucket_weights.pop()
                bucket_counts.pop()

                bucket_actuals[-1] = merged_actual
                bucket_predictions[-1] = merged_pred
                bucket_weights[-1] = merged_weight
                bucket_counts[-1] = merged_count

        self._isotonic_thresholds = bucket_predictions
        self._isotonic_values = bucket_actuals
        self._isotonic_fitted = True
        logger.debug(f"Isotonic regression fitted: {len(bucket_predictions)} bins from {n} samples")

    def calibrate_isotonic(self, confidence: float) -> float:
        """
        应用Isotonic回归校准

        在阈值之间线性插值，保持单调性。
        """
        if not self._isotonic_fitted or not self._isotonic_thresholds:
            return confidence

        thresholds = self._isotonic_thresholds
        values = self._isotonic_values

        if confidence <= thresholds[0]:
            return values[0]
        if confidence >= thresholds[-1]:
            return values[-1]

        # 二分查找
        lo, hi = 0, len(thresholds) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if thresholds[mid] <= confidence:
                lo = mid
            else:
                hi = mid

        # 线性插值
        if thresholds[hi] - thresholds[lo] > 0:
            t = (confidence - thresholds[lo]) / (thresholds[hi] - thresholds[lo])
            return values[lo] + t * (values[hi] - values[lo])
        return values[lo]

    def calibrate_platt(self, confidence: float) -> float:
        """Apply Platt-scaled calibration: returns 1/(1+exp(-(A*c+B)))."""
        if not self._platt_fitted:
            return confidence
        A, B = self._platt_params["A"], self._platt_params["B"]
        z = A * confidence + B
        z = max(-50.0, min(50.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    # ------------------------------------------------------------------
    # Reliability Diagram
    # ------------------------------------------------------------------

    def get_reliability_diagram(self, n_buckets: int = 10) -> List[Dict[str, Any]]:
        """Return bucket-level data for plotting a reliability diagram.

        Each bucket contains:
          - predicted_mean: mean predicted confidence in the bucket
          - actual_rate:     proportion of positive outcomes
          - count:           number of samples in the bucket
        """
        if not self._calibration_data:
            return []

        buckets: Dict[int, Dict[str, float]] = {}
        for sample in self._calibration_data:
            key = min(n_buckets - 1, int(sample["predicted"] * n_buckets))
            if key not in buckets:
                buckets[key] = {"predicted_sum": 0.0, "actual_sum": 0.0, "count": 0.0}
            w = sample.get("_weight", 1.0)
            buckets[key]["predicted_sum"] += sample["predicted"] * w
            buckets[key]["actual_sum"] += sample["actual"] * w
            buckets[key]["count"] += w

        result = []
        for key in sorted(buckets.keys()):
            b = buckets[key]
            result.append({
                "bucket": key,
                "predicted_mean": b["predicted_sum"] / b["count"] if b["count"] > 0 else 0.0,
                "actual_rate": b["actual_sum"] / b["count"] if b["count"] > 0 else 0.0,
                "count": b["count"],
            })
        return result

    # ------------------------------------------------------------------
    # Calibration Drift Detection
    # ------------------------------------------------------------------

    def detect_calibration_drift(self) -> Tuple[bool, float]:
        """Detect whether calibration quality has deteriorated over time.

        Compares the mean Brier score of the most recent 50 samples against
        the earliest 50 samples in the buffer.  Returns (drifted, change_pct).

        drifted=True when the Brier score has increased by more than 30%.
        """
        if len(self._brier_scores) < 100:
            return False, 0.0

        scores = list(self._brier_scores)
        recent_mean = statistics.mean(scores[-50:])
        earliest_mean = statistics.mean(scores[:50])

        if earliest_mean < 1e-9:
            return False, 0.0

        change_pct = (recent_mean - earliest_mean) / earliest_mean * 100.0
        drifted = change_pct > 30.0
        if drifted:
            logger.warning(f"Calibration drift detected: Brier score increased by {change_pct:.1f}%")
        return drifted, change_pct

    def get_calibration_error(self) -> float:
        if not self._calibration_data:
            return 0.0

        errors = []
        for sample in self._calibration_data:
            calibrated, _ = self.calibrate(sample["predicted"])
            error = (calibrated - sample["actual"]) ** 2
            errors.append(error)

        return statistics.mean(errors) if errors else 0.0

    def get_calibration_curve(self) -> Dict[str, Any]:
        if not self._calibration_map:
            return {}

        sorted_keys = sorted(self._calibration_map.keys())
        curve = [
            {"predicted": k, "calibrated": self._calibration_map[k]["value"]}
            for k in sorted_keys
        ]

        return {"curve": curve, "is_calibrated": self._is_calibrated, "sample_count": len(self._calibration_data)}

    def get_stats(self) -> Dict[str, Any]:
        return {
            "is_calibrated": self._is_calibrated,
            "sample_count": len(self._calibration_data),
            "bucket_count": len(self._calibration_map),
            "calibration_error": self.get_calibration_error(),
            "brier_score": self.get_brier_score(),
            "platt_fitted": self._platt_fitted,
            "min_samples_required": self._min_samples,
        }

    def reset(self):
        self._calibration_data = []
        self._calibration_map = {}
        self._is_calibrated = False
        self._brier_scores.clear()
        self._platt_params = {"A": 0.0, "B": 0.0}
        self._platt_fitted = False
        logger.info("Confidence calibrator reset")
