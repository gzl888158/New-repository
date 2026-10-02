"""eval 包：企业级评估模块。

注意：为避免循环导入，本包 __init__ 仅导出轻量级工具（_base）。
重型模块请直接从子模块导入：
    from eval.decision_quality_evaluator import DecisionQualityEvaluator
    from eval.review_engine import ReviewEngine
    from eval.verification_runner import VerificationRunner

历史路径兼容：core.alert_evaluator / decision.decision_quality_evaluator /
review.review_engine / verification.verification_runner 均以 re-export stub 形式
指向本包对应模块，外部导入无需修改。
"""
from eval._base import safe_int, safe_float, safe_div, safe_finite

__all__ = [
    "safe_int",
    "safe_float",
    "safe_div",
    "safe_finite",
]
