"""
决策相关的 REST API 路由。
"""
from typing import Any, Dict, List
from loguru import logger


class DecisionsAPI:
    def __init__(self, decision_coordinator=None, rule_engine=None, ensemble_maker=None, 
                 decision_validator=None, decision_evaluator=None, confidence_calibrator=None):
        self._decision_coordinator = decision_coordinator
        self._rule_engine = rule_engine
        self._ensemble_maker = ensemble_maker
        self._decision_validator = decision_validator
        self._decision_evaluator = decision_evaluator
        self._confidence_calibrator = confidence_calibrator

    def register_routes(self, app):
        @app.get("/api/v1/decisions")
        async def get_decisions():
            if self._decision_coordinator:
                return {"decisions": self._decision_coordinator.get_pending_decisions()}
            return {"decisions": []}

        @app.get("/api/v1/decisions/{decision_id}")
        async def get_decision(decision_id: str):
            if self._decision_coordinator:
                return self._decision_coordinator.get_decision(decision_id) or {"error": "Decision not found"}
            return {"error": "Decision coordinator not available"}

        @app.post("/api/v1/decisions")
        async def submit_decision(data: Dict[str, Any]):
            if self._decision_coordinator:
                from decision.decision_coordinator import Decision, DecisionType, DecisionPriority
                import uuid
                decision = Decision(
                    decision_id=data.get("decision_id") or str(uuid.uuid4()),
                    decision_type=DecisionType(data.get("decision_type", "signal")),
                    data=data.get("data", {}),
                    confidence=data.get("confidence", 0.5),
                    priority=DecisionPriority(data.get("priority", "normal")),
                    source=data.get("source", "api"),
                )
                result = await self._decision_coordinator.submit_decision(decision)
                return result
            return {"error": "Decision coordinator not available"}

        @app.get("/api/v1/decisions/metrics")
        async def get_decision_metrics():
            metrics = {}
            if self._decision_evaluator:
                metrics["quality"] = self._decision_evaluator.get_metrics()
            if self._confidence_calibrator:
                metrics["calibration"] = self._confidence_calibrator.get_stats()
            if self._ensemble_maker:
                metrics["ensemble"] = self._ensemble_maker.get_stats()
            return metrics

        @app.get("/api/v1/decisions/rules")
        async def get_rules():
            if self._rule_engine:
                return {"rules": [rule.to_dict() for rule in self._rule_engine.get_rules()]}
            return {"rules": []}

        @app.post("/api/v1/decisions/rules")
        async def add_rule(data: Dict[str, Any]):
            if self._rule_engine:
                from decision.rule_based_engine import Rule, RuleOperator, RuleActionType
                rule = Rule(
                    name=data.get("name"),
                    conditions=data.get("conditions", []),
                    action_type=RuleActionType(data.get("action_type", "APPROVE")),
                    action_params=data.get("action_params", {}),
                    priority=data.get("priority", 1),
                )
                self._rule_engine.add_rule(rule)
                return {"status": "success", "rule": rule.to_dict()}
            return {"error": "Rule engine not available"}

        @app.get("/api/v1/decisions/calibration")
        async def get_calibration():
            if self._confidence_calibrator:
                return self._confidence_calibrator.get_calibration_curve()
            return {"error": "Confidence calibrator not available"}

        logger.info("Decision API routes registered")
