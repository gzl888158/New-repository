"""
交易流水线相关的 REST API 路由。
"""
from typing import Any, Dict, List
from loguru import logger


class PipelineAPI:
    def __init__(self, pipeline_orchestrator=None, signal_pipeline=None, anomaly_detector=None, recovery_handler=None):
        self._pipeline_orchestrator = pipeline_orchestrator
        self._signal_pipeline = signal_pipeline
        self._anomaly_detector = anomaly_detector
        self._recovery_handler = recovery_handler

    def register_routes(self, app):
        @app.get("/api/v1/pipeline/status")
        async def get_pipeline_status():
            if self._pipeline_orchestrator:
                return {"status": self._pipeline_orchestrator.get_status()}
            return {"status": "unavailable"}

        @app.get("/api/v1/pipeline/metrics")
        async def get_pipeline_metrics():
            if self._pipeline_orchestrator:
                return self._pipeline_orchestrator.get_metrics()
            return {"error": "Pipeline orchestrator not available"}

        @app.post("/api/v1/pipeline/execute")
        async def execute_pipeline(data: Dict[str, Any]):
            if self._pipeline_orchestrator:
                result = await self._pipeline_orchestrator.execute_pipeline(data.get("signal_data", {}))
                return result
            return {"error": "Pipeline orchestrator not available"}

        @app.post("/api/v1/pipeline/pause")
        async def pause_pipeline():
            if self._pipeline_orchestrator:
                await self._pipeline_orchestrator.pause()
                return {"status": "success", "message": "Pipeline paused"}
            return {"error": "Pipeline orchestrator not available"}

        @app.post("/api/v1/pipeline/resume")
        async def resume_pipeline():
            if self._pipeline_orchestrator:
                await self._pipeline_orchestrator.resume()
                return {"status": "success", "message": "Pipeline resumed"}
            return {"error": "Pipeline orchestrator not available"}

        @app.get("/api/v1/pipeline/anomalies")
        async def get_anomalies(limit: int = 50, severity: str = None):
            if self._anomaly_detector:
                from app.services.trading_pipeline import AnomalySeverity
                severity_enum = AnomalySeverity(severity) if severity else None
                anomalies = self._anomaly_detector.get_anomalies(limit, severity_enum)
                return {"anomalies": [a.to_dict() for a in anomalies]}
            return {"anomalies": []}

        @app.get("/api/v1/pipeline/anomalies/summary")
        async def get_anomaly_summary():
            if self._anomaly_detector:
                return self._anomaly_detector.get_anomaly_summary()
            return {"error": "Anomaly detector not available"}

        @app.get("/api/v1/pipeline/recovery/tasks")
        async def get_recovery_tasks(limit: int = 50, status: str = None):
            if self._recovery_handler:
                from app.services.trading_pipeline import RecoveryStatus
                status_enum = RecoveryStatus(status) if status else None
                tasks = self._recovery_handler.get_tasks(limit, status_enum)
                return {"tasks": [t.to_dict() for t in tasks]}
            return {"tasks": []}

        @app.get("/api/v1/pipeline/recovery/summary")
        async def get_recovery_summary():
            if self._recovery_handler:
                return self._recovery_handler.get_recovery_summary()
            return {"error": "Recovery handler not available"}

        @app.get("/api/v1/pipeline/signal_history")
        async def get_signal_history(limit: int = 100):
            if self._signal_pipeline:
                return {"signals": self._signal_pipeline.get_signal_history(limit)}
            return {"signals": []}

        logger.info("Pipeline API routes registered")
