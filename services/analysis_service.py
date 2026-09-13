"""
策略分析服务，定时分析各策略表现并执行参数优化与报告生成。
"""
import asyncio
from typing import Dict, Any
from loguru import logger


class AnalysisService:
    def __init__(self, analyzer, optimizer, alert_manager=None):
        self._analyzer = analyzer
        self._optimizer = optimizer
        self._alert_manager = alert_manager
        self._analysis_interval = 3600
        self._optimization_interval = 86400
        self._running = False

    async def start(self):
        self._running = True
        asyncio.create_task(self._analysis_loop())
        asyncio.create_task(self._optimization_loop())
        logger.info("Analysis service started")

    async def shutdown(self):
        self._running = False
        logger.info("Analysis service shutdown")

    async def _analysis_loop(self):
        while self._running:
            try:
                analysis = self._analyzer.analyze_all_strategies()
                report = self._analyzer.generate_analysis_report()
                
                if report["summary"]["performance_rating"] in ["差", "一般"]:
                    logger.warning(f"Performance alert: {report['summary']['performance_rating']}, PnL: {report['summary']['total_pnl']:.2f}")
                
                if analysis["shortcomings"]["total_shortcomings"] > 5:
                    logger.warning(f"High number of shortcomings: {analysis['shortcomings']['total_shortcomings']}")
            except Exception as e:
                logger.error(f"Analysis loop error: {e}")
            
            await asyncio.sleep(self._analysis_interval)

    async def _optimization_loop(self):
        while self._running:
            try:
                recommendations = await self._optimizer.optimize_all_strategies()
                
                optimized_count = sum(1 for s in recommendations.values() if isinstance(s, dict) and s.get("optimized"))
                if optimized_count > 0:
                    logger.info(f"Optimization completed: {optimized_count} strategies optimized")
                    
                    applied = await self._optimizer.apply_optimizations(recommendations)
                    if applied["total_applied"] > 0:
                        await self._optimizer.persist_config()
                        logger.info(f"Applied {applied['total_applied']} optimizations and persisted config")
                
                progress = self._optimizer.get_learning_progress()
                if progress["improvement_rate"] > 0.5:
                    logger.info(f"Learning improvement rate: {progress['improvement_rate']:.1%}")
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
            
            await asyncio.sleep(self._optimization_interval)