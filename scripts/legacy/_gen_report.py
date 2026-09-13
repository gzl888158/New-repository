import asyncio
import json
import os
from datetime import datetime

from data.sqlite_storage import SQLiteStorage
from data.redis_cache import RedisCache
from core.trade_journal import TradeJournal
from core.okx_client import OKXClient
from analysis.historical_analyzer import HistoricalAnalyzer
from analysis.contribution_analyzer import get_contribution_analyzer
from analysis.strategy_optimizer import StrategyOptimizer
from analysis.intelligent_analysis_agent import IntelligentAnalysisAgent
from decision.decision_quality_evaluator import DecisionQualityEvaluator
from services.market_regime_engine import MarketRegimeEngine
from configs.settings import load_config

cfg = load_config()

storage = SQLiteStorage(cfg)
redis_cache = RedisCache(cfg)
journal = TradeJournal(cfg, storage, redis_cache, None)

okx_client = None
try:
    okx_client = OKXClient(cfg)
    print("OKXClient: available")
except Exception as e:
    print("OKXClient: unavailable ->", e)

regime_engine = MarketRegimeEngine(cfg, okx_client)
analyzer = HistoricalAnalyzer(journal)
contrib = get_contribution_analyzer(
    sqlite_storage=storage, trade_journal=journal, config=cfg, okx_client=okx_client
)
optimizer = StrategyOptimizer(journal, cfg)
dqe = DecisionQualityEvaluator()

agent = IntelligentAnalysisAgent(
    config=cfg, trade_journal=journal, sqlite_storage=storage,
    market_regime_engine=regime_engine, decision_quality_evaluator=dqe,
    okx_client=okx_client, historical_analyzer=analyzer,
    contribution_analyzer=contrib, strategy_optimizer=optimizer,
)

analysis = asyncio.run(agent.generate_analysis_report(include_adx=True))
optimization = asyncio.run(agent.generate_optimization_report())

# 落盘第一份报告
os.makedirs("data/reports", exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
payload = {
    "report_title": "企业级智能交易记录分析报告（第一份）",
    "generated_at": datetime.now().isoformat(),
    "analysis_report": analysis,
    "optimization_report": optimization,
}
path = f"data/reports/intelligent_analysis_{ts}.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

# 打印结构化摘要
print("\n" + "=" * 60)
print("【市场状态】", json.dumps(analysis["market_state"], ensure_ascii=False))
print("\n【ADX 确认摘要】", json.dumps(analysis["adx_confirmation"].get("summary", {}), ensure_ascii=False))
adx_symbols = analysis["adx_confirmation"].get("symbols", [])
for s in adx_symbols:
    print(f"   {s.get('symbol')}: adx={s.get('adx')} direction={s.get('direction')} "
          f"confirmation={s.get('confirmation')} mtf={s.get('mtf_agreement')}")

print("\n【策略方向】")
for d in analysis["strategy_direction"].get("strategy_directions", []):
    print(f"   {d['strategy']}: 健康度={d['health_grade']} 方向={d['direction']} "
          f"胜率={d['win_rate']} 盈亏比={d['profit_factor']} PnL={d['total_pnl']}")

print("\n【信号质量】", json.dumps(analysis["signal_quality"].get("summary", {}), ensure_ascii=False))

print("\n【分析摘要】", json.dumps(analysis["summary"], ensure_ascii=False))
print("\n【优化摘要】", json.dumps(optimization["summary"], ensure_ascii=False))
print("\n【优先级行动项】")
for a in optimization.get("priority_actions", []):
    print(f"   [{a.get('priority')}] {a.get('type')}: {a.get('reason', '')}")

print("\n已保存报告:", path)
