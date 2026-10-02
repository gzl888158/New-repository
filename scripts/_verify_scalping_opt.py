"""验证方向4修复：scaler 回测参数分化 + 参数键对齐 config。"""
import os, sys, math

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from datetime import datetime, timedelta, timezone
from backtest.backtest_engine import BacktestEngine
from analysis.parameter_optimization.orchestrator import ParameterOptimizationOrchestrator

# 1. 生成合成 K 线（带噪声的正弦波动，含均值回归特征）
candles = []
t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
price = 100.0
for i in range(600):
    wave = math.sin(i / 20.0) * 3.0
    noise = math.sin(i / 7.0) * 0.8
    open_p = price
    close_p = 100.0 + wave + noise
    high_p = max(open_p, close_p) + 0.4
    low_p = min(open_p, close_p) - 0.4
    candles.append({
        "timestamp": t0 + timedelta(minutes=i),
        "open": open_p, "high": high_p, "low": low_p, "close": close_p,
        "volume": 1000.0,
    })
    price = close_p

engine = BacktestEngine({})

# 2. 不同参数组合的 summary 对比
def run(**kw):
    base = dict(rsi_period=4, rsi_oversold=30.0, rsi_overbought=70.0,
                profit_target_min=0.008, stop_loss=0.005, max_hold_minutes=10)
    base.update(kw)
    r = engine.run_scalping_with_candles(candles, "API3-USDT-SWAP", **base)
    s = r.summary()
    return (s.get("total_trades", 0), round(s.get("net_pnl", 0.0), 4))

a = run()
b = run(rsi_oversold=20.0, rsi_overbought=80.0)
c = run(profit_target_min=0.015, stop_loss=0.01)
d = run(rsi_period=14, max_hold_minutes=60)

print("base          trades/net_pnl =", a)
print("rsi 20/80     trades/net_pnl =", b)
print("tp0.015/sl0.01 trades/net_pnl =", c)
print("rsi14/mh60    trades/net_pnl =", d)

results = {a, b, c, d}
assert len(results) > 1, "fitness 仍无分化！不同参数回测结果完全相同"

# 3. 参数键对齐 config
import yaml
with open(os.path.join(BASE, "config.yaml"), encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
scalping_cfg_keys = set(cfg["strategies"]["scalping"].keys())

opt = ParameterOptimizationOrchestrator({})
param_keys = {p.name for p in opt.define_strategy_params("scalping")}
print("\n优化参数键:", sorted(param_keys))
missing = param_keys - scalping_cfg_keys
print("config 中缺失的键:", missing or "无")
assert not missing, f"优化参数键未对齐 config: {missing}"

print("\n[PASS] scalping 回测参数分化 + 参数键对齐 config")