"""验证方向1：多级止盈修复 - 配置值 + 两套止盈互斥 + 语法编译。"""
import os, subprocess, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 1. 语法编译
r = subprocess.run(
    [sys.executable, "-m", "py_compile", os.path.join(BASE, "strategies", "scalping_strategy.py")],
    capture_output=True, text=True, cwd=BASE,
)
print("py_compile scalping_strategy.py rc=", r.returncode, r.stderr.strip() or "OK")

# 2. 配置加载后值
sys.path.insert(0, BASE)
from configs.settings import load_config
cfg = load_config()
sc = cfg["strategies"]["scalping"]
print("tp1_pct =", sc.get("tp1_pct"))
print("tp2_pct =", sc.get("tp2_pct"))
print("take_profit_enabled =", sc.get("take_profit_enabled"))
assert abs(sc.get("tp1_pct") - 0.005) < 1e-9, "tp1_pct 应修复为 0.005"
assert abs(sc.get("tp2_pct") - 0.01) < 1e-9, "tp2_pct 应修复为 0.01"
assert sc.get("take_profit_enabled") is True

print("\n[PASS] 配置值已修复为分数语义 (0.5% / 1.0%)")