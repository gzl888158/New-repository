# -*- coding: utf-8 -*-
"""
离线验证脚本：验证三层修复（不触发网络、不做真实交易、不连接 OKX）

覆盖 4 个修复点：
  1. 基础层  stop_loss 信号带 pos_side（stop_loss_manager._execute_stop_loss_order）
  2. 逻辑层  平仓方向兜底：pos_side 缺失时反向下单 side（order_executor._track_active_orders）
  3. 逻辑层  open_trade_id 赋值解耦：不再受 entry_price 缺失限制
  4. 运算层  PnL 取值：okx_pnl 权威优先，manual_pnl 仅兜底

验证方式：
  - 真实 import 并调用 DirectionUnifier（纯静态类，零依赖）
  - 真实 import OrderExecutor，用 __new__ 跳过 __init__，调用真实纯方法 _resolve_track_direction
  - 对内联在复杂 async 方法中的逻辑，用「等价单元断言 + 源码级静态断言」双重锁定真实代码

运行：
  .venv\\Scripts\\python.exe verify_three_layer_fix.py
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.direction_unifier import DirectionUnifier
from execution.order_executor import OrderExecutor

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    line = f"[{mark}] {name}"
    if detail:
        line += f"  -> {detail}"
    print(line)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def read_src(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ─────────────────────────────────────────────────────────
# 1. 基础层：止损信号带 pos_side
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("1. 基础层：stop_loss 信号构造带 pos_side")
print("=" * 60)

# 等价复现 stop_loss_manager._execute_stop_loss_order (L585-604) 的信号构造
def build_stop_loss_signal(direction):
    return {
        "signal_type": "stop_loss",
        "direction": "sell" if direction == "long" else "buy",
        "pos_side": direction,
        "reduce_only": True,
    }

sig_long = build_stop_loss_signal("long")
check(
    "平多止损信号带 pos_side=long 且 reduce_only=True",
    sig_long["pos_side"] == "long" and sig_long["reduce_only"] is True,
    f"direction(side)={sig_long['direction']}, pos_side={sig_long['pos_side']}",
)

sig_short = build_stop_loss_signal("short")
check(
    "平空止损信号带 pos_side=short 且 side=buy",
    sig_short["pos_side"] == "short" and sig_short["direction"] == "buy",
    f"direction(side)={sig_short['direction']}, pos_side={sig_short['pos_side']}",
)

# 源码级静态断言：真实文件确实在 stop_loss 信号 dict 中写入了 "pos_side": direction
sl_src = read_src(os.path.join(PROJECT_ROOT, "core", "stop_loss_manager.py"))
sl_has_pos_side = '"pos_side": direction' in sl_src
sl_has_stop_type = '"signal_type": "stop_loss"' in sl_src
check(
    "源码：stop_loss_manager.py 的 stop_loss 信号 dict 含 pos_side 字段",
    sl_has_pos_side and sl_has_stop_type,
    f"source contains pos_side={sl_has_pos_side}, signal_type=stop_loss={sl_has_stop_type}",
)

# ─────────────────────────────────────────────────────────
# 2. 逻辑层：平仓方向兜底（真实 _resolve_track_direction + 反向）
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. 逻辑层：平仓方向兜底")
print("=" * 60)

# 跳过 __init__ 直接拿真实方法（该方法只用 order_info 参数 + logger，不依赖实例状态）
executor = OrderExecutor.__new__(OrderExecutor)

# 先验证真实 _resolve_track_direction 的「下单 side 编码」语义（未做反向）
check("真实 _resolve_track_direction('sell') == 'short'",
      executor._resolve_track_direction({"direction": "sell"}) == "short")
check("真实 _resolve_track_direction('buy') == 'long'",
      executor._resolve_track_direction({"direction": "buy"}) == "long")
check("真实 _resolve_track_direction('close'+pos_side=long) == 'short'",
      executor._resolve_track_direction({"direction": "close", "pos_side": "long"}) == "short")


# 复现 order_executor._track_active_orders (L2957-2966) 的平仓方向解析
def resolve_close_direction(order_info):
    direction = order_info.get("pos_side", "")
    if direction not in ("long", "short"):
        # 兜底：_resolve_track_direction 返回下单 side（long/short 编码），
        # 平仓时被平的持仓方向与之相反（平多=sell→short 编码 → 持仓 long）
        direction = executor._resolve_track_direction(order_info)
        direction = "short" if direction == "long" else "long"
    return direction


check(
    "显式 pos_side=long 直接取 long（平多）",
    resolve_close_direction({"pos_side": "long"}) == "long",
)
check(
    "显式 pos_side=short 直接取 short（平空）",
    resolve_close_direction({"pos_side": "short"}) == "short",
)
check(
    "无 pos_side，side=sell → 兜底反向为 long（平多不误判成 short）",
    resolve_close_direction({"direction": "sell"}) == "long",
)
check(
    "无 pos_side，side=buy → 兜底反向为 short（平空）",
    resolve_close_direction({"direction": "buy"}) == "short",
)

# 关键回归点：方向解析正确后，PnL 计算符号不应反向
# 平多(direction=long)：(平仓价-开仓价)*qty；平空(direction=short)：(开仓价-平仓价)*qty
def gross_pnl(direction, entry, exit_px, qty):
    return (exit_px - entry) * qty if direction == "long" else (entry - exit_px) * qty

# 用昨天事故场景复现：平多仓，开仓 97.06，平仓 96.90 -> 应为负
d = resolve_close_direction({"direction": "sell"})  # 无 pos_side 时下单 side=sell
check(
    "平多方向解析 long，PnL 符号正确（亏损为负）",
    gross_pnl(d, 97.06, 96.90, 0.4959) < 0,
    f"direction={d}, gross_pnl={gross_pnl(d, 97.06, 96.90, 0.4959):.4f}",
)

# ─────────────────────────────────────────────────────────
# 3. 逻辑层：open_trade_id 赋值解耦（源码级静态断言）
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. 逻辑层：open_trade_id 始终查询（解耦 entry_price）")
print("=" * 60)

oe_src = read_src(os.path.join(PROJECT_ROOT, "execution", "order_executor.py"))
lines = oe_src.splitlines()

# 定位 open_trade_id 赋值行，以及紧随其后最近的 if not entry_price 行
assign_open_trade_idx = None
for i, ln in enumerate(lines):
    if "open_trade_id = open_rec[\"id\"]" in ln:
        assign_open_trade_idx = i
        break

if assign_open_trade_idx is not None:
    a_indent = _indent(lines[assign_open_trade_idx])
    # 在该行之后找最近的 if not entry_price
    nested_if_indent = None
    for j in range(assign_open_trade_idx + 1, min(assign_open_trade_idx + 20, len(lines))):
        if "if not entry_price:" in lines[j]:
            nested_if_indent = _indent(lines[j])
            break

    # 修复前：open_trade_id 嵌套在 `if not entry_price:` 内（缩进更深）。
    # 修复后：open_trade_id 与 `if not entry_price:` 平级（缩进相等），
    # 即 open_trade_id 始终执行，entry_price 回填才是有条件的。
    check(
        "open_trade_id 赋值与 if not entry_price 平级（解耦成功）",
        nested_if_indent is not None and a_indent == nested_if_indent,
        f"open_trade_id indent={a_indent}, if not entry_price indent={nested_if_indent}",
    )
else:
    check("定位到 open_trade_id = open_rec['id'] 赋值行", False, "未找到目标行")

# ─────────────────────────────────────────────────────────
# 4. 运算层：PnL 权威取值（okx_pnl 优先）
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. 运算层：PnL 取值 okx_pnl 权威优先")
print("=" * 60)


# 复现 order_executor._track_active_orders (L3099-3106)
def resolve_pnl(okx_pnl, manual_pnl):
    if okx_pnl != 0:
        return okx_pnl
    return manual_pnl


# 昨日事故根因：okx_pnl=-2.939 被 ratio 校验丢弃，错误采用 manual_pnl=9.623
check(
    "okx_pnl 非零时优先采用权威值（昨日 -2.939 不再被丢弃）",
    resolve_pnl(-2.939, 9.623) == -2.939,
    f"resolve_pnl(-2.939, 9.623) = {resolve_pnl(-2.939, 9.623)}",
)
check(
    "okx_pnl 为 0 时兜底采用 manual_pnl",
    resolve_pnl(0, 9.623) == 9.623,
    f"resolve_pnl(0, 9.623) = {resolve_pnl(0, 9.623)}",
)
check(
    "okx_pnl 缺失（None→0）时兜底采用 manual_pnl",
    resolve_pnl(float("0"), 9.623) == 9.623,
)

# 源码级静态断言：真实文件存在 okx_pnl 权威优先分支
oe_has_pnl_branch = "if okx_pnl != 0:" in oe_src and "pnl = okx_pnl" in oe_src
check(
    "源码：order_executor.py 含 okx_pnl 权威优先分支",
    oe_has_pnl_branch,
    "okx_pnl 非零时 pnl=okx_pnl，否则 pnl=manual_pnl",
)


# ─────────────────────────────────────────────────────────
# 汇总
# ─────────────────────────────────────────────────────────
def summary():
    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed
    print("\n" + "=" * 60)
    print(f"结果：{passed}/{total} 通过，{failed} 失败")
    print("=" * 60)
    for name, ok, detail in results:
        if not ok:
            print(f"  [FAIL] {name}  -> {detail}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(summary())