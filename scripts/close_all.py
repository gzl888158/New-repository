"""查询当前持仓并平掉所有仓位"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.settings import load_config
from core.okx_client import OKXClient
import time

c = load_config()
k = OKXClient(c)

print("=" * 70)
print("当前持仓")
print("=" * 70)
ps = k.get_positions()
active_positions = []
if ps:
    for p in ps:
        pos = float(p.get('pos', 0))
        if pos != 0:
            active_positions.append(p)
            print(f"  {p.get('instId'):20s} {p.get('posSide'):6s} qty={pos:>10.4f} "
                  f"entry={p.get('avgPx'):>10s} upl={p.get('upl'):>8s}")
else:
    print("  无持仓")

if not active_positions:
    print("\n[OK] 已无持仓，无需平仓")
else:
    print(f"\n共有 {len(active_positions)} 个持仓需要平仓")
    print("=" * 70)
    print("开始平仓")
    print("=" * 70)

    for p in active_positions:
        symbol = p.get('instId')
        pos_side = p.get('posSide')  # long or short
        qty = abs(float(p.get('pos', 0)))

        # 平仓：long仓位用sell，short仓位用buy
        close_side = "sell" if pos_side == "long" else "buy"

        print(f"平仓: {symbol} {pos_side} qty={qty} ... ", end="", flush=True)

        result = k.place_order(
            symbol=symbol,
            side=close_side,
            order_type="market",
            quantity=qty,
            leverage=int(p.get('lever', 1)),
            reduce_only=True,
            pos_side=pos_side
        )

        if result and result.get("ordId"):
            print(f"订单已提交 ordId={result.get('ordId')}")
        else:
            print(f"失败: {result}")

        time.sleep(0.5)  # 避免频率限制

    print("\n等待成交...")
    time.sleep(3)

    print("\n" + "=" * 70)
    print("检查最终持仓")
    print("=" * 70)
    final_ps = k.get_positions()
    remaining = [p for p in final_ps if float(p.get('pos', 0)) != 0]
    if remaining:
        for p in remaining:
            print(f"  {p.get('instId'):20s} {p.get('posSide'):6s} qty={float(p.get('pos', 0)):>10.4f}")
    else:
        print("  [OK] 所有持仓已平仓")