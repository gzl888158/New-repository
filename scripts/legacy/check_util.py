"""验证修复后的资金利用率计算"""
import os
os.environ['DASHBOARD_TOKEN'] = 'AGLuB4FdRbMT7DMSaEHhAGLSXsCsgDWe'
from dashboard_api import fetch_okx_account, fetch_okx_positions, calculate_used_margin, load_config
load_config()

account_data = fetch_okx_account()
if account_data:
    total_equity = float(account_data.get("totalEq", 0) or 0)
    available_balance = 0.0
    usdt_frozen = 0.0
    for detail in account_data.get("details", []):
        if detail.get("ccy") == "USDT":
            available_balance = float(detail.get("availBal", 0) or 0)
            usdt_frozen = float(detail.get("frozenBal", 0) or 0)
            if total_equity <= 0:
                total_equity = float(detail.get("eq", 0) or 0)
            break
    if available_balance <= 0:
        available_balance = float(account_data.get("availEq", 0) or 0)
    # 原始计算方式
    used_margin_v1 = usdt_frozen if usdt_frozen > 0 else max(0.0, total_equity - available_balance)
    # 新增positions API计算方式（更准确）
    used_margin_v2 = calculate_used_margin()
    # 最终采用两者中较大值
    used_margin = max(used_margin_v1, used_margin_v2)
    utilization_rate_v1 = used_margin_v1 / total_equity if total_equity > 0 else 0.0
    utilization_rate_v2 = used_margin_v2 / total_equity if total_equity > 0 else 0.0
    utilization_rate = used_margin / total_equity if total_equity > 0 else 0.0

    print("=== 修复后的资金利用率 ===")
    print(f"总权益: {total_equity:.2f} USDT")
    print(f"可用余额: {available_balance:.2f} USDT")
    print(f"冻结保证金(frozenBal): {usdt_frozen:.2f} USDT")
    print(f"已用保证金(V1 frozenBal): {used_margin_v1:.2f} USDT (利用率 {utilization_rate_v1*100:.2f}%)")
    print(f"已用保证金(V2 positions API): {used_margin_v2:.2f} USDT (利用率 {utilization_rate_v2*100:.2f}%)")
    print(f"已用保证金(最终): {used_margin:.2f} USDT")
    print(f"资金利用率: {utilization_rate*100:.2f}%")
    print(f"目标利用率: 85.00%")
    print(f"闲置资金: {available_balance:.2f} USDT")
    print(f"状态: {'低' if utilization_rate < 0.5 else '正常' if utilization_rate < 0.95 else '高'}")

    # 显示持仓详情
    positions = fetch_okx_positions()
    print(f"\n=== 持仓详情 ({len(positions)}个) ===")
    for pos in positions:
        pos_qty = float(pos.get("pos", 0) or 0)
        if pos_qty == 0:
            continue
        symbol = pos.get("instId", "")
        margin = pos.get("margin", "")
        imr = pos.get("imr", "")
        lever = float(pos.get("lever", 1) or 1)
        notional = float(pos.get("notionalUsd", 0) or 0)
        m = float(margin) if margin else (float(imr) if imr else (notional / lever if lever > 0 and notional > 0 else 0))
        print(f"  {symbol}: pos={pos_qty}, margin={margin or 'N/A'}, imr={imr or 'N/A'}, lever={lever}, notional={notional:.2f}, calc_margin={m:.2f}")

