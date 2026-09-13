"""将 USDT 转换为 USDC 并划转至合约账户"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.settings import load_config
from core.okx_client import OKXClient
import time

c = load_config()
k = OKXClient(c)

print("=" * 70)
print("步骤1: 查询当前 USDT 和 USDC 余额")
print("=" * 70)

path = "/api/v5/account/balance"
headers = k._get_headers("GET", path)
r = k._session.get(f"{k.rest_url}{path}", headers=headers)
data = r.json()

usdt_balance = 0
usdc_balance = 0

if data.get("code") == "0" and data.get("data"):
    details = data["data"][0].get("details", [])
    for d in details:
        if d.get("ccy") == "USDT":
            usdt_balance = float(d.get("eq", 0))
            print(f"  USDT: {usdt_balance:.2f}")
        elif d.get("ccy") == "USDC":
            usdc_balance = float(d.get("eq", 0))
            print(f"  USDC: {usdc_balance:.2f}")

# 获取 USDT-USDC 现货价格
print("\n" + "=" * 70)
print("步骤2: 获取 USDT-USDC 现货价格")
print("=" * 70)

ticker = k.get_ticker("USDT-USDC")
if ticker:
    bid_price = float(ticker.get("bidPx", 0))
    ask_price = float(ticker.get("askPx", 0))
    print(f"  USDT-USDC 买一: {bid_price}")
    print(f"  USDT-USDC 卖一: {ask_price}")
    
    # 用 USDT 买 USDC（以卖一价买入）
    buy_amount_usdt = usdt_balance - 1  # 留1 USDT备用
    buy_amount_usdc = buy_amount_usdt / ask_price
    print(f"\n  计划买入: {buy_amount_usdc:.4f} USDC")
    print(f"  花费: {buy_amount_usdt:.2f} USDT")
    
    # 下单买入 USDC
    print("\n" + "=" * 70)
    print("步骤3: 下单买入 USDC")
    print("=" * 70)
    
    order_body = {
        "instId": "USDT-USDC",
        "tdMode": "cash",  # 现货现金模式
        "side": "buy",
        "ordType": "market",
        "sz": str(buy_amount_usdt),  # 用USDT金额买入
        "ccy": "USDT"  # 计价货币
    }
    
    print(f"  下单请求: {order_body}")
    
    path_order = "/api/v5/trade/order"
    body_str = json.dumps(order_body)
    headers_order = k._get_headers("POST", path_order, body_str)
    r_order = k._session.post(f"{k.rest_url}{path_order}", headers=headers_order, data=body_str)
    result = r_order.json()
    
    print(f"  下单结果: {result}")
    
    if result.get("code") == "0" and result.get("data"):
        ord_id = result["data"][0].get("ordId")
        print(f"  订单ID: {ord_id}")
        
        # 等待成交
        print("\n  等待成交...")
        time.sleep(3)
        
        # 查询订单状态
        path_status = f"/api/v5/trade/order?instId=USDT-USDC&ordId={ord_id}"
        headers_status = k._get_headers("GET", path_status)
        r_status = k._session.get(f"{k.rest_url}{path_status}", headers=headers_status)
        status_data = r_status.json()
        
        if status_data.get("code") == "0" and status_data.get("data"):
            order_info = status_data["data"][0]
            print(f"  订单状态: {order_info.get('state')}")
            print(f"  成交数量: {order_info.get('fillSz')} USDC")
            print(f"  成交均价: {order_info.get('avgPx')}")
            
            # 查询新余额
            time.sleep(1)
            r_bal = k._session.get(f"{k.rest_url}{path}", headers=headers)
            bal_data = r_bal.json()
            if bal_data.get("code") == "0":
                details = bal_data["data"][0].get("details", [])
                print("\n  新余额:")
                for d in details:
                    if d.get("ccy") in ["USDT", "USDC"]:
                        print(f"    {d.get('ccy')}: {d.get('eq')}")
        
        # 划转 USDC 到合约账户
        print("\n" + "=" * 70)
        print("步骤4: 划转 USDC 到合约账户")
        print("=" * 70)
        
        transfer_body = {
            "ccy": "USDC",
            "amt": "120",
            "from": "6",  # 资金账户
            "to": "18"   # 合约账户
        }
        
        print(f"  划转请求: {transfer_body}")
        
        path_transfer = "/api/v5/asset/transfer"
        body_str2 = json.dumps(transfer_body)
        headers_transfer = k._get_headers("POST", path_transfer, body_str2)
        r_transfer = k._session.post(f"{k.rest_url}{path_transfer}", headers=headers_transfer, data=body_str2)
        trans_result = r_transfer.json()
        
        print(f"  划转结果: {trans_result}")
        
        # 验证
        time.sleep(1)
        r_final = k._session.get(f"{k.rest_url}{path}", headers=headers)
        final_data = r_final.json()
        if final_data.get("code") == "0":
            details = final_data["data"][0].get("details", [])
            print("\n  最终余额:")
            for d in details:
                if d.get("ccy") in ["USDT", "USDC"]:
                    print(f"    {d.get('ccy')}: eq={d.get('eq')} availBal={d.get('availBal')}")
                    
else:
    print(f"查询失败: {data}")