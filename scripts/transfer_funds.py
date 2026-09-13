"""划转资金到合约账户"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.settings import load_config
from core.okx_client import OKXClient
import time

c = load_config()
k = OKXClient(c)

print("=" * 70)
print("当前账户资金")
print("=" * 70)

# 查询各个账户余额
path = "/api/v5/account/balance"
headers = k._get_headers("GET", path)
r = k._session.get(f"{k.rest_url}{path}", headers=headers)
data = r.json()

if data.get("code") == "0" and data.get("data"):
    acct = data["data"][0]
    print(f"totalEq={acct.get('totalEq')}")
    details = acct.get("details", [])
    
    spot_balance = 0
    funding_balance = 0
    contract_balance = 0
    
    for d in details:
        ccy = d.get("ccy")
        eq = float(d.get("eq", 0))
        if ccy == "USDT":
            print(f"  ccy={ccy} eq={eq:.2f} availBal={d.get('availBal')} frozen={d.get('frozenBal')}")
            # 判断账户类型（通过余额位置判断）
            if eq > 0:
                # 默认是资金账户
                spot_balance = eq
    
    # 查询合约账户余额（逐仓模式下合约账户可能不同）
    print("\n查询合约账户余额...")
    path2 = "/api/v5/account/positions?instType=SWAP"
    headers2 = k._get_headers("GET", path2)
    r2 = k._session.get(f"{k.rest_url}{path2}", headers=headers2)
    pos_data = r2.json()
    if pos_data.get("code") == "0":
        print(f"  合约持仓数: {len(pos_data.get('data', []))}")
        
    # 尝试划转
    print("\n" + "=" * 70)
    print("划转 USDT 到合约账户")
    print("=" * 70)
    
    # 资金账户 -> 合约账户 (USDT)
    transfer_body = {
        "ccy": "USDT",
        "amt": "120",
        "from": "6",  # 资金账户
        "to": "18"   # 合约账户
    }
    
    print(f"划转请求: {transfer_body}")
    
    path3 = "/api/v5/asset/transfer"
    body_str = json.dumps(transfer_body)
    headers3 = k._get_headers("POST", path3, body_str)
    r3 = k._session.post(f"{k.rest_url}{path3}", headers=headers3, data=body_str)
    result = r3.json()
    
    print(f"划转结果: {result}")
    
    # 等待1秒后验证
    time.sleep(1)
    
    # 再次查询
    r4 = k._session.get(f"{k.rest_url}{path}", headers=headers)
    data4 = r4.json()
    if data4.get("code") == "0" and data4.get("data"):
        acct4 = data4["data"][0]
        details4 = acct4.get("details", [])
        print("\n划转后余额:")
        for d in details4:
            if d.get("ccy") == "USDT":
                print(f"  USDT: eq={d.get('eq')} availBal={d.get('availBal')}")
                
        # 查询合约账户特定余额
        print("\n查询合约账户 USDT...")
        # 在逐仓模式下，资金在各合约下分别管理
        # 查询某个合约的保证金余额
        path5 = "/api/v5/account/positions?instId=ADA-USDT-SWAP"
        headers5 = k._get_headers("GET", path5)
        r5 = k._session.get(f"{k.rest_url}{path5}", headers=headers5)
        pos5 = r5.json()
        if pos5.get("code") == "0":
            print(f"  ADA-USDT-SWAP 持仓数据: {pos5.get('data')}")
            
        # 查询账户配置确认账户类型
        path6 = "/api/v5/account/config"
        headers6 = k._get_headers("GET", path6)
        r6 = k._session.get(f"{k.rest_url}{path6}", headers=headers6)
        cfg = r6.json()
        if cfg.get("code") == "0":
            print(f"\n账户配置: {cfg.get('data')}")

else:
    print(f"查询失败: {data}")