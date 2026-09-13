"""直接划转USDT到合约账户"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.settings import load_config
from core.okx_client import OKXClient

c = load_config()
k = OKXClient(c)

# 查询资金账户USDT余额
path = "/api/v5/account/balance"
headers = k._get_headers("GET", path)
r = k._session.get(f"{k.rest_url}{path}", headers=headers)
data = r.json()

if data.get("code") == "0" and data.get("data"):
    details = data["data"][0].get("details", [])
    print("所有资产:")
    for d in details:
        if float(d.get("eq", 0)) > 0.01:
            print(f"  {d.get('ccy')}: eq={d.get('eq')} avail={d.get('availBal')}")

# 尝试划转
transfer_body = {
    "ccy": "USDT",
    "amt": "120",
    "from": "6",  # 资金账户
    "to": "18"   # 合约账户
}

print(f"\n划转请求: {transfer_body}")

path_transfer = "/api/v5/asset/transfer"
body_str = json.dumps(transfer_body)
headers_transfer = k._get_headers("POST", path_transfer, body_str)
r_transfer = k._session.post(f"{k.rest_url}{path_transfer}", headers=headers_transfer, data=body_str)
trans_result = r_transfer.json()

print(f"划转结果: {trans_result}")

# 也尝试其他账户类型
print("\n尝试其他账户类型...")
for from_type in ["6", "18", "8"]:
    for to_type in ["6", "18", "8"]:
        if from_type == to_type:
            continue
        transfer_body2 = {
            "ccy": "USDT",
            "amt": "1",
            "from": from_type,
            "to": to_type
        }
        body_str2 = json.dumps(transfer_body2)
        headers2 = k._get_headers("POST", path_transfer, body_str2)
        r2 = k._session.post(f"{k.rest_url}{path_transfer}", headers=headers2, data=body_str2)
        result2 = r2.json()
        if result2.get("code") == "0":
            print(f"  from={from_type} to={to_type}: SUCCESS")
        else:
            msg = result2.get("msg", "")
            if "Insufficient" not in msg:
                print(f"  from={from_type} to={to_type}: {result2.get('code')}")