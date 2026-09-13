import sys, yaml
sys.path.insert(0, '.')
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
from core.okx_client import OKXClient
client = OKXClient(config)

pos = client.get_positions()
print('Positions:', len(pos) if pos else 0)
for p in (pos or []):
    print(f"  {p.get('instId','')}: pos={p.get('pos',0)}, side={p.get('posSide','')}")

acct = client.get_account_info()
if acct:
    for d in acct.get("details", []):
        if d.get("ccy") == "USDT":
            print(f"USDT: eq={d.get('eq')}, availBal={d.get('availBal')}, cashBal={d.get('cashBal')}, frozenBal={d.get('frozenBal')}")
