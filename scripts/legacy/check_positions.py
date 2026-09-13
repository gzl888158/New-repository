"""Check positions and pending orders"""
from dotenv import load_dotenv; load_dotenv()
from configs.settings import load_config
from core.okx_client import OKXClient

config = load_config()
client = OKXClient(config)

# Check open positions
pos = client.get_positions()
print("=== Positions ===")
for p in pos:
    print(f"  {p.get('instId')}: posSide={p.get('posSide')}, pos={p.get('pos')}, avgPx={p.get('avgPx')}, margin={p.get('margin')}, mgnMode={p.get('mgnMode')}, lever={p.get('lever')}")

# Check pending orders
orders = client.get_pending_orders()
print("=== Pending Orders ===")
for o in orders:
    print(f"  {o.get('instId')}: side={o.get('side')}, sz={o.get('sz')}, ordType={o.get('ordType')}, state={o.get('state')}")

# Check account config
acct_config = client.get_account_config()
print(f"\n=== Account Config ===\n  acctLv: {acct_config.get('acctLv') if acct_config else 'N/A'}")
if acct_config:
    print(f"  posMode: {acct_config.get('posMode')}")
