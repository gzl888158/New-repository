from dotenv import load_dotenv; load_dotenv()
from configs.settings import load_config
from core.okx_client import OKXClient
import json

config = load_config()
client = OKXClient(config)

for oid in ['3779582735011258368', '3779582742527451136']:
    r = client._make_request('POST', '/api/v5/trade/cancel-order', json.dumps({'instId': 'XRP-USDT-SWAP', 'ordId': oid}))
    print(f"Cancel {oid}: {r.get('code')}")
