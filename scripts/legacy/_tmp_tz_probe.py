import base64, hashlib, hmac, os, time, json
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OKX_API_KEY")
secret = os.getenv("OKX_SECRET_KEY")
passphrase = os.getenv("OKX_PASSPHRASE")
proxies = {"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"}

def okx_get(path):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    msg = ts + "GET" + path
    sign = base64.b64encode(hmac.new(secret.encode(), msg.encode(), hashlib.sha256).digest()).decode()
    headers = {
        "OK-ACCESS-KEY": api_key,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": ts,
        "OK-ACCESS-PASSPHRASE": passphrase,
        "Content-Type": "application/json",
    }
    return requests.get("https://www.okx.com" + path, headers=headers, proxies=proxies, timeout=20).json()

# 取 NEAR 账单
data = okx_get("/api/v5/account/bills?instType=SWAP&instId=NEAR-USDT-SWAP&limit=20")
print("now local:", datetime.now())
print("now utc:", datetime.utcnow())
bills = data.get("data", [])
for b in bills:
    ts = int(b.get("ts", "0"))
    dt_local = datetime.fromtimestamp(ts / 1000)
    dt_utc = datetime.utcfromtimestamp(ts / 1000)
    print(f"subType={b.get('subType')} pnl={b.get('pnl')} ts_local={dt_local} ts_utc={dt_utc}")
