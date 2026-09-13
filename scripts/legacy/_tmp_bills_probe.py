import base64, hashlib, hmac, os, time, json
import requests
from dotenv import load_dotenv
from datetime import datetime

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
    r = requests.get("https://www.okx.com" + path, headers=headers, proxies=proxies, timeout=20)
    return r.json()

data = okx_get("/api/v5/account/bills?instType=SWAP&limit=20")
print("code:", data.get("code"), "msg:", data.get("msg"))
bills = data.get("data", [])
print("count:", len(bills))
for b in bills[:20]:
    print(json.dumps({k: b.get(k) for k in ('instId','posSide','side','type','pnl','fee','fillPx','ts')}, ensure_ascii=False))
