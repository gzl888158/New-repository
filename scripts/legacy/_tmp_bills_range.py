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

# 拉全部账单，看时间范围
CLOSE_LONG = {5, 9, 11}
CLOSE_SHORT = {6, 10, 12}
all_bills = []
before = None
for _ in range(200):
    path = "/api/v5/account/bills?instType=SWAP&limit=100"
    if before:
        path += f"&before={before}"
    d = okx_get(path)
    if not d or d.get("code") != "0":
        print("stop code", d.get("code"), d.get("msg"))
        break
    page = d.get("data", [])
    if not page:
        break
    all_bills.extend(page)
    oldest = min(int(b.get("ts","0")) for b in page)
    before = str(oldest)
    if len(page) < 100:
        break

print("total bills fetched:", len(all_bills))
ts_list = sorted(int(b.get("ts","0")) for b in all_bills)
print("min ts local:", datetime.fromtimestamp(ts_list[0]/1000))
print("max ts local:", datetime.fromtimestamp(ts_list[-1]/1000))

# 找 SOL short 平仓账单（subType 平空）在 7月20-26 之间的
sol_close = []
for b in all_bills:
    if b.get("instId") != "SOL-USDT-SWAP":
        continue
    st = int(b.get("subType","0") or 0)
    if st not in CLOSE_SHORT:
        continue
    pnl = float(b.get("pnl","0") or 0)
    if pnl == 0:
        continue
    dt = datetime.fromtimestamp(int(b["ts"])/1000)
    if datetime(2026,7,19) <= dt <= datetime(2026,7,27):
        sol_close.append((dt, st, pnl))
print("SOL short close bills 7/19-7/27:", len(sol_close))
for x in sol_close[:20]:
    print(x)
