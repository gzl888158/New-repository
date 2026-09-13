"""检查IP白名单和OKX API连通性"""
import requests
import os
import hmac
import hashlib
import base64
import time
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('OKX_API_KEY')
secret = os.getenv('OKX_SECRET_KEY')
passphrase = os.getenv('OKX_PASSPHRASE')

proxies = {'http': 'http://127.0.0.1:7897', 'https': 'http://127.0.0.1:7897'}

# 1. 获取本地直连公网IP
print("=== 1. 本地直连公网IP ===")
try:
    r = requests.get('http://myip.ipip.net', timeout=10, proxies={'http': None, 'https': None})
    print(r.text.strip())
except Exception as e:
    print(f"获取失败: {e}")

# 2. 获取代理出口IP（OKX看到的IP）
print("\n=== 2. 代理出口IP（OKX看到的IP）===")
try:
    r = requests.get('https://api.ip.sb/ip', proxies=proxies, timeout=15)
    print(f"代理出口IP: {r.text.strip()}")
except Exception as e:
    print(f"api.ip.sb 失败: {e}")
    try:
        r = requests.get('https://ifconfig.me/ip', proxies=proxies, timeout=15)
        print(f"代理出口IP: {r.text.strip()}")
    except Exception as e2:
        print(f"ifconfig.me 也失败: {e2}")

# 3. 测试OKX公共API
print("\n=== 3. OKX公共API测试 ===")
try:
    r = requests.get('https://www.okx.com/api/v5/public/time', proxies=proxies, timeout=15)
    print(f"状态: {r.status_code}, 响应: {r.json().get('msg', 'OK')}")
except Exception as e:
    print(f"失败: {e}")

# 4. 测试OKX私有API（验证IP白名单）
print("\n=== 4. OKX私有API测试（验证IP白名单）===")
try:
    ts = time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime())
    method = 'GET'
    path = '/api/v5/account/balance'
    msg = ts + method + path
    sign = base64.b64encode(
        hmac.new(secret.encode(), msg.encode(), hashlib.sha256).digest()
    ).decode()

    headers = {
        'OK-ACCESS-KEY': api_key,
        'OK-ACCESS-SIGN': sign,
        'OK-ACCESS-TIMESTAMP': ts,
        'OK-ACCESS-PASSPHRASE': passphrase
    }

    r = requests.get('https://www.okx.com' + path, headers=headers, proxies=proxies, timeout=15)
    d = r.json()
    print(f"HTTP状态: {r.status_code}")
    print(f"API Code: {d.get('code')}, Msg: {d.get('msg')}")

    if d.get('code') == '0' and d.get('data'):
        bal = d['data'][0]
        print(f"✅ IP白名单验证通过！")
        print(f"   账户权益: {bal.get('totalEq')} USDT")
        print(f"   保证金: {bal.get('margin')} USDT")
    elif d.get('code') == '50013':
        print(f"❌ IP白名单不匹配！当前IP不在白名单中。")
        print(f"   需要在OKX网站更新API Key的IP白名单")
    else:
        print(f"⚠️ 其他错误: {d.get('msg')}")
except Exception as e:
    print(f"请求失败: {e}")

print("\n=== 总结 ===")
print("如果私有API返回code=0，说明IP白名单正常，无需修改。")
print("如果返回code=50013，需要登录OKX网站更新API Key的IP白名单。")
