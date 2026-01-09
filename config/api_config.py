# -*- coding: utf-8 -*-
"""
欧易API配置文件（已打包密钥，无需登录直接运行）
⚠️ 安全警告：本文件包含敏感API密钥，禁止上传到GitHub、网盘、社交平台！
⚠️ 泄露风险：一旦密钥泄露，立即前往欧易API页面禁用该API，避免资产损失！
"""

# -------------------------- 已填入你的欧易API信息 --------------------------
API_KEY = "b9781f6b-08a0-469b-9674-ae3ff3fc9744"  # 你的OKX API Key
SECRET_KEY = "68AA1EAC3B22BEBA32765764D10F163D"  # 你的OKX Secret Key
PASSPHRASE = "Gzl123.@"  # 你的OKX API密码（已填写）
# --------------------------------------------------------------------------

# 欧易API固定配置（无需修改）
API_BASE_URL = "https://api.okx.com"  # 合约交易主网地址
TIMEOUT = 10  # 接口请求超时时间（单位：秒）
PROXIES = None  # 无需代理，留空即可

# 导出配置（供其他模块调用，无需修改）
__all__ = ["API_KEY", "SECRET_KEY", "PASSPHRASE", "API_BASE_URL", "TIMEOUT", "PROXIES"]