"""
可插拔短信发送器（验证码短信）
================================
支持三种 provider：
- console  : 控制台/日志模式，不真实发送短信，验证码打到日志（本地联调/无短信服务商时使用）
- aliyun   : 阿里云短信 Dysmsapi（RPC 签名 HMAC-SHA1）
- tencent  : 腾讯云短信 SMS（TC3-HMAC-SHA256 签名）

配置来自 config.yaml 的 auth.sms 段，由 dashboard_api.py 在初始化时注入。
"""
import base64
import hashlib
import hmac
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Tuple

import requests
from loguru import logger


class SmsSendError(Exception):
    """短信发送失败异常。"""


class SmsSender:
    """短信验证码发送器（根据 provider 分发）。"""

    def __init__(self, config: dict = None):
        config = config or {}
        self.provider = str(config.get("provider", "console")).lower()
        self.aliyun = config.get("aliyun", {}) or {}
        self.tencent = config.get("tencent", {}) or {}
        logger.info(f"SmsSender initialized: provider={self.provider}")

    def send_verification_code(self, phone: str, code: str) -> Tuple[bool, str]:
        """发送验证码短信，返回 (是否成功, 说明信息)。"""
        if self.provider == "console":
            return self._send_console(phone, code)
        if self.provider == "aliyun":
            return self._send_aliyun(phone, code)
        if self.provider == "tencent":
            return self._send_tencent(phone, code)
        return False, f"不支持的短信服务商: {self.provider}"

    # ── console 模式 ──────────────────────────────────────────
    def _send_console(self, phone: str, code: str) -> Tuple[bool, str]:
        # 脱敏手机号，避免日志泄露完整号码
        masked = phone[:3] + "****" + phone[-4:]
        logger.warning(f"[SMS-CONSOLE] 验证码 {code} -> {masked} (未真实发送，仅日志模式)")
        # 返回 code 供本地联调时在响应中回显（生产 aliyun/tencent 不会回显）
        return True, f"验证码已生成（console 模式）: {code}"

    # ── 阿里云 Dysmsapi ───────────────────────────────────────
    def _send_aliyun(self, phone: str, code: str) -> Tuple[bool, str]:
        access_key_id = self.aliyun.get("access_key_id", "")
        access_key_secret = self.aliyun.get("access_key_secret", "")
        sign_name = self.aliyun.get("sign_name", "")
        template_code = self.aliyun.get("template_code", "")
        if not all([access_key_id, access_key_secret, sign_name, template_code]):
            return False, "阿里云短信未配置（access_key_id/access_key_secret/sign_name/template_code）"

        params = {
            "Action": "SendSms",
            "Version": "2017-05-25",
            "Format": "JSON",
            "RegionId": "cn-hangzhou",
            "SignatureMethod": "HMAC-SHA1",
            "SignatureVersion": "1.0",
            "SignatureNonce": str(uuid.uuid4()),
            "Timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "AccessKeyId": access_key_id,
            "PhoneNumbers": phone,
            "SignName": sign_name,
            "TemplateCode": template_code,
            "TemplateParam": json.dumps({"code": code}, ensure_ascii=False),
        }
        signature = self._aliyun_signature(params, access_key_secret)
        params["Signature"] = signature
        try:
            resp = requests.post(
                "https://dysmsapi.aliyuncs.com/",
                data=params,
                timeout=10,
            )
            data = resp.json()
            if data.get("Code") == "OK":
                return True, "验证码已发送"
            return False, f"阿里云短信发送失败: {data.get('Message', data.get('Code', '未知错误'))}"
        except Exception as e:
            logger.error(f"aliyun sms error: {e}")
            return False, f"阿里云短信异常: {e}"

    @staticmethod
    def _percent_encode(value: str) -> str:
        import urllib.parse
        return urllib.parse.quote(str(value), safe="~")

    @staticmethod
    def _aliyun_signature(params: dict, secret: str) -> str:
        """阿里云 RPC 签名（HMAC-SHA1，POST）。"""
        import urllib.parse
        sorted_params = sorted(params.items(), key=lambda kv: kv[0])
        query_string = "&".join(
            f"{urllib.parse.quote(str(k), safe='~')}={urllib.parse.quote(str(v), safe='~')}"
            for k, v in sorted_params
        )
        string_to_sign = "POST&%2F&" + urllib.parse.quote(query_string, safe="~")
        key = (secret + "&").encode("utf-8")
        signature = base64.b64encode(
            hmac.new(key, string_to_sign.encode("utf-8"), hashlib.sha1).digest()
        ).decode("utf-8")
        return signature

    # ── 腾讯云 SMS ────────────────────────────────────────────
    def _send_tencent(self, phone: str, code: str) -> Tuple[bool, str]:
        secret_id = self.tencent.get("secret_id", "")
        secret_key = self.tencent.get("secret_key", "")
        sdk_app_id = self.tencent.get("sdk_app_id", "")
        sign_name = self.tencent.get("sign_name", "")
        template_id = self.tencent.get("template_id", "")
        region = self.tencent.get("region", "ap-guangzhou")
        if not all([secret_id, secret_key, sdk_app_id, sign_name, template_id]):
            return False, "腾讯云短信未配置（secret_id/secret_key/sdk_app_id/sign_name/template_id）"

        payload = json.dumps({
            "PhoneNumberSet": [f"+86{phone}"],
            "SmsSdkAppId": sdk_app_id,
            "SignName": sign_name,
            "TemplateId": template_id,
            "TemplateParamSet": [code],
        })
        headers = self._tencent_headers(secret_id, secret_key, payload, region)
        try:
            resp = requests.post(
                "https://sms.tencentcloudapi.com/",
                headers=headers,
                data=payload,
                timeout=10,
            )
            data = resp.json()
            resp_data = data.get("Response", {})
            if "Error" in resp_data:
                err = resp_data["Error"]
                return False, f"腾讯云短信发送失败: {err.get('Message', err.get('Code', '未知错误'))}"
            return True, "验证码已发送"
        except Exception as e:
            logger.error(f"tencent sms error: {e}")
            return False, f"腾讯云短信异常: {e}"

    @staticmethod
    def _tencent_headers(secret_id: str, secret_key: str, payload: str, region: str) -> dict:
        """腾讯云 TC3-HMAC-SHA256 签名。"""
        service = "sms"
        host = "sms.tencentcloudapi.com"
        algorithm = "TC3-HMAC-SHA256"
        timestamp = int(time.time())
        date = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")

        # 1. CanonicalRequest
        canonical_headers = f"content-type:application/json; charset=utf-8\nhost:{host}\nx-tc-action:sendSms\n"
        signed_headers = "content-type;host;x-tc-action"
        hashed_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        canonical_request = (
            "POST\n/\n\n" + canonical_headers + "\n" + signed_headers + "\n" + hashed_payload
        )

        # 2. StringToSign
        credential_scope = f"{date}/{service}/tc3_request"
        hashed_canonical = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
        string_to_sign = f"{algorithm}\n{timestamp}\n{credential_scope}\n{hashed_canonical}"

        # 3. Signature
        def _hmac(key, msg):
            return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

        secret_date = _hmac(("TC3" + secret_key).encode("utf-8"), date)
        secret_service = _hmac(secret_date, service)
        secret_signing = _hmac(secret_service, "tc3_request")
        signature = hmac.new(
            secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        authorization = (
            f"{algorithm} Credential={secret_id}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        return {
            "Authorization": authorization,
            "Content-Type": "application/json; charset=utf-8",
            "Host": host,
            "X-TC-Action": "SendSms",
            "X-TC-Timestamp": str(timestamp),
            "X-TC-Version": "2021-01-11",
            "X-TC-Region": region,
        }
