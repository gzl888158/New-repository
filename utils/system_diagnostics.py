"""
系统自诊断模块：启动时全面检查关键组件状态，快速发现配置/网络/依赖问题。
"""
import os
import sys
import platform
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class DiagLevel(Enum):
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


def _safe_float(value: Any, default: float = 0.0) -> float:
    """安全 float 转换：None/NaN/Inf/非法值回退 default。"""
    if value is None:
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v or v in (float("inf"), float("-inf")):
        return default
    return v


@dataclass
class DiagItem:
    name: str
    level: DiagLevel
    detail: str = ""
    suggestion: str = ""


@dataclass
class DiagResult:
    items: List[DiagItem] = field(default_factory=list)
    ok_count: int = 0
    warn_count: int = 0
    fail_count: int = 0

    @property
    def healthy(self) -> bool:
        return self.fail_count == 0

    def add(self, name: str, level: DiagLevel, detail: str = "", suggestion: str = ""):
        self.items.append(DiagItem(name, level, detail, suggestion))
        if level == DiagLevel.OK:
            self.ok_count += 1
        elif level == DiagLevel.WARN:
            self.warn_count += 1
        elif level == DiagLevel.FAIL:
            self.fail_count += 1

    def report(self) -> str:
        lines = []
        lines.append(f"OK={self.ok_count} WARN={self.warn_count} FAIL={self.fail_count}")
        lines.append("-" * 60)
        for item in self.items:
            icon = {"OK": "✓", "WARN": "⚠", "FAIL": "✗"}.get(item.level.value, "?")
            lines.append(f"  [{icon}] {item.name}: {item.detail}")
            if item.suggestion:
                lines.append(f"       → {item.suggestion}")
        lines.append("-" * 60)
        return "\n".join(lines)


class SystemDiagnostics:
    """系统自诊断器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    async def run_all(self) -> DiagResult:
        """运行所有诊断检查"""
        result = DiagResult()

        # 1. Python 版本
        self._check_python_version(result)

        # 2. 关键依赖
        self._check_dependencies(result)

        # 3. 配置文件
        await self._check_config(result)

        # 4. 磁盘空间
        self._check_disk_space(result)

        # 5. 数据目录
        self._check_data_dirs(result)

        # 6. 可选依赖
        self._check_optional_deps(result)

        # 7. 环境变量
        self._check_env_vars(result)

        return result

    def _check_python_version(self, r: DiagResult):
        """Python 版本检查"""
        v = sys.version_info
        ver_str = f"{v.major}.{v.minor}.{v.micro}"
        if v >= (3, 11):
            r.add("Python 版本", DiagLevel.OK, f"Python {ver_str}")
        elif v >= (3, 9):
            r.add("Python 版本", DiagLevel.WARN, f"Python {ver_str}", "建议升级到 3.11+")
        else:
            r.add("Python 版本", DiagLevel.FAIL, f"Python {ver_str}", "需要 Python 3.9+")

    def _check_dependencies(self, r: DiagResult):
        """关键依赖检查"""
        required = {
            "yaml": "PyYAML",
            "aiohttp": "aiohttp",
            "loguru": "loguru",
            "pydantic": "pydantic",
            "aiohttp": "aiohttp（WebSocket）",
        }
        missing = []
        checked = set()
        for mod, name in required.items():
            if mod in checked:
                continue
            checked.add(mod)
            try:
                __import__(mod)
            except ImportError:
                missing.append(name)

        if not missing:
            r.add("核心依赖", DiagLevel.OK, "全部就绪")
        else:
            r.add("核心依赖", DiagLevel.FAIL, f"缺少: {', '.join(missing)}",
                  f"pip install {' '.join(m.split(chr(40))[0].strip() for m in missing)}")

    async def _check_config(self, r: DiagResult):
        """配置文件检查"""
        from configs.settings import load_config

        try:
            validated = load_config()
            # 检查策略配置
            strategies = self.config.get("strategies", {})
            enabled = [k for k, v in strategies.items() if v.get("enabled")]
            r.add("配置验证", DiagLevel.OK, f"通过, 启用策略: {len(enabled)}个 ({', '.join(enabled) or '无'})")

            # 检查交易对
            tier1 = self.config.get("currencies", {}).get("tier1_symbols", [])
            tier2 = self.config.get("currencies", {}).get("tier2_symbols", [])
            tier3 = self.config.get("currencies", {}).get("tier3_symbols", [])
            total = len(tier1) + len(tier2) + len(tier3)
            r.add("交易对配置", DiagLevel.OK if total > 0 else DiagLevel.WARN,
                  f"T1={len(tier1)} T2={len(tier2)} T3={len(tier3)} 总计={total}")

            # 检查 API 密钥数量（兼容 Pydantic model 和 dict）
            api_keys = []
            if hasattr(validated, 'okx'):
                api_keys = validated.okx.api_keys or []
                if validated.okx.api_key and not validated.okx.api_key.startswith("${"):
                    api_keys = [{"api_key": validated.okx.api_key}] + api_keys
            elif isinstance(validated, dict):
                okx = validated.get("okx", {})
                api_keys = okx.get("api_keys", [])
            api_key_count = len(api_keys)
            r.add("API 密钥", DiagLevel.OK if api_key_count >= 2 else DiagLevel.WARN,
                  f"{api_key_count}组密钥", "建议配置 2+ 组密钥以支持故障切换")

        except Exception as e:
            r.add("配置验证", DiagLevel.FAIL, str(e)[:80], "检查 config.yaml 格式和参数范围")

    def _check_disk_space(self, r: DiagResult):
        """磁盘空间检查"""
        try:
            import shutil
            usage = shutil.disk_usage(os.getcwd())
            free_gb = usage.free / (1024 ** 3)
            total_gb = usage.total / (1024 ** 3)
            if free_gb < 1:
                r.add("磁盘空间", DiagLevel.FAIL, f"剩余 {free_gb:.1f}GB / {total_gb:.0f}GB",
                      "清理磁盘空间以避免日志写入失败")
            elif free_gb < 5:
                r.add("磁盘空间", DiagLevel.WARN, f"剩余 {free_gb:.1f}GB / {total_gb:.0f}GB")
            else:
                r.add("磁盘空间", DiagLevel.OK, f"剩余 {free_gb:.1f}GB / {total_gb:.0f}GB")
        except Exception:
            r.add("磁盘空间", DiagLevel.WARN, "无法检测")

    def _check_data_dirs(self, r: DiagResult):
        """数据目录检查"""
        for d in ["logs", "data"]:
            if not os.path.isdir(d):
                try:
                    os.makedirs(d, exist_ok=True)
                    r.add(f"目录 {d}", DiagLevel.WARN, "已自动创建")
                except Exception as e:
                    r.add(f"目录 {d}", DiagLevel.FAIL, str(e))
            elif not os.access(d, os.W_OK):
                r.add(f"目录 {d}", DiagLevel.FAIL, "无写入权限")
            else:
                r.add(f"目录 {d}", DiagLevel.OK, "可读写")

        # 检查数据库
        db_path = self.config.get("sqlite", {}).get("db_path", "./data/trading.db")
        if os.path.exists(db_path):
            size_mb = os.path.getsize(db_path) / (1024 * 1024)
            r.add("交易数据库", DiagLevel.OK, f"{size_mb:.1f}MB")
        else:
            r.add("交易数据库", DiagLevel.OK, "首次运行，将自动创建")

    def _check_optional_deps(self, r: DiagResult):
        """可选依赖检查"""
        optional = {
            "scipy": "scipy（组合优化）",
            "sklearn": "scikit-learn（ML模型）",
            "redis": "redis（分布式缓存）",
            "psutil": "psutil（系统监控）",
        }
        for mod, desc in optional.items():
            try:
                __import__(mod)
                r.add(desc, DiagLevel.OK, "已安装")
            except ImportError:
                r.add(desc, DiagLevel.WARN, "未安装", f"安装后可启用对应功能: pip install {mod}")

    def _check_env_vars(self, r: DiagResult):
        """环境变量检查"""
        # 检查代理设置
        proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
        if proxy:
            r.add("HTTP 代理", DiagLevel.OK, proxy[:60])
        else:
            r.add("HTTP 代理", DiagLevel.WARN, "未设置，直连可能受限")

    async def check_connectivity(self) -> DiagResult:
        """网络连通性检查（需要网络）"""
        result = DiagResult()
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                # 测试 OKX API
                try:
                    async with session.get("https://www.okx.com/api/v5/public/time",
                                           timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get("code") == "0":
                                result.add("OKX API", DiagLevel.OK, f"连通, 服务器时间: {data['data'][0]['ts']}")
                            else:
                                result.add("OKX API", DiagLevel.WARN, f"返回异常: code={data.get('code')}")
                        else:
                            result.add("OKX API", DiagLevel.FAIL, f"HTTP {resp.status}")
                except Exception as e:
                    result.add("OKX API", DiagLevel.FAIL, str(e)[:60],
                               "检查代理设置或网络连接")

                # 测试 WebSocket
                try:
                    import aiohttp
                    async with session.ws_connect("wss://ws.okx.com:8443/ws/v5/public",
                                                   timeout=5, autoclose=True) as ws:
                        await ws.close()
                    result.add("OKX WebSocket", DiagLevel.OK, "连通")
                except Exception as e:
                    result.add("OKX WebSocket", DiagLevel.FAIL, str(e)[:60])
        except ImportError:
            result.add("网络检查", DiagLevel.WARN, "aiohttp 不可用，跳过")
        return result

    async def check_account(self) -> Dict[str, Any]:
        """账户状态检查"""
        status = {"checked": False, "balance_usdt": None, "position_count": 0}
        try:
            from core.okx_client import OKXClient
            client = OKXClient(config=self.config)
            try:
                bal = await client.get_account_balance()
                if bal:
                    # 解析 USDT 余额
                    if isinstance(bal, list):
                        for item in bal:
                            if isinstance(item, dict) and item.get("ccy") == "USDT":
                                status["balance_usdt"] = _safe_float(item.get("availEq", 0))
                                break
                    elif isinstance(bal, dict):
                        status["balance_usdt"] = _safe_float(bal.get("totalEq", 0))
                status["checked"] = True
            finally:
                await client.close()
        except Exception as e:
            status["error"] = str(e)[:100]
        return status
