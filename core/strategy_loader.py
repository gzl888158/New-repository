"""
策略加载器 (StrategyLoader)
============================
动态策略加载、自动发现、配置验证、依赖解析、工厂模式的统一入口。

核心定位：
- 消除 scheduler.py 中硬编码的6处 import + 6处 ClassName() + 6处注入
- 利用 StrategyDescriptor.class_path 实现 importlib 动态加载
- 自动发现 strategies/ 包下的所有策略类
- 配置驱动：遍历 config["strategies"] 动态创建实例
- 依赖自动注入：统一处理 okx_client / redis_cache / adaptive_controller 等
- 拓扑排序：根据依赖关系确定启动顺序
- 预加载验证：加载前检查类存在性、必需参数、接口兼容性
- 延迟加载：非关键策略按需加载，加速启动
- 热加载就绪：策略实例支持运行时配置更新

使用方式：
    loader = StrategyLoader(config)
    instances = loader.load_all(
        okx_client=client, redis_cache=cache,
        adaptive_controller=ac, stop_loss_manager=slm, coordinator=coord
    )
    # instances: Dict[str, BaseStrategy] — 可直接注册到 StrategyManager
"""

import os
import sys
import json
import time
import hashlib
import importlib
import importlib.util
import inspect
import pkgutil
import threading
import concurrent.futures
from typing import Dict, Any, Optional, List, Tuple, Type, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger


# ============================================================
# 数据模型
# ============================================================

class LoadStatus(Enum):
    """加载状态"""
    PENDING = "pending"           # 待加载
    VALIDATING = "validating"     # 验证中
    LOADING = "loading"           # 加载中
    LOADED = "loaded"             # 已加载
    SKIPPED = "skipped"           # 跳过（disabled）
    FAILED = "failed"             # 加载失败
    DEFERRED = "deferred"         # 延迟加载


@dataclass
class LoadResult:
    """加载结果"""
    strategy_name: str
    status: LoadStatus
    instance: Optional[Any] = None
    class_path: str = ""
    error: Optional[str] = None
    load_duration_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    config_keys_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "status": self.status.value,
            "class_path": self.class_path,
            "error": self.error,
            "load_duration_ms": round(self.load_duration_ms, 2),
            "warnings": self.warnings,
            "has_instance": self.instance is not None,
        }


@dataclass
class LoadReport:
    """加载报告"""
    total: int = 0
    loaded: int = 0
    skipped: int = 0
    failed: int = 0
    deferred: int = 0
    total_duration_ms: float = 0.0
    results: List[LoadResult] = field(default_factory=list)
    start_order: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "loaded": self.loaded,
            "skipped": self.skipped,
            "failed": self.failed,
            "deferred": self.deferred,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "start_order": self.start_order,
            "timestamp": self.timestamp,
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class StrategyDiscoveryInfo:
    """策略发现信息"""
    name: str                       # 策略名称 (grid/trend/...)
    class_path: str                 # 完整类路径
    module_path: str                # 模块文件路径
    category: str = "contract"      # contract/spot/hybrid
    display_name: str = ""          # 显示名称
    description: str = ""           # 描述
    enabled_by_default: bool = True
    dependencies: List[str] = field(default_factory=list)
    config_schema_keys: List[str] = field(default_factory=list)
    constructor_params: List[str] = field(default_factory=list)  # 构造函数参数名
    has_start_method: bool = False
    has_stop_method: bool = False
    has_update_config: bool = False
    source: str = "auto"           # auto / manual / plugin

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "class_path": self.class_path,
            "module_path": self.module_path,
            "category": self.category,
            "display_name": self.display_name,
            "description": self.description,
            "enabled_by_default": self.enabled_by_default,
            "dependencies": self.dependencies,
            "config_schema_keys": self.config_schema_keys,
            "constructor_params": self.constructor_params,
            "has_start_method": self.has_start_method,
            "has_stop_method": self.has_stop_method,
            "has_update_config": self.has_update_config,
            "source": self.source,
        }


@dataclass
class InterfaceContract:
    """策略接口契约 — 定义策略类必须实现的方法签名"""
    required_methods: List[str] = field(default_factory=lambda: [
        "update_config",   # 热更新配置
    ])
    recommended_methods: List[str] = field(default_factory=lambda: [
        "start", "stop", "pause", "resume", "get_health",
    ])
    # 方法签名规格: 方法名 -> (参数列表, 是否必需)
    method_signatures: Dict[str, Tuple[List[str], bool]] = field(default_factory=lambda: {
        "update_config": (["config"], True),
        "start": ([], False),
        "stop": ([], False),
        "pause": ([], False),
        "resume": ([], False),
        "get_health": ([], False),
    })

    @classmethod
    def check(cls, strategy_class: Type) -> Tuple[bool, List[str], List[str]]:
        """
        验证策略类是否符合接口契约

        Returns:
            (passed, missing_required, missing_recommended)
        """
        contract = cls()
        missing_required = []
        missing_recommended = []

        for method_name, (params, is_required) in contract.method_signatures.items():
            method = getattr(strategy_class, method_name, None)
            if method is None or not callable(method):
                if is_required:
                    missing_required.append(method_name)
                else:
                    missing_recommended.append(method_name)
                continue

            # 检查方法签名
            try:
                sig = inspect.signature(method)
                actual_params = [p for p in sig.parameters if p not in ('self',)]
                for expected_param in params:
                    if expected_param not in actual_params:
                        missing_recommended.append(
                            f"{method_name}(missing param: '{expected_param}')"
                        )
                        break
            except Exception:
                pass

        passed = len(missing_required) == 0
        return passed, missing_required, missing_recommended


# ============================================================
# 发现结果磁盘缓存
# ============================================================

class DiscoveryDiskCache:
    """策略发现结果磁盘缓存 — 加速重启时的策略发现"""

    CACHE_DIR = "./data/strategy_loader"
    CACHE_FILE = "discovery_cache.json"
    CACHE_TTL_SECONDS = 3600  # 1 小时有效

    @classmethod
    def _get_cache_path(cls) -> str:
        return os.path.join(cls.CACHE_DIR, cls.CACHE_FILE)

    @classmethod
    def _compute_package_hash(cls, package_path: str, base_dir: str) -> str:
        """计算策略包的文件哈希，检测文件变更"""
        try:
            package = importlib.import_module(package_path)
            package_dir = package.__path__[0] if hasattr(package, '__path__') else base_dir
        except ImportError:
            package_dir = os.path.join(base_dir, package_path) if base_dir else package_path

        if not os.path.isdir(package_dir):
            return ""

        hasher = hashlib.sha256()
        for root, dirs, files in sorted(os.walk(package_dir)):
            dirs.sort()
            for fname in sorted(files):
                if not fname.endswith('.py') or fname.startswith('__'):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, 'rb') as f:
                        hasher.update(f.read())
                except Exception:
                    pass

        return hasher.hexdigest()

    @classmethod
    def save(cls, discoveries: List[StrategyDiscoveryInfo],
             package_path: str = "strategies", base_dir: str = "") -> bool:
        """缓存发现结果到磁盘"""
        try:
            os.makedirs(cls.CACHE_DIR, exist_ok=True)

            data = {
                "timestamp": datetime.now().isoformat(),
                "package_path": package_path,
                "package_hash": cls._compute_package_hash(package_path, base_dir),
                "discoveries": [d.to_dict() for d in discoveries],
                "count": len(discoveries),
            }

            with open(cls._get_cache_path(), 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.debug(f"Discovery cache saved ({len(discoveries)} entries)")
            return True
        except Exception as e:
            logger.debug(f"Failed to save discovery cache: {e}")
            return False

    @classmethod
    def load(cls, package_path: str = "strategies",
             base_dir: str = "") -> Optional[List[StrategyDiscoveryInfo]]:
        """从磁盘加载缓存的发现结果"""
        cache_path = cls._get_cache_path()
        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 检查 TTL
            cache_time = datetime.fromisoformat(data.get("timestamp", "2000-01-01T00:00:00"))
            age_seconds = (datetime.now() - cache_time).total_seconds()
            if age_seconds > cls.CACHE_TTL_SECONDS:
                logger.debug(f"Discovery cache expired ({age_seconds:.0f}s > {cls.CACHE_TTL_SECONDS}s)")
                return None

            # 检查文件哈希是否匹配
            current_hash = cls._compute_package_hash(package_path, base_dir)
            cached_hash = data.get("package_hash", "")
            if current_hash and cached_hash and current_hash != cached_hash:
                logger.debug("Discovery cache invalidated: package files changed")
                return None

            # 重建 StrategyDiscoveryInfo 对象
            discoveries = []
            for d in data.get("discoveries", []):
                discoveries.append(StrategyDiscoveryInfo(
                    name=d.get("name", ""),
                    class_path=d.get("class_path", ""),
                    module_path=d.get("module_path", ""),
                    category=d.get("category", "contract"),
                    display_name=d.get("display_name", ""),
                    description=d.get("description", ""),
                    enabled_by_default=d.get("enabled_by_default", True),
                    dependencies=d.get("dependencies", []),
                    config_schema_keys=d.get("config_schema_keys", []),
                    constructor_params=d.get("constructor_params", []),
                    has_start_method=d.get("has_start_method", False),
                    has_stop_method=d.get("has_stop_method", False),
                    has_update_config=d.get("has_update_config", False),
                    source=d.get("source", "cache"),
                ))

            logger.info(f"Discovery cache loaded ({len(discoveries)} entries, "
                        f"age={age_seconds:.0f}s)")
            return discoveries

        except Exception as e:
            logger.debug(f"Failed to load discovery cache: {e}")
            return None

    @classmethod
    def invalidate(cls) -> bool:
        """清除缓存"""
        cache_path = cls._get_cache_path()
        try:
            if os.path.exists(cache_path):
                os.remove(cache_path)
                logger.debug("Discovery cache invalidated")
            return True
        except Exception as e:
            logger.debug(f"Failed to invalidate cache: {e}")
            return False


# ============================================================
# 策略自动发现器
# ============================================================

class StrategyDiscovery:
    """
    策略自动发现器

    扫描 strategies/ 目录，自动发现所有策略类。
    支持：
    - 包扫描（pkgutil）
    - 类成员检测（inspect）
    - 构造函数签名提取
    - 策略元数据提取
    """

    # 已知的策略类后缀（用于自动发现时过滤）
    STRATEGY_CLASS_SUFFIXES = ("Strategy", "StrategyBase")

    # 策略基类标记（用于识别策略类）
    STRATEGY_BASE_MARKERS = (
        "PersistentStrategy",
        "BaseStrategy",
    )

    # 已知的策略名称→class_path 映射（兜底，自动发现失败时使用）
    KNOWN_STRATEGIES: Dict[str, Dict[str, Any]] = {
        "grid": {
            "class_path": "strategies.grid_strategy.GridStrategy",
            "display_name": "网格策略",
            "category": "contract",
            "description": "波动率自适应网格交易，区间内高抛低吸",
            "config_schema_keys": ["grid_count_max", "grid_count_min", "min_grid_spacing",
                                    "max_grid_spacing", "atr_multiplier", "leverage"],
        },
        "trend": {
            "class_path": "strategies.trend_strategy.TrendStrategy",
            "display_name": "趋势策略",
            "category": "contract",
            "description": "多周期共振趋势跟踪，金字塔加仓",
            "dependencies": ["market_regime"],
            "config_schema_keys": ["max_concurrent_positions", "initial_position_ratio",
                                    "addition_ratio", "max_additions", "min_signal_quality"],
        },
        "scalping": {
            "class_path": "strategies.scalping_strategy.ScalpingStrategy",
            "display_name": "抢单策略",
            "category": "contract",
            "description": "高频动量+均值回归短线交易",
            "config_schema_keys": ["max_positions", "max_hold_minutes", "min_signal_quality",
                                    "max_stop_loss_pct", "position_sizing_mode", "risk_lock_quality_boost"],
        },
        "arbitrage": {
            "class_path": "strategies.arbitrage_strategy.ArbitrageStrategy",
            "display_name": "套利策略",
            "category": "contract",
            "description": "资金费率/基差/跨品种套利",
            "dependencies": ["trend"],
            "config_schema_keys": ["funding_rate_threshold", "basis_threshold",
                                    "max_hold_hours", "hedge_leverage", "risk_lock_quality_boost"],
        },
        "spot_grid": {
            "class_path": "strategies.spot_grid_strategy.SpotGridStrategy",
            "display_name": "现货网格",
            "category": "spot",
            "description": "现货区间网格交易",
            "enabled_by_default": False,
        },
        "spot_martingale": {
            "class_path": "strategies.spot_martingale_strategy.SpotMartingaleStrategy",
            "display_name": "现货马丁格尔",
            "category": "spot",
            "description": "马丁格尔分批买入，反弹获利",
            "enabled_by_default": False,
        },
        "ema_trend": {
            "class_path": "strategies.ema_trend_strategy.EmaTrendStrategy",
            "display_name": "EMA趋势跟踪",
            "category": "contract",
            "description": "多周期EMA趋势跟踪，波动率过滤震荡",
            "enabled_by_default": False,
            "config_schema_keys": ["ema_fast", "ema_slow", "min_signal_quality"],
        },
        "donchian_breakout": {
            "class_path": "strategies.donchian_breakout_strategy.DonchianBreakoutStrategy",
            "display_name": "唐奇安通道突破",
            "category": "contract",
            "description": "N周期高低点突破，ATR尾随止盈止损",
            "enabled_by_default": False,
            "config_schema_keys": ["donchian_period", "min_signal_quality", "risk_lock_quality_boost"],
        },
        "momentum_rotation": {
            "class_path": "strategies.momentum_rotation_strategy.MomentumRotationStrategy",
            "display_name": "多币种动量轮动",
            "category": "contract",
            "description": "定时计算品种涨跌幅，做多强势、剔除弱势",
            "enabled_by_default": False,
            "config_schema_keys": ["lookback", "top_n", "bottom_n"],
        },
        "bollinger_mean_reversion": {
            "class_path": "strategies.bollinger_mean_reversion_strategy.BollingerMeanReversionStrategy",
            "display_name": "布林均值回归",
            "category": "contract",
            "description": "带趋势过滤器的布林均值回归，轻仓逆势博弈",
            "enabled_by_default": False,
            "config_schema_keys": ["boll_period", "boll_std_mult", "min_signal_quality"],
        },
        "oscillation_harvest": {
            "class_path": "strategies.oscillation_harvest_strategy.OscillationHarvestStrategy",
            "display_name": "震荡收割",
            "category": "contract",
            "description": "震荡区间支撑/阻力均值回归，反复收割区间波动（高周期趋势确认+触及次数区间验证+布林带双确认）",
            "dependencies": ["market_regime"],
            "enabled_by_default": False,
            "config_schema_keys": ["lookback_bars", "support_band_pct", "min_band_width_pct",
                                    "rsi_period", "rsi_oversold", "rsi_overbought", "min_signal_quality",
                                    "confirm_bar", "confirm_trend_threshold", "min_band_touch_count",
                                    "touch_proximity_pct", "use_bollinger_confirm", "boll_period", "boll_std_mult",
                                    "adaptive_enabled", "risk_lock_quality_boost"],
        },
    }

    @classmethod
    def discover_from_package(cls, package_path: str,
                               base_dir: str = None) -> List[StrategyDiscoveryInfo]:
        """
        从 strategies/ 包自动发现策略类

        Args:
            package_path: 包路径，如 'strategies'
            base_dir: 项目根目录，用于补全 sys.path

        Returns:
            发现的策略信息列表
        """
        results = []

        try:
            if base_dir and base_dir not in sys.path:
                sys.path.insert(0, base_dir)

            package = importlib.import_module(package_path)
            if not hasattr(package, '__path__'):
                return results

            package_dir = package.__path__[0]

            for _, module_name, is_pkg in pkgutil.iter_modules([package_dir]):
                if is_pkg or module_name.startswith('_'):
                    continue

                full_module_name = f"{package_path}.{module_name}"

                try:
                    module = importlib.import_module(full_module_name)
                    discovered = cls._extract_from_module(
                        module, full_module_name, package_dir
                    )
                    results.extend(discovered)
                except Exception as e:
                    logger.debug(f"Failed to scan module {full_module_name}: {e}")

        except ImportError as e:
            logger.debug(f"Package '{package_path}' import failed (may not exist): {e}")

        return results

    @classmethod
    def _extract_from_module(cls, module, module_path: str,
                              package_dir: str) -> List[StrategyDiscoveryInfo]:
        """从模块中提取策略类"""
        results = []

        for attr_name in dir(module):
            obj = getattr(module, attr_name, None)
            if not inspect.isclass(obj):
                continue
            if obj.__module__ != module.__name__:
                continue  # 跳过导入的类

            # 检查是否为策略类
            if not cls._is_strategy_class(obj):
                continue

            # 提取信息
            name = cls._infer_strategy_name(attr_name, obj)
            info = cls._build_discovery_info(
                name, obj, attr_name, module_path, package_dir
            )
            if info:
                results.append(info)

        return results

    @classmethod
    def _is_strategy_class(cls, obj: Type) -> bool:
        """判断是否为策略类"""
        # 按名称后缀
        name = obj.__name__
        if name.endswith(cls.STRATEGY_CLASS_SUFFIXES):
            return True

        # 按基类标记
        for base in obj.__mro__:
            if base.__name__ in cls.STRATEGY_BASE_MARKERS:
                return True

        return False

    @classmethod
    def _infer_strategy_name(cls, class_name: str, obj: Type) -> str:
        """从类名推断策略名称"""
        # "GridStrategy" -> "grid"
        # "TrendStrategy" -> "trend"
        name = class_name
        for suffix in cls.STRATEGY_CLASS_SUFFIXES:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break

        # CamelCase → snake_case
        result = []
        for i, ch in enumerate(name):
            if ch.isupper() and i > 0:
                result.append('_')
            result.append(ch.lower())
        return ''.join(result).strip('_')

    @classmethod
    def _build_discovery_info(cls, name: str, obj: Type, class_name: str,
                                module_path: str, package_dir: str) -> Optional[StrategyDiscoveryInfo]:
        """构建策略发现信息"""
        # 提取构造函数参数
        try:
            sig = inspect.signature(obj.__init__)
            params = [p for p in sig.parameters if p not in ('self', 'args', 'kwargs')]
        except Exception:
            params = []

        # 获取源文件路径
        try:
            source_file = inspect.getfile(obj)
        except Exception:
            source_file = os.path.join(package_dir, f"{name}_strategy.py")

        # 获取已知元数据
        known = cls.KNOWN_STRATEGIES.get(name, {})

        return StrategyDiscoveryInfo(
            name=name,
            class_path=f"strategies.{module_path.split('.')[-1]}.{class_name}",
            module_path=source_file,
            category=known.get("category", "contract"),
            display_name=known.get("display_name", class_name),
            description=known.get("description", ""),
            enabled_by_default=known.get("enabled_by_default", True),
            dependencies=known.get("dependencies", []),
            config_schema_keys=known.get("config_schema_keys", []),
            constructor_params=params,
            has_start_method=hasattr(obj, 'start') and callable(getattr(obj, 'start', None)),
            has_stop_method=hasattr(obj, 'stop') and callable(getattr(obj, 'stop', None)),
            has_update_config=hasattr(obj, 'update_config') and callable(getattr(obj, 'update_config', None)),
            source="auto",
        )

    @classmethod
    def get_known_mapping(cls) -> Dict[str, str]:
        """获取已知策略的 name→class_path 映射（兜底用）"""
        return {k: v["class_path"] for k, v in cls.KNOWN_STRATEGIES.items()}


# ============================================================
# 策略验证器
# ============================================================

class StrategyValidator:
    """
    策略预加载验证器

    在实例化策略之前验证：
    - 策略类是否存在且可导入
    - 配置段是否存在且包含必要字段
    - 策略是否在配置中被启用
    - 构造函数参数是否满足
    - 严格模式：缺少配置键时报错
    - 接口契约：验证策略类方法签名
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._validation_cache: Dict[str, Tuple[bool, str]] = {}

        # 验证配置
        loader_cfg = self.config.get("strategy_loader", {})
        validation_cfg = loader_cfg.get("validation", {})
        self._strict_mode = validation_cfg.get("strict_mode", False)
        self._warn_missing_config = validation_cfg.get("warn_missing_config", True)
        self._check_constructor = validation_cfg.get("check_constructor", True)
        self._check_interface = validation_cfg.get("check_interface", True)

    def validate_descriptor(self, info: StrategyDiscoveryInfo) -> Tuple[bool, str, List[str]]:
        """
        验证策略描述符

        Returns:
            (is_valid, error_message, warnings)
        """
        warnings = []

        # 1. 检查类路径
        if not info.class_path:
            return False, "class_path is empty", warnings

        # 2. 检查类是否存在且可导入
        class_exists, class_error = self._check_class_exists(info.class_path)
        if not class_exists:
            return False, f"Class not found: {class_error}", warnings

        # 3. 检查配置段
        strategies_cfg = self.config.get("strategies", {})
        strategy_cfg = strategies_cfg.get(info.name, {})

        # 4. 检查 enabled 状态
        is_enabled = strategy_cfg.get("enabled", info.enabled_by_default)
        if not is_enabled:
            return False, f"Strategy '{info.name}' is disabled in config", warnings

        # 5. 检查关键配置键
        for key in info.config_schema_keys:
            if key not in strategy_cfg:
                msg = f"Missing config key: '{key}'"
                if self._strict_mode:
                    return False, msg, warnings
                elif self._warn_missing_config:
                    warnings.append(msg)

        # 6. 检查构造函数兼容性
        if self._check_constructor:
            if len(info.constructor_params) < 2:
                msg = f"Strategy constructor has only {len(info.constructor_params)} params (expected >= 2)"
                if self._strict_mode:
                    return False, msg, warnings
                warnings.append(msg)

            # 检查常见必需参数
            expected_params = {"config", "okx_client"}
            actual_params = set(info.constructor_params)
            missing_params = expected_params - actual_params
            if missing_params:
                msg = f"Constructor missing expected params: {missing_params}"
                if self._strict_mode:
                    return False, msg, warnings
                warnings.append(msg)

        # 7. 检查接口契约
        if self._check_interface:
            try:
                cls = self._load_class_safe(info.class_path)
                if cls:
                    passed, missing_required, missing_recommended = InterfaceContract.check(cls)
                    if not passed:
                        msg = f"Interface contract violation: missing required methods {missing_required}"
                        return False, msg, warnings
                    for rec in missing_recommended:
                        warnings.append(f"Recommended method missing: {rec}")
            except Exception as e:
                warnings.append(f"Interface check skipped: {e}")

        return True, "", warnings

    def _check_class_exists(self, class_path: str) -> Tuple[bool, str]:
        """检查类是否可以通过 importlib 导入"""
        cache_key = f"class:{class_path}"
        if cache_key in self._validation_cache:
            return self._validation_cache[cache_key]

        try:
            parts = class_path.rsplit('.', 1)
            if len(parts) != 2:
                return False, f"Invalid class path format: {class_path}"

            module_path, class_name = parts
            module = importlib.import_module(module_path)
            if not hasattr(module, class_name):
                return False, f"Class '{class_name}' not found in module '{module_path}'"

            cls = getattr(module, class_name)
            if not inspect.isclass(cls):
                return False, f"'{class_name}' is not a class"

            result = (True, "")
        except ImportError as e:
            result = (False, f"Module import failed: {e}")
        except Exception as e:
            result = (False, str(e))

        self._validation_cache[cache_key] = result
        return result

    def _load_class_safe(self, class_path: str) -> Optional[Type]:
        """安全加载类（失败返回 None）"""
        try:
            parts = class_path.rsplit('.', 1)
            if len(parts) != 2:
                return None
            module_path, class_name = parts
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name, None)
            return cls if inspect.isclass(cls) else None
        except Exception:
            return None

    def validate_all(self, discoveries: List[StrategyDiscoveryInfo]) -> Dict[str, Tuple[bool, str, List[str]]]:
        """批量验证"""
        results = {}
        for info in discoveries:
            results[info.name] = self.validate_descriptor(info)
        return results

    def get_config(self) -> Dict[str, Any]:
        """获取验证器配置"""
        return {
            "strict_mode": self._strict_mode,
            "warn_missing_config": self._warn_missing_config,
            "check_constructor": self._check_constructor,
            "check_interface": self._check_interface,
        }


# ============================================================
# 依赖解析器
# ============================================================

class DependencyResolver:
    """
    策略依赖解析器

    使用拓扑排序确定策略启动顺序。
    依赖关系定义在 StrategyDiscoveryInfo.dependencies 中。
    """

    @staticmethod
    def resolve(discoveries: List[StrategyDiscoveryInfo]) -> Tuple[List[str], Dict[str, Set[str]]]:
        """
        解析依赖，返回启动顺序

        Returns:
            (start_order, dependency_graph)
        """
        # 构建邻接表
        name_to_info = {d.name: d for d in discoveries}
        graph: Dict[str, Set[str]] = {}  # name -> {依赖它的策略}
        in_degree: Dict[str, int] = {}    # name -> 入度（依赖数）

        for info in discoveries:
            in_degree[info.name] = len(info.dependencies)
            graph.setdefault(info.name, set())

        for info in discoveries:
            for dep in info.dependencies:
                # 跳过外部依赖（如 market_regime）
                if dep not in name_to_info:
                    continue
                graph.setdefault(dep, set()).add(info.name)

        # Kahn 拓扑排序
        queue = [name for name, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for dependent in graph.get(node, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # 检测循环依赖
        if len(order) < len(discoveries):
            remaining = set(d.name for d in discoveries) - set(order)
            logger.warning(f"Circular dependency detected among: {remaining}")
            # 将剩余的策略追加到末尾（打破循环）
            order.extend(remaining)

        return order, graph


# ============================================================
# 策略工厂
# ============================================================

class StrategyFactory:
    """
    策略工厂 — 根据 class_path 动态实例化策略

    自动注入以下依赖（通过构造函数参数名匹配）：
    - config: 系统配置字典
    - okx_client: OKX API 客户端
    - redis_cache: Redis 缓存客户端
    - adaptive_controller: 自适应控制器（通过 setter 注入）
    - stop_loss_manager: 止损管理器（通过 setter 注入）
    - coordinator: 策略协调器（通过 setter 注入）
    """

    # 构造函数参数名 → 默认注入源 映射
    CONSTRUCTOR_POSITIONAL_MAP = [
        ("config", None),         # 始终通过参数注入
        ("okx_client", None),
        ("redis_cache", None),
    ]

    # Setter 注入映射
    SETTER_INJECTORS = {
        "set_adaptive_controller": "adaptive_controller",
        "set_stop_loss_manager": "stop_loss_manager",
        "set_coordinator": "coordinator",
        "set_regime_engine": "market_regime_engine",
    }

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._class_cache: Dict[str, Type] = {}
        self._instance_count = 0

    def create(self, class_path: str, **dependencies) -> Tuple[Optional[Any], Optional[str]]:
        """
        动态创建策略实例

        Args:
            class_path: 策略类路径 (e.g., 'strategies.grid_strategy.GridStrategy')
            **dependencies: 注入的依赖项 (okx_client, redis_cache, ...)

        Returns:
            (instance, error_message)
        """
        try:
            # 1. 加载类
            cls = self._load_class(class_path)

            # 2. 解析构造函数参数
            sig = inspect.signature(cls.__init__)
            param_names = [p for p in sig.parameters if p not in ('self', 'args', 'kwargs')]

            # 3. 构建参数
            kwargs = {}
            for param_name in param_names:
                if param_name == "config":
                    # config 参数：传入策略专属配置 + 全局配置
                    kwargs[param_name] = self.config
                elif param_name in dependencies:
                    kwargs[param_name] = dependencies[param_name]
                else:
                    logger.debug(f"Strategy constructor param '{param_name}' not provided, "
                                f"class will use default")

            # 4. 实例化
            instance = cls(**kwargs)

            # 5. Setter 注入
            self._inject_setters(instance, dependencies)

            self._instance_count += 1
            logger.info(f"StrategyFactory created instance: {class_path} "
                        f"(#{self._instance_count})")

            return instance, None

        except ImportError as e:
            return None, f"Import failed: {e}"
        except TypeError as e:
            return None, f"Constructor error: {e}"
        except Exception as e:
            return None, f"Instantiation failed: {e}"

    def _load_class(self, class_path: str) -> Type:
        """动态加载类（带缓存）"""
        if class_path in self._class_cache:
            return self._class_cache[class_path]

        parts = class_path.rsplit('.', 1)
        if len(parts) != 2:
            raise ImportError(f"Invalid class_path: {class_path}")

        module_path, class_name = parts
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        if not inspect.isclass(cls):
            raise TypeError(f"'{class_name}' is not a class")

        self._class_cache[class_path] = cls
        return cls

    def _inject_setters(self, instance: Any, dependencies: Dict[str, Any]) -> None:
        """通过 setter 方法注入额外依赖"""
        for setter_name, dep_key in self.SETTER_INJECTORS.items():
            dep_value = dependencies.get(dep_key)
            if dep_value is None:
                continue
            setter = getattr(instance, setter_name, None)
            if setter and callable(setter):
                try:
                    setter(dep_value)
                    logger.debug(f"Injected {dep_key} via {setter_name}")
                except Exception as e:
                    logger.debug(f"Setter injection {setter_name} failed: {e}")


# ============================================================
# 策略加载器（主入口）
# ============================================================

class StrategyLoader:
    """
    策略加载器 — 策略系统的中央加载入口

    替代 scheduler.py 中硬编码的策略创建逻辑。
    提供从配置到策略实例的完整加载流水线：

    1. 自动发现 strategies/ 包中的策略类
    2. 验证策略类和配置
    3. 解析依赖关系确定启动顺序
    4. 通过工厂动态创建策略实例
    5. 返回完整的加载报告

    增强特性：
    - 重试机制（指数退避）
    - 加载超时控制
    - Dry-Run 预加载预览
    - 发现结果磁盘缓存
    - 接口契约验证
    - 加载进度回调
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.discoverer = StrategyDiscovery()
        self.validator = StrategyValidator(config)
        self.resolver = DependencyResolver()
        self.factory = StrategyFactory(config)

        # 加载统计
        self._load_count = 0
        self._last_report: Optional[LoadReport] = None
        self._cached_discoveries: Optional[List[StrategyDiscoveryInfo]] = None

        # 加载选项
        loader_cfg = config.get("strategy_loader", {})
        self._auto_discover = loader_cfg.get("auto_discover", True)
        self._fail_fast = loader_cfg.get("fail_fast", False)
        self._lazy_load = loader_cfg.get("lazy_load", False)
        self._skip_disabled = loader_cfg.get("skip_disabled", True)
        self._package_path = loader_cfg.get("package_path", "strategies")

        # 重试配置
        self._max_retry_attempts = loader_cfg.get("max_retry_attempts", 2)
        self._retry_delay_seconds = loader_cfg.get("retry_delay_seconds", 5)
        self._retry_backoff_multiplier = loader_cfg.get("retry_backoff_multiplier", 2.0)

        # 超时配置
        self._load_timeout_seconds = loader_cfg.get("load_timeout_seconds", 30)

        # 磁盘缓存
        self._use_disk_cache = loader_cfg.get("use_disk_cache", True)
        self._cache_ttl_seconds = loader_cfg.get("cache_ttl_seconds", 3600)

        # 进度回调
        self._progress_callbacks: List[Callable[[str, str, float], None]] = []

        logger.info(f"StrategyLoader initialized: auto_discover={self._auto_discover}, "
                    f"lazy_load={self._lazy_load}, "
                    f"retry={self._max_retry_attempts}x, timeout={self._load_timeout_seconds}s")

    # ── 发现 ─────────────────────────────────────────────────────

    def discover(self, force: bool = False) -> List[StrategyDiscoveryInfo]:
        """
        发现所有可用策略

        Returns:
            策略发现信息列表
        """
        if self._cached_discoveries and not force:
            return self._cached_discoveries

        discoveries = []
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # 尝试从磁盘缓存加载
        if self._use_disk_cache and not force:
            cached = DiscoveryDiskCache.load(self._package_path, base_dir)
            if cached:
                self._cached_discoveries = cached
                return cached

        if self._auto_discover:
            # 自动扫描策略包
            auto_discovered = self.discoverer.discover_from_package(
                self._package_path, base_dir
            )

            if auto_discovered:
                logger.info(f"Auto-discovered {len(auto_discovered)} strategies: "
                           f"{[d.name for d in auto_discovered]}")
                discoveries = auto_discovered

                # 保存到磁盘缓存
                if self._use_disk_cache:
                    DiscoveryDiskCache.save(discoveries, self._package_path, base_dir)

        # 如果自动发现失败，使用已知映射兜底
        if not discoveries:
            logger.info("Auto-discovery yielded no results, falling back to known strategies")
            known = self.discoverer.get_known_mapping()
            for name, class_path in known.items():
                known_info = self.discoverer.KNOWN_STRATEGIES.get(name, {})
                discoveries.append(StrategyDiscoveryInfo(
                    name=name,
                    class_path=class_path,
                    module_path="",
                    category=known_info.get("category", "contract"),
                    display_name=known_info.get("display_name", name),
                    description=known_info.get("description", ""),
                    enabled_by_default=known_info.get("enabled_by_default", True),
                    dependencies=known_info.get("dependencies", []),
                    config_schema_keys=known_info.get("config_schema_keys", []),
                    source="manual",
                ))

        self._cached_discoveries = discoveries
        return discoveries

    # ── 验证 ─────────────────────────────────────────────────────

    def validate(self, discoveries: List[StrategyDiscoveryInfo] = None) -> Dict[str, Tuple[bool, str, List[str]]]:
        """验证所有发现的策略"""
        discoveries = discoveries or self.discover()
        return self.validator.validate_all(discoveries)

    # ── 进度回调 ───────────────────────────────────────────────

    def on_progress(self, callback: Callable[[str, str, float], None]) -> None:
        """
        注册加载进度回调

        Args:
            callback: (strategy_name, status, progress_pct) -> None
        """
        self._progress_callbacks.append(callback)

    def _notify_progress(self, strategy_name: str, status: str, progress_pct: float) -> None:
        """通知所有进度回调"""
        for cb in self._progress_callbacks:
            try:
                cb(strategy_name, status, progress_pct)
            except Exception:
                pass

    # ── 加载 ─────────────────────────────────────────────────────

    def load_all(self, **dependencies) -> Tuple[Dict[str, Any], LoadReport]:
        """
        加载所有启用的策略（含重试机制和超时控制）

        Args:
            **dependencies: 注入的依赖项
                - okx_client: OKX API 客户端 (必需)
                - redis_cache: Redis 缓存客户端 (必需)
                - adaptive_controller: 自适应控制器
                - stop_loss_manager: 止损管理器
                - coordinator: 策略协调器

        Returns:
            (instances_dict, load_report)
        """
        start_time = datetime.now()
        report = LoadReport()

        # 1. 发现
        discoveries = self.discover()
        report.total = len(discoveries)
        self._notify_progress("", "discovery_complete", 0.05)

        # 2. 验证
        validations = self.validate(discoveries)
        self._notify_progress("", "validation_complete", 0.10)

        # 3. 解析依赖
        start_order, dep_graph = self.resolver.resolve(discoveries)
        report.start_order = start_order
        logger.info(f"Strategy start order (topological): {start_order}")

        # 4. 按顺序加载
        strategies_cfg = self.config.get("strategies", {})
        instances = {}
        total_to_load = len(start_order)
        loaded_so_far = 0

        for strategy_name in start_order:
            result = LoadResult(strategy_name=strategy_name, status=LoadStatus.PENDING)

            try:
                # 查找发现信息
                info = next((d for d in discoveries if d.name == strategy_name), None)
                if not info:
                    result.status = LoadStatus.FAILED
                    result.error = f"Discovery info not found for '{strategy_name}'"
                    report.results.append(result)
                    report.failed += 1
                    if self._fail_fast:
                        break
                    continue

                result.class_path = info.class_path

                # 检查 enabled 状态
                strategy_cfg = strategies_cfg.get(strategy_name, {})
                is_enabled = strategy_cfg.get("enabled", info.enabled_by_default)
                if not is_enabled and self._skip_disabled:
                    result.status = LoadStatus.SKIPPED
                    report.results.append(result)
                    report.skipped += 1
                    logger.info(f"Strategy '{strategy_name}' disabled, skipped")
                    self._notify_progress(strategy_name, "skipped",
                                         0.10 + 0.85 * (loaded_so_far / max(total_to_load, 1)))
                    loaded_so_far += 1
                    continue

                # 验证
                is_valid, error_msg, warnings = validations.get(strategy_name, (True, "", []))
                if not is_valid:
                    result.status = LoadStatus.FAILED
                    result.error = error_msg
                    result.warnings = warnings
                    report.results.append(result)
                    report.failed += 1
                    if self._fail_fast:
                        break
                    continue

                result.warnings = warnings

                # 延迟加载判断
                if self._lazy_load and len(instances) >= 2:
                    result.status = LoadStatus.DEFERRED
                    report.results.append(result)
                    report.deferred += 1
                    self._notify_progress(strategy_name, "deferred",
                                         0.10 + 0.85 * (loaded_so_far / max(total_to_load, 1)))
                    loaded_so_far += 1
                    continue

                # 创建实例（含重试+超时）
                self._notify_progress(strategy_name, "loading",
                                     0.10 + 0.85 * (loaded_so_far / max(total_to_load, 1)))
                result.status = LoadStatus.LOADING
                instance, error = self._load_with_retry_and_timeout(
                    info.class_path, strategy_name, **dependencies
                )

                if error:
                    result.status = LoadStatus.FAILED
                    result.error = error
                    report.results.append(result)
                    report.failed += 1
                    logger.error(f"Failed to load strategy '{strategy_name}': {error}")
                    if self._fail_fast:
                        break
                    continue

                # 加载成功
                instances[strategy_name] = instance
                result.status = LoadStatus.LOADED
                result.instance = instance
                result.load_duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                report.results.append(result)
                report.loaded += 1

                self._load_count += 1
                logger.info(f"Strategy '{strategy_name}' loaded successfully "
                           f"({result.load_duration_ms:.1f}ms)")
                self._notify_progress(strategy_name, "loaded",
                                     0.10 + 0.85 * ((loaded_so_far + 1) / max(total_to_load, 1)))

            except Exception as e:
                result.status = LoadStatus.FAILED
                result.error = str(e)
                report.results.append(result)
                report.failed += 1
                logger.error(f"Unexpected error loading '{strategy_name}': {e}")
                if self._fail_fast:
                    break

            loaded_so_far += 1

        report.total_duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        self._last_report = report

        self._notify_progress("", "complete", 1.0)

        logger.info(f"StrategyLoader complete: {report.loaded} loaded, "
                    f"{report.skipped} skipped, {report.failed} failed, "
                    f"{report.deferred} deferred "
                    f"({report.total_duration_ms:.1f}ms)")

        return instances, report

    def _load_with_retry_and_timeout(self, class_path: str, strategy_name: str,
                                      **dependencies) -> Tuple[Optional[Any], Optional[str]]:
        """
        带重试和超时的策略加载

        1. 使用 concurrent.futures 实现超时控制
        2. 失败后指数退避重试
        """
        last_error = None

        for attempt in range(self._max_retry_attempts + 1):
            if attempt > 0:
                # 指数退避
                delay = self._retry_delay_seconds * (self._retry_backoff_multiplier ** (attempt - 1))
                logger.warning(f"Retrying '{strategy_name}' (attempt {attempt}/{self._max_retry_attempts}) "
                              f"after {delay:.1f}s delay")
                time.sleep(delay)

            # 使用线程池实现超时
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.factory.create, class_path, **dependencies)
                try:
                    instance, error = future.result(timeout=self._load_timeout_seconds)
                    if instance is not None:
                        if attempt > 0:
                            logger.info(f"Strategy '{strategy_name}' loaded on retry attempt {attempt}")
                        return instance, None
                    last_error = error
                except concurrent.futures.TimeoutError:
                    last_error = (f"Loading timed out after {self._load_timeout_seconds}s "
                                 f"(attempt {attempt + 1}/{self._max_retry_attempts + 1})")
                    logger.warning(last_error)
                except Exception as e:
                    last_error = f"Load error: {e}"
                    logger.warning(f"Strategy '{strategy_name}' load attempt {attempt + 1} failed: {e}")

        return None, last_error

    # ── Dry-Run 预加载 ─────────────────────────────────────────

    def dry_run(self) -> Dict[str, Any]:
        """
        Dry-Run 预加载预览

        执行完整的发现、验证、依赖解析，但不创建策略实例。
        返回详细的预览报告，用于启动前检查。

        Returns:
            {
                "ready": bool,               # 是否可以正常加载
                "total": int,                 # 总策略数
                "enabled": int,               # 启用的策略数
                "valid": int,                 # 通过验证的策略数
                "invalid": int,               # 验证失败的策略数
                "disabled": int,              # 禁用的策略数
                "start_order": [str],         # 启动顺序
                "details": [...],             # 每个策略的详细信息
                "issues": [...],              # 需要注意的问题
                "interface_violations": [...], # 接口契约违规
            }
        """
        result = {
            "ready": True,
            "total": 0,
            "enabled": 0,
            "valid": 0,
            "invalid": 0,
            "disabled": 0,
            "start_order": [],
            "details": [],
            "issues": [],
            "interface_violations": [],
        }

        # 1. 发现
        discoveries = self.discover(force=True)
        result["total"] = len(discoveries)

        # 2. 验证
        validations = self.validate(discoveries)
        strategies_cfg = self.config.get("strategies", {})

        for info in discoveries:
            strategy_cfg = strategies_cfg.get(info.name, {})
            is_enabled = strategy_cfg.get("enabled", info.enabled_by_default)
            is_valid, error_msg, warnings = validations.get(info.name, (True, "", []))

            detail = {
                "name": info.name,
                "class_path": info.class_path,
                "category": info.category,
                "enabled": is_enabled,
                "valid": is_valid,
                "error": error_msg if not is_valid else None,
                "warnings": warnings,
                "dependencies": info.dependencies,
            }

            if is_enabled:
                result["enabled"] += 1
                if is_valid:
                    result["valid"] += 1
                else:
                    result["invalid"] += 1
                    result["issues"].append(f"[{info.name}] Validation failed: {error_msg}")
                    result["ready"] = False
            else:
                result["disabled"] += 1

            # 检查接口契约
            if is_enabled and is_valid:
                try:
                    cls = self.validator._load_class_safe(info.class_path)
                    if cls:
                        contract_ok, missing_req, missing_rec = InterfaceContract.check(cls)
                        detail["interface_ok"] = contract_ok
                        detail["interface_missing_required"] = missing_req
                        detail["interface_missing_recommended"] = missing_rec
                        if not contract_ok:
                            result["interface_violations"].append({
                                "strategy": info.name,
                                "missing_required": missing_req,
                            })
                            result["issues"].append(
                                f"[{info.name}] Missing required methods: {missing_req}"
                            )
                except Exception:
                    pass

            result["details"].append(detail)

        # 3. 依赖解析
        enabled_discoveries = [d for d in discoveries
                               if strategies_cfg.get(d.name, {}).get("enabled", d.enabled_by_default)]
        if enabled_discoveries:
            start_order, dep_graph = self.resolver.resolve(enabled_discoveries)
            result["start_order"] = start_order

            # 检查循环依赖
            if len(start_order) < len(enabled_discoveries):
                missing = set(d.name for d in enabled_discoveries) - set(start_order)
                result["issues"].append(f"Circular dependency detected among: {missing}")
                result["ready"] = False

        logger.info(f"Dry-run complete: {result['valid']}/{result['enabled']} enabled strategies valid, "
                    f"ready={result['ready']}")
        return result

    def load_single(self, strategy_name: str, **dependencies) -> Tuple[Optional[Any], LoadResult]:
        """
        加载单个策略（用于延迟加载）

        Args:
            strategy_name: 策略名称
            **dependencies: 注入的依赖项
        """
        discoveries = self.discover()
        info = next((d for d in discoveries if d.name == strategy_name), None)
        if not info:
            result = LoadResult(
                strategy_name=strategy_name,
                status=LoadStatus.FAILED,
                error=f"Strategy '{strategy_name}' not found in discoveries"
            )
            return None, result

        strategies_cfg = self.config.get("strategies", {})
        strategy_cfg = strategies_cfg.get(strategy_name, {})
        if not strategy_cfg.get("enabled", info.enabled_by_default):
            result = LoadResult(
                strategy_name=strategy_name,
                status=LoadStatus.SKIPPED,
                class_path=info.class_path,
            )
            return None, result

        start = datetime.now()
        instance, error = self.factory.create(info.class_path, **dependencies)

        result = LoadResult(
            strategy_name=strategy_name,
            status=LoadStatus.LOADED if instance else LoadStatus.FAILED,
            instance=instance,
            class_path=info.class_path,
            error=error,
            load_duration_ms=(datetime.now() - start).total_seconds() * 1000,
        )

        return instance, result

    # ── 热重载 ───────────────────────────────────────────────────

    def reload_config(self, strategy_name: str, new_config: Dict[str, Any],
                       instance: Any = None) -> Tuple[bool, str]:
        """
        热重载策略配置

        通过策略实例的 update_config() 方法注入新配置。
        """
        instance = instance or self._find_instance(strategy_name)
        if not instance:
            return False, f"Instance not found for '{strategy_name}'"

        if hasattr(instance, 'update_config') and callable(instance.update_config):
            try:
                instance.update_config(new_config)
                logger.info(f"Strategy '{strategy_name}' config hot-reloaded")
                return True, "Config updated"
            except Exception as e:
                return False, f"update_config failed: {e}"
        else:
            return False, "Instance does not support update_config"

    def _find_instance(self, strategy_name: str) -> Optional[Any]:
        """从最近的报告中查找实例"""
        if self._last_report:
            for result in self._last_report.results:
                if result.strategy_name == strategy_name and result.instance:
                    return result.instance
        return None

    # ── 查询 ─────────────────────────────────────────────────────

    def get_last_report(self) -> Optional[LoadReport]:
        """获取最后一次加载报告"""
        return self._last_report

    def get_load_stats(self) -> Dict[str, Any]:
        """获取加载统计"""
        return {
            "total_loads": self._load_count,
            "auto_discover": self._auto_discover,
            "lazy_load": self._lazy_load,
            "package_path": self._package_path,
            "discovered_count": len(self._cached_discoveries) if self._cached_discoveries else 0,
            "last_report": self._last_report.to_dict() if self._last_report else None,
            "factory_instances": self.factory._instance_count,
            "retry_config": {
                "max_attempts": self._max_retry_attempts,
                "delay_seconds": self._retry_delay_seconds,
                "backoff_multiplier": self._retry_backoff_multiplier,
            },
            "timeout_seconds": self._load_timeout_seconds,
            "disk_cache_enabled": self._use_disk_cache,
            "validator_config": self.validator.get_config(),
        }

    def get_discovery_summary(self) -> List[Dict[str, Any]]:
        """获取发现摘要"""
        discoveries = self.discover()
        return [d.to_dict() for d in discoveries]

    def invalidate_cache(self) -> bool:
        """清除发现缓存（强制下次重新扫描）"""
        self._cached_discoveries = None
        return DiscoveryDiskCache.invalidate()


# ── 全局单例 ────────────────────────────────────────────────────

import threading

_loader: Optional[StrategyLoader] = None
_loader_lock = threading.Lock()


def get_strategy_loader(config: Dict[str, Any] = None) -> StrategyLoader:
    """获取全局策略加载器"""
    global _loader
    with _loader_lock:
        if _loader is None:
            _loader = StrategyLoader(config)
        elif config:
            _loader.config.update(config)
        return _loader


def reset_strategy_loader() -> None:
    """重置加载器（测试用）"""
    global _loader
    with _loader_lock:
        _loader = None
