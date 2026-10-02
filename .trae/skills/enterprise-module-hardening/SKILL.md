---
name: "enterprise-module-hardening"
description: "Enterprise-grade module hardening: type safety, fail-closed error handling, JSON-safe outputs, NaN/Inf elimination. Invoke when user asks to 强化/企业级/harden a module."
---

# Enterprise Module Hardening

将现有模块强化为企业级标准的标准化工作流。核心目标：**类型安全、fail-closed、JSON 安全、不污染调用方、无 NaN/Inf 输出**。

## 触发条件

当用户请求包含以下语义时调用：
- "企业级 `<module>` 模块"
- "强化 `<module>` 模块"
- "harden `<module>`"
- "enterprise-grade `<module>`"

## 标准工作流

### 第 1 步：模块勘探

```
Glob/Read 目标模块的全部 .py 文件
Grep 定位高风险模式：
  - float('inf') / float('nan')
  - int() / float() 直接转换外部输入
  - a / b 无除零防护
  - dict[key] 直接取值（无 .get）
  - json.dumps 无 default=
  - 直接修改传入的 dict/list
  - try/except: pass 静默吞错
```

### 第 2 步：建立安全工具层（`_base.py`）

若模块已有多个文件，创建 `_base.py`（或在已有 helpers 中）统一放置以下工具，**不依赖 eval/visualize** 避免循环导入：

```python
def safe_float(value, default=0.0) -> float:
    """None / 非法 / NaN / Inf -> default"""
def safe_int(value, default=0) -> int: ...
def safe_div(n, d, default=0.0) -> float:
    """除零 / NaN / Inf -> default"""
def safe_finite(value, default=0.0) -> float: ...
def _sanitize_for_json(obj) -> Any:
    """递归清洗，确保 json.dumps 可用且不含 NaN/Inf"""
def safe_json_dumps(obj, **kwargs) -> str:
    """带异常兜底的 json.dumps，失败返回 '{}'"""
```

### 第 3 步：逐项加固

| 加固项 | 规则 |
|---|---|
| **外部输入类型安全** | 所有 config / 指标 / API 响应 / 环境变量 必须经 `safe_*` 转换后再使用 |
| **消除 NaN/Inf 输出** | 搜索 `float('inf')`、`float('nan')`；计算结果用 `safe_finite` 兜底 |
| **除零防护** | 所有 `/` 运算改用 `safe_div`，或显式判断分母为 0 |
| **fail-closed** | 涉及资金/安全/开仓的决策路径，异常时一律阻断而非放行 |
| **不污染调用方** | 函数接收的 dict/list 先 `copy.deepcopy` 再操作 |
| **JSON 安全** | 所有对外返回的 dict 先经 `_sanitize_for_json` 清洗；`json.dumps` 用 `default=str` + try/except |
| **异常处理** | 禁止裸 `except: pass`；至少 `logger.warning`，关键路径需记录堆栈 |
| **配置边界** | 数值型配置参数钳制到合理范围（如 `max_connections = max(1, ...)`） |

### 第 4 步：验证

```bash
# 1. 编译检查
.venv/Scripts/python.exe -m py_compile <module>/*.py

# 2. 冒烟测试（必测项）
- safe_* 工具对 None/NaN/Inf/非法字符串的处理
- 除零场景返回 0.0 而非崩溃
- float('inf') 已消除
- json.dumps 结果不含 'NaN' / 'Infinity'
- fail-closed 路径在异常时确实阻断

# 3. 全量回归
.venv/Scripts/python.exe -m pytest tests/unit/ -q --tb=line
```

## 关键反模式（必须修复）

1. `float('inf')` 作为返回值 → 改为 `0.0` 或合理默认
2. `value / denominator` 无防零 → `safe_div(value, denominator)`
3. `int(config["x"])` 直接转换 → `safe_int(config.get("x"), default)`
4. `costs["total"] + x if cond else costs["total"]` 三元运算符优先级 bug → 加括号
5. `json.dumps(obj)` 无 default → `json.dumps(obj, default=str)` + try/except
6. `state["meta"] = ...` 直接修改传入 dict → 先 deepcopy

## 设计约束

- 优先编辑现有文件，仅在必要时新建（如 `_base.py`）
- 不主动创建文档文件
- 向后兼容：保留原有函数签名和返回结构，仅增强健壮性
- 安全工具函数放在被加固模块内部，不跨模块复用 eval/visualize 的实现（避免循环依赖）
