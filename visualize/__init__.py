"""visualize 包：企业级交易数据可视化模块。

提供基于 matplotlib 的服务端图表生成，输出 base64 PNG / SVG / 文件路径，
可嵌入日报、复盘报告与 Dashboard。

设计原则（与 eval 包一致）：
- 所有外部输入经 safe_* 转换，禁止裸 int()/float()
- 禁止 NaN/Inf 出现在输出与坐标中
- fail-closed：图表生成失败返回 {"error": ...}，绝不向上抛异常
- matplotlib 强制 Agg 后端，无需 GUI；中文字体自动探测
- 每个 Figure 使用后立即 close，避免内存泄漏

子模块：
- visualize._base     : 共享工具（安全转换、调色板、图像编码、字体探测）
- visualize.charts    : 单图生成器（权益曲线、回撤、PnL 分布、策略对比、时段热力）
- visualize.report    : 组合可视化报表
"""
from visualize._base import (
    CHART_PALETTE,
    safe_int,
    safe_float,
    safe_div,
    safe_finite,
    fig_to_base64_png,
    fig_to_svg,
    save_figure,
)
from visualize.report import VisualizationReport, generate_report

__all__ = [
    "CHART_PALETTE",
    "safe_int",
    "safe_float",
    "safe_div",
    "safe_finite",
    "fig_to_base64_png",
    "fig_to_svg",
    "save_figure",
    "VisualizationReport",
    "generate_report",
]
