"""
统一持仓方向处理工具 - Unified Position Direction Handler

功能：
1. 将任意方向字符串（long/short/buy/sell/大小写混合）规范化到标准格式
2. long/short 与 buy/sell 之间的双向转换
3. 方向判断与反向计算
4. 方向字符串验证

在整个交易系统中，不同模块（OKX API、策略引擎、风控、订单管理）
对方向的表示方式不一致，该模块统一处理所有方向字符串的规范化。
"""

from typing import Tuple, Optional
from loguru import logger


class DirectionUnifier:
    """
    统一持仓方向处理类

    所有方法均为静态方法，无需实例化即可使用。

    方向映射规则：
        - long 方向：'long', 'LONG', 'buy', 'BUY'          → 'long'
        - short 方向：'short', 'SHORT', 'sell', 'SELL'     → 'short'

    转换规则：
        - long ↔ buy
        - short ↔ sell
    """

    # ── 核心映射表 ──────────────────────────────────────

    # 方向 → 规范化结果
    _NORMALIZE_MAP: dict = {
        "long": "long",
        "short": "short",
        "buy": "long",
        "sell": "short",
    }

    # 方向 → 交易方向
    _TO_SIDE_MAP: dict = {
        "long": "buy",
        "short": "sell",
        "buy": "buy",
        "sell": "sell",
    }

    # 交易方向 → 方向
    _TO_POS_SIDE_MAP: dict = {
        "buy": "long",
        "sell": "short",
        "long": "long",
        "short": "short",
    }

    # 有效方向集合
    _VALID_DIRECTIONS: set = {"long", "short", "buy", "sell"}

    # ── 静态方法 ────────────────────────────────────────

    @staticmethod
    def normalize(direction: str) -> str:
        """
        将任意方向字符串规范化到标准格式 ('long', 'short')

        支持输入：
            - 'long', 'LONG', 'Long'         → 'long'
            - 'short', 'SHORT', 'Short'      → 'short'
            - 'buy', 'BUY', 'Buy'            → 'long'
            - 'sell', 'SELL', 'Sell'         → 'short'

        Args:
            direction: 需要规范化的方向字符串

        Returns:
            规范化后的方向字符串，'long' 或 'short'

        Raises:
            ValueError: 输入方向无效时抛出

        Example:
            >>> DirectionUnifier.normalize("BUY")
            'long'
            >>> DirectionUnifier.normalize("sell")
            'short'
        """
        if not isinstance(direction, str):
            logger.error(f"DirectionUnifier.normalize: 输入类型无效 (expected str, got {type(direction).__name__})")
            raise ValueError(
                f"方向必须为字符串类型，收到: {type(direction).__name__}"
            )

        normalized = direction.strip().lower()
        if normalized not in DirectionUnifier._VALID_DIRECTIONS:
            logger.error(f"DirectionUnifier.normalize: 无效方向字符串 '{direction}'")
            raise ValueError(
                f"无效方向 '{direction}'，有效值: long, short, buy, sell（不区分大小写）"
            )

        return DirectionUnifier._NORMALIZE_MAP[normalized]

    @staticmethod
    def to_side(pos_side: str) -> str:
        """
        将持仓方向 (long/short) 转换为交易方向 (buy/sell)

        long → buy, short → sell

        Args:
            pos_side: 持仓方向字符串，'long' 或 'short'（不区分大小写）

        Returns:
            交易方向字符串，'buy' 或 'sell'

        Raises:
            ValueError: 输入方向无效时抛出

        Example:
            >>> DirectionUnifier.to_side("long")
            'buy'
            >>> DirectionUnifier.to_side("SHORT")
            'sell'
        """
        normalized = DirectionUnifier.normalize(pos_side)
        return DirectionUnifier._TO_SIDE_MAP[normalized]

    @staticmethod
    def to_pos_side(side: str) -> str:
        """
        将交易方向 (buy/sell) 转换为持仓方向 (long/short)

        buy → long, sell → short

        Args:
            side: 交易方向字符串，'buy' 或 'sell'（不区分大小写）

        Returns:
            持仓方向字符串，'long' 或 'short'

        Raises:
            ValueError: 输入方向无效时抛出

        Example:
            >>> DirectionUnifier.to_pos_side("buy")
            'long'
            >>> DirectionUnifier.to_pos_side("SELL")
            'short'
        """
        normalized = DirectionUnifier.normalize(side)
        return DirectionUnifier._TO_POS_SIDE_MAP[normalized]

    @staticmethod
    def is_long(direction: str) -> bool:
        """
        判断方向是否为多头 (long)

        支持 long/short/buy/sell 任意格式（不区分大小写）

        Args:
            direction: 方向字符串

        Returns:
            是否为多头方向

        Raises:
            ValueError: 输入方向无效时抛出

        Example:
            >>> DirectionUnifier.is_long("buy")
            True
            >>> DirectionUnifier.is_long("SHORT")
            False
        """
        return DirectionUnifier.normalize(direction) == "long"

    @staticmethod
    def is_short(direction: str) -> bool:
        """
        判断方向是否为空头 (short)

        支持 long/short/buy/sell 任意格式（不区分大小写）

        Args:
            direction: 方向字符串

        Returns:
            是否为空头方向

        Raises:
            ValueError: 输入方向无效时抛出

        Example:
            >>> DirectionUnifier.is_short("sell")
            True
            >>> DirectionUnifier.is_short("LONG")
            False
        """
        return DirectionUnifier.normalize(direction) == "short"

    @staticmethod
    def opposite(direction: str) -> str:
        """
        返回相反方向

        long → short, short → long
        也支持 buy/sell 输入：buy → sell, sell → buy
        输出格式与输入格式一致：输入 long/short 则输出 long/short，
        输入 buy/sell 则输出 buy/sell。

        Args:
            direction: 方向字符串，long/short/buy/sell（不区分大小写）

        Returns:
            相反方向的规范化字符串

        Raises:
            ValueError: 输入方向无效时抛出

        Example:
            >>> DirectionUnifier.opposite("long")
            'short'
            >>> DirectionUnifier.opposite("buy")
            'sell'
        """
        if not isinstance(direction, str):
            logger.error(f"DirectionUnifier.opposite: 输入类型无效 (expected str, got {type(direction).__name__})")
            raise ValueError(
                f"方向必须为字符串类型，收到: {type(direction).__name__}"
            )

        key = direction.strip().lower()
        if key not in DirectionUnifier._VALID_DIRECTIONS:
            logger.error(f"DirectionUnifier.opposite: 无效方向字符串 '{direction}'")
            raise ValueError(
                f"无效方向 '{direction}'，有效值: long, short, buy, sell（不区分大小写）"
            )

        normalized = DirectionUnifier._NORMALIZE_MAP[key]
        opposite_normalized = "short" if normalized == "long" else "long"

        # 保持输出格式与输入格式一致：buy/sell 输入 → buy/sell 输出
        if key in ("buy", "sell"):
            return DirectionUnifier._TO_SIDE_MAP[opposite_normalized]
        return opposite_normalized

    @staticmethod
    def normalize_pos_side(pos_side: str) -> str:
        """
        专门用于持仓方向 (posSide) 的规范化

        与 normalize() 功能相同，但命名更语义化，适用于 OKX API 的 posSide 字段处理。
        当输入已经是 long/short 时直接返回，当输入是 buy/sell 时映射为 long/short。

        Args:
            pos_side: 持仓方向字符串

        Returns:
            规范化后的持仓方向，'long' 或 'short'

        Raises:
            ValueError: 输入方向无效时抛出

        Example:
            >>> DirectionUnifier.normalize_pos_side("long")
            'long'
            >>> DirectionUnifier.normalize_pos_side("BUY")
            'long'
        """
        return DirectionUnifier.normalize(pos_side)

    @staticmethod
    def validate_direction(direction: str) -> Tuple[Optional[str], Optional[str]]:
        """
        验证方向字符串并返回 (规范化结果, 错误信息)

        Args:
            direction: 需要验证的方向字符串

        Returns:
            (normalized_direction, error_message) 元组
            - 验证通过时: ('long' 或 'short', None)
            - 验证失败时: (None, 错误描述字符串)

        Example:
            >>> DirectionUnifier.validate_direction("buy")
            ('long', None)
            >>> DirectionUnifier.validate_direction("invalid")
            (None, "无效方向 'invalid'，有效值: long, short, buy, sell（不区分大小写）")
            >>> DirectionUnifier.validate_direction(123)
            (None, "方向必须为字符串类型，收到: int")
        """
        try:
            normalized = DirectionUnifier.normalize(direction)
            return (normalized, None)
        except ValueError as e:
            return (None, str(e))
        except Exception as e:
            logger.error(f"DirectionUnifier.validate_direction: 意外错误 - {e}")
            return (None, f"方向验证失败: {str(e)}")