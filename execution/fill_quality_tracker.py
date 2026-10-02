"""订单成交质量追踪模块

记录每笔订单的理论价 vs 实际成交价，计算滑点统计：
- 平均滑点 / 最大滑点 / 95分位滑点
- 按策略、按 symbol 聚合
- 通过 dashboard 暴露API
"""
import asyncio
import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from loguru import logger


class FillQualityTracker:
    def __init__(self, config: Dict[str, Any], sqlite_storage=None):
        self.config = config
        self.sqlite_storage = sqlite_storage
        self._db_path = config.get("sqlite", {}).get("db_path", "./data/trading.db")

        # 内存缓存最近的成交记录（最近1000条）
        self._recent_fills: List[Dict[str, Any]] = []
        self._max_cache = 1000

    def record_fill(self, order_id: str, symbol: str, strategy_name: str,
                    side: str, expected_price: float, filled_price: float,
                    quantity: float, order_type: str = "market",
                    slippage_tolerance: float = 0.01):
        """记录一笔订单的成交质量数据

        Args:
            order_id: 交易所订单ID
            symbol: 合约ID
            strategy_name: 策略名
            side: buy/sell
            expected_price: 理论价（下单时的price）
            filled_price: 实际成交价
            quantity: 成交数量
            order_type: 订单类型 market/limit
            slippage_tolerance: 滑点容忍度（配置值）
        """
        try:
            if expected_price <= 0 or filled_price <= 0:
                return

            # 滑点 = (实际成交价 - 理论价) / 理论价，按方向修正
            # 买入：正值=付出更高价（不利）；卖出：正值=卖得更低（不利）
            raw_slippage = (filled_price - expected_price) / expected_price
            # 方向修正：买入方向滑点为正=不利；卖出方向滑点为负=不利，统一取"不利方向"
            direction_adjusted_slippage = raw_slippage if side == "buy" else -raw_slippage
            abs_slippage = abs(raw_slippage)

            record = {
                "order_id": order_id,
                "symbol": symbol,
                "strategy_name": strategy_name,
                "side": side,
                "expected_price": expected_price,
                "filled_price": filled_price,
                "quantity": quantity,
                "order_type": order_type,
                "raw_slippage": raw_slippage,
                "abs_slippage": abs_slippage,
                "direction_adjusted_slippage": direction_adjusted_slippage,
                "exceeds_tolerance": abs_slippage > slippage_tolerance,
                "timestamp": datetime.now().isoformat(),
            }

            self._recent_fills.append(record)
            if len(self._recent_fills) > self._max_cache:
                self._recent_fills.pop(0)

            # 持久化到 SQLite
            self._persist_fill(record)

            if record["exceeds_tolerance"]:
                logger.warning(
                    f"Fill slippage exceeds tolerance: {symbol} {side} "
                    f"expected={expected_price:.4f} filled={filled_price:.4f} "
                    f"slippage={abs_slippage:.4%}"
                )
        except Exception as e:
            logger.error(f"Error recording fill quality: {e}")

    def _persist_fill(self, record: Dict[str, Any]):
        """持久化到 fill_quality 表"""
        conn = None
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fill_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT,
                    symbol TEXT,
                    strategy_name TEXT,
                    side TEXT,
                    expected_price REAL,
                    filled_price REAL,
                    quantity REAL,
                    order_type TEXT,
                    raw_slippage REAL,
                    abs_slippage REAL,
                    direction_adjusted_slippage REAL,
                    exceeds_tolerance INTEGER,
                    timestamp TEXT
                )
            """)
            conn.execute("""
                INSERT INTO fill_quality
                (order_id, symbol, strategy_name, side, expected_price, filled_price,
                 quantity, order_type, raw_slippage, abs_slippage,
                 direction_adjusted_slippage, exceeds_tolerance, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record["order_id"], record["symbol"], record["strategy_name"],
                record["side"], record["expected_price"], record["filled_price"],
                record["quantity"], record["order_type"], record["raw_slippage"],
                record["abs_slippage"], record["direction_adjusted_slippage"],
                1 if record["exceeds_tolerance"] else 0,
                record["timestamp"]
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error persisting fill quality: {e}")
        finally:
            if conn is not None:
                conn.close()

    def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """获取最近N小时的成交质量统计"""
        conn = None
        try:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

            # 总体统计
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_fills,
                    AVG(abs_slippage) as avg_slippage,
                    MAX(abs_slippage) as max_slippage,
                    AVG(direction_adjusted_slippage) as avg_unfavorable_slippage,
                    SUM(exceeds_tolerance) as tolerance_exceeded_count
                FROM fill_quality
                WHERE timestamp >= ?
            """, (cutoff,))
            overall = dict(cursor.fetchone() or {})

            # 按策略聚合
            cursor = conn.execute("""
                SELECT
                    strategy_name,
                    COUNT(*) as fills,
                    AVG(abs_slippage) as avg_slippage,
                    MAX(abs_slippage) as max_slippage,
                    SUM(exceeds_tolerance) as exceeded_count
                FROM fill_quality
                WHERE timestamp >= ?
                GROUP BY strategy_name
            """, (cutoff,))
            by_strategy = [dict(r) for r in cursor.fetchall()]

            # 按symbol聚合
            cursor = conn.execute("""
                SELECT
                    symbol,
                    COUNT(*) as fills,
                    AVG(abs_slippage) as avg_slippage,
                    MAX(abs_slippage) as max_slippage
                FROM fill_quality
                WHERE timestamp >= ?
                GROUP BY symbol
                ORDER BY avg_slippage DESC
                LIMIT 10
            """, (cutoff,))
            by_symbol = [dict(r) for r in cursor.fetchall()]

            # 95分位滑点（用Python计算）
            cursor = conn.execute("""
                SELECT abs_slippage FROM fill_quality WHERE timestamp >= ?
            """, (cutoff,))
            slippages = sorted([r["abs_slippage"] for r in cursor.fetchall()])
            p95_slippage = slippages[int(len(slippages) * 0.95)] if slippages else 0

            return {
                "hours": hours,
                "overall": {
                    "total_fills": overall.get("total_fills", 0),
                    "avg_slippage": round(overall.get("avg_slippage", 0) or 0, 6),
                    "max_slippage": round(overall.get("max_slippage", 0) or 0, 6),
                    "p95_slippage": round(p95_slippage, 6),
                    "avg_unfavorable_slippage": round(overall.get("avg_unfavorable_slippage", 0) or 0, 6),
                    "tolerance_exceeded_count": overall.get("tolerance_exceeded_count", 0) or 0,
                    "tolerance_exceeded_rate": (
                        round((overall.get("tolerance_exceeded_count", 0) or 0) / overall["total_fills"], 4)
                        if overall.get("total_fills", 0) > 0 else 0
                    ),
                },
                "by_strategy": by_strategy,
                "by_symbol": by_symbol,
            }
        except Exception as e:
            logger.error(f"Error getting fill statistics: {e}")
            return {"error": str(e)}
        finally:
            if conn is not None:
                conn.close()

    def export_status(self) -> Dict[str, Any]:
        """导出当前状态供dashboard读取"""
        return self.get_statistics(hours=24)
