"""
分库分表基础设施 — ShardRouter
==============================

为「订单 / 行情」等追加型、高增长数据提供按时间分片的路由与表名管理。

设计要点：
1. 按月分片：`{base}_YYYYMM`（如 trade_records_202609、ticker_202608）。
2. 表名合法性校验（防 SQL 注入）：表名必须匹配 `[A-Za-z_][A-Za-z0-9_]*`，
   分片后缀必须是 6 位数字月份。
3. 惰性建表：`ensure_shard` 由调用方提供 DDL 模板，保证与热表 schema 一致。
4. 分片发现：`iter_shards` 从 `sqlite_master` 枚举「热表 + 已有分片」，
   热表在前、分片按月份字典序（即时间序）排列。
"""

import re
from datetime import datetime
from typing import Callable, List

from sqlalchemy import text as sql_text

# 合法表名：字母/下划线开头，后续可含数字
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# 分片后缀：恰好 6 位数字月份（YYYYMM）
_SHARD_RE_TEMPLATE = r"{base}_\d{{6}}"


def is_valid_identifier(name: str) -> bool:
    """表名合法性校验，杜绝注入。"""
    return bool(name) and bool(_IDENT_RE.match(name))


def month_suffix(dt: datetime) -> str:
    """返回 YYYYMM 月份后缀。"""
    return dt.strftime("%Y%m")


class ShardRouter:
    """按月份分片的命名/发现/建表工具。"""

    def __init__(self, base_table: str):
        if not is_valid_identifier(base_table):
            raise ValueError(f"Illegal base table name: {base_table!r}")
        self.base_table = base_table
        self._shard_re = re.compile(_SHARD_RE_TEMPLATE.format(base=re.escape(base_table)))

    def shard_name(self, dt: datetime) -> str:
        """按时间生成分片表名：{base}_YYYYMM。"""
        return f"{self.base_table}_{month_suffix(dt)}"

    def is_shard(self, table_name: str) -> bool:
        """判断表名是否为本 base 的月份分片。"""
        return bool(self._shard_re.match(table_name))

    def iter_shards(self, conn) -> List[str]:
        """枚举「热表 + 已有分片」，热表在前、分片按月份字典序（时间序）排列。

        conn 为 SQLAlchemy Connection（与数据层 sqlite_storage 一致）。
        """
        rowset = conn.execute(sql_text("SELECT name FROM sqlite_master WHERE type='table'"))
        names = [r[0] for r in rowset.fetchall()]
        shards = sorted(n for n in names if self.is_shard(n))
        result: List[str] = []
        if self.base_table in names:
            result.append(self.base_table)
        result.extend(shards)
        return result

    def ensure_shard(self, conn, dt: datetime, create_sql: Callable[[str], str]) -> str:
        """若分片不存在则建表，返回分片表名。

        conn 为 SQLAlchemy Connection；create_sql(name) 返回该分片的 CREATE TABLE 语句。
        """
        name = self.shard_name(dt)
        conn.execute(sql_text(create_sql(name)))
        return name