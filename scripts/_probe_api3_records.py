"""查询 trade_records 中 API3 残留记录的完整字段，确认要修正的项。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sqlalchemy import text
from configs.settings import load_config
from data.sqlite_storage import SQLiteStorage

config = load_config()
storage = SQLiteStorage(config)

conn = storage.get_connection()
rows = conn.execute(text(
    "SELECT * FROM trade_records WHERE symbol LIKE '%API3%' ORDER BY create_time"
)).fetchall()
print(f"=== trade_records 中 API3 完整字段（共 {len(rows)} 条）===")
for r in rows:
    print(dict(r._mapping))

# 同时查 trades 表里刚补记的记录
print("\n=== trades 表里 manual_override API3 记录 ===")
rows2 = conn.execute(text(
    "SELECT * FROM trades WHERE symbol='API3-USDT-SWAP'"
)).fetchall()
for r in rows2:
    print(dict(r._mapping))

conn.close()
