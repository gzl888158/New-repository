import sqlite3, json

db = r'e:\新建文件夹\okx_quant_trading\data\trading.db'
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

print("=== trade_records GPS ===")
for r in cur.execute("SELECT * FROM trade_records WHERE symbol LIKE 'GPS%' ORDER BY id DESC LIMIT 40"):
    d = dict(r)
    print(json.dumps(d, ensure_ascii=False, default=str))

print("\n=== trades GPS ===")
for r in cur.execute("SELECT * FROM trades WHERE symbol LIKE 'GPS%' ORDER BY trade_id DESC LIMIT 40"):
    d = dict(r)
    print(json.dumps(d, ensure_ascii=False, default=str))

conn.close()
