# -*- coding: utf-8 -*-
import sqlite3, json, os
os.chdir(r'e:\新建文件夹\okx_quant_trading')

db = sqlite3.connect('data/trading.db')
cur = db.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]

for t in tables:
    cur.execute(f"PRAGMA table_info({t})")
    cols = [r[1] for r in cur.fetchall()]
    if 'status' not in cols:
        continue
    try:
        cur.execute(f"SELECT * FROM {t} WHERE status='open'")
        rows = cur.fetchall()
    except Exception as e:
        continue
    if rows:
        print(f"\n=== {t} OPEN 记录 ({len(rows)}) ===")
        for row in rows:
            d = dict(zip(cols, row))
            print(json.dumps(d, ensure_ascii=False, default=str))

db.close()
print("\n=== 完成 ===")
