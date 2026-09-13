# -*- coding: utf-8 -*-
import sqlite3

c = sqlite3.connect("file:data/trading.db?mode=ro", uri=True)
tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
print("TABLES:", tables)
for t in tables:
    cols = c.execute(f"PRAGMA table_info({t})").fetchall()
    print(f"\n[{t}]")
    for col in cols:
        print("   ", col[1], col[2])
