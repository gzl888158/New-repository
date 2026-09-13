import sqlite3
c = sqlite3.connect('data/trading.db')
for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall():
    print(r[0])
