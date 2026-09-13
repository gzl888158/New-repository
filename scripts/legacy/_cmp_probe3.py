import sqlite3, json
from datetime import datetime

conn = sqlite3.connect("file:data/trading.db?mode=ro", uri=True)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

def q(sql, p=()):
    return [dict(r) for r in cur.execute(sql, p).fetchall()]

CUT = "2026-08-19 09:15:00"

print("=== grid direction for trades entered after 09:15 ===")
for r in q("SELECT direction, COUNT(*) c, MIN(entry_time) mn, MAX(entry_time) mx FROM trades WHERE strategy_name='grid' AND entry_time >= ? GROUP BY direction", (CUT,)):
    print(r)

print("\n=== grid direction for trades exited after 09:15 ===")
for r in q("SELECT direction, COUNT(*) c FROM trades WHERE strategy_name='grid' AND exit_time >= ? GROUP BY direction", (CUT,)):
    print(r)

print("\n=== all grid shorts after 09:15 (entry/exit) ===")
for r in q("SELECT trade_id, symbol, direction, entry_time, exit_time, pnl_usdt, win FROM trades WHERE strategy_name='grid' AND direction='short' AND exit_time >= ? ORDER BY exit_time", (CUT,)):
    print(r)

print("\n=== all trades (any strategy) entered after 09:15 by direction ===")
for r in q("SELECT strategy_name, direction, COUNT(*) c FROM trades WHERE entry_time >= ? GROUP BY strategy_name, direction", (CUT,)):
    print(r)

print("\n=== account_history range & latest ===")
print(q("SELECT COUNT(*) c, MIN(timestamp) mn, MAX(timestamp) mx FROM account_history"))
print(q("SELECT timestamp, total_equity, available_balance, used_margin, margin_rate FROM account_history ORDER BY timestamp DESC LIMIT 3"))
print("\n--- before 08:45 latest ---")
print(q("SELECT timestamp, total_equity, used_margin FROM account_history WHERE timestamp <= '2026-08-19 08:45:00' ORDER BY timestamp DESC LIMIT 1"))
print("\n--- after 09:15 earliest & latest ---")
print(q("SELECT timestamp, total_equity, used_margin FROM account_history WHERE timestamp >= '2026-08-19 09:15:00' ORDER BY timestamp ASC LIMIT 1"))
print(q("SELECT timestamp, total_equity, used_margin FROM account_history WHERE timestamp >= '2026-08-19 09:15:00' ORDER BY timestamp DESC LIMIT 1"))

conn.close()

# decision_audit_chain analysis
try:
    with open("data/decision_audit_chain.json", encoding="utf-8") as f:
        chain = json.load(f)
    entries = chain.get("entries", [])
    print("\n=== decision_audit_chain entries:", len(entries), "===")
    # field sample keys
    if entries:
        print("sample keys:", sorted(entries[0].keys()))
        print("sample meta_verdict values:", set(e.get("meta_verdict") for e in entries[:100]))
        print("sample executed values:", set(e.get("executed") for e in entries[:100]))
        print("sample outcome values:", set(e.get("outcome") for e in entries[:100]))
    post = [e for e in entries if e.get("timestamp", "") >= "2026-08-19T09:15:00"]
    print("entries after 09:15:", len(post))
    from collections import Counter
    print("meta_verdict after 09:15:", Counter(e.get("meta_verdict") for e in post))
    print("executed after 09:15:", Counter(str(e.get("executed")) for e in post))
    print("min ts:", min((e.get("timestamp","") for e in entries), default=""))
    print("max ts:", max((e.get("timestamp","") for e in entries), default=""))
except Exception as e:
    print("decision_audit_chain error:", e)
