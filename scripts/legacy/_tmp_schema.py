import sqlite3
c = sqlite3.connect('data/trading.db')
for t in ['trades', 'trading_signals', 'signal_generator_history', 'stop_loss_audit', 'fill_quality']:
    print("=== " + t + " ===")
    for r in c.execute(f"PRAGMA table_info({t})").fetchall():
        print(r)
    print()
