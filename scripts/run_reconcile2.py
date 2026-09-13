import sys
import os
# 确保使用项目根目录的模块，但不触发 core/__init__.py 的循环导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 直接导入，避免 __init__.py 链式导入
from core.okx_client import OKXClient
from data.sqlite_storage import SQLiteStorage
from configs.settings import load_config

config = load_config()
okx_client = OKXClient(config)
sqlite_storage = SQLiteStorage(config)

# 获取数据库中pnl=0的closed记录
session = sqlite_storage._Session()
try:
    from data.sqlite_storage import TradeRecord
    records = session.query(TradeRecord).filter(
        TradeRecord.status == "closed"
    ).order_by(TradeRecord.close_time.desc()).limit(500).all()
    
    suspicious_records = [r for r in records if not r.pnl or r.pnl == 0]
    print(f"Found {len(suspicious_records)} suspicious closed records (pnl=0/null)")
    
    # 找到最早的close_time
    earliest_close_time = None
    for rec in suspicious_records:
        if rec.close_time:
            if earliest_close_time is None or rec.close_time < earliest_close_time:
                earliest_close_time = rec.close_time
    if earliest_close_time:
        print(f"Earliest close_time: {earliest_close_time}")
        earliest_ts_ms = str(int(earliest_close_time.timestamp() * 1000))
        print(f"Earliest ts_ms: {earliest_ts_ms}")
    else:
        earliest_ts_ms = None
finally:
    session.close()

# 测试分页拉取
print("\n=== 分页拉取账单 ===")
all_bills = okx_client.get_all_bills_paginated(
    bill_type="2",
    earliest_ts_ms=earliest_ts_ms,
    max_pages=10
)
print(f"Total bills retrieved: {len(all_bills)}")

# 按symbol统计
symbol_counts = {}
for bill in all_bills:
    sym = bill.get("instId", "")
    symbol_counts[sym] = symbol_counts.get(sym, 0) + 1
print(f"Bills by symbol: {symbol_counts}")

# 构建索引
bills_index = {}
bills_by_symbol = {}
for bill in all_bills:
    symbol = bill.get("instId", "")
    ts = bill.get("ts", "")
    ord_id = bill.get("ordId", "")
    
    if ord_id:
        bills_index[ord_id] = bill
    
    key = f"{symbol}_{ts}"
    bills_index[key] = bill
    
    if symbol not in bills_by_symbol:
        bills_by_symbol[symbol] = []
    bills_by_symbol[symbol].append(bill)

# 重新获取suspicious_records进行对账
session = sqlite_storage._Session()
try:
    from data.sqlite_storage import TradeRecord
    records = session.query(TradeRecord).filter(
        TradeRecord.status == "closed"
    ).order_by(TradeRecord.close_time.desc()).limit(500).all()
    suspicious_records = [r for r in records if not r.pnl or r.pnl == 0]
finally:
    session.close()

# 逐条对账
corrected = 0
failed = 0
matched_count = 0
for rec in suspicious_records:
    try:
        if rec.pnl is not None and rec.pnl != 0:
            continue
        
        matched_bill = None
        
        # 优先按orderId匹配
        if rec.id in bills_index:
            matched_bill = bills_index[rec.id]
        
        # 其次按symbol+close_time
        if not matched_bill and rec.close_time:
            ts_str = str(int(rec.close_time.timestamp() * 1000))
            key = f"{rec.symbol}_{ts_str}"
            if key in bills_index:
                matched_bill = bills_index[key]
        
        # 最后按symbol匹配最近的
        if not matched_bill and rec.symbol in bills_by_symbol:
            sorted_bills = sorted(bills_by_symbol[rec.symbol], key=lambda x: x.get("ts", ""), reverse=True)
            for bill in sorted_bills:
                if float(bill.get("pnl", "0")) != 0:
                    matched_bill = bill
                    break
        
        if matched_bill:
            matched_count += 1
            okx_pnl = float(matched_bill.get("pnl", "0"))
            okx_fee = abs(float(matched_bill.get("fee", "0")))
            okx_fill_px = float(matched_bill.get("fillPx", "0"))
            net_pnl = okx_pnl - okx_fee if okx_pnl != 0 else 0
            
            if net_pnl != 0 or okx_fill_px > 0:
                updates = {"pnl": net_pnl}
                if okx_fill_px > 0 and (not rec.filled_price or rec.filled_price == 0):
                    updates["filled_price"] = okx_fill_px
                if rec.price and rec.price > 0 and rec.quantity and rec.quantity > 0:
                    cost = rec.price * rec.quantity
                    updates["pnl_percent"] = net_pnl / cost * 100 if cost > 0 else 0
                
                sqlite_storage.update_trade_record(rec.id, updates)
                corrected += 1
                if corrected <= 20:
                    print(f"Corrected: {rec.symbol} pnl={net_pnl:.4f}")
        else:
            failed += 1
    except Exception as e:
        failed += 1
        print(f"Failed: {rec.id[:16]}...: {e}")

print(f"\n=== 对账结果 ===")
print(f"Matched: {matched_count}, Corrected: {corrected}, Failed: {failed}")
