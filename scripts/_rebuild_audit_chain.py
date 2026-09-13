"""一次性重建策略审计日志哈希链（修复进程重启导致的 event_id 重复与链断裂）。"""
import json
import hashlib
import shutil

path = r"e:\新建文件夹\okx_quant_trading\data\strategy_manager\audit.jsonl"
backup = path + ".bak"

# 1. 备份原文件
shutil.copy2(path, backup)

# 2. 读取全部记录（保持文件顺序 = 时间顺序）
entries = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        entries.append(json.loads(line))

# 3. 重新编号 event_id 并重算哈希链
prev_hash = ""
for i, e in enumerate(entries, start=1):
    e["event_id"] = f"strat_audit_{i:06d}"
    e["hash_prev"] = prev_hash
    e.pop("hash_current", None)
    content = json.dumps(e, sort_keys=True, ensure_ascii=False, default=str)
    e["hash_current"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    prev_hash = e["hash_current"]

# 4. 写回
with open(path, "w", encoding="utf-8") as f:
    for e in entries:
        f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")

# 5. 验证链连续
continuous = True
for i in range(1, len(entries)):
    if entries[i]["hash_prev"] != entries[i - 1]["hash_current"]:
        continuous = False
        print(f"VERIFY FAIL at index {i}")
        break

print(f"rebuilt {len(entries)} entries, chain_continuous={continuous}, backup={backup}")
