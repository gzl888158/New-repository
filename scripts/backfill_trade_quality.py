"""企业级历史数据回填：统一 trade_records 的口径。

一次性修复三类脏数据：
1. pnl_percent 统一为保证金收益率 pnl/margin*100（修复历史口径不一致）
2. signal_type 空值兜底为 'unknown'（区分「未记录」与 NULL）
3. exit_reason 空值兜底（可选）

用法：
    python scripts/backfill_trade_quality.py          # dry-run，仅统计
    python scripts/backfill_trade_quality.py --apply  # 实际回填
"""
import sqlite3
import sys

DB = r"e:\新建文件夹\okx_quant_trading\data\trading.db"


def main():
    apply = "--apply" in sys.argv
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    # 1) pnl_percent 口径不一致（closed 且 margin>0 且 pnl!=0，但 pnl_percent != pnl/margin*100）
    pct_bad = c.execute("""
        SELECT COUNT(*) FROM trade_records
        WHERE status='closed' AND margin > 0 AND pnl != 0
          AND ABS(COALESCE(pnl_percent,0) - pnl/margin*100) > 0.0001
    """).fetchone()[0]

    # 2) signal_type 空
    st_null = c.execute("""
        SELECT COUNT(*) FROM trade_records
        WHERE signal_type IS NULL OR signal_type = ''
    """).fetchone()[0]

    # 3) exit_reason 空
    er_null = c.execute("""
        SELECT COUNT(*) FROM trade_records
        WHERE exit_reason IS NULL OR exit_reason = ''
    """).fetchone()[0]

    print(f"[dry-run] pnl_percent 口径不一致: {pct_bad}")
    print(f"[dry-run] signal_type 空值: {st_null}")
    print(f"[dry-run] exit_reason 空值: {er_null}")

    if not apply:
        print("\n未执行回填（加 --apply 生效）")
        conn.close()
        return

    if pct_bad:
        c.execute("""
            UPDATE trade_records
            SET pnl_percent = pnl / margin * 100
            WHERE status='closed' AND margin > 0 AND pnl != 0
              AND ABS(COALESCE(pnl_percent,0) - pnl/margin*100) > 0.0001
        """)
        print(f"  已回填 pnl_percent: {c.rowcount} 行")

    if st_null:
        c.execute("UPDATE trade_records SET signal_type='unknown' WHERE signal_type IS NULL OR signal_type=''")
        print(f"  已回填 signal_type: {c.rowcount} 行")

    if er_null:
        c.execute("UPDATE trade_records SET exit_reason='close' WHERE exit_reason IS NULL OR exit_reason=''")
        print(f"  已回填 exit_reason: {c.rowcount} 行")

    conn.commit()
    conn.close()
    print("\n回填完成")


if __name__ == "__main__":
    main()
