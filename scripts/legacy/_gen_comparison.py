# -*- coding: utf-8 -*-
"""生成 P0/P1 优化前后对比分析报告（只读查询 trading.db + 本地状态文件）。"""
import sqlite3
import json
import os
from datetime import datetime
from collections import Counter

DB = "data/trading.db"
CUT = "2026-08-19 09:15:00"          # 优化后起点
BASELINE_CUT = "2026-08-19 08:45:00"  # 优化前基线采集点


def conn_ro():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


def q(cur, sql, p=()):
    return [dict(r) for r in cur.execute(sql, p).fetchall()]


def agg_trades(cur, where_sql, params=()):
    """按方向/策略聚合 trades 表：胜率、PnL、手续费、笔数、磨损型占比。"""
    rows = q(cur, f"""
        SELECT strategy_name, direction,
               COUNT(*) AS cnt,
               SUM(CASE WHEN win=1 THEN 1 ELSE 0 END) AS wins,
               SUM(CASE WHEN win=0 THEN 1 ELSE 0 END) AS losses,
               SUM(pnl_usdt) AS pnl_usdt,
               SUM(fees) AS fees,
               SUM(CASE WHEN pnl_usdt >= -0.01 AND pnl_usdt <= 0 THEN 1 ELSE 0 END) AS wear
        FROM trades
        WHERE {where_sql}
        GROUP BY strategy_name, direction
    """, params)
    return rows


def summarize(rows):
    """把聚合行汇总成策略级与总体统计。"""
    by_strategy = {}
    total = {"cnt": 0, "wins": 0, "losses": 0, "pnl_usdt": 0.0, "fees": 0.0, "wear": 0}
    by_direction = {}
    for r in rows:
        s = r["strategy_name"]
        d = r["direction"]
        cnt = r["cnt"] or 0
        wins = r["wins"] or 0
        losses = r["losses"] or 0
        pnl = r["pnl_usdt"] or 0.0
        fees = r["fees"] or 0.0
        wear = r["wear"] or 0
        if s not in by_strategy:
            by_strategy[s] = {"cnt": 0, "wins": 0, "losses": 0, "pnl_usdt": 0.0, "fees": 0.0, "wear": 0}
        b = by_strategy[s]
        b["cnt"] += cnt; b["wins"] += wins; b["losses"] += losses
        b["pnl_usdt"] += pnl; b["fees"] += fees; b["wear"] += wear
        total["cnt"] += cnt; total["wins"] += wins; total["losses"] += losses
        total["pnl_usdt"] += pnl; total["fees"] += fees; total["wear"] += wear
        if d not in by_direction:
            by_direction[d] = {"cnt": 0, "wins": 0, "pnl_usdt": 0.0, "fees": 0.0}
        by_direction[d]["cnt"] += cnt
        by_direction[d]["wins"] += wins
        by_direction[d]["pnl_usdt"] += pnl
        by_direction[d]["fees"] += fees

    def winrate(x):
        return round(x["wins"] / x["cnt"], 4) if x["cnt"] else None

    def finalize(x):
        return {
            "cnt": x["cnt"], "wins": x["wins"], "losses": x["losses"],
            "win_rate": winrate(x),
            "pnl_usdt": round(x["pnl_usdt"], 6), "fees": round(x["fees"], 6),
            "wear_cnt": x.get("wear", 0),
            "wear_ratio": round(x.get("wear", 0) / x["cnt"], 4) if x.get("cnt") else None,
        }

    return {
        "overall": finalize(total),
        "by_strategy": {k: finalize(v) for k, v in by_strategy.items()},
        "by_direction": {k: {"cnt": v["cnt"], "wins": v["wins"],
                              "win_rate": round(v["wins"]/v["cnt"], 4) if v["cnt"] else None,
                              "pnl_usdt": round(v["pnl_usdt"], 6), "fees": round(v["fees"], 6)}
                         for k, v in by_direction.items()},
    }


def main():
    cur = conn_ro().cursor()

    # 1) 交易数据：优化后（exit_time >= 09:15）
    post_rows = agg_trades(cur, "exit_time >= ?", (CUT,))
    post = summarize(post_rows)

    # 2) 交易数据：优化前参考（exit_time < 08:45，用于同口径交叉验证）
    pre_rows = agg_trades(cur, "exit_time < ?", (BASELINE_CUT,))
    pre = summarize(pre_rows)

    # 3) 账户权益 / 资金利用率（account_history）
    acc_before = q(cur, "SELECT total_equity, used_margin, timestamp FROM account_history "
                        "WHERE timestamp <= ? ORDER BY timestamp DESC LIMIT 1", (BASELINE_CUT,))
    acc_after = q(cur, "SELECT total_equity, used_margin, timestamp FROM account_history "
                       "WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT 1", (CUT,))
    acc_after_earliest = q(cur, "SELECT total_equity, used_margin, timestamp FROM account_history "
                                "WHERE timestamp >= ? ORDER BY timestamp ASC LIMIT 1", (CUT,))

    def util(r):
        if not r or not r[0].get("total_equity"):
            return None
        e = r[0]["total_equity"]
        m = r[0]["used_margin"] or 0.0
        return {"total_equity": round(e, 4), "used_margin": round(m, 4),
                "utilization": round(m / e, 4) if e else None, "timestamp": r[0]["timestamp"]}

    capital = {
        "baseline_point": util(acc_before),
        "post_opt_earliest": util(acc_after_earliest),
        "post_opt_latest": util(acc_after),
    }

    # 4) 信号审批率
    sig = {"baseline": {"total_audits": 2214, "approved": 194,
                        "approve_rate": round(194 / 2214, 4)}}
    agent_state = {}
    try:
        with open("data/intelligent_agent_state.json", encoding="utf-8") as f:
            agent_state = json.load(f)
    except Exception as e:
        agent_state = {"_error": str(e)}
    ds = agent_state.get("decision_stats", {})
    sig["current_decision_stats"] = {
        "total_audits": ds.get("total_audits"),
        "approved": ds.get("approved"),
        "rejected": ds.get("rejected"),
        "reduced": ds.get("reduced"),
        "delayed": ds.get("delayed"),
        "approve_rate": round(ds.get("approved", 0) / ds.get("total_audits", 1), 4) if ds.get("total_audits") else None,
        "note": "decision_stats 计数器已于 last_reset_time 重置，无法与基线 2214 次审计做同口径减法对比",
        "last_reset_time": agent_state.get("last_reset_time"),
    }
    # 决策审计链（时间戳维度，meta_verdict 均为 proceed，非审批通过语义，仅供参考）
    chain_after = 0
    try:
        with open("data/decision_audit_chain.json", encoding="utf-8") as f:
            chain = json.load(f)
        entries = chain.get("entries", [])
        chain_after = sum(1 for e in entries if e.get("timestamp", "") >= "2026-08-19T09:15:00")
    except Exception:
        pass
    sig["decision_audit_chain_entries_after_opt"] = chain_after

    # 5) P0/P1 配置落地核验
    config_checks = {}
    try:
        import yaml
        with open("config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        g = cfg.get("strategies", {}).get("grid", {})
        t = cfg.get("strategies", {}).get("trend", {})
        s = cfg.get("strategies", {}).get("scalping", {})
        config_checks = {
            "grid_long_only": g.get("long_only"),
            "grid_min_grid_spacing": g.get("min_grid_spacing"),
            "grid_min_signal_quality": g.get("min_signal_quality"),
            "trend_shared_vote_adx_floor": t.get("shared_vote_adx_floor"),
            "scalping_min_signal_quality": s.get("min_signal_quality"),
        }
    except Exception as e:
        config_checks = {"_error": str(e)}
    learned = agent_state.get("learned_params", {})
    config_checks["learned_grid_min_signal_quality"] = learned.get("grid_min_signal_quality")
    config_checks["learned_grid_min_confidence_threshold"] = learned.get("grid_min_confidence_threshold")
    config_checks["learned_scalping_min_signal_quality"] = learned.get("scalping_min_signal_quality")

    # 6) 黑名单核验（P0-3 清理 DOT/LINK/ADA）
    blacklist = agent_state.get("blacklist", {})
    config_checks["current_blacklist_keys"] = sorted(blacklist.keys())

    # 样本充足性判定
    def sufficiency(cnt):
        if cnt >= 20:
            return "样本充足"
        return "样本不足，结论待观察"

    post_suff = {
        "overall": sufficiency(post["overall"]["cnt"]),
        "grid": sufficiency(post["by_strategy"].get("grid", {}).get("cnt", 0)),
        "trend": sufficiency(post["by_strategy"].get("trend", {}).get("cnt", 0)),
        "scalping": sufficiency(post["by_strategy"].get("scalping", {}).get("cnt", 0)),
    }

    # P0/P1 各项优化效果判定
    grid_post_cnt = post["by_strategy"].get("grid", {}).get("cnt", 0)
    grid_post_wear = post["by_strategy"].get("grid", {}).get("wear_ratio")
    grid_shorts_post = next(
        (r["cnt"] for r in post_rows if r["strategy_name"] == "grid" and r["direction"] == "short"), 0
    )
    bl_keys = set(config_checks.get("current_blacklist_keys", []))

    p0p1_verdicts = [
        {
            "id": "P0-1",
            "item": "grid 策略 long_only（只做多，关闭做空）",
            "verdict": "无效（未生效）",
            "reason": "config.yaml 已写入 grid.long_only=true，但优化后（09:15 之后）实盘仍有 4 笔 grid 空单（08-20 14:51~14:55 入场）。"
                      "疑似运行进程未重启加载新配置，或 long_only 未在 grid 下单链路上真正生效。",
            "evidence": {"grid_short_trades_post_opt": grid_shorts_post},
        },
        {
            "id": "P0-2",
            "item": "grid min_grid_spacing 0.01→0.015",
            "verdict": "待观察",
            "reason": "config.yaml 已写入 min_grid_spacing=0.015，但优化后 grid 磨损型占比反而升至 "
                      f"{grid_post_wear if grid_post_wear is not None else 'N/A'}（基线约 0.44），且 long_only 未生效造成混淆，样本仅 {grid_post_cnt} 笔，无法归因。",
            "evidence": {"grid_wear_ratio_post_opt": grid_post_wear, "grid_post_cnt": grid_post_cnt},
        },
        {
            "id": "P0-3",
            "item": "清理 DOT/LINK/ADA 过期黑名单",
            "verdict": "有效",
            "reason": "当前黑名单仅剩 XRP/SUI/UNI（均为未过期项），DOT/LINK/ADA 已不再出现在黑名单中。",
            "evidence": {"current_blacklist_keys": sorted(bl_keys)},
        },
        {
            "id": "P0-4",
            "item": "grid 信号阈值下调（config 0.45→0.35；learned 0.40→0.30、0.40→0.30）",
            "verdict": "待观察（配置已落地但交易频率未提升）",
            "reason": "config grid.min_signal_quality=0.35、learned grid_min_signal_quality=0.30、grid_min_confidence_threshold=0.30 均已写入。"
                      f"但优化后 grid 仅成交 {grid_post_cnt} 笔（约 31 小时），频率低于基线日均水平，暗示阈值下调未在实盘生效（可能因进程未重启）。",
            "evidence": {
                "config_grid_min_signal_quality": config_checks.get("grid_min_signal_quality"),
                "learned_grid_min_signal_quality": config_checks.get("learned_grid_min_signal_quality"),
                "learned_grid_min_confidence_threshold": config_checks.get("learned_grid_min_confidence_threshold"),
                "grid_post_opt_cnt": grid_post_cnt,
            },
        },
        {
            "id": "P1-1",
            "item": "trend shared_vote_adx_floor 15.0→20.0",
            "verdict": "待观察",
            "reason": "config.yaml 已写入 trend.shared_vote_adx_floor=20.0。优化后 trend 胜率升至 62.5%（5/8），但样本仅 8 笔且 PnL 仍为负（-0.3649 USDT），无法确认因果。",
            "evidence": {"trend_post_opt": post["by_strategy"].get("trend")},
        },
        {
            "id": "P1-2",
            "item": "scalping_min_signal_quality 0.15→0.12",
            "verdict": "待观察",
            "reason": "learned scalping_min_signal_quality=0.12 已写入（config scalping.min_signal_quality=0.1）。但优化后 scalping 无任何成交，无样本可评估。",
            "evidence": {"scalping_post_opt_cnt": post["by_strategy"].get("scalping", {}).get("cnt", 0)},
        },
        {
            "id": "P1-3",
            "item": "修复 core/intelligent_agent.py 的 _save_state 过期黑名单持久化 bug",
            "verdict": "有效",
            "reason": "_save_state 现仅持久化未过期黑名单（until > now），当前黑名单仅含 3 个未过期项，无过期残留，修复已落地。",
            "evidence": {"current_blacklist_keys": sorted(bl_keys)},
        },
    ]

    report = {
        "report_title": "P0/P1 优化前后对比分析报告",
        "generated_at": datetime.now().isoformat(),
        "baseline_captured_at": "2026-08-19 08:45:00",
        "post_opt_start": CUT,
        "data_source": {"trades_table": "data/trading.db (trades, read-only)",
                        "account_history": "data/trading.db (account_history, read-only)",
                        "signal_state": "data/intelligent_agent_state.json, data/decision_audit_chain.json"},
        "baseline": {
            "overall": {"total_trades": 1046, "wins": 380, "losses": 666, "win_rate": 0.367},
            "grid": {"total_trades": 1032, "win_rate": 0.361, "pnl_usdt": -1.0915, "fees": 2.5855},
            "direction": {"long": {"cnt": 274, "pnl_usdt": 6.0311}, "short": {"cnt": 772, "pnl_usdt": -0.0303}},
            "capital_utilization": {"total_equity": 208.0, "used_margin": 52.0, "utilization": 0.25},
            "signal_approval": sig["baseline"],
        },
        "pre_opt_computed_from_db": pre,  # exit_time < 08:45 同口径参考
        "post_optimization": {
            "sample_size": post["overall"]["cnt"],
            "sample_sufficient": post["overall"]["cnt"] >= 20,
            "overall": post["overall"],
            "by_strategy": post["by_strategy"],
            "by_direction": post["by_direction"],
        },
        "capital_utilization": capital,
        "signal_approval": sig,
        "config_landing_checks": config_checks,
        "sample_sufficiency": post_suff,
        "p0p1_verdicts": p0p1_verdicts,
    }

    os.makedirs("data/reports", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"data/reports/comparison_report_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print("已保存报告:", out_path)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
