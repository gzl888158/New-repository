"""
解决资金不足和极端波动问题
1. 优化资金分配
2. 调整极端波动阈值
"""
import json
import os
import yaml
from datetime import datetime
import shutil


def analyze_capital_allocation():
    """分析当前资金分配"""
    with open("./config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    with open("./data/risk_status.json", "r", encoding="utf-8") as f:
        risk = json.load(f)
    
    current_equity = risk.get("current_equity", 0)
    initial_capital = config.get("trading", {}).get("total_capital", 125)
    trading_ratio = config.get("trading", {}).get("trading_capital_ratio", 0.90)
    
    trading_capital = current_equity * trading_ratio
    
    allocations = {
        "grid": config.get("trading", {}).get("grid_allocation", 0.10),
        "spot_grid": config.get("trading", {}).get("spot_grid_allocation", 0.12),
        "spot_martingale": config.get("trading", {}).get("spot_martingale_allocation", 0.10),
        "trend": config.get("trading", {}).get("trend_allocation", 0.28),
        "scalping": config.get("trading", {}).get("scalping_allocation", 0.20),
        "arbitrage": config.get("trading", {}).get("arbitrage_allocation", 0.20),
    }
    
    total_allocation = sum(allocations.values())
    
    print("=" * 60)
    print("CAPITAL ALLOCATION ANALYSIS")
    print("=" * 60)
    print(f"Current Equity: {current_equity:.2f} USDT")
    print(f"Trading Capital ({trading_ratio*100:.0f}%): {trading_capital:.2f} USDT")
    print(f"Initial Capital: {initial_capital:.2f} USDT")
    print()
    print("STRATEGY ALLOCATIONS:")
    print("-" * 40)
    
    for strategy, alloc in allocations.items():
        capital = trading_capital * alloc
        print(f"  {strategy:20s}: {alloc*100:5.1f}% = {capital:8.2f} USDT")
    
    print("-" * 40)
    print(f"  {'TOTAL':20s}: {total_allocation*100:5.1f}%")
    print()
    
    issues = []
    
    if total_allocation > 1.0:
        issues.append(f"Total allocation {total_allocation*100:.1f}% exceeds 100%")
    
    for strategy, alloc in allocations.items():
        capital = trading_capital * alloc
        min_required = 20  # 最小需要的资金
        if capital < min_required:
            issues.append(f"{strategy} allocation too low: {capital:.2f} USDT < {min_required} USDT minimum")
    
    if issues:
        print("ISSUES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
    
    return {
        "current_equity": current_equity,
        "trading_capital": trading_capital,
        "allocations": allocations,
        "total_allocation": total_allocation,
        "issues": issues,
    }


def optimize_capital_allocation(analysis: dict) -> dict:
    """优化资金分配"""
    trading_capital = analysis["trading_capital"]
    
    # 根据当前权益调整分配比例
    # 权益较高时，提高主动策略分配；权益较低时，提高防御策略分配
    if trading_capital >= 500:
        # 高权益：增加趋势和剥头皮分配
        new_allocations = {
            "grid_allocation": 0.08,
            "spot_grid_allocation": 0.10,
            "spot_martingale_allocation": 0.08,
            "trend_allocation": 0.30,
            "scalping_allocation": 0.24,
            "arbitrage_allocation": 0.20,
        }
    elif trading_capital >= 200:
        # 中等权益：均衡分配
        new_allocations = {
            "grid_allocation": 0.10,
            "spot_grid_allocation": 0.12,
            "spot_martingale_allocation": 0.10,
            "trend_allocation": 0.28,
            "scalping_allocation": 0.20,
            "arbitrage_allocation": 0.20,
        }
    else:
        # 低权益：减少高风险策略，增加稳定策略
        new_allocations = {
            "grid_allocation": 0.12,
            "spot_grid_allocation": 0.15,
            "spot_martingale_allocation": 0.08,
            "trend_allocation": 0.25,
            "scalping_allocation": 0.15,
            "arbitrage_allocation": 0.25,
        }
    
    total = sum(new_allocations.values())
    
    return {
        "allocations": new_allocations,
        "total": total,
        "trading_capital": trading_capital,
    }


def adjust_volatility_threshold():
    """调整极端波动阈值"""
    with open("./config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    current_threshold = config.get("strategies", {}).get("grid", {}).get("volatility_threshold", 0.01)
    
    # 提高阈值以减少误报
    new_threshold = min(current_threshold * 1.5, 0.02)  # 提高到1.5倍，但不超过2%
    
    return {
        "current": current_threshold,
        "recommended": new_threshold,
        "reason": "Increase threshold to reduce false positives during normal market conditions",
    }


def apply_fixes(capital_fix: dict, volatility_fix: dict):
    """应用修复"""
    # 备份配置文件
    backup_path = f"./config.yaml.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy("./config.yaml", backup_path)
    print(f"Config backed up to: {backup_path}")
    
    # 读取配置
    with open("./config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 应用资金分配修复
    for key, value in capital_fix["allocations"].items():
        if "trading" not in config:
            config["trading"] = {}
        config["trading"][key] = value
    
    # 应用波动阈值修复
    if "strategies" not in config:
        config["strategies"] = {}
    if "grid" not in config["strategies"]:
        config["strategies"]["grid"] = {}
    config["strategies"]["grid"]["volatility_threshold"] = volatility_fix["recommended"]
    
    # 增加极端波动暂停时间（从30分钟改为15分钟）
    config["strategies"]["grid"]["extreme_volatility_pause_minutes"] = 15
    
    # 保存配置
    with open("./config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print("\nConfiguration updated successfully!")
    
    return config


def create_strategy_state_reset():
    """创建策略状态重置指令"""
    reset_path = "./data/strategy_state/reset_instructions.json"
    os.makedirs(os.path.dirname(reset_path), exist_ok=True)
    
    instructions = {
        "timestamp": datetime.now().isoformat(),
        "actions": [
            {
                "strategy": "grid",
                "action": "clear_extreme_mode",
                "reason": "Volatility threshold adjusted, resume normal operation"
            },
            {
                "strategy": "all",
                "action": "recalculate_capital",
                "reason": "Capital allocation optimized"
            }
        ]
    }
    
    with open(reset_path, "w", encoding="utf-8") as f:
        json.dump(instructions, f, indent=2, ensure_ascii=False)
    
    print(f"Reset instructions created: {reset_path}")
    
    return instructions


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("TRADING ISSUE FIX SCRIPT")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. 分析资金分配
    print("\n[1] ANALYZING CAPITAL ALLOCATION...")
    analysis = analyze_capital_allocation()
    
    # 2. 优化资金分配
    print("\n[2] OPTIMIZING CAPITAL ALLOCATION...")
    capital_fix = optimize_capital_allocation(analysis)
    
    print("\nNEW ALLOCATIONS:")
    print("-" * 40)
    for key, value in capital_fix["allocations"].items():
        capital = capital_fix["trading_capital"] * value
        print(f"  {key:25s}: {value*100:5.1f}% = {capital:8.2f} USDT")
    print(f"  {'TOTAL':25s}: {capital_fix['total']*100:5.1f}%")
    
    # 3. 调整波动阈值
    print("\n[3] ADJUSTING VOLATILITY THRESHOLD...")
    volatility_fix = adjust_volatility_threshold()
    print(f"  Current threshold: {volatility_fix['current']:.2%}")
    print(f"  New threshold: {volatility_fix['recommended']:.2%}")
    print(f"  Reason: {volatility_fix['reason']}")
    
    # 4. 应用修复
    print("\n[4] APPLYING FIXES...")
    apply_fixes(capital_fix, volatility_fix)
    
    # 5. 创建重置指令
    print("\n[5] CREATING RESET INSTRUCTIONS...")
    create_strategy_state_reset()
    
    print("\n" + "=" * 60)
    print("FIX COMPLETED")
    print("=" * 60)
    print("\nNEXT STEPS:")
    print("1. Restart the trading system to apply new configuration")
    print("2. Monitor logs for improved capital allocation")
    print("3. Verify grid strategies resume normal operation after volatility pause expires")
    
    return 0


if __name__ == "__main__":
    main()