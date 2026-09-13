"""
交易系统重置脚本
解决长时间无交易问题，重置策略状态和回撤
"""
import json
import os
import sys
from datetime import datetime


def reset_strategy_states():
    """重置所有策略状态"""
    state_dir = "./data/strategy_state"
    if not os.path.exists(state_dir):
        print(f"[INFO] State directory not found: {state_dir}")
        return
    
    files = [f for f in os.listdir(state_dir) if f.endswith(".json")]
    print(f"[INFO] Found {len(files)} strategy state files")
    
    for filename in files:
        filepath = os.path.join(state_dir, filename)
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                state = json.load(f)
            
            modified = False
            
            if "current_drawdown" in state:
                original = state["current_drawdown"]
                if original > 1.0 or original < 0:
                    state["current_drawdown"] = 0.0
                    modified = True
                    print(f"[FIX] {filename}: current_drawdown {original:.2%} -> 0.0%")
            
            if "_current_drawdown" in state:
                original = state["_current_drawdown"]
                if original > 1.0 or original < 0:
                    state["_current_drawdown"] = 0.0
                    modified = True
                    print(f"[FIX] {filename}: _current_drawdown {original:.2%} -> 0.0%")
            
            if "_equity_initialized" in state:
                state["_equity_initialized"] = False
                modified = True
                print(f"[FIX] {filename}: _equity_initialized reset to False")
            
            if "_max_daily_equity" in state:
                state["_max_daily_equity"] = 0.0
                modified = True
                print(f"[FIX] {filename}: _max_daily_equity reset to 0")
            
            if "daily_reset_date" in state:
                state["daily_reset_date"] = None
                modified = True
                print(f"[FIX] {filename}: daily_reset_date reset to None")
            
            if modified:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(state, f, ensure_ascii=False, indent=2)
                print(f"[INFO] Saved changes to {filename}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")


def create_risk_control_reset():
    """创建风控重置指令"""
    control_path = "./data/risk_control.json"
    
    control_data = {
        "action": "reset_pause",
        "timestamp": datetime.now().isoformat(),
        "reason": "manual reset via reset_trading.py"
    }
    
    with open(control_path, "w", encoding="utf-8") as f:
        json.dump(control_data, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] Created risk control reset at {control_path}")


def check_current_status():
    """检查当前系统状态"""
    print("\n" + "="*60)
    print("CURRENT SYSTEM STATUS")
    print("="*60)
    
    risk_path = "./data/risk_status.json"
    if os.path.exists(risk_path):
        with open(risk_path, "r", encoding="utf-8") as f:
            risk_data = json.load(f)
        
        print(f"\nRisk Control:")
        print(f"  Current Equity: {risk_data.get('current_equity', 0):.2f} USDT")
        print(f"  Peak Equity: {risk_data.get('peak_equity', 0):.2f} USDT")
        print(f"  Max Drawdown Threshold: {risk_data.get('max_drawdown', 0):.2%}")
        print(f"  Is Paused: {risk_data.get('is_paused', False)}")
        print(f"  Pause Reason: {risk_data.get('pause_reason', 'None')}")
        print(f"  Tier Triggered: {risk_data.get('tier_triggered', {})}")
        print(f"  Process Running: {risk_data.get('process_running', False)}")
    
    state_dir = "./data/strategy_state"
    if os.path.exists(state_dir):
        print(f"\nStrategy States:")
        for filename in os.listdir(state_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(state_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        state = json.load(f)
                    
                    drawdown = state.get("current_drawdown") or state.get("_current_drawdown", 0)
                    daily_pnl = state.get("daily_pnl", 0)
                    daily_start = state.get("daily_start_equity", 0)
                    
                    print(f"  {filename}:")
                    print(f"    Drawdown: {drawdown:.2%}")
                    print(f"    Daily PnL: {daily_pnl:.2f} USDT")
                    print(f"    Daily Start: {daily_start:.2f} USDT")
                except Exception:
                    print(f"  {filename}: Error reading")


def main():
    """主函数"""
    print("="*60)
    print("TRADING SYSTEM RESET TOOL")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    check_current_status()
    
    print("\n" + "="*60)
    print("PERFORMING RESET...")
    print("="*60)
    
    reset_strategy_states()
    create_risk_control_reset()
    
    print("\n" + "="*60)
    print("RESET COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Restart the trading system")
    print("2. Monitor the logs for any issues")
    print("3. Check that strategies start generating signals")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())