# -*- coding: utf-8 -*-
import os, sys, asyncio, json
sys.path.insert(0, r'e:\新建文件夹\okx_quant_trading')
os.chdir(r'e:\新建文件夹\okx_quant_trading')

async def main():
    from core.okx_client import OKXClient
    from configs.settings import load_config
    cfg = load_config()
    client = OKXClient(cfg)
    
    try:
        pos = await client.get_positions_async()
        print(f"=== 持仓原始数据 (共{len(pos) if pos else 0}条) ===")
        if pos:
            total_margin = 0
            for p in pos:
                inst = p.get('instId', '?')
                pos_side = p.get('posSide', '?')
                pos_qty = p.get('pos', '0')
                # 只显示有实际仓位的
                try:
                    qty = float(pos_qty) if pos_qty not in ('', None) else 0
                except:
                    qty = 0
                if qty == 0:
                    continue
                def f(x):
                    try:
                        return float(x) if x not in ('', None) else 0.0
                    except:
                        return 0.0
                avg = f(p.get('avgPx', 0))
                upl = f(p.get('upl', 0))
                margin = f(p.get('imr', 0))
                mmr = f(p.get('mmr', 0))
                total_margin += margin
                print(f"{inst:15s} | {pos_side:6s} | 数量={qty:+.4f} | 均价={avg:.6f} | 浮盈={upl:+.4f} | 保证金={margin:.4f} | 维持保证金={mmr:.4f}")
            print(f"\n总保证金占用: {total_margin:.4f} USDT")
        else:
            print("无持仓")
    except Exception as e:
        print(f"持仓查询失败: {e}")
        # 打印原始
        try:
            pos = await client.get_positions_async()
            print(json.dumps(pos, ensure_ascii=False, indent=2, default=str)[:3000])
        except Exception as e2:
            print(f"再次失败: {e2}")
    
    client.close()

asyncio.run(main())
