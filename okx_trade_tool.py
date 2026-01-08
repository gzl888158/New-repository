import os
import time
import csv
import pandas as pd
from okx import OKXClient
from datetime import datetime

# ========== 从环境变量读取 API 密钥（安全无泄露） ==========
API_KEY = os.getenv("OKX_API_KEY")
SECRET_KEY = os.getenv("OKX_SECRET_KEY")
PASSPHRASE = os.getenv("OKX_PASSPHRASE")
FLAG = os.getenv("OKX_FLAG", "0")  # 默认实盘

# ========== 初始化欧易客户端 ==========
client = OKXClient(
    api_key=API_KEY,
    secret_key=SECRET_KEY,
    passphrase=PASSPHRASE,
    flag=FLAG
)

# ========== 交易日志保存（自动写入 CSV） ==========
def init_trade_log():
    if not os.path.exists("trade_history.csv"):
        with open("trade_history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["时间", "币种", "方向", "仓位", "止盈价", "止损价", "保证金使用率", "AI建议"])

def write_trade_log(inst_id, side, sz, tp_price, sl_price, mgn_ratio, ai_tip):
    init_trade_log()
    with open("trade_history.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            inst_id,
            side,
            sz,
            tp_price,
            sl_price,
            mgn_ratio,
            ai_tip
        ])

# ========== 核心策略：止盈止损 + AI 分析 + 风险控制 ==========
class OKXTradeBot:
    def __init__(self, inst_id="BTC-USDT-SWAP", leverage=10):
        self.inst_id = inst_id
        self.leverage = leverage
        self.set_leverage()

    # 设置杠杆
    def set_leverage(self):
        res = client.trade.set_leverage(
            instId=self.inst_id,
            lever=self.leverage,
            mgnMode="cross"
        )
        if res["code"] == "0":
            print(f"✅ {self.inst_id} 杠杆设置为 {self.leverage} 倍")
        else:
            print(f"❌ 杠杆设置失败：{res['msg']}")

    # 获取保证金使用率（风险预警）
    def get_margin_ratio(self):
        res = client.account.get_account()
        mgn_ratio = float(res["data"][0]["mgnRatio"]) * 100
        if mgn_ratio >= 80:
            print("🚨 保证金≥80%：强制限制开仓！")
        elif mgn_ratio >= 70:
            print("⚠️ 保证金≥70%：建议减仓！")
        elif mgn_ratio >= 50:
            print("ℹ️ 保证金≥50%：仓位偏重！")
        return f"{mgn_ratio:.2f}%"

    # 30分钟 AI 行情解读（专业指标+小白话术）
    def ai_market_analysis(self):
        # 获取 15m K线 + 多空比
        candles = client.market.get_candlesticks(instId=self.inst_id, bar="15m", limit=20)
        last_close = float(candles["data"][0][4])
        prev_close = float(candles["data"][1][4])
        long_short = client.market.get_long_short_ratio(instId=self.inst_id, period="15m")
        ratio = float(long_short["data"][0]["longShortRatio"])

        # 趋势判断
        trend = "上涨" if last_close > prev_close else "下跌"
        ratio_tip = "多头占优" if ratio > 1.2 else "空头占优" if ratio < 0.8 else "多空平衡"
        ai_tip = f"{self.inst_id} 当前价 {last_close:.2f} USDT，15m {trend}，多空比 {ratio:.2f}（{ratio_tip}）→ 建议：{'持有多单' if trend == '上涨' and ratio>1.2 else '持有空单' if trend == '下跌' and ratio<0.8 else '观望'}"
        print(f"📊 AI 解读：{ai_tip}")
        return ai_tip

    # 止盈止损开仓
    def place_tp_sl_order(self, side="buy", sz="0.01", tp_pct=5, sl_pct=2):
        ticker = client.market.get_ticker(instId=self.inst_id)
        last_price = float(ticker["data"][0]["last"])
        tp_price = last_price * (1 + tp_pct/100) if side == "buy" else last_price * (1 - tp_pct/100)
        sl_price = last_price * (1 - sl_pct/100) if side == "buy" else last_price * (1 + sl_pct/100)

        res = client.trade.place_order(
            instId=self.inst_id,
            tdMode="cross",
            side=side,
            ordType="market",
            sz=sz,
            tpTriggerPx=str(tp_price),
            tpOrdPx=str(tp_price),
            slTriggerPx=str(sl_price),
            slOrdPx=str(sl_price)
        )

        if res["code"] == "0":
            mgn_ratio = self.get_margin_ratio()
            ai_tip = self.ai_market_analysis()
            print(f"✅ {side} 单开仓成功！止盈 {tp_price:.2f} | 止损 {sl_price:.2f}")
            write_trade_log(self.inst_id, side, sz, tp_price, sl_price, mgn_ratio, ai_tip)
        else:
            print(f"❌ 开仓失败：{res['msg']}")

# ========== 启动机器人 ==========
if __name__ == "__main__":
    bot = OKXTradeBot(inst_id="BTC-USDT-SWAP", leverage=10)
    bot.ai_market_analysis()  # 执行 AI 分析
    bot.get_margin_ratio()    # 检查保证金风险
    # 如需自动开仓，取消下面注释 ↓
    # bot.place_tp_sl_order(side="buy", sz="0.01")