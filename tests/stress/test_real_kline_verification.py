"""真实历史K线离线复核（real historical K-line offline review，direction ③）
========================================================================
方向②用「合成震荡 K 线」对 grid_spacing × tp_pct × sl_pct 做笛卡尔积扫描，
得出「震荡行情应选宽网格(gs≥0.04) + 宽止损(sl≥0.02)」的结论。但合成行情是
±6% 正弦波 + 0.15% 噪声的「完美均值回归」，真实 1H K 线混杂趋势/噪声，结论未必迁移。

本套（direction ③）用 OKX 真实历史 1H K 线（近 30 天，缓存于
`tests/stress/fixtures/real_klines_1H.json`）离线复核方向②结论，验证三点：

  1. 「宽网格(gs≥0.04)」在真实数据上是否稳健成立（≥80% 币种最优 gs 落在宽网格区）
  2. 「宽止损(sl≥0.02)」是否稳健（真实数据最优 sl 是否因币种而异）
  3. 生产默认参数 (gs=0.03/tp=0.02/sl=0.02) 纯参数下是否次优（远低于最优）

离线特性：优先读本地缓存 fixture；缓存缺失时尝试网络抓取并回写；网络不可用则 pytest.skip，
保证本套测试在无网 CI 环境不拖垮全量回归。

与 tests/stress/test_parameter_grid_scan.py（合成数据扫描）区分：
  - 后者用确定性合成 K 线验证「参数空间最优区间」的逻辑。
  - 本套用真实 K 线复核「合成结论能否迁移到真实市场」，是方向②结论的外推性验证。
"""
import json
import os
from datetime import datetime
from itertools import product
from typing import Dict, List, Optional, Tuple

import pytest

from backtest.backtest_engine import BacktestEngine


FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "real_klines_1H.json")
SYMBOLS = ["ETH-USDT-SWAP", "SOL-USDT-SWAP", "XRP-USDT-SWAP", "DOGE-USDT-SWAP", "SUI-USDT-SWAP"]

GRID_SPACINGS = [0.01, 0.02, 0.03, 0.04, 0.05]
TP_PCTS = [0.005, 0.01, 0.015, 0.02, 0.025]
SL_PCTS = [0.005, 0.01, 0.015, 0.02, 0.03]


def _engine() -> BacktestEngine:
    return BacktestEngine({
        "execution": {
            "taker_fee": 0.0005,
            "maker_fee": 0.0002,
            "funding_rate": 0.0001,
            "funding_interval_hours": 8,
        },
        "strategies": {},
    })


def _restore_timestamp(ts) -> datetime:
    """把缓存 JSON 里的时间戳还原为 datetime（JSON 序列化时 datetime 被转成字符串）。"""
    if isinstance(ts, datetime):
        return ts
    return datetime.fromisoformat(str(ts))


def _load_klines(symbol: str) -> List[Dict]:
    """从缓存加载真实 K 线；缓存缺失则网络抓取并回写；网络不可用返回空列表。"""
    candles = None
    if os.path.exists(FIXTURE_PATH):
        try:
            with open(FIXTURE_PATH, encoding="utf-8") as f:
                raw = json.load(f)
            if symbol in raw:
                candles = raw[symbol]
        except (json.JSONDecodeError, OSError):
            candles = None

    if not candles:
        # 缓存缺失 → 尝试网络抓取（需代理可用）
        try:
            import yaml
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            with open(os.path.join(base, "config.yaml"), encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            fetched = BacktestEngine(cfg).fetch_historical_klines(symbol, bar="1H", days=30)
            if len(fetched) < 60:
                return []
            candles = fetched
        except Exception:
            return []

    return [{
        "timestamp": _restore_timestamp(c["timestamp"]),
        "open": float(c["open"]),
        "high": float(c["high"]),
        "low": float(c["low"]),
        "close": float(c["close"]),
        "volume": float(c.get("volume", 0.0)),
    } for c in candles]


def _run_one(eng: BacktestEngine, candles, gs, tp, sl, min_q) -> Dict[str, float]:
    result = eng.run_grid_with_candles(
        candles, "X-USDT-SWAP",
        grid_spacing=gs, tp_pct=tp, sl_pct=sl,
        min_signal_quality=min_q,
        trend_filter_threshold=0.0,
        wear_fee_multiple=0.0,
    )
    summary = result.summary()
    if "error" in summary:
        return {"roi": -100.0, "trades": 0, "win": 0.0}
    return {
        "roi": float(summary["roi_percent"]),
        "trades": int(summary["total_trades"]),
        "win": float(summary["win_rate_percent"]),
    }


# 模块级扫描结果缓存：min_q -> {symbol: (best_combo, default_combo, rows)}
_scan_cache: Dict[float, Dict[str, Tuple]] = {}


def _scan_all(min_q: float) -> Dict[str, Tuple]:
    """对所有币种跑完整笛卡尔积扫描（结果缓存，多个测试共享，避免重复回测）。"""
    if min_q in _scan_cache:
        return _scan_cache[min_q]

    eng = _engine()
    result = {}
    for symbol in SYMBOLS:
        candles = _load_klines(symbol)
        if len(candles) < 60:
            continue
        rows = []
        for gs, tp, sl in product(GRID_SPACINGS, TP_PCTS, SL_PCTS):
            m = _run_one(eng, candles, gs, tp, sl, min_q)
            rows.append((gs, tp, sl, m["roi"], m["trades"], m["win"]))
        best = max(rows, key=lambda r: r[3])
        default = next(r for r in rows if r[0] == 0.03 and r[1] == 0.02 and r[2] == 0.02)
        result[symbol] = (best, default, rows)
    _scan_cache[min_q] = result
    return result


@pytest.fixture(scope="module")
def real_scan():
    """模块级 fixture：返回纯参数扫描（min_q=0）结果，网络/缓存不可用则 skip。"""
    if not os.path.exists(FIXTURE_PATH) and not _load_klines(SYMBOLS[0]):
        pytest.skip("真实K线缓存缺失且网络不可用，跳过 direction ③ 离线复核")
    scan = _scan_all(0.0)
    if len(scan) < 3:
        pytest.skip("真实K线数据不足，跳过")
    return scan


@pytest.fixture(scope="module")
def real_scan_production():
    """模块级 fixture：返回生产信号门槛（min_q=0.35）扫描结果。"""
    if not os.path.exists(FIXTURE_PATH) and not _load_klines(SYMBOLS[0]):
        pytest.skip("真实K线缓存缺失且网络不可用，跳过")
    return _scan_all(0.35)


# ─────────────────────────────────────────────────────────────
# 方向③ 核心复核
# ─────────────────────────────────────────────────────────────
class TestWideGridRobustOnRealData:
    def test_wide_grid_wins_majority_on_real_data(self, real_scan):
        """复核方向②「宽网格」结论：真实数据上 ≥80% 币种的最优 grid_spacing ≥ 0.04。

        方向②合成震荡数据发现最优落在宽网格区(gs≥0.04)，本测试验证该结论在真实 1H K 线
        上是否稳健——若真实数据上宽网格也一致胜出，则「抬升 grid_spacing」是可执行的调参建议；
        若只有合成数据成立，则说明方向②结论是合成行情伪影。
        """
        wide_wins = sum(1 for s, (best, _, _) in real_scan.items() if best[0] >= 0.04)
        total = len(real_scan)
        print("\n[方向③宽网格复核]")
        for s, (best, default, _) in real_scan.items():
            print(f"  {s:16s} 最优gs={best[0]:.3f} 最优ROI={best[3]:.2f}% "
                  f"默认ROI={default[3]:.2f}% 宽网格={'Y' if best[0] >= 0.04 else 'N'}")
        print(f"  >> 宽网格胜出 {wide_wins}/{total}")
        assert wide_wins >= total * 0.8, (
            f"真实数据上宽网格只胜出 {wide_wins}/{total}，方向②「宽网格」结论不稳健"
        )

    def test_production_default_suboptimal_on_real_data(self, real_scan):
        """复核方向②「生产默认次优」：真实数据上生产默认 (gs=0.03) 纯参数 ROI 应显著低于最优。"""
        for s, (best, default, _) in real_scan.items():
            assert best[3] > default[3], (
                f"{s}: 最优 ROI={best[3]:.2f}% 未优于生产默认 ROI={default[3]:.2f}%"
            )


class TestWideStopLossNotRobust:
    def test_optimal_sl_varies_across_symbols(self, real_scan):
        """复核方向②「宽止损」结论：真实数据上最优 sl_pct 应因币种而异（部分窄止损更优）。

        方向②合成震荡数据发现最优 sl 落在 [0.02,0.03]（宽止损），但真实数据混杂趋势，
        窄止损能快速止血。若真实数据上最优 sl 出现 <0.02 的情况（窄止损），
        则「宽止损」结论是合成数据伪影，不应无差别抬高 stop_loss_pct。
        """
        sls = sorted({best[2] for best, _, _ in real_scan.values()})
        print(f"\n[方向③宽止损复核] 真实数据各币种最优 sl_pct 集合 = {sls}")
        assert min(sls) < 0.02, (
            f"真实数据所有币种最优 sl_pct 都 ≥0.02（{sls}），与「宽止损是合成伪影」的预期不符，"
            f"需重新评估方向②宽止损结论"
        )


class TestSignalQualityDominant:
    def test_signal_quality_beats_pure_parameters(self, real_scan, real_scan_production):
        """复核 20260914 报告「信号质量才是命门」：min_signal_quality=0.35 下最优 ROI
        应高于纯参数（min_q=0）最优 ROI，且胜率 > 50%。

        方向②在纯参数空间（关闭信号门槛）找最优，本测试验证生产信号门槛叠加后，
        收益与胜率是否进一步提升——印证「信号质量」比「网格间距/止损」更根本。
        """
        pure_best = max(best[3] for best, _, _ in real_scan.values())
        prod_best = max(best[3] for best, _, _ in real_scan_production.values())
        print(f"\n[方向③信号质量复核] 纯参数最优 ROI={pure_best:.2f}% vs 生产门槛最优 ROI={prod_best:.2f}%")
        assert prod_best > pure_best, (
            f"生产信号门槛最优 ROI={prod_best:.2f}% 未高于纯参数最优 ROI={pure_best:.2f}%，"
            f"与「信号质量是命门」结论矛盾"
        )
        # 生产门槛下胜率应显著高于纯参数的 ~14%（见 20260914 报告）
        prod_win = max(best[5] for best, _, _ in real_scan_production.values())
        print(f"  生产门槛最优胜率={prod_win:.1f}%")
        assert prod_win > 50.0, f"生产门槛最优胜率={prod_win:.1f}% 未突破 50% 盈亏平衡线"


class TestSignalQualityFlipsGridSpacing:
    """信号质量门槛翻转 grid_spacing 最优方向（方向③最核心洞察）。

    纯参数（关闭信号门槛）下最优 grid_spacing 落在宽网格区(≥0.04)；
    生产信号门槛(0.35)下最优 grid_spacing 翻转为窄网格区(≤0.03)。

    经济学直觉：信号质量门槛与宽网格是两种可互相替代的「噪声过滤器」——
    已用信号门槛提纯入场，就不需要再用宽网格过滤，窄网格反而提高资金利用率。

    因此方向②「抬升 grid_spacing 到 [0.04,0.05]」的结论只在「关闭信号门槛」的退化场景成立；
    生产环境信号门槛已开启（config grid.min_signal_quality=0.122039 且 locked），
    应保持窄网格甚至收窄，而非抬升。
    """

    def test_signal_quality_prefers_narrow_grid(self, real_scan, real_scan_production):
        """信号门槛下最优 grid_spacing 应落在窄网格区(≤0.03)，与纯参数的宽网格相反。"""
        prod_gs = [best[0] for best, _, _ in real_scan_production.values()]
        print(f"\n[方向③网格方向翻转] 纯参数最优gs集合={sorted(best[0] for best, _, _ in real_scan.values())} "
              f"vs 信号门槛最优gs集合={sorted(prod_gs)}")
        assert all(gs <= 0.03 for gs in prod_gs), (
            f"信号门槛下最优 grid_spacing 应 ≤0.03（窄网格），实际={prod_gs}，"
            f"与「信号门槛翻转网格方向」洞察矛盾"
        )

    def test_signal_quality_narrows_vs_pure_wide(self, real_scan, real_scan_production):
        """信号门槛下的最优 gs 应严格窄于纯参数下的最优 gs（方向翻转）。"""
        pure_gs = [best[0] for best, _, _ in real_scan.values()]
        prod_gs = [best[0] for best, _, _ in real_scan_production.values()]
        flip_count = sum(1 for p, q in zip(pure_gs, prod_gs) if q < p)
        total = len(prod_gs)
        print(f"[方向③翻转] 信号门槛使最优 gs 变窄的币种: {flip_count}/{total}")
        assert flip_count >= total * 0.8, (
            f"信号门槛只使 {flip_count}/{total} 币种的最优 gs 变窄，翻转现象不显著"
        )
