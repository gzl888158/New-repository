"""
PnL对账模块
定期从OKX拉取账单(bills)和已实现盈亏历史，校正数据库中的pnl记录
解决历史PnL=0的假数据问题
"""
import asyncio
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from loguru import logger


def _finite(value: Any, default: float = 0.0) -> float:
    """安全数值转换：None/非法字符串/NaN/Inf 统一回退到 default。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


class PnLReconciler:
    """PnL对账器：数据库 vs OKX实际账单"""

    def __init__(self, config: Dict[str, Any], okx_client, sqlite_storage):
        self.config = config
        self.okx_client = okx_client
        self.sqlite_storage = sqlite_storage

        # 对账周期：持仓对账15分钟，完整对账1小时（P2: 6h→1h，缩短手动平仓后 pnl 显示 0 的窗口）
        self._position_reconcile_interval = 15 * 60  # 15 min
        self._full_reconcile_interval = 1 * 3600  # 1 h
        self._reconcile_interval = self._position_reconcile_interval
        self._last_full_reconcile: Optional[datetime] = None
        # 最近一次对账时间
        self._last_reconcile: Optional[datetime] = None
        # 对账统计
        self._stats = {
            "total_reconciled": 0,
            "corrected_records": 0,
            "failed_corrections": 0,
            "last_run": None
        }

    @staticmethod
    def _norm_side(side) -> str:
        """统一方向口径：buy/long -> long，sell/short -> short，其余原样返回。"""
        s = (side or "").strip().lower()
        if s in ("buy", "long"):
            return "long"
        if s in ("sell", "short"):
            return "short"
        return s

    async def start(self):
        """启动对账循环"""
        logger.info("PnLReconciler started, position_interval=15min, full_interval=1h")
        # 启动后先等5分钟再首次执行（避开系统启动高峰）
        await asyncio.sleep(300)
        while True:
            try:
                now = datetime.now()
                # 完整对账：1小时一次
                if (self._last_full_reconcile is None or
                    (now - self._last_full_reconcile).total_seconds() >= self._full_reconcile_interval):
                    await self.reconcile_all()
                    self._last_full_reconcile = now
                else:
                    # 持仓对账：15分钟一次（轻量版）
                    await self._reconcile_open_positions()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"PnL reconciliation loop error: {e}")
            await asyncio.sleep(self._reconcile_interval)

    async def reconcile_all(self) -> Dict[str, Any]:
        """执行完整对账流程"""
        logger.info("=" * 50)
        logger.info("Starting PnL reconciliation with OKX...")
        start_time = datetime.now()

        # 1. 先对账 open 记录与 OKX 实际持仓：幽灵持仓标 closed（pnl 暂空）
        open_result = await self._reconcile_open_positions()

        # 2. 再回填已平仓记录 pnl/fees（含刚标记的幽灵仓：平仓账单仍在 7 天窗口内，
        #    可在同一对账周期内立即回填，避免下一周期 close_time 被清理时刻污染导致漏配）
        closed_result = await self._reconcile_closed_records()

        # 3. 对账账户总盈亏
        account_result = await self._reconcile_account_pnl()

        self._last_reconcile = datetime.now()
        self._last_full_reconcile = self._last_reconcile
        self._stats["last_run"] = self._last_reconcile.isoformat()
        self._stats["total_reconciled"] += 1

        duration = (datetime.now() - start_time).total_seconds()
        result = {
            "duration_seconds": duration,
            "closed_reconciled": closed_result,
            "open_reconciled": open_result,
            "account_reconciled": account_result,
            "timestamp": self._last_reconcile.isoformat()
        }
        logger.info(f"PnL reconciliation completed in {duration:.1f}s: "
                    f"corrected={closed_result['corrected']}, failed={closed_result['failed']}")
        logger.info("=" * 50)
        return result

    async def _reconcile_closed_records(self) -> Dict[str, Any]:
        """对账已平仓记录：从 OKX 平仓账单(close bill, pnl!=0)回填 pnl/filled_price/fees。

        修复历史缺陷：原实现用 orderId / symbol+close_time 匹配，幽灵仓记录的 close_time 是
        清理时刻而非真实平仓时刻，orderId 是开仓订单号，两者都匹配不到平仓账单，导致 pnl=0
        假数据无法回填。现改为 subType 精确识别平仓账单 + symbol+side+[create_time, close_time] 窗口匹配。
        """
        corrected = 0
        failed = 0
        checked = 0

        # 平仓子类型（OKX 账单 subType 真实枚举）：
        #   平多：5=平多、100=强减平多、104=强平平多、112=交割平多
        #   平空：6=平空、101=强减平空、105=强平平空、113=交割平空
        # 注：9=自动减仓/10=穿仓补偿/11=系统换币/12=策略划拨 均非直接平仓账单，剔除。
        CLOSE_LONG_SUBTYPES = {5, 100, 104, 112}
        CLOSE_SHORT_SUBTYPES = {6, 101, 105, 113}
        TIME_BUFFER = timedelta(seconds=60)

        try:
            # 跨「热表 + 月份分片」定位 pnl 缺失(0/null)的已平仓记录
            records = self.sqlite_storage.get_closed_records_missing_pnl(limit=500)

            logger.info(f"Found {len(records)} closed records with pnl=0/null")

            # OKX bills 接口仅保留最近 7 天账单，超出范围的历史 ghost_close 记录
            # 无法逐笔回填，反复尝试只会累积 failed 并浪费 API/日志。过滤掉，避免噪音。
            bills_window = datetime.now() - timedelta(days=7)
            recoverable = [r for r in records
                           if r.get("close_time") and r["close_time"] >= bills_window]
            skipped = len(records) - len(recoverable)
            if skipped:
                logger.info(f"Skipped {skipped} closed records older than 7 days "
                            f"(unrecoverable from bills API)")
            records = recoverable

            if not records:
                return {"checked": 0, "corrected": 0, "failed": 0}

            # 分页拉取交易账单(type=2 交易，含开平仓)，后续按 subType 筛出平仓账单
            okx_bills = self.okx_client.get_all_bills_paginated(
                bill_type="2",
                earliest_ts_ms=None,
                max_pages=10
            )
            if not okx_bills:
                logger.warning("No bills returned from OKX")
                return {"checked": len(records), "corrected": 0, "failed": len(records)}

            # 构建平仓账单列表：[(symbol, side, dt, pnl, fee, fill_px, sz), ...]
            close_bills = []
            for bill in okx_bills:
                try:
                    subtype = int(bill.get("subType", "0") or 0)
                except (TypeError, ValueError):
                    continue
                if subtype in CLOSE_LONG_SUBTYPES:
                    side = "long"
                elif subtype in CLOSE_SHORT_SUBTYPES:
                    side = "short"
                else:
                    continue  # 开仓或非平仓账单，跳过
                pnl = _finite(bill.get("pnl", "0") or 0, 0.0)
                # 保留 fee 符号：OKX 账单 fee 负值表示支付手续费、正值表示返佣，
                # 绝对值会丢失方向性导致返佣被误当作成本扣减。
                fee = _finite(bill.get("fee", "0") or 0, 0.0)
                # 修复历史缺陷：原实现先按 pnl==0 过滤，误杀「平价平仓、仅剩手续费」的
                # 剥头皮平仓账单（pnl=0 但 fee<0），导致这类记录的 pnl/fees 永远无法回填，
                # 污染手续费统计与胜率。仅当 pnl 与 fee 同时为 0 才跳过（无入账价值）。
                if pnl == 0 and fee == 0:
                    continue
                ts = bill.get("ts", "")
                try:
                    dt = datetime.fromtimestamp(int(ts) / 1000) if ts else None
                except (TypeError, ValueError, OverflowError):
                    dt = None
                if not dt:
                    continue
                fill_px = _finite(bill.get("fillPx", "0") or 0, 0.0)
                sz = abs(_finite(bill.get("sz", "0") or 0, 0.0))
                close_bills.append((bill.get("instId", ""), side, dt, pnl, fee, fill_px, sz))

            logger.info(f"Filtered {len(close_bills)} close bills from {len(okx_bills)} trade bills")

            # 逐条对账：symbol + side + [create_time-buffer, close_time+buffer] 时间窗，选最接近 close_time 的平仓账单
            for rec in records:
                checked += 1
                try:
                    rec_side = self._norm_side(rec["side"])
                    if not rec.get("close_time"):
                        failed += 1
                        continue

                    lower = (rec["create_time"] - TIME_BUFFER) if rec.get("create_time") else None
                    upper = rec["close_time"] + TIME_BUFFER

                    # 收集时间窗内所有平仓账单（部分平仓会产生多笔），后续聚合而非取单笔
                    matched = []
                    for b_symbol, b_side, b_dt, b_pnl, b_fee, b_px, b_sz in close_bills:
                        if b_symbol != rec["symbol"] or b_side != rec_side:
                            continue
                        if lower and b_dt < lower:
                            continue
                        if b_dt > upper:
                            continue
                        matched.append((b_pnl, b_fee, b_px, b_sz, b_dt))

                    if matched:
                        # 部分平仓多笔账单聚合：pnl/fee 求和，fill_px 按成交张数(sz)加权
                        total_pnl = sum(b[0] for b in matched)
                        total_fee = sum(b[1] for b in matched)
                        total_sz = sum(b[3] for b in matched)
                        # 真实平仓时刻取最晚一笔账单（多笔部分平仓时以最后一笔为准）
                        close_dt = max(b[4] for b in matched)
                        if total_sz > 0:
                            px = sum(b[2] * b[3] for b in matched) / total_sz
                        else:
                            px = matched[-1][2]
                        # 净额口径：bill.pnl 为毛盈亏，fee 带符号，net = pnl + fee
                        net_pnl = total_pnl + total_fee
                        updates = {"pnl": net_pnl}
                        if px > 0:
                            updates["filled_price"] = px
                        if abs(total_fee) > 0:
                            updates["fees"] = abs(total_fee)
                        # 回填真实平仓时间：幽灵仓 close_time 原为清理时刻（now），
                        # 与真实平仓账单时间错位会导致 7 天可恢复窗口判定与时间归因失真。
                        updates["close_time"] = close_dt.isoformat(sep=" ")
                        # P1: margin 回填 —— 幽灵仓/sync 记录常缺 margin（=0），用开仓名义价值 / 杠杆反算。
                        #     仅基于开仓价格(rec["price"])与数量，不能用 close 的 filled_price 反算。
                        if not rec.get("margin"):
                            try:
                                _entry_px = float(rec.get("price") or 0)
                                _qty = float(rec.get("quantity") or 0)
                                _lev = float(rec.get("leverage") or 0)
                                if _entry_px > 0 and _qty > 0 and _lev > 0:
                                    _calc_margin = _entry_px * _qty / _lev
                                    if _calc_margin > 0:
                                        updates["margin"] = _calc_margin
                            except (TypeError, ValueError, ZeroDivisionError):
                                pass
                        # 标签治理：幽灵仓匹配到真实平仓账单 → 说明是「成交回执丢失的真实平仓」，
                        # 从 ghost_close/ghost_cleanup 重标为 recovered_close，恢复绩效归因
                        # （与真正的幽灵关闭（无任何账单匹配）区分开）。
                        # P1-对账器保留 manual_close：手动平仓标签优先级最高，不被覆盖
                        prev_reason = (rec.get("exit_reason") or "").strip()
                        if prev_reason != "manual_close" and prev_reason in ("ghost_close", "ghost_cleanup", "reconciled", ""):
                            updates["exit_reason"] = "recovered_close"
                        # 记录可能在热表或月份分片，原地回写对应分片
                        shard = rec.get("_shard", "trade_records")
                        self.sqlite_storage.update_trade_in_shard(shard, rec["id"], updates)
                        corrected += 1
                        if corrected <= 10:
                            logger.info(f"Corrected {rec['symbol']} {rec['id'][:16]}...: pnl={net_pnl:.4f}, "
                                        f"fillPx={px:.4f}, fee={total_fee:.4f} (aggregated {len(matched)} bills)")
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                    logger.debug(f"Failed to reconcile record {rec['id'][:16]}...: {e}")

        except Exception as e:
            logger.error(f"Error in _reconcile_closed_records: {e}")

        self._stats["corrected_records"] += corrected
        self._stats["failed_corrections"] += failed
        return {"checked": checked, "corrected": corrected, "failed": failed}

    async def _reconcile_open_positions(self) -> Dict[str, Any]:
        """对账open记录与OKX实际持仓：数据库open但OKX已平仓的，标记为closed"""
        ghost_count = 0
        okx_position_symbols = set()

        try:
            # 获取OKX实际持仓。fail-closed：API 失败或返回 None 时绝不动库，
            # 否则空结果会让所有 open 记录被误判为幽灵持仓并批量关闭。
            checked_query = getattr(self.okx_client, "get_positions_checked", None)
            okx_positions = (
                checked_query()
                if callable(checked_query)
                else self.okx_client.get_positions()
            )
            if okx_positions is None:
                logger.error("Failed to fetch OKX positions for reconciliation, abort ghost cleanup")
                return {"ghost_positions_cleaned": 0, "error": "okx_positions_none"}
            for pos in okx_positions:
                pos_qty = _finite(pos.get("pos", 0), 0.0)
                if pos_qty != 0:
                    okx_position_symbols.add(pos.get("instId", ""))

            # 获取数据库所有open记录
            db_open_records = self.sqlite_storage.get_all_open_records()

            for rec in db_open_records:
                if rec["symbol"] not in okx_position_symbols:
                    # P1-保留 manual_close：已标记手动平仓的记录不视为幽灵仓
                    if (rec.get("exit_reason") or "").strip() == "manual_close":
                        continue
                    # 数据库open但OKX无持仓：幽灵持仓
                    ghost_count += 1
                    logger.warning(f"Ghost position detected: {rec['symbol']} (id={rec['id'][:16]}...)")
                    # 标记为closed，pnl保持null（后续可从bills补全）
                    # 标签治理：与 order_executor._reconcile_positions 统一使用 exit_reason='ghost_close'，
                    # 避免此处漏标导致该记录沉沦为 NULL exit_reason，无法与真实平仓记录区分。
                    self.sqlite_storage.update_trade_record(
                        rec["id"],
                        {"status": "closed", "close_time": datetime.now().isoformat(),
                         "exit_reason": "ghost_close"}
                    )

        except Exception as e:
            logger.error(f"Error in _reconcile_open_positions: {e}")

        if ghost_count > 0:
            logger.warning(f"Found and cleaned {ghost_count} ghost positions")
        return {"ghost_positions_cleaned": ghost_count}

    def _fetch_funding_fee(self) -> float:
        """从 OKX 资金费账单（type=7）拉取对账窗口内的实际资金费率净额（带符号）。

        负值 = 支付资金费（成本），正值 = 收取资金费（收益）。OKX 无独立滑点/点差
        账单，故残差中仅资金费可精确入账，滑点/点差归入 slippage_spread。

        fail-open：失败/无数据时返回 0.0，仅影响残差拆分精度，不阻断 discrepancy 主口径。
        """
        try:
            bills = self.okx_client.get_all_bills_paginated(bill_type="8", max_pages=3)
            if not isinstance(bills, list) or not bills:
                return 0.0
            total = 0.0
            for b in bills:
                if not isinstance(b, dict):
                    continue
                try:
                    total += float(b.get("fee", 0) or 0)
                except (TypeError, ValueError):
                    continue
            return total
        except Exception as e:
            logger.debug(f"Failed to fetch funding fee for reconciliation: {e}")
            return 0.0

    async def _reconcile_account_pnl(self) -> Dict[str, Any]:
        """对账账户总盈亏：数据库累计pnl vs OKX账户权益变化。

        除日志外，将 discrepancy 分解为「未实现盈亏」与「未归因残差」并持久化，
        使资金费率/滑点/点差/存取款等未入账成本可被追溯（解决只记日志不回写）。
        残差进一步拆分为资金费（OKX 账单精确入账）与滑点+点差+记录噪声残差。
        """
        try:
            # 数据库累计已实现 pnl（以 TradeJournal trades.pnl_usdt 权威口径为准，
            # trade_records 的 pnl 长期被 ghost_close 污染导致严重低估）
            db_total_pnl = self.sqlite_storage.get_authoritative_realized_pnl()

            # OKX账户权益 - 动态基准 = 实际总盈亏（含未实现）
            account = self.okx_client.get_account_info()
            if not account:
                logger.error("Failed to fetch OKX account info for reconciliation, abort account PnL reconcile")
                return {"error": "account_info_none"}
            okx_equity = _finite(account.get("totalEq", 0), 0.0)
            # 未实现盈亏：OKX账户余额接口的 upl 字段（浮盈浮亏）
            unrealized_pnl = _finite(account.get("upl", 0) or 0, 0.0)

            # 动态基准：静态 total_capital 会把出入金也误计入盈亏。改用持久化的
            # 基准权益（首次用配置 total_capital），并在检测到外部资金流动时同步调整基准。
            config_capital = _finite(self.config.get("trading", {}).get("total_capital", 100) or 0, 0.0)
            prev = self.sqlite_storage.get_latest_pnl_reconciliation()
            baseline = (prev.get("initial_capital") if prev else None) or config_capital

            # 防漂移护栏：baseline（初始资本）理论上恒为非负、且应贴近配置 total_capital。
            # 当 db 已实现盈亏口径失真时，残差推断会把真实盈亏误判为出入金，导致 baseline
            # 漂移失控（曾漂移到负数 -672，制造 discrepancy=848 假象）。检测到历史失真即重置。
            if baseline < 0 or (config_capital > 0 and
                                abs(baseline - config_capital) > config_capital * 0.5):
                logger.warning(
                    f"baseline 异常 ({baseline:.4f}) 与配置 total_capital({config_capital:.4f}) "
                    f"偏差过大，判定为历史 external_flow 漂移，重置为配置值"
                )
                baseline = config_capital

            external_flow = 0.0
            if prev and prev.get("okx_equity") is not None:
                equity_delta = okx_equity - prev["okx_equity"]
                realized_delta = db_total_pnl - (prev.get("db_realized_pnl") or 0.0)
                unrealized_delta = unrealized_pnl - (prev.get("unrealized_pnl") or 0.0)
                # 出入金 = 权益变化 - 已实现盈亏变化 - 未实现盈亏变化
                external_flow = equity_delta - realized_delta - unrealized_delta
                # 阈值：超过 1 USDT 且占权益 0.5% 以上才视为出入金（过滤噪声/费率残差）
                threshold = max(1.0, okx_equity * 0.005)
                if abs(external_flow) > threshold:
                    # 单次出入金上限：真实出入金（充值/提现）应有明确操作，单次不会超过
                    # 权益的 50%（或 100 USDT 兜底）。超过则大概率是 db 失真误判，告警不调整。
                    max_single_flow = max(okx_equity * 0.5, 100.0)
                    if abs(external_flow) > max_single_flow:
                        logger.warning(
                            f"external_flow {external_flow:+.4f} 超过单次上限 {max_single_flow:.2f}，"
                            f"疑似 db 已实现盈亏口径失真，拒绝自动调整 baseline，需人工核对出入金"
                        )
                    else:
                        baseline += external_flow
                        if baseline < 0:
                            logger.warning(f"baseline 漂移到负数 {baseline:.4f}，clamp 到 0（需人工核对）")
                            baseline = 0.0
                        logger.info(f"External capital flow detected: {external_flow:+.4f} USDT, "
                                    f"baseline adjusted to {baseline:.4f}")

            okx_total_pnl = okx_equity - baseline

            discrepancy = okx_total_pnl - db_total_pnl
            # 未归因残差 = 资金费率 + 滑点 + 点差 + 存取款 + 其他未入账项
            unattributed = discrepancy - unrealized_pnl

            # 残差入账：资金费从 OKX 资金费账单精确入账，剩余归入滑点+点差+记录噪声残差
            funding_fee = self._fetch_funding_fee()
            slippage_spread = unattributed - funding_fee

            recon = {
                "initial_capital": baseline,
                "okx_equity": okx_equity,
                "okx_total_pnl": okx_total_pnl,
                "db_realized_pnl": db_total_pnl,
                "unrealized_pnl": unrealized_pnl,
                "discrepancy": discrepancy,
                "unattributed": unattributed,
                "funding_fee": funding_fee,
                "slippage_spread": slippage_spread,
                "external_flow": external_flow,
            }
            # 回写：持久化对账快照
            self.sqlite_storage.save_pnl_reconciliation(recon)

            logger.info(f"Account PnL reconciliation: DB={db_total_pnl:.4f}, OKX={okx_total_pnl:.4f}, "
                        f"unrealized={unrealized_pnl:.4f}, discrepancy={discrepancy:.4f}, "
                        f"unattributed={unattributed:.4f} (funding_fee={funding_fee:.4f}, "
                        f"slippage_spread={slippage_spread:.4f})")

            return {**recon}
        except Exception as e:
            logger.error(f"Error in _reconcile_account_pnl: {e}")
            return {"error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """获取对账统计"""
        return {
            **self._stats,
            "last_reconcile": self._last_reconcile.isoformat() if self._last_reconcile else None
        }
