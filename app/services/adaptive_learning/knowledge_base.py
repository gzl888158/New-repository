"""
知识库 - 交易经验存储、案例检索与知识蒸馏

核心能力:
  - 结构化经验存储 (市场状态 → 策略行为 → 结果)
  - 基于相似度的案例检索 (k-NN)
  - 知识蒸馏 (旧经验压缩为新规则)
  - 经验衰减 (时间加权遗忘)
  - 持久化存储
"""
import asyncio
import json
import os
import time
from collections import OrderedDict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from app.services.enterprise import EnterpriseServiceMixin


class ExperienceType(Enum):
    SUCCESS = "success"         # 盈利交易经验
    FAILURE = "failure"         # 亏损交易教训
    PATTERN = "pattern"         # 市场模式识别
    RULE = "rule"               # 提炼的规则
    ANOMALY = "anomaly"         # 异常事件记录


class Experience:
    """经验条目"""
    def __init__(self, exp_type: ExperienceType, market_state: Dict[str, Any],
                 strategy: str, action: str, outcome: Dict[str, Any],
                 confidence: float = 0.5, tags: List[str] = None):
        self.exp_type = exp_type
        self.market_state = market_state  # 市场状态特征向量
        self.strategy = strategy
        self.action = action
        self.outcome = outcome            # {pnl, return_pct, duration, ...}
        self.confidence = confidence
        self.tags = tags or []
        self.timestamp = datetime.now()
        self.access_count = 0
        self.weight = 1.0                 # 衰减权重
        self.id = f"exp_{int(time.time()*1000)}_{hash(str(outcome)) % 10000}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.exp_type.value,
            "market_state": self.market_state,
            "strategy": self.strategy,
            "action": self.action,
            "outcome": self.outcome,
            "confidence": self.confidence,
            "tags": self.tags,
            "timestamp": self.timestamp.isoformat(),
            "access_count": self.access_count,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Experience":
        exp = cls(
            exp_type=ExperienceType(d["type"]),
            market_state=d["market_state"],
            strategy=d["strategy"],
            action=d["action"],
            outcome=d["outcome"],
            confidence=d.get("confidence", 0.5),
            tags=d.get("tags", []),
        )
        exp.id = d.get("id", f"exp_{int(time.time()*1000)}")
        exp.access_count = d.get("access_count", 0)
        exp.weight = d.get("weight", 1.0)
        exp.timestamp = datetime.fromisoformat(d["timestamp"])
        return exp


class KnowledgeBase(EnterpriseServiceMixin):
    """交易知识库"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        kb_cfg = self._config.get("knowledge_base", {})

        # ── 经验存储 ──
        self._experiences: Dict[str, Experience] = {}  # id -> Experience
        self._index: Dict[str, List[str]] = {}         # strategy -> [exp_id]

        # ── 衰减配置 ──
        self._decay_half_life = kb_cfg.get("decay_half_life_hours", 168)  # 7天半衰期
        self._max_experiences = kb_cfg.get("max_experiences", 10000)
        self._min_weight = kb_cfg.get("min_weight", 0.01)

        # ── 蒸馏配置 ──
        self._distill_threshold = kb_cfg.get("distill_threshold", 50)  # 同类经验50条触发蒸馏
        self._rules: Dict[str, Dict[str, Any]] = {}

        # ── 持久化 ──
        self._persist_dir = kb_cfg.get("persist_dir", "./data/knowledge")
        self._persist_interval = kb_cfg.get("persist_interval_seconds", 300)
        self._last_persist = 0

        # ── 统计 ──
        self._stats = {"total_stored": 0, "total_retrieved": 0, "total_distilled": 0,
                        "by_type": {}, "by_strategy": {}}

        self._lock = asyncio.Lock()
        self._running = False

        os.makedirs(self._persist_dir, exist_ok=True)
        self._load()
        logger.info(f"KnowledgeBase initialized: max={self._max_experiences}, "
                     f"decay_half_life={self._decay_half_life}h")

    async def start(self):
        self._running = True
        logger.info("KnowledgeBase started")

    async def stop(self):
        self._running = False
        await self._persist()
        logger.info("KnowledgeBase stopped")

    # ===================== 存储 =====================

    async def store_experience(self, experience: Experience) -> str:
        """存储经验"""
        async with self._lock:
            # 容量控制
            if len(self._experiences) >= self._max_experiences:
                self._prune()

            self._experiences[experience.id] = experience
            self._index.setdefault(experience.strategy, []).append(experience.id)
            self._stats["total_stored"] += 1
            self._stats["by_type"][experience.exp_type.value] = \
                self._stats["by_type"].get(experience.exp_type.value, 0) + 1
            self._stats["by_strategy"][experience.strategy] = \
                self._stats["by_strategy"].get(experience.strategy, 0) + 1

            # 自动持久化
            if time.time() - self._last_persist > self._persist_interval:
                asyncio.create_task(self._persist())
                self._last_persist = time.time()

            # 蒸馏检查
            strategy_count = len(self._index.get(experience.strategy, []))
            if strategy_count >= self._distill_threshold:
                asyncio.create_task(self._distill_strategy(experience.strategy))

            return experience.id

    async def store_trade_result(self, strategy: str, symbol: str, market_state: Dict,
                                  action: str, pnl: float, return_pct: float,
                                  duration: float, confidence: float = 0.5,
                                  tags: List[str] = None):
        """便捷方法：存储交易结果"""
        exp_type = ExperienceType.SUCCESS if pnl > 0 else ExperienceType.FAILURE
        exp = Experience(
            exp_type=exp_type,
            market_state=market_state,
            strategy=strategy,
            action=action,
            outcome={"pnl": pnl, "return_pct": return_pct, "duration": duration,
                      "symbol": symbol},
            confidence=confidence,
            tags=tags or [],
        )
        return await self.store_experience(exp)

    # ===================== 检索 =====================

    async def query_similar(self, market_state: Dict[str, Any], strategy: str = None,
                             exp_type: ExperienceType = None, top_k: int = 5,
                             min_similarity: float = 0.3) -> List[Experience]:
        """检索相似市场状态下的经验"""
        async with self._lock:
            candidates = []
            if strategy and strategy in self._index:
                ids = self._index[strategy]
                candidates = [self._experiences[eid] for eid in ids
                              if eid in self._experiences and
                              (exp_type is None or self._experiences[eid].exp_type == exp_type)]
            else:
                candidates = list(self._experiences.values())
                if exp_type:
                    candidates = [e for e in candidates if e.exp_type == exp_type]

            if not candidates:
                return []

            # 计算相似度
            scored = []
            for exp in candidates:
                sim = self._compute_similarity(market_state, exp.market_state)
                if sim >= min_similarity:
                    exp.access_count += 1
                    exp.weight *= 0.999  # 微小衰减
                    scored.append((sim * exp.weight, exp))

            scored.sort(key=lambda x: x[0], reverse=True)
            self._stats["total_retrieved"] += min(len(scored), top_k)
            return [exp for _, exp in scored[:top_k]]

    async def query_by_tags(self, tags: List[str], top_k: int = 10) -> List[Experience]:
        """按标签检索"""
        async with self._lock:
            matches = []
            for exp in self._experiences.values():
                if any(t in exp.tags for t in tags):
                    matches.append(exp)
            matches.sort(key=lambda e: e.weight * (e.outcome.get("pnl", 0) / max(abs(e.outcome.get("pnl", 0)), 1)),
                          reverse=True)
            return matches[:top_k]

    async def get_best_practices(self, strategy: str, top_k: int = 5) -> List[Dict]:
        """获取策略最佳实践"""
        async with self._lock:
            successes = []
            for eid in self._index.get(strategy, []):
                exp = self._experiences.get(eid)
                if exp and exp.exp_type == ExperienceType.SUCCESS:
                    successes.append(exp)
            successes.sort(key=lambda e: e.outcome.get("return_pct", 0) * e.weight, reverse=True)
            return [e.to_dict() for e in successes[:top_k]]

    # ===================== 衰减 =====================

    def _apply_decay(self):
        """应用时间衰减"""
        now = datetime.now()
        for exp in self._experiences.values():
            hours_elapsed = (now - exp.timestamp).total_seconds() / 3600
            exp.weight = max(self._min_weight,
                             2 ** (-hours_elapsed / self._decay_half_life))

    def _prune(self):
        """裁剪低权重经验"""
        self._apply_decay()
        sorted_exps = sorted(self._experiences.items(),
                             key=lambda x: x[1].weight)
        to_remove = sorted_exps[:max(1, len(sorted_exps) - self._max_experiences + 500)]
        for eid, exp in to_remove:
            del self._experiences[eid]
            if exp.strategy in self._index:
                self._index[exp.strategy] = [i for i in self._index[exp.strategy] if i != eid]

    # ===================== 蒸馏 =====================

    async def _distill_strategy(self, strategy: str):
        """将策略经验蒸馏为规则"""
        async with self._lock:
            successes = []
            failures = []
            for eid in self._index.get(strategy, []):
                exp = self._experiences.get(eid)
                if not exp:
                    continue
                if exp.exp_type == ExperienceType.SUCCESS:
                    successes.append(exp)
                elif exp.exp_type == ExperienceType.FAILURE:
                    failures.append(exp)

            if len(successes) < 10:
                return

            # 分析成功案例的共性
            avg_confidence = sum(e.confidence for e in successes) / len(successes)
            avg_pnl = sum(e.outcome.get("return_pct", 0) for e in successes) / len(successes)

            # 提取常见市场特征
            feature_means = {}
            for exp in successes:
                for k, v in exp.market_state.items():
                    if isinstance(v, (int, float)):
                        feature_means[k] = feature_means.get(k, 0) + v / len(successes)

            rule = {
                "strategy": strategy,
                "confidence_threshold": max(0.3, avg_confidence - 0.1),
                "target_return": avg_pnl,
                "feature_profile": feature_means,
                "success_count": len(successes),
                "failure_count": len(failures),
                "success_rate": len(successes) / max(len(successes) + len(failures), 1),
                "distilled_at": datetime.now().isoformat(),
            }

            self._rules[strategy] = rule
            self._stats["total_distilled"] += 1
            logger.info(f"Knowledge distilled for {strategy}: "
                         f"success_rate={rule['success_rate']:.1%}, "
                         f"avg_return={avg_pnl:.2%}")

    def get_rule(self, strategy: str) -> Optional[Dict]:
        """获取策略蒸馏规则"""
        return self._rules.get(strategy)

    # ===================== 相似度 =====================

    def _compute_similarity(self, state1: Dict, state2: Dict) -> float:
        """计算两个市场状态的余弦相似度"""
        common_keys = set(state1.keys()) & set(state2.keys())
        if not common_keys:
            return 0.0

        dot = 0.0
        norm1 = 0.0
        norm2 = 0.0
        for k in common_keys:
            v1 = float(state1.get(k, 0)) if isinstance(state1.get(k), (int, float)) else 0
            v2 = float(state2.get(k, 0)) if isinstance(state2.get(k), (int, float)) else 0
            dot += v1 * v2
            norm1 += v1 * v1
            norm2 += v2 * v2

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return max(0.0, min(1.0, dot / ((norm1 ** 0.5) * (norm2 ** 0.5))))

    # ===================== 持久化 =====================

    async def _persist(self):
        """持久化到磁盘"""
        try:
            data = {
                "experiences": [e.to_dict() for e in self._experiences.values()],
                "index": self._index,
                "rules": self._rules,
                "stats": self._stats,
            }
            path = os.path.join(self._persist_dir, "knowledge_base.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, default=str)
            self._last_persist = time.time()
            logger.debug(f"KnowledgeBase persisted: {len(self._experiences)} experiences")
        except Exception as e:
            self._handle_exception(
                e, module="KnowledgeBase", function="_persist",
                severity="medium", category="persistence",
            )

    def _load(self):
        """从磁盘加载"""
        path = os.path.join(self._persist_dir, "knowledge_base.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for exp_dict in data.get("experiences", []):
                exp = Experience.from_dict(exp_dict)
                self._experiences[exp.id] = exp
            self._index = data.get("index", {})
            self._rules = data.get("rules", {})
            self._stats = data.get("stats", self._stats)
            logger.info(f"KnowledgeBase loaded: {len(self._experiences)} experiences, "
                         f"{len(self._rules)} rules")
        except Exception as e:
            self._handle_exception(
                e, module="KnowledgeBase", function="_load",
                severity="medium", category="persistence",
            )

    # ===================== 统计查询 =====================

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "total_experiences": len(self._experiences),
            "total_rules": len(self._rules),
            "strategies_indexed": list(self._index.keys()),
        }

    def get_experiences(self, strategy: str = None, exp_type: ExperienceType = None,
                        limit: int = 50) -> List[Dict]:
        """获取经验列表"""
        exps = list(self._experiences.values())
        if strategy:
            exps = [e for e in exps if e.strategy == strategy]
        if exp_type:
            exps = [e for e in exps if e.exp_type == exp_type]
        exps.sort(key=lambda e: e.weight, reverse=True)
        return [e.to_dict() for e in exps[:limit]]
