"""
遗传算法参数优化器 (Genetic Algorithm Optimizer)

生产级遗传算法实现，用于交易策略参数优化：
  - 多种选择策略：锦标赛、轮盘赌、排名选择
  - 多种交叉算子：模拟二进制交叉(SBX)、均匀交叉、单点交叉
  - 多项式变异 + 自适应变异率
  - 精英保留 + 种群多样性维护
  - 收敛检测 + 早期停止
  - 并行适应度评估
  - Pareto 前沿多目标优化
"""
import asyncio
import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Callable, Set
import numpy as np
from loguru import logger


# ═══════════════════════════════════════════════════════════════
# 枚举与数据模型
# ═══════════════════════════════════════════════════════════════

class SelectionMethod(Enum):
    """选择策略"""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"

class CrossoverMethod(Enum):
    """交叉策略"""
    SBX = "sbx"                   # 模拟二进制交叉
    UNIFORM = "uniform"           # 均匀交叉
    SINGLE_POINT = "single_point" # 单点交叉

class ConvergenceReason(Enum):
    MAX_GENERATIONS = "max_generations"
    LOW_VARIANCE = "low_variance"
    NO_IMPROVEMENT = "no_improvement"
    TARGET_REACHED = "target_reached"

@dataclass
class ParameterDef:
    """参数定义"""
    name: str
    type: str = "float"              # float / int / categorical
    low: float = 0.0
    high: float = 1.0
    step: float = 0.0                # 离散化步长（0=连续）
    categories: List[Any] = None     # 分类参数的取值列表
    log_scale: bool = False          # 是否对数尺度搜索
    description: str = ""

    def sample(self) -> float:
        if self.type == "categorical" and self.categories:
            return random.choice(self.categories)
        if self.log_scale:
            log_low, log_high = math.log10(max(self.low, 1e-10)), math.log10(max(self.high, 1e-10))
            v = 10 ** (log_low + random.random() * (log_high - log_low))
        else:
            v = self.low + random.random() * (self.high - self.low)
        if self.type == "int":
            v = round(v)
        elif self.step > 0:
            v = round(v / self.step) * self.step
        return self.clamp(v)

    def clamp(self, value: float) -> float:
        if value is None:
            value = self.low
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = self.low
        if not math.isfinite(value):
            value = self.low
        if not math.isfinite(self.low):
            low = 0.0
        else:
            low = self.low
        if not math.isfinite(self.high):
            high = low
        else:
            high = self.high
        v = min(max(value, low), high)
        if self.type == "int":
            v = round(v)
        elif self.step > 0:
            v = round(v / self.step) * self.step
        return v


@dataclass
class Individual:
    """种群个体"""
    genes: List[float]               # 参数值列表
    fitness: float = float('-inf')
    objectives: Dict[str, float] = field(default_factory=dict)
    generation: int = 0
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = f"{random.randint(0, 999999):06d}"


@dataclass
class GenerationStats:
    """每代统计"""
    generation: int
    best_fitness: float
    avg_fitness: float
    median_fitness: float
    worst_fitness: float
    std_fitness: float
    population_diversity: float       # 种群基因标准差均值
    elapsed_seconds: float
    evaluations: int


@dataclass
class GAOptimizationResult:
    """遗传算法优化结果"""
    best_individual: Individual
    best_params: Dict[str, float]
    best_fitness: float
    generation_stats: List[GenerationStats] = field(default_factory=list)
    convergence_reason: ConvergenceReason = ConvergenceReason.MAX_GENERATIONS
    convergence_generation: int = 0
    total_evaluations: int = 0
    total_time_seconds: float = 0.0
    final_population: List[Individual] = field(default_factory=list)
    fitness_history: List[float] = field(default_factory=list)
    pareto_front: List[Individual] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        def _finite(v: Any, default: float = 0.0) -> Any:
            """将数值安全转为有限 float；非数值（如字符串类别）原样保留，NaN/Inf 归零。"""
            if isinstance(v, bool):
                return v
            try:
                f = float(v)
            except (TypeError, ValueError):
                return v
            return f if math.isfinite(f) else default

        return {
            "best_params": {k: _finite(v) for k, v in self.best_params.items()},
            "best_fitness": _finite(self.best_fitness),
            "convergence_reason": self.convergence_reason.value,
            "convergence_generation": _finite(self.convergence_generation),
            "total_evaluations": _finite(self.total_evaluations),
            "total_time_seconds": _finite(self.total_time_seconds),
            "generation_count": len(self.generation_stats),
            "final_population_size": len(self.final_population),
            "pareto_front_size": len(self.pareto_front),
            "fitness_history": [round(f, 6) for f in self.fitness_history[-20:] if math.isfinite(f)],
            "timeline": [
                {"gen": s.generation,
                 "best": _finite(s.best_fitness),
                 "avg": _finite(s.avg_fitness),
                 "diversity": _finite(s.population_diversity)}
                for s in self.generation_stats
            ],
            "timestamp": self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════
# 遗传算法优化器
# ═══════════════════════════════════════════════════════════════

class GeneticOptimizer:
    """遗传算法参数优化器"""

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config.get("genetic_optimizer", {}) if config else {}
        self._population_size = cfg.get("population_size", 50)
        self._generations = cfg.get("generations", 100)
        self._elite_count = cfg.get("elite_count", 3)
        self._tournament_size = cfg.get("tournament_size", 3)
        self._crossover_rate = cfg.get("crossover_rate", 0.8)
        self._mutation_rate = cfg.get("mutation_rate", 0.1)
        self._mutation_strength = cfg.get("mutation_strength", 0.2)
        self._selection_method = SelectionMethod(cfg.get("selection_method", "tournament"))
        self._crossover_method = CrossoverMethod(cfg.get("crossover_method", "sbx"))
        # 自适应变异
        self._adaptive_mutation = cfg.get("adaptive_mutation", True)
        self._min_mutation_rate = cfg.get("min_mutation_rate", 0.01)
        # 收敛检测
        self._convergence_patience = cfg.get("convergence_patience", 20)
        self._convergence_variance = cfg.get("convergence_variance_threshold", 1e-6)
        self._target_fitness = cfg.get("target_fitness", float('inf'))
        # 多样性维护
        self._crowding_distance = cfg.get("crowding_distance", False)
        self._diversity_threshold = cfg.get("diversity_threshold", 0.05)
        self._restart_on_stagnation = cfg.get("restart_on_stagnation", False)
        # 并行
        self._parallel_eval = cfg.get("parallel_eval", True)
        self._max_workers = cfg.get("max_workers", 4)
        # SBX参数
        self._sbx_eta = cfg.get("sbx_eta", 20)
        self._pm_eta = cfg.get("pm_eta", 20)

        self._seed = cfg.get("seed", None)

        self._param_defs: List[ParameterDef] = []
        self._fitness_fn: Optional[Callable] = None
        self._population: List[Individual] = []
        self._best_individual: Optional[Individual] = None
        self._generation_stats: List[GenerationStats] = []
        self._fitness_history: List[float] = []
        self._eval_count = 0
        self._lock = asyncio.Lock()

        logger.info(f"GeneticOptimizer initialized: pop={self._population_size}, "
                    f"gen={self._generations}, selection={self._selection_method.value}, "
                    f"crossover={self._crossover_method.value}")

    # ── 参数定义 ──────────────────────────────────────────────

    def set_param_defs(self, param_defs: List[ParameterDef]):
        """设置搜索空间参数定义"""
        self._param_defs = param_defs
        logger.info(f"Parameter definitions set: {len(param_defs)} parameters")

    def add_param(self, name: str, low: float = 0, high: float = 1,
                  param_type: str = "float", step: float = 0,
                  categories: List = None, log_scale: bool = False,
                  description: str = ""):
        """添加单个参数定义"""
        self._param_defs.append(ParameterDef(
            name=name, type=param_type, low=low, high=high,
            step=step, categories=categories, log_scale=log_scale,
            description=description,
        ))

    # ── 种群初始化 ────────────────────────────────────────────

    def _initialize_population(self) -> List[Individual]:
        """初始化随机种群（含拉丁超立方采样）"""
        population = []
        n = len(self._param_defs)
        # 使用拉丁超立方采样提高初始覆盖
        lhs_segments = min(self._population_size, 20)
        for i in range(lhs_segments):
            genes = []
            for pd in self._param_defs:
                if pd.log_scale:
                    log_low = math.log10(max(pd.low, 1e-10))
                    log_high = math.log10(max(pd.high, 1e-10))
                    seg = (i + random.random()) / lhs_segments
                    v = 10 ** (log_low + seg * (log_high - log_low))
                else:
                    seg = (i + random.random()) / lhs_segments
                    v = pd.low + seg * (pd.high - pd.low)
                if pd.type == "int":
                    v = round(v)
                elif pd.step > 0:
                    v = round(v / pd.step) * pd.step
                genes.append(pd.clamp(v))
            population.append(Individual(genes=genes, generation=0))

        # 填充剩余个体
        for _ in range(max(0, self._population_size - lhs_segments)):
            genes = [pd.sample() for pd in self._param_defs]
            population.append(Individual(genes=genes, generation=0))

        random.shuffle(population)
        return population[:self._population_size]

    # ── 适应度评估 ────────────────────────────────────────────

    def set_fitness_fn(self, fn: Callable[[Dict[str, float]], float]):
        """设置适应度函数 fn(params_dict) -> float"""
        self._fitness_fn = fn

    async def _evaluate_individual(self, ind: Individual) -> float:
        """评估单个个体适应度"""
        if not self._fitness_fn:
            return 0.0
        params = {pd.name: ind.genes[i] for i, pd in enumerate(self._param_defs)}
        try:
            result = self._fitness_fn(params)
            if hasattr(result, '__await__'):
                result = await result
            value = float(result)
            if not math.isfinite(value):
                return float('-inf')
            return value
        except Exception as e:
            logger.debug(f"Fitness evaluation error: {e}")
            return float('-inf')

    async def _evaluate_population(self, population: List[Individual]):
        """并行评估种群适应度"""
        if self._parallel_eval:
            tasks = [self._evaluate_individual(ind) for ind in population]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for ind, r in zip(population, results):
                if isinstance(r, Exception):
                    ind.fitness = float('-inf')
                elif r is None:
                    ind.fitness = float('-inf')
                else:
                    try:
                        v = float(r)
                    except (TypeError, ValueError):
                        v = float('-inf')
                    ind.fitness = v if math.isfinite(v) else float('-inf')
                self._eval_count += 1
        else:
            for ind in population:
                ind.fitness = await self._evaluate_individual(ind)
                self._eval_count += 1

    # ── 选择操作 ──────────────────────────────────────────────

    def _tournament_select(self, population: List[Individual]) -> Individual:
        """锦标赛选择"""
        if not population:
            raise ValueError("Cannot select from empty population")
        k = max(1, min(self._tournament_size, len(population)))
        candidates = random.sample(population, k)
        return max(candidates, key=lambda x: x.fitness)

    def _roulette_select(self, population: List[Individual]) -> Individual:
        """轮盘赌选择"""
        if not population:
            raise ValueError("Cannot select from empty population")
        finite_fits = [ind.fitness for ind in population if math.isfinite(ind.fitness)]
        if not finite_fits:
            return random.choice(population)
        min_fit = min(finite_fits)
        shifted = []
        for ind in population:
            f = ind.fitness if math.isfinite(ind.fitness) else min_fit
            shifted.append(max(f - min_fit + 1e-10, 1e-10))
        total = sum(shifted)
        if total <= 0 or not math.isfinite(total):
            return random.choice(population)
        pick = random.random() * total
        cumulative = 0
        for ind, fit in zip(population, shifted):
            cumulative += fit
            if cumulative >= pick:
                return ind
        return population[-1]

    def _rank_select(self, population: List[Individual]) -> Individual:
        """排名选择（线性排名）"""
        if not population:
            raise ValueError("Cannot select from empty population")
        sorted_pop = sorted(population, key=lambda x: x.fitness)
        n = len(sorted_pop)
        ranks = [i + 1 for i in range(n)]  # 1 = worst, n = best
        total = sum(ranks)
        pick = random.random() * total
        cumulative = 0
        for ind, rank in zip(sorted_pop, ranks):
            cumulative += rank
            if cumulative >= pick:
                return ind
        return sorted_pop[-1]

    def _select_parent(self, population: List[Individual]) -> Individual:
        if self._selection_method == SelectionMethod.TOURNAMENT:
            return self._tournament_select(population)
        elif self._selection_method == SelectionMethod.ROULETTE:
            return self._roulette_select(population)
        else:
            return self._rank_select(population)

    # ── 交叉操作 ──────────────────────────────────────────────

    def _sbx_crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        """模拟二进制交叉 (Simulated Binary Crossover)

        强化：分类参数不能做算术交叉（否则产生非法取值），改为按交叉率交换。
        """
        n = len(p1.genes)
        c1_genes, c2_genes = [], []
        for i in range(n):
            pd = self._param_defs[i] if i < len(self._param_defs) else None
            if pd and pd.type == "categorical":
                # 分类参数：交换保持合法取值
                if random.random() < self._crossover_rate:
                    c1_genes.append(p2.genes[i])
                    c2_genes.append(p1.genes[i])
                else:
                    c1_genes.append(p1.genes[i])
                    c2_genes.append(p2.genes[i])
                continue
            if random.random() < self._crossover_rate:
                u = random.random()
                if u <= 0.5:
                    beta = (2.0 * u) ** (1.0 / (max(self._sbx_eta, 0) + 1))
                else:
                    beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (max(self._sbx_eta, 0) + 1))
                v1 = 0.5 * ((1 + beta) * p1.genes[i] + (1 - beta) * p2.genes[i])
                v2 = 0.5 * ((1 - beta) * p1.genes[i] + (1 + beta) * p2.genes[i])
                if pd:
                    v1, v2 = pd.clamp(v1), pd.clamp(v2)
            else:
                v1, v2 = p1.genes[i], p2.genes[i]
            c1_genes.append(v1)
            c2_genes.append(v2)
        return Individual(genes=c1_genes), Individual(genes=c2_genes)

    def _uniform_crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        """均匀交叉"""
        c1_genes, c2_genes = [], []
        for i in range(len(p1.genes)):
            if random.random() < self._crossover_rate:
                if random.random() < 0.5:
                    c1_genes.append(p1.genes[i])
                    c2_genes.append(p2.genes[i])
                else:
                    c1_genes.append(p2.genes[i])
                    c2_genes.append(p1.genes[i])
            else:
                c1_genes.append(p1.genes[i])
                c2_genes.append(p2.genes[i])
        return Individual(genes=c1_genes), Individual(genes=c2_genes)

    def _single_point_crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        """单点交叉"""
        if random.random() < self._crossover_rate and len(p1.genes) > 1:
            point = random.randint(1, len(p1.genes) - 1)
            c1_genes = p1.genes[:point] + p2.genes[point:]
            c2_genes = p2.genes[:point] + p1.genes[point:]
            return Individual(genes=c1_genes), Individual(genes=c2_genes)
        return Individual(genes=p1.genes.copy()), Individual(genes=p2.genes.copy())

    def _crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        if self._crossover_method == CrossoverMethod.SBX:
            return self._sbx_crossover(p1, p2)
        elif self._crossover_method == CrossoverMethod.UNIFORM:
            return self._uniform_crossover(p1, p2)
        else:
            return self._single_point_crossover(p1, p2)

    # ── 变异操作 ──────────────────────────────────────────────

    def _polynomial_mutation(self, ind: Individual, mutation_rate: float) -> Individual:
        """多项式变异

        强化：分类参数不做算术变异，改为重新采样一个随机类别。
        """
        genes = ind.genes.copy()
        for i in range(len(genes)):
            if random.random() < mutation_rate:
                pd = self._param_defs[i] if i < len(self._param_defs) else None
                if pd and pd.type == "categorical":
                    genes[i] = pd.sample()
                    continue
                u = random.random()
                if u <= 0.5:
                    delta = (2.0 * u) ** (1.0 / (max(self._pm_eta, 0) + 1)) - 1.0
                else:
                    delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (max(self._pm_eta, 0) + 1))
                if pd:
                    range_val = pd.high - pd.low
                    genes[i] = pd.clamp(genes[i] + delta * range_val * self._mutation_strength)
                else:
                    genes[i] = max(0, genes[i] + delta * 0.1)
        return Individual(genes=genes)

    # ── 多样性计算 ────────────────────────────────────────────

    def _compute_diversity(self, population: List[Individual]) -> float:
        """计算种群多样性（基因标准差的均值）"""
        if len(population) <= 1:
            return 0.0
        n = len(population[0].genes)
        gene_matrix = np.array([ind.genes for ind in population], dtype=float)
        gene_matrix = np.nan_to_num(gene_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        stds = np.std(gene_matrix, axis=0)
        # 归一化各参数标准差
        for i, pd in enumerate(self._param_defs[:n]):
            range_val = pd.high - pd.low if pd.high > pd.low else 1.0
            stds[i] = stds[i] / max(range_val, 1e-10)
        result = float(np.mean(stds))
        return result if math.isfinite(result) else 0.0

    # ── 收敛检测 ──────────────────────────────────────────────

    def _check_convergence(self, gen: int) -> Optional[ConvergenceReason]:
        if gen >= self._generations:
            return ConvergenceReason.MAX_GENERATIONS
        if self._best_individual and math.isfinite(self._best_individual.fitness) and self._best_individual.fitness >= self._target_fitness:
            return ConvergenceReason.TARGET_REACHED
        if len(self._fitness_history) >= self._convergence_patience:
            recent = [f for f in self._fitness_history[-self._convergence_patience:] if math.isfinite(f)]
            if len(recent) >= 2 and max(recent) - min(recent) < self._convergence_variance:
                return ConvergenceReason.LOW_VARIANCE
            if len(recent) > 5:
                improvements = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i - 1])
                if improvements <= 1:
                    return ConvergenceReason.NO_IMPROVEMENT
        return None

    def _compute_adaptive_mutation_rate(self, gen: int, diversity: float) -> float:
        """自适应变异率：多样性低时增加变异"""
        mutation_rate = self._mutation_rate if math.isfinite(self._mutation_rate) else 0.1
        min_mutation_rate = self._min_mutation_rate if math.isfinite(self._min_mutation_rate) else 0.01
        if not self._adaptive_mutation:
            return mutation_rate
        if not math.isfinite(diversity):
            diversity = 0.0
        # 基于代数的指数衰减
        gen_factor = mutation_rate * (1.0 - 0.3 * gen / max(self._generations, 1))
        # 基于多样性的反比调整
        div_threshold = self._diversity_threshold if math.isfinite(self._diversity_threshold) else 0.05
        if diversity < div_threshold * 0.5:
            div_factor = 2.5  # 极度低多样性
        elif diversity < div_threshold:
            div_factor = 1.5  # 低多样性
        else:
            div_factor = 1.0
        result = max(min_mutation_rate, gen_factor * div_factor)
        return result if math.isfinite(result) else min_mutation_rate

    def _should_restart(self, diversity: float, gen: int) -> bool:
        """判断是否需要重启种群"""
        if not self._restart_on_stagnation:
            return False
        if gen < self._generations * 0.3:
            return False
        patience_half = max(self._convergence_patience // 2, 5)
        if len(self._fitness_history) < patience_half:
            return False
        recent = [f for f in self._fitness_history[-patience_half:] if math.isfinite(f)]
        if len(recent) >= 2 and max(recent) - min(recent) < self._convergence_variance * 10:
            logger.info(f"Restarting population at gen {gen} due to stagnation")
            return True
        return False

    def _inject_random_individuals(self, population: List[Individual], count: int):
        """注入随机新个体保持多样性"""
        for i in range(min(count, len(population))):
            genes = [pd.sample() for pd in self._param_defs]
            population[i] = Individual(genes=genes)

    # ── Pareto 非支配排序 ─────────────────────────────────────

    def _compute_pareto_front(self, population: List[Individual]) -> List[Individual]:
        """计算 Pareto 前沿（多目标）"""
        if not population:
            return []
        if not any(ind.objectives for ind in population):
            return [max(population, key=lambda x: x.fitness)]
        # NSGA-II 风格非支配排序
        n = len(population)
        dominated_count = [0] * n
        dominates_list = [[] for _ in range(n)]
        fronts = [[]]
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(population[i].objectives, population[j].objectives):
                    dominates_list[i].append(j)
                    dominated_count[j] += 1
                elif self._dominates(population[j].objectives, population[i].objectives):
                    dominates_list[j].append(i)
                    dominated_count[i] += 1
            if dominated_count[i] == 0:
                fronts[0].append(i)
        # 只返回第一前沿
        return [population[i] for i in fronts[0]]

    @staticmethod
    def _dominates(obj_a: Dict[str, float], obj_b: Dict[str, float]) -> bool:
        """判断 A 是否 Pareto 支配 B"""
        keys = set(obj_a.keys()) & set(obj_b.keys())
        if not keys:
            return False
        at_least_better = False
        for k in keys:
            if obj_a[k] < obj_b[k]:
                return False
            if obj_a[k] > obj_b[k]:
                at_least_better = True
        return at_least_better

    # ── 主优化循环 ────────────────────────────────────────────

    async def optimize(self) -> GAOptimizationResult:
        """执行遗传算法优化"""
        if not self._param_defs:
            raise ValueError("No parameter definitions set")
        if not self._fitness_fn:
            raise ValueError("No fitness function set")
        if self._population_size < 1 or self._generations < 1:
            raise ValueError("population_size and generations must be positive")

        start_time = time.time()
        # 重置运行状态，避免多次调用累积污染
        self._eval_count = 0
        self._generation_stats = []
        self._fitness_history = []
        self._best_individual = None
        self._population = []

        # 确定性：设置随机种子
        if self._seed is not None:
            random.seed(self._seed)
            np.random.seed(self._seed)

        # 初始化种群
        self._population = self._initialize_population()
        await self._evaluate_population(self._population)
        self._best_individual = max(self._population, key=lambda x: x.fitness)

        for gen in range(1, self._generations + 1):
            # 排序种群
            sorted_pop = sorted(self._population, key=lambda x: x.fitness, reverse=True)

            # 精英保留
            elites = [deepcopy(sorted_pop[i]) for i in range(min(self._elite_count, len(sorted_pop)))]
            for e in elites:
                e.generation = gen

            # 构建新种群
            new_population = list(elites)

            while len(new_population) < self._population_size:
                p1 = self._select_parent(sorted_pop)
                p2 = self._select_parent(sorted_pop)
                c1, c2 = self._crossover(p1, p2)

                # 自适应变异率
                div = self._compute_diversity(self._population + new_population)
                mut_rate = self._compute_adaptive_mutation_rate(gen, div)

                c1 = self._polynomial_mutation(c1, mut_rate)
                c2 = self._polynomial_mutation(c2, mut_rate)
                c1.generation, c2.generation = gen, gen
                new_population.extend([c1, c2])

            self._population = new_population[:self._population_size]

            # 评估 + 替换
            await self._evaluate_population(self._population[self._elite_count:])
            current_best = max(self._population, key=lambda x: x.fitness)

            if current_best.fitness > self._best_individual.fitness:
                self._best_individual = deepcopy(current_best)

            self._fitness_history.append(self._best_individual.fitness)

            # 统计
            fitnesses = [ind.fitness for ind in self._population]
            finite_fitnesses = [f for f in fitnesses if math.isfinite(f)]
            if finite_fitnesses:
                avg_fitness = float(np.mean(finite_fitnesses))
                median_fitness = float(np.median(finite_fitnesses))
                worst_fitness = float(np.min(finite_fitnesses))
                std_fitness = float(np.std(finite_fitnesses))
            else:
                avg_fitness = median_fitness = worst_fitness = std_fitness = 0.0
            diversity = self._compute_diversity(self._population)
            stats = GenerationStats(
                generation=gen,
                best_fitness=self._best_individual.fitness,
                avg_fitness=avg_fitness,
                median_fitness=median_fitness,
                worst_fitness=worst_fitness,
                std_fitness=std_fitness,
                population_diversity=diversity,
                elapsed_seconds=time.time() - start_time,
                evaluations=len(self._population),
            )
            self._generation_stats.append(stats)

            # 收敛检测
            conv = self._check_convergence(gen)
            if conv:
                if conv != ConvergenceReason.LOW_VARIANCE or gen > self._convergence_patience:
                    logger.info(f"GA converged at gen {gen}: {conv.value}, best={self._best_individual.fitness:.4f}")
                    break

            # 停滞重启
            if self._should_restart(diversity, gen):
                self._inject_random_individuals(self._population, self._population_size // 2)
                await self._evaluate_population(self._population)

            # 定期日志
            if gen % 20 == 0 or gen == 1 or conv:
                logger.info(
                    f"GA gen {gen}: best={self._best_individual.fitness:.4f}, "
                    f"avg={stats.avg_fitness:.4f}, div={diversity:.4f}, time={stats.elapsed_seconds:.1f}s"
                )

        # 构建结果
        params = {pd.name: self._best_individual.genes[i]
                 for i, pd in enumerate(self._param_defs)}
        pareto = self._compute_pareto_front(self._population)

        return GAOptimizationResult(
            best_individual=self._best_individual,
            best_params=params,
            best_fitness=self._best_individual.fitness,
            generation_stats=self._generation_stats,
            convergence_reason=conv or ConvergenceReason.MAX_GENERATIONS,
            convergence_generation=len(self._generation_stats),
            total_evaluations=self._eval_count,
            total_time_seconds=time.time() - start_time,
            final_population=sorted(self._population, key=lambda x: x.fitness, reverse=True),
            fitness_history=self._fitness_history,
            pareto_front=pareto,
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "population_size": self._population_size,
            "generations": self._generations,
            "parameters": len(self._param_defs),
            "evaluations": self._eval_count,
            "best_fitness": self._best_individual.fitness if self._best_individual else None,
            "selection": self._selection_method.value,
            "crossover": self._crossover_method.value,
        }
