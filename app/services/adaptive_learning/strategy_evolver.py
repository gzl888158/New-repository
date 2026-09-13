"""策略进化器：通过遗传/进化算法在多策略群体中演化并迁移最优策略。"""
from enum import Enum
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Awaitable, Tuple
from loguru import logger

from app.services.enterprise import EnterpriseServiceMixin
import statistics
import random
import math
import asyncio
import copy


class EvolutionStrategy(Enum):
    RANDOM = "random"
    GRADIENT = "gradient"
    GENETIC = "genetic"
    SIMULATED_ANNEALING = "simulated_annealing"
    CMA_ES = "cma_es"


class EvolutionStatus(Enum):
    IDLE = "idle"
    EVOLVING = "evolving"
    EVALUATING = "evaluating"
    PAUSED = "paused"


class MigrationTopology(Enum):
    RING = "ring"
    FULLY_CONNECTED = "fully_connected"
    RANDOM = "random"


# ──────────────────────────────────────────────
#  StrategyEvolver
# ──────────────────────────────────────────────
class StrategyEvolver(EnterpriseServiceMixin):
    """Enhanced strategy evolver with elitism, adaptive mutation,
    diversity maintenance, convergence detection, and CMA-ES."""

    def __init__(self, config: Dict[str, Any]):
        self._config = config

        # ── strategy_evolver 专属配置段（统一读取，避免污染顶层配置）──
        se = config.get("strategy_evolver", {})

        self._strategy = EvolutionStrategy(
            se.get("evolution_strategy", "gradient")
        )
        self._status = EvolutionStatus.IDLE
        self._population_size = se.get("population_size", 10)
        self._mutation_rate = se.get("mutation_rate", 0.1)
        self._crossover_rate = se.get("crossover_rate", 0.5)
        self._temperature = se.get("initial_temperature", 1.0)
        self._cooling_rate = se.get("cooling_rate", 0.99)
        self._generations = se.get("generations", 50)

        # Elitism
        self._elite_count = se.get("elite_count", 2)
        self._elite_archive: Dict[str, List[Dict[str, Any]]] = {}

        # Adaptive mutation
        self._adaptive_mutation = se.get("adaptive_mutation", True)
        self._min_mutation_rate = se.get("min_mutation_rate", 0.01)
        self._max_mutation_rate = se.get("max_mutation_rate", 0.5)
        self._base_mutation_rate = self._mutation_rate

        # Stagnation detection
        self._stagnation_generations = se.get("stagnation_generations", 5)
        self._stagnation_threshold = se.get("stagnation_threshold", 0.001)

        # Convergence detection
        self._convergence_patience = se.get("convergence_patience", 10)
        self._convergence_min_improvement = se.get(
            "convergence_min_improvement", 0.0001
        )
        self._convergence_variance_threshold = se.get(
            "convergence_variance_threshold", 0.001
        )

        # CMA-ES
        cma = se.get("cma_es", {})
        self._cma_population_factor = cma.get("population_factor", 4)
        self._cma_sigma = cma.get("sigma", 0.3)

        # Internal state
        self._strategies: Dict[str, Dict[str, Any]] = {}
        self._evolution_history: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._diversity_history: List[float] = []
        self._convergence_info: Dict[str, Any] = {}
        self._current_population: List[Dict[str, float]] = []
        self._current_fitnesses: List[float] = []

    # ── Lifecycle ──────────────────────────────────

    async def start(self):
        async with self._lock:
            if self._status == EvolutionStatus.PAUSED:
                self._status = EvolutionStatus.IDLE
        logger.info("Strategy evolver started")

    async def stop(self):
        async with self._lock:
            self._status = EvolutionStatus.PAUSED
        logger.info("Strategy evolver stopped")

    # ── Registration ───────────────────────────────

    def register_strategy(
        self,
        strategy_name: str,
        parameters: Dict[str, Any],
        fitness_function: Optional[Callable[..., Any]] = None,
    ):
        # Initialize with a random valid starting point (not the parameter definition dict)
        initial_params = self._random_params(parameters)
        self._strategies[strategy_name] = {
            "parameters": parameters,
            "fitness_function": fitness_function,
            "best_fitness": float("-inf"),
            "best_parameters": initial_params,
            "history": [],
        }
        # 初始化精英档案
        if strategy_name not in self._elite_archive:
            self._elite_archive[strategy_name] = []
        logger.info(f"Registered strategy for evolution: {strategy_name}")

    # ── Evolve batch ───────────────────────────────

    async def evolve_strategies(self) -> List[Dict[str, Any]]:
        results = []
        for strategy_name in list(self._strategies.keys()):
            try:
                result = await self.evolve(strategy_name)
                if result.get("status") not in ("paused", "unknown_strategy"):
                    results.append(result)
            except Exception as e:
                self._handle_exception(
                    e, {"strategy": strategy_name}, module="StrategyEvolver",
                    function="evolve_strategies", severity="low", category="evolution",
                )
        return results

    # ── Main evolve dispatch ───────────────────────

    async def evolve(self, strategy_name: str) -> Dict[str, Any]:
        async with self._lock:
            if self._status == EvolutionStatus.PAUSED:
                return {"status": "paused"}
            if strategy_name not in self._strategies:
                return {"error": f"Strategy {strategy_name} not registered"}
            self._status = EvolutionStatus.EVOLVING

        strategy_data = self._strategies[strategy_name]
        # Reset per-run diversity & convergence tracking
        self._diversity_history = []
        self._convergence_info = {}
        self._stagnation_counter = 0
        self._last_best_fitness = float("-inf")

        try:
            if self._strategy == EvolutionStrategy.RANDOM:
                result = await self._random_search(strategy_name)
            elif self._strategy == EvolutionStrategy.GRADIENT:
                result = await self._gradient_descent(strategy_name)
            elif self._strategy == EvolutionStrategy.GENETIC:
                result = await self._genetic_algorithm(strategy_name)
            elif self._strategy == EvolutionStrategy.SIMULATED_ANNEALING:
                result = await self._simulated_annealing(strategy_name)
            elif self._strategy == EvolutionStrategy.CMA_ES:
                result = await self._cma_es(strategy_name)
            else:
                result = {"status": "unknown_strategy"}

            self._evolution_history.append(
                {
                    "strategy_name": strategy_name,
                    "timestamp": datetime.now(),
                    "result": result,
                }
            )
            async with self._lock:
                self._status = EvolutionStatus.IDLE
            return result

        except Exception as e:
            self._handle_exception(
                e, {"strategy": strategy_name}, module="StrategyEvolver",
                function="evolve", severity="high", category="evolution",
            )
            self._notify_sync(
                f"策略进化失败: {strategy_name}",
                f"Strategy evolution failed for {strategy_name}: {e}",
                priority="warning",
                category="adaptive_learning",
            )
            async with self._lock:
                self._status = EvolutionStatus.IDLE
            return {"status": "failed", "error": str(e)}

    # ── Random search ──────────────────────────────

    async def _random_search(self, strategy_name: str) -> Dict[str, Any]:
        strategy_data = self._strategies[strategy_name]
        parameters = strategy_data["parameters"]
        best_fitness = strategy_data["best_fitness"]
        best_params = strategy_data["best_parameters"].copy()
        patience_counter = 0

        for gen in range(self._generations):
            candidate_params = self._random_params(parameters)
            fitness = await self._evaluate_fitness(strategy_name, candidate_params)

            if fitness > best_fitness:
                best_fitness = fitness
                best_params = candidate_params.copy()
                patience_counter = 0
            else:
                patience_counter += 1

            self._update_elite_archive(strategy_name, candidate_params, fitness)

            if await self._check_convergence(strategy_name, best_fitness, patience_counter):
                self._convergence_info = {
                    "reason": "no_improvement",
                    "generation": gen + 1,
                    "patience": patience_counter,
                }
                logger.info(f"Converged at generation {gen + 1}: no improvement")
                break

        strategy_data["best_fitness"] = best_fitness
        strategy_data["best_parameters"] = best_params
        strategy_data["history"].append({"fitness": best_fitness, "params": best_params})

        return {
            "status": "success",
            "strategy": "random_search",
            "best_fitness": best_fitness,
            "best_parameters": best_params,
            "evaluations": self._generations,
        }

    # ── Gradient descent ───────────────────────────

    async def _gradient_descent(self, strategy_name: str) -> Dict[str, Any]:
        strategy_data = self._strategies[strategy_name]
        parameters = strategy_data["parameters"]
        current_params = strategy_data["best_parameters"].copy()
        best_fitness = strategy_data["best_fitness"]

        for _ in range(self._generations):
            for param_name, param_info in parameters.items():
                step = param_info.get("step", 0.01)
                original_value = current_params[param_name]

                current_params[param_name] = original_value + step
                fitness_up = await self._evaluate_fitness(strategy_name, current_params)

                current_params[param_name] = original_value - step
                fitness_down = await self._evaluate_fitness(strategy_name, current_params)

                current_params[param_name] = original_value

                if fitness_up > best_fitness:
                    best_fitness = fitness_up
                    current_params[param_name] = original_value + step
                elif fitness_down > best_fitness:
                    best_fitness = fitness_down
                    current_params[param_name] = original_value - step

            self._update_elite_archive(strategy_name, current_params, best_fitness)

        strategy_data["best_fitness"] = best_fitness
        strategy_data["best_parameters"] = current_params.copy()
        strategy_data["history"].append({"fitness": best_fitness, "params": current_params})

        return {
            "status": "success",
            "strategy": "gradient_descent",
            "best_fitness": best_fitness,
            "best_parameters": current_params,
            "evaluations": self._generations * len(parameters) * 2,
        }

    # ── Genetic algorithm ──────────────────────────

    async def _genetic_algorithm(self, strategy_name: str) -> Dict[str, Any]:
        strategy_data = self._strategies[strategy_name]
        parameters = strategy_data["parameters"]
        n_params = len(parameters)

        population = [
            self._random_params(parameters) for _ in range(self._population_size)
        ]

        # Use adaptive mutation internally for genetic
        current_mutation_rate = self._mutation_rate

        for gen in range(self._generations):
            fitness_scores = [
                await self._evaluate_fitness(strategy_name, p) for p in population
            ]

            # Sort
            sorted_indices = sorted(
                range(len(fitness_scores)),
                key=lambda i: fitness_scores[i],
                reverse=True,
            )
            population = [population[i] for i in sorted_indices]
            fitness_scores = [fitness_scores[i] for i in sorted_indices]

            best_fitness = fitness_scores[0]
            best_params = population[0].copy()

            # Update elite archive
            for p, f in zip(population, fitness_scores):
                self._update_elite_archive(strategy_name, p, f)

            # ── Convergence check ──
            improvement = best_fitness - self._last_best_fitness
            if improvement < self._convergence_min_improvement:
                self._stagnation_counter += 1
            else:
                self._stagnation_counter = 0
            self._last_best_fitness = best_fitness

            if await self._check_convergence(
                strategy_name, best_fitness, self._stagnation_counter
            ):
                self._convergence_info = {
                    "reason": "no_improvement",
                    "generation": gen + 1,
                    "patience": self._stagnation_counter,
                }
                logger.info(f"GA converged at generation {gen + 1}")
                break

            # ── Adaptive mutation rate ──
            if self._adaptive_mutation:
                diversity = self._compute_diversity(population, parameters)
                self._diversity_history.append(diversity)
                current_mutation_rate = self._adjust_mutation_rate(diversity)
                logger.debug(
                    f"Gen {gen}: diversity={diversity:.4f}, "
                    f"mutation_rate={current_mutation_rate:.4f}"
                )

            # ── Elitism ──
            new_population: List[Dict[str, float]] = []

            # Keep elite individuals
            for idx in range(min(self._elite_count, len(population))):
                new_population.append(copy.deepcopy(population[idx]))

            while len(new_population) < self._population_size:
                # Tournament selection with crowding
                parent1 = self._tournament_select_crowded(
                    population, fitness_scores, parameters
                )
                parent2 = self._tournament_select_crowded(
                    population, fitness_scores, parameters
                )

                child = self._crossover(parent1, parent2)
                child = self._mutate_with_rate(child, parameters, current_mutation_rate)
                new_population.append(child)

            population = new_population
            self._current_population = population
            self._current_fitnesses = fitness_scores

            # ── Variance convergence ──
            if n_params > 1:
                param_vars = self._parameter_variance(population, parameters)
                if all(v < self._convergence_variance_threshold for v in param_vars):
                    self._convergence_info = {
                        "reason": "low_variance",
                        "generation": gen + 1,
                    }
                    logger.info(
                        f"GA converged at gen {gen + 1}: population variance low"
                    )
                    break

        strategy_data["best_fitness"] = best_fitness
        strategy_data["best_parameters"] = best_params
        strategy_data["history"].append({"fitness": best_fitness, "params": best_params})

        return {
            "status": "success",
            "strategy": "genetic_algorithm",
            "best_fitness": best_fitness,
            "best_parameters": best_params,
            "generations": self._generations,
            "population_size": self._population_size,
            "final_mutation_rate": current_mutation_rate if self._adaptive_mutation else self._mutation_rate,
            "convergence": self._convergence_info if self._convergence_info else None,
        }

    # ── Simulated annealing ────────────────────────

    async def _simulated_annealing(self, strategy_name: str) -> Dict[str, Any]:
        strategy_data = self._strategies[strategy_name]
        parameters = strategy_data["parameters"]
        current_params = strategy_data["best_parameters"].copy()
        current_fitness = await self._evaluate_fitness(strategy_name, current_params)
        best_fitness = current_fitness
        best_params = current_params.copy()
        temperature = self._temperature
        patience_counter = 0

        for gen in range(self._generations):
            candidate_params = self._mutate(current_params.copy(), parameters)
            candidate_fitness = await self._evaluate_fitness(
                strategy_name, candidate_params
            )
            delta = candidate_fitness - current_fitness

            if delta > 0 or random.random() < self._acceptance_probability(
                delta, temperature
            ):
                current_params = candidate_params
                current_fitness = candidate_fitness

                if current_fitness > best_fitness:
                    best_fitness = current_fitness
                    best_params = current_params.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                patience_counter += 1

            self._update_elite_archive(strategy_name, candidate_params, candidate_fitness)
            temperature *= self._cooling_rate

            if await self._check_convergence(strategy_name, best_fitness, patience_counter):
                self._convergence_info = {
                    "reason": "no_improvement",
                    "generation": gen + 1,
                    "patience": patience_counter,
                }
                break

        strategy_data["best_fitness"] = best_fitness
        strategy_data["best_parameters"] = best_params
        strategy_data["history"].append({"fitness": best_fitness, "params": best_params})

        return {
            "status": "success",
            "strategy": "simulated_annealing",
            "best_fitness": best_fitness,
            "best_parameters": best_params,
            "evaluations": self._generations,
            "final_temperature": temperature,
        }

    # ── CMA-ES ─────────────────────────────────────

    async def _cma_es(self, strategy_name: str) -> Dict[str, Any]:
        """Basic CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

        Maintains a multivariate normal distribution over parameter space.
        Samples λ candidates, evaluates, selects μ best to update mean
        and covariance.
        """
        strategy_data = self._strategies[strategy_name]
        parameters = strategy_data["parameters"]
        param_names = list(parameters.keys())
        n = len(param_names)  # dimensionality

        if n == 0:
            return {"status": "failed", "error": "No parameters to optimize"}

        # CMA-ES hyperparameters
        lamb = self._cma_population_factor * n  # offspring count
        if lamb < 4:
            lamb = 4
        mu = lamb // 2  # parents count
        weights = [math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)]
        weights_sum = sum(weights)
        weights = [w / weights_sum for w in weights]
        mu_eff = 1.0 / sum(w * w for w in weights)

        sigma = self._cma_sigma

        # Initialize mean using current best or centre of parameter ranges
        mean = []
        bounds_low = []
        bounds_high = []
        for name in param_names:
            info = parameters[name]
            lo, hi = info["min"], info["max"]
            bounds_low.append(lo)
            bounds_high.append(hi)
            # Start at center of range
            mean.append((lo + hi) / 2.0)

        # Covariance matrix: initialize as identity scaled by sigma²
        C = [[0.0] * n for _ in range(n)]
        for i in range(n):
            C[i][i] = 1.0

        # Evolution path accumulators
        pc = [0.0] * n  # evolution path for C
        ps = [0.0] * n  # evolution path for sigma

        # Strategy parameters
        cc = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        cs = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        c1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
        cmu = min(
            1.0 - c1,
            2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff),
        )
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + cs

        best_fitness_all = float("-inf")
        best_params_all = {}
        patience_counter = 0
        last_best = float("-inf")

        # Cholesky decomposition helper
        def cholesky(A: List[List[float]]) -> List[List[float]]:
            n_a = len(A)
            L = [[0.0] * n_a for _ in range(n_a)]
            for i in range(n_a):
                for j in range(i + 1):
                    s = sum(L[i][k] * L[j][k] for k in range(j))
                    if i == j:
                        val = A[i][i] - s
                        if val <= 0:
                            val = 1e-10
                        L[i][j] = math.sqrt(val)
                    else:
                        denom = L[j][j]
                        if denom == 0:
                            denom = 1e-10
                        L[i][j] = (A[i][j] - s) / denom
            return L

        def clamp(val: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, val))

        for gen in range(self._generations):
            # Sample λ candidates from N(mean, sigma² * C)
            L_mat = cholesky(C)  # Cholesky decomposition of C

            candidates_params: List[Dict[str, float]] = []
            candidates_raw: List[List[float]] = []  # raw N(0,1) vectors

            for _ in range(lamb):
                # Sample z ~ N(0, I)
                z = [random.gauss(0, 1) for _ in range(n)]
                # Transform: x = mean + sigma * L * z
                x = list(mean)
                for i in range(n):
                    for j in range(n):
                        x[i] += sigma * L_mat[i][j] * z[j]
                # Clamp to bounds
                candidate = {}
                for idx, name in enumerate(param_names):
                    candidate[name] = clamp(x[idx], bounds_low[idx], bounds_high[idx])
                candidates_params.append(candidate)
                candidates_raw.append(z)

            # Evaluate
            fitnesses = [
                await self._evaluate_fitness(strategy_name, cp)
                for cp in candidates_params
            ]

            # Sort by fitness descending
            sorted_idx = sorted(
                range(lamb), key=lambda i: fitnesses[i], reverse=True
            )
            sorted_fitnesses = [fitnesses[i] for i in sorted_idx]
            sorted_candidates = [candidates_params[i] for i in sorted_idx]
            sorted_z = [candidates_raw[i] for i in sorted_idx]

            gen_best_fitness = sorted_fitnesses[0]
            gen_best_params = sorted_candidates[0].copy()

            if gen_best_fitness > best_fitness_all:
                best_fitness_all = gen_best_fitness
                best_params_all = gen_best_params
                patience_counter = 0
            else:
                patience_counter += 1

            self._update_elite_archive(strategy_name, gen_best_params, gen_best_fitness)

            # ── Update mean: weighted average of μ best ──
            new_mean = [0.0] * n
            for i in range(mu):
                w = weights[i]
                candidate = sorted_candidates[i]
                for j, name in enumerate(param_names):
                    new_mean[j] += w * candidate[name]
            mean = new_mean

            # ── Update evolution paths ──
            # z_w = (mean_old - mean) / sigma  (approximation)
            y_w = [
                (mean[i] - (sum(weights[k] * sorted_candidates[k][param_names[i]] for k in range(mu))))
                for i in range(n)
            ]
            # Actually simpler: compute weighted z
            z_w = [0.0] * n
            for i in range(mu):
                w = weights[i]
                for j in range(n):
                    z_w[j] += w * sorted_z[i][j]

            # ps update
            for i in range(n):
                ps[i] = (1.0 - cs) * ps[i] + math.sqrt(cs * (2.0 - cs) * mu_eff) * z_w[i]

            # sigma update
            ps_norm = math.sqrt(sum(v * v for v in ps))
            expected_norm = math.sqrt(n) * (
                1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n)
            )
            sigma *= math.exp((cs / damps) * (ps_norm / expected_norm - 1.0))
            sigma = max(1e-12, min(sigma, 10.0))

            # pc update
            for i in range(n):
                pc[i] = (1.0 - cc) * pc[i] + math.sqrt(cc * (2.0 - cc) * mu_eff) * z_w[i]

            # C update
            # C = (1 - c1 - cmu) * C + c1 * (pc * pc^T) + cmu * weighted sum(z_i * z_i^T)
            for i in range(n):
                for j in range(n):
                    C[i][j] *= (1.0 - c1 - cmu)

            # Rank-1 update
            for i in range(n):
                for j in range(n):
                    C[i][j] += c1 * pc[i] * pc[j]

            # Rank-mu update
            for k in range(mu):
                w = weights[k]
                z_k = sorted_z[k]
                for i in range(n):
                    for j in range(n):
                        C[i][j] += cmu * w * z_k[i] * z_k[j]

            if await self._check_convergence(strategy_name, best_fitness_all, patience_counter):
                self._convergence_info = {
                    "reason": "no_improvement",
                    "generation": gen + 1,
                    "patience": patience_counter,
                }
                logger.info(f"CMA-ES converged at generation {gen + 1}")
                break

        strategy_data["best_fitness"] = best_fitness_all
        strategy_data["best_parameters"] = best_params_all
        strategy_data["history"].append(
            {"fitness": best_fitness_all, "params": best_params_all}
        )

        return {
            "status": "success",
            "strategy": "cma_es",
            "best_fitness": best_fitness_all,
            "best_parameters": best_params_all,
            "generations": self._generations,
            "final_sigma": sigma,
        }

    # ── Elitism ────────────────────────────────────

    def _update_elite_archive(
        self,
        strategy_name: str,
        params: Dict[str, float],
        fitness: float,
    ):
        """Maintain elite archive of top-K best individuals ever seen."""
        if strategy_name not in self._elite_archive:
            self._elite_archive[strategy_name] = []

        archive = self._elite_archive[strategy_name]

        # Check if this is a new best
        is_duplicate = any(
            self._params_equal(params, entry["params"]) for entry in archive
        )
        if is_duplicate:
            return

        archive.append(
            {
                "params": copy.deepcopy(params),
                "fitness": fitness,
                "timestamp": datetime.now().isoformat(),
            }
        )
        # Keep only top elite_count
        archive.sort(key=lambda x: x["fitness"], reverse=True)
        if len(archive) > self._elite_count:
            archive[:] = archive[: self._elite_count]

    def get_elite_archive(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return the elite archive for all strategies."""
        return copy.deepcopy(self._elite_archive)

    @staticmethod
    def _params_equal(a: Dict[str, float], b: Dict[str, float]) -> bool:
        if set(a.keys()) != set(b.keys()):
            return False
        return all(
            abs(a[k] - b[k]) < 1e-9 for k in a
        )

    # ── Adaptive mutation ──────────────────────────

    def _adjust_mutation_rate(self, diversity: float) -> float:
        """Dynamically adjust mutation rate based on population diversity.

        Low diversity → high mutation (explore more).
        High diversity → low mutation (exploit good solutions).
        """
        # diversity normally in [0, ~1] range, clamp
        div = max(0.0, min(1.0, diversity))
        # When diversity=0 → max rate, when diversity=1 → base rate
        rate = self._max_mutation_rate - div * (
            self._max_mutation_rate - self._base_mutation_rate
        )
        # Clamp to safe bounds
        return max(self._min_mutation_rate, min(self._max_mutation_rate, rate))

    def _mutate_with_rate(
        self,
        params: Dict[str, float],
        parameters_info: Dict[str, Any],
        mutation_rate: float,
    ) -> Dict[str, float]:
        """Mutate using a specific mutation rate (for adaptive mutation)."""
        for name, info in parameters_info.items():
            if random.random() < mutation_rate:
                step = info.get("step", 0.01)
                mutation = (random.random() - 0.5) * 2 * step
                params[name] = max(
                    info["min"], min(info["max"], params[name] + mutation)
                )
        return params

    # ── Diversity / Crowding distance ──────────────

    def _compute_diversity(
        self,
        population: List[Dict[str, float]],
        parameters: Dict[str, Any],
    ) -> float:
        """Average pairwise Euclidean distance in parameter space (normalized)."""
        if len(population) < 2:
            return 0.0

        param_names = list(parameters.keys())
        if not param_names:
            return 0.0

        distances: List[float] = []
        # Normalization factors per parameter
        ranges = {
            name: parameters[name]["max"] - parameters[name]["min"]
            for name in param_names
        }

        for i in range(min(len(population), 20)):  # sample at most 20 for efficiency
            for j in range(i + 1, min(len(population), 20)):
                sq_sum = 0.0
                for name in param_names:
                    rng = ranges[name]
                    if rng == 0:
                        rng = 1.0
                    diff = (population[i][name] - population[j][name]) / rng
                    sq_sum += diff * diff
                distances.append(math.sqrt(sq_sum / len(param_names)))

        if not distances:
            return 0.0
        return statistics.mean(distances)

    def _compute_crowding_distances(
        self,
        population: List[Dict[str, float]],
        fitnesses: List[float],
        parameters: Dict[str, Any],
    ) -> List[float]:
        """Compute crowding distance for each individual.

        Used for diversity preservation in multi-objective / niche scenarios.
        """
        n = len(population)
        if n <= 2:
            return [float("inf")] * n

        param_names = list(parameters.keys())
        distances = [0.0] * n

        # For each objective dimension (here: fitness + each param)
        # We add crowding based on parameter spread
        # Main objective: fitness
        sorted_idx = sorted(range(n), key=lambda i: fitnesses[i])
        f_min = fitnesses[sorted_idx[0]]
        f_max = fitnesses[sorted_idx[-1]]
        if f_max - f_min > 1e-12:
            distances[sorted_idx[0]] = float("inf")
            distances[sorted_idx[-1]] = float("inf")
            for k in range(1, n - 1):
                distances[sorted_idx[k]] += (
                    fitnesses[sorted_idx[k + 1]] - fitnesses[sorted_idx[k - 1]]
                ) / (f_max - f_min)

        # Secondary: parameter dimensions
        for name in param_names:
            sorted_idx = sorted(range(n), key=lambda i: population[i][name])
            p_min = population[sorted_idx[0]][name]
            p_max = population[sorted_idx[-1]][name]
            if p_max - p_min > 1e-12:
                distances[sorted_idx[0]] += float("inf")
                distances[sorted_idx[-1]] += float("inf")
                for k in range(1, n - 1):
                    distances[sorted_idx[k]] += (
                        population[sorted_idx[k + 1]][name]
                        - population[sorted_idx[k - 1]][name]
                    ) / (p_max - p_min)

        return distances

    def _tournament_select_crowded(
        self,
        population: List[Dict[str, float]],
        fitnesses: List[float],
        parameters: Dict[str, Any],
        tournament_size: int = 3,
    ) -> Dict[str, float]:
        """Tournament selection using crowding comparison operator.

        Prefers higher fitness; when fitness is close, prefers higher crowding distance.
        """
        indices = random.sample(range(len(population)), min(tournament_size, len(population)))

        if not indices:
            return population[0]

        # Use crowding distances from last computation or compute fresh
        crowding = self._compute_crowding_distances(
            population, fitnesses, parameters
        )

        best_idx = indices[0]
        for idx in indices[1:]:
            if fitnesses[idx] > fitnesses[best_idx]:
                best_idx = idx
            elif abs(fitnesses[idx] - fitnesses[best_idx]) < 1e-6:
                if (
                    len(crowding) > idx
                    and len(crowding) > best_idx
                    and crowding[idx] > crowding[best_idx]
                ):
                    best_idx = idx

        return population[best_idx].copy()

    def _parameter_variance(
        self,
        population: List[Dict[str, float]],
        parameters: Dict[str, Any],
    ) -> List[float]:
        """Normalized per-parameter variance across the population."""
        if len(population) < 2:
            return [0.0] * len(parameters)

        variances = []
        for name, info in parameters.items():
            vals = [p[name] for p in population]
            rng = info["max"] - info["min"]
            if rng == 0:
                variances.append(0.0)
            else:
                variances.append(statistics.variance(vals) / (rng * rng))
        return variances

    def compute_diversity(self, strategy_name: str) -> float:
        """Public method: compute current population diversity."""
        if (
            not self._current_population
            or strategy_name not in self._strategies
        ):
            return 0.0
        parameters = self._strategies[strategy_name]["parameters"]
        return self._compute_diversity(self._current_population, parameters)

    def get_diversity(self) -> Dict[str, Any]:
        """Return diversity metrics."""
        return {
            "diversity_history": list(self._diversity_history),
            "current_diversity": (
                self._diversity_history[-1] if self._diversity_history else 0.0
            ),
        }

    def get_population_health(self) -> Dict[str, Any]:
        """Assess population health: diversity, stagnation, variance."""
        result: Dict[str, Any] = {
            "status": "unknown",
            "diversity": 0.0,
            "stagnant": False,
            "converged": False,
        }

        if self._diversity_history:
            result["diversity"] = self._diversity_history[-1]
            result["trend"] = "stable"
            if len(self._diversity_history) >= 3:
                recent = self._diversity_history[-3:]
                if recent[-1] < recent[0] * 0.7:
                    result["trend"] = "declining"
                elif recent[-1] > recent[0] * 1.3:
                    result["trend"] = "increasing"

        if self._convergence_info:
            result["converged"] = True
            result["convergence_reason"] = self._convergence_info.get("reason")

        if (
            self._diversity_history
            and self._diversity_history[-1] < 0.05
            and self._stagnation_counter >= self._stagnation_generations
        ):
            result["stagnant"] = True

        if result["converged"]:
            result["status"] = "converged"
        elif result["stagnant"]:
            result["status"] = "stagnant"
        elif result["diversity"] > 0.1:
            result["status"] = "healthy"
        else:
            result["status"] = "precarious"

        return result

    # ── Convergence detection ──────────────────────

    async def _check_convergence(
        self,
        strategy_name: str,
        current_best: float,
        no_improvement_count: int,
    ) -> bool:
        """Check if evolution has converged."""
        if no_improvement_count >= self._convergence_patience:
            return True
        return False

    def get_convergence(self) -> Dict[str, Any]:
        """Return convergence status and report."""
        report = {
            "converged": bool(self._convergence_info),
            "reason": self._convergence_info.get("reason") if self._convergence_info else None,
            "generation": self._convergence_info.get("generation") if self._convergence_info else None,
            "patience": self._convergence_info.get("patience", 0),
            "current_stagnation": self._stagnation_counter,
            "stagnation_threshold": self._convergence_patience,
            "last_best_fitness": self._last_best_fitness,
            "population_variance": self._parameter_variance(
                self._current_population,
                next(iter(self._strategies.values()), {"parameters": {}}).get("parameters", {}),
            )
            if self._current_population
            else [],
        }
        return report

    # ── Utility methods ────────────────────────────

    def _random_params(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        return {
            name: random.uniform(info["min"], info["max"])
            for name, info in parameters.items()
        }

    def _crossover(
        self, parent1: Dict[str, float], parent2: Dict[str, float]
    ) -> Dict[str, float]:
        child = {}
        for name in parent1:
            if random.random() < self._crossover_rate:
                child[name] = parent1[name]
            else:
                child[name] = parent2[name]
        return child

    def _mutate(
        self, params: Dict[str, float], parameters_info: Dict[str, Any]
    ) -> Dict[str, float]:
        for name, info in parameters_info.items():
            if random.random() < self._mutation_rate:
                step = info.get("step", 0.01)
                mutation = (random.random() - 0.5) * 2 * step
                params[name] = max(
                    info["min"], min(info["max"], params[name] + mutation)
                )
        return params

    def _acceptance_probability(self, delta: float, temperature: float) -> float:
        if delta >= 0:
            return 1.0
        return math.exp(delta / max(temperature, 1e-12))

    async def _evaluate_fitness(
        self, strategy_name: str, params: Dict[str, float]
    ) -> float:
        strategy_data = self._strategies[strategy_name]
        fitness_function = strategy_data.get("fitness_function")

        if fitness_function:
            try:
                result = fitness_function(params)
                if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
                    return await result
                return result
            except Exception as e:
                self._handle_exception(
                    e, module="StrategyEvolver", function="_evaluate_fitness",
                    severity="medium", category="evolution",
                )
                return float("-inf")

        return sum(params.values())

    # ── Public API ─────────────────────────────────

    def get_strategy(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        return self._strategies.get(strategy_name)

    def get_strategies(self) -> Dict[str, Any]:
        return self._strategies

    def get_status(self) -> str:
        return self._status.value

    def get_evolution_summary(self) -> Dict[str, Any]:
        summary = {
            "status": self._status.value,
            "evolution_strategy": self._strategy.value,
            "population_size": self._population_size,
            "mutation_rate": self._mutation_rate,
            "crossover_rate": self._crossover_rate,
            "generations": self._generations,
            "adaptive_mutation": self._adaptive_mutation,
            "convergence": self.get_convergence(),
            "diversity": self.get_diversity(),
            "strategies": {},
        }

        for name, data in self._strategies.items():
            history = data.get("history", [])
            if history:
                recent_fitness = [h["fitness"] for h in history[-10:]]
                avg_fitness = statistics.mean(recent_fitness)
                best_fitness = max(recent_fitness)
            else:
                avg_fitness = 0
                best_fitness = data["best_fitness"]

            summary["strategies"][name] = {
                "best_fitness": best_fitness,
                "avg_recent_fitness": avg_fitness,
                "best_parameters": data["best_parameters"],
                "history_length": len(history),
                "elite_count": len(self._elite_archive.get(name, [])),
            }

        return summary

    def pause(self):
        self._status = EvolutionStatus.PAUSED
        logger.info("Strategy evolver paused")

    def resume(self):
        self._status = EvolutionStatus.IDLE
        logger.info("Strategy evolver resumed")

    def reset(self):
        for data in self._strategies.values():
            data["best_fitness"] = float("-inf")
            data["best_parameters"] = self._random_params(data["parameters"])
            data["history"] = []
        self._evolution_history = []
        self._elite_archive = {}
        self._diversity_history = []
        self._convergence_info = {}
        self._current_population = []
        self._current_fitnesses = []
        self._stagnation_counter = 0
        self._last_best_fitness = float("-inf")
        logger.info("Strategy evolver reset")


# ──────────────────────────────────────────────
#  MultiPopulationEvolver
# ──────────────────────────────────────────────
class MultiPopulationEvolver:
    """Run multiple isolated sub-populations with periodic migration.

    Each sub-population uses its own StrategyEvolver with potentially
    different mutation/crossover rates and strategies.

    Migration topologies:
    - RING: each sub-pop sends its best to the next in a ring
    - FULLY_CONNECTED: each sub-pop sends to all others
    - RANDOM: each sub-pop sends to a random other

    Migration replaces the worst individuals in the receiving population.
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        se = config.get("strategy_evolver", {})

        self._num_populations = se.get("num_populations", 3)
        self._migration_interval = se.get("migration_interval", 5)
        self._migration_rate = se.get("migration_rate", 0.2)
        topo = se.get("migration_topology", "ring")
        try:
            self._migration_topology = MigrationTopology(topo)
        except ValueError:
            self._migration_topology = MigrationTopology.RING

        # Build sub-population configs
        sub_pop_configs = se.get("sub_populations", [])
        if not sub_pop_configs:
            sub_pop_configs = [{} for _ in range(self._num_populations)]

        self._evolvers: List[StrategyEvolver] = []
        for i in range(self._num_populations):
            sub_cfg = sub_pop_configs[i] if i < len(sub_pop_configs) else {}
            pop_config = copy.deepcopy(config)
            # Overlay sub-pop specific settings
            for key in (
                "mutation_rate",
                "crossover_rate",
                "population_size",
                "evolution_strategy",
                "generations",
            ):
                if key in sub_cfg:
                    pop_config[key] = sub_cfg[key]
            self._evolvers.append(StrategyEvolver(pop_config))

        self._lock = asyncio.Lock()
        self._merged_elite: Dict[str, List[Dict[str, Any]]] = {}
        logger.info(
            f"MultiPopulationEvolver initialized: {self._num_populations} "
            f"populations, topology={self._migration_topology.value}"
        )

    def register_strategy(
        self,
        strategy_name: str,
        parameters: Dict[str, Any],
        fitness_function: Optional[Callable[..., Any]] = None,
    ):
        """Register the same strategy across all sub-populations."""
        for evolver in self._evolvers:
            evolver.register_strategy(strategy_name, parameters, fitness_function)

    def get_evolver(self, index: int) -> Optional[StrategyEvolver]:
        """Access a specific sub-population evolver."""
        if 0 <= index < len(self._evolvers):
            return self._evolvers[index]
        return None

    def get_all_evolvers(self) -> List[StrategyEvolver]:
        return list(self._evolvers)

    async def run_multi_population(
        self,
        strategy_name: str,
        total_generations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run multi-population evolution with periodic migration.

        Args:
            strategy_name: The strategy to evolve.
            total_generations: Override generations per sub-population.

        Returns:
            Merged results from all sub-populations.
        """
        if total_generations is None:
            total_generations = self._evolvers[0]._generations if self._evolvers else 50

        # All sub-populations start from the same registered strategy
        for evolver in self._evolvers:
            if strategy_name not in evolver._strategies:
                return {
                    "status": "error",
                    "error": f"Strategy {strategy_name} not registered in all sub-populations",
                }

        all_results: List[Dict[str, Any]] = []
        params = self._evolvers[0]._strategies[strategy_name]["parameters"]

        for gen in range(total_generations):
            # Each population evolves one generation
            gen_results = await asyncio.gather(
                *[
                    self._evolve_one_generation(evolver, strategy_name, gen_idx)
                    for gen_idx, evolver in enumerate(self._evolvers)
                ]
            )
            all_results.extend(gen_results)

            # Periodic migration
            if (gen + 1) % self._migration_interval == 0 and len(self._evolvers) > 1:
                await self._migrate(strategy_name, params)

        # Collect best from all sub-populations
        all_elite: List[Dict[str, Any]] = []
        for evolver in self._evolvers:
            elite = evolver.get_elite_archive().get(strategy_name, [])
            all_elite.extend(elite)

        all_elite.sort(key=lambda x: x["fitness"], reverse=True)

        overall_best = all_elite[0] if all_elite else {
            "params": self._evolvers[0]._strategies[strategy_name]["best_parameters"],
            "fitness": float("-inf"),
        }

        # Update the first evolver's strategy with the merged best
        self._evolvers[0]._strategies[strategy_name]["best_fitness"] = overall_best[
            "fitness"
        ]
        self._evolvers[0]._strategies[strategy_name]["best_parameters"] = (
            overall_best["params"]
        )

        self._merged_elite[strategy_name] = all_elite

        return {
            "status": "success",
            "strategy": "multi_population",
            "best_fitness": overall_best["fitness"],
            "best_parameters": overall_best["params"],
            "num_populations": len(self._evolvers),
            "total_generations": total_generations,
            "migration_count": total_generations // self._migration_interval,
            "sub_population_results": all_results,
            "elite_archive": all_elite[:10],
        }

    async def _evolve_one_generation(
        self,
        evolver: StrategyEvolver,
        strategy_name: str,
        pop_index: int,
    ) -> Dict[str, Any]:
        """Evolve one generation for a single sub-population."""
        strategy_data = evolver._strategies[strategy_name]
        parameters = strategy_data["parameters"]

        # For genetic: create initial population if needed
        if not evolver._strategies.get(strategy_name):
            return {"error": "not registered", "pop": pop_index}

        # Use the evolver's own evolve method (which runs all generations at once)
        # For step-by-step we need to run individual generations inline
        # We'll simulate one generation of genetic evolution
        pop_size = evolver._population_size
        if not evolver._current_population:
            evolver._current_population = [
                evolver._random_params(parameters) for _ in range(pop_size)
            ]

        fitnesses = [
            await evolver._evaluate_fitness(strategy_name, p)
            for p in evolver._current_population
        ]

        sorted_idx = sorted(
            range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True
        )
        evolver._current_population = [evolver._current_population[i] for i in sorted_idx]
        fitnesses = [fitnesses[i] for i in sorted_idx]

        best_f = fitnesses[0]
        best_p = evolver._current_population[0].copy()

        # Elitism
        new_pop = []
        elite_count = min(evolver._elite_count, len(evolver._current_population))
        for idx in range(elite_count):
            new_pop.append(copy.deepcopy(evolver._current_population[idx]))

        # Adaptive mutation
        mutation_rate = evolver._mutation_rate
        if evolver._adaptive_mutation:
            diversity = evolver._compute_diversity(
                evolver._current_population, parameters
            )
            mutation_rate = evolver._adjust_mutation_rate(diversity)

        while len(new_pop) < pop_size:
            p1_idx = random.randint(0, max(1, pop_size // 2) - 1)
            p2_idx = random.randint(0, max(1, pop_size // 2) - 1)
            p1 = evolver._current_population[p1_idx]
            p2 = evolver._current_population[p2_idx]
            child = evolver._crossover(p1, p2)
            child = evolver._mutate_with_rate(child, parameters, mutation_rate)
            new_pop.append(child)

        evolver._current_population = new_pop
        evolver._current_fitnesses = fitnesses

        # Update elite archive
        evolver._update_elite_archive(strategy_name, best_p, best_f)

        return {
            "pop": pop_index,
            "best_fitness": best_f,
            "avg_fitness": statistics.mean(fitnesses) if fitnesses else 0.0,
        }

    async def _migrate(
        self, strategy_name: str, parameters: Dict[str, Any]
    ):
        """Perform migration between sub-populations based on topology."""
        n = len(self._evolvers)
        if n < 2:
            return

        # Collect best individuals from each population
        bests: List[Dict[str, float]] = []
        for evolver in self._evolvers:
            if evolver._current_population and evolver._current_fitnesses:
                bests.append(evolver._current_population[0].copy())
            else:
                bests.append({})

        emigrant_count = max(1, int(self._migration_rate * self._evolvers[0]._population_size))

        if self._migration_topology == MigrationTopology.RING:
            for i in range(n):
                sender = i
                receiver = (i + 1) % n
                await self._send_migrants(bests[sender], self._evolvers[receiver], emigrant_count)

        elif self._migration_topology == MigrationTopology.FULLY_CONNECTED:
            for i in range(n):
                for j in range(n):
                    if i != j:
                        await self._send_migrants(bests[i], self._evolvers[j], emigrant_count)

        elif self._migration_topology == MigrationTopology.RANDOM:
            for i in range(n):
                j = random.choice([x for x in range(n) if x != i])
                await self._send_migrants(bests[i], self._evolvers[j], emigrant_count)

        logger.debug(
            f"Migration completed: {n} populations, topology={self._migration_topology.value}"
        )

    async def _send_migrants(
        self,
        migrant: Dict[str, float],
        receiver: StrategyEvolver,
        count: int,
    ):
        """Replace worst individuals in receiver with copies of migrant."""
        if not migrant or not receiver._current_population:
            return

        pop = receiver._current_population
        # Replace last `count` individuals
        for i in range(min(count, len(pop))):
            pop[-1 - i] = copy.deepcopy(migrant)

        logger.debug(f"Sent {count} migrants to population")

    def get_diversity(self) -> Dict[str, Any]:
        """Aggregate diversity from all sub-populations."""
        result: Dict[str, Any] = {"per_population": []}
        total_div = 0.0
        for i, evolver in enumerate(self._evolvers):
            div_data = evolver.get_diversity()
            result["per_population"].append(
                {"pop_index": i, "diversity": div_data["current_diversity"]}
            )
            total_div += div_data["current_diversity"]
        result["average_diversity"] = (
            total_div / len(self._evolvers) if self._evolvers else 0.0
        )
        return result

    def get_elite_archive(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return merged elite archives from all sub-populations."""
        if self._merged_elite:
            return copy.deepcopy(self._merged_elite)
        merged: Dict[str, List[Dict[str, Any]]] = {}
        for evolver in self._evolvers:
            for name, entries in evolver.get_elite_archive().items():
                if name not in merged:
                    merged[name] = []
                merged[name].extend(entries)
        for name in merged:
            merged[name].sort(key=lambda x: x["fitness"], reverse=True)
            merged[name] = merged[name][:10]
        self._merged_elite = merged
        return copy.deepcopy(merged)

    def get_convergence(self) -> Dict[str, Any]:
        """Aggregate convergence from all sub-populations."""
        result: Dict[str, Any] = {"per_population": []}
        converged_count = 0
        for i, evolver in enumerate(self._evolvers):
            conv = evolver.get_convergence()
            conv["pop_index"] = i
            result["per_population"].append(conv)
            if conv["converged"]:
                converged_count += 1
        result["all_converged"] = converged_count == len(self._evolvers)
        result["converged_count"] = converged_count
        return result

    async def start(self):
        for evolver in self._evolvers:
            await evolver.start()

    async def stop(self):
        for evolver in self._evolvers:
            await evolver.stop()

    def pause(self):
        for evolver in self._evolvers:
            evolver.pause()

    def resume(self):
        for evolver in self._evolvers:
            evolver.resume()

    def reset(self):
        for evolver in self._evolvers:
            evolver.reset()
        self._merged_elite = {}
