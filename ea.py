import concurrent.futures

import numpy as np

from model import Geneset

rng = np.random.Generator(bit_generator=np.random.PCG64(0xDEADBEEF))


class EA:
    def __init__(
        self,
        population_size=10,
        tournaments=2,
        reproductive_rate=2,
        parents=2,
        champions_per_tournament=0,
        max_age=2,
        min_iters=100,
        window_size=1 / 2,
        min_improvement_rate=0.01,
        sa_prob=0,
        temperature_update_interval=10,
        mutation_strategy="inversion",
        xover_strategy="cycle",
        mutation_prob=None,
    ):
        assert population_size >= 2, "Population size must be at least 2"
        assert tournaments >= 1, "Tournaments must be at least 1"
        assert reproductive_rate >= 1, "Reproductive rate must be at least 1"
        assert parents >= 2, "Parents must be at least 2"
        assert (
            champions_per_tournament >= 0
        ), "Champions per tournament must be at least 1. Set to 0 to disable"
        assert max_age >= 1, "Max age must be at least 1"
        assert (
            min_iters >= 1 / window_size
        ), "Minimum iterations must be at least (1 / window size)"
        assert 0 < window_size <= 1, "Window size must be in (0, 1]"
        assert (
            min_improvement_rate > 0
        ), "Minimum improvement rate must be non-negative. Default to 0.1"
        assert (
            0 <= sa_prob < 1
        ), "SA probability must be between 0 and 1. Default to 0 (disabled)"
        assert (
            temperature_update_interval >= 1
        ), "The number of iterations between each update of the temperature must be at least 1"
        assert mutation_strategy in [
            "scramble",
            "swap",
            "insert",
            "inversion",
        ], "Mutation strategy must be scramble, swap, insertion, or inversion (default)"
        assert xover_strategy in [
            None,
            "cycle",
            # "pmx",
            # "ox",
            # "erx",
            "inverover",
        ], "Crossover strategy must be cycle (default), pmx, ox, erx, or inverover"
        if mutation_strategy == "scramble":
            assert (
                0 <= mutation_prob < 1
            ), "Mutation probability must be between 0 and 1 for scramble strategy"

        self.population_size = population_size
        self.tournaments = tournaments
        self.reproductive_rate = reproductive_rate
        self.parents = parents
        self.champions_per_tournament = champions_per_tournament
        self.max_age = max_age
        self.min_iters = min_iters
        self.window_size = window_size
        self.min_improvement_rate = min_improvement_rate
        self.sa_prob = sa_prob
        self.mutation_strategy = mutation_strategy
        self.xover_strategy = xover_strategy
        self.mutation_prob = mutation_prob
        self.temperature_update_interval = temperature_update_interval

    def xover(self, population, strategy: str = "cycle"):
        # Randomly crossover pairs of parents to create self.population_size children
        children = np.empty(
            self.population_size * self.reproductive_rate, dtype=Geneset
        )
        for i in range(self.population_size * self.reproductive_rate):
            parent1, parent2 = rng.choice(population, 2, replace=False)
            match strategy:
                case None:
                    child = parent1
                case "cycle":
                    child = parent1.cycle_xover(parent2)
                case "inverover":
                    child = parent1.inverover_xover(parent2)
                case _:
                    raise NotImplementedError(f"Strategy {strategy} not implemented")
            children[i] = child

        return children

    def select_parents(self, population: np.ndarray, temperature=1):
        population.sort()
        selected = np.empty(self.parents // self.tournaments, dtype=Geneset)
        i = 0
        for g in population:
            if i >= self.parents // self.tournaments:
                break
            # if g.age >= self.max_age:
            #     continue
            if rng.random() < (self.sa_prob * temperature):
                continue
            # if i < self.champions_per_tournament:
            #     g.age -= 1
            selected[i] = g
            i += 1
        if (
            i < self.parents // self.tournaments
        ):  # finished the loop and there's still empty slots to fill
            # print(i, selected, "sa_prob or temperature too high")
            selected[i:] = population[: self.parents // self.tournaments - i]

        return selected

    def select(self, population):
        # Select the best individuals from the population with simulated annealing
        return np.sort(population)[: self.population_size]

    def mutate(self, population: np.ndarray):
        mutated_population = np.empty_like(population)
        for i, child in enumerate(population):
            mutated_population[i] = child.mutate(
                self.mutation_strategy, self.mutation_prob
            )
        return mutated_population

    def run(self, starting_geneset: np.ndarray):

        # Geneset is an array of all numbers from 0 to df.shape[0] - 1, in order of travelling
        # Fix the starting point at the first one and only mutate the rest

        # create population as copies of the starting geneset
        population = np.array([starting_geneset] * self.population_size)

        best_geneset = population[np.argmin([g.cost for g in population])]
        i = 0
        temp = 1
        history = []

        with concurrent.futures.ThreadPoolExecutor() as pool:
            while True:
                i += 1

                # Parent selection
                batches = np.array_split(population, self.tournaments)
                futures = {
                    pool.submit(self.select_parents, batch, temp) for batch in batches
                }
                parents = np.concatenate(
                    [f.result() for f in concurrent.futures.as_completed(futures)],
                )

                # Crossover
                children = self.xover(parents, self.xover_strategy)
                futures = {
                    pool.submit(
                        child.mutate, self.mutation_strategy, self.mutation_prob
                    )
                    for child in children
                }

                # Mutation
                mutated_children = np.array(
                    [f.result() for f in concurrent.futures.as_completed(futures)]
                )

                # Survivor selection
                population = np.concatenate([population, mutated_children])
                population = self.select(population)
                # for g in population: # Aging seems to be bugged, not sure why but might have to do with the hash function
                # g.age += 1

                history.append([g.cost for g in population])
                if population[0].cost < best_geneset.cost:
                    best_geneset = population[0]

                improvement_rate = (
                    (history[-int(i * self.window_size)][0] - history[-1][0])
                    / (int(i * self.window_size))
                    if (int(i * self.window_size)) > 0
                    else (history[0][0] - history[-1][0]) / i
                )
                if i % self.temperature_update_interval == 0:
                    local_improvement_rate = (
                        history[-self.temperature_update_interval][0] - history[-1][0]
                    ) / (self.temperature_update_interval)
                    if local_improvement_rate <= self.min_improvement_rate * 10:
                        temp *= 1.2
                    else:
                        temp /= 1.1

                if i >= self.min_iters and improvement_rate < self.min_improvement_rate:
                    break

            return best_geneset, i, history
