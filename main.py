"""
TSP
Using Tournament Selection, Crossover, and Mutation
"""

import itertools
import concurrent.futures
import pandas as pd
import numpy as np
import geopy.distance
from typing import Self
from icecream import ic
import matplotlib.pyplot as plt

FILENAME = "italy.csv"
POPULATION_SIZE = 50
TOURNAMENTS = 2
REPRODUCTIVE_RATE = 2
PARENTS = 10
CHAMPIONS_PER_TOURNAMENT = 0
MUTATION_RATE = 0.4
MAX_AGE = 1
MIN_ITERS = 400
WINDOW_SIZE = 2

rng = np.random.Generator(np.random.PCG64(0xDEADBEEF))


class Geneset:
    def __init__(self, genes):
        self.genes = genes
        self.cost = None
        self.len = genes.shape[0]
        self.age = 0

    def compute_cost(self, dist_mat):
        self.cost = 0
        for i in range(self.len):
            self.cost += dist_mat[self.genes[i], self.genes[i - 1]]

    def mutate(self, prob: float) -> Self:  # TODO make it better :)
        """Return a new Geneset mutating each gene with probability `prob`
            Mutation is done by swapping two adjacent genes
            Always keep the first gene fixed
        Args:
            prob (float)

        Returns:
            Geneset
        """
        new_geneset = Geneset(self.genes.copy())
        for i in range(2, len(new_geneset.genes)):
            if rng.random() < prob:
                new_geneset.genes[i], new_geneset.genes[i - 1] = (
                    new_geneset.genes[i - 1],
                    new_geneset.genes[i],
                )
        return new_geneset

    def __repr__(self):
        return f"Geneset({self.genes})"

    def __lt__(self, other):
        return self.cost < other.cost

    def __le__(self, other):
        return self.cost <= other.cost


def xover(population):  # TODO proper crossover
    # Randomly crossover pairs of parents to create POPULATION_SIZE children
    # Always keep the first gene fixed
    children = np.empty(POPULATION_SIZE * REPRODUCTIVE_RATE, dtype=Geneset)
    for i in range(POPULATION_SIZE * REPRODUCTIVE_RATE):
        parent1, parent2 = rng.choice(population, 2, replace=False)
        child = Geneset(parent1.genes.copy())
        start, end = rng.choice(range(1, child.len), 2, replace=False)
        child.genes[start:end] = parent2.genes[start:end]
        child.genes = parent1.genes
        children[i] = child

    return children


def select_parents(population: np.ndarray):
    population.sort()
    selected = population[: PARENTS // TOURNAMENTS]
    for g in selected[CHAMPIONS_PER_TOURNAMENT:]:
        g.age += 1
    return selected


def select(population):
    # Select the best individuals from the population
    return np.sort(population)[:POPULATION_SIZE]


def mutate(population: np.ndarray):
    mutated_population = np.empty_like(population)
    # np.apply_along_axis(, 0, population)
    for i, child in enumerate(population):
        mutated_population[i] = child.mutate(MUTATION_RATE / child.len)
    return mutated_population


def compute_costs(population, distance_matrix):
    for geneset in population:
        geneset.compute_cost(distance_matrix)


def main(filename):

    df = pd.read_csv(filename, header=None, names=["name", "lat", "lon"])
    distance_matrix = np.zeros((df.shape[0], df.shape[0]))
    for c1, c2 in itertools.combinations(df.itertuples(), 2):
        distance_matrix[c1.Index, c2.Index] = distance_matrix[c2.Index, c1.Index] = (
            geopy.distance.geodesic((c1.lat, c1.lon), (c2.lat, c2.lon)).km
        )
    # Geneset is an array of all numbers from 0 to df.shape[0] - 1, in order of travelling
    # Fix the starting point at the first one and only mutate the rest

    # according to some short tests the second one is faster
    # genes = np.insert(np.random.permutation(df.shape[0] - 1) + 1, 0, 0)
    # geneset = np.insert(np.random.permutation(np.arange(1, df.shape[0])), 0, 0)

    population = np.array(
        [
            Geneset(genes)
            for genes in [
                np.insert(rng.permutation(np.arange(1, df.shape[0])), 0, 0)
                for _ in range(POPULATION_SIZE)
            ]
        ]
    )
    compute_costs(population, distance_matrix)

    best_geneset = population[0].genes
    best_cost = population[0].cost
    i = 0
    history = []

    with concurrent.futures.ThreadPoolExecutor(TOURNAMENTS) as pool:
        while True:
            i += 1
            batches = np.array_split(population, TOURNAMENTS)

            futures = {pool.submit(select_parents, batch) for batch in batches}
            parents = np.concatenate(
                [f.result() for f in concurrent.futures.as_completed(futures)],
            )
            children = xover(parents)
            futures = {pool.submit(child.mutate, MUTATION_RATE) for child in children}
            mutated_children = np.array(
                [f.result() for f in concurrent.futures.as_completed(futures)]
            )
            concurrent.futures.wait(
                {
                    pool.submit(geneset.compute_cost, distance_matrix)
                    for geneset in mutated_children
                }
            )

            population = np.concatenate([population, mutated_children])
            population = select(population)

            history.append([g.cost for g in population])
            if population[0].cost < best_cost:
                best_cost = population[0].cost
                best_geneset = population[0].genes

            improvement_rate = (
                (history[-1][0] - history[-i // WINDOW_SIZE][0]) / (i // WINDOW_SIZE)
                if i > MIN_ITERS
                else np.inf
            )
            if improvement_rate < 0.01:
                break

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for j, h in enumerate(history):
        ax.scatter(
            [j] * len(h),
            h,
            color="blue",
            alpha=1 / POPULATION_SIZE,
            marker=".",
        )
    ax.plot([h[0] for h in history], color="red")
    ax.set_title(f"iterations: {i} - best cost: {best_cost:.2f}")
    print(f"Iterations: {i}")
    print(f"Path cost: {best_cost:.2f}")
    print(f"visited cities:\n{df.iloc[best_geneset].name}")
    print(
        f"Wolfram-coded path: {{{', '.join(f'{best_geneset + 1}'[1:-1].split())}, 1}}"
    )
    plt.show()


if __name__ == "__main__":
    main(FILENAME)
