"""
TSP
Using Tournament Selection, Crossover, and Mutation
"""

import itertools
import multiprocessing
import pandas as pd
import numpy as np
import geopy.distance
from typing import Self
from icecream import ic

FILENAME = "vanuatu.csv"
POPULATION_SIZE = 50
TOURNAMENTS = 5
REPRODUCTIVE_RATE = 2
PARENTS = 10
CHAMPIONS = 1
MUTATION_RATE = 2
MAX_AGE = 2

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

    def mutate(self, prob: float) -> Self:
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


def xover(population):
    # Randomly crossover pairs of parents to create POPULATION_SIZE children
    # Always keep the first gene fixed
    children = []
    for _ in range(POPULATION_SIZE * REPRODUCTIVE_RATE):
        parent1, parent2 = rng.choice(population, 2, replace=False)
        child = Geneset(parent1.genes.copy())
        start, end = rng.choice(range(1, child.len), 2, replace=False)
        child.genes[start:end] = parent2.genes[start:end]
        children.append(child)

    return children


def select_parents(population):
    return population[:PARENTS]


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

    population = [
        Geneset(genes)
        for genes in [
            np.insert(rng.permutation(np.arange(1, df.shape[0])), 0, 0)
            for _ in range(POPULATION_SIZE)
        ]
    ]
    [geneset.compute_cost(distance_matrix) for geneset in population]

    with multiprocessing.Pool() as pool:
        while True:
            # ic(population)
            batches = np.array_split(population, PARENTS)

            parents = pool.map(select_parents, batches)
            # parents = select_parents(population)
            # ic(parents)
            children = pool.map(xover, parents)
            # ic(children)
            mutated_children = pool.map(mutate, children)

            # ic(mutated_children)
            pool.starmap(
                compute_costs,
                [(children, distance_matrix) for children in mutated_children],
            )
            ic(mutated_children)
            population = select(population + mutated_children)
            break


if __name__ == "__main__":
    main(FILENAME)
