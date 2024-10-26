"""
TSP
Using Tournament Selection, Crossover, and Mutation
"""

import concurrent.futures

import matplotlib.pyplot as plt
import numpy as np
from icecream import ic

from model import Geneset, get_df, get_distance_matrix

POPULATION_SIZE = 50
TOURNAMENTS = 2
REPRODUCTIVE_RATE = 2
PARENTS = 10
CHAMPIONS_PER_TOURNAMENT = 1
MUTATION_RATE = 0.5
MAX_AGE = 1
MIN_ITERS = 100
WINDOW_SIZE = 2
IMPROVEMENT_RATE_EARLY_STOP = 0.01
SA_PROB = 0.1

rng = np.random.Generator(bit_generator=np.random.PCG64(0xDEADBEEF))


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
    # Select the best individuals from the population with simulated annealing
    selected = np.empty(POPULATION_SIZE, dtype=Geneset)
    np.sort(population)
    i = 0
    for g in population:
        if i >= POPULATION_SIZE:
            break
        if g.age > MAX_AGE or rng.random() < SA_PROB:
            continue
        selected[i] = g
        i += 1
    else:
        selected[i:] = population[: POPULATION_SIZE - i]
    return selected


def mutate(population: np.ndarray):
    mutated_population = np.empty_like(population)
    # np.apply_along_axis(, 0, population)
    for i, child in enumerate(population):
        mutated_population[i] = child.mutate(MUTATION_RATE / child.len)
    return mutated_population


def compute_costs(population, distance_matrix):
    for geneset in population:
        geneset.compute_cost(distance_matrix)


def main(filename, starting_geneset: np.ndarray = None):
    def plot(history, geneset, df):
        plt.subplots(1, 2, figsize=(16, 10))

        # plot improvement graph
        ax = plt.subplot(121)
        ax.set_yscale("log")
        for j, h in enumerate(history):
            ax.scatter(
                [j] * len(h),
                h,
                color="blue",
                alpha=1 / POPULATION_SIZE,
                marker=".",
            )
        ax.plot([h[0] for h in history], color="red")
        ax.set_title(f"iterations: {len(history)}")

        # plot map
        plt.subplot(122)
        geneset.plot(df)
        plt.show()

    # Geneset is an array of all numbers from 0 to df.shape[0] - 1, in order of travelling
    # Fix the starting point at the first one and only mutate the rest
    df = get_df(filename)
    distance_matrix = get_distance_matrix(df)

    if not starting_geneset:
        population = np.array(
            [
                Geneset(genes)
                for genes in [
                    np.insert(rng.permutation(np.arange(1, df.shape[0])), 0, 0)
                    for _ in range(POPULATION_SIZE)
                ]
            ]
        )
    else:
        # create population as copies of the starting geneset
        population = np.array([starting_geneset] * POPULATION_SIZE)

    compute_costs(population, distance_matrix)

    best_geneset = population[np.argmin([g.cost for g in population])]
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
            if population[0].cost < best_geneset.cost:
                best_geneset = population[0]

            improvement_rate = (
                (history[-i // WINDOW_SIZE][0] - history[-1][0]) / (i // WINDOW_SIZE)
                if i > MIN_ITERS
                else np.inf
            )
            if improvement_rate < IMPROVEMENT_RATE_EARLY_STOP:
                break

    print(f"Iterations: {i}")
    print(f"Path cost: {best_geneset.cost:.2f}")
    print(f"visited cities:\n{df.iloc[best_geneset.genes].name}")
    # print(f"Wolfram-coded path: {best_geneset.format_wolfram()}")
    plot(history, best_geneset, df)
