"""
TSP
Using Tournament Selection, Crossover, and Mutation
"""

import time

import matplotlib.pyplot as plt

from ea import EA
from greedy import main as greedy
from model import get_df

PLOT = True

instances = {
    "vanuatu.csv": {
        "population_size": 20,
        "tournaments": 2,
        "reproductive_rate": 2,
        "parents": 2,
        "champions_per_tournament": 0,
        "max_age": 1,
        "min_iters": 1,
        "window_size": 2,
        "min_improvement_rate": 0.01,
        "sa_prob": 0.1,
        "mutation_strategy": "swap",
        "xover_strategy": None,
        "mutation_prob": None,
    },
    "italy.csv": {
        "population_size": 50,
        "tournaments": 4,
        "reproductive_rate": 2,
        "parents": 8,
        "champions_per_tournament": 1,
        "max_age": 2,
        "min_iters": 50,
        "window_size": 2,
        "min_improvement_rate": 0.01,
        "sa_prob": 0.1,
        "mutation_strategy": "inversion",
        "xover_strategy": "cycle",
        "mutation_prob": None,
    },
    "russia.csv": {
        "population_size": 30,
        "tournaments": 5,
        "reproductive_rate": 2,
        "parents": 10,
        "champions_per_tournament": 1,
        "max_age": 2,
        "min_iters": 200,
        "window_size": 3,
        "min_improvement_rate": 0.2,
        "sa_prob": 0.1,
        "mutation_strategy": "inversion",
        "xover_strategy": "cycle",
        "mutation_prob": None,
    },
    "us.csv": {
        "population_size": 20,
        "tournaments": 4,
        "reproductive_rate": 2,
        "parents": 8,
        "champions_per_tournament": 1,
        "max_age": 2,
        "min_iters": 500,
        "window_size": 3,
        "min_improvement_rate": 0.3,
        "sa_prob": 0.05,
        "mutation_strategy": "inversion",
        "xover_strategy": "cycle",
        "mutation_prob": None,
    },
    "china.csv": {
        "population_size": 10,
        "tournaments": 2,
        "reproductive_rate": 2,
        "parents": 4,
        "champions_per_tournament": 1,
        "max_age": 2,
        "min_iters": 500,
        "window_size": 3,
        "min_improvement_rate": 0.5,
        "sa_prob": 0.05,
        "mutation_strategy": "inversion",
        "xover_strategy": "cycle",
        "mutation_prob": None,
    },
}


def plot(history, geneset, population_size, elapsed_time):
    plt.subplots(1, 2, figsize=(16, 10))

    # plot improvement graph
    ax = plt.subplot(121)
    # ax.set_yscale("log")
    for j, h in enumerate(history):
        ax.scatter(
            [j] * len(h),
            h,
            color="blue",
            alpha=1 / population_size,
            marker=".",
        )
    ax.plot([h[0] for h in history], color="red")
    ax.set_title(f"iterations: {len(history)} - {elapsed_time}")

    # plot map
    plt.subplot(122)
    geneset.plot()


if __name__ == "__main__":
    for filename, instance in instances.items():
        df = get_df(filename)

        start = time.perf_counter()
        starting_geneset = greedy(df)
        ea = EA(**instance)
        best_geneset, iterations, history = ea.run(starting_geneset)
        end = time.perf_counter()
        elapsed = end - start
        elapsed = (
            f"{elapsed:.2f}s"
            if elapsed < 60
            else f"{elapsed//60:.0f}m {elapsed%60:.2f}s"
        )
        print(f"Iterations: {iterations}")
        print(f"Elapsed time: {elapsed}")
        print(f"Path cost: {best_geneset.cost:.2f}")
        print(f"Visited cities:\n{df.iloc[best_geneset._true_genes].name}")
        # print(f"Wolfram-coded path: {best_geneset.format_wolfram()}")
        if PLOT:
            plot(
                history,
                best_geneset,
                population_size=instance["population_size"],
                elapsed_time=elapsed,
            )
            plt.savefig(f"plots/{filename.split('.')[0]}.png")
