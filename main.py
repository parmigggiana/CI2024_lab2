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
        "min_iters": 2,
        "window_size": 1,
        "min_improvement_rate": 100,
        "sa_prob": 0.1,
        "mutation_strategy": "swap",
        "xover_strategy": None,
    },
    "italy.csv": {
        "population_size": 50,
        "tournaments": 5,
        "reproductive_rate": 3,
        "parents": 10,
        "champions_per_tournament": 0,
        "max_age": 1,
        "min_iters": 100,
        "window_size": 2 / 3,
        "min_improvement_rate": 0.1,
        "sa_prob": 0.2,
        "temperature_update_interval": 20,
        "mutation_strategy": "inversion",
        "xover_strategy": "inverover",
    },
    "russia.csv": {
        "population_size": 50,
        "tournaments": 5,
        "reproductive_rate": 3,
        "parents": 10,
        "champions_per_tournament": 1,
        "max_age": 2,
        "min_iters": 500,
        "window_size": 1 / 2,
        "min_improvement_rate": 0.02,
        "sa_prob": 0.2,
        "temperature_update_interval": 30,
        "mutation_strategy": "inversion",
        "xover_strategy": "inverover",
    },
    "us.csv": {
        "population_size": 20,
        "tournaments": 4,
        "reproductive_rate": 3,
        "parents": 8,
        "champions_per_tournament": 1,
        "max_age": 3,
        "min_iters": 1000,
        "window_size": 1 / 3,
        "min_improvement_rate": 0.02,
        "sa_prob": 0.25,
        "temperature_update_interval": 50,
        "mutation_strategy": "inversion",
        "xover_strategy": "inverover",
    },
    "china.csv": {
        "population_size": 15,
        "tournaments": 2,
        "reproductive_rate": 3,
        "parents": 4,
        "champions_per_tournament": 1,
        "max_age": 3,
        "min_iters": 2000,
        "window_size": 1 / 3,
        "min_improvement_rate": 0.02,
        "sa_prob": 0.3,
        "temperature_update_interval": 50,
        "mutation_strategy": "inversion",
        "xover_strategy": "inverover",
    },
}


def plot(history, geneset, population_size, elapsed_time):
    plt.subplots(1, 2, figsize=(16, 10))

    # plot improvement graph
    ax = plt.subplot(121)
    ax.set_yscale("log")
    for j, h in enumerate(history):
        ax.scatter(
            [j] * len(h),
            h,
            color="blue",
            alpha=1 / population_size,
            marker=".",
        )
    ax.plot([h[0] for h in history], color="red")
    ax.set_title(f"generations: {len(history)} - {elapsed_time}")

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
