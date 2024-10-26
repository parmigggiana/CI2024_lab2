"""
TSP
Using Tournament Selection, Crossover, and Mutation
"""

from ea import EA
from greedy import main as greedy
from model import get_df
import matplotlib.pyplot as plt

PLOT = True

instances = {
    "vanuatu.csv": {
        "population_size": 10,
        "tournaments": 1,
        "reproductive_rate": 5,
        "parents": 2,
        "champions_per_tournament": 0,
        "max_age": 1,
        "min_iters": 10,
        "window_size": 2,
        "min_improvement_rate": 0.01,
        "sa_prob": 0.05,
        "mutation_strategy": "inversion",
        "xover_strategy": "cycle",
        "mutation_prob": None,
    },
    "italy.csv": {
        "population_size": 20,
        "tournaments": 4,
        "reproductive_rate": 2,
        "parents": 8,
        "champions_per_tournament": 1,
        "max_age": 2,
        "min_iters": 200,
        "window_size": 2,
        "min_improvement_rate": 0.01,
        "sa_prob": 0.05,
        "mutation_strategy": "inversion",
        "xover_strategy": "cycle",
        "mutation_prob": None,
    },
    "russia.csv": {
        "population_size": 50,
        "tournaments": 5,
        "reproductive_rate": 2,
        "parents": 10,
        "champions_per_tournament": 1,
        "max_age": 2,
        "min_iters": 100,
        "window_size": 2,
        "min_improvement_rate": 0.01,
        "sa_prob": 0.05,
        "mutation_strategy": "inversion",
        "xover_strategy": "cycle",
        "mutation_prob": None,
    },
    "us.csv": {
        "population_size": 50,
        "tournaments": 5,
        "reproductive_rate": 2,
        "parents": 10,
        "champions_per_tournament": 1,
        "max_age": 2,
        "min_iters": 100,
        "window_size": 2,
        "min_improvement_rate": 0.01,
        "sa_prob": 0.05,
        "mutation_strategy": "inversion",
        "xover_strategy": "cycle",
        "mutation_prob": None,
    },
    "china.csv": {
        "population_size": 50,
        "tournaments": 5,
        "reproductive_rate": 2,
        "parents": 10,
        "champions_per_tournament": 1,
        "max_age": 2,
        "min_iters": 100,
        "window_size": 2,
        "min_improvement_rate": 0.01,
        "sa_prob": 0.05,
        "mutation_strategy": "inversion",
        "xover_strategy": "cycle",
        "mutation_prob": None,
    },
}


def plot(history, geneset, population_size):
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
    ax.set_title(f"iterations: {len(history)}")

    # plot map
    plt.subplot(122)
    geneset.plot()
    plt.show()


if __name__ == "__main__":
    for filename, instance in instances.items():
        df = get_df(filename)

        starting_geneset = greedy(df)
        ea = EA(**instance)
        best_geneset, iterations, history = ea.run(starting_geneset)

        print(f"Iterations: {iterations}")
        print(f"Path cost: {best_geneset.cost:.2f}")
        print(f"Visited cities:\n{df.iloc[best_geneset._true_genes].name}")
        # print(f"Wolfram-coded path: {best_geneset.format_wolfram()}")
        if PLOT:
            plot(history, best_geneset, population_size=instance["population_size"])
