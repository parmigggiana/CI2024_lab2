import matplotlib.pyplot as plt
import numpy as np
from icecream import ic

from model import Geneset, get_df, get_distance_matrix


def main(filename):
    df = get_df(filename)
    distance_matrix = get_distance_matrix(df)

    geneset = Geneset(
        np.zeros((len(distance_matrix)), dtype=np.uint)
    )  # init the first node
    for i in range(1, df.shape[0]):
        mask = np.ones(distance_matrix.shape[0], dtype=bool)
        mask[geneset.genes] = False
        next_shortest_node = np.argmin(
            np.where(mask, distance_matrix[geneset.genes[i - 1]], np.inf)
        )
        geneset.genes[i] = next_shortest_node
        # ic(i, geneset.genes[i - 1], next_shortest_node)
    geneset.compute_cost(distance_matrix)
    return geneset


if __name__ == "__main__":
    geneset = main("vanuatu.csv")
    plt.figure()
    geneset.plot(get_df("vanuatu.csv"))
    plt.show()
