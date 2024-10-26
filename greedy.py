import numpy as np

from model import Geneset


def main(df):
    from model import dist_mat

    geneset = Geneset(np.zeros((len(dist_mat)), dtype=np.uint))  # init the first node
    for i in range(1, df.shape[0]):
        mask = np.ones(dist_mat.shape[0], dtype=bool)
        mask[geneset._true_genes] = False
        next_shortest_node = np.argmin(
            np.where(mask, dist_mat[geneset._true_genes[i - 1]], np.inf)
        )
        geneset._true_genes[i] = next_shortest_node
        # ic(i, geneset.genes[i - 1], next_shortest_node)
    return geneset
