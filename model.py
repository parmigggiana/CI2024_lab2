import itertools
from typing import Self

import geopy.distance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from icecream import ic
from mpl_toolkits.basemap import Basemap

rng = np.random.Generator(bit_generator=np.random.PCG64(0xDEADBEEF))

explored_genesets = {}
dist_mat = None
df = None


def get_df(filename):
    global df, dist_mat
    df = pd.read_csv(filename, header=None, names=["name", "lat", "lon"])
    dist_mat = np.zeros((df.shape[0], df.shape[0]))
    for c1, c2 in itertools.combinations(df.itertuples(), 2):
        dist_mat[c1.Index, c2.Index] = dist_mat[c2.Index, c1.Index] = (
            geopy.distance.geodesic((c1.lat, c1.lon), (c2.lat, c2.lon)).km
        )
    return df


class Geneset:
    def __init__(self, genes: np.ndarray):
        if genes[1] < genes[-1]:  # make sure that direction is always the same
            genes = np.roll(genes[::-1], 1)
        self._genes = genes[1:]
        self._true_genes = genes
        self._len = genes.shape[0] - 1
        self.age = 0

    @property
    def cost(self):
        if self not in explored_genesets:
            self.compute_cost()
        return explored_genesets[self]

    def compute_cost(self):
        global explored_genesets
        cost = 0
        for i in range(self._len + 1):
            cost += dist_mat[self._true_genes[i], self._true_genes[i - 1]]
        explored_genesets[self] = cost

    def mutate(
        self,
        strategy: str = "scramble",
        prob: float = None,
    ) -> Self:
        """Return a new Geneset mutating according to the selected strategy
        valid strategies are:
        - scramble (default)
            Select some random alleles, scramble them
        - swap
            Select two random genes, swap them
        - insert
            Select two loci, move the second one after the first
        - inversion
            Select two random loci, invert all the alleles between them.
        Args:
            prob (float)

        Returns:
            Geneset
        """
        new_genes = self._genes.copy()
        match strategy:
            case "scramble":
                assert prob is not None, "Scramble mutation requires a probability"
                mask = rng.random(self._len) < prob
                scrambled = new_genes[mask]
                rng.shuffle(scrambled)
                new_genes[mask] = scrambled
            case "swap":
                swap_indices = rng.choice(self._len, 2, replace=False)
                new_genes[swap_indices] = new_genes[swap_indices[::-1]]
            case "insert":
                insert_indices = rng.choice(self._len, 2, replace=False)
                if insert_indices[0] > insert_indices[1]:
                    insert_indices = insert_indices[::-1]
                new_genes = np.concatenate(
                    [
                        self._genes[: insert_indices[0]],
                        [self._genes[insert_indices[1]]],
                        self._genes[insert_indices[0] : insert_indices[1]],
                        self._genes[insert_indices[1] + 1 :],
                    ]
                )
            case "inversion":
                invert_indices = rng.choice(self._len, 2, replace=False)
                if invert_indices[0] > invert_indices[1]:
                    new_genes = new_genes[::-1]
                    invert_indices = invert_indices[::-1]
                    invert_indices[0] += 1
                new_genes[invert_indices[0] : invert_indices[1]] = new_genes[
                    invert_indices[0] : invert_indices[1]
                ][::-1]
            case _:
                raise ValueError(f"Invalid mutation strategy: {strategy}")

        return Geneset(np.concatenate([[self._true_genes[0]], new_genes]))

    def __repr__(self):
        return f"Geneset({self._true_genes})"

    def __lt__(self, other):
        return self.cost < other.cost

    def __le__(self, other):
        return self.cost <= other.cost

    def __hash__(self) -> int:
        min_index = np.nonzero(self._true_genes == 0)[0][0]
        rotated_genes = np.roll(self._true_genes, -min_index)
        return hash(tuple(rotated_genes))

    def format_wolfram(self):
        return "Uninplemented"
        # BUG This is valid in wolfram but the indexes don't correspond to the correct cities
        # return "{" + ", ".join(map(str, self.genes + 1)) + ", 1}"

    def plot(self):
        points = df[["lat", "lon"]].values[self._true_genes]

        m = Basemap(
            projection="merc",
            resolution=None,
            llcrnrlon=min(points[:, 1]) - 1,
            llcrnrlat=min(points[:, 0]) - 1,
            urcrnrlon=max(points[:, 1]) + 1,
            urcrnrlat=max(points[:, 0]) + 1,
            lat_ts=points[0][0],
        )

        m.shadedrelief()

        # Convert latitude and longitude to map projection coordinates
        x, y = m(points[:, 1], points[:, 0])

        m.plot([x[-1], x[0]], [y[-1], y[0]], "-", color="orange")
        m.plot(x, y, "o-", color="orange")
        plt.title(f"cost: {self.cost:.2f}")

    def cycle_xover(self, parent2):
        selected_loci = rng.choice(np.arange(self._len), 2, replace=False)
        child = np.zeros_like(self._genes)
        if selected_loci[0] > selected_loci[1]:
            selected_loci = selected_loci[::-1]

        child[selected_loci[0] : selected_loci[1]] = self._genes[
            selected_loci[0] : selected_loci[1]
        ]

        j = 0
        for i in np.arange(parent2._len):
            if j == selected_loci[0]:
                j = selected_loci[1]
                break
            if parent2._genes[i] not in child:
                child[j] = parent2._genes[i]
                j += 1

        for i in np.arange(parent2._len):
            if parent2._genes[i] not in child:
                child[j] = parent2._genes[i]
                j += 1
        return Geneset(np.concatenate([[self._true_genes[0]], child]))
