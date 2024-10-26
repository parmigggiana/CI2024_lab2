import itertools
from typing import Self

import geopy.distance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap

rng = np.random.Generator(bit_generator=np.random.PCG64(0xDEADBEEF))


def get_df(filename):
    return pd.read_csv(filename, header=None, names=["name", "lat", "lon"])


def get_distance_matrix(df):
    distance_matrix = np.zeros((df.shape[0], df.shape[0]))
    for c1, c2 in itertools.combinations(df.itertuples(), 2):
        distance_matrix[c1.Index, c2.Index] = distance_matrix[c2.Index, c1.Index] = (
            geopy.distance.geodesic((c1.lat, c1.lon), (c2.lat, c2.lon)).km
        )
    return distance_matrix


class Geneset:
    def __init__(self, genes: np.ndarray):
        self.genes = genes
        self.cost = None
        self.len = genes.shape[0]
        self.age = 0

    def compute_cost(self, dist_mat):
        self.cost = 0
        for i in range(self.len):
            self.cost += dist_mat[self.genes[i], self.genes[i - 1]]

    def mutate(
        self,
        strategy: str = "scramble",
        prob: float = None,
    ) -> Self:  # TODO make it better :)
        """Return a new Geneset mutating according to the selected strategy
        valid strategies are:
        - scramble (default)
            Select some random alleles, scramble them
        - swap
            Select two random genes, swap them
        Args:
            prob (float)

        Returns:
            Geneset
        """
        new_geneset = Geneset(self.genes.copy())
        match strategy:
            case "scramble":
                assert prob is not None, "Scramble mutation requires a probability"
                mask = rng.random(self.len) < prob
                scrambled = new_geneset.genes[mask]
                rng.shuffle(scrambled)
                new_geneset.genes[mask] = scrambled
            case "swap":
                swap_indices = rng.choice(self.len, 2, replace=False)
                new_geneset.genes[swap_indices] = new_geneset.genes[swap_indices[::-1]]
            case _:
                raise ValueError(f"Invalid mutation strategy: {strategy}")
        return new_geneset

    def __repr__(self):
        return f"Geneset({self.genes})"

    def __lt__(self, other):
        return self.cost < other.cost

    def __le__(self, other):
        return self.cost <= other.cost

    def format_wolfram(self):
        return "Uninplemented"
        # BUG This is valid in wolfram but the indexes don't correspond to the correct cities
        # return "{" + ", ".join(map(str, self.genes + 1)) + ", 1}"

    def plot(self, df):
        points = df[["lat", "lon"]].values[self.genes]
        m = Basemap(
            projection="aeqd",
            resolution=None,
            llcrnrlon=min(points[:, 1]) - 1,
            llcrnrlat=min(points[:, 0]) - 1,
            urcrnrlon=max(points[:, 1]) + 1,
            urcrnrlat=max(points[:, 0]) + 1,
            lat_0=points[len(points) // 2, 0],
            lon_0=points[len(points) // 2, 1],
        )

        m.shadedrelief()

        # Convert latitude and longitude to map projection coordinates
        x, y = m(points[:, 1], points[:, 0])

        m.plot([x[-1], x[0]], [y[-1], y[0]], "-", color="orange")
        m.plot(x, y, "o-", color="orange")
        plt.title(f"cost: {self.cost:.2f}")
