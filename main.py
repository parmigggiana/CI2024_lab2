from ea import main as ea
from greedy import main as greedy

FILENAME = "italy.csv"

if __name__ == "__main__":
    starting_geneset = greedy(FILENAME)
    ea(FILENAME, starting_geneset)
