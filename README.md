# TSP Problem with EA
This python module can be started by running
```sh
pip install -r requirements.txt
python main.py
```
It can be configured by changing the variables `PLOT` and `instances` in the file `main.py`

PLOT is not a live plot, instead the result is saved to png files in the `plots/` directory, it has a mild impact on the overall performance.
`instances` is a dictionary with an entry for each source file containing parameters for the EA.

For each of these instances, a greedy search is performed to obtain a starting solution. That is passed to the ea module which iterates performing tournament selection for the parents, then producing offspring with one of the configurable crossover strategies, mutating them with one of the configurable mutation strategies and then selecting the best offspring to maintain population size constant over the generations.

The available crossover strategies are:
- cycle
- inverover

The available mutation strategies are:
- scramble
- swap
- insert
- inversion

The ea algorithm hyperparameters are:
- **population_size** - the number of individuals in each generation
- **tournaments** - the number of tournaments in each generation
- **reproductive_rate** - the rate of children generated in each generation to population_size
- **parents** - the number of pareants selected in total
- **champions_per_tournament** - the number of champions for each tournament (champions don't age)
- **max_age** - the max age accepted. Individuals with age > max_age can not be selected as parents
- **min_iters** - the minimum numbers of generations
- **window_size** - improvement rate is calculated over the last $i \times window\_size$ generations
- **min_improvement_rate** - steady state is reached when the improvement rate is lower than min_improvement_rate and the current best solution is accepted
- **sa_prob** - the base probability of not selecting an individual for the next generation
- **temperature_update_interval** - the number of generations between each update of the temperature
- **mutation_strategy** - the algorithm used for mutation
- **xover_strategy** - the algorithm used for crossover
- **mutation_prob** - parameter for scramble mutation, ignored for every other strategy

> [!WARNING] Aging is disabled at the moment because it seemed to be bugged