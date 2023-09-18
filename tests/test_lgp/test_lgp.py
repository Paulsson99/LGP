import numpy as np

from LGP.LGP import LGP
from LGP._typing import Chromosome
from LGP.selection import SelectionBase
from LGP.crossover import CrossoverBase
from LGP.mutation import MutationBase


class NoMutation(MutationBase):
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        return chromosome

class MaxSelection(SelectionBase):
    def select(self, fitness: list[float]) -> int:
        return int(np.argmax(fitness))
    
class NoCrossover(CrossoverBase):
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> tuple[Chromosome, Chromosome]:
        return parent1, parent2

def fitness_func(chromosome: Chromosome) -> float:
    return len(chromosome)


def test_log_maximize():
    lgp = LGP(
        population=[((0, 0, 0, 0),) * i for i in range(10)],
        selection_method=MaxSelection(),
        crossover_method=NoCrossover(),
        mutation_method=NoMutation(),
        fitness_func=fitness_func,
        minimize=False,
    )

    lgp.run(generations=1)
    assert lgp.best_fitness == 9
    assert lgp.population == [((0, 0, 0, 0),) * 9 for _ in range(10)]
    
    lgp.population = [((0, 0, 0, 0),) * (i + 1) for i in range(10)]

    lgp.run(generations=1)
    assert lgp.best_fitness == 10
    assert lgp.all_time_best_fitness == 10
    assert lgp.population == [((0, 0, 0, 0),) * 10 for _ in range(10)]

    lgp.population = [((0, 0, 0, 0),) * i for i in range(10)]

    lgp.run(generations=1)
    assert lgp.best_fitness == 9
    assert lgp.all_time_best_fitness == 10
    assert lgp.population == [((0, 0, 0, 0),) * 9 for _ in range(10)]


def test_log_minimize():
    lgp = LGP(
        population=[((0, 0, 0, 0),) * (i + 1) for i in range(10)],
        selection_method=MaxSelection(),
        crossover_method=NoCrossover(),
        mutation_method=NoMutation(),
        fitness_func=fitness_func,
        minimize=True,
    )

    lgp.run(generations=1)
    assert lgp.best_fitness == 1
    assert lgp.all_time_best_fitness == 1
    assert lgp.population == [((0, 0, 0, 0),) for _ in range(10)]

    lgp.population = [((0, 0, 0, 0),) * i for i in range(10)]

    lgp.run(generations=1)
    assert lgp.best_fitness == 0
    assert lgp.all_time_best_fitness == 0
    assert lgp.population == [tuple() for _ in range(10)]

    lgp.population = [((0, 0, 0, 0),) * (i + 1) for i in range(10)]

    lgp.run(generations=1)
    assert lgp.best_fitness == 1
    assert lgp.all_time_best_fitness == 0
    assert lgp.population == [((0, 0, 0, 0),) for _ in range(10)]
