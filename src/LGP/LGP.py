from tqdm import trange
import numpy as np
from typing import Optional, Callable
import operator

from ._typing import Chromosome
from LGP.selection import SelectionBase
from LGP.crossover import CrossoverBase
from LGP.mutation import MutationBase
from LGP.fitness import FitnessBase


class LGP:
    """
    Implementation of Linear Genetic Programming
    
    Parameters:
    - population:           List of chromosomes
    - selection_method:     Instance of a Selection class
    - crossover_method:     Instance of a Crossover class
    - mutation_method:      Instance of a Mutation class
    - fitness_func:         Fitness function
    - minimize (bool):      True if the algorithm should minimize the fitness function
    - elitism (bool):       Turn on elitism
    - len_punishment:       Punish long chromosomes. This parameter decides how much better a chromosome that is twice as long must be to be considered equal
    """

    def __init__(
            self, 
            population: list[Chromosome], 
            selection_method: SelectionBase,
            crossover_method: CrossoverBase,
            mutation_method: MutationBase | list[MutationBase],
            fitness_func: FitnessBase,
            minimize: bool = False,
            elitism: bool = False,
            len_punishment: float = 0.0,
    ) -> None:
        self.population = population
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method if isinstance(mutation_method, list) else [mutation_method]
        self.fitness_func = fitness_func
        self.minimize = minimize
        self.elitism = elitism
        self.len_punishment = len_punishment

        self.population_size = len(population)

        self.best_fitness: Optional[float] = None
        self.best_individual: Chromosome = tuple()

        self.all_time_best_fitness: Optional[float] = None
        self.all_time_best_individual: Chromosome = tuple()

        self.avg_fitness_log = []
        self.best_fitness_log = []

        # Callbacks
        self.new_best_callback: list[Callable[[Chromosome], None]] = []
        self.generation_callback: list[Callable[[LGP], None]] = []

    def add_new_best_callback(self, callback: Callable[[Chromosome], None]) -> None:
        self.new_best_callback.append(callback)

    def add_generation_callback(self, callback: Callable[["LGP"], None]) -> None:
        self.generation_callback.append(callback)

    def _log(self, fitness: list[float]) -> None:
        """
        Log the results
        """
        # Different operators depending on minimize of maximize
        best_index_op = np.argmin if self.minimize else np.argmax
        fitness_comparison = operator.__lt__ if self.minimize else operator.__gt__

        # Get the best individual in the current population
        best_fitness_index = best_index_op(fitness)
        self.best_fitness = fitness[best_fitness_index]
        self.best_individual = self.population[best_fitness_index]

        # Update the all time best individual
        if self.all_time_best_fitness is None or fitness_comparison(self.best_fitness, self.all_time_best_fitness):
            self.all_time_best_fitness = self.best_fitness
            self.all_time_best_individual = self.best_individual

            for callback in self.new_best_callback:
                callback(self.all_time_best_individual)

        # Save info to logs
        avg_fitness = np.mean(fitness)
        self.avg_fitness_log.append(avg_fitness)
        self.best_fitness_log.append(self.best_fitness)

    def _mutate(self, chromosome: Chromosome) -> Chromosome:
        for mutation in self.mutation_method:
            chromosome = mutation.mutate(chromosome)
        return chromosome

    def run(self, generations: int) -> Chromosome:
        pbar = trange(generations, desc="Best fitness: ???")

        for g in pbar:
            fitness = self.fitness_func(self.population)
            self._log(fitness)
            pbar.desc = f"Best fitness: {self.all_time_best_fitness:0.2f}"
            
            new_population = []

            if self.elitism:
                self.population.append(self.all_time_best_individual)
                self.population.append(self._mutate(self.all_time_best_individual))
                self.population.append(self.best_individual)
                self.population.append(self._mutate(self.best_individual))

            # Negate the fitness if minimize is true
            if self.minimize:
                fitness = [-f for f in fitness]

            # Add some punishment for longer chromosomes
            fitness = [f - self.len_punishment * len(c) for f, c in zip(fitness, self.population)]

            while len(new_population) < self.population_size:
                # Select parents
                parent1_index = self.selection_method.select(fitness)
                parent2_index = self.selection_method.select(fitness)

                parent1 = self.population[parent1_index]
                parent2 = self.population[parent2_index]

                # Generate offspring
                offspring1, offspring2 = self.crossover_method.crossover(parent1, parent2)
                
                # Mutate offspring
                offspring1 = self._mutate(offspring1)
                offspring2 = self._mutate(offspring2)

                # Add to the new population
                new_population.append(offspring1)
                new_population.append(offspring2)

            self.population = new_population

            for callback in self.generation_callback:
                callback(self)
        
        return self.all_time_best_individual
