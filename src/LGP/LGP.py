from tqdm import trange
import numpy as np
from typing import Optional
import operator

from ._typing import Chromosome, Fitness
from LGP.selection import SelectionBase
from LGP.crossover import CrossoverBase
from LGP.mutation import MutationBase


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
    """

    def __init__(
            self, 
            population: list[Chromosome], 
            selection_method: SelectionBase,
            crossover_method: CrossoverBase,
            mutation_method: MutationBase,
            fitness_func: Fitness,
            minimize: bool = False,
            elitism: bool = False,
    ) -> None:
        self.population = population
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.fitness_func = fitness_func
        self.minimize = minimize
        self.elitism = elitism

        self.population_size = len(population)

        self.best_fitness: Optional[float] = None
        self.best_individual: Chromosome = tuple()

        self.all_time_best_fitness: Optional[float] = None
        self.all_time_best_individual: Chromosome = tuple()

        self.avg_fitness_log = []
        self.best_fitness_log = []

    def _calc_fitness(self) -> list[float]:
        """
        Calculate the fitness function for every individual
        """
        fitness = []
        for individual in self.population:
            f = self.fitness_func(individual)
            fitness.append(f)
        return fitness


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

        # Save info to logs
        avg_fitness = np.mean(fitness)
        self.avg_fitness_log.append(avg_fitness)
        self.best_fitness_log.append(self.best_fitness)

    def run(self, generations: int) -> Chromosome:
        pbar = trange(generations, desc="Best fitness: ???")

        for g in pbar:
            fitness = self._calc_fitness()
            self._log(fitness)
            pbar.desc = f"Best fitness: {self.all_time_best_fitness:0.2f}"
            
            new_population = []

            if self.elitism:
                self.population.append(self.all_time_best_individual)
                self.population.append(self.mutation_method.mutate(self.all_time_best_individual))
                self.population.append(self.best_individual)
                self.population.append(self.mutation_method.mutate(self.best_individual))

            # Negate the fitness if minimize is true
            if self.minimize:
                fitness = [-f for f in fitness]

            while len(new_population) < self.population_size:
                # Select parents
                parent1_index = self.selection_method.select(fitness)
                parent2_index = self.selection_method.select(fitness)

                parent1 = self.population[parent1_index]
                parent2 = self.population[parent2_index]

                # Generate offspring
                offspring1, offspring2 = self.crossover_method.crossover(parent1, parent2)
                
                # Mutate offspring
                offspring1 = self.mutation_method.mutate(offspring1)
                offspring2 = self.mutation_method.mutate(offspring2)

                # Add to the new population
                new_population.append(offspring1)
                new_population.append(offspring2)

            self.population = new_population
        
        return self.all_time_best_individual
