from abc import ABC, abstractmethod
import random

from LGP._typing import Chromosome

from ._typing import Chromosome


class CrossoverBase(ABC):

    @abstractmethod
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> tuple[Chromosome, Chromosome]:
        """
        Perform crossover between two parents
        """


class TwoPointCrossover(CrossoverBase):
    """
    Perform two point crossover between two individuals
    
    1111 | 22 | 333
    44 | 555 | 666666
    ->
    1111 | 555 | 333
    44 | 22 | 666666

    Paramters:
    - pCross (float):       The probability to perform crossover
    - max_length (int):     The maximum lenght of a chromosome
    """

    def __init__(self, pCross: float, max_length: int) -> None:
        super().__init__()
        assert 0.0 <= pCross <= 1.0
        assert max_length > 0

        self.pCross = pCross
        self.max_length = max_length

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> tuple[Chromosome, Chromosome]:
        if random.random() > self.pCross:
            return parent1, parent2
        
        len1 = len(parent1)
        len2 = len(parent2)

        # Crossover points in parent 1
        p11 = random.randint(0, len1)
        p12 = random.randint(0, len1)
        p11, p12 = sorted((p11, p12))
        # Crossover points in parent 2
        p21 = random.randint(0, len2)
        p22 = random.randint(0, len2)
        p21, p22 = sorted((p21, p22))

        offspring1 = parent1[:p11] + parent2[p21:p22] + parent1[p12:]
        offspring2 = parent2[:p21] + parent1[p11:p12] + parent2[p22:]

        if len(offspring1) > self.max_length:
            offspring1 = offspring1[:self.max_length]

        if len(offspring2) > self.max_length:
            offspring2 = offspring2[:self.max_length]

        return offspring1, offspring2
