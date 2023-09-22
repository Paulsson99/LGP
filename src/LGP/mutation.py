from abc import ABC, abstractmethod
import random
from typing import Callable, Optional

from LGP._typing import Chromosome, Instruction
from LGP.population import random_instruction


DecayFunction = Callable[[int], float]


# Decay functions for the mutation rate
def linear_decay(pMax: float, pMin: float, rate: float, offset: int = 0) -> DecayFunction:
    """
    A linear decay function between pMax and pMin
    
    Parameters:
    - pMax (float):     The maximum mutation probability
    - pMin (float):     The minimum mutation probability
    - rate (float):     The rate of decay measured in decay / generation
    - offset (int):     Delay the decay until this generation
    
    Returns:
    - decay function:   A function that can be called with a generation number to give the corresponding mutation rate
    """
    def _linear_decay(generation: int) -> float:
        if generation < offset:
            return pMax
        
        pMutate = pMax - rate * (generation + 1)
        if pMutate < pMin:
            return pMin
        return pMutate
    
    return _linear_decay


class MutationBase(ABC):

    def __init__(self, pMutate: float, nVar: int, nConst: int, nOp: int, update_func: Optional[DecayFunction]) -> None:
        super().__init__()
        self.pMutate = pMutate
        self.nVar = nVar
        self.nConst = nConst
        self.nTot = nVar + nConst
        self.nOp = nOp

        if update_func is None:
            update_func = lambda x: pMutate

        self.update_func = update_func

    @abstractmethod
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Mutate a chromosome

        Parameters:
        - chromosome: A chromosome

        Returns:
        - chromosome: A new and mutated chromosome
        """

    def update(self, generation: int) -> None:
        """
        Update the mutation depending on the generation number
        """
        self.pMutate = self.update_func(generation)


class InstructionMutation(MutationBase):
    """
    Mutate any instruction on a Chromosome

    Paramters:
    - pMutate (float):      The probability to mutate a random instruction
    - nVar (int):           The number of variable registers
    - nConst (int):         The number of constant registers
    - nOp (int):            The number of operators
    """

    def __init__(self, pMutate: float, nVar: int, nConst: int, nOp: int, update_func: Optional[DecayFunction] = None) -> None:
        super().__init__(pMutate, nVar, nConst, nOp, update_func)

    def mutate(self, chromosome: Chromosome) -> Chromosome:
        def _mutate_intruction(intruction: Instruction) -> Instruction:
            r = random.random()
            if r < 0.25:
                return random.randint(0, self.nTot - 1), intruction[1], intruction[2], intruction[3]
            elif r < 0.5:
                return intruction[0], random.randint(0, self.nTot - 1), intruction[2], intruction[3]
            elif r < 0.75:
                return intruction[0], intruction[1], random.randint(0, self.nOp - 1), intruction[3]
            else:
                return intruction[0], intruction[1], intruction[2], random.randint(0, self.nVar - 1)
            
        return tuple(_mutate_intruction(instruction) if random.random() < self.pMutate else instruction for instruction in chromosome)


class InsertMutation(MutationBase):

    def __init__(self, pInsert: float, nVar: int, nConst: int, nOp: int, update_func: Optional[DecayFunction] = None, max_len: Optional[int] = None) -> None:
        super().__init__(pInsert, nVar, nConst, nOp, update_func)
        self.max_len = max_len

    def mutate(self, chromosome: Chromosome) -> Chromosome:
        if self.max_len is not None and len(chromosome) > self.max_len:
            return chromosome
        
        new_chromosome = []
        for instruction in chromosome:
            if random.random() < self.pMutate:
                new_chromosome.append(random_instruction(self.nVar, self.nConst, self.nOp))
            new_chromosome.append(instruction)

        # Insert at the end
        if random.random() < self.pMutate:
            new_chromosome.append(random_instruction(self.nVar, self.nConst, self.nOp))
        
        return tuple(new_chromosome)


class DeleteMutation(MutationBase):

    def __init__(self, pDelete: float, nVar: int, nConst: int, nOp: int, update_func: Optional[DecayFunction] = None, min_len: Optional[int] = None) -> None:
        super().__init__(pDelete, nVar, nConst, nOp, update_func)
        self.min_len = min_len

    def mutate(self, chromosome: Chromosome) -> Chromosome:
        if self.min_len is not None and len(chromosome) < self.min_len:
            return chromosome
        
        new_chromosome = []
        for instruction in chromosome:
            if random.random() < self.pMutate:
                continue
            new_chromosome.append(instruction)

        return tuple(new_chromosome)
