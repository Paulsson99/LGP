from abc import ABC, abstractmethod
import random

from LGP._typing import Chromosome, Instruction
from LGP.population import random_instruction


class MutationBase(ABC):

    @abstractmethod
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Mutate a chromosome

        Parameters:
        - chromosome: A chromosome

        Returns:
        - chromosome: A new and mutated chromosome
        """


class InstructionMutation(MutationBase):
    """
    Mutate any instruction on a Chromosome

    Paramters:
    - pMutate (float):      The probability to mutate a random instruction
    - nVar (int):           The number of variable registers
    - nConst (int):         The number of constant registers
    - nOp (int):            The number of operators
    """

    def __init__(self, pMutate: float, nVar: int, nConst: int, nOp: int) -> None:
        super().__init__()
        self.pMutate = pMutate
        self.nVar = nVar
        self.nConst = nConst
        self.nTot = nVar + nConst
        self.nOp = nOp

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

    def __init__(self, pInsert: float, nVar: int, nConst: int, nOp: int) -> None:
        super().__init__()
        self.pInsert = pInsert
        self.nVar = nVar
        self.nConst = nConst
        self.nTot = nVar + nConst
        self.nOp = nOp

    def mutate(self, chromosome: Chromosome) -> Chromosome:
        new_chromosome = []
        for instruction in chromosome:
            if random.random() < self.pInsert:
                new_chromosome.append(random_instruction(self.nVar, self.nConst, self.nOp))
            new_chromosome.append(instruction)

        # Insert at the end
        if random.random() < self.pInsert:
            new_chromosome.append(random_instruction(self.nVar, self.nConst, self.nOp))
        
        return tuple(new_chromosome)


class DeleteMutation(MutationBase):

    def __init__(self, pDelete: float, nVar: int, nConst: int, nOp: int) -> None:
        super().__init__()
        self.pDelete = pDelete
        self.nVar = nVar
        self.nConst = nConst
        self.nTot = nVar + nConst
        self.nOp = nOp

    def mutate(self, chromosome: Chromosome) -> Chromosome:
        new_chromosome = []
        for instruction in chromosome:
            if random.random() < self.pDelete:
                continue
            new_chromosome.append(instruction)

        return tuple(new_chromosome)


class MultipleMutations(MutationBase):

    def __init__(self, *mutations: MutationBase) -> None:
        super().__init__()
        self.mutations = mutations
    
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        for mutation in self.mutations:
            chromosome = mutation.mutate(chromosome)
        return chromosome
