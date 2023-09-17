from abc import ABC, abstractmethod
import random

from ._typing import Chromosome, Instruction


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