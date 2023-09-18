import random

from LGP._typing import Chromosome, Instruction


def random_instruction(nVar: int, nConst: int, nOp: int) -> Instruction:
    """
    Return a random instruction
    """
    nTot = nVar + nConst
    return random.randrange(nTot), random.randrange(nTot), random.randrange(nOp), random.randrange(nVar)


def random_individual(size: int, nVar: int, nConst: int, nOp: int) -> Chromosome:
    """
    Return a random individual
    """
    return tuple(random_instruction(nVar, nConst, nOp) for _ in range(size))


def random_population(population_size: int, min_size: int, max_size: int, nVar: int, nConst: int, nOp: int) -> list[Chromosome]:
    """
    Return a random population
    """
    return [
        random_individual(random.randint(min_size, max_size), nVar, nConst, nOp)
        for _ in range(population_size)
    ]
