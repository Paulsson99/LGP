from typing import Callable

Instruction = tuple[int, int, int, int]
Chromosome = tuple[Instruction,...]
Operator = Callable[[float, float], float]
Fitness = Callable[[Chromosome], float]