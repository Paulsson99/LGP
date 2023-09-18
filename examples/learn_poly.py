import  numpy as np
import matplotlib.pyplot as plt

from LGP.LGP import LGP
from LGP.fitness import MimicTrainingData
from LGP.selection import TournamentSelection
from LGP.crossover import TwoPointCrossover
from LGP.mutation import InstructionMutation
from LGP.evaluation import Operators
from LGP.population import random_population


# Params
NVAR = 4
CONST_REG = [1.0, 2.0, 3.0]
NCONST = len(CONST_REG)
OPS = [
    Operators.Add,
    Operators.Sub,
    Operators.Mult
]
NOPS = len(OPS)
POPULATION_SIZE = 500
MIN_LEN = 10
MAX_LEN = 100

GENERATIONS = 3000

pTour = 0.8
tournament_size = 4
pCross = 0.6
pMutate = 0.07


# Create the training data
p = [1, 4, -10, -7]
x = np.linspace(-5, 5).reshape((-1, 1))
y = np.polyval(p, x)

# Inspect the training data
fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()

# Set up the training loop
fitness_func = MimicTrainingData(x, y, nVar=NVAR, constReg=CONST_REG, operators=OPS)
selection = TournamentSelection(pTour=pTour, size=tournament_size)
crossover = TwoPointCrossover(pCross=pCross, max_length=2 * MAX_LEN)
mutation = InstructionMutation(pMutate=pMutate, nVar=NVAR, nConst=NCONST, nOp=NOPS)
population = random_population(
    population_size=POPULATION_SIZE,
    min_size=MIN_LEN,
    max_size=MAX_LEN,
    nVar=NVAR,
    nConst=NCONST,
    nOp=NOPS,
)

lgp = LGP(
    population=population,
    selection_method=selection,
    crossover_method=crossover,
    mutation_method=mutation,
    fitness_func=fitness_func,
    minimize=True,
    elitism=True,
)

lgp.run(GENERATIONS)
