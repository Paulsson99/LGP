import  numpy as np
import matplotlib.pyplot as plt

from LGP.LGP import LGP
from LGP.fitness import MimicTrainingData
from LGP.selection import TournamentSelection
from LGP.crossover import TwoPointCrossover
from LGP.mutation import InstructionMutation, InsertMutation, DeleteMutation, linear_decay
from LGP.evaluation import Operators, evaluate
from LGP.population import random_population
from LGP._typing import Chromosome


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
pMutate = 0.7
pInsert = 0.05
pDelete = 0.05


instruction_decay = linear_decay(pMutate, 0.01, pMutate / 500, offset=25)
insertion_decay = linear_decay(pInsert, 0.01, pInsert / 500, offset=25)
deletion_decay = linear_decay(pDelete, 0.01, pDelete / 500, offset=25)


# Create the training data
def training_data():
    p = [1, 4, -10, -7]
    x = np.linspace(-5, 5).reshape((-1, 1))
    y = np.polyval(p, x)
    return x, y


def main():
    x, y = training_data()

    # Inspect the training data
    fig, ax = plt.subplots()
    ax.plot(x, y)
    best_line, = ax.plot(x, [0] * len(x), 'r--')
    plt.show(block=False)

    # Set up the training loop
    fitness_func = MimicTrainingData(x, y, nVar=NVAR, constReg=CONST_REG, operators=OPS)
    selection = TournamentSelection(pTour=pTour, size=tournament_size)
    crossover = TwoPointCrossover(pCross=pCross, max_length=2 * MAX_LEN)

    # Mutation
    instruction_mutation = InstructionMutation(pMutate=pMutate, nVar=NVAR, nConst=NCONST, nOp=NOPS, update_func=instruction_decay)
    insert_mutation = InsertMutation(pInsert=pInsert, nVar=NVAR, nConst=NCONST, nOp=NOPS, update_func=insertion_decay)
    delete_mutation = DeleteMutation(pDelete=pDelete, nVar=NVAR, nConst=NCONST, nOp=NOPS, update_func=deletion_decay)
    mutations = [instruction_mutation, insert_mutation, delete_mutation]

    population = random_population(
        population_size=POPULATION_SIZE,
        min_size=MIN_LEN,
        max_size=MAX_LEN,
        nVar=NVAR,
        nConst=NCONST,
        nOp=NOPS,
    )

    # Update function for the plot
    def update(chromosome: Chromosome):
        y = []
        for xp in x:
            varReg = [float(xp[0])] + [0] * (NVAR - 1)
            out = evaluate(chromosome, OPS, varReg, CONST_REG)
            y.append(out[0])
        best_line.set_ydata(y)
        fig.canvas.draw()
        fig.canvas.flush_events()


    lgp = LGP(
        population=population,
        selection_method=selection,
        crossover_method=crossover,
        mutation_method=mutations,
        fitness_func=fitness_func,
        minimize=True,
        elitism=True,
        len_punishment=0.01
    )

    lgp.add_new_best_callback(update)

    lgp.run(GENERATIONS)


if __name__ == '__main__':
    main()
