import numpy as np

from LGP.fitness import MimicTrainingData
from LGP.evaluation import Operators


def test_fitness_call(mocker):
    x = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    y = np.array([[2, 2, 1], [3, 4, 0], [4, 4, 2]])

    fitness_func = MimicTrainingData(x=x, y=y, nVar=6, operators=[Operators.Add], constReg=[1, 2, 3])

    mocker.patch("LGP.fitness.evaluate", return_value=[0, 0, 0])

    fitness = fitness_func([tuple()])

    # Assert the average fitness is correct
    assert fitness == [14 / len(x)]