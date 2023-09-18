from abc import ABC, abstractmethod
import numpy as np
from multiprocessing import Pool

from LGP._typing import Chromosome, Operator
from LGP.evaluation import evaluate


class FitnessBase(ABC):

    @abstractmethod
    def __call__(self, populaiton: list[Chromosome]) -> list[float]:
        """
        Calculate the fitness of the population
        
        Parameters:
        - population: The current population
        
        Returns:
        - fitness: The fitness for every individual in the population
        """


class MimicTrainingData(FitnessBase):

    def __init__(self, x: np.ndarray, y: np.ndarray, nVar: int, constReg: list[float], operators: list[Operator]) -> None:
        super().__init__()
        assert len(x.shape) == 2
        assert len(y.shape) == 2

        assert x.shape[0] == y.shape[0]

        self.training_samples = x.shape[0]
        self.input_len = x.shape[1]
        self.output_len = y.shape[1]
        assert self.input_len <= nVar
        assert self.output_len <= nVar

        self.x = x
        self.y = y

        self.nVar = nVar
        self.constReg = constReg
        self.operators = operators

    def __call__(self, populaiton: list[Chromosome]) -> list[float]:
        
        fitness = []
        for individual in populaiton:
            tot_error = 0
            for xp, yp in zip(self.x, self.y):
                varReg = [float(xp[i]) if i < self.input_len else 0.0 for i in range(self.nVar)]
                yh = evaluate(individual, self.operators, varReg, self.constReg)[:self.output_len]

                diff = yp - yh
                error = np.sqrt(np.sum(diff * diff))

                tot_error += error
            
            fitness.append(tot_error / self.training_samples)
        
        return fitness


class MimicTrainingDataMultiProcessing(MimicTrainingData):

    def __init__(self, x: np.ndarray, y: np.ndarray, nVar: int, constReg: list[float], operators: list[Operator], workers: int = 4) -> None:
        super().__init__(x, y, nVar, constReg, operators)
        self.workers = workers

    def fitness(self, individual: Chromosome) -> float:
        tot_error = 0
        for xp, yp in zip(self.x, self.y):
            varReg = [float(xp[i]) if i < self.input_len else 0.0 for i in range(self.nVar)]
            yh = evaluate(individual, self.operators, varReg, self.constReg)[:self.output_len]

            diff = yp - yh
            error = np.sqrt(np.sum(diff * diff))

            tot_error += error
        return tot_error / self.training_samples

    def __call__(self, populaiton: list[Chromosome]) -> list[float]:
        with Pool(processes=self.workers) as pool:
            return pool.map(self.fitness, populaiton)
