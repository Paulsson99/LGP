from abc import ABC, abstractmethod
import random


class SelectionBase(ABC):
    """
    The selection class is responsible for selecting one individual at a time based on its fitness
    """

    @abstractmethod
    def select(self, fitness: list[float]) -> int:
        """
        Select an individual and return the index of it

        Parameters:
        - fitness (list[float]):    The fitness for each individual in the populaiton

        Returns:
        - winner (int):             The winner of the selection
        """


class TournamentSelection(SelectionBase):
    """
    A class representing tournament selection in a genetic algorithm.
    
    Parameters:
    - pTour (float):    The probability to choose the best individual in each tournament
    - size (int):       The number of individuals to compete in each tournament
    """

    def __init__(self, pTour: float, size: int = 2) -> None:
        super().__init__()
        
        assert 0.0 <= pTour <= 1.0
        assert size > 0
        
        self.pTour = pTour
        self.size = size

    def select(self, fitness: list[float]) -> int:
        # Select the individuals to participate in the tournament
        tournament_indecies = random.choices(range(len(fitness)), k=self.size)

        # Sort them according to fitness (best last)
        tournament_indecies.sort(key=lambda i: fitness[i])

        while len(tournament_indecies) > 1:
            # Get the best remaning individual
            best_individual = tournament_indecies.pop()

            # Return the best individual with probability pTour
            if random.random() < self.pTour:
                return best_individual
        
        # Return the worst individual with probability (1 - pTour)^size
        return tournament_indecies[0]
