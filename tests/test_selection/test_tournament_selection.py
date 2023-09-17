import pytest
from LGP.selection import TournamentSelection


@pytest.mark.parametrize(
        ("pTour", "size"),
        (
            (-1, 2),
            (2, 2),
            (0.5, -1),
            (0.5, 0),
        )
)
def test_invalid_init(pTour, size):
    with pytest.raises(AssertionError):
        TournamentSelection(pTour, size)


def test_selection_of_best_individual(mocker):
    """
    With a high pTour the best individual should always be returned
    """
    tournamentSize = 5
    tournament = TournamentSelection(1, tournamentSize)

    population_fitness = list(range(tournamentSize))
    mocker.patch("LGP.selection.random.choices", return_value=list(range(tournamentSize)))

    assert tournament.select(population_fitness) == tournamentSize - 1


def test_selection_of_worst_individual(mocker):
    """
    With a low pTour the worst individual should always be returned
    """
    tournamentSize = 5
    tournament = TournamentSelection(0, tournamentSize)

    population_fitness = list(range(tournamentSize))
    mocker.patch("LGP.selection.random.choices", return_value=list(range(tournamentSize)))

    assert tournament.select(population_fitness) == 0


def test_selection_of_middle_individual(mocker):
    """
    Test if the selection returns an average individual if the random sequenze allows for it
    """
    tournamentSize = 5
    tournament = TournamentSelection(0.5, tournamentSize)

    population_fitness = list(range(tournamentSize))
    random_seq = [0.8, 0.7, 0.2]

    mocker.patch("LGP.selection.random.choices", return_value=list(range(tournamentSize)))
    mocker.patch("LGP.selection.random.random", side_effect=random_seq)

    assert tournament.select(population_fitness) == tournamentSize - len(random_seq)
