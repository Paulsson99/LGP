import pytest

from LGP.crossover import TwoPointCrossover


CHROMOSOME_TEST_LENGTH = 100


@pytest.mark.parametrize(
    ("p11", "p12", "p21", "p22"),
    [
        (30, 60, 30, 60),
        (0, 0, 0, 0),
        (100, 100, 100, 100),
        (0, 100, 0, 100),
        (0, 20, 90, 100),
    ]
)
def test_crossover_points(mocker, p11, p12, p21, p22):

    crossover = TwoPointCrossover(1, 200)

    parent1 = [1] * CHROMOSOME_TEST_LENGTH
    parent2 = [2] * CHROMOSOME_TEST_LENGTH

    expected_offspring1 = [1] * p11 + [2] * (p22 - p21) + [1] * (CHROMOSOME_TEST_LENGTH - p12)
    expected_offspring2 = [2] * p21 + [1] * (p12 - p11) + [2] * (CHROMOSOME_TEST_LENGTH - p22)

    # Get the correct random elements
    mocker.patch("LGP.crossover.random.randint", side_effect=[p11, p12, p21, p22])

    offspring1, offspring2 = crossover.crossover(parent1, parent2)

    assert offspring1 == expected_offspring1
    assert offspring2 == expected_offspring2


def test_max_length_during_crossover(mocker):

    crossover = TwoPointCrossover(1, 110)

    parent1 = [1] * CHROMOSOME_TEST_LENGTH
    parent2 = [2] * CHROMOSOME_TEST_LENGTH

    # Get the correct random elements
    mocker.patch("LGP.crossover.random.randint", side_effect=[50, 60, 10, 90])

    offspring1, offspring2 = crossover.crossover(parent1, parent2)

    assert len(offspring1) == 110
    assert len(offspring2) == 30


def test_low_crossover_prob():
    crossover = TwoPointCrossover(0, 200)

    parent1 = [1] * CHROMOSOME_TEST_LENGTH
    parent2 = [2] * CHROMOSOME_TEST_LENGTH

    offspring1, offspring2 = crossover.crossover(parent1, parent2)

    assert offspring1 == parent1
    assert offspring2 == parent2
