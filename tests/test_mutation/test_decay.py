import pytest

from LGP.mutation import InstructionMutation, linear_decay


@pytest.mark.parametrize(
        "generations", (1, 2, 3, 4, 5)
)
def test_linear_decay_rate(generations):
    """
    Check that the mutation is down to pMin after generations
    """
    rate = 1.0 / generations
    decay_func = linear_decay(1.0, 0.0, rate)
    decayMut = InstructionMutation(1.0, 3, 3, 3, update_func=decay_func)

    for g in range(generations):
        assert decayMut.pMutate > 0.0
        decayMut.update(g)

    assert decayMut.pMutate == 0.0


@pytest.mark.parametrize(
        "generations", (0, 1, 2, 3, 4, 5)
)
def test_decay_offset(generations):
    """
    Check that the mutation is offset correctly
    """
    rate = 1.0
    decay_func = linear_decay(1.0, 0.0, rate, offset=generations)
    decayMut = InstructionMutation(1.0, 3, 3, 3, update_func=decay_func)

    g = 0

    for g in range(generations):
        decayMut.update(g)
        assert decayMut.pMutate == 1.0
    decayMut.update(g + 1)
    assert decayMut.pMutate == 0.0
