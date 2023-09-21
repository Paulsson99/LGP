import pytest

from LGP.mutation import InstuctionMutationDecay


@pytest.mark.parametrize(
        "generations", (1, 2, 3, 4, 5)
)
def test_decay(generations):
    """
    Check that the mutation is down to pMin after generations
    """
    decayMut = InstuctionMutationDecay(1.0, 0.0, 1.0, 3, 3, 3)

    for g in range(generations):
        decayMut.update(g)

    assert decayMut.pMutate == 0.0