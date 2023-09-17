import pytest

from LGP.evaluation import evaluate, Operators


@pytest.mark.parametrize(
        ("chromosome", "result"),
        (
            [   # Do nothing
                tuple(),
                [0.0, 0.0, 0.0]
            ],
            [   # 1 + 1 = 2
                ((3, 3, 0, 0),),
                [2.0, 0, 0]
            ],
            [    # 3 * 3 = 9
                ((5, 5, 1, 2),),
                [0.0, 0.0, 9.0]
            ],
        )
)
def test_evaluation(chromosome, result):
    varReg = [0.0, 0.0, 0.0]
    constReg = [1.0, 2.0, 3.0]
    operations = [
        Operators.Add,
        Operators.Mult,
    ]

    assert evaluate(chromosome=chromosome, operations=operations, varReg=varReg, constReg=constReg) == result
