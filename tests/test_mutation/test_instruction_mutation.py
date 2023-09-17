from LGP.mutation import InstructionMutation


def test_high_mutation_rate(mocker):
    mocker.patch("LGP.mutation.random.randint", return_value=1)

    chromosome = tuple((0, 0, 0, 0) for _ in range(10))

    mutation = InstructionMutation(1.0, 3, 3, 3)

    new_chromosome = mutation.mutate(chromosome)
    assert new_chromosome != chromosome
    assert len(new_chromosome) == len(chromosome)


def test_low_mutation_rate(mocker):
    mocker.patch("LGP.mutation.random.randint", return_value=1)

    chromosome = tuple((0, 0, 0, 0) for _ in range(10))

    mutation = InstructionMutation(0.0, 3, 3, 3)

    new_chromosome = mutation.mutate(chromosome)
    assert new_chromosome == chromosome
    assert len(new_chromosome) == len(chromosome)


def test_mutation_variation():

    chromosome = tuple((-1, -1, -1, -1) for _ in range(10_000))

    mutation = InstructionMutation(1.0, 5, 5, 5)

    new_chromosome = mutation.mutate(chromosome)

    operand1_set = set(instruction[0] for instruction in new_chromosome if instruction[0] != -1)
    operand2_set = set(instruction[1] for instruction in new_chromosome if instruction[1] != -1)
    operator_set = set(instruction[2] for instruction in new_chromosome if instruction[2] != -1)
    destination_set = set(instruction[3] for instruction in new_chromosome if instruction[3] != -1)

    assert operand1_set == set(range(10))
    assert operand2_set == set(range(10))
    assert operator_set == set(range(5))
    assert destination_set == set(range(5))

    assert len(new_chromosome) == len(chromosome)