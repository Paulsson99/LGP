import pytest

from LGP.evaluation import Register


def test_set_var_register():
    r = Register([-1] * 10, [1] * 10)
    
    for i in range(10):
        r[i] = i
    
    assert r.varReg == list(range(10))
    assert r.constReg == [1] * 10


def test_set_const_register():
    r = Register([-1] * 10, [1] * 10)
    
    with pytest.raises(IndexError):
        r[10] = 10
    
    assert r.varReg == [-1] * 10
    assert r.constReg == [1] * 10


def test_get_register():
    r = Register(list(range(10)), list(range(10, 20)))
    
    for i in range(20):
        assert r[i] == i
