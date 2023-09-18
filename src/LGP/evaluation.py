import math

from ._typing import Operator, Chromosome


class Operators:

    @staticmethod
    def Add(x: float, y: float) -> float:
        return x + y

    @staticmethod
    def Sub(x: float, y: float) -> float:
        return x - y

    @staticmethod
    def Mult(x: float, y: float) -> float:
        return x * y

    @staticmethod
    def Div(x: float, y: float) -> float:
        if y == 0:
            return 10_000_000
        return x / y

    @staticmethod
    def Sin(x: float, y: float) -> float:
        return math.sin(x)

    @staticmethod
    def Cos(x: float, y: float) -> float:
        return math.cos(x)

    @staticmethod
    def Sqrt(x: float, y: float) -> float:
        return math.sqrt(abs(x))

    @staticmethod
    def Atan2(x: float, y: float) -> float:
        return math.atan2(y, x)


class Register:
    """
    The register is used for the evaluation of a Chromosome
    
    Parameters:
    - varReg:           The variable register
    - constReg:         The constant register
    """

    def __init__(self, varReg: list[float], constReg: list[float]) -> None:
        self.nVar = len(varReg)
        self.nConst = len(constReg)

        self.varReg = varReg
        self.constReg = constReg

    def __getitem__(self, index: int) -> float:
        """
        Get an item from the register
        """
        if index < self.nVar:
            return self.varReg[index]
        return self.constReg[index - self.nVar]
    
    def __setitem__(self, index: int, val: float):
        """
        Set an item in the register
        """
        self.varReg[index] = val


def evaluate(chromosome: Chromosome, operations: list[Operator], varReg: list[float], constReg: list[float]) -> list[float]:
    register = Register(varReg, constReg)
    for operandIndex1, operandIndex2, operatorIndex, destinationIndex in chromosome:
        op1 = register[operandIndex1]
        op2 = register[operandIndex2]
        operator = operations[operatorIndex]
        register[destinationIndex] = operator(op1, op2)
        
    return register.varReg
