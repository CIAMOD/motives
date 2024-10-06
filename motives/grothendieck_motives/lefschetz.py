import sympy as sp

from .symbol import Symbol

class Lefschetz(Symbol):
    """
    A symbol in an expression tree that represents a Lefschetz motive.

    Parameters:
    -----------
    parent : Node
        The parent node of the Lefschetz motive.
    id_ : hashable
        Because the Lefschetz motive is unique, the id is always "Lefschetz". It should
        not be changed.

    Methods:
    --------
    sigma(degree: int) -> ET
        Applies the sigma operation to the current Lefschetz motive.
    lambda_(degree: int) -> ET
        Applies the lambda operation to the current Lefschetz motive.
    adams(degree: int) -> ET
        Applies the adams operation to the current Lefschetz motive.

    Properties:
    -----------
    sympy : sp.Symbol
        The sympy representation of the Lefschetz motive.

    Attributes:
    -----------
    L_VAR : sympy.Symbol
        The value of the Lefschetz motive.
    """

    L_VAR: sp.Symbol = sp.Symbol("L")

    def __init__(self, parent=None, id_="Lefschetz"):
        id_ = "Lefschetz"
        super().__init__(self.L_VAR, parent, id_)