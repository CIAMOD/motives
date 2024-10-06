from typing import Hashable, Optional

from ..core.operand import Operand
from ..core.node import Node

class Motive(Operand):
    """
    An abstract operand node in an expression tree that represents a motive.

    Parameters:
    -----------
    parent : Node
        The parent node of the motive.
    id_ : hashable
        The id of the motive, used to identify it (if two operands
        have the same id, they are the same operand).

    Methods:
    --------
    sigma(degree: int) -> ET
        Applies the sigma operation to the current motive.
    lambda_(degree: int) -> ET
        Applies the lambda operation to the current motive.
    adams(degree: int) -> ET
        Applies the adams operation to the current motive.

    Properties:
    -----------
    sympy : sp.Symbol
        The sympy representation of the motive.
    """

    def __init__(self, parent: Optional[Node] = None, id_: Hashable = None):
        super().__init__(parent, id_)