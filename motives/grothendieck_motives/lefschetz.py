from __future__ import annotations
import sympy as sp

from .polynomial_1_var import Polynomial1Var

from ..utils import SingletonMeta

class Lefschetz(Polynomial1Var, metaclass=SingletonMeta):
    """
    Represents a Lefschetz motive in an expression tree in the Grothendieck lambda-ring of
    varieties, the Grothendieck rin of Chow motives or in any extension or completion of such rings
    begin considered.

    It represents the motivic class of the complex line L=[A^1].

    The Lefschetz motive is represented as a 1-dimensional object named "L". 
    
    This class ensures that the Lefschetz motive is a singleton, meaning only one instance 
    can exist throughout the program.
    """

    def __new__(cls) -> Lefschetz:
        """
        Creates a new instance of the Lefschetz motive, ensuring singleton behavior.

        The Lefschetz motive is represented by the symbol 'L' in expressions.

        Returns:
        --------
        Lefschetz
            The singleton instance of the `Lefschetz` motive.
        """
        return sp.Symbol.__new__(cls, "L")
