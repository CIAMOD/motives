from __future__ import annotations
import sympy as sp

from ..polynomial_1_var import Polynomial1Var
from ..utils import SingletonMeta


class Lefschetz(Polynomial1Var, metaclass=SingletonMeta):
    """
    Represents the Lefschetz motive in the Grothendieck lambda-ring of varieties or
    the Grothendieck ring of Chow motives. It is a fundamental object in the
    study of motivic classes, specifically representing the complex line \( \mathbb{A}^1 \).

    This class enforces the Singleton pattern, ensuring that only one instance
    of the Lefschetz motive exists throughout the program.
    """

    is_real = True

    def __new__(cls) -> Lefschetz:
        """
        Creates and returns the singleton instance of the Lefschetz motive.

        The Lefschetz motive is symbolized by 'L' in expressions.

        Returns:
            Lefschetz: The singleton instance representing the Lefschetz motive.
        """
        return sp.Symbol.__new__(cls, "L")
