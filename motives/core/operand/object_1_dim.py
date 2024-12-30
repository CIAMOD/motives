import sympy as sp

from .operand import Operand


class Object1Dim(Operand):
    """
    An abstract class representing a one-dimensional object in the expression tree.

    This class inherits from Operand and is meant to be extended by any specific one-dimensional
    objects. It defines the abstract methods for Adams and Lambda operations that must be implemented
    by subclasses.

    Attributes:
    -----------
    name : str
        The name of the one-dimensional object.
    """

    def _sympystr(self, printer: sp.StrPrinter) -> str:
        """
        Provides a string representation of the one-dimensional object for SymPy printing.

        Args:
        -----
        printer : sp.StrPrinter
            The SymPy printer used for formatting the string representation.

        Returns:
        --------
        str
            The string representation of the one-dimensional object.
        """
        return self.__repr__()

    def get_adams_var(self, i: int) -> sp.Expr:
        """
        Returns the one-dimensional object with an Adams operation applied to it.

        Args:
        -----
        i : int
            The degree of the Adams operator to apply.

        Returns:
        --------
        sp.Expr
            The one-dimensional object with the Adams operator applied.
        """
        raise NotImplementedError("This method must be implemented in all subclasses.")

    def get_lambda_var(self, i: int) -> sp.Expr:
        """
        Returns the one-dimensional object with a Lambda operation applied to it.

        Args:
        -----
        i : int
            The degree of the Lambda operator to apply.

        Returns:
        --------
        sp.Expr
            The one-dimensional object with the Lambda operator applied.
        """
        raise NotImplementedError("This method must be implemented in all subclasses.")

    def get_max_adams_degree(self) -> int:
        """
        Computes the maximum Adams degree of this one-dimensional object.

        Returns:
        --------
        int
            The maximum Adams degree of this one-dimensional object.
        """
        return 1

    def get_max_groth_degree(self) -> int:
        """
        Computes the maximum Grothendieck degree of this one-dimensional object.

        Returns:
        --------
        int
            The maximum sigma or lambda degree of this one-dimensional object.
        """
        return 0
