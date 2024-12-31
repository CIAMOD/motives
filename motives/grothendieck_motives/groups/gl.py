from .semisimple_g import SemisimpleG


class GL(SemisimpleG):
    """
    Represents a GL_n bundle in an expression tree.

    A GL_n bundle represents the product \prod_{k=0}^{n-1} (L^n - L^k), where L is the Lefschetz motive.
    This class supports operations related to Adams and Lambda transformations.

    Attributes:
    -----------
    n : int
        The dimension of the GL_n bundle.
    lef : Lefschetz
        The Lefschetz motive associated with the bundle.
    _et_repr : sp.Expr
        The GL_n bundle as a sympy expression.
    _lambda_vars : dict[int, sp.Expr]
        A dictionary of the lambda variables generated for this GL_n bundle.
    """

    def __new__(cls, n: int, *args, **kwargs):
        """
        Creates a new instance of the GL group with the specified dimension n.

        Parameters:
        -----------
        n : int
            The dimension of the group GL(n).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns:
        --------
        GL
            A new instance of the GL class.
        """
        new_sl = SemisimpleG.__new__(cls, list(range(2, n + 2)), n**2)
        return new_sl

    def __init__(self, n: int, *args, **kwargs):
        """
        Initializes the A class with the specified dimension n.

        Parameters:
        -----------
        n : int
            The dimension of the group A(n).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(list(range(2, n + 2)), n**2)
        self.n = n

    def __repr__(self) -> str:
        """
        Returns the string representation of the GL_n bundle.

        Returns:
        --------
        str
            A string representation in the form "GL_n".
        """
        return f"GL_{self.n}"

    def _hashable_content(self) -> tuple:
        """
        Returns the hashable content of the GL_n bundle.

        Returns:
        --------
        tuple
            A tuple containing the dimension of the GL_n bundle.
        """
        return (self.n,)
