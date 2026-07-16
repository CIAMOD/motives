import sympy as sp

from .power_operations import Wedge


import sympy as sp

from motives.core.lambda_ring_expr import LambdaRingExpr


import sympy as sp

from motives.core.lambda_ring_expr import LambdaRingExpr


class Dual(LambdaRingExpr, sp.Function):
    """
    Formal dual of a vector-bundle expression.

    ``Dual(E)`` represents the dual bundle ``E^∨``. It is stored as an
    operation node whose child is the original vector-bundle expression.

    Dualization satisfies

    ``Dual(Dual(E)) = E``.

    If the homogeneous Chern-character components of ``E`` are
    ``ch_i(E)``, then

    ``ch_i(E^∨) = (-1)^i ch_i(E)``.

    Parameters
    ----------
    operand
        Vector bundle or vector-bundle expression whose dual is constructed.
    """

    nargs = 1

    @classmethod
    def eval(cls, operand):
        """
        Apply immediate canonical simplifications.

        Double dualization is removed according to

        ``(E^∨)^∨ = E``.

        Scalar expressions are left unchanged by dualization.

        Parameters
        ----------
        operand
            Expression to dualize.

        Returns
        -------
        sympy.Expr or None
            Simplified expression when an immediate simplification applies,
            or ``None`` to keep the formal ``Dual`` node.
        """
        if operand.is_Number:
            return operand

        if isinstance(operand, cls):
            return operand.child

        return None

    @property
    def child(self) -> sp.Expr:
        """
        Return the expression whose dual is represented.

        Returns
        -------
        sympy.Expr
            Operand stored inside the dual node.
        """
        return self.args[0]

    @property
    def scheme(self):
        """
        Return the base scheme of the operand when available.

        Returns
        -------
        Scheme or None
            Scheme associated with the operand, or ``None`` when it cannot
            be determined directly.
        """
        return getattr(self.child, "scheme", None)

    @property
    def rank(self):
        """
        Return the rank of the dual bundle when available.

        Dualization preserves rank:

        ``rank(E^∨) = rank(E)``.

        Returns
        -------
        sympy.Expr or None
            Rank of the operand, or ``None`` when it cannot be determined.
        """
        return getattr(self.child, "rank", None)

    @property
    def max_degree(self):
        """
        Return the maximum characteristic degree of the operand.

        Returns
        -------
        int or None
            Maximum degree associated with the operand.
        """
        return getattr(self.child, "max_degree", None)

    def _sympystr(self, printer) -> str:
        """
        Return the plain-text representation ``Dual(E)``.

        The explicit function notation is used in plain text to avoid
        confusing dualization with the multiplication operator ``*``.
        """
        return f"Dual({printer.doprint(self.child)})"

    def _latex(self, printer) -> str:
        """
        Return the LaTeX representation ``E^\\vee``.
        """
        return rf"\left({printer._print(self.child)}\right)^\vee"


def Hom(E: sp.Expr, F: sp.Expr) -> sp.Expr:
    """
    Construct the homomorphism bundle from ``E`` to ``F``.

    Mathematically,

    ``Hom(E, F) = E^∨ ⊗ F``.

    In the symbolic K-theory representation, multiplication denotes tensor
    product, so this function returns ``Dual(E) * F``.

    Parameters
    ----------
    E
        Source vector bundle or vector-bundle expression.
    F
        Target vector bundle or vector-bundle expression.

    Returns
    -------
    sympy.Expr
        Symbolic expression representing ``E^∨ ⊗ F``.
    """
    E = sp.sympify(E)
    F = sp.sympify(F)

    return Dual(E) * F


def End(E: sp.Expr) -> sp.Expr:
    """
    Construct the endomorphism bundle of ``E``.

    Mathematically,

    ``End(E) = Hom(E, E) = E^∨ ⊗ E``.

    Parameters
    ----------
    E
        Vector bundle or vector-bundle expression.

    Returns
    -------
    sympy.Expr
        Symbolic expression representing ``E^∨ ⊗ E``.
    """
    E = sp.sympify(E)

    return Hom(E, E)


def Det(E: sp.Expr) -> sp.Expr:
    """
    Construct the determinant line bundle of ``E``.

    For a vector bundle of rank ``r``,

    ``Det(E) = Λ^r(E)``.

    The rank must currently be a known non-negative integer because the
    determinant is constructed using the existing formal exterior-power
    operation ``Wedge(r, E)``.

    Parameters
    ----------
    E
        Vector bundle whose determinant is constructed. It must have a
        ``rank`` attribute containing a known non-negative integer.

    Returns
    -------
    sympy.Expr
        The determinant line bundle ``Λ^rank(E)(E)``.

    Raises
    ------
    TypeError
        If ``E`` does not have a ``rank`` attribute.
    ValueError
        If the rank is not a known non-negative integer.
    """
    E = sp.sympify(E)
    rank = getattr(E, "rank", None)

    if rank is None:
        raise TypeError("Det currently requires a vector bundle with a rank attribute.")

    rank = sp.sympify(rank)

    if not isinstance(rank, sp.Integer):
        raise ValueError("Det requires the rank of the vector bundle to be a known integer.")

    rank = int(rank)

    if rank < 0:
        raise ValueError("The rank of a vector bundle must be non-negative.")

    if rank == 0:
        return sp.Integer(1)

    if rank == 1:
        return E

    return Wedge(rank, E)