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