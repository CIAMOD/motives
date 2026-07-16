import sympy as sp

from motives.core.operator.ring_operator import Lambda_, Sigma

from ..objects.power_bundles import Wedge, SymPower


def wedge(self, n: int):
    """
    Construct the formal n-th exterior power of an expression.

    This method creates a ``Wedge`` node representing

    ``Λⁿ(self)``

    without expanding it using lambda-ring identities.

    Parameters
    ----------
    n : int
        Exterior-power degree.

    Returns
    -------
    sympy.Expr
        ``1`` when ``n = 0``, ``self`` when ``n = 1``, and a formal
        ``Wedge`` object otherwise.

    Raises
    ------
    ValueError
        If ``n`` is negative.

    Notes
    -----
    This method does not use bundle ranks and therefore does not automatically
    impose relations such as ``Λⁿ(E) = 0`` when ``n > rank(E)``.

    Use ``to_wedge`` when an expansion in terms of exterior powers of the
    atomic expressions is required.
    """
    if n < 0:
        raise ValueError("n must be nonnegative")
    if n == 0:
        return sp.Integer(1)
    if n == 1:
        return self

    return Wedge(n, self)


def to_wedge(self, n: int | None = None):
    """
    Expand an exterior power using the lambda-ring conversion machinery.

    When ``n`` is supplied, this method expands ``Λⁿ(self)``. When ``n`` is
    omitted and ``self`` is already a ``Wedge`` object, it expands that formal
    operation.

    For example, the second exterior power of a direct sum satisfies

    ``Λ²(E ⊕ F) = Λ²(E) ⊕ (E ⊗ F) ⊕ Λ²(F)``.

    Parameters
    ----------
    n : int or None, optional
        Exterior-power degree. If omitted, ``self`` must already be a
        ``Wedge`` object in order for an expansion to occur.

    Returns
    -------
    sympy.Expr
        Expanded expression whose remaining formal exterior-power atoms are
        represented by ``Wedge`` objects. If ``n`` is omitted and ``self`` is
        not a ``Wedge`` object, ``self`` is returned unchanged.

    Raises
    ------
    ValueError
        If ``n`` is negative.

    Notes
    -----
    The expansion is computed through the package's internal ``Sigma``
    representation and then relabelled using ``Wedge`` nodes. The relabelling
    changes the representation and printing, not the algebraic expression.
    """
    if n is None:
        if not isinstance(self, Wedge):
            return self

        expr = sp.expand(self.to_sigma())
        return _replace_sigma_by_wedge(expr)

    if n < 0:
        raise ValueError("n must be nonnegative")
    if n == 0:
        return sp.Integer(1)
    if n == 1:
        return self

    expr = sp.expand(self.sigma(n).to_sigma())
    return _replace_sigma_by_wedge(expr)


def _replace_sigma_by_wedge(expr):
    """
    Relabel internal ``Sigma`` nodes as ``Wedge`` nodes.

    Parameters
    ----------
    expr : sympy.Expr
        Expression produced by the sigma-expansion backend.

    Returns
    -------
    sympy.Expr
        Expression in which each ordinary ``Sigma`` node has been replaced by
        a ``Wedge`` node with the same arguments.

    Notes
    -----
    This is a representation-level transformation:

    ``Sigma(n, E) -> Wedge(n, E)``.

    It does not perform an additional mathematical expansion or simplification.
    Existing ``Wedge`` instances are left unchanged.
    """
    return expr.replace(
        lambda x: isinstance(x, Sigma) and not isinstance(x, Wedge),
        lambda x: Wedge(*x.args)
    )


def sym(self, n: int):
    """
    Construct the formal n-th symmetric power of an expression.

    This method creates a ``SymPower`` node representing

    ``Symⁿ(self)``

    without expanding it using lambda-ring identities.

    Parameters
    ----------
    n : int
        Symmetric-power degree.

    Returns
    -------
    sympy.Expr
        ``1`` when ``n = 0``, ``self`` when ``n = 1``, and a formal
        ``SymPower`` object otherwise.

    Raises
    ------
    ValueError
        If ``n`` is negative.

    Notes
    -----
    Use ``to_sym`` when an expansion in terms of symmetric powers of the
    atomic expressions is required.
    """
    if n < 0:
        raise ValueError("n must be nonnegative")
    if n == 0:
        return sp.Integer(1)
    if n == 1:
        return self

    return SymPower(n, self)


def to_sym(self, n: int | None = None):
    """
    Expand a symmetric power using the lambda-ring conversion machinery.

    When ``n`` is supplied, this method expands ``Symⁿ(self)``. When ``n`` is
    omitted and ``self`` is already a ``SymPower`` object, it expands that
    formal operation.

    For example,

    ``Sym²(E ⊕ F) = Sym²(E) ⊕ (E ⊗ F) ⊕ Sym²(F)``.

    Parameters
    ----------
    n : int or None, optional
        Symmetric-power degree. If omitted, ``self`` must already be a
        ``SymPower`` object in order for an expansion to occur.

    Returns
    -------
    sympy.Expr
        Expanded expression whose remaining formal symmetric-power atoms are
        represented by ``SymPower`` objects. If ``n`` is omitted and ``self``
        is not a ``SymPower`` object, ``self`` is returned unchanged.

    Raises
    ------
    ValueError
        If ``n`` is negative.

    Notes
    -----
    The expansion is computed through the package's internal ``Lambda_``
    representation and then relabelled using ``SymPower`` nodes. This is an
    internal implementation convention.
    """
    if n is None:
        if not isinstance(self, SymPower):
            return self

        expr = sp.expand(self.to_lambda())
        return _replace_lambda_by_sym(expr)

    if n < 0:
        raise ValueError("n must be nonnegative")
    if n == 0:
        return sp.Integer(1)
    if n == 1:
        return self

    expr = sp.expand(self.lambda_(n).to_lambda())
    return _replace_lambda_by_sym(expr)


def _replace_lambda_by_sym(expr):
    """
    Relabel internal ``Lambda_`` nodes as ``SymPower`` nodes.

    Parameters
    ----------
    expr : sympy.Expr
        Expression produced by the lambda-expansion backend.

    Returns
    -------
    sympy.Expr
        Expression in which each ordinary ``Lambda_`` node has been replaced
        by a ``SymPower`` node with the same arguments.

    Notes
    -----
    This transformation only changes the node type used for representation:

    ``Lambda_(n, E) -> SymPower(n, E)``.

    It does not perform an additional algebraic operation. Existing
    ``SymPower`` instances are left unchanged.
    """
    return expr.replace(
        lambda x: isinstance(x, Lambda_) and not isinstance(x, SymPower),
        lambda x: SymPower(*x.args)
    )