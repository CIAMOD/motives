from motives.core.operator.ring_operator import Sigma, Lambda_


class Wedge(Sigma):
    """
    Formal exterior-power expression.

    ``Wedge(n, E)`` represents the n-th exterior power ``Λⁿ(E)``.

    The class inherits from ``Sigma`` because the current conversion machinery
    uses ``Sigma`` nodes as the internal backend for exterior-power expansions.
    This inheritance is an implementation detail and should not be interpreted
    as a mathematical identification of exterior powers with the usual sigma
    operation.

    Parameters
    ----------
    degree
        Exterior-power degree.
    child
        Expression to which the exterior power is applied.

    Notes
    -----
    The arguments are stored using the underlying ring-operator convention and
    can be accessed through ``degree`` and ``child``.
    """

    def _sympystr(self, printer):
        """Return the plain-text representation ``∧n(operand)``."""
        degree, operand = self.args
        return f"∧{printer.doprint(degree)}({printer.doprint(operand)})"

    def _latex(self, printer):
        """Return the LaTeX representation of the exterior-power expression."""
        degree, operand = self.args
        return (
            r"\wedge^{%s}\left(%s\right)"
            % (printer._print(degree), printer._print(operand))
        )


class SymPower(Lambda_):
    """
    Formal symmetric-power expression.

    ``SymPower(n, E)`` represents the n-th symmetric power ``Symⁿ(E)``.

    The class inherits from ``Lambda_`` because the current conversion
    machinery uses ``Lambda_`` nodes as the internal backend for
    symmetric-power expansions. This inheritance is an implementation detail,
    not a mathematical identification with the usual exterior-power
    lambda operation.

    Parameters
    ----------
    degree
        Symmetric-power degree.
    child
        Expression to which the symmetric power is applied.

    Notes
    -----
    The arguments are stored using the underlying ring-operator convention and
    can be accessed through ``degree`` and ``child``.
    """

    def _sympystr(self, printer) -> str:
        """Return the plain-text representation ``Symn(operand)``."""
        return f"Sym{self.degree}({printer.doprint(self.child)})"

    def _latex(self, printer) -> str:
        """Return the LaTeX representation of the sym-power expression."""
        degree, operand = self.args
        return (
            r"\operatorname{Sym}^{%s}\left(%s\right)"
            % (printer._print(degree), printer._print(operand))
        )