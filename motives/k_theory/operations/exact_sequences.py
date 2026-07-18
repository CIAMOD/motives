from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import sympy as sp

from ..chern import chern_character, chern_class
from ..objects.vector_bundle import VectorBundle


Invariant = Literal["chern_character", "chern_classes"]


def _validate_exact_sequence(sequence: Sequence[sp.Expr]) -> tuple[sp.Expr, ...]:
    """
    Validate and normalize the non-zero terms of an exact sequence.

    The elements are interpreted in their given order, with the zero objects
    at both ends omitted. For example, ``[E, F, G]`` represents

        0 -> E -> F -> G -> 0.

    This function checks that the input is an ordered sequence containing at
    least two terms, that at least one ``VectorBundle`` appears in its
    expressions, and that all vector bundles are defined over the same scheme.
    Each term is converted into a SymPy expression.

    Exactness is assumed and cannot be verified because the morphisms between
    the terms are not provided.

    Parameters
    ----------
    sequence : Sequence[sp.Expr]
        Ordered non-zero terms of the exact sequence. Terms may be individual
        vector bundles or SymPy expressions containing vector bundles.

    Returns
    -------
    tuple[sp.Expr, ...]
        The normalized terms as a tuple of SymPy expressions.

    Raises
    ------
    TypeError
        If ``sequence`` is not an ordered sequence.

    ValueError
        If the sequence contains fewer than two terms, contains no vector
        bundles, or includes vector bundles defined over different schemes.
    """
    if isinstance(sequence, (str, bytes)) or not isinstance(sequence, Sequence):
        raise TypeError("sequence must be an ordered sequence of expressions.")

    if len(sequence) < 2:
        raise ValueError("An exact sequence must contain at least two non-zero terms.")

    terms = tuple(sp.sympify(term) for term in sequence)
    bundles: set[VectorBundle] = set()

    for term in terms:
        bundles.update(term.atoms(VectorBundle))

    if not bundles:
        raise ValueError("The exact sequence must contain at least one VectorBundle.")

    reference_scheme = next(iter(bundles)).scheme

    for bundle in bundles:
        if bundle.scheme != reference_scheme:
            raise ValueError("All vector bundles must be defined over the same scheme.")

    return terms


def _get_solve_variables(
    solve_for: VectorBundle | sp.Symbol | Sequence[VectorBundle | sp.Symbol],
    invariant: Invariant,
    component: int | None
) -> tuple[sp.Symbol, ...]:
    """
    Return the symbolic Chern components that must be solved for.

    ``solve_for`` may contain vector bundles or explicit SymPy symbols. A
    vector bundle is expanded into its Chern-character or Chern-class symbols.
    If ``component`` is provided, only that component is selected. Otherwise,
    all symbolic components are selected, excluding ``c_0 = 1`` for Chern
    classes. Numeric components, such as a known rank, are ignored.

    Parameters
    ----------
    solve_for : VectorBundle | sp.Symbol | Sequence[VectorBundle | sp.Symbol]
        Vector bundles or explicit symbols to solve for.

    invariant : {"chern_character", "chern_classes"}
        Invariant whose components are selected for vector-bundle targets.

    component : int | None
        Specific component to select, or ``None`` to select every symbolic
        component.

    Returns
    -------
    tuple[sp.Symbol, ...]
        Ordered tuple of non-repeated symbols passed to ``sympy.solve``.

    Raises
    ------
    TypeError
        If a target is neither a ``VectorBundle`` nor a SymPy symbol.

    ValueError
        If no symbolic variables can be obtained from ``solve_for``.
    """
    if isinstance(solve_for, (VectorBundle, sp.Symbol)):
        targets = (solve_for,)
    elif isinstance(solve_for, Sequence) and not isinstance(solve_for, (str, bytes)):
        targets = tuple(solve_for)
    else:
        raise TypeError("solve_for must be a VectorBundle, a Symbol, or a sequence of them.")

    variables: list[sp.Symbol] = []

    for target in targets:
        if isinstance(target, VectorBundle):
            if invariant == "chern_character":
                values = (target.ch(component),) if component is not None else target.ch()
            else:
                values = (target.c(component),) if component is not None else target.c()[1:]

            for value in values:
                value = sp.sympify(value)

                if isinstance(value, sp.Symbol):
                    variables.append(value)

        elif isinstance(target, sp.Symbol):
            variables.append(target)

        else:
            raise TypeError("Every solve target must be a VectorBundle or Symbol.")

    variables = list(dict.fromkeys(variables))

    if not variables:
        raise ValueError("No symbolic variables were found in solve_for.")

    return tuple(variables)


def exact_sequence_realtions(
    sequence: Sequence[sp.Expr],
    invariant: Invariant = "chern_character",
    component: int | None = None
) -> sp.Equality | tuple[sp.Equality, ...]:
    """
    Construct the Chern equations associated with an exact sequence.

    The input contains only the non-zero terms of the sequence. For example,

        [E, F, G]

    is interpreted as

        0 -> E -> F -> G -> 0.

    The exact sequence determines the alternating relation in K-theory

        [E_0] - [E_1] + [E_2] - [E_3] + ... = 0,

    which is rearranged by placing even-indexed terms on the left and
    odd-indexed terms on the right:

        E_0 + E_2 + ... = E_1 + E_3 + ...

    The selected Chern invariant is then applied to both expressions.

    With ``invariant="chern_character"``, the returned equations express the
    additivity of the Chern character. Degree zero is included and gives the
    corresponding rank equation.

    With ``invariant="chern_classes"``, the equations are obtained from total
    Chern classes. This automatically produces the Whitney product relations,
    such as

        c_1(F) = c_1(E) + c_1(G),

        c_2(F) = c_2(E) + c_1(E)c_1(G) + c_2(G).

    The component ``c_0 = 1`` is omitted when all Chern-class equations are
    requested because it only gives the trivial identity ``1 = 1``.

    Parameters
    ----------
    sequence : Sequence[sp.Expr]
        Ordered non-zero terms of the exact sequence. Terms may be vector
        bundles or SymPy expressions containing vector bundles.

    invariant : {"chern_character", "chern_classes"}, default="chern_character"
        Chern invariant used to construct the equations.

    component : int | None, default=None
        Specific component to compute. If ``None``, all available components
        are returned.

    Returns
    -------
    sp.Equality
        A single SymPy equation when ``component`` is specified.

    tuple[sp.Equality, ...]
        One SymPy equation per available component when ``component`` is
        ``None``.

    Raises
    ------
    TypeError
        If the sequence is not an ordered sequence.

    ValueError
        If the sequence is invalid, mixes schemes, contains no vector bundles,
        or ``invariant`` is not supported.

    Notes
    -----
    The function assumes that the supplied sequence is exact. It does not
    verify exactness because no morphisms are provided.
    """
    terms = _validate_exact_sequence(sequence)

    if invariant not in {"chern_character", "chern_classes"}:
        raise ValueError("invariant must be 'chern_character' or 'chern_classes'.")

    even_expression = sp.Add(*terms[::2])
    odd_expression = sp.Add(*terms[1::2])

    invariant_function = chern_character if invariant == "chern_character" else chern_class

    if component is not None:
        left = invariant_function(even_expression, component=component)
        right = invariant_function(odd_expression, component=component)

        return sp.Eq(sp.expand(left), sp.expand(right), evaluate=False)

    left_components = invariant_function(even_expression)
    right_components = invariant_function(odd_expression)

    first_component = 0 if invariant == "chern_character" else 1
    max_components = min(len(left_components), len(right_components))

    return tuple(
        sp.Eq(
            sp.expand(left_components[index]),
            sp.expand(right_components[index]),
            evaluate=False
        )
        for index in range(first_component, max_components)
    )


def solve_exact_sequence(
    sequences: Sequence[sp.Expr] | Sequence[Sequence[sp.Expr]],
    solve_for: VectorBundle | sp.Symbol | Sequence[VectorBundle | sp.Symbol],
    invariant: Invariant = "chern_character",
    component: int | None = None
) -> list[dict[sp.Symbol, sp.Expr]]:
    """
    Construct and solve the Chern equations from one or several exact sequences.

    A single sequence can be passed directly as ``[E, F, G]``. Several
    sequences can be combined by passing ``[[E, F, G], [G, H, K]]``. Each
    sequence is converted into equations by ``exact_sequence_realtions`` and all
    equations are solved together with ``sympy.solve``.

    Equations are converted from ``Eq(lhs, rhs)`` to ``lhs - rhs = 0`` and
    trivial identities are removed before solving. Vector bundles in
    ``solve_for`` are automatically expanded into their symbolic Chern
    components.

    Parameters
    ----------
    sequences : Sequence[sp.Expr] | Sequence[Sequence[sp.Expr]]
        One exact sequence or a collection of exact sequences.

    solve_for : VectorBundle | sp.Symbol | Sequence[VectorBundle | sp.Symbol]
        Vector bundles or explicit component symbols to solve for.

    invariant : {"chern_character", "chern_classes"}, default="chern_character"
        Invariant used to generate the equations and unknown variables.

    component : int | None, default=None
        Specific component to solve. If ``None``, all symbolic components are
        included.

    Returns
    -------
    list[dict[sp.Symbol, sp.Expr]]
        Solutions returned by ``sympy.solve(..., dict=True)``. Each dictionary
        maps a solved component to its expression. A single solution is still
        returned inside a one-element list.

    Raises
    ------
    ValueError
        If no sequence is supplied, all generated equations are trivial, no
        symbolic solve variables are found, or the requested variables do not
        appear in the equations.
    """
    if not sequences:
        raise ValueError("At least one exact sequence is required.")

    first_element = sequences[0]
    normalized_sequences = (sequences,) if isinstance(first_element, sp.Expr) else tuple(sequences)
    equations: list[sp.Equality] = []

    for sequence in normalized_sequences:
        sequence_equations = exact_sequence_realtions(sequence=sequence, invariant=invariant, component=component)

        if isinstance(sequence_equations, sp.Equality):
            equations.append(sequence_equations)
        else:
            equations.extend(sequence_equations)

    residuals = [sp.simplify(equation.lhs - equation.rhs) for equation in equations]
    residuals = [residual for residual in residuals if residual != 0]

    if not residuals:
        raise ValueError("The exact sequences do not produce any non-trivial equations.")

    variables = _get_solve_variables(solve_for=solve_for, invariant=invariant, component=component)
    missing_variables = [variable for variable in variables if not any(residual.has(variable) for residual in residuals)]

    if missing_variables:
        raise ValueError(f"The variables {missing_variables} do not appear in the generated equations.")

    return sp.solve(residuals, variables, dict=True)