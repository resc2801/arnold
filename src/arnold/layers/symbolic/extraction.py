# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Symbolic extraction functions for KAN layers.

Provides the main API functions for converting trained KAN layers
into symbolic mathematical expressions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import sympy as sp

    from arnold.layers.core.kan_base import KANBase


def kan_to_polynomial(
    layer: "KANBase",
    input_symbols: list["sp.Symbol"] | None = None,
    simplify: bool = True,
) -> "sp.Expr":
    r"""
    Convert a trained KAN layer to a symbolic SymPy expression.

    This is the primary function for extracting interpretable mathematical
    expressions from trained KAN layers.

    Parameters
    ----------
    layer : KANBase
        Trained KAN layer to convert.
    input_symbols : list[sp.Symbol] | None, optional
        Symbolic variables for inputs. Default: [x_0, x_1, ..., x_n].
    simplify : bool, default=True
        Whether to simplify the resulting expression.

    Returns
    -------
    sp.Expr
        Symbolic SymPy expression representing the layer's computation.

    Examples
    --------
    >>> from arnold.layers import Legendre
    >>> from arnold.layers.symbolic import kan_to_polynomial
    >>>
    >>> layer = Legendre(units=1, degree=3)
    >>> layer.build((None, 2))
    >>> # After training...
    >>> expr = kan_to_polynomial(layer)
    >>> print(expr)
    0.5*x_0**2 - 0.3*x_1 + 1.2*x_0*x_1

    Notes
    -----
    This function enables:

    - **Interpretability**: Understand what function the KAN learned
    - **Symbolic differentiation**: Compute derivatives symbolically
    - **Export**: Convert to Mathematica, MATLAB, or LaTeX
    - **Publications**: Include explicit formulas in papers

    See Also
    --------
    kan_to_latex : Convert to LaTeX representation
    extract_coefficients : Get raw coefficient values
    """
    from arnold.layers.symbolic.sympy_backend import SymPyBackend

    backend = SymPyBackend()
    expr = backend.to_expression(layer, input_symbols)

    if simplify:
        import sympy as sp

        expr = sp.simplify(expr)

    return expr


def kan_to_latex(
    layer: "KANBase",
    simplify: bool = True,
    mode: str = "inline",
) -> str:
    r"""
    Convert a trained KAN layer to LaTeX representation.

    Parameters
    ----------
    layer : KANBase
        Trained KAN layer to convert.
    simplify : bool, default=True
        Whether to simplify before converting.
    mode : {'inline', 'equation', 'align'}, default='inline'
        LaTeX mode:
        - 'inline': Just the expression
        - 'equation': Wrapped in \\begin{equation}
        - 'align': Wrapped in \\begin{align}

    Returns
    -------
    str
        LaTeX string representation.

    Examples
    --------
    >>> latex = kan_to_latex(layer)
    >>> print(latex)
    0.5 x_{0}^{2} - 0.3 x_{1} + 1.2 x_{0} x_{1}
    >>>
    >>> # For papers
    >>> latex = kan_to_latex(layer, mode='equation')
    >>> print(latex)
    \\begin{equation}
    f(x) = 0.5 x_{0}^{2} - 0.3 x_{1} + 1.2 x_{0} x_{1}
    \\end{equation}
    """
    import sympy as sp

    expr = kan_to_polynomial(layer, simplify=simplify)
    latex_str = sp.latex(expr)

    if mode == "equation":
        return f"\\begin{{equation}}\nf(x) = {latex_str}\n\\end{{equation}}"
    elif mode == "align":
        return f"\\begin{{align}}\nf(x) &= {latex_str}\n\\end{{align}}"
    else:
        return latex_str


def extract_coefficients(
    layer: "KANBase",
    as_dict: bool = True,
) -> dict[str, Any] | list[Any]:
    r"""
    Extract learned coefficients from a trained KAN layer.

    Parameters
    ----------
    layer : KANBase
        Trained KAN layer.
    as_dict : bool, default=True
        If True, return as dict mapping names to values.
        If False, return as flat list.

    Returns
    -------
    dict[str, Any] | list[Any]
        Coefficient values, either as dict or list.

    Examples
    --------
    >>> coeffs = extract_coefficients(layer)
    >>> print(coeffs.keys())
    dict_keys(['legendre/kernel:0', 'legendre/bias:0'])
    >>> print(coeffs['legendre/kernel:0'].shape)
    (2, 1, 4)
    """
    from arnold.layers.symbolic.sympy_backend import SymPyBackend

    backend = SymPyBackend()
    coeffs = backend.get_coefficients(layer)

    if as_dict:
        return coeffs
    else:
        return list(coeffs.values())
