# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Symbolic simplification utilities.

Provides convenience wrappers around SymPy's simplification routines
for KAN expression manipulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import sympy as sp


def _ensure_sympy():
    r"""Ensure SymPy is installed."""
    try:
        import sympy  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "SymPy is required for symbolic simplification. "
            "Install it with: pip install sympy"
        ) from e


def simplify_expression(expr: sp.Expr, **kwargs) -> sp.Expr:
    r"""
    Simplify a symbolic expression.

    Applies SymPy's main simplification routine with optional
    control over the strategy.

    Parameters
    ----------
    expr : sp.Expr
        SymPy expression to simplify.
    **kwargs
        Additional keyword arguments passed to ``sympy.simplify``.

    Returns
    -------
    sp.Expr
        Simplified expression.

    Examples
    --------
    >>> import sympy as sp
    >>> from arnold.layers.symbolic import simplify_expression
    >>>
    >>> x = sp.Symbol('x')
    >>> expr = x**2 + 2*x + 1
    >>> simplified = simplify_expression(expr)
    >>> print(simplified)
    (x + 1)**2
    """
    _ensure_sympy()
    import sympy as sp

    return sp.simplify(expr, **kwargs)


def expand_expression(expr: sp.Expr, **kwargs) -> sp.Expr:
    r"""
    Expand a symbolic expression.

    Expands products and powers, distributing multiplication over addition.

    Parameters
    ----------
    expr : sp.Expr
        SymPy expression to expand.
    **kwargs
        Additional keyword arguments passed to ``sympy.expand``.

    Returns
    -------
    sp.Expr
        Expanded expression.

    Examples
    --------
    >>> import sympy as sp
    >>> from arnold.layers.symbolic import expand_expression
    >>>
    >>> x = sp.Symbol('x')
    >>> expr = (x + 1)**3
    >>> expanded = expand_expression(expr)
    >>> print(expanded)
    x**3 + 3*x**2 + 3*x + 1
    """
    _ensure_sympy()
    import sympy as sp

    return sp.expand(expr, **kwargs)


def factor_expression(expr: sp.Expr, **kwargs) -> sp.Expr:
    r"""
    Factor a polynomial expression.

    Factors a polynomial over the rationals.

    Parameters
    ----------
    expr : sp.Expr
        SymPy expression to factor.
    **kwargs
        Additional keyword arguments passed to ``sympy.factor``.

    Returns
    -------
    sp.Expr
        Factored expression.

    Examples
    --------
    >>> import sympy as sp
    >>> from arnold.layers.symbolic import factor_expression
    >>>
    >>> x = sp.Symbol('x')
    >>> expr = x**2 - 1
    >>> factored = factor_expression(expr)
    >>> print(factored)
    (x - 1)*(x + 1)
    """
    _ensure_sympy()
    import sympy as sp

    return sp.factor(expr, **kwargs)


def collect_terms(
    expr: sp.Expr,
    symbols: list[sp.Symbol] | sp.Symbol,
    **kwargs,
) -> sp.Expr:
    r"""
    Collect terms with respect to specified symbols.

    Rewrites the expression collecting coefficients of powers
    of the specified symbols.

    Parameters
    ----------
    expr : sp.Expr
        SymPy expression.
    symbols : list[sp.Symbol] | sp.Symbol
        Symbol(s) to collect terms for.
    **kwargs
        Additional keyword arguments passed to ``sympy.collect``.

    Returns
    -------
    sp.Expr
        Expression with collected terms.

    Examples
    --------
    >>> import sympy as sp
    >>> from arnold.layers.symbolic import collect_terms
    >>>
    >>> x, y = sp.symbols('x y')
    >>> expr = x*y + x - 3 + 2*x**2 - y*x**2 + x**3
    >>> collected = collect_terms(expr, x)
    >>> print(collected)
    x**3 + x**2*(2 - y) + x*(y + 1) - 3
    """
    _ensure_sympy()
    import sympy as sp

    return sp.collect(expr, symbols, **kwargs)


def trigsimp_expression(expr: sp.Expr, **kwargs) -> sp.Expr:
    r"""
    Simplify trigonometric expressions.

    Applies trigonometric simplifications.

    Parameters
    ----------
    expr : sp.Expr
        SymPy expression with trigonometric functions.
    **kwargs
        Additional keyword arguments passed to ``sympy.trigsimp``.

    Returns
    -------
    sp.Expr
        Trigonometrically simplified expression.

    Examples
    --------
    >>> import sympy as sp
    >>> from arnold.layers.symbolic import trigsimp_expression
    >>>
    >>> x = sp.Symbol('x')
    >>> expr = sp.sin(x)**2 + sp.cos(x)**2
    >>> simplified = trigsimp_expression(expr)
    >>> print(simplified)
    1
    """
    _ensure_sympy()
    import sympy as sp

    return sp.trigsimp(expr, **kwargs)


def polynomial_degree(expr: sp.Expr, symbol: sp.Symbol) -> int:
    r"""
    Get the polynomial degree with respect to a symbol.

    Parameters
    ----------
    expr : sp.Expr
        SymPy expression.
    symbol : sp.Symbol
        Symbol to compute degree with respect to.

    Returns
    -------
    int
        Polynomial degree.

    Examples
    --------
    >>> import sympy as sp
    >>> from arnold.layers.symbolic import polynomial_degree
    >>>
    >>> x = sp.Symbol('x')
    >>> expr = 3*x**4 + 2*x**2 - 1
    >>> deg = polynomial_degree(expr, x)
    >>> print(deg)
    4
    """
    _ensure_sympy()
    import sympy as sp

    poly = sp.Poly(expr, symbol)
    return poly.degree()


def coefficient_list(
    expr: sp.Expr,
    symbol: sp.Symbol,
) -> list:
    r"""
    Extract coefficient list for a polynomial.

    Returns coefficients from highest degree to lowest.

    Parameters
    ----------
    expr : sp.Expr
        SymPy polynomial expression.
    symbol : sp.Symbol
        Symbol for polynomial variable.

    Returns
    -------
    list
        List of coefficients from highest to lowest degree.

    Examples
    --------
    >>> import sympy as sp
    >>> from arnold.layers.symbolic import coefficient_list
    >>>
    >>> x = sp.Symbol('x')
    >>> expr = 3*x**3 + 2*x - 5
    >>> coeffs = coefficient_list(expr, x)
    >>> print(coeffs)  # [3, 0, 2, -5] for x^3 + 0*x^2 + 2*x - 5
    [3, 0, 2, -5]
    """
    _ensure_sympy()
    import sympy as sp

    poly = sp.Poly(expr, symbol)
    return poly.all_coeffs()
