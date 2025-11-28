# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
ARNOLD Symbolic Package.

This package provides symbolic computation capabilities for KAN layers,
enabling interpretability and mathematical analysis of trained networks.

Classes
-------
SymbolicBase
    Abstract base class for symbolic representations.
SymPyBackend
    Backend for SymPy-based symbolic computation.

Functions
---------
kan_to_polynomial
    Convert a trained KAN layer to a symbolic SymPy expression.
kan_to_latex
    Convert a trained KAN layer to LaTeX representation.
simplify_expression
    Simplify a symbolic KAN expression.
extract_coefficients
    Extract polynomial coefficients from a trained KAN.

Examples
--------
>>> from arnold.layers.symbolic import kan_to_polynomial, kan_to_latex
>>> from arnold.layers import Legendre
>>>
>>> # Train a KAN layer
>>> layer = Legendre(units=1, degree=3)
>>> model = tf.keras.Sequential([layer])
>>> model.compile(optimizer='adam', loss='mse')
>>> model.fit(X_train, y_train, epochs=100)
>>>
>>> # Extract symbolic representation
>>> expr = kan_to_polynomial(layer)
>>> print(expr)
0.5*x_0**2 - 0.3*x_1 + 1.2*x_0*x_1
>>>
>>> # Get LaTeX for publication
>>> latex = kan_to_latex(layer)
>>> print(latex)
0.5 x_0^2 - 0.3 x_1 + 1.2 x_0 x_1
"""

from __future__ import annotations

from arnold.layers.symbolic.base import SymbolicBase
from arnold.layers.symbolic.extraction import (
    extract_coefficients,
    kan_to_latex,
    kan_to_polynomial,
)
from arnold.layers.symbolic.simplification import (
    collect_terms,
    expand_expression,
    factor_expression,
    simplify_expression,
)
from arnold.layers.symbolic.sympy_backend import SymPyBackend


__all__ = [
    # Base
    "SymbolicBase",
    "SymPyBackend",
    # Extraction
    "kan_to_polynomial",
    "kan_to_latex",
    "extract_coefficients",
    # Simplification
    "simplify_expression",
    "expand_expression",
    "factor_expression",
    "collect_terms",
]
