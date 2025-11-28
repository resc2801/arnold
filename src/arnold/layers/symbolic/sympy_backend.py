# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
SymPy backend for symbolic computation.

Provides the main backend implementation using SymPy for symbolic
mathematical operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from arnold.layers.symbolic.base import SymbolicBase

if TYPE_CHECKING:
    import sympy as sp

    from arnold.layers.core.kan_base import KANBase


def _ensure_sympy():
    """Ensure SymPy is available, raise helpful error if not."""
    try:
        import sympy  # noqa: F401

        return sympy
    except ImportError as e:
        raise ImportError(
            "SymPy is required for symbolic operations. "
            "Install it with: pip install sympy"
        ) from e


class SymPyBackend(SymbolicBase):
    r"""
    SymPy-based backend for symbolic computation.

    Provides methods to convert KAN layers into symbolic SymPy expressions,
    enabling mathematical analysis, simplification, and export.

    Examples
    --------
    >>> backend = SymPyBackend()
    >>> expr = backend.to_expression(layer)
    >>> latex = backend.to_latex(layer)
    """

    def __init__(self):
        """Initialize the SymPy backend."""
        self._sp = _ensure_sympy()

    def to_expression(
        self,
        layer: "KANBase",
        input_symbols: list["sp.Symbol"] | None = None,
    ) -> "sp.Expr":
        r"""
        Convert a KAN layer to a symbolic SymPy expression.

        Parameters
        ----------
        layer : KANBase
            The trained KAN layer to convert.
        input_symbols : list[sp.Symbol] | None, optional
            Symbols for input variables. If None, generates x_0, x_1, etc.

        Returns
        -------
        sp.Expr
            Symbolic expression representing the layer's computation.

        Notes
        -----
        The conversion process:

        1. Extract layer weights and configuration
        2. Generate symbolic basis functions
        3. Combine with learned coefficients
        4. Optionally simplify the result
        """
        sp = self._sp

        # Get layer configuration
        if not layer.built:
            raise ValueError("Layer must be built before symbolic conversion")

        # Determine input dimension from layer
        input_dim = self._get_input_dim(layer)

        # Create input symbols if not provided
        if input_symbols is None:
            input_symbols = [sp.Symbol(f"x_{i}") for i in range(input_dim)]

        # Get basis type and build symbolic representation
        basis_expr = self._build_basis_expression(layer, input_symbols)

        return basis_expr

    def to_latex(self, layer: "KANBase") -> str:
        r"""
        Convert a KAN layer to LaTeX representation.

        Parameters
        ----------
        layer : KANBase
            The trained KAN layer to convert.

        Returns
        -------
        str
            LaTeX string representation.
        """
        sp = self._sp
        expr = self.to_expression(layer)
        return sp.latex(expr)

    def get_coefficients(self, layer: "KANBase") -> dict[str, Any]:
        r"""
        Extract learned coefficients from a layer.

        Parameters
        ----------
        layer : KANBase
            The trained KAN layer.

        Returns
        -------
        dict[str, Any]
            Dictionary mapping coefficient names to their NumPy values.
        """
        if not layer.built:
            raise ValueError("Layer must be built before extracting coefficients")

        coefficients = {}
        for weight in layer.weights:
            coefficients[weight.name] = weight.numpy()

        return coefficients

    def _get_input_dim(self, layer: "KANBase") -> int:
        """Get the input dimension from a layer."""
        # Try to get from layer's input_spec or stored dimension
        if hasattr(layer, "input_dim") and layer.input_dim is not None:
            return layer.input_dim
        if hasattr(layer, "_input_dim") and layer._input_dim is not None:
            return layer._input_dim

        # Try to infer from weights
        for weight in layer.weights:
            if "kernel" in weight.name.lower():
                return weight.shape[0]

        # Default fallback
        return 1

    def _build_basis_expression(
        self,
        layer: "KANBase",
        input_symbols: list["sp.Symbol"],
    ) -> "sp.Expr":
        """Build the symbolic expression for the layer's basis."""
        sp = self._sp

        # Get layer type name
        layer_type = type(layer).__name__.lower()

        # Get weights
        coeffs = self.get_coefficients(layer)

        # Find the main kernel weight
        kernel = None
        for name, val in coeffs.items():
            if "kernel" in name.lower() or "weight" in name.lower():
                kernel = val
                break

        if kernel is None:
            # Return simple sum if no kernel found
            return sum(input_symbols)

        # Get degree if available
        degree = getattr(layer, "degree", kernel.shape[-1] - 1 if len(kernel.shape) > 1 else 0)

        # Build basis-specific expression
        if "legendre" in layer_type:
            return self._legendre_expression(input_symbols, kernel, degree)
        elif "chebyshev" in layer_type:
            return self._chebyshev_expression(input_symbols, kernel, degree)
        elif "hermite" in layer_type:
            return self._hermite_expression(input_symbols, kernel, degree)
        elif "fourier" in layer_type:
            return self._fourier_expression(input_symbols, kernel, degree)
        else:
            # Generic polynomial expansion
            return self._generic_polynomial_expression(input_symbols, kernel, degree)

    def _legendre_expression(
        self,
        symbols: list["sp.Symbol"],
        kernel,
        degree: int,
    ) -> "sp.Expr":
        """Build Legendre polynomial expression."""
        sp = self._sp
        from sympy import legendre

        result = sp.Integer(0)
        for i, x in enumerate(symbols):
            for k in range(min(degree + 1, kernel.shape[-1] if len(kernel.shape) > 1 else 1)):
                coeff = float(kernel[i, 0, k]) if len(kernel.shape) > 2 else float(kernel.flat[k])
                if abs(coeff) > 1e-10:
                    result += coeff * legendre(k, x)
        return result

    def _chebyshev_expression(
        self,
        symbols: list["sp.Symbol"],
        kernel,
        degree: int,
    ) -> "sp.Expr":
        """Build Chebyshev polynomial expression."""
        sp = self._sp
        from sympy import chebyshevt

        result = sp.Integer(0)
        for i, x in enumerate(symbols):
            for k in range(min(degree + 1, kernel.shape[-1] if len(kernel.shape) > 1 else 1)):
                coeff = float(kernel[i, 0, k]) if len(kernel.shape) > 2 else float(kernel.flat[k])
                if abs(coeff) > 1e-10:
                    result += coeff * chebyshevt(k, x)
        return result

    def _hermite_expression(
        self,
        symbols: list["sp.Symbol"],
        kernel,
        degree: int,
    ) -> "sp.Expr":
        """Build Hermite polynomial expression."""
        sp = self._sp
        from sympy import hermite

        result = sp.Integer(0)
        for i, x in enumerate(symbols):
            for k in range(min(degree + 1, kernel.shape[-1] if len(kernel.shape) > 1 else 1)):
                coeff = float(kernel[i, 0, k]) if len(kernel.shape) > 2 else float(kernel.flat[k])
                if abs(coeff) > 1e-10:
                    result += coeff * hermite(k, x)
        return result

    def _fourier_expression(
        self,
        symbols: list["sp.Symbol"],
        kernel,
        degree: int,
    ) -> "sp.Expr":
        """Build Fourier series expression."""
        sp = self._sp

        result = sp.Integer(0)
        x = symbols[0] if symbols else sp.Symbol("x")

        # Fourier: 1, cos(x), sin(x), cos(2x), sin(2x), ...
        num_terms = kernel.shape[-1] if len(kernel.shape) > 0 else 1
        idx = 0
        for k in range((num_terms + 1) // 2):
            if idx < num_terms:
                coeff = float(kernel.flat[idx])
                if k == 0:
                    result += coeff  # constant term
                else:
                    result += coeff * sp.cos(k * x)
                idx += 1
            if k > 0 and idx < num_terms:
                coeff = float(kernel.flat[idx])
                result += coeff * sp.sin(k * x)
                idx += 1

        return result

    def _generic_polynomial_expression(
        self,
        symbols: list["sp.Symbol"],
        kernel,
        degree: int,
    ) -> "sp.Expr":
        """Build generic polynomial expression."""
        sp = self._sp

        result = sp.Integer(0)
        for i, x in enumerate(symbols):
            for k in range(min(degree + 1, kernel.shape[-1] if len(kernel.shape) > 1 else degree + 1)):
                try:
                    coeff = float(kernel[i, 0, k]) if len(kernel.shape) > 2 else float(kernel.flat[k])
                except (IndexError, ValueError):
                    coeff = 0.0
                if abs(coeff) > 1e-10:
                    result += coeff * x**k

        return result
