# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Base classes for symbolic computation.

Provides abstract interfaces for symbolic representation of KAN layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import sympy as sp

    from arnold.layers.core.kan_base import KANBase


class SymbolicBase(ABC):
    r"""
    Abstract base class for symbolic representations of KAN layers.

    Provides the interface for converting trained KAN layers into
    symbolic mathematical expressions.

    Methods
    -------
    to_expression(layer, input_symbols)
        Convert a KAN layer to a symbolic expression.
    to_latex(layer)
        Convert a KAN layer to LaTeX representation.
    get_coefficients(layer)
        Extract learned coefficients from a layer.
    """

    @abstractmethod
    def to_expression(
        self,
        layer: KANBase,
        input_symbols: list[sp.Symbol] | None = None,
    ) -> sp.Expr:
        r"""
        Convert a KAN layer to a symbolic expression.

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
        """
        raise NotImplementedError

    @abstractmethod
    def to_latex(self, layer: KANBase) -> str:
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
        raise NotImplementedError

    @abstractmethod
    def get_coefficients(self, layer: KANBase) -> dict[str, Any]:
        r"""
        Extract learned coefficients from a layer.

        Parameters
        ----------
        layer : KANBase
            The trained KAN layer.

        Returns
        -------
        dict[str, Any]
            Dictionary mapping coefficient names to their values.
        """
        raise NotImplementedError


class BasisSymbolic(ABC):
    r"""
    Interface for basis-specific symbolic representations.

    Each basis type (Legendre, Chebyshev, Fourier, etc.) implements this
    interface to provide its symbolic representation.
    """

    @abstractmethod
    def basis_symbol(self, degree: int, variable: sp.Symbol) -> sp.Expr:
        r"""
        Get the symbolic expression for a single basis function.

        Parameters
        ----------
        degree : int
            The degree/order of the basis function.
        variable : sp.Symbol
            The symbolic variable.

        Returns
        -------
        sp.Expr
            Symbolic expression for the basis function.
        """
        raise NotImplementedError

    @abstractmethod
    def basis_name(self) -> str:
        r"""
        Get the name of this basis type.

        Returns
        -------
        str
            Name like "Legendre", "Chebyshev", "Fourier", etc.
        """
        raise NotImplementedError
