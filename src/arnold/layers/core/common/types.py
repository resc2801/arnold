# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Type definitions for ARNOLD KAN layers.

This module provides:

- :class:`DomainSpec` — Domain specification for basis functions
- :class:`BasisProtocol` — Protocol for basis function implementations
- :class:`NormalizationScheme` — Enumeration of normalization strategies
- :class:`EvaluationStrategy` — Enumeration of evaluation algorithms

These types enable type-safe, IDE-friendly development across the library.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol, Tuple, Optional, runtime_checkable

import tensorflow as tf


class NormalizationScheme(Enum):
    r"""Normalization schemes for orthogonal polynomials.
    
    Attributes
    ----------
    NONE
        No normalization. Raw polynomial values.
    ORTHONORMAL
        Orthonormal normalization: :math:`\int w(x) P_n(x)^2 dx = 1`
    MONIC
        Monic normalization: leading coefficient is 1.
    STANDARDIZED
        Standardized to :math:`P_n(1) = 1` (where applicable).
    """
    NONE = auto()
    ORTHONORMAL = auto()
    MONIC = auto()
    STANDARDIZED = auto()


class EvaluationStrategy(Enum):
    r"""Algorithms for polynomial evaluation.
    
    Attributes
    ----------
    RECURRENCE
        Standard three-term recurrence. Simple but may accumulate errors.
    CLENSHAW
        Clenshaw algorithm. Backward-stable for Chebyshev and similar.
    HORNER
        Horner's method. Optimal for monomial basis.
    MATRIX
        Matrix-based evaluation. Vectorized, GPU-friendly.
    """
    RECURRENCE = auto()
    CLENSHAW = auto()
    HORNER = auto()
    MATRIX = auto()


@dataclass(frozen=True)
class DomainSpec:
    r"""Specification for the canonical domain of a basis function.
    
    Parameters
    ----------
    lower : float
        Lower bound of the domain.
    upper : float
        Upper bound of the domain.
    periodic : bool, default=False
        Whether the domain is periodic (e.g., for Fourier basis).
    unbounded_lower : bool, default=False
        True if lower bound is :math:`-\infty`.
    unbounded_upper : bool, default=False
        True if upper bound is :math:`+\infty`.
    
    Examples
    --------
    >>> DomainSpec(-1.0, 1.0)  # Chebyshev, Legendre
    DomainSpec(lower=-1.0, upper=1.0, periodic=False, ...)
    
    >>> DomainSpec(0.0, float('inf'), unbounded_upper=True)  # Laguerre
    DomainSpec(lower=0.0, upper=inf, ...)
    
    >>> DomainSpec(-float('inf'), float('inf'), unbounded_lower=True, unbounded_upper=True)  # Hermite
    DomainSpec(lower=-inf, upper=inf, ...)
    """
    lower: float
    upper: float
    periodic: bool = False
    unbounded_lower: bool = False
    unbounded_upper: bool = False
    
    @property
    def is_bounded(self) -> bool:
        """True if domain is bounded on both sides."""
        return not self.unbounded_lower and not self.unbounded_upper
    
    @property
    def center(self) -> float:
        """Center of the domain (only valid for bounded domains)."""
        if not self.is_bounded:
            return 0.0
        return (self.lower + self.upper) / 2.0
    
    @property
    def width(self) -> float:
        """Width of the domain (only valid for bounded domains)."""
        if not self.is_bounded:
            return float('inf')
        return self.upper - self.lower


# Standard domain constants
DOMAIN_UNIT_INTERVAL = DomainSpec(0.0, 1.0)
DOMAIN_SYMMETRIC = DomainSpec(-1.0, 1.0)
DOMAIN_POSITIVE = DomainSpec(0.0, float('inf'), unbounded_upper=True)
DOMAIN_REAL_LINE = DomainSpec(
    -float('inf'), float('inf'), 
    unbounded_lower=True, 
    unbounded_upper=True
)
DOMAIN_UNIT_CIRCLE = DomainSpec(-3.141592653589793, 3.141592653589793, periodic=True)


@runtime_checkable
class BasisProtocol(Protocol):
    r"""Protocol for basis function implementations.
    
    All basis classes (polynomial, RBF, wavelet, etc.) should implement
    this protocol to ensure consistent API across the library.
    
    Methods
    -------
    evaluate(x, degree)
        Evaluate basis functions at points x up to given degree.
    domain
        Return the canonical domain specification.
    """
    
    def evaluate(self, x: tf.Tensor, degree: int) -> tf.Tensor:
        r"""Evaluate basis functions.
        
        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape ``(..., input_dim)``.
        degree : int
            Maximum degree/order of basis functions.
        
        Returns
        -------
        tf.Tensor
            Basis matrix of shape ``(..., input_dim, num_basis)``.
        """
        ...
    
    @property
    def domain(self) -> DomainSpec:
        """Return canonical domain for this basis."""
        ...


@dataclass
class RecurrenceCoefficients:
    r"""Coefficients for three-term recurrence relation.
    
    The recurrence relation is:
    
    .. math::
        P_{n+1}(x) = (a_n x + b_n) P_n(x) - c_n P_{n-1}(x)
    
    Parameters
    ----------
    a : tf.Tensor
        Coefficient :math:`a_n` for each degree n.
    b : tf.Tensor
        Coefficient :math:`b_n` for each degree n.
    c : tf.Tensor
        Coefficient :math:`c_n` for each degree n.
    """
    a: tf.Tensor
    b: tf.Tensor
    c: tf.Tensor


__all__ = [
    "NormalizationScheme",
    "EvaluationStrategy",
    "DomainSpec",
    "BasisProtocol",
    "RecurrenceCoefficients",
    # Domain constants
    "DOMAIN_UNIT_INTERVAL",
    "DOMAIN_SYMMETRIC",
    "DOMAIN_POSITIVE",
    "DOMAIN_REAL_LINE",
    "DOMAIN_UNIT_CIRCLE",
]
