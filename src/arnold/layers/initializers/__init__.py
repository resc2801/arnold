# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
ARNOLD Initializers Package.

This package provides KAN-specific weight initializers optimized for different
basis function types.

Classes
-------
KANInitializer
    Abstract base class for all KAN initializers.
PolynomialInitializer
    Initializer for polynomial basis coefficients.
RBFInitializer
    Initializer for radial basis function parameters.
SpectralInitializer
    Initializer for spectral/Fourier basis coefficients.
OrthogonalInitializer
    Orthogonal matrix initializer for KAN layers.

Functions
---------
create_trainable_param
    Create a trainable parameter with optional constraints.
create_bounded_param
    Create a parameter bounded to an interval.

Examples
--------
>>> from arnold.layers.initializers import PolynomialInitializer
>>>
>>> # Initialize polynomial coefficients with degree scaling
>>> init = PolynomialInitializer(degree=10, normalize=True)
"""

from __future__ import annotations

from arnold.layers.initializers.base import (
    KANInitializer,
    create_bounded_param,
    create_trainable_param,
)
from arnold.layers.initializers.orthogonal_init import OrthogonalInitializer
from arnold.layers.initializers.polynomial import PolynomialInitializer
from arnold.layers.initializers.rbf import RBFInitializer
from arnold.layers.initializers.spectral import SpectralInitializer

__all__ = [
    # Base
    "KANInitializer",
    # Specialized
    "PolynomialInitializer",
    "RBFInitializer",
    "SpectralInitializer",
    "OrthogonalInitializer",
    # Functions
    "create_trainable_param",
    "create_bounded_param",
]
