# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
ARNOLD Mixed Basis Package.

This package provides layers that combine multiple basis function families
into a single representation.

Classes
-------
MixedBasis
    Combine multiple different basis functions with learned weights.
ProductBasis
    Tensor product of basis functions (e.g., Legendre × Fourier).
AttentionBasis
    Attention-weighted combination of basis functions.

Examples
--------
>>> from arnold.layers.mixed import MixedBasis, ProductBasis
>>>
>>> # Combine polynomial and spectral bases
>>> layer = MixedBasis(units=32, bases=['legendre', 'fourier'])
>>>
>>> # Tensor product for multivariate functions
>>> layer = ProductBasis(units=32, basis_x='legendre', basis_y='fourier')
"""

from __future__ import annotations

from arnold.layers.mixed.attention_basis import AttentionBasis
from arnold.layers.mixed.mixed_basis import MixedBasis
from arnold.layers.mixed.product_basis import ProductBasis

__all__ = [
    "MixedBasis",
    "ProductBasis",
    "AttentionBasis",
]
