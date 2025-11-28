# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
ARNOLD Architectures Package.

This package provides high-level KAN architecture implementations that
combine layers into specific network designs.

Classes
-------
OriginalKAN
    Original B-spline KAN architecture from Liu et al. (2024).
CompactKAN
    Compact/efficient KAN variant with reduced parameters.
KalmanKAN
    Recursive filter-based KAN architecture.
MLPBasis
    MLP as a learnable basis function.
HyperKAN
    Experimental hypernetwork-based KAN.

Examples
--------
>>> from arnold.layers.architectures import OriginalKAN, CompactKAN
>>>
>>> # Original KAN with 3 layers
>>> model = OriginalKAN(
...     layer_dims=[2, 32, 32, 1],
...     spline_order=3,
...     grid_size=5
... )
>>>
>>> # Compact KAN for efficiency
>>> model = CompactKAN(
...     layer_dims=[2, 64, 1],
...     basis='legendre',
...     degree=8
... )
"""

from __future__ import annotations

from arnold.layers.architectures.ckan import CompactKAN
from arnold.layers.architectures.hyper_kan import HyperKAN
from arnold.layers.architectures.kalman_kan import KalmanKAN
from arnold.layers.architectures.mlp_basis import MLPBasis
from arnold.layers.architectures.original_kan import OriginalKAN


__all__ = [
    "OriginalKAN",
    "CompactKAN",
    "KalmanKAN",
    "MLPBasis",
    "HyperKAN",
]
