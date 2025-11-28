# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
ARNOLD Regularizers Package.

This package provides KAN-specific regularizers for controlling weight behavior
during training.

Classes
-------
KANRegularizer
    Abstract base class for all KAN regularizers.
L1Regularizer
    L1 (Lasso) regularization.
L2Regularizer
    L2 (Ridge) regularization.
L1L2Regularizer
    Combined L1 + L2 (Elastic Net) regularization.
SparsityRegularizer
    Entropy-based sparsity regularization.
SmoothnessRegularizer
    Finite-difference smoothness regularization.
CurvatureRegularizer
    Second-derivative curvature penalty.

Examples
--------
>>> from arnold.layers.regularizers import L1L2Regularizer, SparsityRegularizer
>>>
>>> # Elastic net regularization
>>> reg = L1L2Regularizer(l1=1e-4, l2=1e-3)
>>>
>>> # Promote sparse activations
>>> sparsity_reg = SparsityRegularizer(target_sparsity=0.9)
"""

from __future__ import annotations

from arnold.layers.regularizers.base import KANRegularizer
from arnold.layers.regularizers.curvature import CurvatureRegularizer
from arnold.layers.regularizers.l1_l2 import (
    L1L2Regularizer,
    L1Regularizer,
    L2Regularizer,
)
from arnold.layers.regularizers.smoothness import SmoothnessRegularizer
from arnold.layers.regularizers.sparsity import SparsityRegularizer

__all__ = [
    # Base
    "KANRegularizer",
    # Standard
    "L1Regularizer",
    "L2Regularizer",
    "L1L2Regularizer",
    # KAN-specific
    "SparsityRegularizer",
    "SmoothnessRegularizer",
    "CurvatureRegularizer",
]
