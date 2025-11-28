## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
"""Utility functions for ARNOLD KAN layers."""

from arnold.utils.compilation import kan_function
from arnold.utils.constants import PARAM_EPS
from arnold.utils.constraints import softplus_lower_bound
from arnold.utils.numerics import safe_acos
from arnold.utils.weights import create_bounded_param_logits, create_trainable_param


# Lazy import for optimized_ops to avoid circular dependency
def __getattr__(name):
    """Lazy import for optimized_ops functions to avoid circular imports."""
    _optimized_ops_names = {
        "clenshaw_chebyshev_sum",
        "clenshaw_legendre_sum",
        "clenshaw_generic_sum",
        "q_polynomial_basis_xla",
        "al_salam_carlitz_1st_basis_xla",
        "fused_polynomial_forward",
        "parallel_polynomial_eval",
        "chebyshev_parallel_eval",
        "with_tpu_sharding",
        "configure_tpu_strategy",
        "optimized_polynomial_forward",
    }
    if name in _optimized_ops_names:
        from arnold.utils import optimized_ops
        return getattr(optimized_ops, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Compilation
    "kan_function",
    # Constants
    "PARAM_EPS",
    # Constraints
    "softplus_lower_bound",
    # Numerics
    "safe_acos",
    # Optimized operations (lazy loaded)
    "clenshaw_chebyshev_sum",
    "clenshaw_legendre_sum",
    "clenshaw_generic_sum",
    "q_polynomial_basis_xla",
    "al_salam_carlitz_1st_basis_xla",
    "fused_polynomial_forward",
    "parallel_polynomial_eval",
    "chebyshev_parallel_eval",
    "with_tpu_sharding",
    "configure_tpu_strategy",
    "optimized_polynomial_forward",
    # Weights
    "create_bounded_param_logits",
    "create_trainable_param",
]
