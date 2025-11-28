## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
# ruff: isort:skip_file
"""Shared numerical constants for ARNOLD layers."""

import tensorflow as tf

# Default perturbations tuned for float32/float64 boundaries.
EPS_FLOAT32 = 1e-7
EPS_FLOAT64 = 1e-15
PARAM_EPS = 1e-6
BOUNDARY_EPS = 1e-8


def eps_for_dtype(dtype: tf.dtypes.DType) -> float:
    """
    Return an appropriate epsilon for the given dtype.

    Parameters
    ----------
    dtype : tf.dtypes.DType
        TensorFlow dtype.

    Returns
    -------
    float
        Epsilon magnitude suited to ``dtype``.
    """
    return EPS_FLOAT64 if dtype == tf.float64 else EPS_FLOAT32
