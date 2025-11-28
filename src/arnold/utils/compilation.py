## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
"""
Standardized tf.function decorator helpers for KAN layers.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import tensorflow as tf


def kan_function(jit_compile: bool = True) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Wrap a function with common tf.function settings used across KAN basis evaluations.

    Parameters
    ----------
    jit_compile : bool, default True
        Whether to request XLA compilation. Falls back gracefully when unavailable.

    Returns
    -------
    Callable
        Decorated function with tf.function applied.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        @tf.function(autograph=True, jit_compile=jit_compile, reduce_retracing=True)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        return wrapper

    return decorator
