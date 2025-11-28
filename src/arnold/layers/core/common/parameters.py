# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Parameter handling and domain transformations for basis functions.

This module provides utilities for:

- Domain specification and validation
- Linear and nonlinear domain transformations
- Parameter validation and normalization

Mathematical Background
-----------------------
Most orthogonal polynomials are defined on canonical domains:

- Chebyshev, Legendre, Jacobi: :math:`[-1, 1]`
- Laguerre: :math:`[0, \infty)`
- Hermite: :math:`(-\infty, \infty)`

User inputs often come from arbitrary domains :math:`[a, b]` and must
be mapped to the canonical domain via affine transformation:

.. math::
    t = \frac{2(x - a)}{b - a} - 1 \in [-1, 1]

For semi-infinite and infinite domains, we use nonlinear maps
(e.g., exponential, tanh) to preserve numerical stability.
"""

from __future__ import annotations

from typing import Tuple, Union, Optional

import tensorflow as tf

from .types import DomainSpec, DOMAIN_SYMMETRIC


def scale_to_domain(
    x: tf.Tensor,
    domain: DomainSpec,
    target: DomainSpec = DOMAIN_SYMMETRIC,
) -> tf.Tensor:
    r"""Transform input from one domain to another.
    
    Applies an affine transformation:
    
    .. math::
        t = \frac{(x - a)(d - c)}{b - a} + c
    
    where :math:`[a, b]` is the input domain and :math:`[c, d]` is
    the target domain.
    
    Parameters
    ----------
    x : tf.Tensor
        Input tensor.
    domain : DomainSpec
        Input domain specification.
    target : DomainSpec
        Target domain. Defaults to :math:`[-1, 1]`.
    
    Returns
    -------
    tf.Tensor
        Transformed tensor.
    
    Raises
    ------
    ValueError
        If the domain bounds are invalid (lower >= upper).
    
    Examples
    --------
    >>> from arnold.layers.core.common.types import DomainSpec
    >>> x = tf.constant([0.0, 0.5, 1.0])
    >>> domain = DomainSpec(lower=0.0, upper=1.0)
    >>> t = scale_to_domain(x, domain)  # Maps [0,1] -> [-1,1]
    >>> # t is approximately [-1, 0, 1]
    """
    if domain.lower >= domain.upper:
        raise ValueError(
            f"Invalid domain bounds: lower={domain.lower} >= upper={domain.upper}"
        )
    if target.lower >= target.upper:
        raise ValueError(
            f"Invalid target bounds: lower={target.lower} >= upper={target.upper}"
        )
    
    # Cast bounds to tensor dtype
    dtype = x.dtype
    a = tf.cast(domain.lower, dtype)
    b = tf.cast(domain.upper, dtype)
    c = tf.cast(target.lower, dtype)
    d = tf.cast(target.upper, dtype)
    
    # Affine transformation
    return (x - a) * (d - c) / (b - a) + c


def get_domain_bounds(
    domain: DomainSpec,
) -> Tuple[float, float]:
    r"""Extract domain bounds as a tuple.
    
    Parameters
    ----------
    domain : DomainSpec
        Domain specification.
    
    Returns
    -------
    Tuple[float, float]
        Tuple of (lower, upper) bounds.
    """
    return (domain.lower, domain.upper)


def validate_parameters(
    degree: int,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    min_degree: int = 0,
    max_degree: Optional[int] = None,
) -> None:
    r"""Validate polynomial parameters.
    
    Checks that degree and shape parameters are within valid ranges.
    
    Parameters
    ----------
    degree : int
        Polynomial degree.
    alpha : float, optional
        First shape parameter (e.g., for Jacobi polynomials).
    beta : float, optional
        Second shape parameter.
    min_degree : int
        Minimum allowed degree. Default 0.
    max_degree : int, optional
        Maximum allowed degree. No limit if None.
    
    Raises
    ------
    ValueError
        If any parameter is out of valid range.
    
    Notes
    -----
    For Jacobi polynomials :math:`P_n^{(\alpha, \beta)}(x)`, we require
    :math:`\alpha, \beta > -1` for the weight function to be integrable.
    """
    if degree < min_degree:
        raise ValueError(f"degree must be >= {min_degree}, got {degree}")
    
    if max_degree is not None and degree > max_degree:
        raise ValueError(f"degree must be <= {max_degree}, got {degree}")
    
    if alpha is not None and alpha <= -1:
        raise ValueError(f"alpha must be > -1, got {alpha}")
    
    if beta is not None and beta <= -1:
        raise ValueError(f"beta must be > -1, got {beta}")


def clip_to_domain(
    x: tf.Tensor,
    domain: DomainSpec,
    epsilon: float = 1e-7,
) -> tf.Tensor:
    r"""Clip values to domain with small margin.
    
    Clips input to :math:`[a + \epsilon, b - \epsilon]` to avoid
    boundary singularities.
    
    Parameters
    ----------
    x : tf.Tensor
        Input tensor.
    domain : DomainSpec
        Target domain.
    epsilon : float
        Margin from boundaries. Default ``1e-7``.
    
    Returns
    -------
    tf.Tensor
        Clipped tensor.
    
    Notes
    -----
    This is useful for polynomials with weight functions that are
    singular at the boundaries (e.g., Chebyshev of the first kind).
    """
    dtype = x.dtype
    lower = tf.cast(domain.lower + epsilon, dtype)
    upper = tf.cast(domain.upper - epsilon, dtype)
    return tf.clip_by_value(x, lower, upper)


def normalize_input(
    x: tf.Tensor,
    mean: Optional[tf.Tensor] = None,
    std: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    r"""Normalize input to zero mean and unit variance.
    
    Computes:
    
    .. math::
        \hat{x} = \frac{x - \mu}{\sigma}
    
    Parameters
    ----------
    x : tf.Tensor
        Input tensor.
    mean : tf.Tensor, optional
        Mean to subtract. Computed from x if None.
    std : tf.Tensor, optional
        Standard deviation to divide. Computed from x if None.
    
    Returns
    -------
    tf.Tensor
        Normalized tensor.
    """
    if mean is None:
        mean = tf.reduce_mean(x)
    if std is None:
        std = tf.math.reduce_std(x)
    
    # Avoid division by zero
    std = tf.maximum(std, tf.constant(1e-7, dtype=x.dtype))
    
    return (x - mean) / std


__all__ = [
    "scale_to_domain",
    "get_domain_bounds",
    "validate_parameters",
    "clip_to_domain",
    "normalize_input",
]
