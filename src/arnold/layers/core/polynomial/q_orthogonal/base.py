# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Base class for q-orthogonal polynomial KAN layers.

This module provides the abstract base class for all q-analog polynomial families
in the q-Askey scheme.

Mathematical Background
-----------------------
All q-polynomials share a common parameter :math:`q \in (0, 1)` and use the
q-Pochhammer symbol:

.. math::

    (a; q)_n = \prod_{k=0}^{n-1} (1 - a q^k)

with :math:`(a; q)_0 = 1`.

References
----------
.. [DLMF] NIST Digital Library of Mathematical Functions, §18.27-18.28
   https://dlmf.nist.gov/18.27
.. [KLS] Koekoek, R., Lesky, P. A., & Swarttouw, R. F. (2010).
   Hypergeometric orthogonal polynomials and their q-analogues.
"""
import math
from abc import ABC, abstractmethod

import tensorflow as tf

from ..poly_base import PolynomialBase


tfk = tf.keras


def _inverse_softplus_lower_bound(value: float, lower_bound: float, eps: float = 1e-6) -> float:
    r"""
    Compute logits that produce ``value`` when passed through softplus_lower_bound.

    This is the inverse of :math:`f(x) = \text{softplus}(x) + \text{lower\_bound} + \epsilon`.

    Parameters
    ----------
    value : float
        Target value after transformation.
    lower_bound : float
        Lower bound used in softplus_lower_bound.
    eps : float, default 1e-6
        Epsilon for numerical stability.

    Returns
    -------
    float
        Logit value.

    Raises
    ------
    ValueError
        If value <= lower_bound + eps.
    """
    y = value - lower_bound - eps
    if y <= 0:
        raise ValueError(f"value={value} must be > lower_bound + eps = {lower_bound + eps}")
    if y > 20:
        return y
    return math.log(math.exp(y) - 1)


def _inverse_sigmoid(value: float, eps: float = 1e-6) -> float:
    r"""
    Compute logit that produces ``value`` when passed through sigmoid.

    This is the inverse of :math:`\sigma(x) = 1 / (1 + e^{-x})`.

    Parameters
    ----------
    value : float
        Target value in (0, 1).
    eps : float, default 1e-6
        Epsilon for numerical stability.

    Returns
    -------
    float
        Logit value.
    """
    value = max(eps, min(1 - eps, value))
    return math.log(value / (1 - value))


class QPolynomialBase(PolynomialBase, ABC):
    r"""
    Abstract base class for q-orthogonal polynomial KAN layers.

    All q-polynomials in the q-Hahn class share a common parameter :math:`q \in (0, 1)`.
    This base class provides:

    - q parameter management with sigmoid constraint
    - Common q-Pochhammer symbol computation
    - Common q-power precomputation utilities

    The q-Pochhammer symbol is defined as:

    .. math::

        (a; q)_n = \prod_{k=0}^{n-1} (1 - a q^k)

    with :math:`(a; q)_0 = 1`.

    Parameters
    ----------
    degree : int
        Maximum polynomial degree.
    units : int
        Output dimensionality.
    q : float
        Base parameter, must be in (0, 1). Default is 0.5.
    q_trainable : bool
        Whether q is trainable. Default is True.
    input_clip : tuple[float, float] | None
        Optional input clamp.
    **kwargs :
        Forwarded to :class:`PolynomialBase`.

    Attributes
    ----------
    q_logits : tf.Variable
        Logits for q parameter (sigmoid-constrained to (0, 1)).

    See Also
    --------
    arnold.layers.core.polynomial.orthogonal : Classical orthogonal polynomials
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

        if not 0 < q < 1:
            raise ValueError(f"q must be in (0, 1), got {q}")

        self.q_init = q
        self.q_trainable = q_trainable
        self.q_logits = None

    def build(self, input_shape):
        super().build(input_shape)

        # q ∈ (0, 1) via sigmoid
        q_logit = _inverse_sigmoid(self.q_init)

        self.q_logits = self.add_weight(
            name="q_logits",
            shape=(),
            initializer=tf.constant_initializer(q_logit),
            trainable=self.q_trainable,
            dtype=self.dtype,
        )

    def _get_q(self, dtype=tf.float64):
        """
        Get q parameter constrained to (0, 1).

        Parameters
        ----------
        dtype : tf.DType
            Target dtype for the result.

        Returns
        -------
        tf.Tensor
            Scalar tensor with q value.
        """
        return tf.sigmoid(tf.cast(self.q_logits, dtype))

    def _q_pochhammer(self, a, q, n, dtype=tf.float64):
        r"""
        Compute q-Pochhammer symbol :math:`(a; q)_n`.

        .. math::

            (a; q)_n = \prod_{k=0}^{n-1} (1 - a \cdot q^k)

        Parameters
        ----------
        a : tf.Tensor
            Parameter a.
        q : tf.Tensor
            Base q.
        n : int
            Number of factors.
        dtype : tf.DType
            Computation dtype.

        Returns
        -------
        tf.Tensor
            Scalar tensor with :math:`(a; q)_n`.
        """
        if n == 0:
            return tf.ones((), dtype=dtype)

        # Compute product: (1 - a) * (1 - a*q) * ... * (1 - a*q^{n-1})
        k = tf.range(n, dtype=dtype)
        factors = 1.0 - a * tf.pow(q, k)
        return tf.reduce_prod(factors)

    def _q_pochhammer_ratio(self, a, q, n, m, dtype=tf.float64):
        r"""
        Compute ratio :math:`(a; q)_n / (a; q)_m` for :math:`n \geq m`.

        More numerically stable than computing separately.

        Parameters
        ----------
        a : tf.Tensor
            Parameter a.
        q : tf.Tensor
            Base q.
        n : int
            Numerator index.
        m : int
            Denominator index.
        dtype : tf.DType
            Computation dtype.

        Returns
        -------
        tf.Tensor
            Scalar tensor with the ratio.
        """
        if n == m:
            return tf.ones((), dtype=dtype)
        if n < m:
            # Reciprocal
            return 1.0 / self._q_pochhammer_ratio(a, q, m, n, dtype)

        # (a; q)_n / (a; q)_m = prod_{k=m}^{n-1} (1 - a * q^k)
        k = tf.range(m, n, dtype=dtype)
        factors = 1.0 - a * tf.pow(q, k)
        return tf.reduce_prod(factors)

    def get_config(self):
        config = super().get_config()
        config.update({
            "q": self.q_init,
            "q_trainable": self.q_trainable,
        })
        return config

    @abstractmethod
    def pseudo_vandermonde(self, x):
        """Compute q-polynomial basis; implemented by subclasses."""
        raise NotImplementedError


__all__ = [
    "QPolynomialBase",
    "_inverse_softplus_lower_bound",
    "_inverse_sigmoid",
]
