# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Mathieu function KAN layer.

This module provides :class:`Mathieu`, a KAN layer using Mathieu functions
as basis functions.

Mathematical Background
-----------------------
Mathieu functions are solutions to Mathieu's differential equation:

.. math::

    y'' + (a - 2q\cos 2x) y = 0

The solutions are classified as:

- **Angular Mathieu functions**: :math:`\text{ce}_n(x, q)`, :math:`\text{se}_n(x, q)`
  (periodic, cosine-like and sine-like)
- **Radial Mathieu functions**: Non-periodic solutions

These functions arise in:
- Wave propagation in elliptic coordinates
- Vibrations of elliptic membranes
- Periodic potentials in quantum mechanics
- Electromagnetic wave guides with elliptic cross-section

Parameters
----------
The parameter :math:`q` controls the "ellipticity":
- :math:`q = 0`: Reduces to simple trigonometric functions
- :math:`q \neq 0`: Introduces elliptic character

References
----------
.. [1] NIST DLMF, Ch. 28: Mathieu Functions
.. [2] McLachlan, N.W. (1947). "Theory and Application of Mathieu Functions"
"""
import tensorflow as tf

from arnold.layers.core.special.base import SpecialBase

tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold", name="Mathieu")
class Mathieu(SpecialBase):
    r"""
    Kolmogorov-Arnold Network layer using Mathieu functions.

    Uses angular Mathieu functions ce_n(x, q) and se_n(x, q) as basis.

    Parameters
    ----------
    units : int
        Output dimension.
    max_order : int, default=8
        Maximum order n for Mathieu functions.
    q : float, default=1.0
        Mathieu parameter controlling ellipticity.
    learnable_q : bool, default=False
        Whether to make q a trainable parameter.
    input_clip : tuple[float, float] | None, default=None
        Input clipping for numerical stability.
    **kwargs
        Additional Keras layer arguments.

    Notes
    -----
    For q=0, the Mathieu functions reduce to:
    - ce_n(x, 0) = cos(nx)
    - se_n(x, 0) = sin(nx)

    Example
    -------
    >>> import tensorflow as tf
    >>> from arnold.layers.core.special import Mathieu
    >>>
    >>> layer = Mathieu(max_order=4, q=2.0, units=32)
    >>> x = tf.random.uniform((16, 10), 0, 2*np.pi)
    >>> y = layer(x)  # shape: (16, 32)
    """

    def __init__(
        self,
        *,
        units: int,
        max_order: int = 8,
        q: float = 1.0,
        learnable_q: bool = False,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        self.initial_q = q
        self.learnable_q = learnable_q
        super().__init__(
            units=units,
            max_order=max_order,
            input_clip=input_clip,
            **kwargs,
        )

    def build(self, input_shape):
        """Build the layer with optional trainable q parameter."""
        if self.learnable_q:
            self.q_var = self.add_weight(
                name="q",
                shape=(),
                initializer=tf.keras.initializers.Constant(self.initial_q),
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.q_var = tf.constant(self.initial_q, dtype=self.dtype)

        super().build(input_shape)

    def _get_num_basis_functions(self) -> int:
        """Return number of Mathieu basis functions."""
        # ce_0 through ce_n plus se_1 through se_n
        # = (n+1) + n = 2n + 1
        return 2 * self.max_order + 1

    def special_basis(self, x: tf.Tensor) -> tf.Tensor:
        r"""
        Compute Mathieu function basis.

        Uses Fourier series approximation for ce_n and se_n.

        For small q, Mathieu functions are approximately:

        .. math::

            \text{ce}_n(x, q) \approx \cos(nx) + O(q)
            \text{se}_n(x, q) \approx \sin(nx) + O(q)

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        tf.Tensor
            Basis tensor of shape (batch, input_dim, num_basis).
        """
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)
        q = tf.cast(self.q_var, tf.float64)

        basis = []

        # ce_0(x, q) ≈ 1 + (q/2)(cos(2x) - 1) for small q
        ce0 = 1.0 + (q / 2.0) * (tf.cos(2.0 * x) - 1.0)
        basis.append(ce0)

        for n in range(1, self.max_order + 1):
            n_f = tf.cast(n, tf.float64)

            # ce_n(x, q) ≈ cos(nx) + q/(4n²-4) * (cos((n+2)x) + cos((n-2)x))
            # Simplified approximation for moderate q
            ce_n = tf.cos(n_f * x)
            if n >= 2:
                correction = (q / (4.0 * n_f * n_f - 4.0)) * (
                    tf.cos((n_f + 2.0) * x) + tf.cos((n_f - 2.0) * x)
                )
                ce_n = ce_n + correction
            basis.append(ce_n)

            # se_n(x, q) ≈ sin(nx) + q/(4n²-4) * (sin((n+2)x) - sin((n-2)x))
            se_n = tf.sin(n_f * x)
            if n >= 2:
                correction = (q / (4.0 * n_f * n_f - 4.0)) * (
                    tf.sin((n_f + 2.0) * x) - tf.sin((n_f - 2.0) * x)
                )
                se_n = se_n + correction
            basis.append(se_n)

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "q": float(self.initial_q),
            "learnable_q": self.learnable_q,
        })
        return config
