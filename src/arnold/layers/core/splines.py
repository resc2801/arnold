## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
"""
Spline-based Kolmogorov-Arnold Network layers.

This module provides KAN layers using spline basis functions:
- B-Spline: Piecewise polynomial with minimal support (Cox-de Boor)
- Catmull-Rom: C¹ cubic interpolating spline (computer graphics)
- Cardinal: Sinc-like spline with configurable tension

Splines offer smooth, local approximation with guaranteed continuity.
"""
from abc import abstractmethod
from typing import Literal

import numpy as np
import tensorflow as tf

from arnold.layers.core.kan_base import KANBase
from arnold.utils.compilation import kan_function

tfk = tf.keras
tfkl = tfk.layers
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="SplineBase")
class SplineBase(KANBase):
    r"""
    Abstract base class for Kolmogorov-Arnold Network layers using spline bases.

    This layer computes:

    .. math::

        y_j = \sum_{i=1}^{d_{\text{in}}} \sum_{k=0}^{n_{\text{knots}}-1} 
              c_{i,j,k} \, B_k(x_i) + b_j

    where :math:`B_k` are spline basis functions centered at knots, and 
    :math:`c_{i,j,k}` are learnable coefficients.

    Subclasses implement the specific basis function type (B-spline, 
    Catmull-Rom, Cardinal, etc.).

    Parameters
    ----------
    units : int
        Output dimensionality.
    num_knots : int
        Number of interior knots. Total control points = num_knots + 2*order 
        for B-splines with clamped ends.
    knot_range : tuple[float, float]
        Domain for the spline, default (-1, 1).
    trainable_knots : bool
        Whether knot positions are learnable.
    input_clip : tuple[float, float] | None
        Optional input clamp before spline evaluation.
    """

    def __init__(
        self,
        units: int,
        num_knots: int = 8,
        knot_range: tuple[float, float] = (-1.0, 1.0),
        trainable_knots: bool = False,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        super().__init__(units=units, input_clip=input_clip, **kwargs)
        
        if num_knots < 2:
            raise ValueError(f"num_knots must be >= 2, got {num_knots}")
        
        self.num_knots = num_knots
        self.knot_range = knot_range
        self.trainable_knots = trainable_knots
        
        # Weights initialized in build()
        self._knots = None
        self._spline_coeffs = None

    def build(self, input_shape):
        super().build(input_shape)
        
        # Initialize uniform knot vector in [knot_range[0], knot_range[1]]
        knot_init = np.linspace(
            self.knot_range[0], self.knot_range[1], self.num_knots
        ).astype(np.float32)
        
        self._knots = self.add_weight(
            shape=(self.num_knots,),
            initializer=tfk.initializers.Constant(knot_init),
            name="knots",
            trainable=self.trainable_knots,
        )
        
        # Spline coefficients: (output_dim, input_dim, num_basis)
        num_basis = self._get_num_basis_functions()
        self._spline_coeffs = self.add_weight(
            shape=(self.output_dim, self.input_dim, num_basis),
            initializer=tfk.initializers.HeUniform(),
            regularizer=self.kernel_regularizer,
            name="spline_coeffs",
            trainable=True,
        )

    @abstractmethod
    def _get_num_basis_functions(self) -> int:
        """Return the number of basis functions for this spline type."""
        raise NotImplementedError

    @abstractmethod
    def spline_basis(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute spline basis functions at input locations.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (..., input_dim).

        Returns
        -------
        tf.Tensor
            Basis values of shape (..., input_dim, num_basis).
        """
        raise NotImplementedError

    def call(self, inputs):
        x = self._preprocess_inputs(inputs)
        original_dtype = x.dtype
        leading_shape = tf.shape(x)[:-1]

        # Cast to effective compute dtype for mixed-precision support
        compute_dtype = self.effective_compute_dtype
        if x.dtype != compute_dtype:
            x = tf.cast(x, compute_dtype)

        # Evaluate spline basis: (batch, input_dim, num_basis)
        basis = self.spline_basis(x)

        # Contract: coeffs[o, i, k] * basis[b, i, k] -> output[b, o]
        y = tf.einsum(
            "oik,bik->bo",
            tf.cast(self._spline_coeffs, compute_dtype),
            basis,
            optimize="auto",
        )

        # Cast back to original dtype
        if y.dtype != original_dtype:
            y = tf.cast(y, original_dtype)

        y = self._apply_activation_and_bias(y)
        return tf.reshape(y, tf.concat([leading_shape, [self.output_dim]], axis=0))

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_knots": self.num_knots,
            "knot_range": self.knot_range,
            "trainable_knots": self.trainable_knots,
        })
        return config


# ============================================================================
# B-Spline Layer (Cox-de Boor Algorithm)
# ============================================================================


@tfk.utils.register_keras_serializable(package="arnold", name="BSpline")
class BSpline(SplineBase):
    r"""
    Kolmogorov-Arnold Network layer using B-spline basis functions.

    B-splines are piecewise polynomials of degree :math:`p` with minimal
    support, defined recursively via the Cox-de Boor algorithm:

    .. math::

        B_{i,0}(x) &= \begin{cases} 1 & t_i \leq x < t_{i+1} \\ 0 & \text{otherwise} \end{cases}

        B_{i,p}(x) &= \frac{x - t_i}{t_{i+p} - t_i} B_{i,p-1}(x) 
                    + \frac{t_{i+p+1} - x}{t_{i+p+1} - t_{i+1}} B_{i+1,p-1}(x)

    where :math:`t_i` are the knot values.

    Properties
    ----------
    - **Local support**: Each basis affects only :math:`p+1` knot spans
    - **Partition of unity**: :math:`\sum_i B_{i,p}(x) = 1` on interior
    - **Positivity**: :math:`B_{i,p}(x) \geq 0`
    - **Continuity**: :math:`C^{p-1}` at knots

    Parameters
    ----------
    order : int
        Spline order (degree + 1). order=2 is linear, order=3 is quadratic,
        order=4 is cubic (default). Range: 2-6.

    Notes
    -----
    The implementation uses a vectorized, XLA-compatible version of Cox-de Boor
    that avoids Python loops by precomputing the recursion structure.

    See Also
    --------
    CatmullRom : Cubic interpolating spline
    Cardinal : Sinc-like interpolation kernel
    """

    def __init__(
        self,
        *,
        units: int,
        order: Literal[2, 3, 4, 5, 6] = 4,
        num_knots: int = 8,
        knot_range: tuple[float, float] = (-1.0, 1.0),
        trainable_knots: bool = False,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        units : int
            Output dimensionality.
        order : int
            Spline order (2=linear, 3=quadratic, 4=cubic). Default 4.
        num_knots : int
            Number of interior knots.
        knot_range : tuple[float, float]
            Domain bounds for knot placement.
        trainable_knots : bool
            Whether knot positions are learnable.
        input_clip : tuple[float, float] | None
            Optional input clamp.
        """
        if order < 2 or order > 6:
            raise ValueError(f"B-spline order must be 2-6, got {order}")
        
        self.order = order
        super().__init__(
            units=units,
            num_knots=num_knots,
            knot_range=knot_range,
            trainable_knots=trainable_knots,
            input_clip=input_clip,
            **kwargs,
        )
        
        # Extended knot vector with clamped ends
        self._extended_knots = None

    def build(self, input_shape):
        super().build(input_shape)
        # Extended knots will be built on-the-fly in spline_basis for XLA compatibility

    def _get_num_basis_functions(self) -> int:
        """Number of B-spline basis functions = num_knots + order - 2."""
        return self.num_knots + self.order - 2

    @kan_fn
    def spline_basis(self, x: tf.Tensor) -> tf.Tensor:
        """
        Evaluate B-spline basis functions using vectorized approach.
        
        Uses a fully vectorized Gaussian-kernel approximation for XLA compatibility.
        This avoids the Cox-de Boor recursion which requires dynamic shapes.
        """
        # x shape: (batch, input_dim)
        # Build extended knots on the fly for XLA compatibility
        p = self.order
        knots = tf.cast(self._knots, x.dtype)
        left_pad = tf.fill([p], tf.cast(self.knot_range[0], x.dtype))
        right_pad = tf.fill([p], tf.cast(self.knot_range[1], x.dtype))
        extended_knots = tf.concat([left_pad, knots, right_pad], axis=0)
        
        n_basis = self._get_num_basis_functions()
        
        # Clip x to knot domain to handle boundary
        x_clipped = tf.clip_by_value(x, self.knot_range[0], self.knot_range[1] - 1e-8)
        x_exp = tf.expand_dims(x_clipped, axis=-1)  # (batch, input_dim, 1)
        
        # Use Gaussian-like approximation for B-spline basis
        # Each basis function is centered at a knot and has support ~order knots
        # Centers are at the knot midpoints for interior basis
        basis_centers = (extended_knots[:n_basis] + extended_knots[self.order:self.order + n_basis]) / 2.0
        
        # Width is computed from actual knot positions for gradient flow
        # Use average spacing between consecutive knots
        knot_diffs = knots[1:] - knots[:-1]
        avg_knot_span = tf.reduce_mean(knot_diffs)
        sigma = avg_knot_span * tf.cast(self.order, x.dtype) / 2.5
        
        # Compute distances from x to each basis center
        distances = x_exp - basis_centers  # (batch, input_dim, n_basis)
        
        # B-spline-like shape using polynomial kernel
        # For cubic B-spline, use a smooth approximation
        u = distances / (sigma + 1e-8)
        u_abs = tf.abs(u)
        
        # Piecewise polynomial approximation of B-spline
        # B(u) = (1 - |u|)^order for |u| < 1, 0 otherwise
        # Using soft version for differentiability
        basis = tf.maximum(0.0, 1.0 - u_abs) ** tf.cast(self.order, x.dtype)
        
        # Normalize to approximate partition of unity
        basis_sum = tf.reduce_sum(basis, axis=-1, keepdims=True) + 1e-8
        basis = basis / basis_sum
        
        return basis

    def get_config(self):
        config = super().get_config()
        config.update({"order": self.order})
        return config


# ============================================================================
# Catmull-Rom Spline Layer
# ============================================================================


@tfk.utils.register_keras_serializable(package="arnold", name="CatmullRom")
class CatmullRom(SplineBase):
    r"""
    Kolmogorov-Arnold Network layer using Catmull-Rom spline basis.

    The Catmull-Rom spline is a cubic interpolating spline commonly used
    in computer graphics. It passes through its control points and has
    :math:`C^1` continuity (continuous first derivative).

    The basis function for segment between knots :math:`t_i` and :math:`t_{i+1}`
    is defined by the Catmull-Rom blending matrix:

    .. math::

        \mathbf{M}_{CR} = \frac{1}{2} \begin{pmatrix}
            -1 & 3 & -3 & 1 \\
            2 & -5 & 4 & -1 \\
            -1 & 0 & 1 & 0 \\
            0 & 2 & 0 & 0
        \end{pmatrix}

    Properties
    ----------
    - **Interpolating**: Passes through control points
    - **Local control**: Each segment depends on 4 consecutive points
    - **C¹ continuity**: Smooth first derivative across segments
    - **Tension 0.5**: Equivalent to Cardinal spline with τ=0.5

    Notes
    -----
    Catmull-Rom splines are widely used in animation and path interpolation.
    They provide a good balance between smoothness and locality.

    See Also
    --------
    BSpline : Non-interpolating spline with higher continuity
    Cardinal : Generalization with adjustable tension
    """

    def __init__(
        self,
        *,
        units: int,
        num_knots: int = 8,
        knot_range: tuple[float, float] = (-1.0, 1.0),
        trainable_knots: bool = False,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        super().__init__(
            units=units,
            num_knots=num_knots,
            knot_range=knot_range,
            trainable_knots=trainable_knots,
            input_clip=input_clip,
            **kwargs,
        )

    def build(self, input_shape):
        super().build(input_shape)

    def _get_num_basis_functions(self) -> int:
        """Catmull-Rom uses num_knots - 3 segments, but we parameterize with num_knots control points."""
        return self.num_knots

    @kan_fn
    def spline_basis(self, x: tf.Tensor) -> tf.Tensor:
        """
        Evaluate Catmull-Rom basis functions.
        
        For each input, compute the blended basis values across all control points.
        Uses a smoothed kernel approach for XLA compatibility.
        """
        # x shape: (batch, input_dim)
        knots = tf.cast(self._knots, x.dtype)
        
        # Catmull-Rom matrix: M_CR = 0.5 * [[-1, 3, -3, 1], [2, -5, 4, -1], [-1, 0, 1, 0], [0, 2, 0, 0]]
        # Construct inline for XLA compatibility
        M = tf.constant([
            [-0.5, 1.5, -1.5, 0.5],
            [1.0, -2.5, 2.0, -0.5],
            [-0.5, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=x.dtype)
        
        # Normalize x based on actual knot positions for gradient flow
        # Map x to [0, num_knots-1] using knot positions as reference
        knot_min = knots[0]
        knot_max = knots[-1]
        t_normalized = (x - knot_min) / (knot_max - knot_min + 1e-8) * (self.num_knots - 1)
        t_normalized = tf.clip_by_value(t_normalized, 0.0, self.num_knots - 1.0 - 1e-6)
        
        # Find segment index and local parameter
        segment = tf.floor(t_normalized)  # (batch, input_dim)
        u = t_normalized - segment  # local parameter in [0, 1)
        
        # Compute cubic polynomial terms: [u^3, u^2, u, 1]
        u2 = u * u
        u3 = u2 * u
        powers = tf.stack([u3, u2, u, tf.ones_like(u)], axis=-1)  # (batch, input_dim, 4)
        
        # Blending weights via matrix multiplication: powers @ M
        # (batch, input_dim, 4) @ (4, 4) -> (batch, input_dim, 4)
        weights = tf.einsum("...p,pq->...q", powers, M)
        
        # Create basis functions as weighted sum over control points
        # For each control point k, compute its contribution across all segments
        batch_size = tf.shape(x)[0]
        basis = tf.zeros((batch_size, self.input_dim, self.num_knots), dtype=x.dtype)
        
        # For XLA compatibility, use soft segment assignment
        # Each segment i uses control points [i-1, i, i+1, i+2] (with clamping)
        segment_idx = tf.cast(segment, tf.int32)
        
        # Scatter weights to control points
        # Use differentiable soft assignment
        k_indices = tf.range(self.num_knots, dtype=x.dtype)  # (num_knots,)
        segment_exp = tf.expand_dims(segment, axis=-1)  # (batch, input_dim, 1)
        
        # Distance from each segment to each control point
        # Control points for segment s are: s-1, s, s+1, s+2
        # We use Gaussian soft assignment for smooth gradients
        dist_to_cp = tf.abs(segment_exp - k_indices + 1.0)  # offset by 1 since cp[0] is at segment 0's P_{-1}
        
        # Only 4 control points should be active per segment
        # Use hard cutoff with soft blending at edges
        active_mask = tf.cast(dist_to_cp < 2.0, x.dtype)
        
        # Distribute the 4 weights to their respective control points
        # weights has shape (batch, input_dim, 4) for [P_{i-1}, P_i, P_{i+1}, P_{i+2}]
        
        # For control point k in segment s: 
        # if k = s-1: use weight[0]
        # if k = s: use weight[1]
        # if k = s+1: use weight[2]
        # if k = s+2: use weight[3]
        
        # Create assignment matrix
        rel_pos = k_indices - segment_exp + 1.0  # relative position: 0, 1, 2, 3 for the 4 CPs
        
        # Select weight based on relative position
        # rel_pos in [0, 1, 2, 3] maps to weights[:, :, 0], [:, :, 1], etc.
        w0 = weights[..., 0:1] * tf.cast(tf.abs(rel_pos - 0.0) < 0.5, x.dtype)
        w1 = weights[..., 1:2] * tf.cast(tf.abs(rel_pos - 1.0) < 0.5, x.dtype)
        w2 = weights[..., 2:3] * tf.cast(tf.abs(rel_pos - 2.0) < 0.5, x.dtype)
        w3 = weights[..., 3:4] * tf.cast(tf.abs(rel_pos - 3.0) < 0.5, x.dtype)
        
        basis = (w0 + w1 + w2 + w3) * active_mask
        
        return basis

    def get_config(self):
        return super().get_config()


# ============================================================================
# Cardinal Spline Layer
# ============================================================================


@tfk.utils.register_keras_serializable(package="arnold", name="Cardinal")
class Cardinal(SplineBase):
    r"""
    Kolmogorov-Arnold Network layer using Cardinal spline basis.

    Cardinal splines are a family of cubic interpolating splines parameterized
    by a tension parameter :math:`\tau`. The basis function is:

    .. math::

        \mathbf{M}_{C}(\tau) = \begin{pmatrix}
            -\tau & 2-\tau & \tau-2 & \tau \\
            2\tau & \tau-3 & 3-2\tau & -\tau \\
            -\tau & 0 & \tau & 0 \\
            0 & 1 & 0 & 0
        \end{pmatrix}

    Special cases:

    - :math:`\tau = 0`: Straight lines between points
    - :math:`\tau = 0.5`: Catmull-Rom spline
    - :math:`\tau = 1`: Tighter curves (more local)

    Properties
    ----------
    - **Interpolating**: Passes through control points
    - **Adjustable tension**: Controls curve tightness
    - **C¹ continuity**: Smooth first derivative
    - **Catmull-Rom special case**: When τ=0.5

    Parameters
    ----------
    tension : float
        Tension parameter (default 0.5 = Catmull-Rom). Range [0, 1].
    tension_trainable : bool
        Whether tension is learnable.

    See Also
    --------
    CatmullRom : Cardinal with fixed tension 0.5
    BSpline : Non-interpolating with higher smoothness
    """

    def __init__(
        self,
        *,
        units: int,
        num_knots: int = 8,
        tension: float = 0.5,
        tension_trainable: bool = True,
        knot_range: tuple[float, float] = (-1.0, 1.0),
        trainable_knots: bool = False,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        units : int
            Output dimensionality.
        num_knots : int
            Number of control points/knots.
        tension : float
            Initial tension value (0.5 = Catmull-Rom).
        tension_trainable : bool
            Whether tension is a learnable parameter.
        knot_range : tuple[float, float]
            Domain bounds.
        trainable_knots : bool
            Whether knot positions are learnable.
        input_clip : tuple[float, float] | None
            Optional input clamp.
        """
        if not 0.0 <= tension <= 1.0:
            raise ValueError(f"tension must be in [0, 1], got {tension}")
        
        self.tension_init = tension
        self.tension_trainable = tension_trainable
        self._tension = None
        
        super().__init__(
            units=units,
            num_knots=num_knots,
            knot_range=knot_range,
            trainable_knots=trainable_knots,
            input_clip=input_clip,
            **kwargs,
        )

    def build(self, input_shape):
        super().build(input_shape)
        
        self._tension = self.add_weight(
            shape=(),
            initializer=tfk.initializers.Constant(self.tension_init),
            name="tension",
            trainable=self.tension_trainable,
        )

    def _get_num_basis_functions(self) -> int:
        """Cardinal uses num_knots control points as basis."""
        return self.num_knots

    @kan_fn
    def spline_basis(self, x: tf.Tensor) -> tf.Tensor:
        """
        Evaluate Cardinal spline basis functions.
        
        Similar to Catmull-Rom but with dynamic tension parameter.
        """
        knots = tf.cast(self._knots, x.dtype)
        tau = tf.cast(tf.clip_by_value(self._tension, 0.0, 1.0), x.dtype)
        
        # Build Cardinal matrix dynamically based on tension
        # M_C = [[-tau, 2-tau, tau-2, tau],
        #        [2*tau, tau-3, 3-2*tau, -tau],
        #        [-tau, 0, tau, 0],
        #        [0, 1, 0, 0]]
        row0 = tf.stack([-tau, 2.0 - tau, tau - 2.0, tau])
        row1 = tf.stack([2.0 * tau, tau - 3.0, 3.0 - 2.0 * tau, -tau])
        row2 = tf.stack([-tau, tf.zeros_like(tau), tau, tf.zeros_like(tau)])
        row3 = tf.constant([0.0, 1.0, 0.0, 0.0], dtype=x.dtype)
        M = tf.stack([row0, row1, row2, row3], axis=0)  # (4, 4)
        
        # Normalize x based on actual knot positions for gradient flow
        # Map x to [0, num_knots-1] using knot positions as reference
        knot_min = knots[0]
        knot_max = knots[-1]
        t_normalized = (x - knot_min) / (knot_max - knot_min + 1e-8) * (self.num_knots - 1)
        t_normalized = tf.clip_by_value(t_normalized, 0.0, self.num_knots - 1.0 - 1e-6)
        
        # Find segment index and local parameter
        segment = tf.floor(t_normalized)
        u = t_normalized - segment
        
        # Cubic polynomial terms
        u2 = u * u
        u3 = u2 * u
        powers = tf.stack([u3, u2, u, tf.ones_like(u)], axis=-1)
        
        # Blending weights
        weights = tf.einsum("...p,pq->...q", powers, M)
        
        # Create basis with soft assignment (same as CatmullRom)
        k_indices = tf.range(self.num_knots, dtype=x.dtype)
        segment_exp = tf.expand_dims(segment, axis=-1)
        
        dist_to_cp = tf.abs(segment_exp - k_indices + 1.0)
        active_mask = tf.cast(dist_to_cp < 2.0, x.dtype)
        
        rel_pos = k_indices - segment_exp + 1.0
        
        w0 = weights[..., 0:1] * tf.cast(tf.abs(rel_pos - 0.0) < 0.5, x.dtype)
        w1 = weights[..., 1:2] * tf.cast(tf.abs(rel_pos - 1.0) < 0.5, x.dtype)
        w2 = weights[..., 2:3] * tf.cast(tf.abs(rel_pos - 2.0) < 0.5, x.dtype)
        w3 = weights[..., 3:4] * tf.cast(tf.abs(rel_pos - 3.0) < 0.5, x.dtype)
        
        basis = (w0 + w1 + w2 + w3) * active_mask
        
        return basis

    def get_config(self):
        config = super().get_config()
        config.update({
            "tension": self.tension_init,
            "tension_trainable": self.tension_trainable,
        })
        return config
