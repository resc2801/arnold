## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
from abc import abstractmethod

import tensorflow as tf

from arnold.layers.core.kan_base import KANBase, detect_hardware


# Lazy import to avoid circular dependency
def _get_optimized_ops():
    """Lazily import optimized_ops to avoid circular imports."""
    from arnold.utils import optimized_ops
    return optimized_ops


tfk = tf.keras
tfkl = tfk.layers


@tfk.utils.register_keras_serializable(package="arnold", name="PolynomialBase")
class PolynomialBase(KANBase):
    r"""
    Abstract base class for Kolmogorov-Arnold Network layers using polynomial bases.

    This layer implements the core KAN computation:

    .. math::

        y_j = \sum_{i=1}^{d_{\text{in}}} \sum_{k=0}^{n} c_{i,k,j} \, P_k(x_i) + b_j

    where :math:`P_k` denotes the degree-:math:`k` polynomial basis function,
    :math:`c_{i,k,j}` are learnable coefficients, and :math:`b_j` is an optional bias.

    Supports mixed-precision training via ``tf.keras.mixed_precision`` and
    hardware-adaptive dtype selection for optimal performance.

    Notes
    -----
    For high polynomial degrees (> 10), float64 computation may be beneficial
    on CPU to avoid numerical instability. Use ``promote_to_float64=True`` or
    rely on ``hardware_adaptive=True`` (default) for automatic selection.

    For memory-constrained scenarios, Tucker decomposition can significantly
    reduce parameter count:

    .. math::

        c_{i,k,j} \approx \sum_{r_1, r_2, r_3} G_{r_1, r_2, r_3} \, A_{i,r_1} \, B_{k,r_2} \, C_{j,r_3}

    where :math:`G` is a small core tensor and :math:`A, B, C` are factor matrices.
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        core_ranks: None | tuple[int, int, int] = None,
        input_clip: tuple[float, float] | None = None,
        promote_to_float64: bool | None = None,
        precision_threshold: int = 10,
        use_clenshaw: bool | None = None,
        use_true_clenshaw: bool = False,
        enable_tpu_sharding: bool = False,
        hardware_adaptive: bool = True,
        **kwargs,
    ):
        """
        :param degree: Maximum degree of the polynomial basis.
        :param core_ranks: Optional Tucker ranks (r1, r2, r3) for decomposed coefficient tensor.
        :param input_clip: Optional (min, max) tuple for clamping inputs prior to basis evaluation.
        :param promote_to_float64: Whether to evaluate basis/contract in float64 when degree exceeds 
            ``precision_threshold`` (cast back to original dtype afterward). If None (default),
            uses hardware-adaptive selection when ``hardware_adaptive=True``.
        :param precision_threshold: Degree threshold above which promotion is considered.
        :param use_clenshaw: Force use of ``clenshaw_basis`` when True, force pseudo-Vandermonde 
            when False; ``None`` auto-selects (Clenshaw for high degree).
        :param use_true_clenshaw: When True, use true Clenshaw summation that fuses basis evaluation
            with coefficient contraction, achieving O(1) memory per degree step. Only supported for
            Chebyshev and Legendre polynomials currently. Requires ``core_ranks=None``.
        :param enable_tpu_sharding: When True, apply TPU sharding annotations for SPMD partitioning.
            This hints to XLA to shard the batch dimension across TPU cores.
        :param hardware_adaptive: If True and ``promote_to_float64=None``, automatically select
            optimal dtype based on detected hardware (CPU: float64 for high degree, GPU/MPS: float32).
        """
        super().__init__(units=units, input_clip=input_clip, **kwargs)

        self.degree = degree
        self.core_ranks = core_ranks
        self.precision_threshold = precision_threshold
        self.use_clenshaw = use_clenshaw
        self.use_true_clenshaw = use_true_clenshaw
        self.enable_tpu_sharding = enable_tpu_sharding
        self.hardware_adaptive = hardware_adaptive
        
        # Handle promote_to_float64 with hardware-adaptive default
        if promote_to_float64 is None and hardware_adaptive:
            # Auto-select based on hardware
            hardware = detect_hardware()
            # Only promote on CPU where float64 is efficient
            self.promote_to_float64 = (hardware == "cpu" and degree > precision_threshold)
            self._detected_hardware = hardware
        else:
            self.promote_to_float64 = promote_to_float64 if promote_to_float64 is not None else True
            self._detected_hardware = None
        
        self.poly_coeffs = None
        self.poly_coeffs_core = None
        self.poly_coeffs_A = None
        self.poly_coeffs_B = None
        self.poly_coeffs_C = None

    def build(self, input_shape):
        super().build(input_shape)

        stddev = 1.0 / (float(self.input_dim) * float(self.degree + 1))
        initializer = tfk.initializers.RandomNormal(mean=0.0, stddev=stddev)

        if self.core_ranks is not None:
            self.r1, self.r2, self.r3 = self.core_ranks
            self.poly_coeffs_core = self.add_weight(
                shape=(self.r1, self.r2, self.r3),
                initializer=initializer,
                regularizer=self.kernel_regularizer,
                trainable=True,
                name="polynomial_coefficients_core",
            )
            self.poly_coeffs_A = self.add_weight(
                shape=(self.input_dim, self.r1),
                initializer=initializer,
                regularizer=self.kernel_regularizer,
                trainable=True,
                name="polynomial_coefficients_input_dim",
            )
            self.poly_coeffs_B = self.add_weight(
                shape=(self.degree + 1, self.r2),
                initializer=initializer,
                regularizer=self.kernel_regularizer,
                trainable=True,
                name="polynomial_coefficients_degree",
            )
            self.poly_coeffs_C = self.add_weight(
                shape=(self.output_dim, self.r3),
                initializer=initializer,
                regularizer=self.kernel_regularizer,
                trainable=True,
                name="polynomial_coefficients_output_dim",
            )
        else:
            self.poly_coeffs = self.add_weight(
                shape=(self.input_dim, self.degree + 1, self.output_dim),
                initializer=initializer,
                regularizer=self.kernel_regularizer,
                trainable=True,
                name="polynomial_coefficients",
            )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def call(self, inputs):
        """
        Forward computation: evaluate basis and combine with coefficients.
        
        Respects mixed-precision policy from ``tf.keras.mixed_precision``.
        
        Supports three evaluation modes:
        1. Standard: Compute full pseudo-Vandermonde, then contract with coefficients
        2. Clenshaw basis: Use tf.scan-based recurrence (memory efficient for high degree)
        3. True Clenshaw: Fuse basis evaluation with coefficient contraction (O(1) memory per step)
        """
        x = self._preprocess_inputs(inputs)
        original_dtype = x.dtype

        # Flatten all leading dims for evaluation, then reshape back.
        leading_shape = tf.shape(x)[:-1]
        x_flat = tf.reshape(x, (-1, self.input_dim))

        # Apply TPU sharding annotation if enabled
        if self.enable_tpu_sharding:
            optimized_ops = _get_optimized_ops()
            x_flat = optimized_ops.with_tpu_sharding(x_flat, "batch")

        # Determine compute dtype: respect mixed-precision policy
        compute_dtype = self.effective_compute_dtype
        
        # Decide whether to promote to float64 for numerical stability
        promote = (
            self.promote_to_float64 
            and self.degree > self.precision_threshold 
            and compute_dtype in (tf.float32, tf.float16, tf.bfloat16)
        )
        
        if promote:
            x_eval = tf.cast(x_flat, tf.float64)
        elif x_flat.dtype != compute_dtype:
            x_eval = tf.cast(x_flat, compute_dtype)
        else:
            x_eval = x_flat

        # True Clenshaw path: fused basis-coefficient evaluation (N1 optimization)
        if self.use_true_clenshaw and self.core_ranks is None:
            y_flat = self._true_clenshaw_forward(x_eval, promote, compute_dtype)
        else:
            # Standard path: compute basis first, then contract
            # Choose evaluation path: Clenshaw for higher degrees if overridden.
            use_clenshaw = self.use_clenshaw
            if use_clenshaw is None:
                use_clenshaw = self.degree > 10

            if use_clenshaw:
                basis = self.clenshaw_basis(x_eval)
            else:
                basis = self.pseudo_vandermonde(x_eval)  # shape (B_flat, input_dim, degree+1)

            if self.core_ranks is not None:
                coeff_dtype = tf.float64 if promote else compute_dtype
                coeff_core = tf.cast(self.poly_coeffs_core, coeff_dtype) if self.poly_coeffs_core.dtype != coeff_dtype else self.poly_coeffs_core
                coeff_A = tf.cast(self.poly_coeffs_A, coeff_dtype) if self.poly_coeffs_A.dtype != coeff_dtype else self.poly_coeffs_A
                coeff_B = tf.cast(self.poly_coeffs_B, coeff_dtype) if self.poly_coeffs_B.dtype != coeff_dtype else self.poly_coeffs_B
                coeff_C = tf.cast(self.poly_coeffs_C, coeff_dtype) if self.poly_coeffs_C.dtype != coeff_dtype else self.poly_coeffs_C
                
                # Apply TPU sharding to coefficients
                if self.enable_tpu_sharding:
                    optimized_ops = _get_optimized_ops()
                    coeff_core = optimized_ops.with_tpu_sharding(coeff_core, "replicated")
                    coeff_A = optimized_ops.with_tpu_sharding(coeff_A, "replicated")
                    coeff_B = optimized_ops.with_tpu_sharding(coeff_B, "replicated")
                    coeff_C = optimized_ops.with_tpu_sharding(coeff_C, "replicated")
                
                # Tucker decomposition: coeffs[i,d,o] ≈ Σ_xyz core[x,y,z] * A[i,x] * B[d,y] * C[o,z]
                y_flat = tf.einsum(
                    "bid,xyz,ix,dy,oz->bo",
                    basis,
                    coeff_core,
                    coeff_A,
                    coeff_B,
                    coeff_C,
                    optimize="auto",
                )
            else:
                coeff_dtype = tf.float64 if promote else compute_dtype
                coeffs = tf.cast(self.poly_coeffs, coeff_dtype) if self.poly_coeffs.dtype != coeff_dtype else self.poly_coeffs
                
                # Apply TPU sharding to coefficients
                if self.enable_tpu_sharding:
                    optimized_ops = _get_optimized_ops()
                    coeffs = optimized_ops.with_tpu_sharding(coeffs, "replicated")
                
                y_flat = tf.einsum(
                    "bid,ido->bo",
                    basis,
                    coeffs,
                    optimize="auto",
                )

        # Cast back to original dtype for output
        if y_flat.dtype != original_dtype:
            y_flat = tf.cast(y_flat, original_dtype)

        y_flat = self._apply_activation_and_bias(y_flat)
        output = tf.reshape(y_flat, tf.concat([leading_shape, [self.output_dim]], axis=0))
        return output

    def _true_clenshaw_forward(self, x: tf.Tensor, promote: bool, compute_dtype) -> tf.Tensor:
        """
        True Clenshaw summation: fuses basis evaluation with coefficient contraction.
        
        This achieves O(1) memory per degree step by not materializing the full basis tensor.
        Subclasses should override this if they support true Clenshaw for their polynomial type.
        
        Default implementation falls back to standard evaluation.
        """
        # Default: fall back to standard evaluation
        use_clenshaw = self.use_clenshaw
        if use_clenshaw is None:
            use_clenshaw = self.degree > 10

        if use_clenshaw:
            basis = self.clenshaw_basis(x)
        else:
            basis = self.pseudo_vandermonde(x)
        
        coeff_dtype = tf.float64 if promote else compute_dtype
        coeffs = tf.cast(self.poly_coeffs, coeff_dtype) if self.poly_coeffs.dtype != coeff_dtype else self.poly_coeffs
        
        return tf.einsum("bid,ido->bo", basis, coeffs, optimize="auto")

    @abstractmethod
    def pseudo_vandermonde(self, x):
        """
        Computes the pseudo-Vandermonde matrix for given `x`.

        :param x: Data to compute the pseudo-vandermonde tensor with.
        :type x: tf.Tensor

        :returns: pseudo-vandermonde tensor
        :rtype: tf.Tensor
        """
        raise NotImplementedError(
            f"Layer {self.__class__.__name__} does not have a `pseudo_vandermonde()` method implemented."
        )

    def clenshaw_basis(self, x: tf.Tensor) -> tf.Tensor:
        """
        Optional fused evaluation for high-degree polynomials.

        By default, falls back to :meth:`pseudo_vandermonde`. Subclasses can
        override to provide a memory-friendly Clenshaw implementation.
        """
        return self.pseudo_vandermonde(x)

    def true_clenshaw_eval(self, x: tf.Tensor, coefficients: tf.Tensor) -> tf.Tensor:
        """
        True Clenshaw summation with fused coefficient contraction.

        This method provides O(1) memory evaluation by fusing basis computation
        with coefficient contraction, avoiding materialization of the full basis
        tensor. This is critical for high-degree polynomials and TPU/XLA efficiency.

        Subclasses should override this method to provide polynomial-specific
        Clenshaw recurrence. The default implementation falls back to the
        standard basis + einsum approach.

        :param x: Input tensor of shape (batch, input_dim).
        :type x: tf.Tensor
        :param coefficients: Coefficient tensor of shape (input_dim, degree+1, output_dim).
        :type coefficients: tf.Tensor
        :returns: Output tensor of shape (batch, output_dim).
        :rtype: tf.Tensor

        .. note::
            When overriding, use ``tf.while_loop`` with fused accumulation:

            .. code-block:: python

                def true_clenshaw_eval(self, x, coefficients):
                    # Clenshaw recurrence: b_{n+1} = 0, b_n = c_n
                    # b_{k-1} = c_{k-1} + 2*x*b_k - b_{k+1}
                    # Result: P(x) = b_0 - x*b_1 (for Chebyshev)
                    ...

        .. seealso::
            :func:`arnold.utils.optimized_ops.true_clenshaw_chebyshev`
            :func:`arnold.utils.optimized_ops.true_clenshaw_legendre`
        """
        # Default: fall back to standard evaluation (no true Clenshaw benefit)
        basis = self.clenshaw_basis(x)
        return tf.einsum("bid,ido->bo", basis, coefficients, optimize="auto")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "degree": self.degree,
                "core_ranks": self.core_ranks,
                "input_clip": self.input_clip,
                "promote_to_float64": self.promote_to_float64,
                "precision_threshold": self.precision_threshold,
                "use_clenshaw": self.use_clenshaw,
                "use_true_clenshaw": self.use_true_clenshaw,
                "enable_tpu_sharding": self.enable_tpu_sharding,
                "hardware_adaptive": self.hardware_adaptive,
            }
        )
        return config
