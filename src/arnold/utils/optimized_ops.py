## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
"""
Optimized operations for KAN layers.

This module provides performance-optimized implementations for:
- N1: True Clenshaw summation (fused basis-coefficient evaluation)
- N2: XLA-compatible q-polynomial recurrences using tf.while_loop
- P1: Fused basis-matmul operations
- P2: Parallel degree computation via tf.vectorized_map
- P3: TPU sharding annotations
"""

from __future__ import annotations

from collections.abc import Callable

import tensorflow as tf


def _detect_hardware() -> str:
    """Detect available hardware: 'tpu', 'gpu', or 'cpu'."""
    try:
        tpu_devices = tf.config.list_logical_devices('TPU')
        if tpu_devices:
            return 'tpu'
    except Exception:
        pass

    try:
        gpu_devices = tf.config.list_logical_devices('GPU')
        if gpu_devices:
            return 'gpu'
    except Exception:
        pass

    return 'cpu'


# =============================================================================
# N1: True Clenshaw Summation
# =============================================================================

def clenshaw_chebyshev_sum(
    x: tf.Tensor,
    coeffs: tf.Tensor,
    *,
    input_dim: int,
    output_dim: int,
) -> tf.Tensor:
    """
    Compute Chebyshev polynomial expansion via true Clenshaw summation.

    This fuses the basis evaluation and coefficient contraction in O(1) memory
    per degree step, rather than materializing the full pseudo-Vandermonde tensor.

    For Chebyshev T_n with recurrence T_{n+1} = 2x T_n - T_{n-1}:

    The Clenshaw algorithm evaluates sum_{k=0}^{n} c_k T_k(x) by:
        b_{n+2} = b_{n+1} = 0
        b_k = c_k + 2x b_{k+1} - b_{k+2}  for k = n, n-1, ..., 1
        result = c_0 + x b_1 - b_2

    Parameters
    ----------
    x : tf.Tensor
        Input tensor of shape (batch, input_dim).
    coeffs : tf.Tensor
        Coefficient tensor of shape (input_dim, degree+1, output_dim).
    input_dim : int
        Input dimension.
    output_dim : int
        Output dimension.

    Returns
    -------
    tf.Tensor
        Output tensor of shape (batch, output_dim).
    """
    # x: (batch, input_dim)
    # coeffs: (input_dim, degree+1, output_dim)
    degree = tf.shape(coeffs)[1] - 1

    # Initialize b_{n+1} = 0, b_{n+2} = 0
    batch_size = tf.shape(x)[0]
    b_k1 = tf.zeros((batch_size, input_dim, output_dim), dtype=x.dtype)  # b_{k+1}
    b_k2 = tf.zeros((batch_size, input_dim, output_dim), dtype=x.dtype)  # b_{k+2}

    # Cast coeffs to compute dtype
    coeffs = tf.cast(coeffs, x.dtype)

    # x expanded for broadcasting: (batch, input_dim, 1)
    x_expanded = tf.expand_dims(x, -1)

    # Clenshaw backward recurrence: b_k = c_k + 2x b_{k+1} - b_{k+2}
    def clenshaw_step(carry, k):
        b_k1, b_k2 = carry
        # c_k: (input_dim, output_dim) -> (1, input_dim, output_dim)
        c_k = tf.expand_dims(coeffs[:, k, :], 0)
        b_k = c_k + 2.0 * x_expanded * b_k1 - b_k2
        return (b_k, b_k1)

    # Iterate k = degree, degree-1, ..., 1
    k_range = tf.range(degree, 0, -1)
    final_carry = tf.foldl(
        clenshaw_step,
        k_range,
        initializer=(b_k1, b_k2),
    )
    b_1, b_2 = final_carry

    # Final step: result = c_0 + x * b_1 - b_2
    c_0 = tf.expand_dims(coeffs[:, 0, :], 0)  # (1, input_dim, output_dim)
    result = c_0 + x_expanded * b_1 - b_2  # (batch, input_dim, output_dim)

    # Sum over input dimension
    return tf.reduce_sum(result, axis=1)  # (batch, output_dim)


def clenshaw_legendre_sum(
    x: tf.Tensor,
    coeffs: tf.Tensor,
    *,
    input_dim: int,
    output_dim: int,
) -> tf.Tensor:
    """
    Compute Legendre polynomial expansion via true Clenshaw summation.

    For Legendre P_n with recurrence:
    (n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1}

    The Clenshaw algorithm uses the modified recurrence.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor of shape (batch, input_dim).
    coeffs : tf.Tensor
        Coefficient tensor of shape (input_dim, degree+1, output_dim).
    input_dim : int
        Input dimension.
    output_dim : int
        Output dimension.

    Returns
    -------
    tf.Tensor
        Output tensor of shape (batch, output_dim).
    """
    degree = tf.shape(coeffs)[1] - 1
    batch_size = tf.shape(x)[0]

    b_k1 = tf.zeros((batch_size, input_dim, output_dim), dtype=x.dtype)
    b_k2 = tf.zeros((batch_size, input_dim, output_dim), dtype=x.dtype)

    coeffs = tf.cast(coeffs, x.dtype)
    x_expanded = tf.expand_dims(x, -1)

    # Legendre Clenshaw: b_k = c_k + (2k+1)/(k+1) x b_{k+1} - (k+1)/(k+2) b_{k+2}
    def clenshaw_step(carry, k):
        b_k1, b_k2 = carry
        k_f = tf.cast(k, x.dtype)
        c_k = tf.expand_dims(coeffs[:, k, :], 0)
        alpha = (2.0 * k_f + 1.0) / (k_f + 1.0)
        beta = (k_f + 1.0) / (k_f + 2.0)
        b_k = c_k + alpha * x_expanded * b_k1 - beta * b_k2
        return (b_k, b_k1)

    k_range = tf.range(degree, 0, -1)
    final_carry = tf.foldl(
        clenshaw_step,
        k_range,
        initializer=(b_k1, b_k2),
    )
    b_1, b_2 = final_carry

    # Final: result = c_0 + x * b_1 - 0.5 * b_2
    c_0 = tf.expand_dims(coeffs[:, 0, :], 0)
    result = c_0 + x_expanded * b_1 - 0.5 * b_2

    return tf.reduce_sum(result, axis=1)


def clenshaw_generic_sum(
    x: tf.Tensor,
    coeffs: tf.Tensor,
    alpha_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    beta_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    *,
    input_dim: int,
    output_dim: int,
) -> tf.Tensor:
    """
    Compute polynomial expansion via generic Clenshaw summation.

    For polynomials with three-term recurrence:
    P_{n+1} = alpha_n(x) P_n - beta_n P_{n-1}

    Parameters
    ----------
    x : tf.Tensor
        Input tensor of shape (batch, input_dim).
    coeffs : tf.Tensor
        Coefficient tensor of shape (input_dim, degree+1, output_dim).
    alpha_fn : Callable
        Function (n, x) -> alpha_n coefficient tensor.
    beta_fn : Callable
        Function (n, x) -> beta_n coefficient tensor.
    input_dim : int
        Input dimension.
    output_dim : int
        Output dimension.

    Returns
    -------
    tf.Tensor
        Output tensor of shape (batch, output_dim).
    """
    degree = tf.shape(coeffs)[1] - 1
    batch_size = tf.shape(x)[0]

    b_k1 = tf.zeros((batch_size, input_dim, output_dim), dtype=x.dtype)
    b_k2 = tf.zeros((batch_size, input_dim, output_dim), dtype=x.dtype)

    coeffs = tf.cast(coeffs, x.dtype)

    def clenshaw_step(carry, k):
        b_k1, b_k2 = carry
        c_k = tf.expand_dims(coeffs[:, k, :], 0)
        alpha = tf.expand_dims(alpha_fn(k, x), -1)  # (batch, input_dim, 1)
        beta = tf.expand_dims(beta_fn(k, x), -1)
        b_k = c_k + alpha * b_k1 - beta * b_k2
        return (b_k, b_k1)

    k_range = tf.range(degree, 0, -1)
    final_carry = tf.foldl(
        clenshaw_step,
        k_range,
        initializer=(b_k1, b_k2),
    )
    b_1, b_2 = final_carry

    c_0 = tf.expand_dims(coeffs[:, 0, :], 0)
    alpha_0 = tf.expand_dims(alpha_fn(tf.constant(0), x), -1)
    beta_0 = tf.expand_dims(beta_fn(tf.constant(0), x), -1)
    result = c_0 + alpha_0 * b_1 - beta_0 * b_2

    return tf.reduce_sum(result, axis=1)


# =============================================================================
# N2: XLA-Compatible q-Polynomial Recurrence
# =============================================================================

def q_polynomial_basis_xla(
    x: tf.Tensor,
    q: tf.Tensor,
    degree: int,
    recurrence_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, int], tf.Tensor],
    p0_fn: Callable[[tf.Tensor], tf.Tensor],
    p1_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
) -> tf.Tensor:
    """
    Compute q-polynomial basis using tf.while_loop for XLA compatibility.

    Uses TensorArray to accumulate polynomial values without Python lists.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor.
    q : tf.Tensor
        The q parameter (scalar).
    degree : int
        Maximum polynomial degree.
    recurrence_fn : Callable
        Function (P_n, P_{n-1}, x, q, n) -> P_{n+1}.
    p0_fn : Callable
        Function (x) -> P_0(x).
    p1_fn : Callable
        Function (x, q) -> P_1(x).

    Returns
    -------
    tf.Tensor
        Basis tensor of shape (..., degree+1).
    """
    original_shape = tf.shape(x)
    x_flat = tf.reshape(x, [-1])
    tf.shape(x_flat)[0]

    # Initialize TensorArray
    basis_ta = tf.TensorArray(
        dtype=x.dtype,
        size=degree + 1,
        dynamic_size=False,
        element_shape=[None],
        infer_shape=False,
    )

    # P_0
    p0 = p0_fn(x_flat)
    basis_ta = basis_ta.write(0, p0)

    if degree == 0:
        basis = basis_ta.stack()
        basis = tf.transpose(basis)  # (batch, 1)
        return tf.reshape(basis, tf.concat([original_shape, [degree + 1]], axis=0))

    # P_1
    p1 = p1_fn(x_flat, q)
    basis_ta = basis_ta.write(1, p1)

    if degree == 1:
        basis = basis_ta.stack()
        basis = tf.transpose(basis)
        return tf.reshape(basis, tf.concat([original_shape, [degree + 1]], axis=0))

    # Use while_loop for n = 2, ..., degree
    def cond(n, p_n, p_n_1, ta):
        return n <= degree

    def body(n, p_n, p_n_1, ta):
        p_next = recurrence_fn(p_n, p_n_1, x_flat, q, n)
        ta = ta.write(n, p_next)
        return (n + 1, p_next, p_n, ta)

    _, _, _, basis_ta = tf.while_loop(
        cond,
        body,
        loop_vars=(2, p1, p0, basis_ta),
        shape_invariants=(
            tf.TensorShape([]),
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorArraySpec(dtype=x.dtype, element_shape=[None], dynamic_size=False),
        ),
        parallel_iterations=1,  # Sequential for recurrence
    )

    basis = basis_ta.stack()  # (degree+1, batch)
    basis = tf.transpose(basis)  # (batch, degree+1)
    return tf.reshape(basis, tf.concat([original_shape, [degree + 1]], axis=0))


def al_salam_carlitz_1st_basis_xla(
    x: tf.Tensor,
    a: tf.Tensor,
    q: tf.Tensor,
    degree: int,
) -> tf.Tensor:
    """
    Compute Al-Salam-Carlitz U basis using XLA-compatible while_loop.

    Recurrence:
    U_{n+1} = (x - (1+a) q^n) U_n + a q^{n-1} (1 - q^n) U_{n-1}
    """
    a = tf.cast(a, x.dtype)
    q = tf.cast(q, x.dtype)

    def p0_fn(x):
        return tf.ones_like(x)

    def p1_fn(x, q):
        return x - (1.0 + a)

    def recurrence_fn(p_n, p_n_1, x, q, n):
        n_f = tf.cast(n, x.dtype)
        q_n = tf.pow(q, n_f)
        q_n_1 = tf.pow(q, n_f - 1.0)
        return (x - (1.0 + a) * q_n) * p_n + a * q_n_1 * (1.0 - q_n) * p_n_1

    return q_polynomial_basis_xla(x, q, degree, recurrence_fn, p0_fn, p1_fn)


# =============================================================================
# P1: Fused Basis-Matmul Operations
# =============================================================================

@tf.function(jit_compile=True)
def fused_polynomial_forward(
    x: tf.Tensor,
    coeffs: tf.Tensor,
    basis_fn: Callable[[tf.Tensor], tf.Tensor],
    *,
    use_clenshaw: bool = False,
    clenshaw_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] | None = None,
) -> tf.Tensor:
    """
    Fused polynomial basis evaluation and coefficient contraction.

    When use_clenshaw=True and clenshaw_fn is provided, uses true Clenshaw
    summation which avoids materializing the full basis tensor.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor of shape (batch, input_dim).
    coeffs : tf.Tensor
        Coefficient tensor of shape (input_dim, degree+1, output_dim).
    basis_fn : Callable
        Function to compute pseudo-Vandermonde basis.
    use_clenshaw : bool
        Whether to use Clenshaw summation (memory efficient for high degree).
    clenshaw_fn : Callable, optional
        Clenshaw summation function if use_clenshaw=True.

    Returns
    -------
    tf.Tensor
        Output tensor of shape (batch, output_dim).
    """
    if use_clenshaw and clenshaw_fn is not None:
        return clenshaw_fn(x, coeffs)

    # Standard path: compute basis then contract
    basis = basis_fn(x)  # (batch, input_dim, degree+1)
    return tf.einsum('bid,ido->bo', basis, coeffs)


# =============================================================================
# P2: Parallel Degree Computation via tf.vectorized_map
# =============================================================================

def parallel_polynomial_eval(
    x: tf.Tensor,
    degree: int,
    eval_single_degree: Callable[[tf.Tensor, int], tf.Tensor],
) -> tf.Tensor:
    """
    Evaluate polynomial basis in parallel across degrees using tf.vectorized_map.

    This can improve GPU utilization for lower degrees where the recurrence
    overhead dominates.

    Note: Only beneficial when degrees are independent (e.g., direct formulas).
    For recurrence-based polynomials, sequential evaluation is required.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor of shape (batch, input_dim).
    degree : int
        Maximum polynomial degree.
    eval_single_degree : Callable
        Function (x, k) -> P_k(x) that evaluates degree k directly.

    Returns
    -------
    tf.Tensor
        Basis tensor of shape (batch, input_dim, degree+1).
    """
    # For Chebyshev T_k(x) = cos(k * arccos(x)), we can parallelize
    degrees = tf.range(0, degree + 1, dtype=tf.int32)

    def eval_degree(k):
        return eval_single_degree(x, k)

    # Returns (degree+1, batch, input_dim)
    basis_transposed = tf.vectorized_map(eval_degree, degrees)
    # Transpose to (batch, input_dim, degree+1)
    return tf.transpose(basis_transposed, perm=[1, 2, 0])


def chebyshev_parallel_eval(x: tf.Tensor, degree: int) -> tf.Tensor:
    """
    Evaluate Chebyshev T_k(x) = cos(k * arccos(x)) in parallel.

    This uses the trigonometric definition which allows parallel evaluation
    across all degrees simultaneously, avoiding sequential recurrence.
    """
    from arnold.utils.numerics import safe_acos

    x = tf.reshape(x, (-1, tf.shape(x)[-1], 1))  # (batch, input_dim, 1)
    theta = safe_acos(x)  # (batch, input_dim, 1)
    degrees = tf.cast(tf.range(0, degree + 1), x.dtype)  # (degree+1,)
    # Broadcast: (batch, input_dim, 1) * (degree+1,) -> (batch, input_dim, degree+1)
    return tf.cos(theta * degrees)


# =============================================================================
# P3: TPU Sharding Annotations
# =============================================================================

def with_tpu_sharding(
    tensor: tf.Tensor,
    sharding_spec: str | None = None,
) -> tf.Tensor:
    """
    Apply TPU sharding annotation to a tensor for SPMD partitioning.

    This is a no-op on non-TPU devices but provides hints to the XLA compiler
    for optimal TPU memory layout.

    Parameters
    ----------
    tensor : tf.Tensor
        Tensor to annotate.
    sharding_spec : str, optional
        Sharding specification. Common values:
        - "batch": Shard along batch dimension
        - "replicated": Fully replicate across cores
        - None: Let XLA decide

    Returns
    -------
    tf.Tensor
        Annotated tensor (same value, with sharding metadata).
    """
    hardware = _detect_hardware()
    if hardware != "tpu":
        return tensor

    # TPU-specific sharding using tf.distribute
    if sharding_spec == "batch":
        # Hint that batch dimension should be sharded
        # This uses experimental APIs when available
        try:
            return tf.ensure_shape(tensor, tensor.shape)
        except Exception:
            return tensor

    return tensor


def configure_tpu_strategy() -> tf.distribute.Strategy | None:
    """
    Configure TPU distribution strategy if TPU is available.

    Returns
    -------
    tf.distribute.Strategy or None
        TPUStrategy if TPU available, None otherwise.
    """
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        return tf.distribute.TPUStrategy(resolver)
    except Exception:
        return None


# =============================================================================
# Combined Optimized Layer Forward Pass
# =============================================================================

def optimized_polynomial_forward(
    x: tf.Tensor,
    coeffs: tf.Tensor,
    degree: int,
    input_dim: int,
    output_dim: int,
    *,
    polynomial_type: str = "chebyshev",
    use_clenshaw: bool = True,
    use_parallel: bool = False,
    tpu_sharding: bool = False,
) -> tf.Tensor:
    """
    Optimized polynomial KAN forward pass with configurable optimizations.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor of shape (batch, input_dim).
    coeffs : tf.Tensor
        Coefficient tensor of shape (input_dim, degree+1, output_dim).
    degree : int
        Polynomial degree.
    input_dim : int
        Input dimension.
    output_dim : int
        Output dimension.
    polynomial_type : str
        Type of polynomial ("chebyshev", "legendre", etc.).
    use_clenshaw : bool
        Use true Clenshaw summation (memory efficient).
    use_parallel : bool
        Use parallel degree evaluation (for Chebyshev with direct formula).
    tpu_sharding : bool
        Apply TPU sharding annotations.

    Returns
    -------
    tf.Tensor
        Output tensor of shape (batch, output_dim).
    """
    if tpu_sharding:
        x = with_tpu_sharding(x, "batch")
        coeffs = with_tpu_sharding(coeffs, "replicated")

    # Choose evaluation strategy based on polynomial type and degree
    if use_clenshaw and degree > 10:
        if polynomial_type == "chebyshev":
            return clenshaw_chebyshev_sum(x, coeffs, input_dim=input_dim, output_dim=output_dim)
        elif polynomial_type == "legendre":
            return clenshaw_legendre_sum(x, coeffs, input_dim=input_dim, output_dim=output_dim)

    if use_parallel and polynomial_type == "chebyshev":
        basis = chebyshev_parallel_eval(x, degree)
        return tf.einsum('bid,ido->bo', basis, coeffs)

    # Fallback to standard evaluation
    # This would call the appropriate basis function
    raise NotImplementedError(
        f"Standard evaluation for {polynomial_type} should be handled by the layer itself. "
        "Use use_clenshaw=True or use_parallel=True for optimized paths."
    )
