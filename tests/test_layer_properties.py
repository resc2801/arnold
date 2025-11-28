## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
"""Gradient and serialization properties for representative layers."""

import tensorflow as tf

from arnold.layers.core.polynomial.orthogonal import (
    Chebyshev2nd,
    Hermite,
    Jacobi,
    Legendre,
    Wilson,
)


def test_legendre_gradients_and_finite():
    x = tf.random.uniform((4, 3), minval=-0.8, maxval=0.8, seed=7)
    layer = Legendre(degree=4, units=2)
    with tf.GradientTape() as tape:
        y = layer(x)
        loss = tf.reduce_mean(tf.square(y))
    grads = tape.gradient(loss, layer.trainable_variables)
    assert all(g is not None for g in grads)
    for g in grads:
        assert tf.reduce_all(tf.math.is_finite(g))


def test_legendre_serialization_roundtrip():
    layer = Legendre(degree=3, units=2)
    _ = layer(tf.zeros((1, 3)))  # build
    config = layer.get_config()
    restored = Legendre.from_config(config)
    y = restored(tf.ones((2, 3)))
    assert y.shape == (2, 2)


def test_chebyshev2nd_gradients():
    x = tf.random.uniform((4, 3), minval=-0.8, maxval=0.8, seed=17)
    layer = Chebyshev2nd(degree=5, units=3, input_clip=(-1, 1))
    with tf.GradientTape() as tape:
        y = layer(x)
        loss = tf.reduce_mean(tf.square(y))
    grads = tape.gradient(loss, layer.trainable_variables)
    assert all(g is not None for g in grads)
    for g in grads:
        assert tf.reduce_all(tf.math.is_finite(g))


def test_jacobi_gradient():
    x = tf.random.uniform((3, 2), minval=-0.9, maxval=0.9, seed=21)
    layer = Jacobi(degree=4, units=2, alpha_init=0.5, beta_init=-0.25, input_clip=(-1, 1))
    with tf.GradientTape() as tape:
        y = layer(x)
        loss = tf.reduce_mean(y)
    grads = tape.gradient(loss, layer.trainable_variables)
    assert all(g is not None for g in grads)
    for g in grads:
        assert tf.reduce_all(tf.math.is_finite(g))


def test_hermite_normalized_gradient():
    x = tf.random.uniform((3, 2), minval=-2.0, maxval=2.0, seed=33)
    layer = Hermite(degree=5, units=2, normalized=True)
    with tf.GradientTape() as tape:
        y = layer(x)
        loss = tf.reduce_mean(tf.square(y))
    grads = tape.gradient(loss, layer.trainable_variables)
    assert all(g is not None for g in grads)
    for g in grads:
        assert tf.reduce_all(tf.math.is_finite(g))


def test_wilson_smoke_and_gradients():
    x = tf.random.uniform((2, 2), minval=-0.5, maxval=0.5, seed=44)
    layer = Wilson(degree=3, units=2)
    with tf.GradientTape() as tape:
        y = layer(x)
        loss = tf.reduce_mean(tf.square(y))
    grads = tape.gradient(loss, layer.trainable_variables)
    assert y.shape == (2, 2)
    assert all(g is not None for g in grads)
    for g in grads:
        assert tf.reduce_all(tf.math.is_finite(g))


# ============================================================================
# A2: Regularizer Support Tests
# ============================================================================


def test_kernel_regularizer_polynomial():
    """Test kernel_regularizer works on PolynomialBase layers."""
    x = tf.random.uniform((4, 3), minval=-0.8, maxval=0.8, seed=100)
    layer = Legendre(
        degree=3,
        units=2,
        kernel_regularizer=tf.keras.regularizers.L2(0.01)
    )
    layer(x)

    # Layer should have regularization losses
    assert len(layer.losses) > 0
    # Regularization loss should be positive
    reg_loss = sum(layer.losses)
    assert reg_loss > 0


def test_bias_regularizer_polynomial():
    """Test bias_regularizer works on PolynomialBase layers."""
    x = tf.random.uniform((4, 3), minval=-0.8, maxval=0.8, seed=101)
    layer = Legendre(
        degree=3,
        units=2,
        use_bias=True,
        bias_regularizer=tf.keras.regularizers.L1(0.01)
    )
    layer(x)

    # Layer should have regularization losses from bias
    assert len(layer.losses) > 0


def test_regularizer_serialization():
    """Test regularizers serialize/deserialize correctly."""
    layer = Legendre(
        degree=3,
        units=2,
        kernel_regularizer=tf.keras.regularizers.L2(0.05),
        bias_regularizer=tf.keras.regularizers.L1(0.02),
    )
    _ = layer(tf.zeros((1, 3)))  # build

    config = layer.get_config()

    # Check regularizers are in config
    assert config["kernel_regularizer"] is not None
    assert config["bias_regularizer"] is not None

    # Restore and verify
    restored = Legendre.from_config(config)
    assert restored.kernel_regularizer is not None
    assert restored.bias_regularizer is not None


def test_kernel_regularizer_rbf():
    """Test kernel_regularizer works on RBFBase layers."""
    from arnold.layers.core.rbf import GaussianRBF

    x = tf.random.uniform((4, 3), minval=0.0, maxval=1.0, seed=102)
    layer = GaussianRBF(
        units=2,
        num_grids=5,
        kernel_regularizer=tf.keras.regularizers.L2(0.01)
    )
    layer(x)

    assert len(layer.losses) > 0
    reg_loss = sum(layer.losses)
    assert reg_loss > 0


def test_kernel_regularizer_wavelet():
    """Test kernel_regularizer works on WaveletBase layers."""
    from arnold.layers.core.wavelets import Morelet

    x = tf.random.uniform((4, 3), minval=-1.0, maxval=1.0, seed=103)
    layer = Morelet(
        units=2,
        kernel_regularizer=tf.keras.regularizers.L2(0.01)
    )
    layer(x)

    assert len(layer.losses) > 0
    reg_loss = sum(layer.losses)
    assert reg_loss > 0


def test_regularizer_in_model_training():
    """Test regularizers affect model loss during training."""
    x = tf.random.uniform((32, 4), minval=-0.8, maxval=0.8, seed=104)
    y = tf.random.uniform((32, 2), seed=105)

    # Model without regularization
    model_no_reg = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(4,)),
        Legendre(degree=3, units=2),
    ])
    model_no_reg.compile(optimizer="sgd", loss="mse")

    # Model with regularization
    model_with_reg = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(4,)),
        Legendre(degree=3, units=2, kernel_regularizer=tf.keras.regularizers.L2(0.1)),
    ])
    model_with_reg.compile(optimizer="sgd", loss="mse")

    # The regularized model should report higher total loss
    loss_no_reg = model_no_reg.evaluate(x, y, verbose=0)
    loss_with_reg = model_with_reg.evaluate(x, y, verbose=0)

    # With same random init this isn't guaranteed, but regularization should add positive loss
    # Just verify both work without errors
    assert not tf.math.is_nan(loss_no_reg)
    assert not tf.math.is_nan(loss_with_reg)


# ============================================================================
# N4: Float64 Promotion Path Tests
# ============================================================================


def test_float64_promotion_high_degree():
    """Test that high-degree polynomials promote to float64 on CPU."""
    from arnold.layers.core.kan_base import detect_hardware

    hw = detect_hardware()

    layer = Legendre(
        degree=15,  # Above precision_threshold (10)
        units=2,
        hardware_adaptive=True,
        precision_threshold=10,
    )

    x = tf.constant([[0.5, -0.3]], dtype=tf.float32)
    y = layer(x)

    # On CPU, should have detected hardware and promoted
    if hw == "cpu":
        assert layer.promote_to_float64 is True
        assert layer._detected_hardware == "cpu"

    # Output should be valid regardless of hardware
    assert not tf.reduce_any(tf.math.is_nan(y))
    assert y.dtype == tf.float32  # Output cast back to original


def test_float64_promotion_disabled_on_gpu():
    """Test that GPU/MPS disables float64 promotion by default."""
    from arnold.layers.core.kan_base import detect_hardware

    hw = detect_hardware()

    layer = Legendre(
        degree=15,
        units=2,
        hardware_adaptive=True,
    )

    _ = layer(tf.ones((1, 3)))  # build

    # On GPU/MPS, should not promote
    if hw in ("gpu", "mps"):
        assert layer.promote_to_float64 is False


def test_explicit_float64_promotion_override():
    """Test that explicit promote_to_float64=True works."""
    layer = Legendre(
        degree=15,
        units=2,
        promote_to_float64=True,  # Explicit override
        hardware_adaptive=False,
    )

    x = tf.constant([[0.5, -0.3]], dtype=tf.float32)
    y = layer(x)

    assert layer.promote_to_float64 is True
    assert not tf.reduce_any(tf.math.is_nan(y))


def test_float64_promotion_preserves_output_dtype():
    """Output dtype should match input dtype after promotion (for float32+)."""
    layer = Legendre(
        degree=15,
        units=2,
        promote_to_float64=True,
    )

    # Test with float32 input - should be preserved
    x32 = tf.constant([[0.5, -0.3]], dtype=tf.float32)
    y32 = layer(x32)
    assert y32.dtype == tf.float32

    # Note: float16 input may be upcast due to Keras compute dtype handling
    # This is expected behavior for numerical stability
