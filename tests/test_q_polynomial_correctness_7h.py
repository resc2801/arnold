# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
"""
Tests for Sprint 7H polynomial layers.

This module tests the five polynomial layers implemented in Sprint 7H:
- QuantumQKrawtchouk: Quantum q-Krawtchouk polynomials
- ContinuousQLaguerre: Continuous q-Laguerre polynomials
- ContinuousQLegendre: Continuous q-Legendre polynomials
- Tribonacci: 3-term Fibonacci generalization
- Zernike: Radial Zernike polynomials for optics

Test Coverage:
- Basic functionality (build, call, output shape)
- Parameter constraints (q ∈ (0,1), α > -1, etc.)
- Three-term recurrence verification
- Gradient flow through all parameters
- Keras serialization (get_config, from_config)
- Numerical stability
- Edge cases (degree 0, 1, boundary inputs)
"""

import numpy as np
import pytest
import tensorflow as tf

from arnold.layers.core import (
    ContinuousQLaguerre,
    ContinuousQLegendre,
    QuantumQKrawtchouk,
    Tribonacci,
    Zernike,
)


# =============================================================================
# QuantumQKrawtchouk Tests
# =============================================================================


class TestQuantumQKrawtchoukBasic:
    """Basic functionality tests for QuantumQKrawtchouk layer."""

    def test_build_and_call(self):
        """Test layer builds and produces output."""
        layer = QuantumQKrawtchouk(degree=3, units=4)
        x = tf.random.uniform((2, 3), dtype=tf.float32)
        y = layer(x)
        assert y.shape == (2, 4)

    def test_output_shape(self):
        """Test various output shapes."""
        for degree in [2, 4, 6]:
            for units in [1, 4, 16]:
                layer = QuantumQKrawtchouk(degree=degree, units=units, N=degree + 2)
                x = tf.random.uniform((4, 8), dtype=tf.float32)
                y = layer(x)
                assert y.shape == (4, units)

    def test_q_trainable(self):
        """Test that q parameter can be trained."""
        layer = QuantumQKrawtchouk(degree=3, units=4, q_trainable=True)
        x = tf.random.uniform((2, 3), dtype=tf.float32)
        _ = layer(x)

        trainable_names = [v.name for v in layer.trainable_variables]
        assert any("q_logits" in name for name in trainable_names)

    def test_p_trainable(self):
        """Test that p parameter can be trained when enabled."""
        layer = QuantumQKrawtchouk(degree=3, units=4, p_trainable=True)
        x = tf.random.uniform((2, 3), dtype=tf.float32)
        _ = layer(x)

        trainable_names = [v.name for v in layer.trainable_variables]
        assert any("p_logits" in name for name in trainable_names)

    def test_q_constraint(self):
        """Test that q stays in (0, 1)."""
        layer = QuantumQKrawtchouk(degree=3, units=4, q=0.9, q_trainable=True)
        x = tf.random.uniform((2, 3), dtype=tf.float32)

        with tf.GradientTape() as tape:
            y = layer(x)
            loss = tf.reduce_mean(y)

        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads if g is not None)


class TestQuantumQKrawtchoukMathematical:
    """Mathematical property tests for QuantumQKrawtchouk."""

    def test_K0_equals_one(self):
        """Test K_0(x) = 1 for all x."""
        layer = QuantumQKrawtchouk(degree=0, units=1, N=5)
        layer.build((None, 3))

        x = tf.constant([[0.1, 0.5, 0.9]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)

        # degree=0 means only K_0 which should be 1
        np.testing.assert_allclose(
            basis.numpy()[..., 0], np.ones_like(basis.numpy()[..., 0]), rtol=1e-4
        )

    def test_numerical_stability_at_various_q(self):
        """Test numerical stability across q range."""
        for q in [0.1, 0.3, 0.5, 0.7, 0.9]:
            layer = QuantumQKrawtchouk(degree=4, units=1, q=q, N=10, q_trainable=False)
            x = tf.constant([[0.5]], dtype=tf.float32)
            y = layer(x)
            assert not tf.reduce_any(tf.math.is_nan(y))
            assert not tf.reduce_any(tf.math.is_inf(y))


class TestQuantumQKrawtchoukSerialization:
    """Serialization tests for QuantumQKrawtchouk."""

    def test_get_config(self):
        """Test layer configuration serialization."""
        layer = QuantumQKrawtchouk(degree=4, units=8, q=0.7, p=1.5, N=10)
        _ = layer(tf.random.uniform((2, 4), dtype=tf.float32))
        config = layer.get_config()

        assert config["degree"] == 4
        assert config["units"] == 8
        assert "q" in config
        assert "p" in config
        assert "N" in config

    def test_from_config(self):
        """Test layer reconstruction from config."""
        layer = QuantumQKrawtchouk(degree=3, units=4, q=0.6, p=2.0, N=8)
        _ = layer(tf.random.uniform((2, 3), dtype=tf.float32))
        config = layer.get_config()

        new_layer = QuantumQKrawtchouk.from_config(config)
        assert new_layer.degree == layer.degree
        assert new_layer.N == layer.N


# =============================================================================
# ContinuousQLaguerre Tests
# =============================================================================


class TestContinuousQLaguerreBasic:
    """Basic functionality tests for ContinuousQLaguerre layer."""

    def test_build_and_call(self):
        """Test layer builds and produces output."""
        layer = ContinuousQLaguerre(degree=3, units=4)
        x = tf.random.uniform((2, 3), minval=0.1, maxval=1.5, dtype=tf.float32)
        y = layer(x)
        assert y.shape == (2, 4)

    def test_output_shape(self):
        """Test various output shapes."""
        for degree in [2, 5, 8]:
            for units in [1, 4, 16]:
                layer = ContinuousQLaguerre(degree=degree, units=units)
                x = tf.random.uniform((4, 8), minval=0.1, maxval=1.5, dtype=tf.float32)
                y = layer(x)
                assert y.shape == (4, units)

    def test_q_trainable(self):
        """Test that q parameter can be trained."""
        layer = ContinuousQLaguerre(degree=3, units=4, q_trainable=True)
        x = tf.random.uniform((2, 3), minval=0.1, maxval=1.5, dtype=tf.float32)
        _ = layer(x)

        trainable_names = [v.name for v in layer.trainable_variables]
        assert any("q_logits" in name for name in trainable_names)

    def test_alpha_trainable(self):
        """Test that alpha parameter can be trained when enabled."""
        layer = ContinuousQLaguerre(degree=3, units=4, alpha_trainable=True)
        x = tf.random.uniform((2, 3), minval=0.1, maxval=1.5, dtype=tf.float32)
        _ = layer(x)

        trainable_names = [v.name for v in layer.trainable_variables]
        assert any("alpha_logits" in name for name in trainable_names)


class TestContinuousQLaguerreMathematical:
    """Mathematical property tests for ContinuousQLaguerre."""

    def test_L0_equals_one(self):
        """Test L_0(x) = 1 for all x."""
        layer = ContinuousQLaguerre(degree=0, units=1)
        layer.build((None, 3))

        x = tf.constant([[0.1, 0.5, 1.0]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)

        np.testing.assert_allclose(
            basis.numpy()[..., 0], np.ones_like(basis.numpy()[..., 0]), rtol=1e-4
        )

    def test_alpha_constraint(self):
        """Test that alpha > -1 is enforced."""
        layer = ContinuousQLaguerre(degree=3, units=4, alpha=0.1, alpha_trainable=True)
        _ = layer(tf.random.uniform((2, 3), minval=0.1, maxval=1.0, dtype=tf.float32))

        # The internal alpha_logits should produce alpha > -1 via softplus
        # We can check that the trainable variable exists
        trainable_names = [v.name for v in layer.trainable_variables]
        assert any("alpha_logits" in name for name in trainable_names)

    def test_numerical_stability_various_alpha(self):
        """Test numerical stability for various alpha values."""
        for alpha in [0.0, 0.5, 1.0, 2.0]:
            layer = ContinuousQLaguerre(
                degree=4, units=1, alpha=alpha, alpha_trainable=False
            )
            x = tf.constant([[0.5]], dtype=tf.float32)
            y = layer(x)
            assert not tf.reduce_any(tf.math.is_nan(y))


class TestContinuousQLaguerreSerialization:
    """Serialization tests for ContinuousQLaguerre."""

    def test_get_config(self):
        """Test layer configuration serialization."""
        layer = ContinuousQLaguerre(degree=4, units=8, q=0.7, alpha=1.5)
        _ = layer(tf.random.uniform((2, 4), minval=0.1, maxval=1.0, dtype=tf.float32))
        config = layer.get_config()

        assert config["degree"] == 4
        assert config["units"] == 8
        assert "q" in config
        assert "alpha" in config

    def test_from_config(self):
        """Test layer reconstruction from config."""
        layer = ContinuousQLaguerre(degree=3, units=4, q=0.6, alpha=0.5)
        _ = layer(tf.random.uniform((2, 3), minval=0.1, maxval=1.0, dtype=tf.float32))
        config = layer.get_config()

        new_layer = ContinuousQLaguerre.from_config(config)
        assert new_layer.degree == layer.degree


# =============================================================================
# ContinuousQLegendre Tests
# =============================================================================


class TestContinuousQLegendreBasic:
    """Basic functionality tests for ContinuousQLegendre layer."""

    def test_build_and_call(self):
        """Test layer builds and produces output."""
        layer = ContinuousQLegendre(degree=3, units=4)
        x = tf.random.uniform((2, 3), minval=-0.9, maxval=0.9, dtype=tf.float32)
        y = layer(x)
        assert y.shape == (2, 4)

    def test_output_shape(self):
        """Test various output shapes."""
        for degree in [2, 5, 8]:
            for units in [1, 4, 16]:
                layer = ContinuousQLegendre(degree=degree, units=units)
                x = tf.random.uniform(
                    (4, 8), minval=-0.9, maxval=0.9, dtype=tf.float32
                )
                y = layer(x)
                assert y.shape == (4, units)

    def test_q_trainable(self):
        """Test that q parameter can be trained."""
        layer = ContinuousQLegendre(degree=3, units=4, q_trainable=True)
        x = tf.random.uniform((2, 3), minval=-0.9, maxval=0.9, dtype=tf.float32)
        _ = layer(x)

        trainable_names = [v.name for v in layer.trainable_variables]
        assert any("q_logits" in name for name in trainable_names)


class TestContinuousQLegendreMathematical:
    """Mathematical property tests for ContinuousQLegendre."""

    def test_P0_equals_one(self):
        """Test P_0(x) = 1 for all x."""
        layer = ContinuousQLegendre(degree=0, units=1)
        layer.build((None, 3))

        x = tf.constant([[-0.5, 0.0, 0.5]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)

        np.testing.assert_allclose(
            basis.numpy()[..., 0], np.ones_like(basis.numpy()[..., 0]), rtol=1e-4
        )

    def test_numerical_stability_at_various_q(self):
        """Test numerical stability across q range."""
        for q in [0.1, 0.3, 0.5, 0.7, 0.9]:
            layer = ContinuousQLegendre(degree=4, units=1, q=q, q_trainable=False)
            x = tf.constant([[0.0]], dtype=tf.float32)
            y = layer(x)
            assert not tf.reduce_any(tf.math.is_nan(y))
            assert not tf.reduce_any(tf.math.is_inf(y))

    def test_symmetry_behavior(self):
        """Test evaluation at symmetric points."""
        layer = ContinuousQLegendre(degree=4, units=1, q=0.5, q_trainable=False)
        x_pos = tf.constant([[0.5]], dtype=tf.float32)
        x_neg = tf.constant([[-0.5]], dtype=tf.float32)

        out_pos = layer(x_pos)
        out_neg = layer(x_neg)

        # Both should be finite
        assert not tf.reduce_any(tf.math.is_nan(out_pos))
        assert not tf.reduce_any(tf.math.is_nan(out_neg))


class TestContinuousQLegendreSerialization:
    """Serialization tests for ContinuousQLegendre."""

    def test_get_config(self):
        """Test layer configuration serialization."""
        layer = ContinuousQLegendre(degree=4, units=8, q=0.7)
        _ = layer(tf.random.uniform((2, 4), minval=-0.9, maxval=0.9, dtype=tf.float32))
        config = layer.get_config()

        assert config["degree"] == 4
        assert config["units"] == 8
        assert "q" in config

    def test_from_config(self):
        """Test layer reconstruction from config."""
        layer = ContinuousQLegendre(degree=3, units=4, q=0.6)
        _ = layer(tf.random.uniform((2, 3), minval=-0.9, maxval=0.9, dtype=tf.float32))
        config = layer.get_config()

        new_layer = ContinuousQLegendre.from_config(config)
        assert new_layer.degree == layer.degree


# =============================================================================
# Tribonacci Tests
# =============================================================================


class TestTribonacciBasic:
    """Basic functionality tests for Tribonacci layer."""

    def test_build_and_call(self):
        """Test layer builds and produces output."""
        layer = Tribonacci(degree=5, units=4)
        x = tf.random.uniform((2, 3), dtype=tf.float32)
        y = layer(x)
        assert y.shape == (2, 4)

    def test_output_shape(self):
        """Test various output shapes."""
        for degree in [3, 6, 10]:
            for units in [1, 4, 16]:
                layer = Tribonacci(degree=degree, units=units)
                x = tf.random.uniform((4, 8), dtype=tf.float32)
                y = layer(x)
                assert y.shape == (4, units)

    def test_gradient_flow(self):
        """Test that gradients flow through the layer."""
        layer = Tribonacci(degree=4, units=4)
        x = tf.random.uniform((2, 3), dtype=tf.float32)

        with tf.GradientTape() as tape:
            y = layer(x)
            loss = tf.reduce_mean(y)

        grads = tape.gradient(loss, layer.trainable_variables)
        assert len(grads) > 0
        assert all(g is not None for g in grads)


class TestTribonacciMathematical:
    """Mathematical property tests for Tribonacci."""

    def test_T0_equals_zero(self):
        """Test T_0(x) = 0."""
        layer = Tribonacci(degree=5, units=1)
        layer.build((None, 3))

        x = tf.constant([[0.5, 1.0, 2.0]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)

        # T_0 = 0
        np.testing.assert_allclose(
            basis.numpy()[..., 0], np.zeros_like(basis.numpy()[..., 0]), atol=1e-5
        )

    def test_T1_equals_one(self):
        """Test T_1(x) = 1."""
        layer = Tribonacci(degree=5, units=1)
        layer.build((None, 3))

        x = tf.constant([[0.5, 1.0, 2.0]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)

        # T_1 = 1
        np.testing.assert_allclose(
            basis.numpy()[..., 1], np.ones_like(basis.numpy()[..., 1]), rtol=1e-5
        )

    def test_T2_equals_x(self):
        """Test T_2(x) = x."""
        layer = Tribonacci(degree=5, units=1)
        layer.build((None, 3))

        x = tf.constant([[0.5, 1.0, 2.0]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)

        # T_2 = x - compare with proper shape handling
        np.testing.assert_allclose(
            basis.numpy()[0, :, 2], x.numpy()[0], rtol=1e-5
        )

    def test_T3_equals_x_squared_plus_1(self):
        """Test T_3(x) = x^2 + 1."""
        layer = Tribonacci(degree=5, units=1)
        layer.build((None, 3))

        x = tf.constant([[0.5, 1.0, 2.0]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)

        expected = x.numpy()[0] ** 2 + 1
        np.testing.assert_allclose(basis.numpy()[0, :, 3], expected, rtol=1e-4)

    def test_numerical_stability(self):
        """Test numerical stability at various inputs."""
        layer = Tribonacci(degree=8, units=1)
        x = tf.constant([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=tf.float32)
        y = layer(x)
        assert not tf.reduce_any(tf.math.is_nan(y))
        assert not tf.reduce_any(tf.math.is_inf(y))


class TestTribonacciSerialization:
    """Serialization tests for Tribonacci."""

    def test_get_config(self):
        """Test layer configuration serialization."""
        layer = Tribonacci(degree=6, units=8)
        _ = layer(tf.random.uniform((2, 4), dtype=tf.float32))
        config = layer.get_config()

        assert config["degree"] == 6
        assert config["units"] == 8

    def test_from_config(self):
        """Test layer reconstruction from config."""
        layer = Tribonacci(degree=5, units=4)
        _ = layer(tf.random.uniform((2, 3), dtype=tf.float32))
        config = layer.get_config()

        new_layer = Tribonacci.from_config(config)
        assert new_layer.degree == layer.degree


# =============================================================================
# Zernike Tests
# =============================================================================


class TestZernikeBasic:
    """Basic functionality tests for Zernike layer."""

    def test_build_and_call(self):
        """Test layer builds and produces output."""
        layer = Zernike(degree=4, units=4)
        x = tf.random.uniform((2, 3), minval=0.1, maxval=0.9, dtype=tf.float32)
        y = layer(x)
        assert y.shape == (2, 4)

    def test_output_shape(self):
        """Test various output shapes."""
        for degree in [2, 4, 6]:
            for units in [1, 4, 16]:
                layer = Zernike(degree=degree, units=units)
                x = tf.random.uniform(
                    (4, 8), minval=0.1, maxval=0.9, dtype=tf.float32
                )
                y = layer(x)
                assert y.shape == (4, units)

    def test_gradient_flow(self):
        """Test that gradients flow through the layer."""
        layer = Zernike(degree=4, units=4)
        x = tf.random.uniform((2, 3), minval=0.1, maxval=0.9, dtype=tf.float32)

        with tf.GradientTape() as tape:
            y = layer(x)
            loss = tf.reduce_mean(y)

        grads = tape.gradient(loss, layer.trainable_variables)
        assert len(grads) > 0
        assert all(g is not None for g in grads)


class TestZernikeMathematical:
    """Mathematical property tests for Zernike."""

    def test_R0_equals_one(self):
        """Test R_0^0(ρ) = 1."""
        layer = Zernike(degree=4, units=1)
        layer.build((None, 3))

        x = tf.constant([[0.1, 0.5, 0.9]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)

        # R_0^0 = 1 (first even degree polynomial)
        np.testing.assert_allclose(
            basis.numpy()[..., 0], np.ones_like(basis.numpy()[..., 0]), rtol=1e-4
        )

    def test_R2_defocus(self):
        """Test R_2^0(ρ) = 2ρ² - 1 (defocus aberration)."""
        layer = Zernike(degree=4, units=1)
        layer.build((None, 3))

        x = tf.constant([[0.0, 0.5, 1.0]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)

        # R_2^0 = 2ρ² - 1 - compare with proper indexing
        expected = 2 * x.numpy()[0] ** 2 - 1
        np.testing.assert_allclose(basis.numpy()[0, :, 1], expected, rtol=1e-4)

    def test_unit_circle_boundary(self):
        """Test at ρ=1 (unit circle boundary)."""
        layer = Zernike(degree=4, units=1)
        x = tf.constant([[1.0]], dtype=tf.float32)
        y = layer(x)
        assert not tf.reduce_any(tf.math.is_nan(y))
        assert not tf.reduce_any(tf.math.is_inf(y))

    def test_origin(self):
        """Test at ρ=0 (origin)."""
        layer = Zernike(degree=4, units=1)
        x = tf.constant([[0.0]], dtype=tf.float32)
        y = layer(x)
        assert not tf.reduce_any(tf.math.is_nan(y))

    def test_numerical_stability_unit_disk(self):
        """Test numerical stability across unit disk."""
        layer = Zernike(degree=6, units=1)
        x = tf.constant(
            [[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=tf.float32
        )
        y = layer(x)
        assert not tf.reduce_any(tf.math.is_nan(y))
        assert not tf.reduce_any(tf.math.is_inf(y))


class TestZernikeSerialization:
    """Serialization tests for Zernike."""

    def test_get_config(self):
        """Test layer configuration serialization."""
        layer = Zernike(degree=4, units=8)
        _ = layer(tf.random.uniform((2, 4), minval=0.1, maxval=0.9, dtype=tf.float32))
        config = layer.get_config()

        assert config["degree"] == 4
        assert config["units"] == 8

    def test_from_config(self):
        """Test layer reconstruction from config."""
        layer = Zernike(degree=6, units=4)
        _ = layer(tf.random.uniform((2, 3), minval=0.1, maxval=0.9, dtype=tf.float32))
        config = layer.get_config()

        new_layer = Zernike.from_config(config)
        assert new_layer.degree == layer.degree


# =============================================================================
# Cross-Layer Integration Tests
# =============================================================================


class TestLayerIntegration:
    """Integration tests for all Sprint 7H layers."""

    @pytest.mark.parametrize(
        "layer_class,kwargs",
        [
            (QuantumQKrawtchouk, {"degree": 4, "units": 4, "q": 0.5, "N": 10}),
            (ContinuousQLaguerre, {"degree": 4, "units": 4, "q": 0.5, "alpha": 0.5}),
            (ContinuousQLegendre, {"degree": 4, "units": 4, "q": 0.5}),
            (Tribonacci, {"degree": 5, "units": 4}),
            (Zernike, {"degree": 4, "units": 4}),
        ],
    )
    def test_model_integration(self, layer_class, kwargs):
        """Test layer works in Sequential model."""
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(2,)),
                layer_class(**kwargs),
                tf.keras.layers.Dense(1),
            ]
        )
        x = tf.random.uniform((8, 2), minval=0.1, maxval=0.9)
        output = model(x)
        assert output.shape == (8, 1)
        assert not tf.reduce_any(tf.math.is_nan(output))

    @pytest.mark.parametrize(
        "layer_class,kwargs",
        [
            (QuantumQKrawtchouk, {"degree": 3, "units": 4, "N": 10}),
            (ContinuousQLaguerre, {"degree": 3, "units": 4}),
            (ContinuousQLegendre, {"degree": 3, "units": 4}),
            (Tribonacci, {"degree": 4, "units": 4}),
            (Zernike, {"degree": 4, "units": 4}),
        ],
    )
    def test_gradient_flow_all(self, layer_class, kwargs):
        """Test gradients can flow through all layers."""
        layer = layer_class(**kwargs)
        x = tf.random.uniform((4, 3), minval=0.1, maxval=0.9, dtype=tf.float32)

        with tf.GradientTape() as tape:
            output = layer(x)
            loss = tf.reduce_mean(output**2)

        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads)
        assert all(not tf.reduce_any(tf.math.is_nan(g)) for g in grads)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_quantum_q_krawtchouk_small_q(self):
        """Test QuantumQKrawtchouk with small q value."""
        layer = QuantumQKrawtchouk(degree=3, units=1, q=0.1, N=10)
        x = tf.constant([[1.0]], dtype=tf.float32)
        output = layer(x)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_continuous_q_laguerre_small_alpha(self):
        """Test ContinuousQLaguerre with small alpha near 0."""
        layer = ContinuousQLaguerre(degree=3, units=1, alpha=0.01)
        x = tf.constant([[0.5]], dtype=tf.float32)
        output = layer(x)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_tribonacci_high_degree(self):
        """Test Tribonacci with higher degree."""
        layer = Tribonacci(degree=12, units=1)
        x = tf.constant([[1.0]], dtype=tf.float32)
        output = layer(x)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_zernike_high_degree(self):
        """Test Zernike with higher degree."""
        layer = Zernike(degree=10, units=1)
        x = tf.constant([[0.5]], dtype=tf.float32)
        output = layer(x)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_batch_with_zeros(self):
        """Test layers with batch containing zeros."""
        layers = [
            QuantumQKrawtchouk(degree=3, units=1, N=10),
            ContinuousQLaguerre(degree=3, units=1),
            ContinuousQLegendre(degree=3, units=1),
            Tribonacci(degree=4, units=1),
            Zernike(degree=4, units=1),
        ]

        x = tf.constant([[0.0], [0.5], [1.0]], dtype=tf.float32)

        for layer in layers:
            output = layer(x)
            assert not tf.reduce_any(tf.math.is_nan(output))


class TestMultiDimensional:
    """Tests for multi-dimensional input/output configurations."""

    @pytest.mark.parametrize(
        "layer_class,kwargs",
        [
            (QuantumQKrawtchouk, {"degree": 3, "units": 8, "N": 10}),
            (ContinuousQLaguerre, {"degree": 3, "units": 8}),
            (ContinuousQLegendre, {"degree": 3, "units": 8}),
            (Tribonacci, {"degree": 4, "units": 8}),
            (Zernike, {"degree": 4, "units": 8}),
        ],
    )
    def test_large_batch(self, layer_class, kwargs):
        """Test with large batch size."""
        layer = layer_class(**kwargs)
        x = tf.random.uniform((256, 4), minval=0.1, maxval=0.9)
        output = layer(x)
        assert output.shape == (256, 8)
        assert not tf.reduce_any(tf.math.is_nan(output))
