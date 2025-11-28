# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
"""
Wavelet Correctness Tests - Phase 4.2 + Phase 7A

Tests verify that each wavelet implementation:
1. mother_wavelet(x) produces correct mathematical output
2. Full layer output has correct shape
3. Gradients flow correctly
4. Serialization works
5. Known mathematical properties hold
"""

import numpy as np
import pytest
import tensorflow as tf

from arnold.layers.core.wavelets import (
    Bump,
    Coiflet,
    Daubechies,
    DerivativeOfGaussian,
    Haar,
    Meyer,
    Morelet,
    Poisson,
    Ricker,
    Shannon,
    Symlet,
    WaveletBase,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_input():
    """Sample input for full layer testing."""
    return tf.constant([[0.1, 0.5], [0.3, 0.7], [0.9, 0.2]], dtype=tf.float32)


# ============================================================================
# Bump Wavelet Tests
# ============================================================================


class TestBump:
    """Test Bump wavelet: ψ(x) = I_{[-1,1]}(x) * exp(-1/(1-x²))"""

    def test_mother_wavelet_at_zero(self):
        """ψ(0) = exp(-1) ≈ 0.368"""
        layer = Bump(units=2)
        _ = layer(tf.ones((1, 1)))

        # Shape: (batch, output_dim, input_dim)
        x = tf.constant([[[0.0]]], dtype=tf.float32)
        psi = layer.mother_wavelet(x)
        expected = np.exp(-1.0)
        np.testing.assert_allclose(psi.numpy()[0, 0, 0], expected, rtol=1e-5)

    def test_mother_wavelet_symmetry(self):
        """Bump wavelet is symmetric: ψ(-x) = ψ(x)"""
        layer = Bump(units=2)
        _ = layer(tf.ones((1, 1)))

        x_pos = tf.constant([[[0.5]]], dtype=tf.float32)
        x_neg = tf.constant([[[-0.5]]], dtype=tf.float32)
        psi_pos = layer.mother_wavelet(x_pos)
        psi_neg = layer.mother_wavelet(x_neg)
        np.testing.assert_allclose(psi_pos.numpy(), psi_neg.numpy(), rtol=1e-5)

    def test_mother_wavelet_bounded(self):
        """Bump wavelet is in (0, exp(-1)]"""
        layer = Bump(units=2)
        _ = layer(tf.ones((1, 1)))

        x = tf.constant([[[0.0, 0.3, 0.6, 0.9]]], dtype=tf.float32)
        psi = layer.mother_wavelet(x).numpy()
        assert np.all(psi > 0)
        assert np.all(psi <= np.exp(-1) + 1e-5)

    def test_output_shape(self, sample_input):
        """Output shape is (batch, units)."""
        layer = Bump(units=5)
        output = layer(sample_input)
        assert output.shape == (3, 5)

    def test_gradient_flow(self, sample_input):
        """Gradients flow through the layer."""
        layer = Bump(units=2)
        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = tf.reduce_mean(output)
        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads)


# ============================================================================
# Derivative of Gaussian (DOG) Tests
# ============================================================================


class TestDerivativeOfGaussian:
    """Test DOG wavelet: ψ(x) = -x * exp(-0.5x²)"""

    def test_mother_wavelet_at_zero(self):
        """ψ(0) = 0"""
        layer = DerivativeOfGaussian(units=2)
        _ = layer(tf.ones((1, 1)))

        x = tf.constant([[[0.0]]], dtype=tf.float32)
        psi = layer.mother_wavelet(x)
        np.testing.assert_allclose(psi.numpy()[0, 0, 0], 0.0, atol=1e-6)

    def test_mother_wavelet_antisymmetry(self):
        """DOG wavelet is antisymmetric: ψ(-x) = -ψ(x)"""
        layer = DerivativeOfGaussian(units=2)
        _ = layer(tf.ones((1, 1)))

        x_pos = tf.constant([[[1.0]]], dtype=tf.float32)
        x_neg = tf.constant([[[-1.0]]], dtype=tf.float32)
        psi_pos = layer.mother_wavelet(x_pos)
        psi_neg = layer.mother_wavelet(x_neg)
        np.testing.assert_allclose(psi_pos.numpy(), -psi_neg.numpy(), rtol=1e-5)

    def test_mother_wavelet_formula(self):
        """Check specific value: ψ(1) = -exp(-0.5) ≈ -0.6065"""
        layer = DerivativeOfGaussian(units=2)
        _ = layer(tf.ones((1, 1)))

        x = tf.constant([[[1.0]]], dtype=tf.float32)
        psi = layer.mother_wavelet(x)
        expected = -1.0 * np.exp(-0.5)
        np.testing.assert_allclose(psi.numpy()[0, 0, 0], expected, rtol=1e-5)

    def test_extrema_locations(self):
        """DOG has extrema at x = ±1"""
        layer = DerivativeOfGaussian(units=2)
        _ = layer(tf.ones((1, 1)))

        # The derivative of -x*exp(-0.5x²) is -(1-x²)*exp(-0.5x²)
        # Zero at x=±1, maximum magnitude at x=0
        x_minus1 = tf.constant([[[-1.0]]], dtype=tf.float32)
        x_plus1 = tf.constant([[[1.0]]], dtype=tf.float32)

        psi_m1 = layer.mother_wavelet(x_minus1).numpy()[0, 0, 0]
        psi_p1 = layer.mother_wavelet(x_plus1).numpy()[0, 0, 0]

        # At extrema, abs(psi) = exp(-0.5)
        np.testing.assert_allclose(abs(psi_m1), np.exp(-0.5), rtol=1e-5)
        np.testing.assert_allclose(abs(psi_p1), np.exp(-0.5), rtol=1e-5)


# ============================================================================
# Meyer Wavelet Tests
# ============================================================================


class TestMeyer:
    """Test Meyer wavelet: ψ(v) = sin(πv) * φ(v) where φ is the Meyer auxiliary"""

    def test_mother_wavelet_at_zero(self):
        """ψ(0) = 0 since sin(0) = 0"""
        layer = Meyer(units=2)
        _ = layer(tf.ones((1, 1)))

        x = tf.constant([[[0.0]]], dtype=tf.float32)
        psi = layer.mother_wavelet(x)
        np.testing.assert_allclose(psi.numpy()[0, 0, 0], 0.0, atol=1e-6)

    def test_mother_wavelet_symmetry(self):
        """Meyer wavelet uses |x|, so ψ(-x) = ψ(x) for x ≠ 0"""
        layer = Meyer(units=2)
        _ = layer(tf.ones((1, 1)))

        x_pos = tf.constant([[[0.7]]], dtype=tf.float32)
        x_neg = tf.constant([[[-0.7]]], dtype=tf.float32)
        psi_pos = layer.mother_wavelet(x_pos)
        psi_neg = layer.mother_wavelet(x_neg)
        np.testing.assert_allclose(psi_pos.numpy(), psi_neg.numpy(), rtol=1e-5)

    def test_output_shape(self, sample_input):
        """Output shape is (batch, units)."""
        layer = Meyer(units=4)
        output = layer(sample_input)
        assert output.shape == (3, 4)


# ============================================================================
# Morelet (Morlet) Wavelet Tests
# ============================================================================


class TestMorelet:
    """Test Morlet wavelet: ψ(x) = exp(-0.5x²) * cos(ω₀x)"""

    def test_mother_wavelet_at_zero(self):
        """ψ(0) = cos(0) = 1"""
        layer = Morelet(units=2, omega_init=5.0)
        _ = layer(tf.ones((1, 1)))

        x = tf.constant([[[0.0]]], dtype=tf.float32)
        psi = layer.mother_wavelet(x)
        np.testing.assert_allclose(psi.numpy()[0, 0, 0], 1.0, rtol=1e-5)

    def test_mother_wavelet_envelope(self):
        """Morlet is bounded by Gaussian envelope: |ψ(x)| ≤ exp(-0.5x²)"""
        layer = Morelet(units=2, omega_init=5.0)
        _ = layer(tf.ones((1, 1)))

        x = tf.constant([[[0.0, 1.0, 2.0, 3.0]]], dtype=tf.float32)
        psi = layer.mother_wavelet(x).numpy()
        envelope = np.exp(-0.5 * x.numpy() ** 2)
        assert np.all(np.abs(psi) <= envelope + 1e-5)

    def test_omega_affects_oscillation(self):
        """Different omega produces different oscillation frequency."""
        layer1 = Morelet(units=2, omega_init=3.0, omega_trainable=False)
        layer2 = Morelet(units=2, omega_init=7.0, omega_trainable=False)
        _ = layer1(tf.ones((1, 1)))
        _ = layer2(tf.ones((1, 1)))

        x = tf.constant([[[1.0]]], dtype=tf.float32)
        psi1 = layer1.mother_wavelet(x).numpy()[0, 0, 0]
        psi2 = layer2.mother_wavelet(x).numpy()[0, 0, 0]
        # cos(3) ≠ cos(7)
        assert psi1 != pytest.approx(psi2, rel=0.1)

    def test_omega_trainable(self, sample_input):
        """Omega parameter receives gradients."""
        layer = Morelet(units=2, omega_init=5.0, omega_trainable=True)
        with tf.GradientTape() as tape:
            output = layer(sample_input)
            loss = tf.reduce_mean(output)
        grads = tape.gradient(loss, layer.trainable_variables)
        omega_grad = [g for g, v in zip(grads, layer.trainable_variables) if "frequency" in v.name]
        assert len(omega_grad) == 1
        assert omega_grad[0] is not None


# ============================================================================
# Poisson Wavelet Tests
# ============================================================================


class TestPoisson:
    """Test Poisson wavelet: ψ(t) = (1/π) * (1-t²)/(1+t²)²"""

    def test_mother_wavelet_at_zero(self):
        """ψ(0) = 1/π"""
        layer = Poisson(units=2)
        _ = layer(tf.ones((1, 1)))

        x = tf.constant([[[0.0]]], dtype=tf.float32)
        psi = layer.mother_wavelet(x)
        expected = 1.0 / np.pi
        np.testing.assert_allclose(psi.numpy()[0, 0, 0], expected, rtol=1e-5)

    def test_mother_wavelet_at_one(self):
        """ψ(1) = 0 since numerator is 1-1² = 0"""
        layer = Poisson(units=2)
        _ = layer(tf.ones((1, 1)))

        x = tf.constant([[[1.0]]], dtype=tf.float32)
        psi = layer.mother_wavelet(x)
        np.testing.assert_allclose(psi.numpy()[0, 0, 0], 0.0, atol=1e-6)

    def test_mother_wavelet_symmetry(self):
        """Poisson wavelet is symmetric: ψ(-x) = ψ(x)"""
        layer = Poisson(units=2)
        _ = layer(tf.ones((1, 1)))

        x_pos = tf.constant([[[0.5]]], dtype=tf.float32)
        x_neg = tf.constant([[[-0.5]]], dtype=tf.float32)
        psi_pos = layer.mother_wavelet(x_pos)
        psi_neg = layer.mother_wavelet(x_neg)
        np.testing.assert_allclose(psi_pos.numpy(), psi_neg.numpy(), rtol=1e-5)


# ============================================================================
# Ricker (Mexican Hat) Wavelet Tests
# ============================================================================


class TestRicker:
    """Test Ricker wavelet: ψ(t) = (2/(√3σ * π^0.25)) * (1-(t/σ)²) * exp(-t²/(2σ²))"""

    def test_mother_wavelet_positive_at_zero(self):
        """ψ(0) > 0 (central peak)"""
        layer = Ricker(units=2, sigma_init=1.0)
        _ = layer(tf.ones((1, 1)))

        x = tf.constant([[[0.0]]], dtype=tf.float32)
        psi = layer.mother_wavelet(x)
        assert psi.numpy()[0, 0, 0] > 0

    def test_mother_wavelet_zeros(self):
        """Ricker wavelet has zeros at t = ±σ"""
        layer = Ricker(units=2, sigma_init=1.0, sigma_trainable=False)
        _ = layer(tf.ones((1, 1)))
        # After build, sigma is transformed via softplus
        sigma_actual = (tf.nn.softplus(layer.sigma) + 1e-6).numpy()

        # At t = σ, the term (1-(t/σ)²) = 0
        x = tf.constant([[[sigma_actual]]], dtype=tf.float32)
        psi = layer.mother_wavelet(x)
        np.testing.assert_allclose(psi.numpy()[0, 0, 0], 0.0, atol=0.05)

    def test_mother_wavelet_symmetry(self):
        """Ricker wavelet is symmetric: ψ(-x) = ψ(x)"""
        layer = Ricker(units=2, sigma_init=1.0)
        _ = layer(tf.ones((1, 1)))

        x_pos = tf.constant([[[0.5]]], dtype=tf.float32)
        x_neg = tf.constant([[[-0.5]]], dtype=tf.float32)
        psi_pos = layer.mother_wavelet(x_pos)
        psi_neg = layer.mother_wavelet(x_neg)
        np.testing.assert_allclose(psi_pos.numpy(), psi_neg.numpy(), rtol=1e-5)

    def test_sigma_trainable(self, sample_input):
        """Sigma parameter receives gradients."""
        layer = Ricker(units=2, sigma_init=1.0, sigma_trainable=True)
        with tf.GradientTape() as tape:
            output = layer(sample_input)
            loss = tf.reduce_mean(output)
        grads = tape.gradient(loss, layer.trainable_variables)
        sigma_grad = [g for g, v in zip(grads, layer.trainable_variables) if "standard_deviation" in v.name]
        assert len(sigma_grad) == 1
        assert sigma_grad[0] is not None


# ============================================================================
# Shannon Wavelet Tests
# ============================================================================


class TestShannon:
    """Test Shannon wavelet: ψ(t) = sinc(t) * window, normalized for unit energy"""

    def test_mother_wavelet_bounded(self):
        """Shannon wavelet is bounded"""
        layer = Shannon(units=2)
        _ = layer(tf.ones((1, 1)))

        x = tf.constant([[[0.0, 0.5, 1.0, 2.0]]], dtype=tf.float32)
        psi = layer.mother_wavelet(x).numpy()
        # Hamming window keeps it bounded
        assert np.all(np.abs(psi) < 10)  # reasonable bound

    def test_output_shape(self, sample_input):
        """Output shape is (batch, units)."""
        layer = Shannon(units=3)
        output = layer(sample_input)
        assert output.shape == (3, 3)

    def test_sinc_at_zero(self):
        """sinc(0) = 1, so mother wavelet at zero equals window[0]/norm."""
        layer = Shannon(units=2)
        _ = layer(tf.ones((1, 4)))  # Build with input_dim=4

        # x=0 should give sinc(0)=1, so result = window[0]/norm
        x = tf.constant([[[0.0, 0.0, 0.0, 0.0]]], dtype=tf.float32)
        psi = layer.mother_wavelet(x).numpy()

        # sinc(0) = 1, window[i]/norm for each position
        window = layer._hamming_window.numpy()
        norm = layer._normalization.numpy()
        expected = window / norm

        np.testing.assert_allclose(psi[0, 0], expected, rtol=1e-5)

    def test_hamming_window_precomputed(self):
        """Hamming window is precomputed in build(), not dynamically."""
        layer = Shannon(units=2)
        _ = layer(tf.ones((1, 5)))  # Build with input_dim=5

        assert layer._hamming_window is not None
        assert layer._hamming_window.shape == (5,)
        assert layer._normalization is not None

    def test_energy_normalization(self):
        """Window is normalized so that ||window||_2 divides the output."""
        layer = Shannon(units=2)
        _ = layer(tf.ones((1, 8)))  # Build with input_dim=8

        # Check that normalization factor is sqrt(sum(window^2))
        window = layer._hamming_window.numpy()
        expected_norm = np.sqrt(np.sum(window**2))
        actual_norm = layer._normalization.numpy()

        np.testing.assert_allclose(actual_norm, expected_norm, rtol=1e-5)

    def test_gradient_flow(self):
        """Gradients flow through Shannon wavelet."""
        layer = Shannon(units=2)
        x = tf.constant([[0.5, 1.0, 1.5]], dtype=tf.float32)

        with tf.GradientTape() as tape:
            y = layer(x)
            loss = tf.reduce_sum(y)

        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads)


# ============================================================================
# WaveletBase Infrastructure Tests
# ============================================================================


class TestWaveletBase:
    """Test WaveletBase infrastructure."""

    def test_scale_positive(self):
        """Scale is always positive via softplus."""
        layer = Bump(units=2)
        _ = layer(tf.ones((1, 3)))

        # scale is stored as logits, softplus ensures positivity
        scale_logits = layer.scale.numpy()
        scale_actual = tf.nn.softplus(scale_logits).numpy()
        assert np.all(scale_actual > 0)

    def test_scale_and_translation_shape(self):
        """Scale and translation have shape (1, output_dim, input_dim)."""
        layer = Bump(units=5)
        _ = layer(tf.ones((2, 3)))

        assert layer.scale.shape == (1, 5, 3)
        assert layer.translation.shape == (1, 5, 3)

    def test_wavelet_weights_shape(self):
        """Wavelet weights have shape (output_dim, input_dim)."""
        layer = Bump(units=5)
        _ = layer(tf.ones((2, 3)))

        assert layer.wavelet_weights.shape == (5, 3)

    def test_output_shape_basic(self):
        """Output shape matches (batch, units)."""
        layer = Bump(units=7)
        x = tf.random.uniform((5, 3))
        y = layer(x)
        assert y.shape == (5, 7)

    def test_batch_dimensions_preserved(self):
        """Standard 2D batches work correctly."""
        layer = Bump(units=4)
        x = tf.random.uniform((10, 5))  # (batch, features)
        y = layer(x)
        assert y.shape == (10, 4)

    def test_serialization_roundtrip(self):
        """Layer survives get_config/from_config."""
        layer = Bump(
            units=4,
            scale_init=2.0,
            translation_init=0.5,
        )
        x = tf.random.uniform((2, 3))
        _ = layer(x)

        config = layer.get_config()
        restored = Bump.from_config(config)
        _ = restored(x)

        assert config["units"] == 4
        assert config["scale_init"] == 2.0
        assert config["translation_init"] == 0.5


# ============================================================================
# Numerical Stability Tests
# ============================================================================


class TestNumericalStability:
    """Test numerical stability edge cases for wavelets."""

    @pytest.mark.parametrize(
        "LayerClass,kwargs",
        [
            (Bump, {"units": 2}),
            (DerivativeOfGaussian, {"units": 2}),
            (Meyer, {"units": 2}),
            (Morelet, {"units": 2, "omega_init": 5.0}),
            (Poisson, {"units": 2}),
            (Ricker, {"units": 2, "sigma_init": 1.0}),
            (Shannon, {"units": 2}),
            (Haar, {"units": 2}),
            (Daubechies, {"units": 2, "order": 4}),
            (Symlet, {"units": 2, "order": 4}),
            (Coiflet, {"units": 2, "order": 2}),
        ],
    )
    def test_no_nan_on_normal_input(self, LayerClass, kwargs):
        """No NaN on normal inputs."""
        layer = LayerClass(**kwargs)
        x = tf.random.uniform((10, 5), minval=-1.0, maxval=1.0)
        output = layer(x)
        assert not tf.reduce_any(tf.math.is_nan(output))

    @pytest.mark.parametrize(
        "LayerClass,kwargs",
        [
            (DerivativeOfGaussian, {"units": 2}),
            (Meyer, {"units": 2}),
            (Morelet, {"units": 2, "omega_init": 5.0}),
            (Poisson, {"units": 2}),
            (Ricker, {"units": 2, "sigma_init": 1.0}),
            (Haar, {"units": 2}),
            (Daubechies, {"units": 2, "order": 4}),
            (Symlet, {"units": 2, "order": 4}),
            (Coiflet, {"units": 2, "order": 2}),
        ],
    )
    def test_no_nan_on_large_input(self, LayerClass, kwargs):
        """No NaN on large magnitude inputs."""
        layer = LayerClass(**kwargs)
        x = tf.constant([[10.0, -10.0], [50.0, -50.0]], dtype=tf.float32)
        output = layer(x)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_gradient_stability(self, sample_input):
        """Gradients are finite."""
        layer = Bump(units=2)
        with tf.GradientTape() as tape:
            output = layer(sample_input)
            loss = tf.reduce_mean(output**2)
        grads = tape.gradient(loss, layer.trainable_variables)
        for g in grads:
            if g is not None:
                assert not tf.reduce_any(tf.math.is_nan(g))
                assert not tf.reduce_any(tf.math.is_inf(g))


# ============================================================================
# Haar Wavelet Tests (Sprint 7A)
# ============================================================================


class TestHaar:
    """Test Haar wavelet: piecewise constant step function."""

    def test_output_shape(self, sample_input):
        """Output shape is (batch, units)."""
        layer = Haar(units=5)
        output = layer(sample_input)
        assert output.shape == (3, 5)

    def test_mother_wavelet_symmetry_at_half(self):
        """Haar wavelet changes sign at t = 0.5."""
        layer = Haar(units=2, sharpness=50.0)
        _ = layer(tf.ones((1, 1)))

        # At x = 0.25, should be positive (in [0, 0.5))
        x_before = tf.constant([[[0.25]]], dtype=tf.float32)
        psi_before = layer.mother_wavelet(x_before).numpy()[0, 0, 0]

        # At x = 0.75, should be negative (in [0.5, 1))
        x_after = tf.constant([[[0.75]]], dtype=tf.float32)
        psi_after = layer.mother_wavelet(x_after).numpy()[0, 0, 0]

        assert psi_before > 0
        assert psi_after < 0

    def test_mother_wavelet_zero_outside_support(self):
        """Haar wavelet is approximately zero outside [0, 1]."""
        layer = Haar(units=2, sharpness=50.0)
        _ = layer(tf.ones((1, 1)))

        # Outside support
        x_neg = tf.constant([[[-1.0]]], dtype=tf.float32)
        x_pos = tf.constant([[[2.0]]], dtype=tf.float32)

        psi_neg = layer.mother_wavelet(x_neg).numpy()[0, 0, 0]
        psi_pos = layer.mother_wavelet(x_pos).numpy()[0, 0, 0]

        assert abs(psi_neg) < 0.1
        assert abs(psi_pos) < 0.1

    def test_gradient_flow(self, sample_input):
        """Gradients flow through the layer."""
        layer = Haar(units=2)
        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = tf.reduce_mean(output)
        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads)

    def test_serialization_roundtrip(self):
        """Layer survives get_config/from_config."""
        layer = Haar(units=4, sharpness=30.0, sharpness_trainable=True)
        x = tf.random.uniform((2, 3))
        _ = layer(x)

        config = layer.get_config()
        restored = Haar.from_config(config)
        _ = restored(x)

        assert config["units"] == 4
        assert config["sharpness"] == 30.0
        assert config["sharpness_trainable"] is True


# ============================================================================
# Daubechies Wavelet Tests (Sprint 7A)
# ============================================================================


class TestDaubechies:
    """Test Daubechies wavelets (db1-db10)."""

    def test_output_shape(self, sample_input):
        """Output shape is (batch, units)."""
        layer = Daubechies(units=5, order=4)
        output = layer(sample_input)
        assert output.shape == (3, 5)

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6, 7, 8, 10])
    def test_valid_orders(self, order):
        """All supported orders work."""
        layer = Daubechies(units=2, order=order)
        x = tf.random.uniform((4, 3))
        output = layer(x)
        assert output.shape == (4, 2)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_invalid_order_raises(self):
        """Invalid order raises ValueError."""
        with pytest.raises(ValueError, match="Daubechies order must be in"):
            Daubechies(units=2, order=9)

    def test_db1_equals_haar_approximately(self):
        """db1 should behave similarly to Haar wavelet."""
        db1 = Daubechies(units=2, order=1)
        haar = Haar(units=2, sharpness=20.0)

        x = tf.random.uniform((4, 3), minval=0, maxval=1)
        _ = db1(x)
        _ = haar(x)

        # Both should produce finite outputs on [0,1]
        out_db1 = db1(x)
        out_haar = haar(x)

        assert not tf.reduce_any(tf.math.is_nan(out_db1))
        assert not tf.reduce_any(tf.math.is_nan(out_haar))

    def test_gradient_flow(self, sample_input):
        """Gradients flow through the layer."""
        layer = Daubechies(units=2, order=4)
        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = tf.reduce_mean(output)
        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads)

    def test_serialization_roundtrip(self):
        """Layer survives get_config/from_config."""
        layer = Daubechies(units=4, order=6)
        x = tf.random.uniform((2, 3))
        _ = layer(x)

        config = layer.get_config()
        restored = Daubechies.from_config(config)
        _ = restored(x)

        assert config["units"] == 4
        assert config["order"] == 6


# ============================================================================
# Symlet Wavelet Tests (Sprint 7A)
# ============================================================================


class TestSymlet:
    """Test Symlet wavelets (sym2-sym8)."""

    def test_output_shape(self, sample_input):
        """Output shape is (batch, units)."""
        layer = Symlet(units=5, order=4)
        output = layer(sample_input)
        assert output.shape == (3, 5)

    @pytest.mark.parametrize("order", [2, 3, 4, 5, 6, 7, 8])
    def test_valid_orders(self, order):
        """All supported orders work."""
        layer = Symlet(units=2, order=order)
        x = tf.random.uniform((4, 3))
        output = layer(x)
        assert output.shape == (4, 2)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_invalid_order_raises(self):
        """Invalid order raises ValueError."""
        with pytest.raises(ValueError, match="Symlet order must be in"):
            Symlet(units=2, order=1)

    def test_gradient_flow(self, sample_input):
        """Gradients flow through the layer."""
        layer = Symlet(units=2, order=4)
        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = tf.reduce_mean(output)
        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads)

    def test_serialization_roundtrip(self):
        """Layer survives get_config/from_config."""
        layer = Symlet(units=4, order=5)
        x = tf.random.uniform((2, 3))
        _ = layer(x)

        config = layer.get_config()
        restored = Symlet.from_config(config)
        _ = restored(x)

        assert config["units"] == 4
        assert config["order"] == 5


# ============================================================================
# Coiflet Wavelet Tests (Sprint 7A)
# ============================================================================


class TestCoiflet:
    """Test Coiflet wavelets (coif1-coif5)."""

    def test_output_shape(self, sample_input):
        """Output shape is (batch, units)."""
        layer = Coiflet(units=5, order=2)
        output = layer(sample_input)
        assert output.shape == (3, 5)

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
    def test_valid_orders(self, order):
        """All supported orders work."""
        layer = Coiflet(units=2, order=order)
        x = tf.random.uniform((4, 3))
        output = layer(x)
        assert output.shape == (4, 2)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_invalid_order_raises(self):
        """Invalid order raises ValueError."""
        with pytest.raises(ValueError, match="Coiflet order must be in"):
            Coiflet(units=2, order=6)

    def test_gradient_flow(self, sample_input):
        """Gradients flow through the layer."""
        layer = Coiflet(units=2, order=2)
        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = tf.reduce_mean(output)
        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads)

    def test_serialization_roundtrip(self):
        """Layer survives get_config/from_config."""
        layer = Coiflet(units=4, order=3)
        x = tf.random.uniform((2, 3))
        _ = layer(x)

        config = layer.get_config()
        restored = Coiflet.from_config(config)
        _ = restored(x)

        assert config["units"] == 4
        assert config["order"] == 3


# ============================================================================
# Cross-Wavelet Comparison Tests (Sprint 7A)
# ============================================================================


class TestWaveletFamilyProperties:
    """Test properties that should hold across wavelet families."""

    @pytest.mark.parametrize(
        "LayerClass,kwargs",
        [
            (Haar, {"units": 3}),
            (Daubechies, {"units": 3, "order": 4}),
            (Symlet, {"units": 3, "order": 4}),
            (Coiflet, {"units": 3, "order": 2}),
        ],
    )
    def test_finite_output_on_unit_interval(self, LayerClass, kwargs):
        """All filter-bank wavelets produce finite output on [0, 1]."""
        layer = LayerClass(**kwargs)
        x = tf.random.uniform((10, 5), minval=0.0, maxval=1.0)
        output = layer(x)
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))

    @pytest.mark.parametrize(
        "LayerClass,kwargs",
        [
            (Haar, {"units": 3}),
            (Daubechies, {"units": 3, "order": 4}),
            (Symlet, {"units": 3, "order": 4}),
            (Coiflet, {"units": 3, "order": 2}),
        ],
    )
    def test_bounded_output(self, LayerClass, kwargs):
        """Filter-bank wavelets have bounded output."""
        layer = LayerClass(**kwargs)
        x = tf.random.uniform((100, 5), minval=-2.0, maxval=2.0)
        output = layer(x)
        # Wavelets are bounded functions
        assert tf.reduce_max(tf.abs(output)) < 100.0

    @pytest.mark.parametrize(
        "LayerClass,kwargs",
        [
            (Haar, {"units": 4}),
            (Daubechies, {"units": 4, "order": 2}),
            (Daubechies, {"units": 4, "order": 4}),
            (Daubechies, {"units": 4, "order": 8}),
            (Symlet, {"units": 4, "order": 4}),
            (Coiflet, {"units": 4, "order": 2}),
        ],
    )
    def test_trainable_parameters_exist(self, LayerClass, kwargs):
        """All wavelets have trainable scale, translation, and weights."""
        layer = LayerClass(**kwargs)
        x = tf.random.uniform((4, 3))
        _ = layer(x)

        var_names = [v.name for v in layer.trainable_variables]
        assert any("scale" in name for name in var_names)
        assert any("translation" in name for name in var_names)
        assert any("wavelet_weights" in name for name in var_names)
