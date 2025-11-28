# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Correctness tests for spectral KAN layers.

This module tests:
- FourierKAN: Fourier/trigonometric basis
- RandomFourierFeatures: RBF kernel approximation via RFF

Tests cover:
- Output shapes and dtypes
- Basis function properties (orthogonality, periodicity)
- Gradient flow
- Serialization roundtrip
- Parameter validation
- Numerical stability
"""
import math

import numpy as np
import pytest
import tensorflow as tf

from arnold.layers.core.registry import get_layer
from arnold.layers.core.spectral import (
    FourierKAN,
    RandomFourierFeatures,
)


# =============================================================================
# FourierKAN Tests
# =============================================================================


class TestFourierKAN:
    """Tests for FourierKAN layer."""

    def test_output_shape(self):
        """Test that output shape is (batch, units)."""
        layer = FourierKAN(units=32, degree=8)
        x = tf.random.uniform((16, 10), -1, 1)
        y = layer(x)
        assert y.shape == (16, 32)

    def test_output_shape_multidim(self):
        """Test with higher-dimensional input."""
        layer = FourierKAN(units=64, degree=4)
        x = tf.random.uniform((8, 4, 5), -1, 1)
        y = layer(x)
        assert y.shape == (8, 4, 64)

    @pytest.mark.parametrize("degree", [1, 4, 8, 16, 32])
    def test_valid_degrees(self, degree):
        """Test various valid degrees."""
        layer = FourierKAN(units=16, degree=degree)
        x = tf.random.uniform((8, 5), -1, 1)
        y = layer(x)
        assert y.shape == (8, 16)
        assert not tf.reduce_any(tf.math.is_nan(y))

    def test_num_basis_functions(self):
        """Test that num_basis = 2*degree + 1."""
        for degree in [1, 5, 10, 20]:
            layer = FourierKAN(units=16, degree=degree)
            layer.build((None, 4))
            assert layer._get_num_basis_functions() == 2 * degree + 1

    def test_basis_periodicity(self):
        """Test that Fourier basis is periodic with period 2π/ω."""
        layer = FourierKAN(units=16, degree=8, frequency=1.0)
        layer.build((None, 1))

        # Points separated by 2π should give same basis values
        x1 = tf.constant([[0.5]])
        x2 = tf.constant([[0.5 + 2 * math.pi]])

        basis1 = layer.spectral_basis(x1)
        basis2 = layer.spectral_basis(x2)

        np.testing.assert_allclose(basis1.numpy(), basis2.numpy(), atol=1e-5)

    def test_basis_dc_component(self):
        """Test that first basis function is constant 1."""
        layer = FourierKAN(units=16, degree=4)
        layer.build((None, 3))

        x = tf.random.uniform((10, 3), -5, 5)
        basis = layer.spectral_basis(x)

        # First component should be 1 for all inputs
        dc = basis[..., 0].numpy()
        np.testing.assert_allclose(dc, np.ones_like(dc), atol=1e-6)

    def test_learnable_frequency(self):
        """Test that frequency is trainable when learnable_frequency=True."""
        layer = FourierKAN(units=16, degree=4, learnable_frequency=True)
        layer.build((None, 5))

        # Check omega is trainable
        assert layer._omega.trainable
        trainable_names = [v.name for v in layer.trainable_variables]
        assert any("omega" in name for name in trainable_names)

    def test_fixed_frequency(self):
        """Test that frequency is not trainable by default."""
        layer = FourierKAN(units=16, degree=4, learnable_frequency=False)
        layer.build((None, 5))

        # Omega should not be in trainable variables
        trainable_names = [v.name for v in layer.trainable_variables]
        assert not any("omega" in name for name in trainable_names)

    def test_custom_frequency(self):
        """Test initialization with custom frequency."""
        freq = 3.14159
        layer = FourierKAN(units=16, degree=4, frequency=freq)
        layer.build((None, 3))

        np.testing.assert_allclose(layer._omega.numpy(), freq, atol=1e-5)

    def test_gradient_flow(self):
        """Test that gradients flow through the layer."""
        layer = FourierKAN(units=16, degree=8)
        x = tf.random.uniform((8, 5), -1, 1)

        with tf.GradientTape() as tape:
            y = layer(x)
            loss = tf.reduce_mean(y ** 2)

        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads)
        assert all(not tf.reduce_any(tf.math.is_nan(g)) for g in grads)

    def test_serialization_roundtrip(self):
        """Test that layer can be serialized and reconstructed."""
        layer = FourierKAN(
            units=32,
            degree=12,
            frequency=2.0,
            learnable_frequency=True,
            use_bias=False,
        )
        layer.build((None, 10))

        config = layer.get_config()
        restored = FourierKAN.from_config(config)
        restored.build((None, 10))

        assert restored.units == layer.units
        assert restored.degree == layer.degree
        assert restored.frequency_init == layer.frequency_init
        assert restored.learnable_frequency == layer.learnable_frequency
        assert restored.use_bias == layer.use_bias

    def test_invalid_degree_raises(self):
        """Test that degree < 1 raises ValueError."""
        with pytest.raises(ValueError, match="degree must be >= 1"):
            FourierKAN(units=16, degree=0)

    def test_invalid_frequency_raises(self):
        """Test that frequency <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="frequency must be > 0"):
            FourierKAN(units=16, degree=4, frequency=-1.0)

    def test_omega_property(self):
        """Test omega property accessor."""
        layer = FourierKAN(units=16, degree=4, frequency=2.5)
        layer.build((None, 3))
        assert layer.omega is not None
        np.testing.assert_allclose(layer.omega.numpy(), 2.5, atol=1e-5)


# =============================================================================
# RandomFourierFeatures Tests
# =============================================================================


class TestRandomFourierFeatures:
    """Tests for RandomFourierFeatures layer."""

    def test_output_shape(self):
        """Test that output shape is (batch, units)."""
        layer = RandomFourierFeatures(units=32, num_features=64)
        x = tf.random.uniform((16, 10), -1, 1)
        y = layer(x)
        assert y.shape == (16, 32)

    def test_output_shape_multidim(self):
        """Test with higher-dimensional input."""
        layer = RandomFourierFeatures(units=64, num_features=128)
        x = tf.random.uniform((8, 4, 5), -1, 1)
        y = layer(x)
        assert y.shape == (8, 4, 64)

    @pytest.mark.parametrize("num_features", [16, 64, 128, 256])
    def test_valid_num_features(self, num_features):
        """Test various valid num_features."""
        layer = RandomFourierFeatures(units=16, num_features=num_features)
        x = tf.random.uniform((8, 5), -1, 1)
        y = layer(x)
        assert y.shape == (8, 16)
        assert not tf.reduce_any(tf.math.is_nan(y))

    def test_num_basis_functions(self):
        """Test that num_basis = num_features."""
        for num_features in [32, 64, 128]:
            layer = RandomFourierFeatures(units=16, num_features=num_features)
            layer.build((None, 4))
            assert layer._get_num_basis_functions() == num_features

    def test_basis_bounded(self):
        """Test that RFF basis is bounded in [-sqrt(2/D), sqrt(2/D)]."""
        num_features = 64
        layer = RandomFourierFeatures(units=16, num_features=num_features)
        layer.build((None, 5))

        x = tf.random.uniform((100, 5), -10, 10)
        basis = layer.spectral_basis(x)

        # Normalization factor is sqrt(2/D)
        bound = np.sqrt(2.0 / num_features)
        assert tf.reduce_all(tf.abs(basis) <= bound + 1e-6)

    def test_trainable_frequencies(self):
        """Test that frequencies are trainable when trainable_frequencies=True."""
        layer = RandomFourierFeatures(
            units=16, num_features=64, trainable_frequencies=True
        )
        layer.build((None, 5))

        # Check omega is trainable
        assert layer._omega.trainable
        trainable_names = [v.name for v in layer.trainable_variables]
        assert any("omega" in name for name in trainable_names)

    def test_trainable_phases(self):
        """Test that phases are trainable when trainable_phases=True."""
        layer = RandomFourierFeatures(
            units=16, num_features=64, trainable_phases=True
        )
        layer.build((None, 5))

        # Check phase is trainable
        assert layer._phase.trainable
        trainable_names = [v.name for v in layer.trainable_variables]
        assert any("phase" in name for name in trainable_names)

    def test_fixed_frequencies_and_phases(self):
        """Test that frequencies/phases are not trainable by default."""
        layer = RandomFourierFeatures(units=16, num_features=64)
        layer.build((None, 5))

        # omega and phase should not be in trainable variables
        trainable_names = [v.name for v in layer.trainable_variables]
        assert not any("omega" in name for name in trainable_names)
        assert not any("phase" in name for name in trainable_names)

    def test_kernel_scale(self):
        """Test that kernel_scale affects frequency sampling."""
        # Two layers with different kernel_scale should have different omega std
        layer1 = RandomFourierFeatures(units=16, num_features=256, kernel_scale=0.5, seed=42)
        layer2 = RandomFourierFeatures(units=16, num_features=256, kernel_scale=2.0, seed=42)
        layer1.build((None, 5))
        layer2.build((None, 5))

        std1 = tf.math.reduce_std(layer1._omega)
        std2 = tf.math.reduce_std(layer2._omega)

        # kernel_scale=2.0 should have 4x variance (2x std) of kernel_scale=0.5
        np.testing.assert_allclose(std2.numpy() / std1.numpy(), 4.0, rtol=0.2)

    def test_seed_reproducibility(self):
        """Test that same seed gives same frequencies."""
        layer1 = RandomFourierFeatures(units=16, num_features=64, seed=12345)
        layer2 = RandomFourierFeatures(units=16, num_features=64, seed=12345)
        layer1.build((None, 5))
        layer2.build((None, 5))

        np.testing.assert_allclose(layer1._omega.numpy(), layer2._omega.numpy())
        np.testing.assert_allclose(layer1._phase.numpy(), layer2._phase.numpy())

    def test_gradient_flow(self):
        """Test that gradients flow through the layer."""
        layer = RandomFourierFeatures(units=16, num_features=64)
        x = tf.random.uniform((8, 5), -1, 1)

        with tf.GradientTape() as tape:
            y = layer(x)
            loss = tf.reduce_mean(y ** 2)

        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads)
        assert all(not tf.reduce_any(tf.math.is_nan(g)) for g in grads)

    def test_serialization_roundtrip(self):
        """Test that layer can be serialized and reconstructed."""
        layer = RandomFourierFeatures(
            units=32,
            num_features=128,
            kernel_scale=0.5,
            trainable_frequencies=True,
            trainable_phases=True,
            seed=42,
            use_bias=False,
        )
        layer.build((None, 10))

        config = layer.get_config()
        restored = RandomFourierFeatures.from_config(config)
        restored.build((None, 10))

        assert restored.units == layer.units
        assert restored.num_features == layer.num_features
        assert restored.kernel_scale == layer.kernel_scale
        assert restored.trainable_frequencies == layer.trainable_frequencies
        assert restored.trainable_phases == layer.trainable_phases
        assert restored.seed == layer.seed
        assert restored.use_bias == layer.use_bias

    def test_invalid_num_features_raises(self):
        """Test that num_features < 1 raises ValueError."""
        with pytest.raises(ValueError, match="num_features must be >= 1"):
            RandomFourierFeatures(units=16, num_features=0)

    def test_invalid_kernel_scale_raises(self):
        """Test that kernel_scale <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="kernel_scale must be > 0"):
            RandomFourierFeatures(units=16, num_features=64, kernel_scale=-1.0)

    def test_frequencies_property(self):
        """Test frequencies property accessor."""
        layer = RandomFourierFeatures(units=16, num_features=64, seed=42)
        layer.build((None, 5))
        assert layer.frequencies is not None
        assert layer.frequencies.shape == (5, 64)

    def test_phases_property(self):
        """Test phases property accessor."""
        layer = RandomFourierFeatures(units=16, num_features=64, seed=42)
        layer.build((None, 5))
        assert layer.phases is not None
        assert layer.phases.shape == (64,)
        # Phases should be in [0, 2π]
        assert tf.reduce_all(layer.phases >= 0)
        assert tf.reduce_all(layer.phases <= 2 * math.pi)


# =============================================================================
# Registry Tests
# =============================================================================


class TestSpectralRegistry:
    """Test that spectral layers are properly registered."""

    def test_fourier_in_registry(self):
        """Test FourierKAN is accessible via registry."""
        layer = get_layer("fourier", units=16, degree=4)
        assert isinstance(layer, FourierKAN)

    def test_fourier_aliases(self):
        """Test FourierKAN aliases."""
        for name in ["fourier", "fourier_kan", "trigonometric"]:
            layer = get_layer(name, units=16, degree=4)
            assert isinstance(layer, FourierKAN)

    def test_rff_in_registry(self):
        """Test RandomFourierFeatures is accessible via registry."""
        layer = get_layer("rff", units=16, num_features=32)
        assert isinstance(layer, RandomFourierFeatures)

    def test_rff_aliases(self):
        """Test RandomFourierFeatures aliases."""
        for name in ["rff", "random_fourier_features", "random_fourier"]:
            layer = get_layer(name, units=16, num_features=32)
            assert isinstance(layer, RandomFourierFeatures)

    def test_spectral_in_categories(self):
        """Test spectral category exists in layer categories."""
        from arnold.layers.core.registry import list_layers_by_category

        categories = list_layers_by_category()
        assert "spectral" in categories
        assert "fourier" in categories["spectral"]
        assert "rff" in categories["spectral"]


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestSpectralNumericalStability:
    """Test numerical stability of spectral layers."""

    @pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
    def test_fourier_dtype_stability(self, dtype):
        """Test FourierKAN with different dtypes."""
        layer = FourierKAN(units=16, degree=16)
        x = tf.random.uniform((8, 5), -10, 10, dtype=dtype)
        y = layer(x)
        assert not tf.reduce_any(tf.math.is_nan(y))
        assert not tf.reduce_any(tf.math.is_inf(y))

    @pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
    def test_rff_dtype_stability(self, dtype):
        """Test RandomFourierFeatures with different dtypes."""
        layer = RandomFourierFeatures(units=16, num_features=64)
        x = tf.random.uniform((8, 5), -10, 10, dtype=dtype)
        y = layer(x)
        assert not tf.reduce_any(tf.math.is_nan(y))
        assert not tf.reduce_any(tf.math.is_inf(y))

    def test_fourier_large_input(self):
        """Test FourierKAN with large input values."""
        layer = FourierKAN(units=16, degree=8)
        x = tf.random.uniform((8, 5), -1000, 1000)
        y = layer(x)
        assert not tf.reduce_any(tf.math.is_nan(y))

    def test_rff_large_input(self):
        """Test RandomFourierFeatures with large input values."""
        layer = RandomFourierFeatures(units=16, num_features=64)
        x = tf.random.uniform((8, 5), -1000, 1000)
        y = layer(x)
        assert not tf.reduce_any(tf.math.is_nan(y))

    def test_fourier_high_degree(self):
        """Test FourierKAN with high degree."""
        layer = FourierKAN(units=16, degree=100)
        x = tf.random.uniform((8, 5), -1, 1)
        y = layer(x)
        assert not tf.reduce_any(tf.math.is_nan(y))

    def test_rff_many_features(self):
        """Test RandomFourierFeatures with many features."""
        layer = RandomFourierFeatures(units=16, num_features=1024)
        x = tf.random.uniform((8, 5), -1, 1)
        y = layer(x)
        assert not tf.reduce_any(tf.math.is_nan(y))


# =============================================================================
# Training Tests
# =============================================================================


class TestSpectralTraining:
    """Test spectral layers in training scenarios."""

    def test_fourier_simple_regression(self):
        """Test FourierKAN can learn a simple periodic function."""
        tf.random.set_seed(42)

        # Generate periodic data
        x_train = tf.random.uniform((200, 1), -math.pi, math.pi)
        y_train = tf.sin(x_train) + 0.5 * tf.cos(2 * x_train)

        # Build model with explicit Input layer
        inputs = tf.keras.Input(shape=(1,))
        fourier = FourierKAN(units=32, degree=4, frequency=1.0)(inputs)
        outputs = tf.keras.layers.Dense(1)(fourier)
        model = tf.keras.Model(inputs, outputs)

        model.compile(optimizer="adam", loss="mse")
        history = model.fit(x_train, y_train, epochs=50, verbose=0)

        # Loss should decrease significantly
        assert history.history["loss"][-1] < history.history["loss"][0] * 0.1

    def test_rff_simple_regression(self):
        """Test RandomFourierFeatures can learn a simple function."""
        tf.random.set_seed(42)

        # Generate smooth data (RBF kernel should work well)
        x_train = tf.random.uniform((200, 2), -1, 1)
        y_train = tf.exp(-tf.reduce_sum(x_train ** 2, axis=-1, keepdims=True))

        # Build model with explicit Input layer
        inputs = tf.keras.Input(shape=(2,))
        rff = RandomFourierFeatures(units=64, num_features=128, kernel_scale=1.0)(inputs)
        outputs = tf.keras.layers.Dense(1)(rff)
        model = tf.keras.Model(inputs, outputs)

        model.compile(optimizer="adam", loss="mse")
        history = model.fit(x_train, y_train, epochs=50, verbose=0)

        # Loss should decrease
        assert history.history["loss"][-1] < history.history["loss"][0] * 0.5

    def test_fourier_model_save_load(self):
        """Test FourierKAN model can be saved and loaded."""
        import os
        import tempfile

        # Use Functional API for reliable serialization
        inputs = tf.keras.Input(shape=(5,))
        layer = FourierKAN(units=16, degree=4)
        outputs = layer(inputs)
        model = tf.keras.Model(inputs, outputs)

        x = tf.random.uniform((4, 5))
        y1 = model(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.keras")
            model.save(path)
            loaded = tf.keras.models.load_model(path)
            y2 = loaded(x)

        np.testing.assert_allclose(y1.numpy(), y2.numpy(), atol=1e-5)

    def test_rff_model_save_load(self):
        """Test RandomFourierFeatures model can be saved and loaded."""
        import os
        import tempfile

        # Use Functional API for reliable serialization
        inputs = tf.keras.Input(shape=(5,))
        layer = RandomFourierFeatures(units=16, num_features=32, seed=42)
        outputs = layer(inputs)
        model = tf.keras.Model(inputs, outputs)

        x = tf.random.uniform((4, 5))
        y1 = model(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.keras")
            model.save(path)
            loaded = tf.keras.models.load_model(path)
            y2 = loaded(x)

        np.testing.assert_allclose(y1.numpy(), y2.numpy(), atol=1e-5)
