# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
"""
RBF Correctness Tests - Phase 4.1

Tests verify that each RBF kernel implementation:
1. get_kernels(r) produces correct mathematical output
2. Full layer output has correct shape
3. Gradients flow correctly
4. Serialization works
"""

import numpy as np
import pytest
import tensorflow as tf

from arnold.layers.core.radial_basis_functions import (
    CauchyRBF,
    CubicRBF,
    ExponentialRBF,
    GaussianRBF,
    InverseMultiQuadricRBF,
    InverseQuadricRBF,
    LinearRBF,
    MultiquadricRBF,
    PowerRBF,
    RBFBase,
    ThinPlateSplineRBF,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_radii():
    """Sample radii tensor for testing get_kernels()."""
    return tf.constant([[[0.0], [0.5], [1.0], [2.0], [5.0]]], dtype=tf.float32)


@pytest.fixture
def sample_input():
    """Sample input for full layer testing."""
    return tf.constant([[0.1, 0.5], [0.3, 0.7], [0.9, 0.2]], dtype=tf.float32)


# ============================================================================
# Parameterless Kernel Tests (Cubic, Linear, ThinPlateSpline)
# ============================================================================


class TestLinearRBF:
    """Test LinearRBF: φ(r) = r"""

    def test_kernel_formula(self, sample_radii):
        """LinearRBF kernel is identity function."""
        layer = LinearRBF(units=2, grid_min=0.0, grid_max=1.0, num_grids=8)
        _ = layer(tf.ones((1, 3)))  # build

        kernels = layer.get_kernels(sample_radii)
        expected = sample_radii

        np.testing.assert_allclose(kernels.numpy(), expected.numpy(), rtol=1e-5)

    def test_output_shape(self, sample_input):
        """Output shape is (batch, units)."""
        layer = LinearRBF(units=5, num_grids=8)
        output = layer(sample_input)
        assert output.shape == (3, 5)

    def test_gradient_flow(self, sample_input):
        """Gradients flow through the layer."""
        layer = LinearRBF(units=2)
        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = tf.reduce_mean(output)
        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads)


class TestCubicRBF:
    """Test CubicRBF: φ(r) = r³"""

    def test_kernel_formula(self, sample_radii):
        """CubicRBF kernel is r^3."""
        layer = CubicRBF(units=2, grid_min=0.0, grid_max=1.0, num_grids=8)
        _ = layer(tf.ones((1, 3)))  # build

        kernels = layer.get_kernels(sample_radii)
        expected = sample_radii**3

        np.testing.assert_allclose(kernels.numpy(), expected.numpy(), rtol=1e-5)

    def test_zero_at_origin(self, sample_radii):
        """φ(0) = 0"""
        layer = CubicRBF(units=2)
        _ = layer(tf.ones((1, 3)))
        r_zero = tf.constant([[[0.0]]], dtype=tf.float32)
        assert layer.get_kernels(r_zero).numpy()[0, 0, 0] == 0.0


class TestThinPlateSplineRBF:
    """Test ThinPlateSplineRBF: φ(r) = r² * ln(r)"""

    def test_kernel_formula_nonzero(self):
        """ThinPlateSplineRBF matches r² * ln(r) for r > 0."""
        layer = ThinPlateSplineRBF(units=2, grid_min=0.0, grid_max=1.0, num_grids=8)
        _ = layer(tf.ones((1, 3)))  # build

        r = tf.constant([[[1.0], [2.0], [np.e]]], dtype=tf.float32)
        kernels = layer.get_kernels(r).numpy()

        # r=1: 1² * ln(1) = 0
        assert np.abs(kernels[0, 0, 0]) < 1e-5
        # r=2: 4 * ln(2) ≈ 2.773
        np.testing.assert_allclose(kernels[0, 1, 0], 4 * np.log(2), rtol=1e-5)
        # r=e: e² * ln(e) = e² ≈ 7.389
        np.testing.assert_allclose(kernels[0, 2, 0], np.e**2, rtol=1e-5)

    def test_near_zero_handling(self):
        """Small r values don't produce NaN."""
        layer = ThinPlateSplineRBF(units=2)
        _ = layer(tf.ones((1, 3)))
        r_small = tf.constant([[[1e-8], [1e-10]]], dtype=tf.float32)
        kernels = layer.get_kernels(r_small)
        assert not tf.reduce_any(tf.math.is_nan(kernels))


# ============================================================================
# Parameterized Kernel Tests
# ============================================================================


class TestGaussianRBF:
    """Test GaussianRBF: φ(r) = exp(-(ε*r)²)"""

    def test_kernel_at_zero(self):
        """φ(0) = 1 for any epsilon."""
        layer = GaussianRBF(units=2, epsilon_trainable=False)
        # Build with input_dim=1 to match our r shape
        _ = layer(tf.ones((1, 1)))

        r_zero = tf.constant([[[0.0]]], dtype=tf.float32)
        kernel_at_zero = layer.get_kernels(r_zero).numpy()[0, 0, 0]
        np.testing.assert_allclose(kernel_at_zero, 1.0, rtol=1e-5)

    def test_kernel_decay(self):
        """Kernel decays monotonically with increasing r."""
        layer = GaussianRBF(units=2, num_grids=4)
        # Build with input_dim=1
        _ = layer(tf.ones((1, 1)))

        # r shape: (batch=1, input_dim=1, num_grids=4)
        r = tf.constant([[[0.0, 0.5, 1.0, 2.0]]], dtype=tf.float32)
        kernels = layer.get_kernels(r).numpy()[0, 0, :]  # get the grid dimension

        # Each successive value should be smaller
        for i in range(len(kernels) - 1):
            assert kernels[i] > kernels[i + 1], f"Kernel not decaying: {kernels}"

    def test_kernel_bounded_01(self):
        """Gaussian kernel is in (0, 1]."""
        layer = GaussianRBF(units=2, num_grids=4)
        _ = layer(tf.ones((1, 1)))

        r = tf.constant([[[0.0, 1.0, 5.0, 10.0]]], dtype=tf.float32)
        kernels = layer.get_kernels(r).numpy()

        assert np.all(kernels > 0)
        assert np.all(kernels <= 1.0)

    def test_gradient_wrt_epsilon(self, sample_input):
        """Gradients flow to epsilon logits."""
        layer = GaussianRBF(units=2, epsilon_trainable=True)
        with tf.GradientTape() as tape:
            output = layer(sample_input)
            loss = tf.reduce_mean(output)
        grads = tape.gradient(loss, layer.trainable_variables)
        epsilon_grad = [g for g, v in zip(grads, layer.trainable_variables) if "epsilon" in v.name]
        assert len(epsilon_grad) == 1
        assert epsilon_grad[0] is not None


class TestExponentialRBF:
    """Test ExponentialRBF: φ(r) = exp(-r/σ)"""

    def test_kernel_at_zero(self):
        """φ(0) = 1 for any sigma."""
        layer = ExponentialRBF(units=2, num_grids=2)
        _ = layer(tf.ones((1, 1)))

        r_zero = tf.constant([[[0.0, 0.0]]], dtype=tf.float32)
        kernel_at_zero = layer.get_kernels(r_zero).numpy()[0, 0, 0]
        np.testing.assert_allclose(kernel_at_zero, 1.0, rtol=1e-5)

    def test_kernel_decay(self):
        """Kernel decays monotonically."""
        layer = ExponentialRBF(units=2, num_grids=4)
        _ = layer(tf.ones((1, 1)))

        r = tf.constant([[[0.0, 0.5, 1.0, 2.0]]], dtype=tf.float32)
        kernels = layer.get_kernels(r).numpy()[0, 0, :]

        for i in range(len(kernels) - 1):
            assert kernels[i] > kernels[i + 1]

    def test_kernel_bounded_01(self):
        """Exponential kernel is in (0, 1]."""
        layer = ExponentialRBF(units=2, num_grids=3)
        _ = layer(tf.ones((1, 1)))

        r = tf.constant([[[0.0, 1.0, 5.0]]], dtype=tf.float32)
        kernels = layer.get_kernels(r).numpy()

        assert np.all(kernels > 0)
        assert np.all(kernels <= 1.0)


class TestMultiquadricRBF:
    """Test MultiquadricRBF: φ(r) = √(1 + (ε*r)²)"""

    def test_kernel_at_zero(self):
        """φ(0) = 1 for any epsilon."""
        layer = MultiquadricRBF(units=2, num_grids=2)
        _ = layer(tf.ones((1, 1)))

        r_zero = tf.constant([[[0.0, 0.0]]], dtype=tf.float32)
        kernel_at_zero = layer.get_kernels(r_zero).numpy()[0, 0, 0]
        np.testing.assert_allclose(kernel_at_zero, 1.0, rtol=1e-5)

    def test_kernel_monotonically_increasing(self):
        """Multiquadric increases with r."""
        layer = MultiquadricRBF(units=2, num_grids=4)
        _ = layer(tf.ones((1, 1)))

        r = tf.constant([[[0.0, 0.5, 1.0, 2.0]]], dtype=tf.float32)
        kernels = layer.get_kernels(r).numpy()[0, 0, :]

        for i in range(len(kernels) - 1):
            assert kernels[i] < kernels[i + 1]

    def test_kernel_lower_bound(self):
        """Multiquadric is >= 1."""
        layer = MultiquadricRBF(units=2, num_grids=3)
        _ = layer(tf.ones((1, 1)))

        r = tf.constant([[[0.0, 1.0, 10.0]]], dtype=tf.float32)
        kernels = layer.get_kernels(r).numpy()

        assert np.all(kernels >= 1.0)


class TestInverseMultiQuadricRBF:
    """Test InverseMultiQuadricRBF: φ(r) = 1 / √(1 + (ε*r)²)"""

    def test_kernel_at_zero(self):
        """φ(0) = 1 for any epsilon."""
        layer = InverseMultiQuadricRBF(units=2, num_grids=2)
        _ = layer(tf.ones((1, 1)))

        r_zero = tf.constant([[[0.0, 0.0]]], dtype=tf.float32)
        kernel_at_zero = layer.get_kernels(r_zero).numpy()[0, 0, 0]
        np.testing.assert_allclose(kernel_at_zero, 1.0, rtol=1e-5)

    def test_kernel_monotonically_decreasing(self):
        """Inverse Multiquadric decreases with r."""
        layer = InverseMultiQuadricRBF(units=2, num_grids=4)
        _ = layer(tf.ones((1, 1)))

        r = tf.constant([[[0.0, 0.5, 1.0, 2.0]]], dtype=tf.float32)
        kernels = layer.get_kernels(r).numpy()[0, 0, :]

        for i in range(len(kernels) - 1):
            assert kernels[i] > kernels[i + 1]

    def test_kernel_bounded_01(self):
        """Inverse Multiquadric is in (0, 1]."""
        layer = InverseMultiQuadricRBF(units=2, num_grids=3)
        _ = layer(tf.ones((1, 1)))

        r = tf.constant([[[0.0, 1.0, 10.0]]], dtype=tf.float32)
        kernels = layer.get_kernels(r).numpy()

        assert np.all(kernels > 0)
        assert np.all(kernels <= 1.0)


class TestInverseQuadricRBF:
    """Test InverseQuadricRBF: φ(r) = 1 / (1 + (ε*r)²)"""

    def test_kernel_at_zero(self):
        """φ(0) = 1 for any epsilon."""
        layer = InverseQuadricRBF(units=2, num_grids=2)
        _ = layer(tf.ones((1, 1)))

        r_zero = tf.constant([[[0.0, 0.0]]], dtype=tf.float32)
        kernel_at_zero = layer.get_kernels(r_zero).numpy()[0, 0, 0]
        np.testing.assert_allclose(kernel_at_zero, 1.0, rtol=1e-5)

    def test_kernel_monotonically_decreasing(self):
        """Inverse Quadric decreases with r."""
        layer = InverseQuadricRBF(units=2, num_grids=4)
        _ = layer(tf.ones((1, 1)))

        r = tf.constant([[[0.0, 0.5, 1.0, 2.0]]], dtype=tf.float32)
        kernels = layer.get_kernels(r).numpy()[0, 0, :]

        for i in range(len(kernels) - 1):
            assert kernels[i] > kernels[i + 1]

    def test_kernel_bounded_01(self):
        """Inverse Quadric is in (0, 1]."""
        layer = InverseQuadricRBF(units=2, num_grids=3)
        _ = layer(tf.ones((1, 1)))

        r = tf.constant([[[0.0, 1.0, 10.0]]], dtype=tf.float32)
        kernels = layer.get_kernels(r).numpy()

        assert np.all(kernels > 0)
        assert np.all(kernels <= 1.0)


class TestCauchyRBF:
    """Test CauchyRBF: φ(r) = 1 / (1 + (r/γ)²)"""

    def test_kernel_at_zero(self):
        """φ(0) = 1 for any gamma."""
        layer = CauchyRBF(units=2, num_grids=2)
        _ = layer(tf.ones((1, 1)))

        r_zero = tf.constant([[[0.0, 0.0]]], dtype=tf.float32)
        kernel_at_zero = layer.get_kernels(r_zero).numpy()[0, 0, 0]
        np.testing.assert_allclose(kernel_at_zero, 1.0, rtol=1e-5)

    def test_kernel_monotonically_decreasing(self):
        """Cauchy decreases with r."""
        layer = CauchyRBF(units=2, num_grids=4)
        _ = layer(tf.ones((1, 1)))

        r = tf.constant([[[0.0, 0.5, 1.0, 2.0]]], dtype=tf.float32)
        kernels = layer.get_kernels(r).numpy()[0, 0, :]

        for i in range(len(kernels) - 1):
            assert kernels[i] > kernels[i + 1]

    def test_kernel_bounded_01(self):
        """Cauchy is in (0, 1]."""
        layer = CauchyRBF(units=2, num_grids=3)
        _ = layer(tf.ones((1, 1)))

        r = tf.constant([[[0.0, 1.0, 10.0]]], dtype=tf.float32)
        kernels = layer.get_kernels(r).numpy()

        assert np.all(kernels > 0)
        assert np.all(kernels <= 1.0)


class TestPowerRBF:
    """Test PowerRBF: φ(r) = r^p"""

    def test_kernel_near_zero(self):
        """φ(0) ≈ 0 for positive power (with floor)."""
        layer = PowerRBF(units=2, num_grids=2)
        _ = layer(tf.ones((1, 1)))

        r_zero = tf.constant([[[0.0, 0.0]]], dtype=tf.float32)
        kernel_at_zero = layer.get_kernels(r_zero).numpy()[0, 0, 0]
        # Due to floor at 1e-6, should be very small
        assert kernel_at_zero < 1e-3

    def test_kernel_positive(self):
        """Power kernel is positive for r > 0."""
        layer = PowerRBF(units=2, num_grids=3)
        _ = layer(tf.ones((1, 1)))

        r = tf.constant([[[0.1, 1.0, 2.0]]], dtype=tf.float32)
        kernels = layer.get_kernels(r).numpy()

        assert np.all(kernels > 0)

    def test_kernel_monotonically_increasing(self):
        """Power kernel increases with r for p > 0."""
        layer = PowerRBF(units=2, num_grids=4)
        _ = layer(tf.ones((1, 1)))

        r = tf.constant([[[0.5, 1.0, 2.0, 3.0]]], dtype=tf.float32)
        kernels = layer.get_kernels(r).numpy()[0, 0, :]

        for i in range(len(kernels) - 1):
            assert kernels[i] < kernels[i + 1]


# ============================================================================
# RBFBase Infrastructure Tests
# ============================================================================


class TestRBFBase:
    """Test RBFBase infrastructure."""

    def test_grid_creation(self):
        """Grid is created correctly."""
        layer = GaussianRBF(units=2, grid_min=0.0, grid_max=1.0, num_grids=5)
        _ = layer(tf.ones((1, 3)))

        expected_grid = np.linspace(0.0, 1.0, 5)
        np.testing.assert_allclose(layer.grid.numpy(), expected_grid, rtol=1e-5)

    def test_radii_calculation(self):
        """Radii are normalized by grid spacing."""
        layer = GaussianRBF(units=2, grid_min=0.0, grid_max=1.0, num_grids=5)
        _ = layer(tf.ones((1, 3)))

        # Input exactly at a grid point should have r=0 for that point
        x = tf.constant([[[0.25]]], dtype=tf.float32)  # grid_min + spacing = 0.25
        radii = layer.radii(x).numpy()

        # Spacing is 0.25, so radii should be |x - grid_i| / spacing
        # grid = [0, 0.25, 0.5, 0.75, 1.0]
        # r at x=0.25: [1, 0, 1, 2, 3]
        expected = np.array([1.0, 0.0, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(radii[0, 0, :], expected, rtol=1e-5)

    def test_output_shape_basic(self):
        """Output shape matches (batch, units)."""
        layer = GaussianRBF(units=7, num_grids=10)
        x = tf.random.uniform((5, 3))
        y = layer(x)
        assert y.shape == (5, 7)

    def test_batch_dimensions_preserved(self):
        """Leading batch dimensions are preserved."""
        layer = GaussianRBF(units=4, num_grids=8)
        x = tf.random.uniform((2, 3, 4, 5))  # complex batch shape
        y = layer(x)
        assert y.shape == (2, 3, 4, 4)

    def test_kernel_weights_shape(self):
        """Kernel weights have correct shape."""
        layer = GaussianRBF(units=5, num_grids=8)
        x = tf.random.uniform((2, 3))
        _ = layer(x)
        assert layer.kernel_weights.shape == (3, 8, 5)  # (input_dim, num_grids, units)

    def test_serialization_roundtrip(self):
        """Layer survives get_config/from_config."""
        layer = GaussianRBF(
            units=4,
            grid_min=-1.0,
            grid_max=2.0,
            num_grids=12,
            epsilon_init=0.5,
        )
        x = tf.random.uniform((2, 3))
        _ = layer(x)

        config = layer.get_config()
        restored = GaussianRBF.from_config(config)
        _ = restored(x)

        assert config["units"] == 4
        assert config["grid_min"] == -1.0
        assert config["grid_max"] == 2.0
        assert config["num_grids"] == 12


# ============================================================================
# Numerical Stability Tests
# ============================================================================


class TestNumericalStability:
    """Test numerical stability edge cases."""

    @pytest.mark.parametrize(
        "LayerClass,kwargs",
        [
            (GaussianRBF, {"units": 2}),
            (ExponentialRBF, {"units": 2}),
            (MultiquadricRBF, {"units": 2}),
            (InverseMultiQuadricRBF, {"units": 2}),
            (InverseQuadricRBF, {"units": 2}),
            (CauchyRBF, {"units": 2}),
            (PowerRBF, {"units": 2}),
            (CubicRBF, {"units": 2}),
            (LinearRBF, {"units": 2}),
            (ThinPlateSplineRBF, {"units": 2}),
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
            (GaussianRBF, {"units": 2}),
            (ExponentialRBF, {"units": 2}),
            (MultiquadricRBF, {"units": 2}),
            (InverseMultiQuadricRBF, {"units": 2}),
            (InverseQuadricRBF, {"units": 2}),
            (CauchyRBF, {"units": 2}),
        ],
    )
    def test_no_nan_on_large_input(self, LayerClass, kwargs):
        """No NaN on large magnitude inputs."""
        layer = LayerClass(**kwargs)
        x = tf.constant([[100.0, -100.0], [1000.0, -1000.0]], dtype=tf.float32)
        output = layer(x)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_gradient_stability(self, sample_input):
        """Gradients are finite."""
        layer = GaussianRBF(units=2)
        with tf.GradientTape() as tape:
            output = layer(sample_input)
            loss = tf.reduce_mean(output**2)
        grads = tape.gradient(loss, layer.trainable_variables)
        for g in grads:
            if g is not None:
                assert not tf.reduce_any(tf.math.is_nan(g))
                assert not tf.reduce_any(tf.math.is_inf(g))
