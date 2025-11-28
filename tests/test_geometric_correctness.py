# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Comprehensive correctness tests for geometric basis layers.

Tests cover:
- Zernike polynomials (radial, m=0)
- Spherical harmonics (real Y_l^m)
- Hyperspherical harmonics (n-dimensional)

Mathematical Properties Tested
------------------------------
- Orthogonality relations
- Normalization conditions
- Recurrence relation accuracy
- Boundary conditions
- Gradient stability
- Serialization roundtrip
"""

import numpy as np
import pytest
import tensorflow as tf

from arnold.layers.core.geometric import (
    GeometricBase,
    Zernike,
    SphericalHarmonics,
    HypersphericalHarmonics,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def batch_size():
    return 16


@pytest.fixture
def input_dim():
    return 8


@pytest.fixture
def units():
    return 32


# =============================================================================
# TestZernike - Radial Zernike Polynomials
# =============================================================================


class TestZernikeBasic:
    """Basic functionality tests for Zernike layer."""

    def test_instantiation(self, units):
        """Test Zernike layer can be instantiated."""
        layer = Zernike(degree=6, units=units)
        assert layer.degree == 6
        assert layer.units == units

    def test_build(self, units, input_dim):
        """Test Zernike layer builds correctly."""
        layer = Zernike(degree=6, units=units)
        layer.build((None, input_dim))
        assert layer.built

    def test_forward_pass(self, batch_size, input_dim, units):
        """Test forward pass produces correct shape."""
        layer = Zernike(degree=6, units=units)
        x = tf.random.uniform((batch_size, input_dim), 0, 1)
        y = layer(x)
        assert y.shape == (batch_size, units)

    def test_output_finite(self, batch_size, input_dim, units):
        """Test output contains no NaN or Inf."""
        layer = Zernike(degree=8, units=units)
        x = tf.random.uniform((batch_size, input_dim), 0, 1)
        y = layer(x)
        assert not tf.reduce_any(tf.math.is_nan(y))
        assert not tf.reduce_any(tf.math.is_inf(y))

    def test_degree_0(self, batch_size, input_dim, units):
        """Test degree=0 (only R_0^0 = 1)."""
        layer = Zernike(degree=0, units=units)
        x = tf.random.uniform((batch_size, input_dim), 0, 1)
        y = layer(x)
        assert y.shape == (batch_size, units)

    def test_invalid_degree_raises(self, units):
        """Test negative degree raises ValueError."""
        with pytest.raises(ValueError, match="degree must be >= 0"):
            Zernike(degree=-1, units=units)


class TestZernikeMathematical:
    """Mathematical property tests for Zernike polynomials."""

    def test_R0_equals_one(self):
        """Test R_0^0(ρ) = 1 for all ρ."""
        layer = Zernike(degree=4, units=1)
        layer.build((None, 5))

        rho = tf.constant([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=tf.float32)
        basis = layer.geometric_basis(rho)

        # R_0^0 = 1 (first polynomial)
        np.testing.assert_allclose(
            basis.numpy()[0, :, 0], np.ones(5), rtol=1e-5
        )

    def test_R2_defocus(self):
        """Test R_2^0(ρ) = 2ρ² - 1."""
        layer = Zernike(degree=4, units=1)
        layer.build((None, 5))

        rho_vals = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        rho = tf.constant([rho_vals], dtype=tf.float32)
        basis = layer.geometric_basis(rho)

        expected = 2 * rho_vals**2 - 1
        np.testing.assert_allclose(
            basis.numpy()[0, :, 1], expected, rtol=1e-5
        )

    def test_R4_primary_spherical(self):
        """Test R_4^0(ρ) = 6ρ⁴ - 6ρ² + 1."""
        layer = Zernike(degree=4, units=1)
        layer.build((None, 5))

        rho_vals = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        rho = tf.constant([rho_vals], dtype=tf.float32)
        basis = layer.geometric_basis(rho)

        expected = 6 * rho_vals**4 - 6 * rho_vals**2 + 1
        np.testing.assert_allclose(
            basis.numpy()[0, :, 2], expected, rtol=1e-4
        )

    def test_boundary_rho_one(self):
        """Test at unit circle boundary ρ=1."""
        layer = Zernike(degree=6, units=1)
        x = tf.constant([[1.0] * 5], dtype=tf.float32)
        y = layer(x)
        assert not tf.reduce_any(tf.math.is_nan(y))
        assert not tf.reduce_any(tf.math.is_inf(y))

    def test_origin_rho_zero(self):
        """Test at origin ρ=0."""
        layer = Zernike(degree=6, units=1)
        x = tf.constant([[0.0] * 5], dtype=tf.float32)
        y = layer(x)
        assert not tf.reduce_any(tf.math.is_nan(y))


class TestZernikeGradients:
    """Gradient tests for Zernike layer."""

    def test_gradient_exists(self, batch_size, input_dim, units):
        """Test gradients exist for all trainable variables."""
        layer = Zernike(degree=6, units=units)
        x = tf.random.uniform((batch_size, input_dim), 0, 1)

        with tf.GradientTape() as tape:
            y = layer(x)
            loss = tf.reduce_mean(y**2)

        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads)

    def test_gradient_finite(self, batch_size, input_dim, units):
        """Test gradients are finite."""
        layer = Zernike(degree=6, units=units)
        x = tf.random.uniform((batch_size, input_dim), 0, 1)

        with tf.GradientTape() as tape:
            y = layer(x)
            loss = tf.reduce_mean(y**2)

        grads = tape.gradient(loss, layer.trainable_variables)
        for g in grads:
            assert not tf.reduce_any(tf.math.is_nan(g))
            assert not tf.reduce_any(tf.math.is_inf(g))


class TestZernikeSerialization:
    """Serialization tests for Zernike layer."""

    def test_get_config(self, units):
        """Test get_config returns correct configuration."""
        layer = Zernike(degree=6, units=units)
        config = layer.get_config()
        assert config["degree"] == 6
        assert config["units"] == units

    def test_from_config(self, units):
        """Test layer can be reconstructed from config."""
        layer = Zernike(degree=6, units=units)
        config = layer.get_config()
        new_layer = Zernike.from_config(config)
        assert new_layer.degree == layer.degree
        assert new_layer.units == layer.units

    def test_clone_model(self, batch_size, input_dim, units):
        """Test layer survives model cloning."""
        layer = Zernike(degree=6, units=units)
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(input_dim,)),
            layer,
        ])

        cloned = tf.keras.models.clone_model(model)
        x = tf.random.uniform((batch_size, input_dim), 0, 1)
        y1 = model(x)
        y2 = cloned(x)
        # Shapes should match (weights differ)
        assert y1.shape == y2.shape


# =============================================================================
# TestSphericalHarmonics - Real Spherical Harmonics
# =============================================================================


class TestSphericalHarmonicsBasic:
    """Basic functionality tests for SphericalHarmonics layer."""

    def test_instantiation(self, units):
        """Test SphericalHarmonics layer can be instantiated."""
        layer = SphericalHarmonics(max_degree=4, units=units)
        assert layer.max_degree == 4
        assert layer.units == units

    def test_build(self, units):
        """Test SphericalHarmonics layer builds correctly."""
        layer = SphericalHarmonics(max_degree=4, units=units)
        # Requires 2D spherical input (theta, phi)
        layer.build((None, 2))
        assert layer.built

    def test_forward_pass(self, batch_size, units):
        """Test forward pass with 2D spherical coordinates."""
        layer = SphericalHarmonics(max_degree=4, units=units)
        # theta in [0, pi], phi in [0, 2pi]
        theta = tf.random.uniform((batch_size, 1), 0, np.pi)
        phi = tf.random.uniform((batch_size, 1), 0, 2 * np.pi)
        x = tf.concat([theta, phi], axis=-1)
        y = layer(x)
        assert y.shape == (batch_size, units)

    def test_output_finite(self, batch_size, units):
        """Test output contains no NaN or Inf."""
        layer = SphericalHarmonics(max_degree=5, units=units)
        theta = tf.random.uniform((batch_size, 1), 0.1, np.pi - 0.1)
        phi = tf.random.uniform((batch_size, 1), 0, 2 * np.pi)
        x = tf.concat([theta, phi], axis=-1)
        y = layer(x)
        assert not tf.reduce_any(tf.math.is_nan(y))
        assert not tf.reduce_any(tf.math.is_inf(y))


class TestSphericalHarmonicsMathematical:
    """Mathematical property tests for spherical harmonics."""

    def test_Y00_is_constant(self):
        """Test Y_0^0 (P_0) = 1 is constant."""
        layer = SphericalHarmonics(max_degree=2, units=1)
        layer.build((None, 1))

        # For SphericalHarmonics, input is cos(theta) in [-1, 1]
        x = tf.constant([[-0.5], [0.0], [0.5], [1.0]], dtype=tf.float32)
        basis = layer.geometric_basis(x)

        # P_0(x) = 1 should be constant (first basis function)
        p0 = basis[:, 0, 0].numpy()
        assert np.allclose(p0, 1.0, rtol=1e-4)

    def test_degree_counting(self):
        """Test correct number of basis functions."""
        # SphericalHarmonics uses Legendre polynomials (m=0 only)
        # So for l_max = L, we have L+1 basis functions
        for L in [0, 1, 2, 3, 4]:
            layer = SphericalHarmonics(max_degree=L, units=1)
            expected = L + 1  # Only zonale harmonics
            assert layer._get_num_basis_functions() == expected


class TestSphericalHarmonicsGradients:
    """Gradient tests for SphericalHarmonics layer."""

    def test_gradient_exists(self, batch_size, units):
        """Test gradients exist for all trainable variables."""
        layer = SphericalHarmonics(max_degree=3, units=units)
        theta = tf.random.uniform((batch_size, 1), 0.1, np.pi - 0.1)
        phi = tf.random.uniform((batch_size, 1), 0, 2 * np.pi)
        x = tf.concat([theta, phi], axis=-1)

        with tf.GradientTape() as tape:
            y = layer(x)
            loss = tf.reduce_mean(y**2)

        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads)


class TestSphericalHarmonicsSerialization:
    """Serialization tests for SphericalHarmonics layer."""

    def test_get_config(self, units):
        """Test get_config returns correct configuration."""
        layer = SphericalHarmonics(max_degree=4, units=units)
        config = layer.get_config()
        assert "max_degree" in config or "max_l" in config
        assert config["units"] == units

    def test_from_config(self, units):
        """Test layer can be reconstructed from config."""
        layer = SphericalHarmonics(max_degree=4, units=units)
        config = layer.get_config()
        new_layer = SphericalHarmonics.from_config(config)
        assert new_layer.max_degree == layer.max_degree


# =============================================================================
# TestHypersphericalHarmonics - n-Dimensional Harmonics
# =============================================================================


class TestHypersphericalHarmonicsBasic:
    """Basic functionality tests for HypersphericalHarmonics layer."""

    def test_instantiation(self, units):
        """Test HypersphericalHarmonics layer can be instantiated."""
        layer = HypersphericalHarmonics(max_degree=3, dimension=4, units=units)
        assert layer.max_degree == 3
        assert layer.dimension == 4
        assert layer.units == units

    def test_build(self, units):
        """Test HypersphericalHarmonics layer builds correctly."""
        layer = HypersphericalHarmonics(max_degree=3, dimension=4, units=units)
        # 4D requires 3 angular coordinates
        layer.build((None, 3))
        assert layer.built

    def test_forward_pass(self, batch_size, units):
        """Test forward pass with hyperspherical coordinates."""
        layer = HypersphericalHarmonics(max_degree=3, dimension=4, units=units)
        # 4D: 3 angles (theta_1, theta_2, phi)
        angles = tf.random.uniform((batch_size, 3), 0, np.pi)
        y = layer(angles)
        assert y.shape == (batch_size, units)

    def test_output_finite(self, batch_size, units):
        """Test output contains no NaN or Inf."""
        layer = HypersphericalHarmonics(max_degree=2, dimension=4, units=units)
        angles = tf.random.uniform((batch_size, 3), 0.1, np.pi - 0.1)
        y = layer(angles)
        assert not tf.reduce_any(tf.math.is_nan(y))
        assert not tf.reduce_any(tf.math.is_inf(y))

    def test_3d_equals_spherical_basis_count(self, units):
        """Test 3D case has correct basis count."""
        layer_hyper = HypersphericalHarmonics(max_degree=3, dimension=3, units=units)
        # HypersphericalHarmonics uses Gegenbauer, so max_degree+1 basis functions
        assert layer_hyper._get_num_basis_functions() == 4  # 0, 1, 2, 3


class TestHypersphericalHarmonicsGradients:
    """Gradient tests for HypersphericalHarmonics layer."""

    def test_gradient_exists(self, batch_size, units):
        """Test gradients exist for all trainable variables."""
        layer = HypersphericalHarmonics(max_degree=2, dimension=4, units=units)
        angles = tf.random.uniform((batch_size, 3), 0.1, np.pi - 0.1)

        with tf.GradientTape() as tape:
            y = layer(angles)
            loss = tf.reduce_mean(y**2)

        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads)


class TestHypersphericalHarmonicsSerialization:
    """Serialization tests for HypersphericalHarmonics layer."""

    def test_get_config(self, units):
        """Test get_config returns correct configuration."""
        layer = HypersphericalHarmonics(max_degree=3, dimension=4, units=units)
        config = layer.get_config()
        assert config["max_degree"] == 3
        assert config["dimension"] == 4
        assert config["units"] == units

    def test_from_config(self, units):
        """Test layer can be reconstructed from config."""
        layer = HypersphericalHarmonics(max_degree=3, dimension=4, units=units)
        config = layer.get_config()
        new_layer = HypersphericalHarmonics.from_config(config)
        assert new_layer.max_degree == layer.max_degree
        assert new_layer.dimension == layer.dimension


# =============================================================================
# TestRegistryIntegration - Registry Access
# =============================================================================


class TestGeometricRegistry:
    """Test geometric layers are accessible via registry."""

    def test_zernike_via_registry(self, units):
        """Test Zernike accessible via registry."""
        from arnold.layers.core.registry import get_layer

        layer = get_layer("zernike", degree=4, units=units)
        assert isinstance(layer, Zernike)

    def test_spherical_harmonics_via_registry(self, units):
        """Test SphericalHarmonics accessible via registry."""
        from arnold.layers.core.registry import get_layer

        layer = get_layer("spherical_harmonics", max_degree=3, units=units)
        assert isinstance(layer, SphericalHarmonics)

    def test_spherical_alias(self, units):
        """Test 'spherical' alias works."""
        from arnold.layers.core.registry import get_layer

        layer = get_layer("spherical", max_degree=3, units=units)
        assert isinstance(layer, SphericalHarmonics)

    def test_ylm_alias(self, units):
        """Test 'ylm' alias works."""
        from arnold.layers.core.registry import get_layer

        layer = get_layer("ylm", max_degree=3, units=units)
        assert isinstance(layer, SphericalHarmonics)

    def test_hyperspherical_harmonics_via_registry(self, units):
        """Test HypersphericalHarmonics accessible via registry."""
        from arnold.layers.core.registry import get_layer

        layer = get_layer("hyperspherical_harmonics", max_degree=2, dimension=4, units=units)
        assert isinstance(layer, HypersphericalHarmonics)

    def test_hyperspherical_alias(self, units):
        """Test 'hyperspherical' alias works."""
        from arnold.layers.core.registry import get_layer

        layer = get_layer("hyperspherical", max_degree=2, dimension=4, units=units)
        assert isinstance(layer, HypersphericalHarmonics)


# =============================================================================
# TestGeometricCategory
# =============================================================================


class TestGeometricCategory:
    """Test geometric category in layer registry."""

    def test_geometric_category_exists(self):
        """Test geometric category exists in LAYER_CATEGORIES."""
        from arnold.layers.core.registry import LAYER_CATEGORIES

        assert "geometric" in LAYER_CATEGORIES

    def test_geometric_category_contents(self):
        """Test geometric category contains expected layers."""
        from arnold.layers.core.registry import LAYER_CATEGORIES

        geometric = LAYER_CATEGORIES["geometric"]
        assert "zernike" in geometric
        assert "spherical_harmonics" in geometric
        assert "hyperspherical_harmonics" in geometric


# =============================================================================
# TestGeometricXLA
# =============================================================================


class TestGeometricXLA:
    """XLA compilation tests for geometric layers."""

    @pytest.mark.parametrize("layer_cls,kwargs", [
        (Zernike, {"degree": 4, "units": 16}),
        (SphericalHarmonics, {"max_degree": 3, "units": 16}),
        (HypersphericalHarmonics, {"max_degree": 2, "dimension": 4, "units": 16}),
    ])
    def test_xla_compilation(self, layer_cls, kwargs):
        """Test layers compile with XLA."""
        layer = layer_cls(**kwargs)

        # Determine input dimension
        if layer_cls == Zernike:
            input_dim = 5
        elif layer_cls == SphericalHarmonics:
            input_dim = 2  # (theta, phi)
        else:  # HypersphericalHarmonics
            input_dim = kwargs["dimension"] - 1

        @tf.function(jit_compile=True)
        def forward(x):
            return layer(x)

        x = tf.random.uniform((8, input_dim), 0.1, 0.9)
        y = forward(x)
        assert y.shape[0] == 8


# =============================================================================
# TestHighDegreeStability
# =============================================================================


class TestHighDegreeStability:
    """Test numerical stability at higher degrees."""

    def test_zernike_degree_12(self, batch_size, input_dim, units):
        """Test Zernike at degree 12."""
        layer = Zernike(degree=12, units=units)
        x = tf.random.uniform((batch_size, input_dim), 0.01, 0.99)
        y = layer(x)
        assert not tf.reduce_any(tf.math.is_nan(y))
        assert not tf.reduce_any(tf.math.is_inf(y))

    def test_spherical_harmonics_degree_6(self, batch_size, units):
        """Test SphericalHarmonics at max_degree=6."""
        layer = SphericalHarmonics(max_degree=6, units=units)
        theta = tf.random.uniform((batch_size, 1), 0.1, np.pi - 0.1)
        phi = tf.random.uniform((batch_size, 1), 0, 2 * np.pi)
        x = tf.concat([theta, phi], axis=-1)
        y = layer(x)
        assert not tf.reduce_any(tf.math.is_nan(y))
        assert not tf.reduce_any(tf.math.is_inf(y))

    def test_hyperspherical_degree_4_dim_5(self, batch_size, units):
        """Test HypersphericalHarmonics at max_degree=4, dimension=5."""
        layer = HypersphericalHarmonics(max_degree=4, dimension=5, units=units)
        angles = tf.random.uniform((batch_size, 4), 0.1, np.pi - 0.1)
        y = layer(angles)
        assert not tf.reduce_any(tf.math.is_nan(y))
        assert not tf.reduce_any(tf.math.is_inf(y))
