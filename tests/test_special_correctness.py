# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Correctness tests for special function layers.

Tests the mathematical correctness of special function implementations:
- Airy functions Ai(x), Bi(x)
- Bessel functions J_ν(x)

Note: Most special function layers are stubs with NotImplementedError.
These tests will expand as implementations are added.
"""

import pytest
import tensorflow as tf

from arnold.layers.core.special import (
    Airy,
    Bessel,
    EllipticFunctions,
    LegendreFunctions,
    Mathieu,
    ParabolicCylinder,
    Slepian,
    SpecialBase,
    Whittaker,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_input():
    """Generate sample input tensor for testing."""
    return tf.constant([[0.0, 0.5, 1.0, -0.5, -1.0]], dtype=tf.float64)


@pytest.fixture
def batch_input():
    """Generate batch input tensor for testing."""
    return tf.constant([
        [0.0, 0.5, 1.0],
        [0.1, 0.6, 1.1],
        [-0.5, -0.3, 0.2],
    ], dtype=tf.float64)


# =============================================================================
# Airy Function Tests
# =============================================================================

class TestAiry:
    """Tests for Airy function layer."""

    def test_airy_instantiation(self):
        """Test Airy layer can be instantiated."""
        layer = Airy(max_order=5, units=32)
        assert layer.max_order == 5
        assert layer.units == 32

    def test_airy_ai_at_zero(self, sample_input):
        """Test Ai(0) ≈ 0.35502805388781724."""
        layer = Airy(max_order=3, units=16)
        # Build the layer
        layer.build(sample_input.shape)

        # Get basis values at x=0
        x_zero = tf.constant([[0.0]], dtype=tf.float64)
        basis = layer.special_basis(x_zero)

        # First component should be Ai(0)
        ai_zero = float(basis[0, 0, 0])
        expected = 0.35502805388781724

        # Power series approximation should be close for small arguments
        assert abs(ai_zero - expected) < 0.01, f"Ai(0) = {ai_zero}, expected ≈ {expected}"

    def test_airy_output_shape(self, batch_input):
        """Test Airy layer output shape."""
        layer = Airy(max_order=4, units=32)
        layer.build(batch_input.shape)

        output = layer(batch_input)

        # Output shape: (batch, units)
        assert output.shape == (3, 32)

    def test_airy_differentiable(self, sample_input):
        """Test Airy layer is differentiable."""
        layer = Airy(max_order=3, units=16)
        layer.build(sample_input.shape)

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = tf.reduce_sum(output)

        grads = tape.gradient(loss, sample_input)
        assert grads is not None
        assert not tf.reduce_any(tf.math.is_nan(grads))


# =============================================================================
# Bessel Function Tests
# =============================================================================

class TestBessel:
    """Tests for Bessel function layer."""

    def test_bessel_instantiation(self):
        """Test Bessel layer can be instantiated."""
        layer = Bessel(max_order=5, units=32)
        assert layer.max_order == 5
        assert layer.units == 32

    def test_bessel_j0_at_zero(self, sample_input):
        """Test J_0(0) = 1."""
        layer = Bessel(max_order=3, units=16)
        layer.build(sample_input.shape)

        # Get basis values at x=0
        x_zero = tf.constant([[0.0]], dtype=tf.float64)
        basis = layer.special_basis(x_zero)

        # J_0(0) = 1
        j0_zero = float(basis[0, 0, 0])
        assert abs(j0_zero - 1.0) < 1e-10, f"J_0(0) = {j0_zero}, expected 1.0"

    def test_bessel_j1_at_zero(self, sample_input):
        """Test J_1(0) = 0."""
        layer = Bessel(max_order=3, units=16)
        layer.build(sample_input.shape)

        # Get basis values at x=0
        x_zero = tf.constant([[0.0]], dtype=tf.float64)
        basis = layer.special_basis(x_zero)

        # J_1(0) = 0 (second order in output)
        if basis.shape[-1] > 1:
            j1_zero = float(basis[0, 0, 1])
            assert abs(j1_zero) < 1e-10, f"J_1(0) = {j1_zero}, expected 0.0"

    def test_bessel_output_shape(self, batch_input):
        """Test Bessel layer output shape."""
        layer = Bessel(max_order=4, units=32)
        layer.build(batch_input.shape)

        output = layer(batch_input)

        # Output shape: (batch, units)
        assert output.shape == (3, 32)

    def test_bessel_differentiable(self, sample_input):
        """Test Bessel layer is differentiable."""
        layer = Bessel(max_order=3, units=16)
        layer.build(sample_input.shape)

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = tf.reduce_sum(output)

        grads = tape.gradient(loss, sample_input)
        assert grads is not None
        assert not tf.reduce_any(tf.math.is_nan(grads))


# =============================================================================
# Stub Layer Tests (verify NotImplementedError)
# =============================================================================

class TestStubLayers:
    """Tests for stub layers that are not yet fully implemented."""

    @pytest.mark.parametrize("LayerClass,name,kwargs", [
        (ParabolicCylinder, "ParabolicCylinder", {"max_order": 5, "units": 32}),
        (Mathieu, "Mathieu", {"max_order": 5, "units": 32}),
        (Whittaker, "Whittaker", {"max_order": 5, "units": 32}),
        (Slepian, "Slepian", {"max_order": 5, "units": 32, "bandwidth": 0.25}),
        (LegendreFunctions, "LegendreFunctions", {"max_order": 5, "units": 32}),
        (EllipticFunctions, "EllipticFunctions", {"max_order": 5, "units": 32}),
    ])
    def test_stub_instantiation(self, LayerClass, name, kwargs):
        """Test stub layers can be instantiated."""
        layer = LayerClass(**kwargs)
        assert layer.max_order == 5
        assert layer.units == 32

    @pytest.mark.parametrize("LayerClass,name,kwargs", [
        (Mathieu, "Mathieu", {"max_order": 5, "units": 32}),
        (Whittaker, "Whittaker", {"max_order": 5, "units": 32}),
        (Slepian, "Slepian", {"max_order": 5, "units": 32, "bandwidth": 0.25}),
        (LegendreFunctions, "LegendreFunctions", {"max_order": 5, "units": 32}),
        (EllipticFunctions, "EllipticFunctions", {"max_order": 5, "units": 32}),
    ])
    def test_stub_output_shape(self, LayerClass, name, kwargs, batch_input):
        """Test stub layers produce correct output shape."""
        layer = LayerClass(**kwargs)
        layer.build(batch_input.shape)
        output = layer(batch_input)
        assert output.shape == (3, 32)


# =============================================================================
# Registry Integration Tests
# =============================================================================

class TestRegistryIntegration:
    """Tests for special layer registry integration."""

    def test_airy_in_registry(self):
        """Test Airy layer is registered."""
        from arnold.layers.core.registry import get_layer, is_registered

        assert is_registered("airy")
        layer = get_layer("airy", max_order=3, units=16)
        assert isinstance(layer, Airy)

    def test_bessel_functions_in_registry(self):
        """Test Bessel functions layer is registered (as bessel_functions)."""
        from arnold.layers.core.registry import get_layer, is_registered

        # Note: "bessel" maps to Bessel polynomial, "bessel_functions" to special
        assert is_registered("bessel_functions")
        layer = get_layer("bessel_functions", max_order=3, units=16)
        assert isinstance(layer, Bessel)

    def test_special_category_exists(self):
        """Test special category exists in registry."""
        from arnold.layers.core.registry import LAYER_CATEGORIES

        assert "special" in LAYER_CATEGORIES
        assert "airy" in LAYER_CATEGORIES["special"]
        assert "bessel_functions" in LAYER_CATEGORIES["special"]


# =============================================================================
# Base Class Tests
# =============================================================================

class TestSpecialBase:
    """Tests for SpecialBase abstract class."""

    def test_special_base_is_abstract(self):
        """Test SpecialBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SpecialBase(max_order=5, units=32)

    def test_concrete_layer_has_required_methods(self):
        """Test concrete layers have required methods from SpecialBase."""
        layer = Airy(max_order=5, units=32)

        assert hasattr(layer, "max_order")
        assert hasattr(layer, "special_basis")
        assert callable(layer.special_basis)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
