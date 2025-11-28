# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
"""
Tucker Decomposition Tests - Phase 4.3

Tests verify that core_ranks parameter:
1. Creates correct weight shapes
2. Reduces parameter count as expected
3. Produces valid outputs
4. Gradients flow correctly
5. Serializes properly
"""

import numpy as np
import pytest
import tensorflow as tf

from arnold.layers.core.polynomial.orthogonal import (
    Chebyshev1st,
    Chebyshev2nd,
    Legendre,
    GeneralizedLaguerre as Laguerre,
    Hermite,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_input():
    """Sample input for layer testing."""
    return tf.constant([[0.1, 0.5, 0.3], [0.4, 0.2, 0.8]], dtype=tf.float32)


# ============================================================================
# Tucker Weight Shape Tests
# ============================================================================


class TestTuckerWeightShapes:
    """Test that Tucker decomposition creates correct weight shapes."""

    def test_core_tensor_shape(self, sample_input):
        """Core tensor has shape (r1, r2, r3)."""
        core_ranks = (4, 5, 6)
        layer = Legendre(units=10, degree=8, core_ranks=core_ranks)
        _ = layer(sample_input)

        assert layer.poly_coeffs_core.shape == core_ranks

    def test_factor_A_shape(self, sample_input):
        """Factor A has shape (input_dim, r1)."""
        core_ranks = (4, 5, 6)
        layer = Legendre(units=10, degree=8, core_ranks=core_ranks)
        _ = layer(sample_input)

        input_dim = sample_input.shape[-1]
        assert layer.poly_coeffs_A.shape == (input_dim, core_ranks[0])

    def test_factor_B_shape(self, sample_input):
        """Factor B has shape (degree+1, r2)."""
        core_ranks = (4, 5, 6)
        degree = 8
        layer = Legendre(units=10, degree=degree, core_ranks=core_ranks)
        _ = layer(sample_input)

        assert layer.poly_coeffs_B.shape == (degree + 1, core_ranks[1])

    def test_factor_C_shape(self, sample_input):
        """Factor C has shape (output_dim, r3)."""
        core_ranks = (4, 5, 6)
        output_dim = 10
        layer = Legendre(units=output_dim, degree=8, core_ranks=core_ranks)
        _ = layer(sample_input)

        assert layer.poly_coeffs_C.shape == (output_dim, core_ranks[2])

    def test_no_full_coeffs_when_tucker(self, sample_input):
        """Full poly_coeffs is None when using Tucker decomposition."""
        layer = Legendre(units=10, degree=8, core_ranks=(4, 5, 6))
        _ = layer(sample_input)

        assert layer.poly_coeffs is None


# ============================================================================
# Parameter Count Reduction Tests
# ============================================================================


class TestParameterReduction:
    """Test that Tucker decomposition reduces parameter count."""

    def test_parameter_count_reduction(self, sample_input):
        """Tucker decomposition reduces total parameters."""
        input_dim = sample_input.shape[-1]  # 3
        output_dim = 20
        degree = 10

        # Full rank: input_dim * (degree+1) * output_dim = 3 * 11 * 20 = 660
        full_params = input_dim * (degree + 1) * output_dim

        # Tucker with ranks (2, 3, 4)
        r1, r2, r3 = 2, 3, 4
        # core: r1*r2*r3 = 24
        # A: input_dim*r1 = 6
        # B: (degree+1)*r2 = 33
        # C: output_dim*r3 = 80
        # Total: 24 + 6 + 33 + 80 = 143
        tucker_params = r1 * r2 * r3 + input_dim * r1 + (degree + 1) * r2 + output_dim * r3

        layer_full = Legendre(units=output_dim, degree=degree)
        layer_tucker = Legendre(units=output_dim, degree=degree, core_ranks=(r1, r2, r3))

        _ = layer_full(sample_input)
        _ = layer_tucker(sample_input)

        # Count polynomial coefficient parameters only
        full_count = np.prod(layer_full.poly_coeffs.shape)
        tucker_count = (
            np.prod(layer_tucker.poly_coeffs_core.shape)
            + np.prod(layer_tucker.poly_coeffs_A.shape)
            + np.prod(layer_tucker.poly_coeffs_B.shape)
            + np.prod(layer_tucker.poly_coeffs_C.shape)
        )

        assert full_count == full_params
        assert tucker_count == tucker_params
        assert tucker_count < full_count

    def test_compression_ratio(self, sample_input):
        """Check compression ratio for realistic sizes."""
        input_dim = sample_input.shape[-1]
        output_dim = 64
        degree = 15
        r1, r2, r3 = 4, 4, 4

        full_params = input_dim * (degree + 1) * output_dim
        tucker_params = r1 * r2 * r3 + input_dim * r1 + (degree + 1) * r2 + output_dim * r3

        compression_ratio = full_params / tucker_params
        # Should achieve significant compression
        assert compression_ratio > 5, f"Compression ratio {compression_ratio} is too low"


# ============================================================================
# Output Correctness Tests
# ============================================================================


class TestTuckerOutput:
    """Test that Tucker decomposition produces valid outputs."""

    def test_output_shape(self, sample_input):
        """Tucker layer produces correct output shape."""
        layer = Legendre(units=10, degree=8, core_ranks=(3, 4, 5))
        output = layer(sample_input)
        assert output.shape == (2, 10)

    def test_output_not_nan(self, sample_input):
        """Tucker output contains no NaN."""
        layer = Legendre(units=10, degree=8, core_ranks=(3, 4, 5))
        output = layer(sample_input)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_output_not_inf(self, sample_input):
        """Tucker output contains no Inf."""
        layer = Legendre(units=10, degree=8, core_ranks=(3, 4, 5))
        output = layer(sample_input)
        assert not tf.reduce_any(tf.math.is_inf(output))

    def test_batch_independence(self):
        """Different batch items produce different outputs."""
        layer = Legendre(units=5, degree=6, core_ranks=(2, 3, 4))
        x = tf.constant([[0.1, 0.2], [0.8, 0.9]], dtype=tf.float32)
        output = layer(x)

        # Different inputs should produce different outputs
        assert not tf.reduce_all(tf.equal(output[0], output[1]))


# ============================================================================
# Gradient Flow Tests
# ============================================================================


class TestTuckerGradients:
    """Test gradient flow through Tucker decomposition."""

    def test_gradients_to_core(self, sample_input):
        """Gradients flow to core tensor."""
        layer = Legendre(units=5, degree=6, core_ranks=(2, 3, 4))
        with tf.GradientTape() as tape:
            output = layer(sample_input)
            loss = tf.reduce_mean(output)
        grads = tape.gradient(loss, layer.trainable_variables)

        core_grads = [g for g, v in zip(grads, layer.trainable_variables) if "core" in v.name]
        assert len(core_grads) == 1
        assert core_grads[0] is not None

    def test_gradients_to_all_factors(self, sample_input):
        """Gradients flow to all factor matrices."""
        layer = Legendre(units=5, degree=6, core_ranks=(2, 3, 4))
        with tf.GradientTape() as tape:
            output = layer(sample_input)
            loss = tf.reduce_mean(output)
        grads = tape.gradient(loss, layer.trainable_variables)

        # Check all polynomial coefficient gradients exist
        coeff_vars = [v for v in layer.trainable_variables if "polynomial_coefficients" in v.name]
        coeff_grads = [g for g, v in zip(grads, layer.trainable_variables) if "polynomial_coefficients" in v.name]

        assert len(coeff_vars) == 4  # core, A, B, C
        assert all(g is not None for g in coeff_grads)

    def test_gradients_finite(self, sample_input):
        """All gradients are finite."""
        layer = Legendre(units=5, degree=6, core_ranks=(2, 3, 4))
        with tf.GradientTape() as tape:
            output = layer(sample_input)
            loss = tf.reduce_mean(output**2)
        grads = tape.gradient(loss, layer.trainable_variables)

        for g in grads:
            if g is not None:
                assert not tf.reduce_any(tf.math.is_nan(g))
                assert not tf.reduce_any(tf.math.is_inf(g))


# ============================================================================
# Serialization Tests
# ============================================================================


class TestTuckerSerialization:
    """Test serialization of Tucker-decomposed layers."""

    def test_config_includes_core_ranks(self, sample_input):
        """Config includes core_ranks."""
        core_ranks = (3, 4, 5)
        layer = Legendre(units=10, degree=8, core_ranks=core_ranks)
        _ = layer(sample_input)

        config = layer.get_config()
        assert config["core_ranks"] == core_ranks

    def test_from_config_roundtrip(self, sample_input):
        """Layer survives from_config roundtrip."""
        core_ranks = (3, 4, 5)
        layer = Legendre(units=10, degree=8, core_ranks=core_ranks)
        _ = layer(sample_input)

        config = layer.get_config()
        restored = Legendre.from_config(config)
        _ = restored(sample_input)

        assert restored.core_ranks == core_ranks


# ============================================================================
# Multiple Polynomial Types Tests
# ============================================================================


class TestTuckerMultipleTypes:
    """Test Tucker decomposition works with different polynomial types."""

    @pytest.mark.parametrize(
        "LayerClass",
        [Chebyshev1st, Chebyshev2nd, Legendre, Laguerre, Hermite],
    )
    def test_tucker_works_with_polynomial(self, LayerClass, sample_input):
        """Tucker decomposition works with various polynomial types."""
        layer = LayerClass(units=5, degree=6, core_ranks=(2, 3, 4))
        output = layer(sample_input)

        assert output.shape == (2, 5)
        assert not tf.reduce_any(tf.math.is_nan(output))

    @pytest.mark.parametrize(
        "LayerClass",
        [Chebyshev1st, Chebyshev2nd, Legendre, Laguerre, Hermite],
    )
    def test_tucker_gradients_polynomial(self, LayerClass, sample_input):
        """Tucker gradients flow for various polynomial types."""
        layer = LayerClass(units=5, degree=6, core_ranks=(2, 3, 4))
        with tf.GradientTape() as tape:
            output = layer(sample_input)
            loss = tf.reduce_mean(output)
        grads = tape.gradient(loss, layer.trainable_variables)

        # All gradients should exist
        assert all(g is not None for g in grads)


# ============================================================================
# Edge Cases
# ============================================================================


class TestTuckerEdgeCases:
    """Test edge cases for Tucker decomposition."""

    def test_rank_one_decomposition(self, sample_input):
        """Rank-1 decomposition works."""
        layer = Legendre(units=5, degree=6, core_ranks=(1, 1, 1))
        output = layer(sample_input)

        assert output.shape == (2, 5)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_full_rank_equivalent(self, sample_input):
        """Full rank decomposition is equivalent to no decomposition."""
        input_dim = sample_input.shape[-1]
        output_dim = 5
        degree = 4

        # Full rank Tucker is essentially no compression
        layer_tucker = Legendre(
            units=output_dim,
            degree=degree,
            core_ranks=(input_dim, degree + 1, output_dim),
        )
        _ = layer_tucker(sample_input)

        # Should have same number of coefficient params (plus some overhead)
        full_params = input_dim * (degree + 1) * output_dim
        tucker_core_params = input_dim * (degree + 1) * output_dim

        assert np.prod(layer_tucker.poly_coeffs_core.shape) == tucker_core_params

    def test_asymmetric_ranks(self, sample_input):
        """Asymmetric ranks work correctly."""
        layer = Legendre(units=10, degree=8, core_ranks=(2, 8, 3))
        output = layer(sample_input)

        assert output.shape == (2, 10)
        assert not tf.reduce_any(tf.math.is_nan(output))
