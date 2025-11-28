# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
"""
Numerical Precision Regression Tests

These tests catch silent precision degradation by:
1. Testing against known reference values (golden tests)
2. Verifying orthogonality properties hold within tolerance
3. Checking that high-degree evaluations don't silently overflow/underflow
4. Ensuring dtype promotions work correctly

Run with: pytest tests/test_precision_regression.py -v
"""

import numpy as np
import pytest
import tensorflow as tf

from arnold.layers import (
    Chebyshev1st,
    Gegenbauer,
    GaussianRBF,
    Hermite,
    Jacobi,
    Legendre,
    Ricker,
)


class TestPolynomialGoldenValues:
    """Test polynomial layers against known reference values."""

    def test_legendre_at_special_points(self):
        """Legendre polynomials have known values at x=0, x=1, x=-1."""
        layer = Legendre(degree=5, units=1, input_clip=(-1.0, 1.0))
        layer.build((None, 1))
        
        # At x=1: P_n(1) = 1 for all n
        x_one = tf.constant([[1.0]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x_one)
        expected_at_one = tf.ones((1, 1, 6), dtype=tf.float32)
        np.testing.assert_allclose(basis.numpy(), expected_at_one.numpy(), rtol=1e-5)
        
        # At x=-1: P_n(-1) = (-1)^n
        x_minus_one = tf.constant([[-1.0]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x_minus_one)
        expected_at_minus_one = np.array([[[1, -1, 1, -1, 1, -1]]], dtype=np.float32)
        np.testing.assert_allclose(basis.numpy(), expected_at_minus_one, rtol=1e-5)
        
        # At x=0: P_n(0) = 0 for odd n, known values for even n
        x_zero = tf.constant([[0.0]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x_zero)
        # P_0(0)=1, P_1(0)=0, P_2(0)=-1/2, P_3(0)=0, P_4(0)=3/8, P_5(0)=0
        expected_at_zero = np.array([[[1.0, 0.0, -0.5, 0.0, 0.375, 0.0]]], dtype=np.float32)
        np.testing.assert_allclose(basis.numpy(), expected_at_zero, rtol=1e-5)

    def test_chebyshev_at_special_points(self):
        """Chebyshev T_n polynomials: T_n(1)=1, T_n(-1)=(-1)^n, T_n(0)=cos(nπ/2)."""
        layer = Chebyshev1st(degree=4, units=1, input_clip=(-1.0, 1.0))
        layer.build((None, 1))
        
        # At x=1: T_n(1) = 1
        x_one = tf.constant([[1.0]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x_one)
        expected = np.ones((1, 1, 5), dtype=np.float32)
        np.testing.assert_allclose(basis.numpy(), expected, rtol=1e-5)
        
        # At x=0: T_0(0)=1, T_1(0)=0, T_2(0)=-1, T_3(0)=0, T_4(0)=1
        x_zero = tf.constant([[0.0]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x_zero)
        expected = np.array([[[1.0, 0.0, -1.0, 0.0, 1.0]]], dtype=np.float32)
        # Use atol for near-zero values
        np.testing.assert_allclose(basis.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_hermite_probabilist_normalization(self):
        """Probabilist's Hermite: He_n(0) matches known values."""
        layer = Hermite(degree=4, units=1, normalized=True, input_clip=(-5.0, 5.0))
        layer.build((None, 1))
        
        # He_0(0)=1, He_1(0)=0, He_2(0)=-1, He_3(0)=0, He_4(0)=3
        x_zero = tf.constant([[0.0]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x_zero)
        expected = np.array([[[1.0, 0.0, -1.0, 0.0, 3.0]]], dtype=np.float32)
        np.testing.assert_allclose(basis.numpy(), expected, rtol=1e-4)


class TestOrthogonalityProperties:
    """Verify orthogonality properties hold within numerical tolerance."""

    def test_legendre_orthogonality_discrete(self):
        """Legendre polynomials are approximately orthogonal under discrete sampling."""
        layer = Legendre(degree=5, units=1, orthonormal=True, input_clip=(-1.0, 1.0))
        layer.build((None, 1))
        
        # Sample on Gauss-Legendre nodes for better orthogonality
        n_samples = 100
        x = tf.constant(np.linspace(-0.99, 0.99, n_samples).reshape(-1, 1), dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)  # (n_samples, 1, degree+1)
        basis = tf.squeeze(basis, axis=1)  # (n_samples, degree+1)
        
        # Compute Gram matrix: should be approximately diagonal for orthonormal basis
        gram = tf.matmul(basis, basis, transpose_a=True) / n_samples
        
        # Off-diagonal elements should be small
        identity = tf.eye(6, dtype=tf.float32)
        off_diag_error = tf.reduce_max(tf.abs(gram - identity * tf.linalg.diag_part(gram)))
        assert off_diag_error < 0.1, f"Off-diagonal error too large: {off_diag_error}"

    def test_chebyshev_orthogonality_weighted(self):
        """Chebyshev T polynomials orthogonal w.r.t. weight 1/sqrt(1-x^2)."""
        layer = Chebyshev1st(degree=4, units=1, orthonormal=True, input_clip=(-1.0, 1.0))
        layer.build((None, 1))
        
        # Use Chebyshev nodes for optimal sampling
        n_samples = 100
        k = np.arange(1, n_samples + 1)
        x_nodes = np.cos((2 * k - 1) * np.pi / (2 * n_samples))
        x = tf.constant(x_nodes.reshape(-1, 1), dtype=tf.float32)
        
        basis = layer.pseudo_vandermonde(x)
        basis = tf.squeeze(basis, axis=1)
        
        # At Chebyshev nodes, T_m and T_n are orthogonal
        gram = tf.matmul(basis, basis, transpose_a=True) / n_samples
        
        # Check approximate diagonality
        diag = tf.linalg.diag_part(gram)
        off_diag = gram - tf.linalg.diag(diag)
        max_off_diag = tf.reduce_max(tf.abs(off_diag))
        assert max_off_diag < 0.15, f"Off-diagonal too large: {max_off_diag}"


class TestHighDegreeStability:
    """Ensure high-degree evaluations don't silently fail."""

    def test_legendre_degree_50_no_nan(self):
        """Legendre at degree 50 should not produce NaN."""
        layer = Legendre(degree=50, units=1, input_clip=(-1.0, 1.0))
        layer.build((None, 1))
        
        x = tf.constant(np.linspace(-0.9, 0.9, 20).reshape(-1, 1), dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)
        
        assert not tf.reduce_any(tf.math.is_nan(basis)), "NaN found in degree-50 Legendre"
        assert not tf.reduce_any(tf.math.is_inf(basis)), "Inf found in degree-50 Legendre"

    def test_chebyshev_degree_50_bounded(self):
        """Chebyshev T_n(x) should be bounded by 1 for |x| <= 1."""
        layer = Chebyshev1st(degree=50, units=1, input_clip=(-1.0, 1.0))
        layer.build((None, 1))
        
        x = tf.constant(np.linspace(-0.99, 0.99, 50).reshape(-1, 1), dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)
        
        max_val = tf.reduce_max(tf.abs(basis))
        assert max_val <= 1.1, f"Chebyshev exceeded bounds: {max_val}"  # Small tolerance

    def test_gegenbauer_moderate_alpha_stable(self):
        """Gegenbauer with moderate alpha should be stable."""
        layer = Gegenbauer(degree=20, units=1, alpha_init=0.5, input_clip=(-1.0, 1.0))
        layer.build((None, 1))
        
        x = tf.constant(np.linspace(-0.9, 0.9, 20).reshape(-1, 1), dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)
        
        assert not tf.reduce_any(tf.math.is_nan(basis)), "NaN in Gegenbauer"
        assert not tf.reduce_any(tf.math.is_inf(basis)), "Inf in Gegenbauer"


class TestDtypePreservation:
    """Verify dtype handling doesn't silently change precision."""

    def test_float32_input_float32_output(self):
        """Float32 input should produce float32 output."""
        layer = Legendre(degree=5, units=2)
        x = tf.constant([[0.5, -0.3]], dtype=tf.float32)
        y = layer(x)
        assert y.dtype == tf.float32, f"Expected float32, got {y.dtype}"

    def test_float64_promotion_returns_float32(self):
        """Even with float64 promotion, output should match input dtype."""
        layer = Legendre(degree=15, units=2, promote_to_float64=True)
        x = tf.constant([[0.5, -0.3]], dtype=tf.float32)
        y = layer(x)
        assert y.dtype == tf.float32, f"Expected float32 after promotion, got {y.dtype}"


class TestRBFPrecision:
    """RBF layers numerical precision tests."""

    def test_gaussian_rbf_no_nan(self):
        """Gaussian RBF should not produce NaN values."""
        layer = GaussianRBF(units=4, num_grids=5, grid_min=-1.0, grid_max=1.0)
        x = tf.constant([[0.0, 0.5, -0.5]], dtype=tf.float32)
        y = layer(x)
        
        assert not tf.reduce_any(tf.math.is_nan(y)), "NaN in GaussianRBF output"
        assert not tf.reduce_any(tf.math.is_inf(y)), "Inf in GaussianRBF output"

    def test_gaussian_rbf_bounded_output(self):
        """Gaussian RBF intermediate values should be bounded."""
        layer = GaussianRBF(units=4, num_grids=5, grid_min=-1.0, grid_max=1.0, epsilon_init=1.0)
        layer.build((None, 3))
        
        x = tf.constant([[0.0, 0.5, -0.5]], dtype=tf.float32)
        # Build and call to get basis
        y = layer(x)
        
        # Output should be finite
        assert tf.reduce_all(tf.math.is_finite(y))


class TestWaveletPrecision:
    """Wavelet layers numerical precision tests."""

    def test_ricker_no_nan(self):
        """Ricker wavelet should not produce NaN values."""
        layer = Ricker(units=4)
        x = tf.constant([[0.0, 0.5, -0.5]], dtype=tf.float32)
        y = layer(x)
        
        assert not tf.reduce_any(tf.math.is_nan(y)), "NaN in Ricker"
        assert not tf.reduce_any(tf.math.is_inf(y)), "Inf in Ricker"

    def test_ricker_finite_output(self):
        """Ricker wavelet output should be finite."""
        layer = Ricker(units=4)
        x = tf.constant(np.linspace(-2, 2, 10).reshape(2, 5).astype(np.float32))
        y = layer(x)
        
        assert tf.reduce_all(tf.math.is_finite(y)), "Non-finite values in Ricker output"


class TestGradientNumericalStability:
    """Ensure gradients don't explode or vanish."""

    def test_legendre_gradient_bounded(self):
        """Gradients through Legendre layer should be bounded."""
        layer = Legendre(degree=10, units=2)
        x = tf.Variable([[0.5, -0.3]], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            y = layer(x)
            loss = tf.reduce_sum(y ** 2)
        
        grads = tape.gradient(loss, layer.trainable_variables)
        
        for grad in grads:
            if grad is not None:
                max_grad = tf.reduce_max(tf.abs(grad))
                assert max_grad < 1e6, f"Gradient explosion: {max_grad}"
                assert not tf.reduce_any(tf.math.is_nan(grad)), "NaN in gradients"

    def test_jacobi_gradient_with_params(self):
        """Jacobi gradients w.r.t. alpha/beta should be stable."""
        layer = Jacobi(degree=5, units=2, alpha_init=0.5, beta_init=0.5)
        x = tf.constant([[0.3, -0.2]], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            y = layer(x)
            loss = tf.reduce_sum(y ** 2)
        
        grads = tape.gradient(loss, layer.trainable_variables)
        
        for grad in grads:
            if grad is not None:
                assert not tf.reduce_any(tf.math.is_nan(grad)), "NaN in Jacobi gradients"
                assert not tf.reduce_any(tf.math.is_inf(grad)), "Inf in Jacobi gradients"


class TestCrossEntropyWithPolynomials:
    """Test numerical stability in typical loss scenarios."""

    def test_legendre_cross_entropy_stable(self):
        """Cross-entropy loss with Legendre output should be stable."""
        layer = Legendre(degree=5, units=10, activation="softmax")
        x = tf.constant(np.random.randn(8, 4).astype(np.float32))
        y_true = tf.constant(np.eye(10)[np.random.randint(0, 10, 8)], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            y_pred = layer(x)
            # Clip to avoid log(0)
            y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0)
            loss = -tf.reduce_mean(tf.reduce_sum(y_true * tf.math.log(y_pred_clipped), axis=-1))
        
        assert not tf.math.is_nan(loss), "NaN loss in cross-entropy"
        assert not tf.math.is_inf(loss), "Inf loss in cross-entropy"
        
        grads = tape.gradient(loss, layer.trainable_variables)
        for grad in grads:
            if grad is not None:
                assert not tf.reduce_any(tf.math.is_nan(grad)), "NaN gradients"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
