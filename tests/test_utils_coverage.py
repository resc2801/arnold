## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.

"""
Tests for utility functions and math helpers to increase coverage.

This file tests:
1. arnold.math.hypergeometric - Pochhammer and generalized hypergeometric
2. arnold.utils.numerics - Safe math functions
3. arnold.layers.core.kan_base - Hardware detection utilities
"""

import numpy as np
import pytest
import tensorflow as tf

from arnold.math.hypergeometric import pochhammer, generalized_hypergeometric
from arnold.utils.numerics import safe_acos, safe_log, safe_reciprocal, clamp_abs
from arnold.layers.core.kan_base import detect_hardware, get_recommended_dtype


# =============================================================================
# Pochhammer Symbol Tests
# =============================================================================

class TestPochhammer:
    """Tests for the Pochhammer (rising factorial) function."""
    
    def test_pochhammer_k_zero(self):
        """(x)_0 = 1 for any x."""
        x = tf.constant([1.0, 2.0, 5.0, -1.0], dtype=tf.float32)
        result = pochhammer(0, x)
        np.testing.assert_allclose(result.numpy(), [1.0, 1.0, 1.0, 1.0], atol=1e-6)
    
    def test_pochhammer_k_one(self):
        """(x)_1 = x."""
        x = tf.constant([1.0, 2.0, 5.0, 0.5], dtype=tf.float32)
        result = pochhammer(1, x)
        np.testing.assert_allclose(result.numpy(), x.numpy(), atol=1e-6)
    
    def test_pochhammer_k_two(self):
        """(x)_2 = x * (x+1)."""
        x = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        result = pochhammer(2, x)
        expected = x.numpy() * (x.numpy() + 1)
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-6)
    
    def test_pochhammer_k_three(self):
        """(x)_3 = x * (x+1) * (x+2)."""
        x = tf.constant([1.0, 2.0], dtype=tf.float32)
        result = pochhammer(3, x)
        expected = [1*2*3, 2*3*4]  # 6, 24
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-6)
    
    def test_pochhammer_positive_integers(self):
        """Test Pochhammer with positive integers uses lgamma path."""
        x = tf.constant([1.0, 2.0, 3.0], dtype=tf.float64)  # All positive
        result = pochhammer(4, x)
        expected = [1*2*3*4, 2*3*4*5, 3*4*5*6]  # 24, 120, 360
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)
    
    def test_pochhammer_with_zero_and_negative(self):
        """Test Pochhammer with zero or negative uses iteration path."""
        x = tf.constant([-1.0, 0.0, 1.0], dtype=tf.float32)  # Mixed signs
        result = pochhammer(2, x)
        # (-1)_2 = -1 * 0 = 0
        # (0)_2 = 0 * 1 = 0
        # (1)_2 = 1 * 2 = 2
        expected = [-1*0, 0*1, 1*2]
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-6)


# =============================================================================
# Generalized Hypergeometric Tests
# =============================================================================

class TestGeneralizedHypergeometric:
    """Tests for the generalized hypergeometric function."""
    
    def test_0F0(self):
        """0F0(;; z) = exp(z)."""
        z = tf.constant([0.5, 1.0, 2.0], dtype=tf.float32)
        result = generalized_hypergeometric([], [], z, num_terms=15)
        expected = np.exp(z.numpy())
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-3)
    
    def test_1F0(self):
        """1F0(a;; z) = (1-z)^(-a) for |z| < 1."""
        a = tf.constant([2.0], dtype=tf.float32)
        z = tf.constant([0.3, 0.5], dtype=tf.float32)
        
        result = generalized_hypergeometric([a], [], z, num_terms=20)
        expected = (1.0 - z.numpy()) ** (-a.numpy()[0])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-2)
    
    def test_1F1_small_z(self):
        """Test 1F1 (Kummer's function) with small z."""
        a = tf.constant([1.0], dtype=tf.float32)
        b = tf.constant([2.0], dtype=tf.float32)
        z = tf.constant([0.2], dtype=tf.float32)
        
        result = generalized_hypergeometric([a], [b], z, num_terms=10)
        assert result.shape == (1,)
        assert not tf.reduce_any(tf.math.is_nan(result))
    
    def test_2F1_small_z(self):
        """Test 2F1 (Gauss hypergeometric) with small z."""
        a1 = tf.constant([1.0], dtype=tf.float32)
        a2 = tf.constant([2.0], dtype=tf.float32)
        b = tf.constant([3.0], dtype=tf.float32)
        z = tf.constant([0.1], dtype=tf.float32)
        
        result = generalized_hypergeometric([a1, a2], [b], z, num_terms=10)
        assert result.shape == (1,)
        assert not tf.reduce_any(tf.math.is_nan(result))


# =============================================================================
# Safe Numerics Tests
# =============================================================================

class TestSafeAcos:
    """Tests for numerically stable arccos."""
    
    def test_normal_values(self):
        """Test safe_acos with normal values."""
        x = tf.constant([0.0, 0.5, -0.5], dtype=tf.float32)
        result = safe_acos(x)
        expected = tf.math.acos(x)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-5)
    
    def test_boundary_clipping(self):
        """Test safe_acos clips values near boundaries."""
        # Values slightly outside [-1, 1] should be clipped
        x = tf.constant([1.0001, -1.0001, 0.999999], dtype=tf.float32)
        result = safe_acos(x)
        
        # Should not produce NaN
        assert not tf.reduce_any(tf.math.is_nan(result))
    
    def test_near_boundaries(self):
        """Test safe_acos at values near boundaries."""
        x = tf.constant([0.9999, -0.9999], dtype=tf.float32)
        result = safe_acos(x)
        
        # acos(0.9999) ≈ 0.014, acos(-0.9999) ≈ 3.127
        assert not tf.reduce_any(tf.math.is_nan(result))
        assert result[0] > 0 and result[0] < 0.1
        assert result[1] > 3.0 and result[1] < np.pi


class TestSafeLog:
    """Tests for numerically stable logarithm."""
    
    def test_positive_values(self):
        """Test safe_log with positive values."""
        x = tf.constant([1.0, 2.0, np.e], dtype=tf.float32)
        result = safe_log(x)
        expected = [0.0, np.log(2), 1.0]
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)
    
    def test_zero_and_negative(self):
        """Test safe_log with zero and negative values (should floor)."""
        x = tf.constant([0.0, -1.0, 1e-10], dtype=tf.float32)
        result = safe_log(x)
        
        # Should not produce -inf or NaN
        assert not tf.reduce_any(tf.math.is_inf(result))
        assert not tf.reduce_any(tf.math.is_nan(result))


class TestSafeReciprocal:
    """Tests for numerically stable reciprocal."""
    
    def test_normal_values(self):
        """Test safe_reciprocal with normal values."""
        x = tf.constant([1.0, 2.0, -2.0], dtype=tf.float32)
        result = safe_reciprocal(x)
        
        # For large |x|, result ≈ 1/x
        expected = x / (x**2 + 1e-7)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-5)
    
    def test_near_zero(self):
        """Test safe_reciprocal near zero."""
        x = tf.constant([1e-10, 0.0, -1e-10], dtype=tf.float32)
        result = safe_reciprocal(x)
        
        # Should not produce inf or NaN
        assert not tf.reduce_any(tf.math.is_inf(result))
        assert not tf.reduce_any(tf.math.is_nan(result))


class TestClampAbs:
    """Tests for clamp_abs utility."""
    
    def test_values_above_threshold(self):
        """Test clamp_abs with values above threshold."""
        x = tf.constant([1.0, -2.0, 0.5], dtype=tf.float32)
        result = clamp_abs(x, eps=0.1)
        
        # Values far from zero should be unchanged
        np.testing.assert_allclose(result.numpy(), x.numpy(), atol=1e-6)
    
    def test_values_near_zero(self):
        """Test clamp_abs with values near zero."""
        x = tf.constant([1e-8, -1e-8, 0.0], dtype=tf.float32)
        result = clamp_abs(x, eps=1e-6)
        
        # Small positive and zero should become +eps
        # Small negative should become -eps
        expected = [1e-6, -1e-6, 1e-6]
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-10)
    
    def test_preserves_sign(self):
        """Test clamp_abs preserves sign for near-zero values."""
        x = tf.constant([1e-8, -1e-8], dtype=tf.float32)
        result = clamp_abs(x, eps=1e-6)
        
        # Positive stays positive, negative stays negative
        assert result[0] > 0
        assert result[1] < 0
    
    def test_zero_becomes_positive_eps(self):
        """Test that exactly zero becomes +eps (consistent convention)."""
        x = tf.constant([0.0], dtype=tf.float32)
        result = clamp_abs(x, eps=1e-6)
        
        # Zero is non-negative, so it maps to +eps
        np.testing.assert_allclose(result.numpy(), [1e-6], atol=1e-10)


# =============================================================================
# Hardware Detection Tests
# =============================================================================

class TestHardwareDetection:
    """Tests for hardware detection utilities."""
    
    def test_detect_hardware_returns_valid(self):
        """Test detect_hardware returns a valid string."""
        hw = detect_hardware()
        assert hw in ("cpu", "gpu", "tpu", "mps")
    
    def test_get_recommended_dtype_cpu(self):
        """Test dtype recommendation for CPU."""
        dtype_low = get_recommended_dtype(5, hardware="cpu")
        dtype_high = get_recommended_dtype(15, hardware="cpu")
        
        assert dtype_low == tf.float32
        assert dtype_high == tf.float64
    
    def test_get_recommended_dtype_gpu(self):
        """Test dtype recommendation for GPU."""
        dtype = get_recommended_dtype(15, hardware="gpu")
        assert dtype == tf.float32
    
    def test_get_recommended_dtype_tpu(self):
        """Test dtype recommendation for TPU."""
        dtype = get_recommended_dtype(15, hardware="tpu")
        assert dtype == tf.float32
    
    def test_get_recommended_dtype_mps(self):
        """Test dtype recommendation for MPS (Apple Metal)."""
        dtype = get_recommended_dtype(15, hardware="mps")
        assert dtype == tf.float32
    
    def test_get_recommended_dtype_auto(self):
        """Test dtype recommendation with auto-detection."""
        dtype = get_recommended_dtype(15)  # No hardware specified
        assert dtype in (tf.float32, tf.float64, tf.bfloat16)
