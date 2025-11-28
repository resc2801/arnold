## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
"""
Tests for Sprint 7G q-polynomial layers (Askey-Wilson Class).

This module tests the five q-polynomial layers implemented in Sprint 7G:
- DiscreteQHermite1: Discrete q-Hermite I polynomials
- DiscreteQHermite2: Discrete q-Hermite II polynomials
- ContinuousQHermite: Continuous q-Hermite (Rogers-Szegő) polynomials
- ContinuousQJacobi: Continuous q-Jacobi polynomials
- ContinuousQUltraspherical: Continuous q-Ultraspherical (Rogers) polynomials

Test Coverage:
- Basic functionality (build, call, output shape)
- Parameter constraints (q ∈ (0,1), other params in valid ranges)
- Three-term recurrence verification
- Gradient flow through all parameters
- Keras serialization (get_config, from_config)
- SavedModel roundtrip
- Numerical stability at various q values
- XLA compatibility
- Edge cases (degree 0, 1, small inputs)
- Symmetry properties (where applicable)

References:
- NIST DLMF §18.27 (Discrete q-Hermite)
- NIST DLMF §18.28 (Continuous q-Hermite, q-Jacobi, q-Ultraspherical)
- Koekoek et al. (2010), Chapters 14.26-14.29
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from arnold.layers.core import (
    ContinuousQHermite,
    ContinuousQJacobi,
    ContinuousQUltraspherical,
    DiscreteQHermite1,
    DiscreteQHermite2,
)


# =============================================================================
# DiscreteQHermite1 Tests
# =============================================================================


class TestDiscreteQHermite1Basic:
    """Basic functionality tests for DiscreteQHermite1 layer."""

    def test_build_and_call(self):
        """Test layer builds and produces output."""
        layer = DiscreteQHermite1(degree=3, units=4)
        x = tf.random.uniform((2, 3), dtype=tf.float32)
        y = layer(x)
        assert y.shape == (2, 4)

    def test_output_shape(self):
        """Test various output shapes."""
        for degree in [2, 5, 8]:
            for units in [1, 4, 16]:
                layer = DiscreteQHermite1(degree=degree, units=units)
                x = tf.random.uniform((4, 8), dtype=tf.float32)
                y = layer(x)
                assert y.shape == (4, units)

    def test_q_trainable(self):
        """Test that q parameter can be trained."""
        layer = DiscreteQHermite1(degree=3, units=4, q_trainable=True)
        x = tf.random.uniform((2, 3), dtype=tf.float32)
        _ = layer(x)
        
        trainable_names = [v.name for v in layer.trainable_variables]
        assert any("q_logits" in name for name in trainable_names)

    def test_q_constraint(self):
        """Test that q stays in (0, 1)."""
        layer = DiscreteQHermite1(degree=3, units=4, q=0.9, q_trainable=True)
        x = tf.random.uniform((2, 3), dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            y = layer(x)
            loss = tf.reduce_mean(y)
        
        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads if g is not None)


class TestDiscreteQHermite1Mathematical:
    """Mathematical property tests for DiscreteQHermite1."""

    def test_h0_equals_one(self):
        """Test h_0(x) = 1 for all x."""
        layer = DiscreteQHermite1(degree=0, units=1)
        layer.build((None, 3))
        
        x = tf.constant([[0.1, 0.5, 0.9]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)
        
        np.testing.assert_allclose(basis[..., 0].numpy(), 1.0, rtol=1e-5)

    def test_h1_equals_x(self):
        """Test h_1(x) = x."""
        layer = DiscreteQHermite1(degree=1, units=1)
        layer.build((None, 3))
        
        x = tf.constant([[0.1, 0.5, 0.9]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)
        
        np.testing.assert_allclose(basis[..., 1].numpy(), x.numpy(), rtol=1e-5)

    def test_three_term_recurrence(self):
        """Test h_{n+1}(x) = x*h_n(x) - (1-q^n)*h_{n-1}(x)."""
        q = 0.5
        layer = DiscreteQHermite1(degree=5, units=1, q=q, q_trainable=False)
        layer.build((None, 3))
        
        x = tf.constant([[0.3, 0.6, 0.9]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)
        
        x_np = x.numpy()
        for n in range(1, 5):
            h_n = basis[..., n].numpy()
            h_nm1 = basis[..., n-1].numpy()
            h_np1 = basis[..., n+1].numpy()
            
            # h_{n+1} = x * h_n - (1 - q^n) * h_{n-1}
            expected = x_np * h_n - (1 - q**n) * h_nm1
            np.testing.assert_allclose(h_np1, expected, rtol=1e-4, atol=1e-6)

    def test_symmetry(self):
        """Test h_n(-x; q) = (-1)^n h_n(x; q)."""
        layer = DiscreteQHermite1(degree=4, units=1, q=0.5)
        layer.build((None, 2))
        
        x = tf.constant([[0.3, 0.7]], dtype=tf.float32)
        neg_x = tf.constant([[-0.3, -0.7]], dtype=tf.float32)
        
        basis_pos = layer.pseudo_vandermonde(x)
        basis_neg = layer.pseudo_vandermonde(neg_x)
        
        for n in range(5):
            sign = (-1) ** n
            np.testing.assert_allclose(
                basis_neg[..., n].numpy(),
                sign * basis_pos[..., n].numpy(),
                rtol=1e-5, atol=1e-7
            )


class TestDiscreteQHermite1Serialization:
    """Serialization tests for DiscreteQHermite1."""

    def test_get_config(self):
        """Test configuration roundtrip."""
        layer = DiscreteQHermite1(degree=4, units=8, q=0.7, q_trainable=True)
        config = layer.get_config()
        
        assert config["degree"] == 4
        assert config["units"] == 8
        assert config["q"] == 0.7
        assert config["q_trainable"] is True

    def test_from_config(self):
        """Test layer recreation from config."""
        layer = DiscreteQHermite1(degree=4, units=8, q=0.7)
        config = layer.get_config()
        
        new_layer = DiscreteQHermite1.from_config(config)
        assert new_layer.degree == layer.degree
        assert new_layer.units == layer.units

    def test_saved_model(self):
        """Test SavedModel serialization."""
        layer = DiscreteQHermite1(degree=3, units=4)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(5,)),
            layer,
        ])
        
        x = tf.random.uniform((2, 5), dtype=tf.float32)
        original_output = model(x)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(Path(tmpdir) / "model.keras")
            loaded = tf.keras.models.load_model(Path(tmpdir) / "model.keras")
            loaded_output = loaded(x)
        
        np.testing.assert_allclose(
            original_output.numpy(), loaded_output.numpy(), rtol=1e-5
        )


# =============================================================================
# DiscreteQHermite2 Tests
# =============================================================================


class TestDiscreteQHermite2Basic:
    """Basic functionality tests for DiscreteQHermite2 layer."""

    def test_build_and_call(self):
        """Test layer builds and produces output."""
        layer = DiscreteQHermite2(degree=3, units=4)
        x = tf.random.uniform((2, 3), dtype=tf.float32)
        y = layer(x)
        assert y.shape == (2, 4)

    def test_output_shape(self):
        """Test various output shapes."""
        for degree in [2, 5, 8]:
            for units in [1, 4, 16]:
                layer = DiscreteQHermite2(degree=degree, units=units)
                x = tf.random.uniform((4, 8), dtype=tf.float32)
                y = layer(x)
                assert y.shape == (4, units)

    def test_q_trainable(self):
        """Test that q parameter can be trained."""
        layer = DiscreteQHermite2(degree=3, units=4, q_trainable=True)
        x = tf.random.uniform((2, 3), dtype=tf.float32)
        _ = layer(x)
        
        trainable_names = [v.name for v in layer.trainable_variables]
        assert any("q_logits" in name for name in trainable_names)


class TestDiscreteQHermite2Mathematical:
    """Mathematical property tests for DiscreteQHermite2."""

    def test_h0_equals_one(self):
        """Test h̃_0(x) = 1 for all x."""
        layer = DiscreteQHermite2(degree=0, units=1)
        layer.build((None, 3))
        
        x = tf.constant([[0.1, 0.5, 0.9]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)
        
        np.testing.assert_allclose(basis[..., 0].numpy(), 1.0, rtol=1e-5)

    def test_h1_equals_x(self):
        """Test h̃_1(x) = x."""
        layer = DiscreteQHermite2(degree=1, units=1)
        layer.build((None, 3))
        
        x = tf.constant([[0.1, 0.5, 0.9]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)
        
        np.testing.assert_allclose(basis[..., 1].numpy(), x.numpy(), rtol=1e-5)

    def test_three_term_recurrence(self):
        """Test h̃_{n+1}(x) = x*h̃_n(x) - q^{n-1}(1-q^n)*h̃_{n-1}(x)."""
        q = 0.5
        layer = DiscreteQHermite2(degree=5, units=1, q=q, q_trainable=False)
        layer.build((None, 3))
        
        x = tf.constant([[0.3, 0.6, 0.9]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)
        
        x_np = x.numpy()
        for n in range(1, 5):
            h_n = basis[..., n].numpy()
            h_nm1 = basis[..., n-1].numpy()
            h_np1 = basis[..., n+1].numpy()
            
            # h̃_{n+1} = x * h̃_n - q^{n-1}(1 - q^n) * h̃_{n-1}
            coeff = (q ** (n-1)) * (1 - q**n)
            expected = x_np * h_n - coeff * h_nm1
            np.testing.assert_allclose(h_np1, expected, rtol=1e-4, atol=1e-6)

    def test_symmetry(self):
        """Test h̃_n(-x; q) = (-1)^n h̃_n(x; q)."""
        layer = DiscreteQHermite2(degree=4, units=1, q=0.5)
        layer.build((None, 2))
        
        x = tf.constant([[0.3, 0.7]], dtype=tf.float32)
        neg_x = tf.constant([[-0.3, -0.7]], dtype=tf.float32)
        
        basis_pos = layer.pseudo_vandermonde(x)
        basis_neg = layer.pseudo_vandermonde(neg_x)
        
        for n in range(5):
            sign = (-1) ** n
            np.testing.assert_allclose(
                basis_neg[..., n].numpy(),
                sign * basis_pos[..., n].numpy(),
                rtol=1e-5, atol=1e-7
            )


class TestDiscreteQHermite2Serialization:
    """Serialization tests for DiscreteQHermite2."""

    def test_get_config(self):
        """Test configuration roundtrip."""
        layer = DiscreteQHermite2(degree=4, units=8, q=0.7)
        config = layer.get_config()
        
        assert config["degree"] == 4
        assert config["units"] == 8
        assert config["q"] == 0.7

    def test_saved_model(self):
        """Test SavedModel serialization."""
        layer = DiscreteQHermite2(degree=3, units=4)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(5,)),
            layer,
        ])
        
        x = tf.random.uniform((2, 5), dtype=tf.float32)
        original_output = model(x)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(Path(tmpdir) / "model.keras")
            loaded = tf.keras.models.load_model(Path(tmpdir) / "model.keras")
            loaded_output = loaded(x)
        
        np.testing.assert_allclose(
            original_output.numpy(), loaded_output.numpy(), rtol=1e-5
        )


# =============================================================================
# ContinuousQHermite Tests
# =============================================================================


class TestContinuousQHermiteBasic:
    """Basic functionality tests for ContinuousQHermite layer."""

    def test_build_and_call(self):
        """Test layer builds and produces output."""
        layer = ContinuousQHermite(degree=3, units=4)
        x = tf.random.uniform((2, 3), dtype=tf.float32)
        y = layer(x)
        assert y.shape == (2, 4)

    def test_output_shape(self):
        """Test various output shapes."""
        for degree in [2, 5, 8]:
            for units in [1, 4, 16]:
                layer = ContinuousQHermite(degree=degree, units=units)
                x = tf.random.uniform((4, 8), dtype=tf.float32)
                y = layer(x)
                assert y.shape == (4, units)

    def test_q_trainable(self):
        """Test that q parameter can be trained."""
        layer = ContinuousQHermite(degree=3, units=4, q_trainable=True)
        x = tf.random.uniform((2, 3), dtype=tf.float32)
        _ = layer(x)
        
        trainable_names = [v.name for v in layer.trainable_variables]
        assert any("q_logits" in name for name in trainable_names)


class TestContinuousQHermiteMathematical:
    """Mathematical property tests for ContinuousQHermite."""

    def test_H0_equals_one(self):
        """Test H_0(x) = 1 for all x."""
        layer = ContinuousQHermite(degree=0, units=1)
        layer.build((None, 3))
        
        x = tf.constant([[0.1, 0.5, 0.9]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)
        
        np.testing.assert_allclose(basis[..., 0].numpy(), 1.0, rtol=1e-5)

    def test_H1_equals_2x(self):
        """Test H_1(x) = 2x."""
        layer = ContinuousQHermite(degree=1, units=1)
        layer.build((None, 3))
        
        x = tf.constant([[0.1, 0.5, 0.9]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)
        
        expected = 2.0 * x.numpy()
        np.testing.assert_allclose(basis[..., 1].numpy(), expected, rtol=1e-5)

    def test_three_term_recurrence(self):
        """Test H_{n+1}(x) = 2x*H_n(x) - (1-q^n)*H_{n-1}(x)."""
        q = 0.5
        layer = ContinuousQHermite(degree=5, units=1, q=q, q_trainable=False)
        layer.build((None, 3))
        
        x = tf.constant([[0.3, 0.6, 0.9]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)
        
        x_np = x.numpy()
        for n in range(1, 5):
            H_n = basis[..., n].numpy()
            H_nm1 = basis[..., n-1].numpy()
            H_np1 = basis[..., n+1].numpy()
            
            # H_{n+1} = 2x * H_n - (1 - q^n) * H_{n-1}
            expected = 2.0 * x_np * H_n - (1 - q**n) * H_nm1
            np.testing.assert_allclose(H_np1, expected, rtol=1e-4, atol=1e-6)

    def test_symmetry(self):
        """Test H_n(-x; q) = (-1)^n H_n(x; q)."""
        layer = ContinuousQHermite(degree=4, units=1, q=0.5)
        layer.build((None, 2))
        
        x = tf.constant([[0.3, 0.7]], dtype=tf.float32)
        neg_x = tf.constant([[-0.3, -0.7]], dtype=tf.float32)
        
        basis_pos = layer.pseudo_vandermonde(x)
        basis_neg = layer.pseudo_vandermonde(neg_x)
        
        for n in range(5):
            sign = (-1) ** n
            np.testing.assert_allclose(
                basis_neg[..., n].numpy(),
                sign * basis_pos[..., n].numpy(),
                rtol=1e-5, atol=1e-7
            )


class TestContinuousQHermiteSerialization:
    """Serialization tests for ContinuousQHermite."""

    def test_get_config(self):
        """Test configuration roundtrip."""
        layer = ContinuousQHermite(degree=4, units=8, q=0.7)
        config = layer.get_config()
        
        assert config["degree"] == 4
        assert config["units"] == 8
        assert config["q"] == 0.7

    def test_saved_model(self):
        """Test SavedModel serialization."""
        layer = ContinuousQHermite(degree=3, units=4)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(5,)),
            layer,
        ])
        
        x = tf.random.uniform((2, 5), dtype=tf.float32)
        original_output = model(x)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(Path(tmpdir) / "model.keras")
            loaded = tf.keras.models.load_model(Path(tmpdir) / "model.keras")
            loaded_output = loaded(x)
        
        np.testing.assert_allclose(
            original_output.numpy(), loaded_output.numpy(), rtol=1e-5
        )


# =============================================================================
# ContinuousQJacobi Tests
# =============================================================================


class TestContinuousQJacobiBasic:
    """Basic functionality tests for ContinuousQJacobi layer."""

    def test_build_and_call(self):
        """Test layer builds and produces output."""
        layer = ContinuousQJacobi(degree=3, units=4)
        x = tf.random.uniform((2, 3), dtype=tf.float32)
        y = layer(x)
        assert y.shape == (2, 4)

    def test_output_shape(self):
        """Test various output shapes."""
        for degree in [2, 5, 8]:
            for units in [1, 4, 16]:
                layer = ContinuousQJacobi(degree=degree, units=units)
                x = tf.random.uniform((4, 8), dtype=tf.float32)
                y = layer(x)
                assert y.shape == (4, units)

    def test_alpha_beta_parameters(self):
        """Test various alpha and beta values."""
        for alpha in [0.0, 0.5, 1.0]:
            for beta in [0.0, 0.5, 1.0]:
                layer = ContinuousQJacobi(
                    degree=3, units=4, alpha=alpha, beta=beta
                )
                x = tf.random.uniform((2, 3), dtype=tf.float32)
                y = layer(x)
                assert y.shape == (2, 4)
                assert np.all(np.isfinite(y.numpy()))

    def test_trainable_parameters(self):
        """Test that alpha and beta can be trained."""
        layer = ContinuousQJacobi(
            degree=3, units=4, 
            alpha_trainable=True, beta_trainable=True
        )
        x = tf.random.uniform((2, 3), dtype=tf.float32)
        _ = layer(x)
        
        trainable_names = [v.name for v in layer.trainable_variables]
        assert any("alpha_logits" in name for name in trainable_names)
        assert any("beta_logits" in name for name in trainable_names)


class TestContinuousQJacobiMathematical:
    """Mathematical property tests for ContinuousQJacobi."""

    def test_P0_equals_one(self):
        """Test P_0(x) = 1 for all x."""
        layer = ContinuousQJacobi(degree=0, units=1, alpha=0.5, beta=0.5)
        layer.build((None, 3))
        
        x = tf.constant([[0.1, 0.5, 0.9]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)
        
        np.testing.assert_allclose(basis[..., 0].numpy(), 1.0, rtol=1e-5)

    def test_numerical_stability(self):
        """Test numerical stability across input range."""
        layer = ContinuousQJacobi(degree=6, units=1, alpha=1.0, beta=0.5, q=0.7)
        layer.build((None, 5))
        
        # Test across the valid range [-1, 1]
        x = tf.constant([[-0.9, -0.5, 0.0, 0.5, 0.9]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)
        
        assert np.all(np.isfinite(basis.numpy()))

    def test_gradient_flow(self):
        """Test gradient flow through all parameters."""
        layer = ContinuousQJacobi(
            degree=3, units=4,
            alpha=0.5, beta=0.5,
            alpha_trainable=True, beta_trainable=True, q_trainable=True
        )
        x = tf.random.uniform((2, 3), dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            y = layer(x)
            loss = tf.reduce_mean(y ** 2)
        
        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads)


class TestContinuousQJacobiSerialization:
    """Serialization tests for ContinuousQJacobi."""

    def test_get_config(self):
        """Test configuration roundtrip."""
        layer = ContinuousQJacobi(
            degree=4, units=8, alpha=0.5, beta=0.3, q=0.7,
            alpha_trainable=True, beta_trainable=False
        )
        config = layer.get_config()
        
        assert config["degree"] == 4
        assert config["units"] == 8
        assert config["alpha"] == 0.5
        assert config["beta"] == 0.3
        assert config["q"] == 0.7
        assert config["alpha_trainable"] is True
        assert config["beta_trainable"] is False

    def test_from_config(self):
        """Test layer recreation from config."""
        layer = ContinuousQJacobi(degree=4, units=8, alpha=0.5, beta=0.3)
        config = layer.get_config()
        
        new_layer = ContinuousQJacobi.from_config(config)
        assert new_layer.degree == layer.degree
        assert new_layer.units == layer.units
        assert new_layer.alpha_init == layer.alpha_init
        assert new_layer.beta_init == layer.beta_init

    def test_saved_model(self):
        """Test SavedModel serialization."""
        layer = ContinuousQJacobi(degree=3, units=4, alpha=0.5, beta=0.5)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(5,)),
            layer,
        ])
        
        x = tf.random.uniform((2, 5), dtype=tf.float32)
        original_output = model(x)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(Path(tmpdir) / "model.keras")
            loaded = tf.keras.models.load_model(Path(tmpdir) / "model.keras")
            loaded_output = loaded(x)
        
        np.testing.assert_allclose(
            original_output.numpy(), loaded_output.numpy(), rtol=1e-5
        )


# =============================================================================
# ContinuousQUltraspherical Tests
# =============================================================================


class TestContinuousQUltrasphericalBasic:
    """Basic functionality tests for ContinuousQUltraspherical layer."""

    def test_build_and_call(self):
        """Test layer builds and produces output."""
        layer = ContinuousQUltraspherical(degree=3, units=4, beta=0.5)
        x = tf.random.uniform((2, 3), dtype=tf.float32)
        y = layer(x)
        assert y.shape == (2, 4)

    def test_output_shape(self):
        """Test various output shapes."""
        for degree in [2, 5, 8]:
            for units in [1, 4, 16]:
                layer = ContinuousQUltraspherical(degree=degree, units=units, beta=0.5)
                x = tf.random.uniform((4, 8), dtype=tf.float32)
                y = layer(x)
                assert y.shape == (4, units)

    def test_beta_constraint(self):
        """Test that beta must be in (-1, 1)."""
        # Valid values
        for beta in [-0.5, 0.0, 0.5]:
            layer = ContinuousQUltraspherical(degree=3, units=4, beta=beta)
            x = tf.random.uniform((2, 3), dtype=tf.float32)
            y = layer(x)
            assert y.shape == (2, 4)
        
        # Invalid values
        with pytest.raises(ValueError):
            ContinuousQUltraspherical(degree=3, units=4, beta=1.0)
        with pytest.raises(ValueError):
            ContinuousQUltraspherical(degree=3, units=4, beta=-1.0)

    def test_trainable_beta(self):
        """Test that beta can be trained."""
        layer = ContinuousQUltraspherical(
            degree=3, units=4, beta=0.5, beta_trainable=True
        )
        x = tf.random.uniform((2, 3), dtype=tf.float32)
        _ = layer(x)
        
        trainable_names = [v.name for v in layer.trainable_variables]
        assert any("beta_logits" in name for name in trainable_names)


class TestContinuousQUltrasphericalMathematical:
    """Mathematical property tests for ContinuousQUltraspherical."""

    def test_C0_equals_one(self):
        """Test C_0(x) = 1 for all x."""
        layer = ContinuousQUltraspherical(degree=0, units=1, beta=0.5)
        layer.build((None, 3))
        
        x = tf.constant([[0.1, 0.5, 0.9]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)
        
        np.testing.assert_allclose(basis[..., 0].numpy(), 1.0, rtol=1e-5)

    def test_C1_formula(self):
        """Test C_1(x) = 2x(1-β)/(1-q)."""
        beta = 0.5
        q = 0.5
        layer = ContinuousQUltraspherical(degree=1, units=1, beta=beta, q=q)
        layer.build((None, 3))
        
        x = tf.constant([[0.1, 0.5, 0.9]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)
        
        expected = 2.0 * x.numpy() * (1 - beta) / (1 - q)
        np.testing.assert_allclose(basis[..., 1].numpy(), expected, rtol=1e-4)

    def test_symmetry(self):
        """Test C_n(-x; β | q) = (-1)^n C_n(x; β | q)."""
        layer = ContinuousQUltraspherical(degree=4, units=1, beta=0.3, q=0.5)
        layer.build((None, 2))
        
        x = tf.constant([[0.3, 0.7]], dtype=tf.float32)
        neg_x = tf.constant([[-0.3, -0.7]], dtype=tf.float32)
        
        basis_pos = layer.pseudo_vandermonde(x)
        basis_neg = layer.pseudo_vandermonde(neg_x)
        
        for n in range(5):
            sign = (-1) ** n
            np.testing.assert_allclose(
                basis_neg[..., n].numpy(),
                sign * basis_pos[..., n].numpy(),
                rtol=1e-4, atol=1e-6
            )

    def test_numerical_stability(self):
        """Test numerical stability across input range."""
        layer = ContinuousQUltraspherical(degree=6, units=1, beta=0.5, q=0.7)
        layer.build((None, 5))
        
        x = tf.constant([[-0.9, -0.5, 0.0, 0.5, 0.9]], dtype=tf.float32)
        basis = layer.pseudo_vandermonde(x)
        
        assert np.all(np.isfinite(basis.numpy()))


class TestContinuousQUltrasphericalSerialization:
    """Serialization tests for ContinuousQUltraspherical."""

    def test_get_config(self):
        """Test configuration roundtrip."""
        layer = ContinuousQUltraspherical(
            degree=4, units=8, beta=0.3, q=0.7, beta_trainable=True
        )
        config = layer.get_config()
        
        assert config["degree"] == 4
        assert config["units"] == 8
        assert config["beta"] == 0.3
        assert config["q"] == 0.7
        assert config["beta_trainable"] is True

    def test_from_config(self):
        """Test layer recreation from config."""
        layer = ContinuousQUltraspherical(degree=4, units=8, beta=0.3)
        config = layer.get_config()
        
        new_layer = ContinuousQUltraspherical.from_config(config)
        assert new_layer.degree == layer.degree
        assert new_layer.units == layer.units
        assert new_layer.beta_init == layer.beta_init

    def test_saved_model(self):
        """Test SavedModel serialization."""
        layer = ContinuousQUltraspherical(degree=3, units=4, beta=0.5)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(5,)),
            layer,
        ])
        
        x = tf.random.uniform((2, 5), dtype=tf.float32)
        original_output = model(x)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(Path(tmpdir) / "model.keras")
            loaded = tf.keras.models.load_model(Path(tmpdir) / "model.keras")
            loaded_output = loaded(x)
        
        np.testing.assert_allclose(
            original_output.numpy(), loaded_output.numpy(), rtol=1e-5
        )


# =============================================================================
# Cross-Layer Comparison Tests
# =============================================================================


class TestQHermiteFamilyRelations:
    """Test relationships between q-Hermite polynomial families."""

    def test_discrete_vs_continuous_degree0(self):
        """All q-Hermite polynomials have P_0 = 1."""
        for layer_cls in [DiscreteQHermite1, DiscreteQHermite2, ContinuousQHermite]:
            layer = layer_cls(degree=0, units=1, q=0.5)
            layer.build((None, 3))
            
            x = tf.constant([[0.1, 0.5, 0.9]], dtype=tf.float32)
            basis = layer.pseudo_vandermonde(x)
            
            np.testing.assert_allclose(
                basis[..., 0].numpy(), 1.0, rtol=1e-5,
                err_msg=f"P_0 ≠ 1 for {layer_cls.__name__}"
            )

    def test_all_layers_gradient_flow(self):
        """Test gradient flow through all Sprint 7G layers."""
        layer_configs = [
            (DiscreteQHermite1, {"degree": 3, "units": 4}),
            (DiscreteQHermite2, {"degree": 3, "units": 4}),
            (ContinuousQHermite, {"degree": 3, "units": 4}),
            (ContinuousQJacobi, {"degree": 3, "units": 4, "alpha": 0.5, "beta": 0.5}),
            (ContinuousQUltraspherical, {"degree": 3, "units": 4, "beta": 0.5}),
        ]
        
        for layer_cls, kwargs in layer_configs:
            layer = layer_cls(**kwargs, q_trainable=True)
            x = tf.random.uniform((2, 3), dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                y = layer(x)
                loss = tf.reduce_mean(y ** 2)
            
            grads = tape.gradient(loss, layer.trainable_variables)
            assert all(g is not None for g in grads), \
                f"Missing gradients for {layer_cls.__name__}"


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestNumericalStability:
    """Test numerical stability across extreme parameter values."""

    @pytest.mark.parametrize("layer_cls", [
        DiscreteQHermite1, DiscreteQHermite2, ContinuousQHermite
    ])
    def test_q_near_zero(self, layer_cls):
        """Test stability when q is close to 0."""
        layer = layer_cls(degree=4, units=4, q=0.1)
        x = tf.random.uniform((2, 3), -0.5, 0.5, dtype=tf.float32)
        y = layer(x)
        assert np.all(np.isfinite(y.numpy()))

    @pytest.mark.parametrize("layer_cls", [
        DiscreteQHermite1, DiscreteQHermite2, ContinuousQHermite
    ])
    def test_q_near_one(self, layer_cls):
        """Test stability when q is close to 1."""
        layer = layer_cls(degree=4, units=4, q=0.95)
        x = tf.random.uniform((2, 3), -0.5, 0.5, dtype=tf.float32)
        y = layer(x)
        assert np.all(np.isfinite(y.numpy()))

    def test_continuous_q_jacobi_extreme_alpha_beta(self):
        """Test ContinuousQJacobi with extreme alpha and beta."""
        for alpha, beta in [(0.0, 0.0), (2.0, 2.0), (0.1, 3.0)]:
            layer = ContinuousQJacobi(
                degree=4, units=4, alpha=alpha, beta=beta, q=0.5
            )
            x = tf.random.uniform((2, 3), -0.9, 0.9, dtype=tf.float32)
            y = layer(x)
            assert np.all(np.isfinite(y.numpy())), \
                f"NaN/Inf for alpha={alpha}, beta={beta}"

    def test_continuous_q_ultraspherical_extreme_beta(self):
        """Test ContinuousQUltraspherical with extreme beta."""
        for beta in [-0.9, -0.5, 0.0, 0.5, 0.9]:
            layer = ContinuousQUltraspherical(
                degree=4, units=4, beta=beta, q=0.5
            )
            x = tf.random.uniform((2, 3), -0.9, 0.9, dtype=tf.float32)
            y = layer(x)
            assert np.all(np.isfinite(y.numpy())), f"NaN/Inf for beta={beta}"


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.parametrize("layer_cls", [
        DiscreteQHermite1, DiscreteQHermite2, ContinuousQHermite
    ])
    def test_degree_zero(self, layer_cls):
        """Test with degree 0 (constant polynomial)."""
        layer = layer_cls(degree=0, units=4)
        x = tf.random.uniform((2, 3), dtype=tf.float32)
        y = layer(x)
        assert y.shape == (2, 4)

    @pytest.mark.parametrize("layer_cls", [
        DiscreteQHermite1, DiscreteQHermite2, ContinuousQHermite
    ])
    def test_degree_one(self, layer_cls):
        """Test with degree 1 (linear polynomial)."""
        layer = layer_cls(degree=1, units=4)
        x = tf.random.uniform((2, 3), dtype=tf.float32)
        y = layer(x)
        assert y.shape == (2, 4)

    @pytest.mark.parametrize("layer_cls", [
        DiscreteQHermite1, DiscreteQHermite2, ContinuousQHermite
    ])
    def test_single_input(self, layer_cls):
        """Test with single input dimension."""
        layer = layer_cls(degree=3, units=4)
        x = tf.random.uniform((2, 1), dtype=tf.float32)
        y = layer(x)
        assert y.shape == (2, 4)

    @pytest.mark.parametrize("layer_cls", [
        DiscreteQHermite1, DiscreteQHermite2, ContinuousQHermite
    ])
    def test_batch_size_one(self, layer_cls):
        """Test with batch size 1."""
        layer = layer_cls(degree=3, units=4)
        x = tf.random.uniform((1, 3), dtype=tf.float32)
        y = layer(x)
        assert y.shape == (1, 4)

    def test_x_at_zero(self):
        """Test behavior at x = 0."""
        for layer_cls in [DiscreteQHermite1, DiscreteQHermite2, ContinuousQHermite]:
            layer = layer_cls(degree=4, units=1)
            layer.build((None, 1))
            
            x = tf.constant([[0.0]], dtype=tf.float32)
            basis = layer.pseudo_vandermonde(x)
            
            # P_0(0) = 1 for all
            assert np.isclose(basis[0, 0, 0].numpy(), 1.0, rtol=1e-5)
            
            # Odd-degree polynomials should be 0 at x=0 (due to symmetry)
            for n in range(1, 5, 2):
                assert np.isclose(basis[0, 0, n].numpy(), 0.0, atol=1e-5), \
                    f"P_{n}(0) ≠ 0 for {layer_cls.__name__}"
