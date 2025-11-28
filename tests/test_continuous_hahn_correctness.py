## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
"""
Tests for Continuous Hahn family polynomial KAN layers.

Tests ContinuousHahn, ContinuousDualHahn, DualHahn, and StieltjesWigert layers.
"""
import numpy as np
import pytest
import tensorflow as tf

from arnold.layers import (
    ContinuousDualHahn,
    ContinuousHahn,
    DualHahn,
    StieltjesWigert,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_input():
    """Standard input for testing."""
    return tf.random.uniform((8, 16), -2.0, 2.0, seed=42)


@pytest.fixture
def positive_input():
    """Positive input for polynomials requiring x > 0."""
    return tf.random.uniform((8, 16), 0.1, 2.0, seed=42)


# =============================================================================
# Basic Functionality Tests
# =============================================================================

class TestContinuousHahnBasic:
    """Basic functionality tests for ContinuousHahn."""

    def test_forward_pass(self, sample_input):
        """Test forward pass produces valid output."""
        layer = ContinuousHahn(degree=5, units=32)
        output = layer(sample_input)
        assert output.shape == (8, 32)
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))

    def test_default_parameters(self):
        """Test default parameter values."""
        layer = ContinuousHahn(degree=5, units=32)
        assert layer.a_init == 1.0
        assert layer.b_init == 1.0
        assert layer.c_init == 1.0
        assert layer.d_init == 1.0

    def test_custom_parameters(self, sample_input):
        """Test with custom parameters."""
        layer = ContinuousHahn(degree=5, units=32, a=2.0, b=1.5, c=2.0, d=1.5)
        output = layer(sample_input)
        assert output.shape == (8, 32)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_trainable_params(self, sample_input):
        """Test trainable parameters."""
        layer = ContinuousHahn(degree=5, units=32, trainable_params=True)
        layer(sample_input)

        trainable_vars = [v for v in layer.trainable_variables if 'logits' in v.name]
        assert len(trainable_vars) == 4  # a, b, c, d

    def test_degrees_1_to_10(self, sample_input):
        """Test various polynomial degrees."""
        for degree in [1, 2, 5, 10]:
            layer = ContinuousHahn(degree=degree, units=32)
            output = layer(sample_input)
            assert output.shape == (8, 32)
            assert not tf.reduce_any(tf.math.is_nan(output))


class TestContinuousDualHahnBasic:
    """Basic functionality tests for ContinuousDualHahn."""

    def test_forward_pass(self, sample_input):
        """Test forward pass produces valid output."""
        layer = ContinuousDualHahn(degree=5, units=32)
        output = layer(sample_input)
        assert output.shape == (8, 32)
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))

    def test_default_parameters(self):
        """Test default parameter values."""
        layer = ContinuousDualHahn(degree=5, units=32)
        assert layer.a_init == 1.0
        assert layer.b_init == 1.0
        assert layer.c_init == 1.0

    def test_custom_parameters(self, sample_input):
        """Test with custom parameters."""
        layer = ContinuousDualHahn(degree=5, units=32, a=2.0, b=1.5, c=0.8)
        output = layer(sample_input)
        assert output.shape == (8, 32)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_trainable_params(self, sample_input):
        """Test trainable parameters."""
        layer = ContinuousDualHahn(degree=5, units=32, trainable_params=True)
        layer(sample_input)

        trainable_vars = [v for v in layer.trainable_variables if 'logits' in v.name]
        assert len(trainable_vars) == 3  # a, b, c


class TestDualHahnBasic:
    """Basic functionality tests for DualHahn."""

    def test_forward_pass(self, sample_input):
        """Test forward pass produces valid output."""
        layer = DualHahn(degree=5, units=32, N=10)
        output = layer(sample_input)
        assert output.shape == (8, 32)
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))

    def test_default_parameters(self):
        """Test default parameter values."""
        layer = DualHahn(degree=5, units=32, N=10)
        assert layer.gamma_init == 1.0
        assert layer.delta_init == 1.0
        assert layer.N == 10

    def test_custom_parameters(self, sample_input):
        """Test with custom parameters."""
        layer = DualHahn(degree=5, units=32, gamma=2.0, delta=1.5, N=15)
        output = layer(sample_input)
        assert output.shape == (8, 32)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_N_must_be_geq_degree(self):
        """Test that N >= degree is enforced."""
        with pytest.raises(ValueError, match="N=3 must be >= degree=5"):
            DualHahn(degree=5, units=32, N=3)

    def test_trainable_params(self, sample_input):
        """Test trainable parameters."""
        layer = DualHahn(degree=5, units=32, N=10, trainable_params=True)
        layer(sample_input)

        trainable_vars = [v for v in layer.trainable_variables if 'logits' in v.name]
        assert len(trainable_vars) == 2  # gamma, delta


class TestStieltjesWigertBasic:
    """Basic functionality tests for StieltjesWigert."""

    def test_forward_pass(self, sample_input):
        """Test forward pass produces valid output."""
        layer = StieltjesWigert(degree=5, units=32, q=0.5)
        output = layer(sample_input)
        assert output.shape == (8, 32)
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))

    def test_default_parameters(self):
        """Test default parameter values."""
        layer = StieltjesWigert(degree=5, units=32)
        assert layer.q_init == 0.5

    def test_custom_q(self, sample_input):
        """Test with custom q parameter."""
        for q in [0.1, 0.3, 0.7, 0.9]:
            layer = StieltjesWigert(degree=5, units=32, q=q)
            output = layer(sample_input)
            assert output.shape == (8, 32)
            assert not tf.reduce_any(tf.math.is_nan(output))

    def test_q_bounds_validation(self):
        """Test q must be in (0, 1)."""
        with pytest.raises(ValueError, match="q must be in"):
            StieltjesWigert(degree=5, units=32, q=0.0)
        with pytest.raises(ValueError, match="q must be in"):
            StieltjesWigert(degree=5, units=32, q=1.0)
        with pytest.raises(ValueError, match="q must be in"):
            StieltjesWigert(degree=5, units=32, q=1.5)

    def test_trainable_params(self, sample_input):
        """Test trainable parameters."""
        layer = StieltjesWigert(degree=5, units=32, trainable_params=True)
        layer(sample_input)

        trainable_vars = [v for v in layer.trainable_variables if 'logits' in v.name]
        assert len(trainable_vars) == 1  # q


# =============================================================================
# Mathematical Properties Tests
# =============================================================================

class TestMathematicalProperties:
    """Test mathematical properties of the polynomial families."""

    def test_continuous_hahn_p0_is_one(self):
        """Test P_0(x) = 1 for ContinuousHahn."""
        layer = ContinuousHahn(degree=5, units=32)
        x = tf.constant([[0.5, 1.0, -0.5]], dtype=tf.float32)
        layer(x)  # Build

        basis = layer.pseudo_vandermonde(x)
        # P_0 should be 1
        np.testing.assert_allclose(basis[..., 0].numpy(), 1.0, atol=1e-5)

    def test_continuous_dual_hahn_s0_is_one(self):
        """Test S_0(x) = 1 for ContinuousDualHahn."""
        layer = ContinuousDualHahn(degree=5, units=32)
        x = tf.constant([[0.5, 1.0, 2.0]], dtype=tf.float32)
        layer(x)  # Build

        basis = layer.pseudo_vandermonde(x)
        np.testing.assert_allclose(basis[..., 0].numpy(), 1.0, atol=1e-5)

    def test_dual_hahn_r0_is_one(self):
        """Test R_0(x) = 1 for DualHahn."""
        layer = DualHahn(degree=5, units=32, N=10)
        x = tf.constant([[0.5, 1.0, 2.0]], dtype=tf.float32)
        layer(x)  # Build

        basis = layer.pseudo_vandermonde(x)
        np.testing.assert_allclose(basis[..., 0].numpy(), 1.0, atol=1e-5)

    def test_stieltjes_wigert_s0_is_one(self):
        """Test S_0(x) = 1 for StieltjesWigert."""
        layer = StieltjesWigert(degree=5, units=32, q=0.5)
        x = tf.constant([[0.5, 1.0, 2.0]], dtype=tf.float32)
        layer(x)  # Build

        basis = layer.pseudo_vandermonde(x)
        np.testing.assert_allclose(basis[..., 0].numpy(), 1.0, atol=1e-5)

    def test_continuous_dual_hahn_uses_x_squared(self, sample_input):
        """Test that ContinuousDualHahn uses x^2 argument."""
        layer = ContinuousDualHahn(degree=5, units=32)

        # Evaluate at x and -x
        x = tf.constant([[1.0, 2.0]], dtype=tf.float32)
        neg_x = tf.constant([[-1.0, -2.0]], dtype=tf.float32)

        layer(x)  # Build

        basis_pos = layer.pseudo_vandermonde(x)
        basis_neg = layer.pseudo_vandermonde(neg_x)

        # Since polynomials are in x^2, they should be equal for x and -x
        np.testing.assert_allclose(basis_pos.numpy(), basis_neg.numpy(), atol=1e-5)


# =============================================================================
# Gradient Flow Tests
# =============================================================================

class TestGradientFlow:
    """Test gradient flow through layers."""

    def test_continuous_hahn_gradient(self, sample_input):
        """Test gradient flow for ContinuousHahn."""
        layer = ContinuousHahn(degree=5, units=32, trainable_params=True)

        with tf.GradientTape() as tape:
            output = layer(sample_input)
            loss = tf.reduce_mean(output)

        grads = tape.gradient(loss, layer.trainable_variables)
        for g in grads:
            assert g is not None
            assert not tf.reduce_any(tf.math.is_nan(g))

    def test_continuous_dual_hahn_gradient(self, sample_input):
        """Test gradient flow for ContinuousDualHahn."""
        layer = ContinuousDualHahn(degree=5, units=32, trainable_params=True)

        with tf.GradientTape() as tape:
            output = layer(sample_input)
            loss = tf.reduce_mean(output)

        grads = tape.gradient(loss, layer.trainable_variables)
        for g in grads:
            assert g is not None
            assert not tf.reduce_any(tf.math.is_nan(g))

    def test_dual_hahn_gradient(self, sample_input):
        """Test gradient flow for DualHahn."""
        layer = DualHahn(degree=5, units=32, N=10, trainable_params=True)

        with tf.GradientTape() as tape:
            output = layer(sample_input)
            loss = tf.reduce_mean(output)

        grads = tape.gradient(loss, layer.trainable_variables)
        for g in grads:
            assert g is not None
            assert not tf.reduce_any(tf.math.is_nan(g))

    def test_stieltjes_wigert_gradient(self, sample_input):
        """Test gradient flow for StieltjesWigert."""
        layer = StieltjesWigert(degree=5, units=32, trainable_params=True)

        with tf.GradientTape() as tape:
            output = layer(sample_input)
            loss = tf.reduce_mean(output)

        grads = tape.gradient(loss, layer.trainable_variables)
        for g in grads:
            assert g is not None
            assert not tf.reduce_any(tf.math.is_nan(g))

    def test_model_training_step(self, sample_input):
        """Test training step with all layers."""
        for LayerClass in [ContinuousHahn, ContinuousDualHahn, StieltjesWigert]:
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(16,)),
                LayerClass(degree=3, units=8),
                tf.keras.layers.Dense(1),
            ])

            model.compile(optimizer='adam', loss='mse')
            targets = tf.random.uniform((8, 1))

            # Single training step should work
            loss = model.train_on_batch(sample_input, targets)
            assert not np.isnan(loss)


# =============================================================================
# XLA Compatibility Tests
# =============================================================================

class TestXLACompatibility:
    """Test XLA compilation compatibility."""

    @pytest.mark.parametrize("LayerClass,kwargs", [
        (ContinuousHahn, {}),
        (ContinuousDualHahn, {}),
        (DualHahn, {"N": 10}),
        (StieltjesWigert, {}),
    ])
    def test_xla_compilation(self, LayerClass, kwargs, sample_input):
        """Test XLA compilation for each layer."""
        layer = LayerClass(degree=5, units=32, **kwargs)

        @tf.function(jit_compile=True)
        def forward(x):
            return layer(x)

        output = forward(sample_input)
        assert output.shape == (8, 32)
        assert not tf.reduce_any(tf.math.is_nan(output))

    @pytest.mark.parametrize("LayerClass,kwargs", [
        (ContinuousHahn, {}),
        (ContinuousDualHahn, {}),
        (DualHahn, {"N": 10}),
        (StieltjesWigert, {}),
    ])
    def test_xla_gradient(self, LayerClass, kwargs, sample_input):
        """Test XLA compilation with gradients."""
        layer = LayerClass(degree=5, units=32, trainable_params=True, **kwargs)

        @tf.function(jit_compile=True)
        def train_step(x):
            with tf.GradientTape() as tape:
                output = layer(x)
                loss = tf.reduce_mean(tf.square(output))
            return tape.gradient(loss, layer.trainable_variables)

        grads = train_step(sample_input)
        for g in grads:
            if g is not None:
                assert not tf.reduce_any(tf.math.is_nan(g))


# =============================================================================
# Serialization Tests
# =============================================================================

class TestSerialization:
    """Test model serialization and deserialization."""

    @pytest.mark.parametrize("LayerClass,kwargs", [
        (ContinuousHahn, {"a": 1.5, "b": 2.0}),
        (ContinuousDualHahn, {"a": 1.5, "b": 2.0, "c": 0.8}),
        (DualHahn, {"gamma": 1.5, "delta": 2.0, "N": 15}),
        (StieltjesWigert, {"q": 0.7}),
    ])
    def test_get_config(self, LayerClass, kwargs, sample_input):
        """Test layer configuration serialization."""
        layer = LayerClass(degree=5, units=32, **kwargs)
        layer(sample_input)  # Build

        config = layer.get_config()

        # Verify key parameters are in config
        assert config["degree"] == 5
        assert config["units"] == 32

    @pytest.mark.parametrize("LayerClass,kwargs", [
        (ContinuousHahn, {}),
        (ContinuousDualHahn, {}),
        (DualHahn, {"N": 10}),
        (StieltjesWigert, {}),
    ])
    def test_from_config(self, LayerClass, kwargs, sample_input):
        """Test layer reconstruction from config."""
        layer1 = LayerClass(degree=5, units=32, **kwargs)
        layer1(sample_input)  # Build

        config = layer1.get_config()
        layer2 = LayerClass.from_config(config)
        layer2(sample_input)  # Build

        # Check that config parameters are preserved
        config2 = layer2.get_config()
        assert config["degree"] == config2["degree"]
        assert config["units"] == config2["units"]

        # Basis outputs should be identical (no learned coefficients involved)
        basis1 = layer1.pseudo_vandermonde(sample_input)
        basis2 = layer2.pseudo_vandermonde(sample_input)
        np.testing.assert_allclose(basis1.numpy(), basis2.numpy(), atol=1e-5)

    @pytest.mark.parametrize("LayerClass,kwargs", [
        (ContinuousHahn, {}),
        (ContinuousDualHahn, {}),
        (DualHahn, {"N": 10}),
        (StieltjesWigert, {}),
    ])
    def test_saved_model_roundtrip(self, LayerClass, kwargs, sample_input, tmp_path):
        """Test SavedModel export and import."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(16,)),
            LayerClass(degree=3, units=8, **kwargs),
        ])

        output_before = model(sample_input)

        # Save and reload using .keras extension
        save_path = str(tmp_path / f"{LayerClass.__name__}_model.keras")
        model.save(save_path)
        loaded_model = tf.keras.models.load_model(save_path)

        output_after = loaded_model(sample_input)
        np.testing.assert_allclose(output_before.numpy(), output_after.numpy(), atol=1e-5)


# =============================================================================
# High Degree Stability Tests
# =============================================================================

class TestHighDegreeStability:
    """Test numerical stability at higher degrees."""

    @pytest.mark.parametrize("LayerClass,kwargs", [
        (ContinuousHahn, {}),
        (ContinuousDualHahn, {}),  # Limited to degree 15 due to polynomial growth
        (DualHahn, {"N": 30}),
        (StieltjesWigert, {}),
    ])
    def test_degree_15(self, LayerClass, kwargs):
        """Test stability at degree 15."""
        x = tf.random.uniform((4, 8), -1.0, 1.0, seed=42)
        layer = LayerClass(degree=15, units=16, **kwargs)

        output = layer(x)
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))

    @pytest.mark.parametrize("LayerClass,kwargs", [
        (ContinuousHahn, {}),
        (ContinuousDualHahn, {}),
        (StieltjesWigert, {}),
    ])
    def test_moderate_x_values(self, LayerClass, kwargs):
        """Test stability for moderate x values."""
        x = tf.constant([[0.1, 0.5, 1.0, 2.0, 5.0]], dtype=tf.float32)
        layer = LayerClass(degree=10, units=8, **kwargs)

        output = layer(x)
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for layers in models."""

    def test_multi_layer_model(self, sample_input):
        """Test model with multiple Wilson class layers."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(16,)),
            ContinuousHahn(degree=3, units=32),
            tf.keras.layers.LayerNormalization(),
            ContinuousDualHahn(degree=3, units=16),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(1),
        ])

        output = model(sample_input)
        assert output.shape == (8, 1)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_mixed_polynomial_model(self, sample_input):
        """Test model mixing Wilson class with other polynomials."""
        from arnold.layers import Legendre

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(16,)),
            ContinuousHahn(degree=3, units=32),
            tf.keras.layers.LayerNormalization(),
            Legendre(degree=3, units=16),
            tf.keras.layers.Dense(1),
        ])

        output = model(sample_input)
        assert output.shape == (8, 1)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_training_convergence(self):
        """Test that a simple model can train."""
        # Simple regression task
        x_train = tf.random.uniform((64, 8), -1.0, 1.0, seed=42)
        y_train = tf.reduce_sum(x_train, axis=-1, keepdims=True)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(8,)),
            ContinuousDualHahn(degree=3, units=16),
            tf.keras.layers.Dense(1),
        ])

        model.compile(optimizer='adam', loss='mse')

        initial_loss = model.evaluate(x_train, y_train, verbose=0)
        model.fit(x_train, y_train, epochs=5, verbose=0)
        final_loss = model.evaluate(x_train, y_train, verbose=0)

        # Loss should decrease
        assert final_loss < initial_loss


# =============================================================================
# Edge Cases and Special Values
# =============================================================================

class TestEdgeCases:
    """Test edge cases and special input values."""

    def test_zero_input(self):
        """Test behavior at x=0."""
        x = tf.zeros((2, 4), dtype=tf.float32)

        for LayerClass, kwargs in [
            (ContinuousHahn, {}),
            (ContinuousDualHahn, {}),
            (DualHahn, {"N": 10}),
            (StieltjesWigert, {}),
        ]:
            layer = LayerClass(degree=5, units=8, **kwargs)
            output = layer(x)
            assert not tf.reduce_any(tf.math.is_nan(output))

    def test_small_input(self):
        """Test behavior for very small inputs."""
        x = tf.constant([[1e-6, 1e-8, 1e-10, 1e-12]], dtype=tf.float32)

        for LayerClass, kwargs in [
            (ContinuousHahn, {}),
            (ContinuousDualHahn, {}),
            (StieltjesWigert, {}),
        ]:
            layer = LayerClass(degree=5, units=8, **kwargs)
            output = layer(x)
            assert not tf.reduce_any(tf.math.is_nan(output))

    def test_negative_input(self):
        """Test behavior for negative inputs."""
        x = tf.constant([[-0.5, -1.0, -2.0, -5.0]], dtype=tf.float32)

        for LayerClass, kwargs in [
            (ContinuousHahn, {}),
            (ContinuousDualHahn, {}),  # Uses x^2, so should work
            (StieltjesWigert, {}),
        ]:
            layer = LayerClass(degree=5, units=8, **kwargs)
            output = layer(x)
            assert not tf.reduce_any(tf.math.is_nan(output))

    def test_degree_1(self):
        """Test minimum degree (1)."""
        x = tf.random.uniform((4, 8), -1.0, 1.0, seed=42)

        for LayerClass, kwargs in [
            (ContinuousHahn, {}),
            (ContinuousDualHahn, {}),
            (DualHahn, {"N": 5}),
            (StieltjesWigert, {}),
        ]:
            layer = LayerClass(degree=1, units=8, **kwargs)
            output = layer(x)
            assert output.shape == (4, 8)
            assert not tf.reduce_any(tf.math.is_nan(output))


# =============================================================================
# Dtype Preservation Tests
# =============================================================================

class TestDtypePreservation:
    """Test dtype handling."""

    @pytest.mark.parametrize("LayerClass,kwargs", [
        (ContinuousHahn, {}),
        (ContinuousDualHahn, {}),
        (DualHahn, {"N": 10}),
        (StieltjesWigert, {}),
    ])
    def test_float32_preserved(self, LayerClass, kwargs):
        """Test that float32 input produces float32 output."""
        x = tf.random.uniform((4, 8), -1.0, 1.0, dtype=tf.float32)
        layer = LayerClass(degree=5, units=16, **kwargs)
        output = layer(x)
        assert output.dtype == tf.float32

    @pytest.mark.parametrize("LayerClass,kwargs", [
        (ContinuousHahn, {}),
        (ContinuousDualHahn, {}),
        (DualHahn, {"N": 10}),
        (StieltjesWigert, {}),
    ])
    def test_float64_input(self, LayerClass, kwargs):
        """Test float64 input handling."""
        x = tf.random.uniform((4, 8), -1.0, 1.0, dtype=tf.float64)
        layer = LayerClass(degree=5, units=16, **kwargs)
        output = layer(x)
        # Should preserve float64 or convert back appropriately
        assert output.dtype in [tf.float32, tf.float64]
        assert not tf.reduce_any(tf.math.is_nan(output))
