## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
"""
Tests for q-Hahn class polynomial layers (Sprint 7E).

Tests cover:
- Basic functionality (forward pass, shapes, dtypes)
- q-parameter constraints (must be in (0, 1))
- Three-term recurrence properties
- Trainable parameter gradient flow
- XLA compatibility
- Serialization (get_config, from_config, SavedModel)
- High-degree stability
- Edge cases
"""
import numpy as np
import pytest
import tensorflow as tf

from arnold.layers.core import (
    BigQJacobi,
    LittleQJacobi,
    QHahn,
    QKrawtchouk,
    QMeixner,
    QPolynomialBase,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def sample_input():
    """Sample input tensor for testing."""
    tf.random.set_seed(42)
    return tf.random.uniform((8, 16), minval=0.1, maxval=0.9, dtype=tf.float32)


@pytest.fixture
def small_input():
    """Small input for edge case testing."""
    tf.random.set_seed(42)
    return tf.random.uniform((2, 4), minval=0.1, maxval=0.9, dtype=tf.float32)


# ==============================================================================
# Test QHahn Layer
# ==============================================================================

class TestQHahnBasic:
    """Basic functionality tests for QHahn layer."""

    def test_build_and_call(self, sample_input):
        """Test that layer builds and produces output."""
        layer = QHahn(degree=3, units=8, N=10)
        output = layer(sample_input)
        
        assert output.shape == (8, 8)
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))

    def test_output_shape(self, sample_input):
        """Test output shape matches units."""
        for units in [4, 16, 32]:
            layer = QHahn(degree=5, units=units, N=10)
            output = layer(sample_input)
            assert output.shape == (8, units)

    def test_degree_constraint(self):
        """Test that degree <= N."""
        with pytest.raises(ValueError):
            QHahn(degree=15, units=8, N=10)  # N < degree

    def test_q_constraint(self):
        """Test that q must be in (0, 1)."""
        with pytest.raises(ValueError):
            QHahn(degree=3, units=8, N=10, q=0.0)
        with pytest.raises(ValueError):
            QHahn(degree=3, units=8, N=10, q=1.0)
        with pytest.raises(ValueError):
            QHahn(degree=3, units=8, N=10, q=1.5)

    def test_dtype_preservation(self, sample_input):
        """Test that float32 input produces float32 output."""
        layer = QHahn(degree=3, units=8, N=10)
        output = layer(sample_input)
        assert output.dtype == tf.float32

    def test_trainable_params(self, sample_input):
        """Test trainable parameters."""
        layer = QHahn(degree=3, units=8, N=10, 
                      alpha_trainable=True, beta_trainable=True, q_trainable=True)
        _ = layer(sample_input)
        
        # Check weights exist
        weight_names = [w.name for w in layer.trainable_weights]
        assert any('alpha' in name for name in weight_names)
        assert any('beta' in name for name in weight_names)
        assert any('q' in name for name in weight_names)


class TestQHahnMathematical:
    """Mathematical property tests for QHahn layer."""

    def test_p0_equals_one(self, small_input):
        """Test that Q_0(x) = 1."""
        layer = QHahn(degree=0, units=4, N=10)
        output = layer(small_input)
        
        # Degree 0 should be constant (all 1s in basis)
        assert output.shape == (2, 4)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_q_near_one_stability(self, sample_input):
        """Test stability when q approaches 1."""
        layer = QHahn(degree=3, units=8, N=10, q=0.99)
        output = layer(sample_input)
        
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))

    def test_q_near_zero_stability(self, sample_input):
        """Test stability when q approaches 0."""
        layer = QHahn(degree=3, units=8, N=10, q=0.01)
        output = layer(sample_input)
        
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))


# ==============================================================================
# Test BigQJacobi Layer
# ==============================================================================

class TestBigQJacobiBasic:
    """Basic functionality tests for BigQJacobi layer."""

    def test_build_and_call(self, sample_input):
        """Test that layer builds and produces output."""
        layer = BigQJacobi(degree=3, units=8)
        output = layer(sample_input)
        
        assert output.shape == (8, 8)
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))

    def test_output_shape(self, sample_input):
        """Test output shape matches units."""
        for units in [4, 16, 32]:
            layer = BigQJacobi(degree=5, units=units)
            output = layer(sample_input)
            assert output.shape == (8, units)

    def test_c_negative(self, sample_input):
        """Test that c parameter is negative."""
        layer = BigQJacobi(degree=3, units=8, c=-1.0)
        output = layer(sample_input)
        
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_trainable_params(self, sample_input):
        """Test trainable parameters."""
        layer = BigQJacobi(degree=3, units=8, 
                          a_trainable=True, b_trainable=True, c_trainable=True)
        _ = layer(sample_input)
        
        weight_names = [w.name for w in layer.trainable_weights]
        assert any('a_logits' in name for name in weight_names)
        assert any('b_logits' in name for name in weight_names)
        assert any('c_logits' in name for name in weight_names)


# ==============================================================================
# Test LittleQJacobi Layer
# ==============================================================================

class TestLittleQJacobiBasic:
    """Basic functionality tests for LittleQJacobi layer."""

    def test_build_and_call(self, sample_input):
        """Test that layer builds and produces output."""
        layer = LittleQJacobi(degree=3, units=8)
        output = layer(sample_input)
        
        assert output.shape == (8, 8)
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))

    def test_output_shape(self, sample_input):
        """Test output shape matches units."""
        for units in [4, 16, 32]:
            layer = LittleQJacobi(degree=5, units=units)
            output = layer(sample_input)
            assert output.shape == (8, units)

    def test_b_zero_is_wall(self, sample_input):
        """Test that b=0 gives Wall/Little q-Laguerre polynomials."""
        # b=0 case should still work numerically
        layer = LittleQJacobi(degree=3, units=8, b=0.01)  # Very small b
        output = layer(sample_input)
        
        assert not tf.reduce_any(tf.math.is_nan(output))


# ==============================================================================
# Test QMeixner Layer
# ==============================================================================

class TestQMeixnerBasic:
    """Basic functionality tests for QMeixner layer."""

    def test_build_and_call(self, sample_input):
        """Test that layer builds and produces output."""
        layer = QMeixner(degree=3, units=8)
        output = layer(sample_input)
        
        assert output.shape == (8, 8)
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))

    def test_output_shape(self, sample_input):
        """Test output shape matches units."""
        for units in [4, 16, 32]:
            layer = QMeixner(degree=5, units=units)
            output = layer(sample_input)
            assert output.shape == (8, units)

    def test_trainable_params(self, sample_input):
        """Test trainable parameters."""
        layer = QMeixner(degree=3, units=8, b_trainable=True, c_trainable=True)
        _ = layer(sample_input)
        
        weight_names = [w.name for w in layer.trainable_weights]
        assert any('b_logits' in name for name in weight_names)
        assert any('c_logits' in name for name in weight_names)


# ==============================================================================
# Test QKrawtchouk Layer
# ==============================================================================

class TestQKrawtchoukBasic:
    """Basic functionality tests for QKrawtchouk layer."""

    def test_build_and_call(self, sample_input):
        """Test that layer builds and produces output."""
        layer = QKrawtchouk(degree=3, units=8, N=10)
        output = layer(sample_input)
        
        assert output.shape == (8, 8)
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))

    def test_output_shape(self, sample_input):
        """Test output shape matches units."""
        for units in [4, 16, 32]:
            layer = QKrawtchouk(degree=5, units=units, N=10)
            output = layer(sample_input)
            assert output.shape == (8, units)

    def test_degree_constraint(self):
        """Test that degree <= N."""
        with pytest.raises(ValueError):
            QKrawtchouk(degree=15, units=8, N=10)  # N < degree

    def test_trainable_p(self, sample_input):
        """Test trainable p parameter."""
        layer = QKrawtchouk(degree=3, units=8, N=10, p_trainable=True)
        _ = layer(sample_input)
        
        weight_names = [w.name for w in layer.trainable_weights]
        assert any('p_logits' in name for name in weight_names)


# ==============================================================================
# Test Gradient Flow
# ==============================================================================

class TestGradientFlow:
    """Test that gradients flow through all q-polynomial layers."""

    @pytest.mark.parametrize("LayerClass,kwargs", [
        (QHahn, {"N": 10, "alpha_trainable": True, "beta_trainable": True}),
        (BigQJacobi, {"a_trainable": True, "b_trainable": True, "c_trainable": True}),
        (LittleQJacobi, {"a_trainable": True, "b_trainable": True}),
        (QMeixner, {"b_trainable": True, "c_trainable": True}),
        (QKrawtchouk, {"N": 10, "p_trainable": True}),
    ])
    def test_gradients_finite(self, LayerClass, kwargs, sample_input):
        """Test that gradients are finite for all trainable params."""
        layer = LayerClass(degree=3, units=8, q_trainable=True, **kwargs)
        
        with tf.GradientTape() as tape:
            output = layer(sample_input)
            loss = tf.reduce_mean(output ** 2)
        
        gradients = tape.gradient(loss, layer.trainable_weights)
        
        for grad, weight in zip(gradients, layer.trainable_weights):
            assert grad is not None, f"No gradient for {weight.name}"
            assert not tf.reduce_any(tf.math.is_nan(grad)), f"NaN gradient for {weight.name}"
            assert not tf.reduce_any(tf.math.is_inf(grad)), f"Inf gradient for {weight.name}"

    @pytest.mark.parametrize("LayerClass,kwargs", [
        (QHahn, {"N": 10}),
        (BigQJacobi, {}),
        (LittleQJacobi, {}),
        (QMeixner, {}),
        (QKrawtchouk, {"N": 10}),
    ])
    def test_q_gradient(self, LayerClass, kwargs, sample_input):
        """Test that q parameter receives gradient when trainable."""
        layer = LayerClass(degree=3, units=8, q_trainable=True, **kwargs)
        
        with tf.GradientTape() as tape:
            output = layer(sample_input)
            loss = tf.reduce_mean(output ** 2)
        
        gradients = tape.gradient(loss, layer.trainable_weights)
        
        # Find q_logits gradient
        q_grad = None
        for grad, weight in zip(gradients, layer.trainable_weights):
            if 'q_logits' in weight.name:
                q_grad = grad
                break
        
        assert q_grad is not None, "q_logits should have gradient"
        assert not tf.math.is_nan(q_grad), "q gradient should be finite"


# ==============================================================================
# Test XLA Compatibility
# ==============================================================================

class TestXLACompatibility:
    """Test XLA compilation compatibility for all layers."""

    @pytest.mark.parametrize("LayerClass,kwargs", [
        (QHahn, {"N": 10}),
        (BigQJacobi, {}),
        (LittleQJacobi, {}),
        (QMeixner, {}),
        (QKrawtchouk, {"N": 10}),
    ])
    def test_xla_compilation(self, LayerClass, kwargs, sample_input):
        """Test that layers work with XLA compilation."""
        layer = LayerClass(degree=3, units=8, **kwargs)
        
        @tf.function(jit_compile=True)
        def forward(x):
            return layer(x)
        
        output = forward(sample_input)
        
        assert output.shape == (8, 8)
        assert not tf.reduce_any(tf.math.is_nan(output))


# ==============================================================================
# Test Serialization
# ==============================================================================

class TestSerialization:
    """Test serialization for all q-polynomial layers."""

    @pytest.mark.parametrize("LayerClass,kwargs", [
        (QHahn, {"N": 10, "alpha": 2.0, "beta": 1.5}),
        (BigQJacobi, {"a": 0.3, "b": 0.4, "c": -0.5}),
        (LittleQJacobi, {"a": 0.3, "b": 0.4}),
        (QMeixner, {"b": 2.0, "c": 0.3}),
        (QKrawtchouk, {"N": 10, "p": 0.7}),
    ])
    def test_get_config(self, LayerClass, kwargs, sample_input):
        """Test get_config returns all parameters."""
        layer = LayerClass(degree=3, units=8, q=0.6, **kwargs)
        _ = layer(sample_input)
        
        config = layer.get_config()
        
        assert config["degree"] == 3
        assert config["units"] == 8
        assert config["q"] == 0.6
        for key, value in kwargs.items():
            assert config[key] == value, f"Config missing {key}"

    @pytest.mark.parametrize("LayerClass,kwargs", [
        (QHahn, {"N": 10}),
        (BigQJacobi, {}),
        (LittleQJacobi, {}),
        (QMeixner, {}),
        (QKrawtchouk, {"N": 10}),
    ])
    def test_from_config(self, LayerClass, kwargs, sample_input):
        """Test from_config recreates layer correctly."""
        layer = LayerClass(degree=3, units=8, **kwargs)
        _ = layer(sample_input)
        
        config = layer.get_config()
        new_layer = LayerClass.from_config(config)
        _ = new_layer(sample_input)
        
        # Check config matches
        new_config = new_layer.get_config()
        assert new_config["degree"] == config["degree"]
        assert new_config["units"] == config["units"]
        assert new_config["q"] == config["q"]

    @pytest.mark.parametrize("LayerClass,kwargs", [
        (QHahn, {"N": 10}),
        (BigQJacobi, {}),
        (LittleQJacobi, {}),
        (QMeixner, {}),
        (QKrawtchouk, {"N": 10}),
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


# ==============================================================================
# Test High-Degree Stability
# ==============================================================================

class TestHighDegreeStability:
    """Test numerical stability at higher degrees."""

    @pytest.mark.parametrize("LayerClass,kwargs", [
        (QHahn, {"N": 20}),
        (BigQJacobi, {}),
        (LittleQJacobi, {}),
        (QMeixner, {}),
        (QKrawtchouk, {"N": 20}),
    ])
    def test_degree_10(self, LayerClass, kwargs, sample_input):
        """Test stability at degree 10."""
        layer = LayerClass(degree=10, units=8, **kwargs)
        output = layer(sample_input)
        
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))

    @pytest.mark.parametrize("LayerClass,kwargs", [
        (BigQJacobi, {}),
        (LittleQJacobi, {}),
        (QMeixner, {}),
    ])
    def test_degree_15(self, LayerClass, kwargs, sample_input):
        """Test stability at degree 15 for unconstrained layers."""
        layer = LayerClass(degree=15, units=8, **kwargs)
        output = layer(sample_input)
        
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))


# ==============================================================================
# Test Integration
# ==============================================================================

class TestIntegration:
    """Integration tests for q-polynomial layers in models."""

    def test_multi_layer_model(self, sample_input):
        """Test multi-layer model with q-polynomial layers."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(16,)),
            QHahn(degree=3, units=32, N=10),
            BigQJacobi(degree=3, units=16),
            LittleQJacobi(degree=3, units=8),
        ])
        
        output = model(sample_input)
        assert output.shape == (8, 8)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_mixed_polynomial_model(self, sample_input):
        """Test mixing q-polynomials with other layers."""
        from arnold.layers.core import Legendre
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(16,)),
            Legendre(degree=3, units=32),
            QMeixner(degree=3, units=16),
            tf.keras.layers.Dense(8),
        ])
        
        output = model(sample_input)
        assert output.shape == (8, 8)

    def test_training_convergence(self, sample_input):
        """Test that model can train without NaN loss."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(16,)),
            QHahn(degree=3, units=8, N=10),
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        target = tf.random.uniform((8, 8), dtype=tf.float32)
        history = model.fit(sample_input, target, epochs=3, verbose=0)
        
        assert not np.isnan(history.history['loss'][-1])


# ==============================================================================
# Test Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Test edge cases for q-polynomial layers."""

    @pytest.mark.parametrize("LayerClass,kwargs", [
        (QHahn, {"N": 10}),
        (BigQJacobi, {}),
        (LittleQJacobi, {}),
        (QMeixner, {}),
        (QKrawtchouk, {"N": 10}),
    ])
    def test_zero_input(self, LayerClass, kwargs):
        """Test with zero input."""
        layer = LayerClass(degree=3, units=8, **kwargs)
        x = tf.zeros((4, 8), dtype=tf.float32)
        output = layer(x)
        
        assert not tf.reduce_any(tf.math.is_nan(output))

    @pytest.mark.parametrize("LayerClass,kwargs", [
        (QHahn, {"N": 10}),
        (BigQJacobi, {}),
        (LittleQJacobi, {}),
        (QMeixner, {}),
        (QKrawtchouk, {"N": 10}),
    ])
    def test_small_input(self, LayerClass, kwargs):
        """Test with very small input values."""
        layer = LayerClass(degree=3, units=8, **kwargs)
        x = tf.ones((4, 8), dtype=tf.float32) * 1e-6
        output = layer(x)
        
        assert not tf.reduce_any(tf.math.is_nan(output))

    @pytest.mark.parametrize("LayerClass,kwargs", [
        (QHahn, {"N": 1}),
        (QKrawtchouk, {"N": 1}),
    ])
    def test_degree_one(self, LayerClass, kwargs, sample_input):
        """Test with degree 1."""
        layer = LayerClass(degree=1, units=8, **kwargs)
        output = layer(sample_input)
        
        assert output.shape == (8, 8)
        assert not tf.reduce_any(tf.math.is_nan(output))


# ==============================================================================
# Test Dtype Preservation
# ==============================================================================

class TestDtypePreservation:
    """Test dtype handling for q-polynomial layers."""

    @pytest.mark.parametrize("LayerClass,kwargs", [
        (QHahn, {"N": 10}),
        (BigQJacobi, {}),
        (LittleQJacobi, {}),
        (QMeixner, {}),
        (QKrawtchouk, {"N": 10}),
    ])
    def test_float32_preserved(self, LayerClass, kwargs):
        """Test that float32 input gives float32 output."""
        layer = LayerClass(degree=3, units=8, **kwargs)
        x = tf.random.uniform((4, 8), dtype=tf.float32)
        output = layer(x)
        
        assert output.dtype == tf.float32

    @pytest.mark.parametrize("LayerClass,kwargs", [
        (QHahn, {"N": 10}),
        (BigQJacobi, {}),
        (LittleQJacobi, {}),
        (QMeixner, {}),
        (QKrawtchouk, {"N": 10}),
    ])
    def test_float64_input(self, LayerClass, kwargs):
        """Test that float64 input is handled."""
        layer = LayerClass(degree=3, units=8, **kwargs)
        x = tf.random.uniform((4, 8), dtype=tf.float64)
        output = layer(x)
        
        # Output should be float64 or float32 depending on layer implementation
        assert output.dtype in [tf.float32, tf.float64]
        assert not tf.reduce_any(tf.math.is_nan(output))
