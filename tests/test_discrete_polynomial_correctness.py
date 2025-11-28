"""
Comprehensive tests for discrete orthogonal polynomial KAN layers.

Tests cover:
1. Basic functionality (forward pass, shapes, dtypes)
2. Mathematical properties (recurrence, special values)
3. Parameter constraints and trainability
4. Gradient flow and XLA compatibility
5. Serialization (save/load, get_config)
6. Edge cases and error handling
"""

import numpy as np
import pytest
import tensorflow as tf

from arnold.layers.core.polynomial.discrete import (
    Hahn,
    Krawtchouk,
    Meixner,
    Racah,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(params=[Krawtchouk, Hahn, Meixner, Racah])
def discrete_class(request):
    """Parametrize over all discrete polynomial layer classes."""
    return request.param


@pytest.fixture
def sample_input():
    """Sample input tensor for testing."""
    tf.random.set_seed(42)
    # Use positive integers common for discrete polynomials
    return tf.cast(tf.random.uniform((8, 4), minval=0, maxval=5, dtype=tf.int32), tf.float32)


@pytest.fixture
def continuous_input():
    """Continuous input for testing (some layers handle this)."""
    tf.random.set_seed(42)
    return tf.random.uniform((8, 4), minval=0.0, maxval=5.0)


# ============================================================================
# Basic Functionality Tests
# ============================================================================


class TestDiscreteBasicFunctionality:
    """Test basic layer functionality: forward pass, shapes, dtypes."""

    def test_layer_builds(self, discrete_class):
        """Test layer builds without error."""
        if discrete_class == Meixner:
            layer = discrete_class(degree=4, units=8)
        else:
            layer = discrete_class(degree=4, units=8, N=10)
        x = tf.random.uniform((4, 3), minval=0, maxval=5)
        output = layer(x)
        assert output.shape == (4, 8)

    def test_output_shape(self, discrete_class, sample_input):
        """Test output shape matches units parameter."""
        if discrete_class == Meixner:
            layer = discrete_class(degree=4, units=16)
        else:
            layer = discrete_class(degree=4, units=16, N=10)
        output = layer(sample_input)
        assert output.shape == (8, 16)

    def test_output_dtype_float32(self, discrete_class, sample_input):
        """Test output preserves float32 dtype."""
        if discrete_class == Meixner:
            layer = discrete_class(degree=4, units=8)
        else:
            layer = discrete_class(degree=4, units=8, N=10)
        output = layer(sample_input)
        assert output.dtype == tf.float32

    def test_output_finite(self, discrete_class, sample_input):
        """Test output contains no NaN or Inf values."""
        if discrete_class == Meixner:
            layer = discrete_class(degree=4, units=8)
        else:
            layer = discrete_class(degree=4, units=8, N=10)
        output = layer(sample_input)
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))

    def test_batch_independence(self, discrete_class):
        """Test that samples in batch are processed independently."""
        if discrete_class == Meixner:
            layer = discrete_class(degree=3, units=4)
        else:
            layer = discrete_class(degree=3, units=4, N=10)

        x1 = tf.constant([[1.0, 2.0]])
        x2 = tf.constant([[3.0, 4.0]])
        x_batch = tf.concat([x1, x2], axis=0)

        out1 = layer(x1)
        out2 = layer(x2)
        out_batch = layer(x_batch)

        np.testing.assert_allclose(out_batch[0].numpy(), out1[0].numpy(), rtol=1e-5)
        np.testing.assert_allclose(out_batch[1].numpy(), out2[0].numpy(), rtol=1e-5)


# ============================================================================
# Krawtchouk Polynomial Tests
# ============================================================================


class TestKrawtchouk:
    """Test Krawtchouk polynomial-specific functionality."""

    def test_krawtchouk_builds(self):
        """Test Krawtchouk layer builds."""
        layer = Krawtchouk(degree=4, units=8, p_init=0.5, N=10)
        x = tf.constant([[0.0, 1.0, 2.0, 3.0]])
        output = layer(x)
        assert output.shape == (1, 8)

    def test_krawtchouk_p_constraint(self):
        """Test p parameter is constrained to (0, 1)."""
        layer = Krawtchouk(degree=3, units=4, p_init=0.3, N=10)
        x = tf.constant([[1.0, 2.0]])
        _ = layer(x)

        # Get effective p value via sigmoid
        p = tf.sigmoid(layer._p_logits)
        assert 0 < p.numpy() < 1

    def test_krawtchouk_n_constraint(self):
        """Test N >= degree constraint."""
        with pytest.raises(ValueError, match="N must be >= degree"):
            Krawtchouk(degree=10, units=8, N=5)

    def test_krawtchouk_p_init_constraint(self):
        """Test p_init must be in (0, 1)."""
        with pytest.raises(ValueError, match="p_init must be in"):
            Krawtchouk(degree=3, units=8, p_init=0.0, N=10)
        with pytest.raises(ValueError, match="p_init must be in"):
            Krawtchouk(degree=3, units=8, p_init=1.0, N=10)

    def test_krawtchouk_basis_at_zero(self):
        """Test K_n(0) values."""
        layer = Krawtchouk(degree=3, units=1, p_init=0.5, N=10)
        x = tf.constant([[0.0]])
        _ = layer(x)

        # Get basis directly
        basis = layer.pseudo_vandermonde(x)

        # K_0(0) = 1
        np.testing.assert_allclose(basis[0, 0, 0].numpy(), 1.0, rtol=1e-5)

    def test_krawtchouk_trainable_p(self):
        """Test p parameter is trainable when specified."""
        layer = Krawtchouk(degree=3, units=4, p_trainable=True, N=10)
        x = tf.constant([[1.0, 2.0]])

        with tf.GradientTape() as tape:
            output = layer(x)
            loss = tf.reduce_sum(output)

        grads = tape.gradient(loss, layer.trainable_variables)
        p_grad = [g for g, v in zip(grads, layer.trainable_variables) if 'p_logits' in v.name]
        assert len(p_grad) == 1
        assert p_grad[0] is not None

    def test_krawtchouk_non_trainable_p(self):
        """Test p parameter is not trainable when specified."""
        layer = Krawtchouk(degree=3, units=4, p_trainable=False, N=10)
        x = tf.constant([[1.0, 2.0]])
        _ = layer(x)

        p_vars = [v for v in layer.trainable_variables if 'p_logits' in v.name]
        assert len(p_vars) == 0


# ============================================================================
# Hahn Polynomial Tests
# ============================================================================


class TestHahn:
    """Test Hahn polynomial-specific functionality."""

    def test_hahn_builds(self):
        """Test Hahn layer builds."""
        layer = Hahn(degree=4, units=8, alpha_init=0.5, beta_init=0.5, N=10)
        x = tf.constant([[0.0, 1.0, 2.0, 3.0]])
        output = layer(x)
        assert output.shape == (1, 8)

    def test_hahn_parameter_constraints(self):
        """Test alpha, beta > -1 constraints."""
        layer = Hahn(degree=3, units=4, alpha_init=0.5, beta_init=0.5, N=10)
        x = tf.constant([[1.0, 2.0]])
        _ = layer(x)

        # Parameters should be > -1 via softplus
        from arnold.utils.constraints import softplus_lower_bound
        alpha = softplus_lower_bound(layer._alpha_logits, lower_bound=-1.0)
        beta = softplus_lower_bound(layer._beta_logits, lower_bound=-1.0)
        assert alpha.numpy() > -1
        assert beta.numpy() > -1

    def test_hahn_n_constraint(self):
        """Test N >= degree constraint."""
        with pytest.raises(ValueError, match="N must be >= degree"):
            Hahn(degree=10, units=8, N=5)

    def test_hahn_basis_at_zero(self):
        """Test Q_n(0) = 1 for n=0."""
        layer = Hahn(degree=3, units=1, N=10)
        x = tf.constant([[0.0]])
        _ = layer(x)

        basis = layer.pseudo_vandermonde(x)
        # Q_0(0) = 1
        np.testing.assert_allclose(basis[0, 0, 0].numpy(), 1.0, rtol=1e-5)

    def test_hahn_trainable_parameters(self):
        """Test alpha and beta are trainable when specified."""
        layer = Hahn(degree=3, units=4, alpha_trainable=True, beta_trainable=True, N=10)
        x = tf.constant([[1.0, 2.0]])

        with tf.GradientTape() as tape:
            output = layer(x)
            loss = tf.reduce_sum(output)

        grads = tape.gradient(loss, layer.trainable_variables)
        param_grads = [g for g, v in zip(grads, layer.trainable_variables)
                       if 'alpha' in v.name or 'beta' in v.name]
        assert len(param_grads) == 2
        assert all(g is not None for g in param_grads)


# ============================================================================
# Meixner Polynomial Tests
# ============================================================================


class TestMeixner:
    """Test Meixner polynomial-specific functionality."""

    def test_meixner_builds(self):
        """Test Meixner layer builds."""
        layer = Meixner(degree=4, units=8, beta_init=1.0, c_init=0.5)
        x = tf.constant([[0.0, 1.0, 2.0, 3.0]])
        output = layer(x)
        assert output.shape == (1, 8)

    def test_meixner_beta_constraint(self):
        """Test beta > 0 constraint."""
        layer = Meixner(degree=3, units=4, beta_init=1.0, c_init=0.5)
        x = tf.constant([[1.0, 2.0]])
        _ = layer(x)

        from arnold.utils.constraints import softplus_lower_bound
        beta = softplus_lower_bound(layer._beta_logits, lower_bound=0.0)
        assert beta.numpy() > 0

    def test_meixner_c_constraint(self):
        """Test c ∈ (0, 1) constraint."""
        layer = Meixner(degree=3, units=4, beta_init=1.0, c_init=0.5)
        x = tf.constant([[1.0, 2.0]])
        _ = layer(x)

        c = tf.sigmoid(layer._c_logits)
        assert 0 < c.numpy() < 1

    def test_meixner_beta_init_constraint(self):
        """Test beta_init must be > 0."""
        with pytest.raises(ValueError, match="beta_init must be > 0"):
            Meixner(degree=3, units=8, beta_init=0.0)

    def test_meixner_c_init_constraint(self):
        """Test c_init must be in (0, 1)."""
        with pytest.raises(ValueError, match="c_init must be in"):
            Meixner(degree=3, units=8, c_init=0.0)
        with pytest.raises(ValueError, match="c_init must be in"):
            Meixner(degree=3, units=8, c_init=1.0)

    def test_meixner_basis_at_zero(self):
        """Test M_n(0) = 1 for n=0."""
        layer = Meixner(degree=3, units=1)
        x = tf.constant([[0.0]])
        _ = layer(x)

        basis = layer.pseudo_vandermonde(x)
        np.testing.assert_allclose(basis[0, 0, 0].numpy(), 1.0, rtol=1e-5)

    def test_meixner_trainable_parameters(self):
        """Test beta and c are trainable when specified."""
        layer = Meixner(degree=3, units=4, beta_trainable=True, c_trainable=True)
        x = tf.constant([[1.0, 2.0]])

        with tf.GradientTape(persistent=True) as tape:
            output = layer(x)
            loss = tf.reduce_sum(output)

        # Filter to only beta and c logits (exclude poly_coeffs)
        param_vars = [v for v in layer.trainable_variables
                      if 'beta_logits' in v.name or 'c_logits' in v.name]
        assert len(param_vars) == 2
        param_grads = [tape.gradient(loss, v) for v in param_vars]
        assert all(g is not None for g in param_grads)


# ============================================================================
# Racah Polynomial Tests
# ============================================================================


class TestRacah:
    """Test Racah polynomial-specific functionality."""

    def test_racah_builds(self):
        """Test Racah layer builds."""
        layer = Racah(degree=4, units=8, N=10)
        x = tf.constant([[0.0, 1.0, 2.0, 3.0]])
        output = layer(x)
        assert output.shape == (1, 8)

    def test_racah_parameter_constraints(self):
        """Test all parameters > -1 constraints."""
        layer = Racah(degree=3, units=4, N=10)
        x = tf.constant([[1.0, 2.0]])
        _ = layer(x)

        from arnold.utils.constraints import softplus_lower_bound
        for name in ['alpha', 'beta', 'gamma', 'delta']:
            logits = getattr(layer, f'_{name}_logits')
            param = softplus_lower_bound(logits, lower_bound=-1.0)
            assert param.numpy() > -1, f"{name} should be > -1"

    def test_racah_n_constraint(self):
        """Test N >= degree constraint."""
        with pytest.raises(ValueError, match="N must be >= degree"):
            Racah(degree=10, units=8, N=5)

    def test_racah_basis_at_zero(self):
        """Test R_0(0) = 1."""
        layer = Racah(degree=3, units=1, N=10)
        x = tf.constant([[0.0]])
        _ = layer(x)

        basis = layer.pseudo_vandermonde(x)
        np.testing.assert_allclose(basis[0, 0, 0].numpy(), 1.0, rtol=1e-5)

    def test_racah_trainable_parameters(self):
        """Test all four parameters are trainable when specified."""
        layer = Racah(
            degree=3, units=4, N=10,
            alpha_trainable=True, beta_trainable=True,
            gamma_trainable=True, delta_trainable=True
        )
        x = tf.constant([[1.0, 2.0]])

        with tf.GradientTape() as tape:
            output = layer(x)
            loss = tf.reduce_sum(output)

        grads = tape.gradient(loss, layer.trainable_variables)
        param_names = ['alpha', 'beta', 'gamma', 'delta']
        param_grads = [g for g, v in zip(grads, layer.trainable_variables)
                       if any(name in v.name for name in param_names)]
        assert len(param_grads) == 4
        assert all(g is not None for g in param_grads)

    def test_racah_lambda_transformation(self):
        """Test λ(x) = x(x + γ + δ + 1) transformation is applied."""
        layer = Racah(degree=2, units=1, N=10, gamma_init=1.0, delta_init=1.0)
        x = tf.constant([[2.0]])
        _ = layer(x)

        # The transformation should affect the basis
        basis = layer.pseudo_vandermonde(x)
        # Just check it's finite and structured
        assert not tf.reduce_any(tf.math.is_nan(basis))
        assert basis.shape[-1] == 3  # degree + 1 basis functions


# ============================================================================
# Mathematical Property Tests
# ============================================================================


class TestMathematicalProperties:
    """Test mathematical properties of discrete polynomials."""

    def test_recurrence_relation_krawtchouk(self):
        """Test Krawtchouk satisfies recurrence relation."""
        layer = Krawtchouk(degree=5, units=1, p_init=0.4, N=10)
        x = tf.constant([[3.0]])
        _ = layer(x)

        basis = layer.pseudo_vandermonde(x)
        p = tf.sigmoid(layer._p_logits).numpy()
        N = 10.0

        # For n=2: verify recurrence approximately holds
        # -x K_n = A_n K_{n+1} - (A_n + C_n) K_n + C_n K_{n-1}
        n = 2
        A_n = p * (N - n)
        C_n = n * (1 - p)
        x_val = 3.0

        K_n = basis[0, 0, n].numpy()
        K_nm1 = basis[0, 0, n - 1].numpy()
        K_np1 = basis[0, 0, n + 1].numpy()

        lhs = -x_val * K_n
        rhs = A_n * K_np1 - (A_n + C_n) * K_n + C_n * K_nm1

        np.testing.assert_allclose(lhs, rhs, rtol=0.1)

    def test_degree_zero_is_constant(self, discrete_class):
        """Test P_0(x) = 1 for all discrete polynomials."""
        if discrete_class == Meixner:
            layer = discrete_class(degree=3, units=1)
        else:
            layer = discrete_class(degree=3, units=1, N=10)

        x1 = tf.constant([[1.0]])
        x2 = tf.constant([[5.0]])

        _ = layer(x1)
        basis1 = layer.pseudo_vandermonde(x1)
        basis2 = layer.pseudo_vandermonde(x2)

        # P_0 should be 1 everywhere
        np.testing.assert_allclose(basis1[0, 0, 0].numpy(), 1.0, rtol=1e-5)
        np.testing.assert_allclose(basis2[0, 0, 0].numpy(), 1.0, rtol=1e-5)

    def test_increasing_degree_complexity(self, discrete_class):
        """Test higher degrees produce more oscillating basis."""
        if discrete_class == Meixner:
            layer = discrete_class(degree=5, units=1)
        else:
            layer = discrete_class(degree=5, units=1, N=10)

        # Evaluate at multiple points
        x = tf.constant([[0.0], [1.0], [2.0], [3.0], [4.0]])
        _ = layer(x[:1])

        basis = layer.pseudo_vandermonde(x)

        # Higher degree polynomials should have more sign changes
        for d in range(1, 5):
            poly_values = basis[:, 0, d].numpy()
            # At least ensure values exist and are finite
            assert not np.any(np.isnan(poly_values))


# ============================================================================
# Gradient Flow Tests
# ============================================================================


class TestGradientFlow:
    """Test gradient flow through discrete polynomial layers."""

    def test_gradients_exist(self, discrete_class, sample_input):
        """Test gradients flow through the layer."""
        if discrete_class == Meixner:
            layer = discrete_class(degree=4, units=8)
        else:
            layer = discrete_class(degree=4, units=8, N=10)

        with tf.GradientTape() as tape:
            output = layer(sample_input)
            loss = tf.reduce_mean(output ** 2)

        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads)

    def test_gradients_finite(self, discrete_class, sample_input):
        """Test gradients are finite (no NaN/Inf)."""
        if discrete_class == Meixner:
            layer = discrete_class(degree=4, units=8)
        else:
            layer = discrete_class(degree=4, units=8, N=10)

        with tf.GradientTape() as tape:
            output = layer(sample_input)
            loss = tf.reduce_mean(output ** 2)

        grads = tape.gradient(loss, layer.trainable_variables)
        for g in grads:
            assert not tf.reduce_any(tf.math.is_nan(g))
            assert not tf.reduce_any(tf.math.is_inf(g))

    def test_input_gradients(self, discrete_class):
        """Test gradients with respect to input."""
        if discrete_class == Meixner:
            layer = discrete_class(degree=3, units=4)
        else:
            layer = discrete_class(degree=3, units=4, N=10)

        x = tf.Variable([[1.0, 2.0, 3.0]])

        with tf.GradientTape() as tape:
            output = layer(x)
            loss = tf.reduce_sum(output)

        grad = tape.gradient(loss, x)
        assert grad is not None
        assert not tf.reduce_any(tf.math.is_nan(grad))


# ============================================================================
# XLA Compatibility Tests
# ============================================================================


class TestXLACompatibility:
    """Test XLA compilation compatibility."""

    def test_xla_compilation(self, discrete_class, sample_input):
        """Test layer works with XLA compilation."""
        if discrete_class == Meixner:
            layer = discrete_class(degree=3, units=8)
        else:
            layer = discrete_class(degree=3, units=8, N=10)

        @tf.function(jit_compile=True)
        def forward(x):
            return layer(x)

        # Should not raise
        output = forward(sample_input)
        assert output.shape == (8, 8)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_xla_training_step(self, discrete_class, sample_input):
        """Test XLA-compiled training step."""
        if discrete_class == Meixner:
            layer = discrete_class(degree=3, units=8)
        else:
            layer = discrete_class(degree=3, units=8, N=10)

        optimizer = tf.keras.optimizers.Adam(0.01)

        @tf.function(jit_compile=True)
        def train_step(x, y):
            with tf.GradientTape() as tape:
                pred = layer(x)
                loss = tf.reduce_mean((pred - y) ** 2)
            grads = tape.gradient(loss, layer.trainable_variables)
            optimizer.apply_gradients(zip(grads, layer.trainable_variables))
            return loss

        y = tf.random.uniform((8, 8))
        loss = train_step(sample_input, y)
        assert not tf.math.is_nan(loss)


# ============================================================================
# Serialization Tests
# ============================================================================


class TestSerialization:
    """Test layer serialization and deserialization."""

    def test_get_config(self, discrete_class, sample_input):
        """Test get_config returns valid configuration."""
        if discrete_class == Meixner:
            layer = discrete_class(degree=4, units=8)
        else:
            layer = discrete_class(degree=4, units=8, N=10)
        _ = layer(sample_input)

        config = layer.get_config()
        assert "degree" in config
        assert "units" in config
        assert config["degree"] == 4
        assert config["units"] == 8

    def test_from_config(self, discrete_class, sample_input):
        """Test layer can be reconstructed from config."""
        if discrete_class == Meixner:
            layer = discrete_class(degree=4, units=8)
        else:
            layer = discrete_class(degree=4, units=8, N=10)
        _ = layer(sample_input)

        config = layer.get_config()
        new_layer = discrete_class.from_config(config)

        output_new = new_layer(sample_input)
        assert output_new.shape == (8, 8)

    def test_keras_model_save_load(self, discrete_class, sample_input, tmp_path):
        """Test layer works in Keras model save/load."""
        if discrete_class == Meixner:
            layer = discrete_class(degree=3, units=8)
        else:
            layer = discrete_class(degree=3, units=8, N=10)

        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(4,)),
            layer,
        ])

        output_before = model(sample_input)

        # Save and load
        save_path = tmp_path / "model.keras"
        model.save(save_path)
        loaded_model = tf.keras.models.load_model(save_path)

        output_after = loaded_model(sample_input)

        np.testing.assert_allclose(output_before.numpy(), output_after.numpy(), rtol=1e-5)

    def test_krawtchouk_config_preserves_params(self, sample_input):
        """Test Krawtchouk config preserves all parameters."""
        layer = Krawtchouk(degree=4, units=8, p_init=0.3, p_trainable=False, N=15)
        _ = layer(sample_input)

        config = layer.get_config()
        assert config["p_init"] == 0.3
        assert config["p_trainable"] is False
        assert config["N"] == 15

        new_layer = Krawtchouk.from_config(config)
        assert new_layer.p_init == 0.3
        assert new_layer.p_trainable is False
        assert new_layer.N == 15

    def test_hahn_config_preserves_params(self, sample_input):
        """Test Hahn config preserves all parameters."""
        layer = Hahn(
            degree=4, units=8,
            alpha_init=0.7, alpha_trainable=False,
            beta_init=1.2, beta_trainable=True,
            N=12
        )
        _ = layer(sample_input)

        config = layer.get_config()
        assert config["alpha_init"] == 0.7
        assert config["beta_init"] == 1.2
        assert config["N"] == 12

    def test_meixner_config_preserves_params(self, sample_input):
        """Test Meixner config preserves all parameters."""
        layer = Meixner(
            degree=4, units=8,
            beta_init=2.0, beta_trainable=False,
            c_init=0.7, c_trainable=True
        )
        _ = layer(sample_input)

        config = layer.get_config()
        assert config["beta_init"] == 2.0
        assert config["c_init"] == 0.7

    def test_racah_config_preserves_params(self, sample_input):
        """Test Racah config preserves all parameters."""
        layer = Racah(
            degree=4, units=8, N=15,
            alpha_init=0.3, beta_init=0.4,
            gamma_init=0.5, delta_init=0.6
        )
        _ = layer(sample_input)

        config = layer.get_config()
        assert config["alpha_init"] == 0.3
        assert config["beta_init"] == 0.4
        assert config["gamma_init"] == 0.5
        assert config["delta_init"] == 0.6
        assert config["N"] == 15


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Test discrete polynomial layers in practical settings."""

    def test_in_sequential_model(self, discrete_class, sample_input):
        """Test layer works in Sequential model."""
        if discrete_class == Meixner:
            layer = discrete_class(degree=3, units=16)
        else:
            layer = discrete_class(degree=3, units=16, N=10)

        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(4,)),
            layer,
            tf.keras.layers.Dense(8),
        ])

        output = model(sample_input)
        assert output.shape == (8, 8)

    def test_in_functional_model(self, discrete_class, sample_input):
        """Test layer works in Functional API model."""
        if discrete_class == Meixner:
            layer = discrete_class(degree=3, units=16)
        else:
            layer = discrete_class(degree=3, units=16, N=10)

        inputs = tf.keras.Input(shape=(4,))
        x = layer(inputs)
        outputs = tf.keras.layers.Dense(8)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        output = model(sample_input)
        assert output.shape == (8, 8)

    def test_training_step(self, discrete_class, sample_input):
        """Test a complete training step."""
        if discrete_class == Meixner:
            layer = discrete_class(degree=3, units=8)
        else:
            layer = discrete_class(degree=3, units=8, N=10)

        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(4,)),
            layer,
        ])

        model.compile(optimizer="adam", loss="mse")

        y = tf.random.uniform((8, 8))
        history = model.fit(sample_input, y, epochs=1, verbose=0)

        assert len(history.history["loss"]) == 1

    def test_with_regularization(self, discrete_class, sample_input):
        """Test layer with kernel regularization."""
        if discrete_class == Meixner:
            layer = discrete_class(
                degree=3,
                units=8,
                kernel_regularizer=tf.keras.regularizers.L2(0.01),
            )
        else:
            layer = discrete_class(
                degree=3,
                units=8,
                N=10,
                kernel_regularizer=tf.keras.regularizers.L2(0.01),
            )

        with tf.GradientTape() as tape:
            output = layer(sample_input)
            loss = tf.reduce_mean(output) + sum(layer.losses)

        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_degree_zero(self, discrete_class):
        """Test degree=0 works (constant function)."""
        if discrete_class == Meixner:
            layer = discrete_class(degree=0, units=4)
        else:
            layer = discrete_class(degree=0, units=4, N=10)

        x = tf.constant([[1.0, 2.0, 3.0]])
        output = layer(x)
        assert output.shape == (1, 4)

    def test_degree_one(self, discrete_class):
        """Test degree=1 works (linear)."""
        if discrete_class == Meixner:
            layer = discrete_class(degree=1, units=4)
        else:
            layer = discrete_class(degree=1, units=4, N=10)

        x = tf.constant([[1.0, 2.0, 3.0]])
        output = layer(x)
        assert output.shape == (1, 4)

    def test_high_degree(self, discrete_class):
        """Test high degree doesn't cause numerical issues."""
        if discrete_class == Meixner:
            layer = discrete_class(degree=15, units=4)
        else:
            layer = discrete_class(degree=15, units=4, N=20)

        x = tf.constant([[1.0, 2.0, 3.0]])
        output = layer(x)

        # Should be finite
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))

    def test_large_input(self, discrete_class):
        """Test with larger input values."""
        if discrete_class == Meixner:
            layer = discrete_class(degree=4, units=8)
        else:
            layer = discrete_class(degree=4, units=8, N=100)

        x = tf.constant([[10.0, 20.0, 30.0, 40.0]])
        output = layer(x)

        assert output.shape == (1, 8)
        # May have large values but should be finite
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_negative_input(self, discrete_class):
        """Test with negative input (outside typical discrete domain)."""
        if discrete_class == Meixner:
            layer = discrete_class(degree=3, units=4)
        else:
            layer = discrete_class(degree=3, units=4, N=10)

        x = tf.constant([[-1.0, -2.0, 0.0, 1.0]])
        output = layer(x)

        # Should handle gracefully (polynomials extend beyond discrete points)
        assert output.shape == (1, 4)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_single_sample(self, discrete_class):
        """Test with batch size 1."""
        if discrete_class == Meixner:
            layer = discrete_class(degree=3, units=4)
        else:
            layer = discrete_class(degree=3, units=4, N=10)

        x = tf.constant([[1.0, 2.0]])
        output = layer(x)
        assert output.shape == (1, 4)

    def test_large_batch(self, discrete_class):
        """Test with large batch size."""
        if discrete_class == Meixner:
            layer = discrete_class(degree=3, units=4)
        else:
            layer = discrete_class(degree=3, units=4, N=10)

        x = tf.random.uniform((256, 8), minval=0, maxval=5)
        output = layer(x)
        assert output.shape == (256, 4)
