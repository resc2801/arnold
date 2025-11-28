# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
"""
MNIST Integration Tests - Phase 4.4

End-to-end integration tests using MNIST-like synthetic data.
Tests verify that KAN layers can train in a realistic setting.
"""

import numpy as np
import pytest
import tensorflow as tf

from arnold.layers.core.polynomial.orthogonal import Legendre, Chebyshev1st
from arnold.layers.core.radial_basis_functions import GaussianRBF


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def synthetic_mnist_data():
    """
    Create synthetic MNIST-like data for fast integration testing.
    
    Real MNIST: 28x28 = 784 features, 10 classes, 60k training samples
    Synthetic: 64 features, 5 classes, 500 training samples (for speed)
    """
    np.random.seed(42)
    n_train = 500
    n_test = 100
    n_features = 64
    n_classes = 5

    # Generate random features (normalized to [-1, 1] for polynomial stability)
    x_train = np.random.randn(n_train, n_features).astype(np.float32)
    x_train = np.clip(x_train, -2, 2) / 2  # Normalize to [-1, 1]

    x_test = np.random.randn(n_test, n_features).astype(np.float32)
    x_test = np.clip(x_test, -2, 2) / 2

    # Generate random labels
    y_train = np.random.randint(0, n_classes, n_train)
    y_test = np.random.randint(0, n_classes, n_test)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "n_classes": n_classes,
        "n_features": n_features,
    }


# ============================================================================
# Model Building Tests
# ============================================================================


class TestModelBuilding:
    """Test that KAN models can be built and compiled."""

    def test_sequential_model_build(self, synthetic_mnist_data):
        """Sequential model with KAN layers builds correctly."""
        data = synthetic_mnist_data
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(data["n_features"],)),
                Legendre(units=32, degree=4),
                tf.keras.layers.ReLU(),
                Legendre(units=data["n_classes"], degree=3),
                tf.keras.layers.Softmax(),
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        # Verify model is built
        assert model.built
        assert len(model.layers) == 4  # InputLayer is not counted in Keras 3

    def test_functional_model_build(self, synthetic_mnist_data):
        """Functional model with KAN layers builds correctly."""
        data = synthetic_mnist_data
        inputs = tf.keras.Input(shape=(data["n_features"],))
        x = Legendre(units=32, degree=4)(inputs)
        x = tf.keras.layers.ReLU()(x)
        x = Legendre(units=data["n_classes"], degree=3)(x)
        outputs = tf.keras.layers.Softmax()(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        assert model.built

    def test_mixed_kan_dense_model(self, synthetic_mnist_data):
        """Mixed model with KAN and Dense layers builds correctly."""
        data = synthetic_mnist_data
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(data["n_features"],)),
                tf.keras.layers.Dense(32, activation="relu"),
                Legendre(units=16, degree=3),
                tf.keras.layers.Dense(data["n_classes"], activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        assert model.built


# ============================================================================
# Training Tests
# ============================================================================


class TestTraining:
    """Test that KAN models can train on data."""

    def test_single_epoch_training(self, synthetic_mnist_data):
        """Model can complete a single epoch of training."""
        data = synthetic_mnist_data
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(data["n_features"],)),
                Legendre(units=16, degree=3),
                tf.keras.layers.Dense(data["n_classes"], activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        history = model.fit(data["x_train"], data["y_train"], epochs=1, batch_size=32, verbose=0)

        assert "loss" in history.history
        assert len(history.history["loss"]) == 1
        assert not np.isnan(history.history["loss"][0])

    def test_multiple_epoch_training(self, synthetic_mnist_data):
        """Model can complete multiple epochs of training."""
        data = synthetic_mnist_data
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(data["n_features"],)),
                Legendre(units=16, degree=3),
                tf.keras.layers.Dense(data["n_classes"], activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        history = model.fit(data["x_train"], data["y_train"], epochs=3, batch_size=32, verbose=0)

        assert len(history.history["loss"]) == 3
        assert all(not np.isnan(loss) for loss in history.history["loss"])

    def test_loss_decreases(self, synthetic_mnist_data):
        """Loss generally decreases over training."""
        data = synthetic_mnist_data
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(data["n_features"],)),
                Legendre(units=16, degree=3),
                tf.keras.layers.Dense(data["n_classes"], activation="softmax"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        history = model.fit(data["x_train"], data["y_train"], epochs=10, batch_size=32, verbose=0)

        # Loss should generally decrease (allow some fluctuation)
        first_loss = history.history["loss"][0]
        last_loss = history.history["loss"][-1]
        assert last_loss < first_loss, f"Loss did not decrease: {first_loss} -> {last_loss}"

    def test_validation_split(self, synthetic_mnist_data):
        """Training with validation split works."""
        data = synthetic_mnist_data
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(data["n_features"],)),
                Legendre(units=16, degree=3),
                tf.keras.layers.Dense(data["n_classes"], activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        history = model.fit(
            data["x_train"], data["y_train"], epochs=2, batch_size=32, validation_split=0.2, verbose=0
        )

        assert "val_loss" in history.history
        assert len(history.history["val_loss"]) == 2


# ============================================================================
# Evaluation Tests
# ============================================================================


class TestEvaluation:
    """Test model evaluation and prediction."""

    def test_evaluate(self, synthetic_mnist_data):
        """Model.evaluate() works correctly."""
        data = synthetic_mnist_data
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(data["n_features"],)),
                Legendre(units=16, degree=3),
                tf.keras.layers.Dense(data["n_classes"], activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        model.fit(data["x_train"], data["y_train"], epochs=1, batch_size=32, verbose=0)
        results = model.evaluate(data["x_test"], data["y_test"], verbose=0)

        assert len(results) == 2  # loss and accuracy
        assert not np.isnan(results[0])  # loss
        assert 0 <= results[1] <= 1  # accuracy

    def test_predict_shape(self, synthetic_mnist_data):
        """Predictions have correct shape."""
        data = synthetic_mnist_data
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(data["n_features"],)),
                Legendre(units=16, degree=3),
                tf.keras.layers.Dense(data["n_classes"], activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        predictions = model.predict(data["x_test"], verbose=0)

        assert predictions.shape == (data["x_test"].shape[0], data["n_classes"])

    def test_predict_probabilities(self, synthetic_mnist_data):
        """Predictions are valid probabilities (sum to 1)."""
        data = synthetic_mnist_data
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(data["n_features"],)),
                Legendre(units=16, degree=3),
                tf.keras.layers.Dense(data["n_classes"], activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        predictions = model.predict(data["x_test"], verbose=0)

        # Each row should sum to 1
        row_sums = np.sum(predictions, axis=1)
        np.testing.assert_allclose(row_sums, np.ones(len(row_sums)), rtol=1e-5)


# ============================================================================
# Different KAN Types Tests
# ============================================================================


class TestDifferentKANTypes:
    """Test different KAN layer types in end-to-end setting."""

    def test_chebyshev_training(self, synthetic_mnist_data):
        """Chebyshev KAN layers can train."""
        data = synthetic_mnist_data
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(data["n_features"],)),
                Chebyshev1st(units=16, degree=3),
                tf.keras.layers.Dense(data["n_classes"], activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        history = model.fit(data["x_train"], data["y_train"], epochs=2, batch_size=32, verbose=0)
        assert not np.isnan(history.history["loss"][-1])

    def test_rbf_training(self, synthetic_mnist_data):
        """RBF KAN layers can train."""
        data = synthetic_mnist_data
        # RBF expects inputs in [0, 1], so we rescale
        x_train = (data["x_train"] + 1) / 2
        x_test = (data["x_test"] + 1) / 2

        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(data["n_features"],)),
                GaussianRBF(units=16, grid_min=0.0, grid_max=1.0, num_grids=8),
                tf.keras.layers.Dense(data["n_classes"], activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        history = model.fit(x_train, data["y_train"], epochs=2, batch_size=32, verbose=0)
        assert not np.isnan(history.history["loss"][-1])

    def test_tucker_decomposition_training(self, synthetic_mnist_data):
        """Tucker-decomposed KAN layers can train."""
        data = synthetic_mnist_data
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(data["n_features"],)),
                Legendre(units=16, degree=4, core_ranks=(8, 4, 8)),
                tf.keras.layers.Dense(data["n_classes"], activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        history = model.fit(data["x_train"], data["y_train"], epochs=2, batch_size=32, verbose=0)
        assert not np.isnan(history.history["loss"][-1])


# ============================================================================
# Model Persistence Tests
# ============================================================================


class TestModelPersistence:
    """Test model saving and loading."""

    def test_save_load_weights(self, synthetic_mnist_data, tmp_path):
        """Model weights can be saved and loaded."""
        import warnings
        
        data = synthetic_mnist_data
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(data["n_features"],)),
                Legendre(units=16, degree=3),
                tf.keras.layers.Dense(data["n_classes"], activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        model.fit(data["x_train"], data["y_train"], epochs=1, batch_size=32, verbose=0)

        # Save weights
        weights_path = tmp_path / "weights.weights.h5"
        model.save_weights(str(weights_path))

        # Create new model and load weights
        model2 = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(data["n_features"],)),
                Legendre(units=16, degree=3),
                tf.keras.layers.Dense(data["n_classes"], activation="softmax"),
            ]
        )
        model2.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        # Build model2 before loading weights
        _ = model2(data["x_train"][:1])
        # Suppress optimizer state mismatch warning (expected when loading weights into fresh model)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")
            model2.load_weights(str(weights_path))

        # Predictions should be identical
        pred1 = model.predict(data["x_test"], verbose=0)
        pred2 = model2.predict(data["x_test"], verbose=0)
        np.testing.assert_allclose(pred1, pred2, rtol=1e-5)

    def test_save_load_full_model(self, synthetic_mnist_data, tmp_path):
        """Full model can be saved and loaded."""
        data = synthetic_mnist_data
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(data["n_features"],)),
                Legendre(units=16, degree=3),
                tf.keras.layers.Dense(data["n_classes"], activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        model.fit(data["x_train"], data["y_train"], epochs=1, batch_size=32, verbose=0)

        # Save full model
        model_path = tmp_path / "model.keras"
        model.save(str(model_path))

        # Load model
        model2 = tf.keras.models.load_model(str(model_path))

        # Predictions should be identical
        pred1 = model.predict(data["x_test"], verbose=0)
        pred2 = model2.predict(data["x_test"], verbose=0)
        np.testing.assert_allclose(pred1, pred2, rtol=1e-5)


# ============================================================================
# SavedModel Format Tests (TF-Serving/TFLite deployment path)
# ============================================================================


class TestSavedModelRoundtrip:
    """Test SavedModel format for TensorFlow Serving and TFLite deployment."""

    def test_savedmodel_roundtrip_legendre(self, synthetic_mnist_data, tmp_path):
        """Legendre model survives SavedModel export/import roundtrip."""
        data = synthetic_mnist_data
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(data["n_features"],)),
                Legendre(units=16, degree=3),
                tf.keras.layers.Dense(data["n_classes"], activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        model.fit(data["x_train"], data["y_train"], epochs=1, batch_size=32, verbose=0)

        # Get original predictions
        pred_original = model.predict(data["x_test"], verbose=0)

        # Export as SavedModel (TF-Serving format)
        savedmodel_path = tmp_path / "saved_model"
        tf.saved_model.save(model, str(savedmodel_path))

        # Reload from SavedModel
        loaded = tf.saved_model.load(str(savedmodel_path))
        
        # Get predictions from loaded model
        infer = loaded.signatures["serving_default"]
        result = infer(tf.constant(data["x_test"]))
        # Output key varies by Keras version
        output_key = list(result.keys())[0]
        pred_loaded = result[output_key].numpy()

        np.testing.assert_allclose(pred_original, pred_loaded, rtol=1e-5)

    def test_savedmodel_roundtrip_chebyshev(self, synthetic_mnist_data, tmp_path):
        """Chebyshev model survives SavedModel export/import roundtrip."""
        data = synthetic_mnist_data
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(data["n_features"],)),
                Chebyshev1st(units=16, degree=4),
                tf.keras.layers.Dense(data["n_classes"], activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        model.fit(data["x_train"], data["y_train"], epochs=1, batch_size=32, verbose=0)

        pred_original = model.predict(data["x_test"], verbose=0)

        savedmodel_path = tmp_path / "saved_model_cheb"
        tf.saved_model.save(model, str(savedmodel_path))

        loaded = tf.saved_model.load(str(savedmodel_path))
        infer = loaded.signatures["serving_default"]
        result = infer(tf.constant(data["x_test"]))
        output_key = list(result.keys())[0]
        pred_loaded = result[output_key].numpy()

        np.testing.assert_allclose(pred_original, pred_loaded, rtol=1e-5)

    def test_savedmodel_roundtrip_rbf(self, synthetic_mnist_data, tmp_path):
        """RBF model survives SavedModel export/import roundtrip."""
        data = synthetic_mnist_data
        x_train = (data["x_train"] + 1) / 2  # Rescale to [0, 1]
        x_test = (data["x_test"] + 1) / 2

        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(data["n_features"],)),
                GaussianRBF(units=16, grid_min=0.0, grid_max=1.0, num_grids=8),
                tf.keras.layers.Dense(data["n_classes"], activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        model.fit(x_train, data["y_train"], epochs=1, batch_size=32, verbose=0)

        pred_original = model.predict(x_test, verbose=0)

        savedmodel_path = tmp_path / "saved_model_rbf"
        tf.saved_model.save(model, str(savedmodel_path))

        loaded = tf.saved_model.load(str(savedmodel_path))
        infer = loaded.signatures["serving_default"]
        result = infer(tf.constant(x_test))
        output_key = list(result.keys())[0]
        pred_loaded = result[output_key].numpy()

        np.testing.assert_allclose(pred_original, pred_loaded, rtol=1e-5)

    def test_savedmodel_roundtrip_tucker(self, synthetic_mnist_data, tmp_path):
        """Tucker-decomposed model survives SavedModel export/import roundtrip."""
        data = synthetic_mnist_data
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(data["n_features"],)),
                Legendre(units=16, degree=4, core_ranks=(8, 4, 8)),
                tf.keras.layers.Dense(data["n_classes"], activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        model.fit(data["x_train"], data["y_train"], epochs=1, batch_size=32, verbose=0)

        pred_original = model.predict(data["x_test"], verbose=0)

        savedmodel_path = tmp_path / "saved_model_tucker"
        tf.saved_model.save(model, str(savedmodel_path))

        loaded = tf.saved_model.load(str(savedmodel_path))
        infer = loaded.signatures["serving_default"]
        result = infer(tf.constant(data["x_test"]))
        output_key = list(result.keys())[0]
        pred_loaded = result[output_key].numpy()

        np.testing.assert_allclose(pred_original, pred_loaded, rtol=1e-5)

    def test_savedmodel_concrete_function(self, synthetic_mnist_data, tmp_path):
        """Verify concrete function can be extracted for XLA compilation."""
        data = synthetic_mnist_data
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(data["n_features"],)),
                Legendre(units=16, degree=3),
                tf.keras.layers.Dense(data["n_classes"], activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        
        # Build model
        _ = model(data["x_train"][:1])

        # Get concrete function (used by XLA, TFLite, etc.)
        @tf.function(input_signature=[tf.TensorSpec(shape=(None, data["n_features"]), dtype=tf.float32)])
        def serve(x):
            return model(x, training=False)

        concrete_fn = serve.get_concrete_function()
        
        # Save with concrete function
        savedmodel_path = tmp_path / "saved_model_concrete"
        tf.saved_model.save(model, str(savedmodel_path), signatures={"serving_default": concrete_fn})

        # Verify it loads and works
        loaded = tf.saved_model.load(str(savedmodel_path))
        pred_original = model.predict(data["x_test"], verbose=0)
        pred_loaded = loaded.signatures["serving_default"](tf.constant(data["x_test"]))
        output_key = list(pred_loaded.keys())[0]
        
        np.testing.assert_allclose(pred_original, pred_loaded[output_key].numpy(), rtol=1e-5)
