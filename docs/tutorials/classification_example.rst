.. Copyright (c) 2025 René Schubotz. All rights reserved.

.. _tutorial-classification:

================================================
Classification with KANs
================================================

This tutorial demonstrates how to use KAN layers for classification tasks,
from binary classification to multi-class problems like MNIST.

.. contents:: In This Tutorial
   :local:
   :depth: 2

----

Objective
---------

By the end of this tutorial, you will:

1. Build KAN-based classifiers for binary and multi-class problems
2. Properly preprocess data for polynomial KANs
3. Choose appropriate architectures for classification
4. Understand when KANs outperform standard networks

----

Binary Classification: Two Moons
--------------------------------

The "two moons" dataset is a classic nonlinear classification problem:

.. code-block:: python
   :linenos:

   import tensorflow as tf
   import numpy as np
   from sklearn.datasets import make_moons
   from arnold.layers import Chebyshev1st, Legendre

   # Generate data
   X, y = make_moons(n_samples=2000, noise=0.15, random_state=42)
   X = X.astype('float32')
   y = y.astype('float32').reshape(-1, 1)

   # Normalize to [-1, 1]
   X_mean, X_std = X.mean(axis=0), X.std(axis=0)
   X = (X - X_mean) / (X_std * 2)  # Roughly in [-1, 1]

   # Split
   split = 1500
   X_train, X_test = X[:split], X[split:]
   y_train, y_test = y[:split], y[split:]

   # Build KAN classifier
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(2,)),
       Chebyshev1st(degree=5, units=16),
       tf.keras.layers.LayerNormalization(),
       Legendre(degree=3, units=8),
       tf.keras.layers.LayerNormalization(),
       tf.keras.layers.Dense(1, activation='sigmoid'),
   ])

   model.compile(
       optimizer=tf.keras.optimizers.Adam(1e-3),
       loss='binary_crossentropy',
       metrics=['accuracy'],
   )

   history = model.fit(
       X_train, y_train,
       epochs=50,
       batch_size=32,
       validation_data=(X_test, y_test),
       verbose=1,
   )

   # Evaluate
   test_loss, test_acc = model.evaluate(X_test, y_test)
   print(f"Test accuracy: {test_acc:.4f}")

**Expected:** ~98-99% accuracy

**Architecture Notes:**

- Use ``Dense`` with ``sigmoid`` for the final layer (not a KAN layer)
- ``binary_crossentropy`` is the standard loss for binary classification
- Input normalization is crucial for polynomial stability

----

Multi-Class Classification: MNIST
---------------------------------

Let's tackle the classic MNIST digit classification:

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from arnold.layers import Chebyshev1st, Legendre

   # Load MNIST
   (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

   # Flatten and normalize to [-1, 1]
   x_train = x_train.reshape(-1, 784).astype('float32')
   x_test = x_test.reshape(-1, 784).astype('float32')
   x_train = (x_train / 127.5) - 1.0  # Scale to [-1, 1]
   x_test = (x_test / 127.5) - 1.0

   # Build KAN classifier
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(784,)),
       
       # First KAN layer with Tucker decomposition (memory efficient)
       Chebyshev1st(
           degree=5, 
           units=128,
           core_ranks=(16, 16, 5),  # Reduces parameters significantly
       ),
       tf.keras.layers.LayerNormalization(),
       tf.keras.layers.Dropout(0.2),
       
       # Second KAN layer
       Legendre(degree=3, units=64),
       tf.keras.layers.LayerNormalization(),
       tf.keras.layers.Dropout(0.2),
       
       # Output layer
       tf.keras.layers.Dense(10, activation='softmax'),
   ])

   # Compile
   model.compile(
       optimizer=tf.keras.optimizers.Adam(1e-3),
       loss='sparse_categorical_crossentropy',
       metrics=['accuracy'],
   )

   # Print model summary
   model.summary()

   # Train
   history = model.fit(
       x_train, y_train,
       epochs=15,
       batch_size=128,
       validation_data=(x_test, y_test),
       verbose=1,
   )

   # Evaluate
   test_loss, test_acc = model.evaluate(x_test, y_test)
   print(f"Test accuracy: {test_acc:.4f}")

**Expected:** ~97-98% accuracy in 15 epochs

**Key Design Choices:**

1. **Tucker decomposition** — 784 input dimensions would explode without it
2. **Dropout** — Regularization prevents overfitting
3. **LayerNormalization** — Stabilizes polynomial outputs
4. **Dense output** — Standard softmax classification head

----

Architecture Patterns for Classification
----------------------------------------

Pattern 1: Small Input Dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For tabular data with few features:

.. code-block:: python

   # E.g., 10-50 features
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(20,)),
       Chebyshev1st(degree=6, units=32),
       tf.keras.layers.LayerNormalization(),
       Legendre(degree=4, units=16),
       tf.keras.layers.LayerNormalization(),
       tf.keras.layers.Dense(num_classes, activation='softmax'),
   ])

Pattern 2: Large Input Dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For images, text embeddings, etc.:

.. code-block:: python

   # E.g., 784+ features (images)
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(784,)),
       
       # Heavy Tucker compression for large input
       Chebyshev1st(
           degree=5, 
           units=128,
           core_ranks=(32, 32, 5),
       ),
       tf.keras.layers.LayerNormalization(),
       tf.keras.layers.Dropout(0.3),
       
       # Standard KAN (smaller input after first layer)
       Legendre(degree=4, units=64),
       tf.keras.layers.LayerNormalization(),
       
       tf.keras.layers.Dense(num_classes, activation='softmax'),
   ])

Pattern 3: Hybrid KAN + MLP
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine KAN expressivity with MLP efficiency:

.. code-block:: python

   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(784,)),
       
       # Dense for dimensionality reduction
       tf.keras.layers.Dense(128, activation='relu'),
       tf.keras.layers.LayerNormalization(),
       
       # KAN for nonlinear feature extraction
       Chebyshev1st(degree=6, units=64),
       tf.keras.layers.LayerNormalization(),
       
       # Dense for classification
       tf.keras.layers.Dense(num_classes, activation='softmax'),
   ])

----

Data Preprocessing
------------------

Proper preprocessing is critical for polynomial KANs:

Scaling to [-1, 1]
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Option 1: Simple rescaling
   x = (x - x.min()) / (x.max() - x.min()) * 2 - 1

   # Option 2: Using Keras layer
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(784,)),
       tf.keras.layers.Rescaling(1./127.5, offset=-1),  # For 0-255 images
       Chebyshev1st(degree=5, units=64),
       ...
   ])

   # Option 3: Standard scaling (good for tabular data)
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   x_train = scaler.fit_transform(x_train)
   x_test = scaler.transform(x_test)
   # Then clip to prevent outliers
   x_train = np.clip(x_train, -3, 3) / 3  # Roughly in [-1, 1]

Handling Categorical Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For mixed tabular data:

.. code-block:: python

   # One-hot encode categorical features
   from sklearn.preprocessing import OneHotEncoder, StandardScaler
   
   # Categorical features: one-hot (no normalization needed)
   cat_encoder = OneHotEncoder(sparse=False)
   X_cat = cat_encoder.fit_transform(X_categorical)
   
   # Numerical features: normalize to [-1, 1]
   num_scaler = StandardScaler()
   X_num = num_scaler.fit_transform(X_numerical)
   X_num = np.clip(X_num, -3, 3) / 3
   
   # Combine
   X = np.hstack([X_num, X_cat])

----

Regularization Techniques
-------------------------

Dropout
~~~~~~~

.. code-block:: python

   Chebyshev1st(degree=6, units=64),
   tf.keras.layers.LayerNormalization(),
   tf.keras.layers.Dropout(0.2),  # After normalization

Weight Regularization
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   Legendre(
       degree=6,
       units=64,
       kernel_regularizer=tf.keras.regularizers.L2(1e-4),
   )

Early Stopping
~~~~~~~~~~~~~~

.. code-block:: python

   callbacks = [
       tf.keras.callbacks.EarlyStopping(
           monitor='val_loss',
           patience=10,
           restore_best_weights=True,
       )
   ]
   
   model.fit(..., callbacks=callbacks)

----

Exercises
---------

**Exercise 1:** Implement a KAN classifier for Fashion-MNIST:

.. code-block:: python

   (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

Can you achieve >90% accuracy?

**Exercise 2:** Try mixing RBF and polynomial KANs:

.. code-block:: python

   from arnold.layers import GaussianRBF
   
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(784,)),
       GaussianRBF(units=64, num_grids=16),  # RBF layer
       tf.keras.layers.LayerNormalization(),
       Legendre(degree=4, units=32),          # Polynomial layer
       tf.keras.layers.LayerNormalization(),
       tf.keras.layers.Dense(10, activation='softmax'),
   ])

**Exercise 3:** Compare training time and accuracy between:

- Pure KAN model
- Pure MLP model
- Hybrid KAN+MLP model

----

Next Steps
----------

- :doc:`time_series_example` — Wavelet KANs for temporal data
- :doc:`../layers/polynomial_kan_layers` — Complete polynomial API
- :doc:`../performance_guide` — GPU/TPU optimization for classification
