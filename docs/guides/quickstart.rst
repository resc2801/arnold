.. Copyright (c) 2025 René Schubotz. All rights reserved.

.. _quickstart:

================================================
Quick Start Guide
================================================

Get up and running with ARNOLD in 5 minutes. This guide covers installation, 
your first KAN model, and essential concepts.

.. contents:: In This Guide
   :local:
   :depth: 2

----

Installation
------------

**Requirements:**

- Python 3.10+
- TensorFlow 2.15+ (or tensorflow-metal for Apple Silicon)

**Install via pip:**

.. code-block:: bash

   pip install arnold-kan

**Verify installation:**

.. code-block:: python

   import arnold
   from arnold.layers import Legendre
   
   # Check version
   print(f"ARNOLD version: {arnold.__version__}")
   
   # Quick test
   layer = Legendre(degree=5, units=10)
   print("✓ Installation successful!")

**Optional: Apple Silicon GPU support:**

.. code-block:: bash

   pip install tensorflow-metal

----

Your First KAN Model
--------------------

Let's build a simple KAN for regression:

.. code-block:: python
   :linenos:

   import tensorflow as tf
   import numpy as np
   from arnold.layers import Chebyshev1st, Legendre

   # Generate synthetic data
   np.random.seed(42)
   X = np.random.uniform(-1, 1, (1000, 2))
   y = np.sin(np.pi * X[:, 0]) + np.cos(2 * np.pi * X[:, 1])

   # Build KAN model
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(2,)),
       Chebyshev1st(degree=5, units=16),   # First KAN layer
       tf.keras.layers.LayerNormalization(),
       Legendre(degree=5, units=8),         # Second KAN layer
       tf.keras.layers.LayerNormalization(),
       Legendre(degree=3, units=1),         # Output layer
   ])

   # Compile and train
   model.compile(optimizer='adam', loss='mse')
   model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)

   # Evaluate
   predictions = model.predict(X[:5])
   print(f"Sample predictions: {predictions.flatten()}")
   print(f"Actual values: {y[:5]}")

**Key points:**

1. KAN layers are **drop-in replacements** for ``Dense`` layers
2. Use ``LayerNormalization`` between KAN layers for stability
3. Start with low degrees (3-5) and increase if needed

----

Understanding KAN Parameters
----------------------------

Every KAN layer has these key parameters:

.. code-block:: python

   layer = Legendre(
       degree=5,           # Maximum polynomial degree
       units=64,           # Output dimension (like Dense)
       use_bias=True,      # Add bias term
       input_clip=(-1, 1), # Clip inputs to basis domain
   )

**Degree**: Controls expressivity. Higher = more flexible but more parameters.

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Degree
     - Use Case
     - Parameters (per input×output)
   * - 1-3
     - Simple relationships
     - 2-4
   * - 5-8
     - Most tasks (default)
     - 6-9
   * - 10-15
     - Complex functions
     - 11-16

**Units**: Output dimension, same as Keras ``Dense``.

**Input Clip**: Maps inputs to the polynomial's natural domain.

----

The Three KAN Families
----------------------

ARNOLD provides three families of basis functions:

Polynomial KANs
~~~~~~~~~~~~~~~

Best for smooth function approximation.

.. code-block:: python

   from arnold.layers import (
       Chebyshev1st,    # Best default choice
       Legendre,        # Great for physics problems
       Hermite,         # For Gaussian-like data
       Jacobi,          # Flexible with α, β parameters
   )

   # Example
   layer = Chebyshev1st(degree=8, units=32)

RBF KANs
~~~~~~~~

Best for interpolation and local features.

.. code-block:: python

   from arnold.layers import (
       GaussianRBF,           # Smooth, localized
       MultiquadricRBF,       # Global influence
       InverseMultiquadricRBF,# Bounded response
   )

   # Example
   layer = GaussianRBF(
       units=32,
       num_grids=16,     # Number of basis centers
       grid_min=-1.0,    # Grid lower bound
       grid_max=1.0,     # Grid upper bound
   )

Wavelet KANs
~~~~~~~~~~~~

Best for multi-scale and time-frequency analysis.

.. code-block:: python

   from arnold.layers import (
       Ricker,           # Mexican hat wavelet
       Morelet,          # Gabor-like wavelet
       Bump,             # Compact support
   )

   # Example
   layer = Ricker(units=32, sigma_init=1.0)

----

Common Patterns
---------------

Pattern 1: Classification
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(784,)),
       tf.keras.layers.Rescaling(1./127.5, offset=-1),  # Normalize to [-1, 1]
       Chebyshev1st(degree=5, units=128),
       tf.keras.layers.LayerNormalization(),
       tf.keras.layers.Dropout(0.2),
       Legendre(degree=3, units=64),
       tf.keras.layers.LayerNormalization(),
       tf.keras.layers.Dense(10, activation='softmax'),
   ])
   
   model.compile(
       optimizer='adam',
       loss='sparse_categorical_crossentropy',
       metrics=['accuracy']
   )

Pattern 2: Regression
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(10,)),
       Legendre(degree=8, units=64, input_clip=(-3, 3)),
       tf.keras.layers.LayerNormalization(),
       Legendre(degree=5, units=32),
       tf.keras.layers.LayerNormalization(),
       Legendre(degree=3, units=1),
   ])
   
   model.compile(optimizer='adam', loss='mse')

Pattern 3: Mixing with Dense Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(100,)),
       tf.keras.layers.Dense(64, activation='relu'),  # Dense for dimensionality reduction
       Chebyshev1st(degree=5, units=32),              # KAN for nonlinear transform
       tf.keras.layers.Dense(10),                      # Dense for output
   ])

Pattern 4: Memory Efficient (Tucker)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For large input dimensions, use Tucker decomposition
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(1000,)),
       Legendre(
           degree=10,
           units=128,
           core_ranks=(16, 8, 16),  # Tucker decomposition ranks
       ),
       Legendre(degree=5, units=10),
   ])

----

Quick Tips
----------

1. **Always normalize inputs** to the polynomial's domain (usually :math:`[-1, 1]`):

   .. code-block:: python

      tf.keras.layers.Rescaling(1./127.5, offset=-1)  # For 0-255 images

2. **Use LayerNormalization** between KAN layers:

   .. code-block:: python

      tf.keras.layers.LayerNormalization()

3. **Start simple** — low degree, shallow network — then add complexity:

   .. code-block:: python

      # Start here
      model = tf.keras.Sequential([
          Chebyshev1st(degree=3, units=32),
          Legendre(degree=3, units=1),
      ])

4. **Add regularization** for large models:

   .. code-block:: python

      Legendre(
          degree=10,
          units=64,
          kernel_regularizer=tf.keras.regularizers.L2(1e-4)
      )

5. **Use gradient clipping** if training is unstable:

   .. code-block:: python

      optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)

----

Next Steps
----------

- :doc:`../basis_selection_guide` — Choose the right basis function
- :doc:`advanced_usage` — Tucker decomposition, custom bases, mixed precision
- :doc:`../performance_guide` — GPU/TPU optimization
- :doc:`../troubleshooting` — Common issues and solutions

----

Example: MNIST with KANs
------------------------

Complete working example:

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from arnold.layers import Chebyshev1st, Legendre
   
   # Load data
   (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
   x_train = x_train.reshape(-1, 784).astype('float32')
   x_test = x_test.reshape(-1, 784).astype('float32')
   
   # Build KAN model
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(784,)),
       tf.keras.layers.Rescaling(1./127.5, offset=-1),  # Normalize to [-1, 1]
       
       Chebyshev1st(degree=5, units=128),
       tf.keras.layers.LayerNormalization(),
       tf.keras.layers.Dropout(0.2),
       
       Legendre(degree=3, units=64),
       tf.keras.layers.LayerNormalization(),
       tf.keras.layers.Dropout(0.2),
       
       tf.keras.layers.Dense(10, activation='softmax'),
   ])
   
   # Compile
   model.compile(
       optimizer=tf.keras.optimizers.Adam(1e-3),
       loss='sparse_categorical_crossentropy',
       metrics=['accuracy'],
   )
   
   # Train
   model.fit(
       x_train, y_train,
       epochs=10,
       batch_size=128,
       validation_data=(x_test, y_test),
   )
   
   # Evaluate
   test_loss, test_acc = model.evaluate(x_test, y_test)
   print(f"Test accuracy: {test_acc:.4f}")

Expected output: ~97-98% accuracy in 10 epochs.
