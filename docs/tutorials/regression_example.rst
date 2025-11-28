.. Copyright (c) 2025 René Schubotz. All rights reserved.

.. _tutorial-regression:

================================================
Regression with Polynomial KANs
================================================

This tutorial covers function approximation and regression tasks using
polynomial KAN layers, from simple 1D functions to complex multivariate problems.

.. contents:: In This Tutorial
   :local:
   :depth: 2

----

Objective
---------

By the end of this tutorial, you will:

1. Fit 1D, 2D, and higher-dimensional functions with KANs
2. Understand degree selection for different function complexities
3. Use Tucker decomposition for high-dimensional problems
4. Compare KAN performance with standard MLPs

----

Prerequisites
-------------

- Completed :doc:`getting_started`
- Basic understanding of polynomial approximation
- TensorFlow/Keras familiarity

----

1D Function Approximation
-------------------------

Let's fit increasingly complex functions to understand how polynomial
degree affects approximation quality.

Smooth Function: sin(πx)
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   import tensorflow as tf
   import numpy as np
   from arnold.layers import Chebyshev1st

   # Generate data
   np.random.seed(42)
   x = np.random.uniform(-1, 1, (2000, 1)).astype('float32')
   y_smooth = np.sin(np.pi * x)

   # Build model
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(1,)),
       Chebyshev1st(degree=5, units=8),
       tf.keras.layers.LayerNormalization(),
       Chebyshev1st(degree=3, units=1),
   ])

   model.compile(optimizer='adam', loss='mse')
   model.fit(x, y_smooth, epochs=100, batch_size=32, verbose=0)

   # Evaluate
   x_test = np.linspace(-1, 1, 200).reshape(-1, 1).astype('float32')
   y_true = np.sin(np.pi * x_test)
   y_pred = model.predict(x_test)
   mse = np.mean((y_true - y_pred)**2)
   print(f"Smooth function MSE: {mse:.8f}")

**Expected:** MSE < 1e-5 (polynomials excel at smooth functions)

High-Frequency Function: sin(10πx)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   y_highfreq = np.sin(10 * np.pi * x)

   # Need higher degree for high-frequency content
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(1,)),
       Chebyshev1st(degree=12, units=16),  # Higher degree
       tf.keras.layers.LayerNormalization(),
       Chebyshev1st(degree=8, units=8),
       tf.keras.layers.LayerNormalization(),
       Chebyshev1st(degree=4, units=1),
   ])

   model.compile(optimizer='adam', loss='mse')
   model.fit(x, y_highfreq, epochs=200, batch_size=32, verbose=0)

**Expected:** MSE ~ 1e-4 (harder, needs more capacity)

Function with Discontinuity
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Polynomials struggle with discontinuities (Gibbs phenomenon):

.. code-block:: python

   # Step function
   y_step = np.sign(x)

   # Even with high degree, Gibbs oscillations appear
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(1,)),
       Chebyshev1st(degree=20, units=32),
       tf.keras.layers.LayerNormalization(),
       Chebyshev1st(degree=10, units=1),
   ])

   model.compile(optimizer='adam', loss='mse')
   model.fit(x, y_step, epochs=300, batch_size=32, verbose=0)

   # Visualize Gibbs oscillations near discontinuity
   y_pred = model.predict(x_test)
   # You'll see oscillations near x=0

**Takeaway:** Use RBF or wavelet KANs for discontinuous functions.

----

Multivariate Regression
-----------------------

KANs naturally extend to multivariate functions. The key insight:
for a function :math:`f(x_1, x_2, \ldots, x_n)`, each KAN layer learns
univariate functions :math:`\phi_i(x_i)` and combines them.

2D Function: Peaks
~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   import tensorflow as tf
   import numpy as np
   from arnold.layers import Legendre

   # Generate 2D data (MATLAB peaks function)
   np.random.seed(42)
   X = np.random.uniform(-3, 3, (5000, 2)).astype('float32')
   x1, x2 = X[:, 0], X[:, 1]
   
   y = (3 * (1 - x1)**2 * np.exp(-x1**2 - (x2 + 1)**2)
        - 10 * (x1/5 - x1**3 - x2**5) * np.exp(-x1**2 - x2**2)
        - 1/3 * np.exp(-(x1 + 1)**2 - x2**2))
   y = y.reshape(-1, 1).astype('float32')

   # Normalize inputs to [-1, 1]
   X_normalized = X / 3.0

   # Build model
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(2,)),
       Legendre(degree=8, units=32),
       tf.keras.layers.LayerNormalization(),
       Legendre(degree=6, units=16),
       tf.keras.layers.LayerNormalization(),
       Legendre(degree=4, units=1),
   ])

   model.compile(
       optimizer=tf.keras.optimizers.Adam(1e-3),
       loss='mse',
   )

   history = model.fit(
       X_normalized, y,
       epochs=200,
       batch_size=64,
       validation_split=0.2,
       verbose=1,
   )

   # Final loss
   print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")

**Expected:** Validation MSE ~ 0.01-0.1 depending on training

High-Dimensional Function (10D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For high dimensions, use Tucker decomposition:

.. code-block:: python
   :linenos:

   # 10D Rosenbrock function variant
   X = np.random.uniform(-1, 1, (10000, 10)).astype('float32')
   
   y = np.zeros((10000, 1), dtype='float32')
   for i in range(9):
       y[:, 0] += (1 - X[:, i])**2 + 10 * (X[:, i+1] - X[:, i]**2)**2
   y = y / 100  # Scale

   # Standard layer: too many parameters
   # 10 inputs × 64 outputs × 9 coefficients = 5,760 per layer
   
   # Tucker decomposition reduces this significantly
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(10,)),
       Legendre(
           degree=8, 
           units=64,
           core_ranks=(8, 8, 8),  # Tucker decomposition
       ),
       tf.keras.layers.LayerNormalization(),
       Legendre(
           degree=6,
           units=32,
           core_ranks=(8, 8, 6),
       ),
       tf.keras.layers.LayerNormalization(),
       Legendre(degree=4, units=1),
   ])

   print(f"Total parameters: {model.count_params():,}")

   model.compile(optimizer='adam', loss='mse')
   model.fit(X, y, epochs=100, batch_size=128, validation_split=0.2, verbose=1)

----

Comparing KANs with MLPs
------------------------

Let's compare KAN and MLP performance on the same task:

.. code-block:: python
   :linenos:

   import tensorflow as tf
   import numpy as np
   from arnold.layers import Chebyshev1st

   # Generate complex 2D function
   np.random.seed(42)
   X = np.random.uniform(-1, 1, (5000, 2)).astype('float32')
   y = (np.sin(3 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1]) + 
        0.5 * np.sin(5 * np.pi * X[:, 0] * X[:, 1])).reshape(-1, 1).astype('float32')

   # Split data
   split = 4000
   X_train, X_test = X[:split], X[split:]
   y_train, y_test = y[:split], y[split:]

   # --- KAN Model ---
   kan_model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(2,)),
       Chebyshev1st(degree=8, units=32),
       tf.keras.layers.LayerNormalization(),
       Chebyshev1st(degree=6, units=16),
       tf.keras.layers.LayerNormalization(),
       Chebyshev1st(degree=4, units=1),
   ])

   kan_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
   kan_history = kan_model.fit(X_train, y_train, epochs=200, batch_size=32, 
                               validation_data=(X_test, y_test), verbose=0)

   # --- MLP Model (same parameter count) ---
   mlp_model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(2,)),
       tf.keras.layers.Dense(64, activation='relu'),
       tf.keras.layers.Dense(32, activation='relu'),
       tf.keras.layers.Dense(16, activation='relu'),
       tf.keras.layers.Dense(1),
   ])

   mlp_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
   mlp_history = mlp_model.fit(X_train, y_train, epochs=200, batch_size=32,
                               validation_data=(X_test, y_test), verbose=0)

   # Compare
   print(f"KAN parameters: {kan_model.count_params():,}")
   print(f"MLP parameters: {mlp_model.count_params():,}")
   print(f"KAN final val_loss: {kan_history.history['val_loss'][-1]:.6f}")
   print(f"MLP final val_loss: {mlp_history.history['val_loss'][-1]:.6f}")

**Typical result:** KANs often achieve 2-10× lower MSE for smooth functions.

----

Degree Selection Guidelines
---------------------------

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Function Type
     - Recommended Degree
     - Rationale
   * - Very smooth (polynomials)
     - 3-5
     - Exact for polynomials up to that degree
   * - Smooth (sin, exp)
     - 5-8
     - Spectral convergence
   * - Moderate complexity
     - 8-12
     - Balance expressivity/stability
   * - High frequency
     - 12-20
     - Need Clenshaw for stability
   * - Nearly discontinuous
     - Use RBF/Wavelet
     - Polynomials have Gibbs phenomenon

**Rule of thumb:** Start with ``degree=5`` and increase if validation loss plateaus.

----

Exercises
---------

**Exercise 1:** Fit the Ackley function (2D optimization test function):

.. math::

   f(x, y) = -20 \exp\left(-0.2 \sqrt{0.5(x^2 + y^2)}\right) 
             - \exp\left(0.5(\cos 2\pi x + \cos 2\pi y)\right) + e + 20

**Exercise 2:** Compare Chebyshev1st, Legendre, and Hermite on the same problem.
Which converges fastest?

**Exercise 3:** Measure training time for increasing input dimensions (2, 5, 10, 20).
How does Tucker decomposition affect training speed?

----

Next Steps
----------

- :doc:`classification_example` — Using KANs for classification
- :doc:`../layers/polynomial_kan_layers` — Complete polynomial API
- :doc:`../guides/advanced_usage` — Tucker decomposition details
