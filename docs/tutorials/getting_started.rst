.. Copyright (c) 2025 René Schubotz. All rights reserved.

.. _tutorial-getting-started:

================================================
Getting Started with ARNOLD
================================================

Welcome to ARNOLD! This tutorial introduces the fundamental concepts of 
Kolmogorov-Arnold Networks and guides you through building your first model.

.. contents:: In This Tutorial
   :local:
   :depth: 2

----

Objective
---------

By the end of this tutorial, you will:

1. Understand what makes KAN layers different from traditional dense layers
2. Install ARNOLD and verify the installation
3. Build and train your first KAN model
4. Know how to choose basic hyperparameters

----

What are KAN Layers?
--------------------

Traditional neural networks use the **MLP architecture**:

.. math::

   y = \sigma(Wx + b)

where :math:`\sigma` is a fixed activation function (ReLU, tanh, etc.) applied 
element-wise after a linear transformation.

**KAN layers** replace this with **learnable univariate functions**:

.. math::

   y_j = \sum_{i=1}^{n} \phi_{ij}(x_i)

Each :math:`\phi_{ij}` is a learnable function, typically represented as a 
polynomial, RBF, or wavelet expansion:

.. math::

   \phi_{ij}(x) = \sum_{k=0}^{d} c_{ijk} P_k(x)

where :math:`P_k` are basis functions and :math:`c_{ijk}` are learnable coefficients.

**Key advantages:**

- Learn the *shape* of activation functions, not just weights
- Better function approximation for smooth functions
- More interpretable (can inspect learned univariate functions)
- Strong theoretical foundations (Kolmogorov-Arnold theorem)

**Architecture comparison:**

.. code-block:: text

   MLP Layer:                    KAN Layer:
   
   Input x ∈ ℝⁿ                  Input x ∈ ℝⁿ
       │                             │
       ▼                             ▼
   ┌─────────┐                  ┌─────────────┐
   │  Wx + b │  Linear          │ V(x) @ C    │  Basis + Coefficients
   └─────────┘                  └─────────────┘
       │                             │
       ▼                             ▼
   ┌─────────┐                  Output y ∈ ℝᵐ
   │   σ(·)  │  Fixed activation     (nonlinear already!)
   └─────────┘
       │
       ▼
   Output y ∈ ℝᵐ

----

Installation
------------

**Prerequisites:**

- Python 3.10 or higher
- TensorFlow 2.15 or higher

**Install via pip:**

.. code-block:: bash

   pip install arnold-kan

**For Apple Silicon (M1/M2/M3):**

.. code-block:: bash

   pip install tensorflow-metal  # GPU acceleration

**Verify installation:**

.. code-block:: python

   import arnold
   from arnold.layers import Legendre
   
   print(f"ARNOLD version: {arnold.__version__}")
   
   # Quick test
   layer = Legendre(degree=5, units=10)
   print("✓ Installation successful!")

----

Your First KAN Model
--------------------

Let's build a simple model to fit a function :math:`f(x) = \sin(\pi x)`:

.. code-block:: python
   :linenos:

   import tensorflow as tf
   import numpy as np
   from arnold.layers import Chebyshev1st, Legendre

   # 1. Generate training data
   np.random.seed(42)
   x_train = np.random.uniform(-1, 1, (1000, 1)).astype('float32')
   y_train = np.sin(np.pi * x_train)

   # 2. Build KAN model
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(1,)),
       Chebyshev1st(degree=5, units=8),    # First KAN layer
       tf.keras.layers.LayerNormalization(),
       Legendre(degree=3, units=1),         # Output layer
   ])

   # 3. Compile
   model.compile(
       optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
       loss='mse',
   )

   # 4. Train
   history = model.fit(
       x_train, y_train,
       epochs=100,
       batch_size=32,
       validation_split=0.2,
       verbose=1,
   )

   # 5. Evaluate
   x_test = np.linspace(-1, 1, 100).reshape(-1, 1).astype('float32')
   y_test = np.sin(np.pi * x_test)
   y_pred = model.predict(x_test)

   # Compute error
   mse = np.mean((y_test - y_pred)**2)
   print(f"Test MSE: {mse:.6f}")

**Expected output:**

.. code-block:: text

   Epoch 100/100
   25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0012 - val_loss: 0.0008
   Test MSE: 0.000523

----

Understanding the Code
----------------------

**Line 11-16: Model Architecture**

.. code-block:: python

   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(1,)),
       Chebyshev1st(degree=5, units=8),     # KAN layer with 8 outputs
       tf.keras.layers.LayerNormalization(),
       Legendre(degree=3, units=1),          # KAN layer with 1 output
   ])

- ``Chebyshev1st(degree=5, units=8)`` — Uses Chebyshev polynomials up to degree 5, 
  outputs 8 values. This is like ``Dense(8)`` but with learnable nonlinearity.

- ``LayerNormalization()`` — Stabilizes training between KAN layers. **Always 
  include this** between KAN layers.

- ``Legendre(degree=3, units=1)`` — Output layer using Legendre polynomials.

**Why Two Different Polynomial Types?**

Mixing polynomial types can improve expressivity. Chebyshev is optimal for 
general approximation; Legendre has uniform weight (good for physical problems).

**Key Hyperparameters:**

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Parameter
     - Meaning
     - Typical Values
   * - ``degree``
     - Maximum polynomial degree
     - 3-8 for most tasks
   * - ``units``
     - Output dimension (like ``Dense``)
     - 8-128 typically
   * - ``use_bias``
     - Add bias term
     - True (default)

----

Visualizing the Result
----------------------

.. code-block:: python

   import matplotlib.pyplot as plt

   plt.figure(figsize=(10, 6))
   plt.plot(x_test, y_test, 'b-', label='True: sin(πx)', linewidth=2)
   plt.plot(x_test, y_pred, 'r--', label='KAN Prediction', linewidth=2)
   plt.xlabel('x')
   plt.ylabel('y')
   plt.title('KAN Function Approximation')
   plt.legend()
   plt.grid(True)
   plt.show()

The KAN should closely match the sine function, demonstrating the network's
ability to learn smooth functions with few parameters.

----

Common Mistakes
---------------

**1. Forgetting LayerNormalization**

.. code-block:: python

   # Wrong: can cause unstable training
   model = tf.keras.Sequential([
       Chebyshev1st(degree=8, units=32),
       Legendre(degree=5, units=10),  # ← No normalization between layers!
   ])

   # Correct: add normalization
   model = tf.keras.Sequential([
       Chebyshev1st(degree=8, units=32),
       tf.keras.layers.LayerNormalization(),  # ← Add this!
       Legendre(degree=5, units=10),
   ])

**2. Inputs Outside Polynomial Domain**

Most polynomials expect inputs in :math:`[-1, 1]`:

.. code-block:: python

   # If your data is in [0, 255] (e.g., images)
   model = tf.keras.Sequential([
       tf.keras.layers.Rescaling(1./127.5, offset=-1),  # Normalize to [-1, 1]
       Chebyshev1st(degree=5, units=32),
       ...
   ])

**3. Starting with Too High Degree**

.. code-block:: python

   # Start simple
   layer = Legendre(degree=3, units=16)
   
   # Increase only if needed
   layer = Legendre(degree=8, units=16)

----

Exercises
---------

**Exercise 1:** Modify the model to fit :math:`f(x) = x^3 - x`:

.. code-block:: python

   y_train = x_train**3 - x_train

How does the error compare to the sine function?

**Exercise 2:** Try using only Hermite polynomials:

.. code-block:: python

   from arnold.layers import Hermite
   
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(1,)),
       Hermite(degree=5, units=8),
       tf.keras.layers.LayerNormalization(),
       Hermite(degree=3, units=1),
   ])

How does training differ?

**Exercise 3:** Experiment with different learning rates (1e-4, 1e-3, 1e-2, 1e-1). 
What's the optimal learning rate for this problem?

----

Next Steps
----------

- :doc:`regression_example` — Multi-dimensional function approximation
- :doc:`classification_example` — Using KANs for classification
- :doc:`../basis_selection_guide` — How to choose the right polynomial basis
- :doc:`../theory/kan_layers` — Mathematical details of KAN architecture
