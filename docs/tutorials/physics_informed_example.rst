.. Copyright (c) 2025 René Schubotz. All rights reserved.

.. _tutorial-physics-informed:

================================================
Physics-Informed KANs
================================================

This tutorial demonstrates how to incorporate physical constraints and prior
knowledge into KAN models for scientific computing applications.

.. contents:: In This Tutorial
   :local:
   :depth: 2

----

Objective
---------

By the end of this tutorial, you will:

1. Understand physics-informed neural networks (PINNs)
2. Build KAN-based PINNs for differential equations
3. Incorporate boundary and initial conditions
4. Solve forward and inverse problems

----

Why KANs for Physics?
---------------------

Physics-Informed Neural Networks (PINNs) embed physical laws (usually PDEs)
into neural network training. KANs offer advantages:

**Theoretical:**

- Kolmogorov-Arnold theorem provides representation foundation
- Polynomial bases have well-understood approximation properties
- Spectral convergence for smooth solutions

**Practical:**

- Legendre/Chebyshev polynomials are natural for spectral methods
- Derivatives of polynomials are polynomials (closed under differentiation)
- Interpretable basis functions

**Comparison with Standard PINNs:**

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Aspect
     - MLP PINNs
     - KAN PINNs
   * - Activation
     - Fixed (tanh, sin)
     - Learnable polynomials
   * - Convergence
     - Algebraic
     - Spectral (for smooth)
   * - Interpretability
     - Black box
     - Polynomial coefficients
   * - Derivatives
     - Automatic diff
     - Polynomial derivatives

----

Example 1: Harmonic Oscillator
------------------------------

Solve the ODE: :math:`\ddot{x} + \omega^2 x = 0`

With initial conditions: :math:`x(0) = 1, \dot{x}(0) = 0`

True solution: :math:`x(t) = \cos(\omega t)`

.. code-block:: python
   :linenos:

   import tensorflow as tf
   import numpy as np
   from arnold.layers import Legendre

   omega = 2.0  # Angular frequency

   # Build KAN model for x(t)
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(1,)),
       Legendre(degree=10, units=16),
       tf.keras.layers.LayerNormalization(),
       Legendre(degree=8, units=8),
       tf.keras.layers.LayerNormalization(),
       Legendre(degree=4, units=1),
   ])

   # Physics-informed loss
   @tf.function
   def physics_loss(t):
       with tf.GradientTape(persistent=True) as tape2:
           tape2.watch(t)
           with tf.GradientTape() as tape1:
               tape1.watch(t)
               x = model(t)
           dx_dt = tape1.gradient(x, t)
       d2x_dt2 = tape2.gradient(dx_dt, t)
       
       # ODE residual: x'' + ω²x = 0
       residual = d2x_dt2 + omega**2 * x
       
       # Initial conditions
       t_0 = tf.constant([[0.0]])
       x_0 = model(t_0)
       with tf.GradientTape() as tape:
           tape.watch(t_0)
           x_0_val = model(t_0)
       dx_0 = tape.gradient(x_0_val, t_0)
       
       ic_loss = (x_0 - 1.0)**2 + dx_0**2  # x(0)=1, x'(0)=0
       
       return tf.reduce_mean(residual**2) + 10.0 * tf.reduce_mean(ic_loss)

   # Training loop
   optimizer = tf.keras.optimizers.Adam(1e-3)

   @tf.function
   def train_step():
       t = tf.random.uniform((100, 1), 0, 2*np.pi/omega)
       with tf.GradientTape() as tape:
           loss = physics_loss(t)
       grads = tape.gradient(loss, model.trainable_variables)
       optimizer.apply_gradients(zip(grads, model.trainable_variables))
       return loss

   # Train
   for epoch in range(1000):
       loss = train_step()
       if epoch % 100 == 0:
           print(f"Epoch {epoch}, Loss: {loss.numpy():.6f}")

   # Evaluate
   t_test = np.linspace(0, 2*np.pi/omega, 100).reshape(-1, 1).astype('float32')
   x_true = np.cos(omega * t_test)
   x_pred = model.predict(t_test)
   
   mse = np.mean((x_true - x_pred)**2)
   print(f"Test MSE: {mse:.8f}")

----

Example 2: Heat Equation
------------------------

Solve the 1D heat equation:

.. math::

   \frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}

Domain: :math:`x \in [0, 1], t \in [0, 1]`

Boundary conditions: :math:`u(0, t) = u(1, t) = 0`

Initial condition: :math:`u(x, 0) = \sin(\pi x)`

.. code-block:: python
   :linenos:

   import tensorflow as tf
   import numpy as np
   from arnold.layers import Legendre, Chebyshev1st

   alpha = 0.1  # Thermal diffusivity

   # Build model for u(x, t)
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(2,)),  # (x, t)
       Legendre(degree=8, units=32),
       tf.keras.layers.LayerNormalization(),
       Legendre(degree=6, units=16),
       tf.keras.layers.LayerNormalization(),
       Legendre(degree=4, units=1),
   ])

   @tf.function
   def pde_loss(x, t):
       xt = tf.concat([x, t], axis=1)
       
       with tf.GradientTape(persistent=True) as tape2:
           tape2.watch([x, t])
           with tf.GradientTape() as tape1:
               tape1.watch([x, t])
               u = model(tf.concat([x, t], axis=1))
           grads = tape1.gradient(u, [x, t])
           u_x, u_t = grads[0], grads[1]
       u_xx = tape2.gradient(u_x, x)
       
       # PDE residual: u_t - α u_xx = 0
       residual = u_t - alpha * u_xx
       return tf.reduce_mean(residual**2)

   @tf.function  
   def boundary_loss():
       t = tf.random.uniform((50, 1), 0, 1)
       
       # u(0, t) = 0
       u_left = model(tf.concat([tf.zeros_like(t), t], axis=1))
       
       # u(1, t) = 0
       u_right = model(tf.concat([tf.ones_like(t), t], axis=1))
       
       return tf.reduce_mean(u_left**2) + tf.reduce_mean(u_right**2)

   @tf.function
   def initial_loss():
       x = tf.random.uniform((50, 1), 0, 1)
       t = tf.zeros_like(x)
       
       u_pred = model(tf.concat([x, t], axis=1))
       u_true = tf.sin(np.pi * x)  # Initial condition
       
       return tf.reduce_mean((u_pred - u_true)**2)

   # Training
   optimizer = tf.keras.optimizers.Adam(1e-3)

   for epoch in range(2000):
       x = tf.random.uniform((200, 1), 0, 1)
       t = tf.random.uniform((200, 1), 0, 1)
       
       with tf.GradientTape() as tape:
           loss = (pde_loss(x, t) + 
                   10.0 * boundary_loss() + 
                   10.0 * initial_loss())
       
       grads = tape.gradient(loss, model.trainable_variables)
       optimizer.apply_gradients(zip(grads, model.trainable_variables))
       
       if epoch % 200 == 0:
           print(f"Epoch {epoch}, Loss: {loss.numpy():.6f}")

----

Choosing Bases for PINNs
------------------------

**Legendre polynomials** are ideal for PINNs because:

1. Orthogonal on :math:`[-1, 1]` with uniform weight
2. Well-conditioned for spectral methods
3. Satisfy Sturm-Liouville form (eigenfunction property)

**For specific problems:**

.. list-table::
   :widths: 30 30 40
   :header-rows: 1

   * - PDE Type
     - Recommended Basis
     - Reason
   * - Elliptic (Laplace, Poisson)
     - Legendre, Chebyshev
     - Spectral accuracy
   * - Parabolic (Heat)
     - Legendre
     - Smooth solutions
   * - Hyperbolic (Wave)
     - Chebyshev
     - Better near boundaries
   * - Quantum mechanics
     - Hermite
     - Natural for Gaussian states

----

Advanced: Inverse Problems
--------------------------

KAN PINNs can also solve inverse problems — estimating unknown parameters
from data:

.. code-block:: python

   # Unknown parameter as trainable variable
   omega_est = tf.Variable(1.0, dtype=tf.float32)

   # Include data fitting loss alongside physics loss
   @tf.function
   def total_loss(t_data, x_data):
       # Physics loss
       physics = physics_loss(t_collocation)
       
       # Data loss
       x_pred = model(t_data)
       data = tf.reduce_mean((x_pred - x_data)**2)
       
       return physics + data

   # Train with gradients w.r.t. both model and omega_est

----

Exercises
---------

**Exercise 1:** Solve the Burgers equation:

.. math::

   \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}

**Exercise 2:** Implement a PINN for the Navier-Stokes equations (2D steady flow).

**Exercise 3:** Use KAN PINNs to estimate an unknown diffusion coefficient from data.

----

Further Reading
---------------

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks"
- Lu, L., Meng, X., et al. (2021). "DeepXDE: A deep learning library for solving differential equations"

----

Next Steps
----------

- :doc:`../layers/polynomial_kan_layers` — Polynomial basis details
- :doc:`../theory/approximation_theory` — Convergence theory
- :doc:`../performance_guide` — GPU optimization for PINNs
