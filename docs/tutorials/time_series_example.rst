.. Copyright (c) 2025 René Schubotz. All rights reserved.

.. _tutorial-time-series:

================================================
Time Series with Wavelet KANs
================================================

This tutorial demonstrates wavelet KAN layers for time series forecasting,
anomaly detection, and multi-scale signal analysis.

.. contents:: In This Tutorial
   :local:
   :depth: 2

----

Objective
---------

By the end of this tutorial, you will:

1. Use wavelet KANs for time series prediction
2. Understand multi-scale analysis with wavelets
3. Combine wavelets with polynomial KANs
4. Build models for anomaly detection

----

Why Wavelets for Time Series?
-----------------------------

Wavelets provide **time-frequency localization** — they capture both:

- **When** something happens (temporal localization)
- **What frequency** is present (spectral localization)

This makes them ideal for:

- Non-stationary signals (changing frequency content)
- Transient detection (sudden changes)
- Multi-scale patterns (trends + details)

Compared to polynomials:

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Property
     - Polynomials
     - Wavelets
   * - Best for
     - Smooth, stationary
     - Non-stationary, transients
   * - Frequency content
     - Low-frequency bias
     - Multi-scale
   * - Edge detection
     - Poor (Gibbs)
     - Excellent

----

Basic Time Series Prediction
----------------------------

Let's predict a synthetic time series with multiple frequency components:

.. code-block:: python
   :linenos:

   import tensorflow as tf
   import numpy as np
   from arnold.layers import Ricker, Morelet, Legendre

   # Generate multi-frequency signal
   np.random.seed(42)
   t = np.linspace(0, 10, 1000)
   signal = (np.sin(2 * np.pi * t) +           # Low frequency
             0.5 * np.sin(10 * np.pi * t) +    # Medium frequency
             0.2 * np.sin(30 * np.pi * t) +    # High frequency
             0.1 * np.random.randn(1000))       # Noise
   signal = signal.astype('float32')

   # Create sequences (look-back = 50)
   def create_sequences(data, lookback):
       X, y = [], []
       for i in range(len(data) - lookback):
           X.append(data[i:i+lookback])
           y.append(data[i+lookback])
       return np.array(X), np.array(y)

   lookback = 50
   X, y = create_sequences(signal, lookback)
   y = y.reshape(-1, 1)

   # Normalize to [-1, 1]
   X = X / np.abs(X).max()
   y = y / np.abs(y).max()

   # Split
   split = 800
   X_train, X_test = X[:split], X[split:]
   y_train, y_test = y[:split], y[split:]

   # Build wavelet-based model
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(lookback,)),
       
       # Multi-scale wavelet layers
       Ricker(units=32, sigma_init=1.0),   # Coarse scale
       tf.keras.layers.LayerNormalization(),
       
       Ricker(units=16, sigma_init=0.3),   # Fine scale
       tf.keras.layers.LayerNormalization(),
       
       # Combine with polynomial for smooth output
       Legendre(degree=3, units=1),
   ])

   model.compile(
       optimizer=tf.keras.optimizers.Adam(1e-3),
       loss='mse',
   )

   history = model.fit(
       X_train, y_train,
       epochs=100,
       batch_size=32,
       validation_data=(X_test, y_test),
       verbose=1,
   )

   # Evaluate
   mse = model.evaluate(X_test, y_test)
   print(f"Test MSE: {mse:.6f}")

----

Multi-Scale Architecture
------------------------

For signals with multiple time scales, use parallel wavelet branches:

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from arnold.layers import Ricker, Bump

   # Functional API for parallel branches
   input_layer = tf.keras.layers.Input(shape=(lookback,))

   # Coarse scale branch (trends)
   coarse = Ricker(units=16, sigma_init=2.0)(input_layer)
   coarse = tf.keras.layers.LayerNormalization()(coarse)

   # Medium scale branch
   medium = Ricker(units=16, sigma_init=0.5)(input_layer)
   medium = tf.keras.layers.LayerNormalization()(medium)

   # Fine scale branch (details)
   fine = Bump(units=16)(input_layer)
   fine = tf.keras.layers.LayerNormalization()(fine)

   # Concatenate multi-scale features
   combined = tf.keras.layers.Concatenate()([coarse, medium, fine])
   output = tf.keras.layers.Dense(1)(combined)

   model = tf.keras.Model(inputs=input_layer, outputs=output)

----

Anomaly Detection
-----------------

Wavelets excel at detecting anomalies (sudden changes, outliers):

.. code-block:: python
   :linenos:

   from arnold.layers import Ricker

   # Create signal with anomalies
   normal_signal = np.sin(2 * np.pi * np.linspace(0, 10, 1000))
   # Insert anomalies
   anomaly_indices = [200, 500, 800]
   for idx in anomaly_indices:
       normal_signal[idx:idx+10] += 2.0  # Sudden spike

   # Build autoencoder for anomaly detection
   encoder = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(50,)),
       Ricker(units=32, sigma_init=0.5),
       tf.keras.layers.LayerNormalization(),
       Ricker(units=8, sigma_init=1.0),   # Bottleneck
   ])

   decoder = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(8,)),
       Ricker(units=32, sigma_init=0.5),
       tf.keras.layers.LayerNormalization(),
       tf.keras.layers.Dense(50),
   ])

   # Train on normal data, anomalies will have high reconstruction error

----

Wavelet Selection for Time Series
---------------------------------

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Wavelet
     - Characteristics
     - Best For
   * - Ricker
     - Smooth, symmetric
     - General time series
   * - Morlet
     - Complex, oscillatory
     - Frequency analysis
   * - Bump
     - Compact support
     - Transient detection
   * - DOG
     - Antisymmetric
     - Edge/step detection

----

Exercises
---------

**Exercise 1:** Predict stock prices (use ``yfinance`` to download data).

**Exercise 2:** Build an ECG anomaly detector using wavelet KANs.

**Exercise 3:** Compare Ricker wavelets at different scales (σ = 0.1, 0.5, 1.0, 2.0).

----

Next Steps
----------

- :doc:`physics_informed_example` — Physics-informed neural networks
- :doc:`../layers/wavelet_kan_layers` — Complete wavelet API
- :doc:`../basis_selection_guide` — Choosing between wavelets and polynomials
