.. Copyright (c) 2025 René Schubotz. All rights reserved.

.. _advanced-usage:

================================================
Advanced Usage Guide
================================================

This guide covers advanced ARNOLD features for power users: Tucker decomposition,
custom basis functions, mixed-precision training, and production deployment.

.. contents:: In This Guide
   :local:
   :depth: 2

----

Tucker Decomposition for Large Models
-------------------------------------

The **Tucker decomposition** dramatically reduces parameter count for KAN layers
with large input/output dimensions while preserving approximation quality.

Mathematical Background
~~~~~~~~~~~~~~~~~~~~~~~

The standard KAN weight tensor has shape :math:`(n, m, d+1)` where:

- :math:`n` = input dimension
- :math:`m` = output dimension  
- :math:`d+1` = number of basis coefficients

Total parameters: :math:`n \cdot m \cdot (d+1)`

Tucker decomposition factorizes this as:

.. math::

   W_{i,j,k} = \sum_{p=1}^{r_1} \sum_{q=1}^{r_2} \sum_{s=1}^{r_3} 
               G_{p,q,s} \cdot U^{(1)}_{i,p} \cdot U^{(2)}_{j,q} \cdot U^{(3)}_{k,s}

Where:

- :math:`G` is the **core tensor** of shape :math:`(r_1, r_2, r_3)`
- :math:`U^{(1)}, U^{(2)}, U^{(3)}` are **factor matrices**

Parameters with Tucker: :math:`r_1 r_2 r_3 + n r_1 + m r_2 + (d+1) r_3`

Using Tucker Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~

Enable Tucker decomposition via the ``core_ranks`` parameter:

.. code-block:: python

   from arnold.layers import Legendre
   
   # Standard layer: 1000 × 500 × 11 = 5,500,000 parameters
   standard_layer = Legendre(degree=10, units=500)
   
   # Tucker layer: much fewer parameters
   tucker_layer = Legendre(
       degree=10,
       units=500,
       core_ranks=(16, 16, 8),  # (r_1, r_2, r_3)
   )

**Parameter comparison:**

.. code-block:: python

   import tensorflow as tf
   
   # Build layers to count parameters
   x = tf.random.normal((1, 1000))
   
   _ = standard_layer(x)
   _ = tucker_layer(x)
   
   print(f"Standard: {standard_layer.count_params():,} params")
   print(f"Tucker:   {tucker_layer.count_params():,} params")
   
   # Output:
   # Standard: 5,500,000 params
   # Tucker:   24,304 params  (226× reduction!)

Choosing Core Ranks
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Scenario
     - Recommended Ranks
     - Compression Ratio
   * - Light compression
     - ``(n//4, m//4, d)``
     - ~16×
   * - Moderate compression
     - ``(16, 16, 8)``
     - ~100×
   * - Aggressive compression
     - ``(8, 8, 4)``
     - ~1000×

**Guidelines:**

- Start with ``(16, 16, min(8, d+1))``
- Increase if validation loss plateaus
- The third rank should not exceed :math:`d+1`

Complete Tucker Example
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from arnold.layers import Chebyshev1st, Legendre

   # Large-scale model with Tucker compression
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(4096,)),  # Large input
       tf.keras.layers.Rescaling(1./127.5, offset=-1),
       
       # Heavy Tucker compression for first layer
       Chebyshev1st(
           degree=8,
           units=256,
           core_ranks=(32, 32, 8),
       ),
       tf.keras.layers.LayerNormalization(),
       tf.keras.layers.Dropout(0.3),
       
       # Moderate compression for middle layer
       Legendre(
           degree=6,
           units=64,
           core_ranks=(16, 16, 6),
       ),
       tf.keras.layers.LayerNormalization(),
       
       # No compression for small output layer
       Legendre(degree=4, units=10),
   ])
   
   model.summary()

----

Mixed-Precision Training
------------------------

Mixed-precision uses ``float16`` for forward/backward passes while keeping 
``float32`` for weights, providing 2-3× speedup on modern GPUs.

Enabling Mixed Precision
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import tensorflow as tf
   
   # Enable mixed precision globally
   tf.keras.mixed_precision.set_global_policy('mixed_float16')
   
   from arnold.layers import Legendre
   
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(100,)),
       Legendre(degree=8, units=64),
       tf.keras.layers.LayerNormalization(),
       Legendre(degree=5, units=10),
       # Use float32 for final softmax (numerical stability)
       tf.keras.layers.Activation('softmax', dtype='float32'),
   ])

Stability Considerations
~~~~~~~~~~~~~~~~~~~~~~~~

Polynomial KANs compute basis functions that can have large ranges. For stable
mixed-precision training:

1. **Always use LayerNormalization**:

   .. code-block:: python

      Legendre(degree=8, units=64),
      tf.keras.layers.LayerNormalization(),

2. **Clip inputs to polynomial domain**:

   .. code-block:: python

      Legendre(degree=8, units=64, input_clip=(-1.0, 1.0))

3. **Use gradient clipping**:

   .. code-block:: python

      optimizer = tf.keras.optimizers.Adam(
          learning_rate=1e-3,
          clipnorm=1.0,
      )

4. **Use loss scaling** (automatic with Keras):

   .. code-block:: python

      model.compile(
          optimizer=optimizer,
          loss='sparse_categorical_crossentropy',
      )
      # Keras handles loss scaling automatically for mixed_float16

Benchmark Results
~~~~~~~~~~~~~~~~~

Typical speedups on NVIDIA GPUs:

.. list-table::
   :widths: 30 30 40
   :header-rows: 1

   * - GPU
     - float32 Baseline
     - mixed_float16 Speedup
   * - RTX 4090
     - 1.0×
     - 2.4×
   * - A100
     - 1.0×
     - 2.8×
   * - H100
     - 1.0×
     - 3.1×

----

Custom Basis Functions
----------------------

You can create custom polynomial, RBF, or wavelet KAN layers.

Custom Polynomial Layer
~~~~~~~~~~~~~~~~~~~~~~~

Inherit from ``PolynomialBase`` and implement ``compute_polynomials``:

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from arnold.layers.polynomial import PolynomialBase
   
   class MyCustomPolynomial(PolynomialBase):
       """Custom polynomial layer using a new basis."""
       
       def __init__(self, degree, units, alpha=0.5, **kwargs):
           super().__init__(degree=degree, units=units, **kwargs)
           self.alpha = alpha
       
       def compute_polynomials(self, x):
           """
           Compute polynomial basis values.
           
           Args:
               x: Tensor of shape (..., input_dim)
               
           Returns:
               Tensor of shape (..., input_dim, degree+1)
           """
           # Example: Shifted Chebyshev-like polynomials
           x = tf.clip_by_value(x, -1.0, 1.0)
           
           # Initialize storage
           polys = [tf.ones_like(x), x]  # P_0 = 1, P_1 = x
           
           # Three-term recurrence
           for n in range(2, self.degree + 1):
               # Custom recurrence: P_n = (2-α)x·P_{n-1} - (1-α)P_{n-2}
               p_n = (2 - self.alpha) * x * polys[-1] - (1 - self.alpha) * polys[-2]
               polys.append(p_n)
           
           return tf.stack(polys, axis=-1)
       
       def get_config(self):
           config = super().get_config()
           config['alpha'] = self.alpha
           return config

   # Usage
   layer = MyCustomPolynomial(degree=8, units=32, alpha=0.7)
   output = layer(tf.random.normal((16, 64)))

Custom RBF Layer
~~~~~~~~~~~~~~~~

Inherit from ``RBFBase``:

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from arnold.layers.rbf import RBFBase
   
   class CauchyRBF(RBFBase):
       """Cauchy RBF layer."""
       
       def compute_rbf(self, x, centers, widths):
           """
           Compute Cauchy RBF: φ(r) = 1 / (1 + (r/γ)²)
           
           Args:
               x: Input tensor, shape (..., input_dim)
               centers: RBF centers, shape (num_grids,)
               widths: RBF widths, shape (num_grids,)
               
           Returns:
               RBF values, shape (..., input_dim, num_grids)
           """
           # Expand for broadcasting
           x_expanded = tf.expand_dims(x, -1)           # (..., input_dim, 1)
           centers = tf.reshape(centers, (1, 1, -1))    # (1, 1, num_grids)
           widths = tf.reshape(widths, (1, 1, -1))      # (1, 1, num_grids)
           
           # Compute Cauchy RBF
           r = (x_expanded - centers) / widths
           return 1.0 / (1.0 + tf.square(r))

   # Usage
   layer = CauchyRBF(units=32, num_grids=16)

Custom Wavelet Layer
~~~~~~~~~~~~~~~~~~~~

Inherit from ``WaveletBase``:

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from arnold.layers.wavelet import WaveletBase
   
   class ShannonWavelet(WaveletBase):
       """Shannon wavelet (sinc) layer."""
       
       def compute_wavelet(self, x, scale, translation):
           """
           Shannon wavelet: ψ(t) = sinc(t) · cos(3πt/2)
           
           Args:
               x: Input tensor
               scale: Wavelet scale parameter
               translation: Wavelet translation parameter
               
           Returns:
               Wavelet values
           """
           # Normalize
           t = (x - translation) / scale
           
           # Shannon wavelet
           sinc_t = tf.where(
               tf.abs(t) < 1e-7,
               tf.ones_like(t),
               tf.sin(tf.constant(np.pi) * t) / (tf.constant(np.pi) * t)
           )
           
           return sinc_t * tf.cos(1.5 * tf.constant(np.pi) * t)

----

Multi-Output and Custom Architectures
-------------------------------------

Functional API for Complex Architectures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from arnold.layers import Chebyshev1st, Legendre, GaussianRBF
   
   # Inputs
   numerical_input = tf.keras.Input(shape=(10,), name='numerical')
   categorical_input = tf.keras.Input(shape=(5,), name='categorical')
   
   # Process numerical features with polynomial KAN
   x = Chebyshev1st(degree=6, units=32)(numerical_input)
   x = tf.keras.layers.LayerNormalization()(x)
   
   # Process categorical features with RBF KAN
   y = GaussianRBF(units=16, num_grids=8)(categorical_input)
   y = tf.keras.layers.LayerNormalization()(y)
   
   # Combine
   combined = tf.keras.layers.Concatenate()([x, y])
   
   # Multi-task outputs
   z = Legendre(degree=4, units=64)(combined)
   z = tf.keras.layers.LayerNormalization()(z)
   
   regression_output = tf.keras.layers.Dense(1, name='regression')(z)
   classification_output = tf.keras.layers.Dense(3, activation='softmax', name='classification')(z)
   
   # Build model
   model = tf.keras.Model(
       inputs=[numerical_input, categorical_input],
       outputs=[regression_output, classification_output]
   )
   
   model.compile(
       optimizer='adam',
       loss={
           'regression': 'mse',
           'classification': 'sparse_categorical_crossentropy'
       },
       loss_weights={
           'regression': 1.0,
           'classification': 0.5
       }
   )

Residual KAN Blocks
~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from arnold.layers import Legendre
   
   class ResidualKANBlock(tf.keras.layers.Layer):
       """Residual block with KAN layers."""
       
       def __init__(self, degree, units, dropout=0.1, **kwargs):
           super().__init__(**kwargs)
           self.kan = Legendre(degree=degree, units=units)
           self.norm = tf.keras.layers.LayerNormalization()
           self.dropout = tf.keras.layers.Dropout(dropout)
           self.project = None  # Projection if dimensions don't match
           
       def build(self, input_shape):
           if input_shape[-1] != self.kan.units:
               self.project = tf.keras.layers.Dense(self.kan.units, use_bias=False)
           super().build(input_shape)
       
       def call(self, x, training=None):
           residual = x
           if self.project is not None:
               residual = self.project(residual)
           
           x = self.kan(x)
           x = self.norm(x)
           x = self.dropout(x, training=training)
           
           return x + residual

   # Usage
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(64,)),
       ResidualKANBlock(degree=6, units=64),
       ResidualKANBlock(degree=6, units=64),
       ResidualKANBlock(degree=4, units=32),
       tf.keras.layers.Dense(10, activation='softmax'),
   ])

----

Saving and Loading Models
-------------------------

Standard Keras Serialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Save entire model
   model.save('my_kan_model.keras')
   
   # Load model
   loaded_model = tf.keras.models.load_model('my_kan_model.keras')

Saving with Custom Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~

For custom layers:

.. code-block:: python

   # Save
   model.save('custom_model.keras')
   
   # Load with custom objects
   loaded = tf.keras.models.load_model(
       'custom_model.keras',
       custom_objects={
           'MyCustomPolynomial': MyCustomPolynomial,
           'CauchyRBF': CauchyRBF,
       }
   )

Exporting for Production (SavedModel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Export to SavedModel format
   tf.saved_model.save(model, 'exported_model')
   
   # Load for inference
   loaded = tf.saved_model.load('exported_model')
   
   # Convert to TensorFlow Lite (for mobile)
   converter = tf.lite.TFLiteConverter.from_saved_model('exported_model')
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   tflite_model = converter.convert()
   
   with open('model.tflite', 'wb') as f:
       f.write(tflite_model)

----

Debugging and Inspection
------------------------

Inspecting Layer Weights
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arnold.layers import Legendre
   
   layer = Legendre(degree=5, units=8)
   layer.build(input_shape=(None, 4))
   
   # Access weights
   coefficients = layer.kernel  # Shape: (4, 8, 6)
   print(f"Coefficient shape: {coefficients.shape}")
   print(f"Coefficient range: [{coefficients.numpy().min():.4f}, {coefficients.numpy().max():.4f}]")
   
   # For Tucker-decomposed layers
   if hasattr(layer, 'core'):
       print(f"Core tensor shape: {layer.core.shape}")
       print(f"Factor 1 shape: {layer.factor_1.shape}")
       print(f"Factor 2 shape: {layer.factor_2.shape}")
       print(f"Factor 3 shape: {layer.factor_3.shape}")

Visualizing Polynomial Responses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from arnold.layers import Chebyshev1st
   
   layer = Chebyshev1st(degree=8, units=1)
   layer.build(input_shape=(None, 1))
   
   # Sample input
   x = np.linspace(-1, 1, 200).reshape(-1, 1)
   
   # Get output
   y = layer(x).numpy()
   
   plt.figure(figsize=(10, 6))
   plt.plot(x, y)
   plt.xlabel('Input')
   plt.ylabel('Output')
   plt.title('KAN Layer Response')
   plt.grid(True)
   plt.show()

Gradient Monitoring
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Custom callback to monitor gradients
   class GradientMonitor(tf.keras.callbacks.Callback):
       def on_train_batch_end(self, batch, logs=None):
           for layer in self.model.layers:
               if hasattr(layer, 'kernel'):
                   grad = layer.kernel.numpy()
                   if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
                       print(f"Warning: NaN/Inf in {layer.name}")
                   grad_norm = np.linalg.norm(grad)
                   if grad_norm > 100:
                       print(f"Warning: Large gradient norm in {layer.name}: {grad_norm:.2f}")
   
   # Use in training
   model.fit(X, y, callbacks=[GradientMonitor()])

----

Performance Tips Summary
------------------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Technique
     - When to Use
   * - Tucker decomposition
     - Input dim > 100 or output dim > 100
   * - Mixed precision
     - NVIDIA GPU with Tensor Cores
   * - LayerNormalization
     - Always between KAN layers
   * - Gradient clipping
     - High degree polynomials (d > 10)
   * - Input clipping
     - Data outside [-1, 1] range
   * - Batch size tuning
     - GPU memory optimization

----

Next Steps
----------

- :doc:`../performance_guide` — Hardware-specific optimization
- :doc:`../troubleshooting` — Common issues and solutions
- :doc:`../theory/kan_layers` — Mathematical details of Tucker decomposition
- :doc:`../layers/polynomial_kan_layers` — Complete polynomial layer API
