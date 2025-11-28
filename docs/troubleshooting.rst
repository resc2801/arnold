.. Copyright (c) 2025 René Schubotz. All rights reserved.
   Licensed under the terms specified in the LICENSE file in the project root.

.. _troubleshooting:

================================================
Troubleshooting & FAQ
================================================

This guide addresses common issues, error messages, and frequently asked questions
when using ARNOLD KAN layers.

.. contents:: In This Chapter
   :local:
   :depth: 2

----

Numerical Issues
----------------

NaN or Inf in outputs
---------------------

**Symptom**: Model outputs contain ``NaN`` or ``Inf`` values during training or inference.

**Common causes and solutions**:

1. **High polynomial degree without input clipping**

   Orthogonal polynomials can grow rapidly outside their natural domain. For example,
   Chebyshev polynomials are defined on :math:`[-1, 1]`.

   .. code-block:: python

      # ❌ Risk of overflow
      layer = Legendre(degree=20, units=64)
      
      # ✅ Safe: inputs clamped to domain
      layer = Legendre(degree=20, units=64, input_clip=(-1.0, 1.0))

2. **Hermite polynomials with large inputs**

   Physicist's Hermite polynomials (:math:`H_n(x)`) grow as :math:`(2x)^n` and can overflow
   even for moderate inputs. Use the default ``input_clip`` or switch to normalized form:

   .. code-block:: python

      # ✅ Uses default input_clip=(-5.0, 5.0)
      layer = Hermite(degree=10, units=64)
      
      # ✅ Probabilist's Hermite (slower growth)
      layer = Hermite(degree=10, units=64, normalized=True)

3. **Learning rate too high**

   Polynomial coefficient gradients can be large. Try reducing the learning rate:

   .. code-block:: python

      optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

4. **Float32 precision insufficient for high degrees**

   For degree > 15, consider enabling float64 promotion on CPU:

   .. code-block:: python

      layer = Legendre(degree=25, units=64, promote_to_float64=True)

Gradient explosion during training
----------------------------------

**Symptom**: Loss becomes ``NaN`` after a few training steps.

**Solutions**:

1. **Use gradient clipping**:

   .. code-block:: python

      optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)

2. **Add kernel regularization**:

   .. code-block:: python

      layer = Chebyshev1st(
          degree=8,
          units=64,
          kernel_regularizer=tf.keras.regularizers.L2(1e-4)
      )

3. **Reduce polynomial degree** or use Tucker decomposition to reduce parameter count:

   .. code-block:: python

      layer = Legendre(
          degree=15,
          units=128,
          core_ranks=(8, 8, 8)  # Tucker decomposition
      )

----

Performance Issues
==================

Slow training on GPU
--------------------

**Symptom**: Training is slower than expected on GPU.

**Solutions**:

1. **Enable XLA compilation** (already default for most layers):

   .. code-block:: python

      model.compile(optimizer='adam', loss='mse', jit_compile=True)

2. **Use mixed precision** for Tensor Core utilization:

   .. code-block:: python

      tf.keras.mixed_precision.set_global_policy('mixed_float16')

3. **Batch sizes should be multiples of 8** (ideally 32/64/128) for optimal GPU utilization.

4. **Avoid q-polynomials with XLA**: Al-Salam–Carlitz and q-hypergeometric polynomials
   use ``jit_compile=False`` due to dynamic recurrences. Consider alternative bases.

Out of memory (OOM)
-------------------

**Symptom**: ``ResourceExhaustedError`` or GPU memory errors.

**Solutions**:

1. **Reduce batch size**.

2. **Use Tucker decomposition** to reduce memory footprint:

   .. code-block:: python

      # Full tensor: 256 × 16 × 128 = 524,288 parameters
      # Tucker: 256×8 + 16×8 + 128×8 + 8×8×8 = 3,712 parameters
      layer = Legendre(
          degree=15,
          units=128,
          input_dim=256,
          core_ranks=(8, 8, 8)
      )

3. **Use gradient checkpointing** for deep networks (TensorFlow 2.x):

   .. code-block:: python

      tf.config.experimental.set_memory_growth(
          tf.config.list_physical_devices('GPU')[0], True
      )

----

Compatibility Issues
====================

Apple Silicon (M1/M2/M3) issues
-------------------------------

**Symptom**: Errors or poor performance on Apple Silicon Macs.

**Solutions**:

1. **Install tensorflow-metal** for GPU acceleration:

   .. code-block:: bash

      pip install tensorflow-metal

2. **MPS does not support float64**. Disable float64 promotion:

   .. code-block:: python

      layer = Legendre(degree=20, units=64, promote_to_float64=False)

3. **Mixed precision may not work** with MPS. Stick to float32:

   .. code-block:: python

      tf.keras.mixed_precision.set_global_policy('float32')

See :doc:`performance_guide` for detailed MPS optimization tips.

SavedModel export fails
-----------------------

**Symptom**: Error when calling ``model.save()`` or exporting to TensorFlow Serving.

**Solutions**:

1. **Build the model first** with a concrete input shape:

   .. code-block:: python

      model.build((None, 32))  # Specify input shape
      model.save("my_model")

2. **Use get_config()/from_config()** for layer-level serialization:

   .. code-block:: python

      config = layer.get_config()
      new_layer = Legendre.from_config(config)

3. **Custom objects must be registered**. ARNOLD layers are already registered
   via ``@tf.keras.utils.register_keras_serializable``.

TPU deployment issues
---------------------

**Symptom**: Errors when training on TPU.

**Solutions**:

1. **Avoid dynamic shapes**. TPU requires static tensor shapes:

   .. code-block:: python

      # ❌ Dynamic batch size
      model(tf.zeros([None, 32]))
      
      # ✅ Fixed batch size
      model(tf.zeros([128, 32]))

2. **q-polynomials not supported** on TPU due to ``jit_compile=False`` requirement.

3. **Float64 not supported** on TPU. Set ``promote_to_float64=False``.

4. **Batch size must be divisible by 8** (or 128 for optimal performance).

See :doc:`performance_guide` for detailed TPU guidance.

----

API & Usage Questions
=====================

How do I choose the right polynomial basis?
-------------------------------------------

See :doc:`basis_selection_guide` for detailed guidance. Quick summary:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Use Case
     - Recommended Basis
     - Notes
   * - General function approximation
     - ``Chebyshev1st``, ``Legendre``
     - Near-optimal for smooth functions
   * - Bounded data in [−1, 1]
     - ``Chebyshev1st``
     - Minimizes Runge phenomenon
   * - Periodic-like data
     - ``FourierKAN``
     - Better for oscillatory patterns
   * - Unbounded inputs
     - ``Hermite`` (normalized)
     - Natural for Gaussian-weighted domains
   * - Integer-valued features
     - ``Charlier``, ``Krawtchouk``
     - Discrete orthogonality

How do I add regularization?
----------------------------

All ARNOLD layers support Keras-standard regularizers:

.. code-block:: python

   from tensorflow.keras import regularizers

   layer = Legendre(
       degree=8,
       units=64,
       kernel_regularizer=regularizers.L2(1e-4),
       bias_regularizer=regularizers.L1(1e-5),
       activity_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
   )

The ``kernel_regularizer`` applies to polynomial coefficients, Tucker factors,
RBF kernel weights, and wavelet weights.

What's the difference between ``units`` and ``output_dim``?
-----------------------------------------------------------

They are the same. ``output_dim`` is deprecated; use ``units`` for consistency
with Keras conventions:

.. code-block:: python

   # ❌ Deprecated (still works, but shows warning)
   layer = Legendre(degree=5, output_dim=64)
   
   # ✅ Preferred
   layer = Legendre(degree=5, units=64)

How do I use Tucker decomposition?
----------------------------------

Tucker decomposition reduces memory and parameter count for large layers:

.. code-block:: python

   # Factorizes the (input_dim, degree+1, output_dim) coefficient tensor
   layer = Legendre(
       degree=15,
       units=128,
       core_ranks=(8, 8, 8)  # Rank for each mode
   )

Choose ranks based on your expressivity vs. efficiency tradeoff. Smaller ranks
mean fewer parameters but potentially less capacity.

Can I train custom polynomial parameters?
-----------------------------------------

Yes! Many polynomial families have trainable parameters:

.. code-block:: python

   # Jacobi with trainable α and β
   layer = Jacobi(
       degree=8,
       units=64,
       alpha_init=0.5,
       alpha_trainable=True,
       beta_init=0.5,
       beta_trainable=True,
   )
   
   # Gegenbauer with trainable λ
   layer = Gegenbauer(
       degree=8,
       units=64,
       lmbda_init=1.0,
       lmbda_trainable=True,
   )

----

Error Messages
==============

``ValueError: `num_grids` must be at least 2``
----------------------------------------------

RBF layers require at least 2 grid points. Increase ``num_grids``:

.. code-block:: python

   layer = GaussianRBF(units=64, num_grids=8)

``ValueError: `grid_max` must be greater than `grid_min``
---------------------------------------------------------

Ensure the grid bounds are ordered correctly:

.. code-block:: python

   layer = GaussianRBF(units=64, grid_min=-1.0, grid_max=1.0)

``UserWarning: Hermite polynomials with degree=N > 15 may overflow``
--------------------------------------------------------------------

This warning indicates potential numerical instability. Options:

1. Use ``input_clip`` to bound inputs (default: ``(-5.0, 5.0)``)
2. Enable ``normalized=True`` for probabilist's Hermite
3. Enable ``promote_to_float64=True`` for higher precision

``TypeError: Cannot convert ... to EagerTensor of dtype float32``
-----------------------------------------------------------------

This usually indicates dtype mismatch. ARNOLD layers cast internally, but ensure
your data is a valid TensorFlow dtype:

.. code-block:: python

   x = tf.constant(data, dtype=tf.float32)
   output = layer(x)

----

Getting Help
============

If your issue isn't covered here:

1. **Check the API documentation** for parameter details
2. **Search GitHub Issues**: https://github.com/resc2801/arnold/issues
3. **Open a new issue** with:
   - ARNOLD version (``pip show arnold-kan``)
   - TensorFlow version (``python -c "import tensorflow; print(tensorflow.__version__)"``)
   - Minimal reproducible example
   - Full error traceback

For mathematical questions about specific polynomial families, see the
:doc:`basis_selection_guide` and individual layer docstrings which include
references to DLMF, Wikipedia, and academic papers.
