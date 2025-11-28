.. Copyright (c) 2025 René Schubotz. All rights reserved.
.. Licensed under the terms specified in the LICENSE file in the project root.

.. _performance-guide:

================================================
Performance & Hardware Optimization Guide
================================================

Keep KAN layers fast and numerically stable across CPU/GPU/TPU/MPS.

.. contents:: In This Chapter
   :local:
   :depth: 2

----

Executive Summary
-----------------

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Hardware
     - Key Optimizations
     - Watch Out For
   * - CUDA GPU
     - Mixed precision, large batches, XLA
     - Keep dimensions as multiples of 8
   * - Apple MPS
     - float32 only, disable XLA
     - Complex ops may hang
   * - TPU
     - Static shapes, TPUStrategy, XLA
     - No float64, batch % 128 = 0
   * - CPU
     - MKL/oneDNN, float64 for precision
     - Cache-friendly batch sizes

----

Execution Modes
---------------
- Prefer ``tf.function(jit_compile=True)`` for heavy models; disable for quick prototyping.
- Use static shapes when possible on TPU/MPS; avoid dynamic ragged indexing.
- Batch sizes: favor larger batches on GPU/TPU to amortize kernel launches; modest batches on CPU.

Data domains
------------
- Keep inputs within basis domains (e.g., :math:`[-1,1]` for orthogonal polys) via preprocessing or ``input_clip``.
- For RBFs, keep units consistent with ``grid_min/grid_max`` to avoid extreme radii.

Precision
---------
- Default compute is float32. For high-degree polynomials (>10) or ill-conditioned bases, enable float64 policy or wrap basis evaluation in a float64 cast.
- Mixed precision: safe for RBFs and wavelets; for high-degree orthogonal polynomials, prefer float32 weights with float64 evaluation.

Kernel selection
----------------
- Polynomials: use lower degrees when possible; for large degrees, prefer Clenshaw-style fused evaluation (planned).
- RBFs: Gaussian/Exponential are tensor-core friendly; avoid very small ``epsilon``/``sigma`` that produce near-Dirac spikes.
- Wavelets: keep scales moderate (``softplus`` already enforces positivity); extreme scales reduce signal-to-noise.

Polynomial evaluation paths
---------------------------
- Low degree (≤10): direct pseudo-Vandermonde is fine.
- High degree (>10): Clenshaw-style recurrences are auto-enabled for Legendre, Chebyshev (T/U/V/W), Jacobi, Gegenbauer, Hermite, Laguerre, Bessel, BannaiIto, Charlier. Uses ``tf.scan`` to reduce graph size.
- Use ``use_clenshaw=True/False`` to force or disable Clenshaw; leave as ``None`` (default) for auto-selection.
- ``promote_to_float64=True`` further stabilizes high-degree evaluation on CPU (auto-disabled on GPU for performance).

**Memory characteristics**:

Current implementation builds the full ``(batch, input_dim, degree+1)`` basis tensor even when using Clenshaw recurrence. This is O(batch × input_dim × degree) memory.

For extreme degrees (>100), consider:

1. Reducing the polynomial degree
2. Using Tucker decomposition (``core_ranks``) to reduce coefficient memory
3. Processing in smaller batches

**Future: True O(1) Clenshaw Summation**

True Clenshaw summation evaluates ``Σ c_n P_n(x)`` without storing all P_n values, using only O(1) memory in degree. This requires fusing the polynomial recurrence with coefficient contraction and is planned for a future release.

**Future: Fused Basis-Matmul Kernels**

For maximum performance, a custom TensorFlow op could fuse basis evaluation with the coefficient matmul into a single kernel, reducing memory bandwidth by avoiding intermediate tensor materialization. This is on the roadmap for performance-critical deployments.

Memory & layout
---------------
- Avoid materializing very high-degree basis tensors when the downstream layer can be fused; reduce degree or use clipping.
- Convolutional wrappers extract patches: choose stride/padding to control intermediate patch volume.

Hardware notes
--------------
- CPU: benefit from vectorized ops; enable MKL/oneDNN; consider smaller degrees to limit cache pressure.
- GPU (CUDA/ROCm): maximize contiguous tensors; favor even multiples of 8/16 for tensor cores; keep jit enabled.
- TPU: prefer static shapes; avoid Python-side control flow; keep batch and feature dims static.
- Apple MPS: see the dedicated section below.

TPU Deployment Guide
~~~~~~~~~~~~~~~~~~~~

TPU provides massive parallelism but requires careful attention to graph structure and data layout.

**Recommended Practices**

1. **Use static shapes**: TPU XLA compilation requires known shapes at compile time.

   .. code-block:: python

       # Bad: dynamic batch
       model.fit(dataset)
       
       # Good: specify batch size
       dataset = dataset.batch(128, drop_remainder=True)

2. **Avoid Python control flow in computation**: Use ``tf.cond`` and ``tf.while_loop`` instead of ``if`` and ``for`` where possible. The clenshaw_basis implementations already use ``tf.scan`` for XLA compatibility.

3. **Enable XLA compilation**:

   .. code-block:: python

       import tensorflow as tf
       
       # Ensure jit_compile is enabled (default for KAN layers)
       model.compile(optimizer='adam', loss='mse', jit_compile=True)

4. **Use TPUStrategy**:

   .. code-block:: python

       resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
       tf.config.experimental_connect_to_cluster(resolver)
       tf.tpu.experimental.initialize_tpu_system(resolver)
       strategy = tf.distribute.TPUStrategy(resolver)
       
       with strategy.scope():
           model = tf.keras.Sequential([
               Legendre(degree=10, units=64),
               Legendre(degree=10, units=1),
           ])
           model.compile(...)

5. **Batch size considerations**: TPU cores work best with large batches divisible by 8 (or 128 for TPU v2/v3). Use ``drop_remainder=True`` in batching.

**Known TPU Limitations**

- Some q-polynomial layers (AlSalamCarlitz, AskeyWilson) use ``jit_compile=False`` due to XLA limitations with dynamic tensor operations.
- float64 is not supported on TPU; ``promote_to_float64`` will be auto-disabled.
- Very high polynomial degrees (>50) may hit XLA compilation time limits.

GPU Optimization Tips
~~~~~~~~~~~~~~~~~~~~~

**CUDA/ROCm Best Practices**

1. **Tensor Core utilization**: Keep dimensions as multiples of 8 (float16) or 16 (bfloat16) for optimal tensor core mapping.

   .. code-block:: python

       # Good: units=64 is divisible by 8
       Legendre(degree=10, units=64)
       
       # Suboptimal: units=50 wastes tensor core capacity
       Legendre(degree=10, units=50)

2. **Mixed precision**: Enable mixed precision for significant speedup on RTX/A100/H100:

   .. code-block:: python

       tf.keras.mixed_precision.set_global_policy('mixed_float16')
       
       # KAN layers automatically handle the precision conversion
       model = tf.keras.Sequential([
           Legendre(degree=10, units=64),
           tf.keras.layers.Dense(10),
       ])

3. **Batch size**: Larger batches amortize kernel launch overhead. Start with 64-256 and tune.

4. **Avoid CPU-GPU transfers**: Keep data on GPU throughout training. Use ``tf.data`` pipelines with ``prefetch``.

5. **XLA compilation**: Keep ``jit_compile=True`` (default) for fused kernels:

   .. code-block:: python

       @tf.function(jit_compile=True)
       def train_step(x, y):
           with tf.GradientTape() as tape:
               pred = model(x, training=True)
               loss = loss_fn(y, pred)
           gradients = tape.gradient(loss, model.trainable_variables)
           optimizer.apply_gradients(zip(gradients, model.trainable_variables))
           return loss

**Multi-GPU Training**

Use ``MirroredStrategy`` for data-parallel training across multiple GPUs:

.. code-block:: python

    strategy = tf.distribute.MirroredStrategy()
    
    with strategy.scope():
        model = tf.keras.Sequential([
            Legendre(degree=10, units=64),
            Legendre(degree=10, units=1),
        ])
        model.compile(optimizer='adam', loss='mse')
    
    # Dataset is automatically distributed
    model.fit(train_dataset, epochs=10)

Apple Silicon (MPS) Guide
~~~~~~~~~~~~~~~~~~~~~~~~~

Apple's Metal Performance Shaders backend offers GPU acceleration on M1/M2/M3 chips. However,
some TensorFlow operations fall back to CPU or are unsupported. Follow these guidelines:

**Recommended Practices**

1. **Use float32 only**: MPS does not support float64. Setting ``tf.keras.mixed_precision.set_global_policy("float64")`` will cause failures. Stick with the default float32.

2. **Disable XLA/JIT on MPS**: Metal does not support XLA compilation. If you experience hangs or failures, disable JIT:

   .. code-block:: python

       import os
       os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
       os.environ["TF_XLA_DISABLE_MPS"] = "1"  # if available

3. **Avoid complex dtypes**: Complex numbers spill to CPU and may hang the execution pipeline.

4. **Prefer static shapes**: Dynamic shapes can cause graph recompilation overhead.

5. **Use moderate batch sizes**: Very large batches may exceed GPU memory. Start with 32-64 and tune.

**Fallback to CPU**

If you encounter MPS-related issues (hangs, incorrect results, unsupported ops), you can force CPU execution:

.. code-block:: python

    import tensorflow as tf
    
    # Option 1: Disable MPS entirely at startup
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable any GPU
    tf.config.set_visible_devices([], "GPU")
    
    # Option 2: Run specific code on CPU
    with tf.device("/CPU:0"):
        model = tf.keras.Sequential([...])
        model.fit(...)

**Checking Device Placement**

To verify where operations run:

.. code-block:: python

    import tensorflow as tf
    
    # List available devices
    print(tf.config.list_physical_devices())
    
    # Check if MPS is available
    from arnold.layers.core.kan_base import detect_hardware
    hw = detect_hardware()
    print(f"Primary hardware: {hw}")  # "mps", "gpu", "tpu", or "cpu"

**Known MPS Limitations**

- Some scatter/gather operations may hang or produce incorrect results
- Reduction operations on very large tensors may be slow
- No XLA support (``jit_compile=True`` will be ignored or fail)
- Some optimizers (e.g., Adam with amsgrad) may have issues

**Troubleshooting Checklist**

1. Update to latest tensorflow-metal plugin: ``pip install -U tensorflow-metal``
2. Ensure macOS is up-to-date (Metal improvements ship with OS updates)
3. If training hangs, try reducing batch size or model complexity
4. If results are wrong, try running on CPU to verify model correctness first
5. For production workloads, consider CUDA GPU or CPU for maximum stability

Developer utilities
-------------------
- Numerical constants live in ``arnold.utils.constants`` (``PARAM_EPS``, ``eps_for_dtype``) to keep boundary handling consistent across families.
- Use ``arnold.utils.compilation.kan_function`` to wrap basis evaluators with standard ``tf.function`` settings; this avoids decorator duplication and lets you toggle ``jit_compile`` when an op is not XLA-friendly (e.g., TensorList).
