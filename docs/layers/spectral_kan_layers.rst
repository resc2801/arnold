.. Copyright (c) 2025 René Schubotz. All rights reserved.

.. _spectral-kan-layers:

====================================================
Spectral KAN Layers
====================================================

Spectral basis functions for periodic and frequency-domain representations.

Overview
--------

Spectral layers use trigonometric or random Fourier features as basis functions,
making them ideal for:

- Periodic functions
- Signal processing applications
- Kernel approximation
- Frequency-domain analysis

Available Layers
----------------

FourierKAN
~~~~~~~~~~

Trigonometric basis expansion using sine and cosine functions.

.. code-block:: python

   from arnold.layers import FourierKAN
   
   layer = FourierKAN(
       units=64,
       degree=16,           # Number of frequency components
       frequency=1.0,       # Base frequency (omega)
       trainable_frequency=True,
   )

**Mathematical Definition:**

For degree :math:`n`, the basis consists of :math:`2n + 1` functions:

.. math::

   \phi_0(x) = 1, \quad
   \phi_{2k-1}(x) = \cos(k \omega x), \quad
   \phi_{2k}(x) = \sin(k \omega x)

for :math:`k = 1, 2, \ldots, n`.

**Parameters:**

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``degree``
     - Required
     - Number of frequency components (determines :math:`2n+1` basis functions)
   * - ``frequency``
     - 1.0
     - Base angular frequency :math:`\omega`
   * - ``trainable_frequency``
     - True
     - Whether to learn :math:`\omega` during training

**Use Cases:**

- Periodic time series
- Fourier series approximation
- Smooth periodic functions


RandomFourierFeatures
~~~~~~~~~~~~~~~~~~~~~

Random Fourier feature expansion for kernel approximation.

.. code-block:: python

   from arnold.layers import RandomFourierFeatures
   
   layer = RandomFourierFeatures(
       units=64,
       num_features=128,       # Number of random features
       kernel_scale=1.0,       # RBF kernel bandwidth
       trainable_frequencies=True,
       trainable_phases=True,
       seed=42,
   )

**Mathematical Definition:**

Approximates shift-invariant kernels via:

.. math::

   \phi(x) = \sqrt{\frac{2}{D}} \cos(\omega^T x + b)

where :math:`\omega \sim p(\omega)` is sampled from the kernel's spectral density
and :math:`b \sim \text{Uniform}[0, 2\pi]`.

For Gaussian RBF kernels:

.. math::

   k(x, y) = \exp\left(-\frac{\|x - y\|^2}{2\sigma^2}\right) 
   \approx \phi(x)^T \phi(y)

**Parameters:**

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``num_features``
     - Required
     - Number of random Fourier features :math:`D`
   * - ``kernel_scale``
     - 1.0
     - RBF kernel bandwidth :math:`\sigma`
   * - ``trainable_frequencies``
     - True
     - Whether to learn frequency vectors
   * - ``trainable_phases``
     - True
     - Whether to learn phase offsets

**Use Cases:**

- Scalable kernel methods
- Large-scale SVM approximation
- Non-linear feature extraction


Registry Access
---------------

All spectral layers are accessible via the registry:

.. code-block:: python

   from arnold.layers.core.registry import get_layer, list_layers_by_category
   
   # Get layer by name
   layer = get_layer("fourier", units=32, degree=8)
   
   # List all spectral layers
   spectral_layers = list_layers_by_category()["spectral"]
   print(spectral_layers)  # ['fourier', 'rff', 'random_fourier_features']

**Aliases:**

- ``fourier``, ``fourier_kan`` → FourierKAN
- ``rff``, ``random_fourier``, ``random_fourier_features`` → RandomFourierFeatures


Performance Considerations
--------------------------

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Layer
     - Parameter Count
     - GPU Efficiency
     - TPU Efficiency
   * - FourierKAN
     - O(n × m × (2d+1))
     - ✓✓✓
     - ✓✓✓
   * - RandomFourierFeatures
     - O(n × m × D)
     - ✓✓✓
     - ✓✓✓

Where n = input dim, m = output units, d = degree, D = num_features.

Both layers are fully XLA-compatible and support mixed precision training.

