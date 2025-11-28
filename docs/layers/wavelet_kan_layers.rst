.. Copyright (c) 2025 René Schubotz. All rights reserved.

.. _wavelet-kan-layers:

================================================
Wavelet KAN Layers
================================================

This chapter provides the complete API reference for wavelet-based KAN layers
in ARNOLD. Wavelet KANs use multi-scale basis functions that excel at 
capturing both localized transients and global trends in data.

.. contents:: In This Chapter
   :local:
   :depth: 2

----

Overview
--------

Wavelet KAN layers approximate univariate functions using scaled and translated wavelets:

.. math::

   \phi(x) = \sum_{j,k} c_{j,k} \, \psi\left(\frac{x - b_k}{a_j}\right)

where:

- :math:`\psi` is the mother wavelet
- :math:`a_j` are scale (dilation) parameters  
- :math:`b_k` are translation parameters
- :math:`c_{j,k}` are learnable coefficients

**Key advantages:**

- **Multi-resolution analysis** — captures features at multiple scales
- **Time-frequency localization** — optimal for non-stationary signals
- **Compact support** — many wavelets have finite support
- **Edge detection** — natural for detecting discontinuities

**When to choose Wavelet KANs:**

- Time series and signal processing
- Functions with multi-scale structure
- Data with transients or sharp transitions
- Audio, seismic, or biomedical signals

**Quick comparison:**

.. list-table::
   :widths: 25 30 45
   :header-rows: 1

   * - Wavelet Type
     - Properties
     - Best For
   * - Ricker (Mexican Hat)
     - Smooth, well-localized
     - Edge detection, general use
   * - Morlet
     - Complex, oscillatory
     - Time-frequency analysis
   * - Shannon
     - Ideal bandpass
     - Signal processing
   * - Meyer
     - Smooth in frequency
     - Smooth functions
   * - Haar
     - Piecewise constant, simplest
     - Edge detection, discontinuities
   * - Daubechies
     - Orthogonal, compact support
     - Signal/image compression
   * - Symlet
     - Near-symmetric Daubechies
     - Signal reconstruction
   * - Coiflet
     - Double vanishing moments
     - Analysis with approximation
   * - Bump
     - Compact support
     - Local features

----

Wavelet Theory Background
-------------------------

**Continuous Wavelet Transform:**

For a signal :math:`f(t)`, the continuous wavelet transform is:

.. math::

   W_f(a, b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} f(t) \, \psi^*\left(\frac{t-b}{a}\right) dt

where :math:`a` controls scale and :math:`b` controls translation.

**Admissibility Condition:**

A function :math:`\psi` is a valid wavelet if:

.. math::

   C_\psi = \int_0^\infty \frac{|\hat{\psi}(\omega)|^2}{\omega} d\omega < \infty

This requires :math:`\hat{\psi}(0) = 0`, meaning wavelets have zero mean.

**Time-Frequency Uncertainty:**

Wavelets obey the Heisenberg uncertainty principle:

.. math::

   \Delta t \cdot \Delta \omega \geq \frac{1}{4\pi}

Each wavelet type achieves different trade-offs between time and frequency localization.

----

Stability Notes
---------------

**Numerical Considerations:**

- Wavelet scales are kept positive via ``softplus`` with a small ``PARAM_EPS`` floor 
  to avoid division by zero
- Basis evaluation is wrapped in the shared ``kan_function`` decorator
- Disable ``jit_compile`` if a platform does not support XLA for these ops

**Input Handling:**

- Use ``input_clip`` if signals wander far from the expected domain
- Extreme translations/scales can reduce numerical signal-to-noise
- Normalize inputs to a reasonable range (e.g., :math:`[-1, 1]` or :math:`[0, 1]`)

**Training Stability:**

- Initialize scales at moderate values (e.g., 1.0)
- Use LayerNormalization between wavelet KAN layers
- Consider gradient clipping for high-frequency wavelets

----

Base Class
----------

All wavelet KAN layers inherit from ``WaveletBase``:

.. autoclass:: arnold.layers.core.wavelets.WaveletBase
    :members: __init__, build, call
    :show-inheritance:

**Architecture:**

.. code-block:: text

   Input x ∈ ℝ^n
       │
       ▼
   ┌────────────────────────┐
   │ Compute Scaled/Shifted │  t = (x - b_k) / a_j
   │ Coordinates            │
   └────────────────────────┘
       │
       ▼
   ┌────────────────────────┐
   │   Evaluate Wavelet     │  ψ(t) for each scale/translation
   │   Ψ ∈ ℝ^{n×J×K}        │
   └────────────────────────┘
       │
       ▼
   ┌────────────────────────┐
   │  Weight Tensor         │  Multiply with coefficients
   │   W ∈ ℝ^{n×m×J×K}      │
   └────────────────────────┘
       │
       ▼
   Output y ∈ ℝ^m

----

Available Wavelets
------------------

**Ricker Wavelet (Mexican Hat)** 

Second derivative of Gaussian. Excellent for edge detection.

.. math::

   \psi(t) = \frac{2}{\sqrt{3\sigma}\pi^{1/4}} 
             \left(1 - \frac{t^2}{\sigma^2}\right) e^{-t^2/(2\sigma^2)}

.. autoclass:: arnold.layers.core.wavelets.Ricker
    :members: __init__

**Morlet Wavelet**

Complex Gabor-like wavelet for time-frequency analysis.

.. math::

   \psi(t) = \pi^{-1/4} e^{-t^2/2} \left(e^{i\omega_0 t} - e^{-\omega_0^2/2}\right)

.. autoclass:: arnold.layers.core.wavelets.Morelet
    :members: __init__

**Meyer Wavelet**

Smooth in frequency domain, excellent localization.

Defined via its Fourier transform with smooth transition functions.

.. autoclass:: arnold.layers.core.wavelets.Meyer
    :members: __init__

**Shannon Wavelet**

Ideal bandpass filter in frequency domain.

.. math::

   \psi(t) = \text{sinc}(t/2) \cos(3\pi t/2)

.. autoclass:: arnold.layers.core.wavelets.Shannon
    :members: __init__

**Derivative of Gaussian Wavelet**

First derivative of Gaussian, also called Gaussian-1 wavelet.

.. math::

   \psi(t) = -\frac{t}{\sigma^2} e^{-t^2/(2\sigma^2)}

.. autoclass:: arnold.layers.core.wavelets.DerivativeOfGaussian
    :members: __init__

**Bump Wavelet**

Infinitely smooth with compact support.

.. math::

   \psi(t) = \begin{cases}
   e^{-1/(1-t^2)} & |t| < 1 \\
   0 & |t| \geq 1
   \end{cases}

.. autoclass:: arnold.layers.core.wavelets.Bump
    :members: __init__

**Poisson Wavelet**

Based on the Poisson kernel, useful for harmonic analysis.

.. autoclass:: arnold.layers.core.wavelets.Poisson
    :members: __init__

----

Filter-Bank Wavelets
--------------------

Filter-bank wavelets (also called discrete wavelets) are defined via their filter
coefficients and form orthonormal bases. They are the standard choice for signal
processing, image compression (JPEG 2000), and denoising applications.

**Haar Wavelet**

The simplest orthogonal wavelet. Piecewise constant with compact support [0, 1].

.. math::

   \psi(t) = \begin{cases}
   1 & 0 \leq t < 1/2 \\
   -1 & 1/2 \leq t < 1 \\
   0 & \text{otherwise}
   \end{cases}

Optimal for detecting discontinuities and edges. Equivalent to db1 (Daubechies order 1).

.. autoclass:: arnold.layers.core.wavelets.Haar
    :members: __init__

**Daubechies Wavelets**

A family of orthogonal wavelets with maximal vanishing moments for their support width.
Higher orders have smoother wavelets but wider support.

- **db1 (Haar):** 1 vanishing moment, support [0, 1]
- **db2:** 2 vanishing moments, support [0, 3]
- **db4:** 4 vanishing moments, support [0, 7]
- **db8:** 8 vanishing moments, support [0, 15]

.. math::

   \phi(t) = \sqrt{2} \sum_{n=0}^{2N-1} h[n] \, \phi(2t - n)

where :math:`h[n]` are the scaling filter coefficients.

.. autoclass:: arnold.layers.core.wavelets.Daubechies
    :members: __init__

**Symlet Wavelets**

Near-symmetric modifications of Daubechies wavelets with improved phase properties.
Preferred when phase distortion matters.

.. autoclass:: arnold.layers.core.wavelets.Symlet
    :members: __init__

**Coiflet Wavelets**

Wavelets with vanishing moments for both the wavelet and scaling function.
Named after Ronald Coifman.

- **coif1:** 6 filter coefficients
- **coif2:** 12 filter coefficients  
- **coif5:** 30 filter coefficients

.. autoclass:: arnold.layers.core.wavelets.Coiflet
    :members: __init__

----

Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import tensorflow as tf
   from arnold.layers import Ricker, Morelet
   
   # Single wavelet layer
   layer = Ricker(
       units=32,
       sigma_init=1.0,  # Initial scale
   )
   
   x = tf.random.uniform((16, 64), -1, 1)
   y = layer(x)  # Shape: (16, 32)

In a Model
~~~~~~~~~~

.. code-block:: python

   from arnold.layers import Ricker, Bump
   
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(100,)),
       
       # Multi-scale wavelet layer
       Ricker(units=64, sigma_init=1.0),
       tf.keras.layers.LayerNormalization(),
       
       # Compact support wavelet
       Bump(units=32),
       tf.keras.layers.LayerNormalization(),
       
       tf.keras.layers.Dense(10, activation='softmax'),
   ])

Time Series Processing
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arnold.layers import Morelet, Ricker
   
   # Multi-resolution time series model
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(256,)),  # Time series input
       
       # Coarse scale for trends
       Ricker(units=64, sigma_init=4.0),
       tf.keras.layers.LayerNormalization(),
       
       # Fine scale for details
       Ricker(units=64, sigma_init=0.5),
       tf.keras.layers.LayerNormalization(),
       
       tf.keras.layers.Concatenate(),
       tf.keras.layers.Dense(1),  # Prediction
   ])

Mixed with Polynomial KANs
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arnold.layers import Legendre, Ricker
   
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(64,)),
       
       # Polynomial for smooth global structure
       Legendre(degree=6, units=32),
       tf.keras.layers.LayerNormalization(),
       
       # Wavelet for local features/edges
       Ricker(units=32, sigma_init=1.0),
       tf.keras.layers.LayerNormalization(),
       
       tf.keras.layers.Dense(1),
   ])

Using Filter-Bank Wavelets
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arnold.layers import Haar, Daubechies, Symlet, Coiflet
   
   # Haar for edge detection (simplest)
   haar_layer = Haar(units=32)
   
   # Daubechies for compression (most common)
   db4_layer = Daubechies(units=32, order=4)
   
   # Symlet for minimal phase distortion
   sym4_layer = Symlet(units=32, order=4)
   
   # Coiflet for smooth approximation
   coif2_layer = Coiflet(units=32, order=2)

   # Combined model
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(128,)),
       Daubechies(units=64, order=4),
       tf.keras.layers.LayerNormalization(),
       Coiflet(units=32, order=2),
       tf.keras.layers.Dense(10, activation='softmax'),
   ])

----

Wavelet Selection Guide
-----------------------

**For Edge Detection:**

Use Ricker (Mexican Hat) — optimal for detecting singularities:

.. code-block:: python

   Ricker(units=32, sigma_init=1.0)

**For Time-Frequency Analysis:**

Use Morlet — provides good balance of time/frequency resolution:

.. code-block:: python

   Morelet(units=32, omega0=5.0)

**For Smooth Functions:**

Use Meyer — smooth in both domains:

.. code-block:: python

   Meyer(units=32)

**For Compact Features:**

Use Bump — infinitely smooth with compact support:

.. code-block:: python

   Bump(units=32)

**For Signal Compression:**

Use Daubechies — industry standard for signal/image compression:

.. code-block:: python

   Daubechies(units=32, order=4)  # db4 is most common

**For Minimal Phase Distortion:**

Use Symlet — near-symmetric with good reconstruction:

.. code-block:: python

   Symlet(units=32, order=4)

**Multi-Scale Analysis:**

Combine wavelets at different scales:

.. code-block:: python

   # Parallel multi-scale processing
   input_layer = tf.keras.layers.Input(shape=(100,))
   
   coarse = Ricker(units=32, sigma_init=4.0)(input_layer)
   medium = Ricker(units=32, sigma_init=1.0)(input_layer)
   fine = Ricker(units=32, sigma_init=0.25)(input_layer)
   
   combined = tf.keras.layers.Concatenate()([coarse, medium, fine])

----

Mathematical Properties
-----------------------

**Wavelet Energy:**

The energy of a wavelet transform is preserved:

.. math::

   \int_{-\infty}^{\infty} |f(t)|^2 dt = \frac{1}{C_\psi} \int_0^\infty \int_{-\infty}^{\infty} 
   |W_f(a,b)|^2 \frac{da \, db}{a^2}

**Localization:**

- **Ricker:** :math:`\Delta t \approx \sigma`, :math:`\Delta \omega \approx 1/\sigma`
- **Morlet:** :math:`\Delta t \approx \sigma`, :math:`\Delta \omega \approx \omega_0/\sigma`
- **Bump:** Compact support in time, smooth decay in frequency

**Vanishing Moments:**

Number of vanishing moments determines regularity detection:

- Ricker: 2 vanishing moments
- Higher-order derivatives of Gaussian: more moments

----

See Also
--------

- :doc:`polynomial_kan_layers` — Polynomial-based alternatives  
- :doc:`radial_basis_kan_layers` — RBF-based alternatives
- :doc:`../basis_selection_guide` — Choosing the right basis
- :doc:`../theory/approximation_theory` — Theoretical foundations
