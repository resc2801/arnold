.. Copyright (c) 2025 René Schubotz. All rights reserved.

.. _kan_layers:

================================================
KAN Layer Mathematics
================================================

This chapter provides the rigorous mathematical formulation of Kolmogorov-Arnold Network 
layers as implemented in ARNOLD. We show how the abstract Kolmogorov-Arnold representation 
theorem is translated into practical, differentiable, GPU-friendly layer operations.

.. contents:: Chapter Contents
   :local:
   :depth: 2

----

The Alternant Tensor Framework
------------------------------

At the heart of ARNOLD's KAN layers is the **alternant tensor**, a generalization of the 
classical alternant matrix from interpolation theory.

Univariate Function Families
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`\{f_{i,j}: [0,1] \to \mathbb{R}\}` be a family of continuous univariate functions 
indexed by :math:`i \in \{1, \ldots, m\}` (output dimension) and :math:`j \in \{1, \ldots, n\}` 
(input dimension).

.. admonition:: Definition — Alternant Tensor

   For a mode-2 input tensor :math:`\mathbf{x} \in \mathbb{R}^{B \times n}` (batch size :math:`B`, 
   input dimension :math:`n`), the **alternant tensor** :math:`A[f_{i,j}](\mathbf{x}) \in \mathbb{R}^{B \times m \times n}` 
   is defined by:

   .. math::

      A[f_{i,j}](\mathbf{x})_{b,i,j} = f_{i,j}(x_{b,j})

   where :math:`b \in \{1, \ldots, B\}`, :math:`i \in \{1, \ldots, m\}`, :math:`j \in \{1, \ldots, n\}`.

The alternant tensor applies each function :math:`f_{i,j}` pointwise to the corresponding 
input coordinate :math:`x_{b,j}` for each batch element :math:`b`.

The KAN Layer Operation
~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: Definition — KAN Layer

   A KAN layer with input dimension :math:`n` and output dimension :math:`m` computes:

   .. math::

      \text{KAN}(\mathbf{x})_{b,i} = \sum_{j=1}^{n} A[f_{i,j}](\mathbf{x})_{b,i,j} = \sum_{j=1}^{n} f_{i,j}(x_{b,j})

   This is a contraction of the alternant tensor over the input dimension axis.

**Comparison with Dense Layers:**

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Operation
     - Dense Layer
     - KAN Layer
   * - Formula
     - :math:`\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}`
     - :math:`y_i = \sum_j f_{i,j}(x_j)`
   * - Parameters
     - :math:`m \times n` weights + :math:`m` biases
     - :math:`m \times n` functions
   * - Nonlinearity
     - External (applied after)
     - Internal (functions :math:`f_{i,j}`)

----

Polynomial KAN Layers
---------------------

The key insight is to represent each :math:`f_{i,j}` as a **polynomial expansion** with 
learnable coefficients.

The Pseudo-Vandermonde Tensor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given a polynomial basis :math:`\mathcal{B} = \{B_0, B_1, \ldots, B_K\}` where :math:`B_k` 
has degree :math:`k`:

.. admonition:: Definition — Pseudo-Vandermonde Tensor

   For input :math:`\mathbf{x} \in \mathbb{R}^{B \times n}` and polynomial basis :math:`\mathcal{B}`, 
   the **pseudo-Vandermonde tensor** :math:`V[\mathcal{B}](\mathbf{x}) \in \mathbb{R}^{B \times n \times (K+1)}` is:

   .. math::

      V[\mathcal{B}](\mathbf{x})_{b,j,k} = B_k(x_{b,j})

   Each slice :math:`V_{b,:,:}` is analogous to a Vandermonde matrix, but uses general 
   polynomial basis functions instead of monomials.

**Examples of Polynomial Bases:**

- **Monomial**: :math:`B_k(x) = x^k`
- **Chebyshev** (1st kind): :math:`T_k(x) = \cos(k \arccos x)`
- **Legendre**: :math:`P_k(x)` satisfying :math:`(k+1)P_{k+1} = (2k+1)xP_k - kP_{k-1}`
- **Hermite**: :math:`H_k(x)` satisfying :math:`H_{k+1} = 2xH_k - 2kH_{k-1}`

The Polynomial Alternant
~~~~~~~~~~~~~~~~~~~~~~~~

We represent each :math:`f_{i,j}` as a linear combination of basis polynomials:

.. math::

   f_{i,j}(x) = \sum_{k=0}^{K} c_{j,k,i} \, B_k(x)

where :math:`\mathbf{c} \in \mathbb{R}^{n \times (K+1) \times m}` is a **learnable coefficient tensor**.

.. admonition:: Definition — Polynomial Alternant Tensor

   Given basis :math:`\mathcal{B}`, input :math:`\mathbf{x}`, and coefficients :math:`\mathbf{c}`:

   .. math::

      A[\mathcal{B}](\mathbf{x})_{b,i,j} = \sum_{k=0}^{K} c_{j,k,i} \, B_k(x_{b,j})

   This is the contraction:

   .. math::

      A[\mathcal{B}](\mathbf{x}) = \mathbf{c} \cdot V[\mathcal{B}](\mathbf{x})

The Polynomial KAN Layer
~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: Definition — Polynomial KAN Layer

   A polynomial KAN layer with input dimension :math:`n`, output dimension :math:`m`, 
   polynomial basis :math:`\mathcal{B}` of degree :math:`K`, and optional bias computes:

   .. math::

      \text{PolyKAN}(\mathbf{x})_{b,i} = \sum_{j=1}^{n} \sum_{k=0}^{K} c_{j,k,i} \, B_k(x_{b,j}) + b_i

   In Einstein notation: :math:`y_{bi} = c_{jki} \, V_{bjk} + b_i` (summing over :math:`j, k`).

**Implementation (einsum):**

.. code-block:: python

   # V: (batch, input_dim, degree+1) — pseudo-Vandermonde
   # c: (input_dim, degree+1, output_dim) — coefficients
   # y: (batch, output_dim)
   
   y = tf.einsum('bjk,jki->bi', V, c) + bias

----

Tucker Decomposition for Efficiency
-----------------------------------

For large layers, the coefficient tensor :math:`\mathbf{c} \in \mathbb{R}^{n \times (K+1) \times m}` 
can be prohibitively large (e.g., :math:`256 \times 16 \times 128 = 524,288` parameters).

**Tucker Decomposition** provides a low-rank factorization:

.. math::

   c_{j,k,i} \approx \sum_{r_1, r_2, r_3} G_{r_1, r_2, r_3} \, A_{j,r_1} \, B_{k,r_2} \, C_{i,r_3}

where:

- :math:`\mathbf{G} \in \mathbb{R}^{R_1 \times R_2 \times R_3}` is a small **core tensor**
- :math:`\mathbf{A} \in \mathbb{R}^{n \times R_1}`, :math:`\mathbf{B} \in \mathbb{R}^{(K+1) \times R_2}`, :math:`\mathbf{C} \in \mathbb{R}^{m \times R_3}` are **factor matrices**

**Parameter Reduction:**

.. math::

   \underbrace{n \times (K+1) \times m}_{\text{Full}} \quad \to \quad 
   \underbrace{R_1 R_2 R_3 + nR_1 + (K+1)R_2 + mR_3}_{\text{Tucker}}

For :math:`n = 256, K = 15, m = 128` with ranks :math:`R_1 = R_2 = R_3 = 8`:

- Full: 524,288 parameters
- Tucker: 512 + 2,048 + 128 + 1,024 = **3,712 parameters** (140× reduction)

**Usage in ARNOLD:**

.. code-block:: python

   layer = Legendre(
       degree=15,
       units=128,
       core_ranks=(8, 8, 8)  # Enable Tucker decomposition
   )

----

Radial Basis Function KAN Layers
--------------------------------

RBF layers use **localized radial kernels** instead of polynomials.

The RBF Kernel Tensor
~~~~~~~~~~~~~~~~~~~~~

Given :math:`K` grid centers :math:`\{\mu_1, \ldots, \mu_K\} \subset [\text{min}, \text{max}]`:

.. admonition:: Definition — Radial Basis Tensor

   For input :math:`\mathbf{x} \in \mathbb{R}^{B \times n}`, grid :math:`\boldsymbol{\mu}`, 
   and radial kernel :math:`\phi`:

   .. math::

      R[\phi](\mathbf{x})_{b,j,k} = \phi\left(\frac{\|x_{b,j} - \mu_k\|}{h}\right)

   where :math:`h = (\max - \min) / (K - 1)` is the grid spacing.

**Common Radial Kernels:**

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Kernel
     - Formula :math:`\phi(r)`
     - Properties
   * - Gaussian
     - :math:`\exp(-(\varepsilon r)^2)`
     - Infinitely smooth, localized
   * - Multiquadric
     - :math:`\sqrt{1 + (\varepsilon r)^2}`
     - Grows with distance
   * - Inverse Multiquadric
     - :math:`1 / \sqrt{1 + (\varepsilon r)^2}`
     - Bounded, decaying
   * - Thin Plate Spline
     - :math:`r^2 \ln r`
     - Classic interpolation

The RBF KAN Layer
~~~~~~~~~~~~~~~~~

.. admonition:: Definition — RBF KAN Layer

   An RBF KAN layer with input dimension :math:`n`, output dimension :math:`m`, 
   :math:`K` grid points, and kernel :math:`\phi` computes:

   .. math::

      \text{RBF-KAN}(\mathbf{x})_{b,i} = \sum_{j=1}^{n} \sum_{k=1}^{K} w_{j,k,i} \, \phi\left(\frac{\|x_{b,j} - \mu_k\|}{h}\right) + b_i

**Implementation:**

.. code-block:: python

   layer = GaussianRBF(
       units=64,
       num_grids=16,        # Number of grid points
       grid_min=-1.0,       # Grid lower bound
       grid_max=1.0,        # Grid upper bound
       epsilon_init=1.0,    # Shape parameter
   )

----

Wavelet KAN Layers
------------------

Wavelet layers use **translated and scaled mother wavelets**.

The Wavelet Transform
~~~~~~~~~~~~~~~~~~~~~

Given a mother wavelet :math:`\psi: \mathbb{R} \to \mathbb{R}`, the daughter wavelets are:

.. math::

   \psi_{s,t}(x) = \frac{1}{\sqrt{s}} \psi\left(\frac{x - t}{s}\right)

where :math:`s > 0` is the **scale** and :math:`t \in \mathbb{R}` is the **translation**.

The :math:`1/\sqrt{s}` normalization ensures :math:`\|\psi_{s,t}\|_2 = \|\psi\|_2`.

.. admonition:: Definition — Wavelet KAN Layer

   A wavelet KAN layer with learnable scales :math:`\mathbf{s}` and translations :math:`\mathbf{t}` computes:

   .. math::

      \text{Wavelet-KAN}(\mathbf{x})_{b,i} = \sum_{j=1}^{n} w_{j,i} \, \frac{1}{\sqrt{s_{j,i}}} 
      \psi\left(\frac{x_{b,j} - t_{j,i}}{s_{j,i}}\right) + b_i

**Common Mother Wavelets:**

- **Ricker (Mexican Hat)**: :math:`\psi(t) = (1 - t^2) e^{-t^2/2}` (second derivative of Gaussian)
- **Morlet**: :math:`\psi(t) = e^{i\omega_0 t} e^{-t^2/2}` (Gaussian-modulated sinusoid)
- **DOG**: :math:`\psi(t) = -t \, e^{-t^2/2}` (first derivative of Gaussian)

**Usage:**

.. code-block:: python

   layer = Ricker(
       units=64,
       sigma_init=1.0,      # Width parameter
       sigma_trainable=True
   )

----

Numerical Stability
-------------------

ARNOLD implements several techniques for numerical stability:

Clenshaw Recurrence
~~~~~~~~~~~~~~~~~~~

For high-degree polynomial evaluation, direct computation of :math:`B_k(x)` can be unstable. 
The **Clenshaw algorithm** evaluates :math:`\sum_k c_k B_k(x)` using the recurrence relation 
**backwards**, accumulating the sum without storing all basis values.

For three-term recurrence :math:`B_{k+1} = \alpha_k x B_k + \beta_k B_{k-1}`:

.. code-block:: python

   def clenshaw(x, coeffs, alpha, beta):
       b_k2, b_k1 = 0, 0
       for k in range(len(coeffs)-1, -1, -1):
           b_k = coeffs[k] + alpha[k] * x * b_k1 + beta[k] * b_k2
           b_k2, b_k1 = b_k1, b_k
       return b_k

This is more stable than forward evaluation for high degrees.

Input Clipping
~~~~~~~~~~~~~~

Orthogonal polynomials are well-conditioned on their natural domain but can explode outside:

.. code-block:: python

   # Chebyshev: defined on [-1, 1]
   layer = Chebyshev1st(degree=10, units=64, input_clip=(-1.0, 1.0))
   
   # Hermite: default clip to prevent overflow
   layer = Hermite(degree=15, units=64)  # input_clip=(-5.0, 5.0) by default

Softplus Constraints
~~~~~~~~~~~~~~~~~~~~

Parameters that must be positive (RBF widths, wavelet scales) are stored as **logits** 
and transformed via softplus:

.. math::

   \sigma = \text{softplus}(\sigma_{\text{logits}}) + \varepsilon = \ln(1 + e^{\sigma_{\text{logits}}}) + \varepsilon

This ensures :math:`\sigma > \varepsilon` with smooth gradients everywhere.

Float64 Promotion
~~~~~~~~~~~~~~~~~

For high polynomial degrees on CPU, ARNOLD can automatically promote computations to float64:

.. code-block:: python

   layer = Legendre(
       degree=25,
       units=64,
       promote_to_float64=True,  # Compute basis in float64
       precision_threshold=10    # Trigger threshold
   )

The output is cast back to the original dtype (e.g., float32).

----

Gradient Flow
-------------

All KAN layer operations are differentiable, enabling gradient-based training:

**Polynomial KAN Gradients:**

.. math::

   \frac{\partial \mathcal{L}}{\partial c_{j,k,i}} = \sum_b \frac{\partial \mathcal{L}}{\partial y_{b,i}} \, B_k(x_{b,j})

**RBF KAN Gradients:**

.. math::

   \frac{\partial \mathcal{L}}{\partial w_{j,k,i}} = \sum_b \frac{\partial \mathcal{L}}{\partial y_{b,i}} \, \phi_k(x_{b,j})

For trainable parameters (e.g., Jacobi's :math:`\alpha, \beta` or RBF's :math:`\varepsilon`), 
gradients flow through the basis evaluation itself.

----

Complexity Analysis
-------------------

**Time Complexity per Layer:**

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Layer Type
     - Forward Pass
     - Backward Pass
   * - Polynomial (full)
     - :math:`O(B \cdot n \cdot K \cdot m)`
     - :math:`O(B \cdot n \cdot K \cdot m)`
   * - Polynomial (Tucker)
     - :math:`O(B \cdot n \cdot K \cdot R)`
     - :math:`O(B \cdot n \cdot K \cdot R)`
   * - RBF
     - :math:`O(B \cdot n \cdot K \cdot m)`
     - :math:`O(B \cdot n \cdot K \cdot m)`
   * - Wavelet
     - :math:`O(B \cdot n \cdot m)`
     - :math:`O(B \cdot n \cdot m)`

where :math:`R = \max(R_1, R_2, R_3)` for Tucker decomposition.

**Space Complexity (Parameters):**

- Polynomial (full): :math:`n \times (K+1) \times m + m`
- Polynomial (Tucker): :math:`R_1 R_2 R_3 + n R_1 + (K+1) R_2 + m R_3 + m`
- RBF: :math:`n \times K \times m + m`
- Wavelet: :math:`n \times m + m` (plus scales/translations)

----

Summary
-------

ARNOLD's KAN layers implement the Kolmogorov-Arnold representation theorem using:

1. **Alternant tensors** as the core mathematical abstraction
2. **Polynomial expansions** with learnable coefficients
3. **RBF kernels** on learnable grids for local approximation
4. **Wavelet transforms** with learnable scales and translations
5. **Tucker decomposition** for parameter efficiency
6. **Numerical stability** via Clenshaw, clipping, and softplus

All operations are:

- Fully differentiable for gradient-based learning
- XLA-compatible for GPU/TPU acceleration
- Keras-native for seamless integration
