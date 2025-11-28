.. Copyright (c) 2025 René Schubotz. All rights reserved.

.. _kolmogorov_arnold_theorem:

================================================
The Kolmogorov-Arnold Representation Theorem
================================================

The Kolmogorov-Arnold Representation Theorem is one of the most remarkable results in 
mathematical analysis, providing a definitive answer to Hilbert's 13th problem. It establishes 
that **any continuous function of multiple variables can be represented using only continuous 
functions of a single variable and the operation of addition**.

This chapter presents the theorem, its historical context, key proofs, and the insights that 
led to Kolmogorov-Arnold Networks.

.. contents:: Chapter Contents
   :local:
   :depth: 2

----

Historical Context: Hilbert's 13th Problem
------------------------------------------

In 1900, David Hilbert presented 23 influential problems at the International Congress of 
Mathematicians. The 13th problem asked:

   *Is it possible to express the roots of a general seventh-degree polynomial equation 
   as a superposition of continuous functions of two variables?*

More generally, Hilbert conjectured that there exist continuous functions of three variables 
that cannot be expressed as finite compositions of continuous functions of two variables.

This question remained open for over half a century until Andrey Kolmogorov [Kolmogorov1957]_ 
and his student Vladimir Arnold [Arnold1957]_ provided a surprising negative answer: **not only 
can all continuous functions of :math:`n` variables be represented using functions of two 
variables, they can be represented using only functions of one variable and addition**.

----

The Theorem
-----------

.. admonition:: Theorem - Kolmogorov-Arnold Representation Theorem
   

   For any continuous function :math:`f: [0,1]^n \to \mathbb{R}`, there exist:
   
   - Continuous functions :math:`\Phi_q: \mathbb{R} \to \mathbb{R}` for :math:`q = 0, 1, \ldots, 2n`
   - Continuous functions :math:`\psi_{q,p}: [0,1] \to \mathbb{R}` for :math:`q = 0, 1, \ldots, 2n` and :math:`p = 1, 2, \ldots, n`
   
   such that:

   .. math::

      f(x_1, x_2, \ldots, x_n) = \sum_{q=0}^{2n} \Phi_q \left( \sum_{p=1}^{n} \psi_{q,p}(x_p) \right)

**Key Observations:**

1. The **inner functions** :math:`\psi_{q,p}` depend only on a single variable :math:`x_p`
2. The **outer functions** :math:`\Phi_q` also depend on a single variable (the sum)
3. The only multi-variable operation is **addition**
4. The number of outer functions :math:`2n + 1` is independent of the complexity of :math:`f`

----

Understanding the Representation
--------------------------------

The theorem can be understood as a two-layer computation:

**Layer 1: Inner Functions (Encoding)**

Each input variable :math:`x_p` is transformed by :math:`2n + 1` different inner functions:

.. math::

   z_q = \sum_{p=1}^{n} \psi_{q,p}(x_p), \quad q = 0, 1, \ldots, 2n

The inner functions :math:`\psi_{q,p}` "encode" the input into :math:`2n + 1` intermediate values.

**Layer 2: Outer Functions (Decoding)**

Each intermediate value :math:`z_q` is transformed by an outer function, and the results are summed:

.. math::

   f(\mathbf{x}) = \sum_{q=0}^{2n} \Phi_q(z_q)

The outer functions :math:`\Phi_q` "decode" the intermediate values into the final output.

**Visualization:**

.. code-block:: text

                        ┌──────┐
                        │ ψ₀,₁ │───┐
              ┌─────────┤      │   │
              │         └──────┘   │    ┌──────┐
              │         ┌──────┐   ├────│  Φ₀  │───┐
              │    ┌────│ ψ₀,₂ │───┘    └──────┘   │
              │    │    └──────┘                    │
     x₁ ──────┼────┼─────────────────────────────────────
              │    │    ┌──────┐                    │
              │    │    │ ψ₁,₁ │───┐                │
              │    │    └──────┘   │    ┌──────┐   │
              │    └────┬──────────┼────│  Φ₁  │───┼───▶ f(x)
              │         │ ψ₁,₂ │───┘    └──────┘   │
     x₂ ──────┼─────────┼──────┘                    │
              │         │         ⋮                  │
              │         │    ┌──────┐               │
              │         │    │ψ₂ₙ,₁│───┐           │
              │         │    └──────┘   │   ┌──────┐│
              └─────────┴────┬──────────┼───│ Φ₂ₙ │┘
                             │ψ₂ₙ,₂│───┘   └──────┘
                             └──────┘

----

Properties of the Inner Functions
---------------------------------

The inner functions :math:`\psi_{q,p}` have remarkable properties:

**Independence from :math:`f`:**
   Kolmogorov showed that the inner functions can be chosen **independently of the target function** 
   :math:`f`. Only the outer functions :math:`\Phi_q` need to be adapted to represent a specific :math:`f`.

**Regularity:**
   The inner functions can be chosen to be Lipschitz continuous with a specific Lipschitz constant.
   Specifically, :math:`\psi_{q,p}` can be taken in the Hölder class with exponent 
   :math:`\alpha = \log 2 / \log(2n + 2)`.

**Constructibility:**
   Explicit constructions of the inner functions exist, though they are typically fractal-like 
   (nowhere differentiable). This has implications for numerical implementation.

----

The Outer Functions
-------------------

While the inner functions can be fixed, the outer functions :math:`\Phi_q` must be carefully 
constructed to represent a given :math:`f`. Key properties:

**Continuity:**
   The outer functions are continuous, but may be highly irregular (nowhere differentiable).

**Function-Specific:**
   Different target functions :math:`f` require different outer functions :math:`\Phi_q`.

**Existence vs. Constructibility:**
   The theorem guarantees existence but doesn't provide an algorithm for constructing :math:`\Phi_q`.

----

Simplifications
---------------

Several mathematicians have simplified the Kolmogorov-Arnold representation:

Lorentz's Simplification
~~~~~~~~~~~~~~~~~~~~~~~~

George Lorentz [Lorentz1962]_ showed that a **single outer function** :math:`\Phi` suffices:

.. admonition:: Theorem - Lorentz's Simplification
   

   For any continuous :math:`f: [0,1]^n \to \mathbb{R}`, there exist a continuous 
   :math:`\Phi: \mathbb{R} \to \mathbb{R}` and continuous :math:`\psi_{q,p}` such that:

   .. math::

      f(\mathbf{x}) = \sum_{q=0}^{2n} \Phi \left( \sum_{p=1}^{n} \psi_{q,p}(x_p) \right)

Sprecher's Simplification
~~~~~~~~~~~~~~~~~~~~~~~~~

David Sprecher [Sprecher1965]_ showed that a **single inner function** :math:`\psi` (appropriately shifted) suffices:

.. admonition:: Theorem - Sprecher's Variant
   

   For any continuous :math:`f: [0,1]^n \to \mathbb{R}`, there exist constants 
   :math:`\eta, \lambda_1, \ldots, \lambda_n`, a continuous :math:`\Phi`, and a single 
   continuous increasing :math:`\psi: [0,1] \to [0,1]` such that:

   .. math::

      f(\mathbf{x}) = \sum_{q=0}^{2n} \Phi \left( \sum_{p=1}^{n} \lambda_p \psi(x_p + \eta q) + q \right)

This variant is computationally significant: only one inner function needs to be learned.

See :doc:`../simplifications` for detailed proofs and discussions.

----

Implications for Neural Networks
--------------------------------

The Kolmogorov-Arnold theorem has profound implications for neural network design:

**Universal Approximation:**
   The theorem proves that a specific two-layer architecture with :math:`2n + 1` hidden units 
   can represent **any** continuous function. This is a stronger statement than the universal 
   approximation theorems for MLPs.

**Fixed Width:**
   Unlike MLPs where the width must grow with problem complexity, the KA representation has 
   fixed width :math:`2n + 1`.

**Learnable Activations:**
   The theorem suggests learning the **activation functions** rather than the weights. This is 
   the core insight behind KAN layers.

**Challenges:**
   The original inner functions are fractal-like and non-smooth. Practical implementations 
   must approximate them with smooth basis expansions (polynomials, RBFs, wavelets).

----

From Theorem to Practice: KAN Layers
------------------------------------

ARNOLD implements the KAN paradigm, as introduced by Liu et al. [Liu2024]_, by:

1. **Replacing** the unknown :math:`\psi_{q,p}` with **learnable polynomial/RBF/wavelet expansions**
2. **Learning** the expansion coefficients during training
3. **Using** smooth basis functions that are numerically stable and GPU-friendly

The resulting architecture:

.. math::

   \text{KAN}(\mathbf{x}) = \sum_{j=1}^{d_{\text{in}}} \sum_{k=0}^{K} c_{j,k} \, B_k(x_j)

where :math:`B_k` are basis functions (Chebyshev, Legendre, Gaussian RBF, wavelets, etc.) 
and :math:`c_{j,k}` are learnable coefficients.

See :doc:`kan_layers` for the detailed mathematical formulation of KAN layers.

----

Key References
--------------

.. [Kolmogorov1957] Kolmogorov, A. N. (1957). 
   "On the representation of continuous functions of several variables by superposition 
   of continuous functions of one variable and addition." 
   *Doklady Akademii Nauk SSSR*, 114:953–956.

.. [Arnold1957] Arnold, V. I. (1957). 
   "On functions of three variables." 
   *Doklady Akademii Nauk SSSR*, 114:679–681.

.. [Sprecher1965] Sprecher, D. A. (1965). 
   "On the structure of continuous functions of several variables." 
   *Transactions of the American Mathematical Society*, 115:340–355.

.. [Lorentz1962] Lorentz, G. G. (1962). 
   "Metric entropy, widths, and superpositions of functions." 
   *American Mathematical Monthly*, 69:469–485.

.. [Liu2024] Liu, Z., Wang, Y., Vaidya, S., et al. (2024). 
   "KAN: Kolmogorov-Arnold Networks." 
   *arXiv preprint arXiv:2404.19756*.
