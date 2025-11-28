.. ARNOLD documentation master file
   Copyright (c) 2025 René Schubotz. All rights reserved.

.. raw:: html

   <style>
   .hero-title { font-size: 2.5em; font-weight: bold; margin-bottom: 0.5em; }
   .hero-subtitle { font-size: 1.2em; color: #666; margin-bottom: 1em; }
   </style>

==========================================
The Book of KANs
==========================================

.. raw:: html

   <p class="hero-subtitle">A Comprehensive Guide to Kolmogorov-Arnold Networks</p>

**ARNOLD** is the definitive, production-ready implementation of Kolmogorov-Arnold Network (KAN) 
layers for TensorFlow/Keras. This documentation serves as both a practical guide for practitioners 
and a mathematical reference for researchers.

.. note::

   **Quick Install**
   
   .. code-block:: bash

      pip install arnold-kan

----

What are Kolmogorov-Arnold Networks?
------------------------------------

Kolmogorov-Arnold Networks are a class of neural networks inspired by the 
**Kolmogorov-Arnold Representation Theorem**, which states that any continuous 
multivariate function can be represented as a composition of univariate functions:

.. math::

   f(\mathbf{x}) = \sum_{q=1}^{2n+1} \Phi_q \left( \sum_{p=1}^{n} \psi_{q,p}(x_p) \right)

Unlike traditional MLPs that use fixed activation functions with learnable weights,
KANs use **learnable activation functions** (represented as polynomial, RBF, or wavelet 
expansions) with fixed summation structure.

.. list-table:: KANs vs. Traditional MLPs
   :widths: 40 30 30
   :header-rows: 1

   * - Property
     - MLP
     - KAN
   * - Activation functions
     - Fixed (ReLU, tanh, etc.)
     - Learnable (polynomials, RBFs, wavelets)
   * - Weights
     - Learnable matrices
     - Fixed summation
   * - Interpretability
     - Black-box
     - Mathematically grounded
   * - Expressivity
     - Universal approximator
     - Universal approximator (by theorem)

Getting Started
---------------

**Drop-in replacement for Dense layers:**

.. code-block:: python
   :linenos:
   :emphasize-lines: 11,13,15

   import tensorflow as tf
   from arnold.layers import Chebyshev1st, Legendre, Bump

   tfk = tf.keras
   tfkl = tfk.layers

   model = tfk.Sequential([
       tfkl.Input(shape=(784,)),
       tfkl.Reshape(target_shape=(784,)),
       tfkl.Rescaling(scale=1./127.5, offset=-1),  # Normalize to [-1, 1]
       Chebyshev1st(degree=5, units=128),          # KAN layer
       tfkl.LayerNormalization(),
       Legendre(degree=3, units=64),               # KAN layer
       tfkl.LayerNormalization(),
       Bump(units=10),                             # Wavelet KAN layer
       tfkl.Activation('softmax')
   ], name="kan_classifier")

   model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

**Key design principles:**

- **Keras-native**: Use ``model.fit()``, ``model.save()``, callbacks, etc.
- **Hardware-adaptive**: Automatic dtype selection for CPU/GPU/TPU/MPS
- **Numerically stable**: Clenshaw recurrence, input clipping, softplus constraints
- **Extensible**: Subclass base layers to add custom basis functions

----

Documentation Structure
-----------------------

This documentation is organized as a comprehensive reference:

**Part I: Foundations**
   Mathematical background, the Kolmogorov-Arnold theorem, and how KANs
   translate representation theory into practical neural network layers.

**Part II: Tutorials**
   Hands-on tutorials with complete, runnable examples for common use cases.

**Part III: Basis Function Catalog**
   Complete reference for all 40+ polynomial, RBF, and wavelet basis functions,
   with mathematical definitions, properties, and usage guidance.

**Part IV: Practical Guides**
   Performance optimization, hardware deployment, and troubleshooting.

**Part V: API Reference**
   Complete API documentation generated from docstrings.

----

.. toctree::
   :maxdepth: 2
   :caption: Part I: Foundations

   theory/index

.. toctree::
   :maxdepth: 2
   :caption: Part II: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Part III: Basis Functions

   layers/index
   basis_selection_guide

.. toctree::
   :maxdepth: 2
   :caption: Part IV: Practical Guides

   guides/index

.. toctree::
   :maxdepth: 1
   :caption: Part V: Reference

   generated/arnold
   simplifications

----

Quick Links
-----------

- :doc:`guides/quickstart` — Get started in 5 minutes
- :doc:`basis_selection_guide` — Choose the right basis function
- :doc:`troubleshooting` — Common issues and solutions
- `GitHub Repository <https://github.com/resc2801/arnold>`_
- `PyPI Package <https://pypi.org/project/arnold-kan/>`_

----

Citation
--------

If you use ARNOLD in your research, please cite:

.. code-block:: bibtex

   @software{arnold2025,
     author = {Schubotz, René},
     title = {ARNOLD: Kolmogorov-Arnold Networks for Keras},
     year = {2025},
     url = {https://github.com/resc2801/arnold},
   }

For the original KAN paper, cite:

.. code-block:: bibtex

   @article{liu2024kan,
     title={KAN: Kolmogorov-Arnold Networks},
     author={Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and others},
     journal={arXiv preprint arXiv:2404.19756},
     year={2024}
   }

----

License
-------

ARNOLD is released under the terms specified in the LICENSE file.
Copyright © 2025 René Schubotz. All rights reserved.

.. Indices and tables
.. ==================
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
