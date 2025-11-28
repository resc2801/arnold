.. Copyright (c) 2025 René Schubotz. All rights reserved.

.. _guides-index:

====================================================
Part IV: Practical Guides
====================================================

This part provides practical guidance for using ARNOLD effectively in
production environments.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   quickstart
   advanced_usage

.. toctree::
   :maxdepth: 2
   :caption: Additional Resources

   ../performance_guide
   ../troubleshooting

----

Guide Overview
--------------

:doc:`quickstart`
   Get up and running with ARNOLD in 5 minutes. Installation, first model,
   essential concepts.

:doc:`advanced_usage`
   Advanced features for power users: Tucker decomposition, custom basis 
   functions, mixed-precision training, and production deployment.

:doc:`../basis_selection_guide`
   Comprehensive guide to choosing the right basis function for your problem.
   Covers polynomials, RBFs, and wavelets with decision flowcharts.

:doc:`../performance_guide`
   Hardware-specific optimization: GPU, TPU, Apple Silicon. Mixed precision,
   XLA compilation, memory management.

:doc:`../troubleshooting`
   Common issues, error messages, and FAQ. Solutions for numerical instability,
   performance problems, and compatibility issues.

----

Quick Navigation
----------------

**I want to...**

- **Get started quickly** → :doc:`quickstart`
- **Understand which polynomial to use** → :doc:`../basis_selection_guide`
- **Reduce memory usage** → :doc:`advanced_usage` (Tucker decomposition)
- **Speed up training on GPU** → :doc:`../performance_guide`
- **Fix NaN/Inf errors** → :doc:`../troubleshooting`
- **Use Apple Silicon** → :doc:`../performance_guide` (MPS section)
- **Deploy to TPU** → :doc:`../performance_guide` (TPU section)
- **Create custom layers** → :doc:`advanced_usage`

----

Recommended Reading Order
-------------------------

**For New Users:**

1. :doc:`quickstart` — Essential concepts and first model
2. :doc:`../basis_selection_guide` — Choose the right basis
3. :doc:`../troubleshooting` — Know what can go wrong

**For Production Deployment:**

1. :doc:`../performance_guide` — Hardware optimization
2. :doc:`advanced_usage` — Tucker, mixed precision, serialization
3. :doc:`../troubleshooting` — Error handling

**For Researchers:**

1. Theory chapters (Part I)
2. :doc:`../basis_selection_guide` — Mathematical background
3. :doc:`advanced_usage` — Custom basis implementation
