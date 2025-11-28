.. Copyright (c) 2025 René Schubotz. All rights reserved.

.. _tutorials-index:

====================================================
Part II: Tutorials & Examples
====================================================

This part provides hands-on tutorials that progressively introduce ARNOLD's
capabilities through complete, runnable examples.

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   getting_started
   regression_example
   classification_example
   time_series_example
   physics_informed_example

----

Learning Path
-------------

**Week 1: Fundamentals**

- :doc:`getting_started` — Installation, first model, core concepts
- :doc:`regression_example` — Function approximation with polynomial KANs

**Week 2: Applications**

- :doc:`classification_example` — Image classification with KANs
- :doc:`time_series_example` — Temporal data with wavelet KANs

**Week 3: Advanced Topics**

- :doc:`physics_informed_example` — Physics-informed neural networks with KANs

----

Tutorial Format
---------------

Each tutorial follows a consistent structure:

1. **Objective** — What you'll learn
2. **Prerequisites** — Required knowledge and setup
3. **Theory** — Brief mathematical background
4. **Implementation** — Step-by-step code walkthrough
5. **Exercises** — Practice problems with solutions
6. **Further Reading** — Links to relevant documentation

----

Running the Tutorials
---------------------

All tutorials are available as Jupyter notebooks in the ``examples/`` directory:

.. code-block:: bash

   # Navigate to examples
   cd examples/
   
   # Start Jupyter
   jupyter notebook

Or run directly in Python:

.. code-block:: bash

   python examples/regression_example.py

----

Quick Links by Topic
--------------------

**Getting Started:**

- Installation and setup → :doc:`getting_started`
- Basic model architecture → :doc:`getting_started`

**Regression Tasks:**

- 1D function fitting → :doc:`regression_example`
- Multivariate regression → :doc:`regression_example`

**Classification Tasks:**

- MNIST with KANs → :doc:`classification_example`
- Binary classification → :doc:`classification_example`

**Time Series:**

- Sequence prediction → :doc:`time_series_example`
- Multi-scale analysis → :doc:`time_series_example`

**Scientific Computing:**

- Physics-informed KANs → :doc:`physics_informed_example`
- PDE solutions → :doc:`physics_informed_example`
