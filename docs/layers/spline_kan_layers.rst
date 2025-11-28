Spline KAN Layers
=================

This module provides Kolmogorov-Arnold Network layers using spline basis functions.
Splines offer smooth, local approximation with guaranteed continuity — ideal for
CAD, animation, signal processing, and smooth function approximation.

Quick Comparison
----------------

.. list-table::
   :widths: 20 15 15 50
   :header-rows: 1

   * - Layer
     - Continuity
     - Interpolating
     - Best For
   * - ``BSpline``
     - :math:`C^{p-1}`
     - No
     - Signal processing, CAD, general approximation
   * - ``CatmullRom``
     - :math:`C^1`
     - Yes
     - Animation, path interpolation, graphics
   * - ``Cardinal``
     - :math:`C^1`
     - Yes
     - Adjustable smoothness, graphics with tension control

Mathematical Background
-----------------------

**B-Splines**

B-splines are piecewise polynomials defined by the Cox-de Boor recursion:

.. math::

   B_{i,0}(x) = \begin{cases} 1 & t_i \leq x < t_{i+1} \\ 0 & \text{otherwise} \end{cases}

.. math::

   B_{i,p}(x) = \frac{x - t_i}{t_{i+p} - t_i} B_{i,p-1}(x) + \frac{t_{i+p+1} - x}{t_{i+p+1} - t_{i+1}} B_{i+1,p-1}(x)

Key properties:

- **Partition of unity**: :math:`\sum_i B_{i,p}(x) = 1` on the interior
- **Non-negativity**: :math:`B_{i,p}(x) \geq 0`
- **Local support**: Each basis spans at most :math:`p+1` knot intervals
- **Smoothness**: :math:`C^{p-1}` continuity at knots

**Catmull-Rom Splines**

Catmull-Rom splines use cubic Hermite interpolation with automatic tangent computation:

.. math::

   \mathbf{M}_{CR} = \frac{1}{2} \begin{pmatrix}
       -1 & 3 & -3 & 1 \\
       2 & -5 & 4 & -1 \\
       -1 & 0 & 1 & 0 \\
       0 & 2 & 0 & 0
   \end{pmatrix}

**Cardinal Splines**

Cardinal splines generalize Catmull-Rom with a tension parameter :math:`\tau`:

.. math::

   \mathbf{M}_C(\tau) = \begin{pmatrix}
       -\tau & 2-\tau & \tau-2 & \tau \\
       2\tau & \tau-3 & 3-2\tau & -\tau \\
       -\tau & 0 & \tau & 0 \\
       0 & 1 & 0 & 0
   \end{pmatrix}

When :math:`\tau = 0.5`, this reduces to Catmull-Rom.

API Reference
-------------

SplineBase
~~~~~~~~~~

.. autoclass:: arnold.layers.core.splines.SplineBase
   :members:
   :show-inheritance:

BSpline
~~~~~~~

.. autoclass:: arnold.layers.core.splines.BSpline
   :members:
   :show-inheritance:

CatmullRom
~~~~~~~~~~

.. autoclass:: arnold.layers.core.splines.CatmullRom
   :members:
   :show-inheritance:

Cardinal
~~~~~~~~

.. autoclass:: arnold.layers.core.splines.Cardinal
   :members:
   :show-inheritance:

Usage Examples
--------------

**Basic B-Spline Layer**

.. code-block:: python

   from arnold.layers import BSpline

   # Cubic B-spline with 16 basis functions
   layer = BSpline(units=64, order=4, num_knots=12)

   # In a Keras model
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(8,)),
       BSpline(units=32, order=4, num_knots=16),
       tf.keras.layers.Dense(10, activation='softmax'),
   ])

**Animation Path with Catmull-Rom**

.. code-block:: python

   from arnold.layers import CatmullRom

   # Catmull-Rom for smooth interpolation
   layer = CatmullRom(units=3, num_knots=10)  # 3D position output

**Adjustable Tension with Cardinal**

.. code-block:: python

   from arnold.layers import Cardinal

   # Tighter curves with higher tension
   layer = Cardinal(
       units=32,
       num_knots=12,
       tension=0.8,  # 0.5 = Catmull-Rom, 1.0 = tightest
       tension_trainable=True,  # Learn optimal tension
   )

**Trainable Knots**

.. code-block:: python

   # Let the network learn optimal knot positions
   layer = BSpline(
       units=64,
       order=4,
       num_knots=16,
       trainable_knots=True,
   )

Choosing a Spline Type
----------------------

**Use B-Spline when:**

- You need high smoothness (:math:`C^2` or higher)
- Partition of unity is important (weights sum to 1)
- General function approximation is the goal
- Working with CAD/signal processing applications

**Use Catmull-Rom when:**

- The spline must pass through control points (interpolating)
- Animation or path interpolation is the use case
- :math:`C^1` continuity is sufficient
- Graphics/game development applications

**Use Cardinal when:**

- You want Catmull-Rom behavior with adjustable "tightness"
- The optimal tension is unknown and should be learned
- Different tension values may work better for different data

Performance Notes
-----------------

All spline layers are XLA-compatible and support:

- Mixed-precision training (``float16``, ``bfloat16``)
- GPU/TPU acceleration
- Gradient-based optimization of all parameters

For best performance:

- Use ``num_knots`` proportional to input complexity (8-32 typical)
- Higher ``order`` (4-5) for smoother approximations
- Consider ``trainable_knots=True`` for adaptive representations
