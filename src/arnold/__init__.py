## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.

"""
ARNOLD: Kolmogorov-Arnold Networks for Keras
=============================================

Production-ready KAN layers with polynomial, RBF, and wavelet bases.

Example usage::

    import arnold
    from arnold.layers import Legendre, Chebyshev1st, GaussianRBF

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10,)),
        Legendre(input_dim=10, output_dim=32, degree=4),
        Chebyshev1st(input_dim=32, output_dim=16, degree=3),
        tf.keras.layers.Dense(1)
    ])

Hardware-adaptive dtype selection::

    from arnold import detect_hardware, get_recommended_dtype
    
    hw = detect_hardware()  # "cpu", "gpu", "tpu", or "mps"
    dtype = get_recommended_dtype(degree=15, hardware=hw)

For more information, see:
- GitHub: https://github.com/resc2801/arnold
- Docs: https://arnold-kan.readthedocs.io
"""

__version__ = "0.1.0"
__author__ = "René Schubotz"
__email__ = "r.schubotz@googlemail.com"

from arnold import layers
from arnold.layers.core.kan_base import detect_hardware, get_recommended_dtype


__all__ = [
    "layers",
    "detect_hardware",
    "get_recommended_dtype",
    "__version__",
]
