Wavelet KAN layers
****************************************

Wavelets are short, wave-like functions that can be adjusted in scale and position. 
The wavelet transform is a process that represents any signal in terms of these scaled and shifted wavelets. 
Wavelets in Kolmogorov-Arnold Networks (KANs) offer a sophisticated approach to function approximation by utilizing wavelet basis functions, 
which are well-suited for capturing both local and global features of complex signals or data distributions.

.. autoclass:: arnold.layers.core.wavelet.WaveletBase
    :members: __init__

.. autoclass:: arnold.layers.core.wavelet.Bump
    :members: __init__

.. autoclass:: arnold.layers.core.wavelet.DerivativeOfGaussian
    :members: __init__

.. autoclass:: arnold.layers.core.wavelet.Meyer
    :members: __init__

.. autoclass:: arnold.layers.core.wavelet.Morelet
    :members: __init__

.. autoclass:: arnold.layers.core.wavelet.Poisson
    :members: __init__

.. autoclass:: arnold.layers.core.wavelet.Ricker
    :members: __init__

.. autoclass:: arnold.layers.core.wavelet.Shannon
    :members: __init__
