.. ARNOLD documentation master file, created by
   sphinx-quickstart on Fri Aug  2 09:46:51 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to :math:`\mathtt{ARNOLD}`!
++++++++++++++++++++++++++++++++++++++++++++

:math:`\mathtt{ARNOLD}` offers `tf.keras` implementations of `Kolmogorov-Arnold Networks (KAN) layers <https://arxiv.org/pdf/2404.19756?trk=public_post_main-feed-card-text>`_ using various basis functions.

Simply use :math:`\texttt{ARNOLD}`'s KAN layers as a drop-in-replacement for ``tf.keras.layers.Dense`` and mix with any standard layers.

.. code-block:: PythonLexer
    :linenos:
    :emphasize-lines: 11,13,15

    import tensorflow as tf
    from arnold.layers.core.polynomial.orthogonal import Chebyshev1st, Legendre
    from arnold.layers.core.wavelet import Bump

    tfk = tf.keras
    tfkl = tfk.layers

    fancy_kan =tfk.Sequential([
            tfkl.Reshape(target_shape=(2, )),
            tfkl.Rescaling(scale=1./127.5, offset=-1),
            Chebyshev1st(input_dim=2, output_dim=8, degree=2),
            tfkl.LayerNormalization(),
            Legendre(input_dim=8, output_dim=6, degree=3),
            tfkl.LayerNormalization(),
            Bump(input_dim=6, output_dim=1),
            tfkl.Activation(tfk.activations.sigmoid)
        ],
        name="fancy_kan" 
    )

Kolmogorov-Arnold Networks (KANs) are a type of neural network inspired by the Kolmogorov-Arnold representation theorem related to the representation of continuous functions. 
In some sense, this theorem states that any multivariate continuous function can be represented as a superposition of continuous univariate functions and addition. 

:math:`\texttt{ARNOLD}` is available via pip 
 
.. code-block:: bash

    python3 pip -m install arnold

In case you are interested, feel free to read some :ref:`formal background <background>` on Kolmogorov-Arnold Networks. 


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Background

   introduction/intro.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Available KAN layers

   layers/polynomial_kan_layers.rst
   layers/radial_basis_kan_layers.rst
   layers/wavelet_kan_layers.rst


.. * :ref:`modindex`
.. * :ref:`search`
.. * :ref:`genindex`