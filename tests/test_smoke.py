## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.

import tensorflow as tf

from arnold.layers.core.polynomial.orthogonal import Legendre
from arnold.layers.core.rbf import GaussianRBF
from arnold.layers.core.wavelets import Bump


def test_legendre_build_call_and_serialize():
    layer = Legendre(degree=2, units=3, use_bias=False)
    x = tf.ones((2, 5))
    y = layer(x)
    assert y.shape == (2, 3)

    config = layer.get_config()
    clone = Legendre.from_config(config)
    clone.build(x.shape)
    y_clone = clone(x)
    assert y_clone.shape == (2, 3)


def test_gaussian_rbf_build_and_shape():
    layer = GaussianRBF(units=4, num_grids=5, use_bias=False)
    x = tf.ones((3, 2))
    y = layer(x)
    assert y.shape == (3, 4)


def test_bump_wavelet_shape():
    layer = Bump(units=2, use_bias=False)
    x = tf.zeros((1, 3))
    y = layer(x)
    assert y.shape == (1, 2)
