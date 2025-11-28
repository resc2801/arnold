## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
"""Correctness/shape/gradient tests for RBF and wavelet layers."""

import tensorflow as tf

from arnold.layers.core.rbf import GaussianRBF
from arnold.layers.core.wavelets import Bump


def test_gaussian_rbf_gradient_and_shape():
    x = tf.random.uniform((5, 4), minval=-1.0, maxval=1.0, seed=11)
    layer = GaussianRBF(units=3)
    with tf.GradientTape() as tape:
        y = layer(x)
        loss = tf.reduce_sum(y)
    grads = tape.gradient(loss, layer.trainable_variables)
    assert y.shape == (5, 3)
    assert all(g is not None for g in grads)
    for g in grads:
        assert tf.reduce_all(tf.math.is_finite(g))


def test_gaussian_rbf_serialization_roundtrip():
    layer = GaussianRBF(units=2, epsilon_init=0.5)
    _ = layer(tf.zeros((1, 3)))  # build
    cfg = layer.get_config()
    restored = GaussianRBF.from_config(cfg)
    y = restored(tf.ones((2, 3)))
    assert y.shape == (2, 2)


def test_wavelet_bump_serialization_roundtrip():
    layer = Bump(units=2)
    _ = layer(tf.zeros((2, 3)))  # build
    cfg = layer.get_config()
    restored = Bump.from_config(cfg)
    y = restored(tf.ones((2, 3)))
    assert y.shape == (2, 2)


def test_wavelet_bump_gradient():
    x = tf.random.uniform((4, 3), minval=-2.0, maxval=2.0, seed=23)
    layer = Bump(units=2, input_clip=(-2, 2))
    with tf.GradientTape() as tape:
        y = layer(x)
        loss = tf.reduce_mean(y)
    grads = tape.gradient(loss, layer.trainable_variables)
    assert all(g is not None for g in grads)
    for g in grads:
        assert tf.reduce_all(tf.math.is_finite(g))
