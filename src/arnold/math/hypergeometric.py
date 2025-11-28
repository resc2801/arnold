## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.

import tensorflow as tf


@tf.function
def pochhammer(k: int, x: tf.Tensor):
    """
    The Pochhammer symbol (rising factorial) is defined as

    pochhammer(k, x) := x * (x + 1) * ... * (x + k -1) = np.prod([(x+i) for i in range(0,k)])

    See also: https://dlmf.nist.gov/5.2#iii

    :param k: The factorial power
    :type k: non-negative integer

    :param x: value to apply pochhammer to
    :type x: tf.Tensor

    :returns: pochhammer(k, x)
    :rtype: tf.Tensor
    """
    assert k >= 0, "Parameter k must be non-negative!"
    if k == 0:
        return tf.ones_like(x)

    if tf.math.reduce_all(tf.math.greater(x, 0.0)):
        # all values in x greater than 0, so it is safe to use lgamma
        # (x)_k = Γ(x+k) / Γ(x)
        return tf.math.exp(tf.math.lgamma(x + k) - tf.math.lgamma(x))
    else:
        # otherwise we need to iterate
        return tf.math.reduce_prod(tf.map_fn(lambda m: x + m, tf.range(0, k, delta=1, dtype=x.dtype)), axis=0)


@tf.function
def generalized_hypergeometric(a_s: list[tf.Tensor], b_s: list[tf.Tensor], z: tf.Tensor, num_terms: int):
    """
    Generalized hypergeometric function.

    # :math:`{}_{p}F_{q}(a_{1},\\ldots ,a_{p};b_{1},\\ldots ,b_{q};z)=\\sum _{n=0}^{\\infty }{\frac {(a_{1})_{n}\\cdots (a_{p})_{n}}{(b_{1})_{n}\\cdots (b_{q})_{n}}}\\,{\frac {z^{n}}{n!}}`

    :param a_s: List of tf.Tensors with same shape
    :type a_s: List[tf.Tensor]

    :param b_s: List of tf.Tensors with same shape
    :type b_s: List[tf.Tensor]

    :param z_s: order-1 tensor
    :type z: tf.Tensor

    :params num_terms: number of summation terms to compute
    :type num_terms: non-negative integer

    :returns: :math:`{}_{p}F_{q}(a_{1},\\ldots ,a_{p};b_{1},\\ldots ,b_{q};z)`
    :rtype: tf.Tensor
    """

    terms = []
    for n in range(num_terms):
        numerator = tf.math.reduce_prod([pochhammer(n, t) for t in a_s], axis=0)
        denominator = tf.math.reduce_prod([pochhammer(n, t) for t in b_s], axis=0)
        coeff = tf.math.divide(numerator, denominator)
        term = tf.math.multiply(coeff, tf.math.pow(z, n) / tf.math.exp(tf.math.lgamma(n + 1.0)))
        terms.append(term)

    return tf.math.reduce_sum(tf.stack(terms), axis=0)
