"""
An implementation of the Transformer decoder architecture.

https://arxiv.org/abs/1706.03762
"""

from math import sqrt

import numpy as np
import tensorflow as tf


def transformer_layer(inputs, num_heads=8, hidden=2048, activation=tf.nn.relu, scope='transformer'):
    """
    Apply a decoder layer of the Transformer.

    Args:
      inputs: a [batch x timesteps x N] Tensor.
      num_heads: the number of attention heads.
      hidden: size of the FC hidden layer.
      scope: the variable scope name.
    """
    inner_dim = inputs.get_shape()[-1].value
    assert not inner_dim % num_heads, 'number of heads must divide d_model'
    with tf.variable_scope(None, default_name=scope):
        # pylint: disable=E1101
        outs = masked_attention(inputs, num_heads=num_heads)
        outs = tf.contrib.layers.layer_norm(inputs + outs, center=True, scale=True,
                                            begin_norm_axis=-1)

        pre_fc = outs
        outs = tf.layers.dense(outs, hidden, activation=activation)
        outs = tf.layers.dense(outs, inner_dim, activation=activation)
        outs = tf.contrib.layers.layer_norm(pre_fc + outs, center=True, scale=True,
                                            begin_norm_axis=-1)

        return outs


def positional_encoding(timesteps, inner_dim, base=10000, dtype=tf.float32):
    """
    Compute the sinusoidal positional encoding for the
    sequence dimensions.

    Args:
      timesteps: the number of timesteps. Either a Tensor
        or an integer.
      inner_dim: the size of the inner vector. An integer.
      base: a parameter controlling the max wavelength.
      dtype: the dtype of the resulting Tensor.

    Returns:
      A [timesteps x inner_dim] Tensor which is meant to
        be added to the inputs.
    """
    positions = tf.cast(tf.expand_dims(tf.range(timesteps), axis=-1), dtype)
    dimensions = tf.cast(tf.expand_dims((tf.range(inner_dim) // 2) * 2, axis=0), dtype)
    arguments = positions / tf.pow(tf.constant(base, dtype=dtype), dimensions / inner_dim)
    sin_mask = tf.tile(tf.expand_dims(tf.range(inner_dim), axis=0), [timesteps, 1])
    return tf.where(tf.equal((sin_mask % 2), 0), tf.cos(arguments), tf.sin(arguments))


def masked_attention(inputs, num_heads=8, scope='attention'):
    """
    Perform masked multi-head attention over a sequence.

    Args:
      inputs: a [batch x timesteps x N] Tensor.
      num_heads: the number of attention heads.
      scope: the variable scope name.
    """
    inner_dim = inputs.get_shape()[-1].value
    assert inner_dim is not None
    _, timesteps, _ = optimized_shape(inputs)
    with tf.variable_scope(None, default_name=scope):
        projected = tf.layers.dense(inputs, inner_dim * 3, name='key_query_value')
        kqv = tf.split(projected, 3, axis=-1)
        keys, queries, values = [split_heads(x, num_heads) for x in kqv]
        logits = tf.matmul(queries, tf.transpose(keys, [0, 1, 3, 2]))
        logits /= sqrt(keys.get_shape()[-1].value)
        logits += upper_triangular(timesteps, dtype=logits.dtype)
        weights = tf.nn.softmax(logits)
        weighted_sum = tf.matmul(weights, values)
        combined = combine_heads(weighted_sum)
        return tf.layers.dense(combined, inner_dim, name='mix_heads')


def split_heads(inputs, num_heads):
    """
    Split up some keys, queries, or values for each head.

    Args:
      inputs: a [batch x timesteps x N] Tensor.
      num_heads: the number of heads to split out.

    Returns:
      A [batch x heads x timesteps x N/heads] Tensor.
    """
    inner_dim = inputs.get_shape()[-1].value
    assert not inner_dim % num_heads
    split_dim = inner_dim // num_heads
    shaped = tf.reshape(inputs, optimized_shape(inputs)[:-1] + (num_heads, split_dim))
    return tf.transpose(shaped, [0, 2, 1, 3])


def combine_heads(keys):
    """
    Combine the results of the heads in multi-head
    attention.

    Args:
      keys: a [batch x heads x timesteps x N/heads] Tensor.

    Returns:
      A [batch x timesteps x N] Tensor.
    """
    batch, heads, timesteps, n_over_heads = optimized_shape(keys)
    moved = tf.transpose(keys, [0, 2, 1, 3])
    return tf.reshape(moved, [batch, timesteps, heads * n_over_heads])


def upper_triangular(size, value=-np.inf, dtype=tf.float32):
    """
    Create an upper-triangular matrix.

    Args:
      size: an integer or 0-D Tensor specifying how large
        the matrix should be.
      value: the value to fill the above-diagonal with.
      dtype: the dtype of the resulting Tensor.

    Returns:
      An upper-triangular matrix. All entries at or below
        the diagonal are zero.
    """
    row_indices = tf.expand_dims(tf.range(size), axis=-1)
    col_indices = tf.expand_dims(tf.range(size), axis=0)
    mask = col_indices > row_indices
    zeros = tf.zeros([size] * 2, dtype=dtype)
    return tf.where(mask, zeros + value, zeros)


def optimized_shape(tensor):
    """
    Get the shape of a Tensor as a tuple, using python
    integers when dimensions are known statically.
    """
    dynamic_shape = tf.shape(tensor)
    static_shape = [x.value for x in tensor.get_shape()]
    return tuple(d if d is not None else dynamic_shape[i] for i, d in enumerate(static_shape))
