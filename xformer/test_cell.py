"""
Tests for the Transformer RNNCell.
"""

import pytest

import numpy as np
import tensorflow as tf

from .transformer import positional_encoding, transformer_layer
from .cell import (LimitedTransformerCell, UnlimitedTransformerCell,
                   inject_at_timestep, sequence_masks)


def test_inject_at_timestep():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            in_seq = tf.constant(np.array([
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                ],
                [
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ],
                [
                    [17, 18, 19, 20],
                    [21, 22, 23, 24],
                ],
            ], dtype='float32'))
            injection = tf.constant(np.array([
                [-1, -2, -3, -4],
                [-5, -6, -7, -8],
                [-9, -10, -11, -12],
            ], dtype='float32'))

            indices = np.array([0, 1, 0], dtype='int32')
            injected = sess.run(inject_at_timestep(indices, in_seq, injection))

            expected = np.array([
                [
                    [-1, -2, -3, -4],
                    [5, 6, 7, 8],
                ],
                [
                    [9, 10, 11, 12],
                    [-5, -6, -7, -8],
                ],
                [
                    [-9, -10, -11, -12],
                    [21, 22, 23, 24],
                ],
            ], dtype='float32')
            assert (injected == expected).all()


def test_sequence_masks():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            indices = tf.constant(np.array([3, 1, 2], dtype='int32'))
            actual = sess.run(sequence_masks(indices, tf.constant(4, dtype=tf.int32), tf.float32))
            expected = np.array([
                [0, 0, 0, 0],
                [0, 0, -np.inf, -np.inf],
                [0, 0, 0, -np.inf],
            ], dtype='float32')
            assert (actual == expected).all()


@pytest.mark.parametrize('cell_cls', [LimitedTransformerCell, UnlimitedTransformerCell])
@pytest.mark.parametrize('num_layers', [1, 2, 6])
def test_basic_equivalence(cell_cls, num_layers):
    """
    Test that both transformer implementations produce the
    same outputs when applied to a properly-sized
    sequence.
    """
    with tf.Graph().as_default():
        with tf.Session() as sess:
            pos_enc = positional_encoding(4, 6, dtype=tf.float64)
            in_seq = tf.get_variable('in_seq',
                                     shape=(3, 4, 6),
                                     initializer=tf.truncated_normal_initializer(),
                                     dtype=tf.float64)
            cell = cell_cls(pos_enc, num_layers=num_layers, num_heads=2, hidden=24)
            actual, _ = tf.nn.dynamic_rnn(cell, in_seq, dtype=tf.float64)
            with tf.variable_scope('rnn', reuse=True):
                with tf.variable_scope('transformer', reuse=True):
                    expected = in_seq + pos_enc
                    for _ in range(num_layers):
                        expected = transformer_layer(expected, num_heads=2, hidden=24)
            sess.run(tf.global_variables_initializer())

            actual, expected = sess.run((actual, expected))

            assert not np.isnan(actual).any()
            assert not np.isnan(expected).any()
            assert actual.shape == expected.shape
            assert np.allclose(actual, expected)


@pytest.mark.parametrize('cell_cls', [UnlimitedTransformerCell])
def test_past_horizon(cell_cls):
    """
    Test the cell when the input sequence is longer than
    the time horizon.
    """
    with tf.Graph().as_default():
        with tf.Session() as sess:
            pos_enc = positional_encoding(4, 6, dtype=tf.float64)
            in_seq = tf.get_variable('in_seq',
                                     shape=(3, 5, 6),
                                     initializer=tf.truncated_normal_initializer(),
                                     dtype=tf.float64)
            cell = cell_cls(pos_enc, num_layers=3, num_heads=2, hidden=24)
            actual, _ = tf.nn.dynamic_rnn(cell, in_seq, dtype=tf.float64)

            def apply_regular(sequence):
                with tf.variable_scope('rnn', reuse=True):
                    with tf.variable_scope('transformer', reuse=True):
                        expected = sequence + pos_enc
                        for _ in range(3):
                            expected = transformer_layer(expected, num_heads=2, hidden=24)
                return expected
            expected = tf.concat([apply_regular(in_seq[:, :-1]),
                                  apply_regular(in_seq[:, 1:])[:, -1:]], axis=1)
            sess.run(tf.global_variables_initializer())

            actual, expected = sess.run((actual, expected))

            assert not np.isnan(actual).any()
            assert not np.isnan(expected).any()
            assert actual.shape == expected.shape
            assert np.allclose(actual, expected)
