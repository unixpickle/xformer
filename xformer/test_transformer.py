"""
Tests for the Transformer implementation.
"""

import os
import pickle

import numpy as np
import tensorflow as tf

from . import transformer


def test_split_heads():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            inputs = np.array([
                [
                    [1, 2, 3, 4, 5, 6],
                    [7, 8, 9, 10, 11, 12],
                    [13, 14, 15, 16, 17, 18],
                ],
                [
                    [-1, -2, -3, -4, -5, -6],
                    [-7, -8, -9, -10, -11, -12],
                    [-13, -14, -15, -16, -17, -18],
                ],
                [
                    [-10, -20, -30, -40, -50, -60],
                    [-70, -80, -90, -100, -110, -120],
                    [-130, -140, -150, -160, -170, -180],
                ],
                [
                    [10, 20, 30, 40, 50, 60],
                    [70, 80, 90, 100, 110, 120],
                    [130, 140, 150, 160, 170, 180],
                ],
            ], dtype='float32')
            outputs = sess.run(transformer.split_heads(tf.constant(inputs), 3))
            expected = np.stack([inputs[..., :2], inputs[..., 2:4], inputs[..., 4:6]], axis=1)
            assert outputs.shape == expected.shape
            assert np.allclose(outputs, expected)
            assert not np.isnan(outputs).any()


def test_combine_heads():
    """
    Test that combine_heads is the inverse of split_heads.
    """
    with tf.Graph().as_default():
        with tf.Session() as sess:
            inputs = np.random.normal(size=(5, 6, 21))
            split = transformer.split_heads(tf.constant(inputs), 7)
            outputs = sess.run(transformer.combine_heads(split))
            assert inputs.shape == outputs.shape
            assert np.allclose(inputs, outputs)


def test_upper_triangular():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            actual = sess.run(transformer.upper_triangular(5, value=3))
            expected = np.array([
                [0, 3, 3, 3, 3],
                [0, 0, 3, 3, 3],
                [0, 0, 0, 3, 3],
                [0, 0, 0, 0, 3],
                [0, 0, 0, 0, 0],
            ], dtype='float32')
            assert np.allclose(actual, expected)
            assert not np.isnan(actual).any()


def test_masking():
    """
    Test that later inputs can't affect earlier outputs.
    """
    with tf.Graph().as_default():
        with tf.Session() as sess:
            inputs = np.random.normal(size=(3, 4, 6)).astype('float32')
            in_seq = tf.placeholder(tf.float32, shape=(3, 4, 6))
            out_seq = transformer.transformer_layer(in_seq, num_heads=2, hidden=24)
            sess.run(tf.global_variables_initializer())

            outputs = sess.run(out_seq, feed_dict={in_seq: inputs})
            for i in [3, 2, 1]:
                inputs[:, i] = np.random.normal(size=(3, 6)).astype('float32')
                cur_outs = sess.run(out_seq, feed_dict={in_seq: inputs})
                assert np.allclose(cur_outs[:, :i], outputs[:, :i])
                assert not np.isnan(cur_outs).any()


def test_regressions():
    """
    Test that the overall model behaves the same way as it
    did when creating this test.
    """
    with tf.Graph().as_default():
        with tf.Session() as sess:
            in_seq = tf.get_variable('in_seq',
                                     shape=(3, 4, 6),
                                     initializer=tf.truncated_normal_initializer(),
                                     dtype=tf.float64)
            out_seq = transformer.transformer_layer(in_seq, num_heads=2, hidden=24)
            sess.run(tf.global_variables_initializer())

            dir_path = os.path.dirname(os.path.realpath(__file__))
            path = os.path.join(dir_path, 'test_data', 'regressions.pkl')
            with open(path, 'rb') as in_file:
                last_result = pickle.load(in_file)
            sess.run([tf.assign(var, val) for var, val
                      in zip(tf.global_variables(), last_result['variables'])])

            actual = sess.run(out_seq)
            expected = last_result['outputs']

            assert actual.shape == expected.shape
            assert np.allclose(actual, expected)
            assert not np.isnan(actual).any()

            # Save the result like so:
            # obj = {
            #     'variables': sess.run(tf.global_variables()),
            #     'outputs': actual,
            # }
            # with open(path, 'wb+') as out_file:
            #     pickle.dump(obj, out_file)


def test_positional_encoding():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            def _make_row(idx):
                return [
                    np.cos(idx / 3 ** (0.0 / 6.0)),
                    np.sin(idx / 3 ** (0.0 / 6.0)),
                    np.cos(idx / 3 ** (2.0 / 6.0)),
                    np.sin(idx / 3 ** (2.0 / 6.0)),
                    np.cos(idx / 3 ** (4.0 / 6.0)),
                    np.sin(idx / 3 ** (4.0 / 6.0)),
                ]
            actual = sess.run(transformer.positional_encoding(5, 6, base=3, dtype=tf.float64))
            expected = np.array([_make_row(i) for i in range(5)], dtype='float64')
            assert actual.shape == expected.shape
            assert np.allclose(actual, expected)
            assert not np.isnan(actual).any()
