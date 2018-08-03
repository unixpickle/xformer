"""
Using Transformer decoders as TF RNNCells.

This helps for two things:

  1. Faster sampling.
  2. Compatibility with RNN-based code.

"""

from math import sqrt

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell  # pylint: disable=E0611

from .transformer import optimized_shape, split_heads, transformer_layer

# Disable warning about "compute_output_shape" not being
# overridden, since most RNNCells don't seem to do so.
# pylint: disable=W0223


class BaseTransformerCell(RNNCell):
    """
    Base class for RNNCells that implement Transformers.
    """

    def __init__(self,
                 pos_encoding,
                 num_layers=6,
                 num_heads=8,
                 hidden=2048,
                 fc_activation=tf.nn.relu,
                 trainable=True,
                 name='transformer',
                 dtype=None):
        """
        Create a new Transformer cell.

        Args:
          pos_encoding: a positional encoding Tensor. The
            time and inner dimensions must both be known
            statically, since the state shape depends on
            both things.
          num_layers: the number of layers.
          num_heads: the number of attention heads.
          hidden: the FC hidden layer size.
          fc_activation: the activation for the FC layers.
          trainable: use trainable variables.
          name: the scope name.
          dtype: the datatype.
        """
        super(BaseTransformerCell, self).__init__(trainable=trainable, name=name, dtype=dtype)
        self.pos_encoding = pos_encoding
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden = hidden
        self.fc_activation = fc_activation

    @property
    def time_horizon(self):
        """
        Get the number of timesteps that the model can see
        at once.
        """
        return self.pos_encoding.get_shape()[0].value

    @property
    def output_size(self):
        return self.pos_encoding.get_shape()[1].value

    def zero_state(self, batch_size, dtype):
        return tuple(tf.zeros([batch_size] + [x.value for x in shape.dims], dtype=dtype)
                     for shape in self.state_size)


class UnlimitedTransformerCell(BaseTransformerCell):
    """
    An RNNCell that implements a Transformer in a way that
    supports arbitrarily long sequences, but is not
    efficient for sequences shorter than the horizon.
    """
    @property
    def state_size(self):
        return [tf.TensorShape(()), self.pos_encoding.get_shape()]

    def call(self, inputs, state):  # pylint: disable=W0221
        timestep_idxs = tf.cast(state[0], tf.int32)
        full_inputs = inject_at_timestep(timestep_idxs, state[1], inputs)
        new_states = [tf.clip_by_value(state[0] + 1.0, 0, self.time_horizon - 1),
                      shift_overflows(timestep_idxs, full_inputs)]
        outputs = full_inputs + self.pos_encoding
        outputs = outputs[:, :tf.reduce_max(timestep_idxs) + 1]
        for _ in range(self.num_layers):
            outputs = transformer_layer(outputs, num_heads=self.num_heads, hidden=self.hidden,
                                        activation=self.fc_activation)
        batch_size = optimized_shape(outputs)[0]
        outputs = tf.gather_nd(outputs,
                               tf.stack([tf.range(batch_size, dtype=tf.int32), timestep_idxs],
                                        axis=-1))
        return outputs, tuple(new_states)


class LimitedTransformerCell(BaseTransformerCell):
    """
    An RNNCell that implements a Transformer, but does not
    support sequences longer than the horizon.

    Back-propagation through this RNN is not efficient,
    since it is not properly batched.
    However, the forward pass is as fast as it could be.

    State tuples are of the form:

        (current_step, keys_1, values_1, keys_2, values_2, ...)

    """
    @property
    def state_size(self):
        return [tf.TensorShape(())] + [self.pos_encoding.get_shape()] * 2 * self.num_layers

    def call(self, inputs, state):  # pylint: disable=W0221
        timestep_idxs = tf.cast(state[0], tf.int32)
        assert_op = tf.Assert(tf.reduce_any(timestep_idxs < self.time_horizon),
                              ['LimitedTransformerCell time horizon exceeded'])
        with tf.control_dependencies([assert_op]):
            layer_inputs = inputs + tf.gather(self.pos_encoding, timestep_idxs)
            new_states = [state[0] + 1.0]
        for layer_idx in range(self.num_layers):
            keys = state[1 + layer_idx * 2]
            values = state[2 + layer_idx * 2]
            new_keys, new_values, layer_inputs = self.transformer_layer(
                timestep_idxs,
                keys,
                values,
                layer_inputs,
            )
            new_states.extend([new_keys, new_values])
        return layer_inputs, tuple(new_states)

    def transformer_layer(self, timestep_idxs, keys, values, inputs, scope='transformer'):
        """
        Apply a layer of the transformer for a timestep.

        Args:
          timestep_idxs: a 1-D Tensor of indices.
          keys: the key history for the layer.
          values: the value history for the layer.
          inputs: the inputs for the current timesteps.

        Returns:
          A tuple (new_keys, new_values, outputs):
            new_keys: the new key history.
            new_values: the new value history.
            outputs: a [batch x N] Tensor from the layer.
        """
        with tf.variable_scope(None, default_name=scope):
            # pylint: disable=E1101
            new_keys, new_values, outputs = self.attention_layer(
                timestep_idxs,
                keys,
                values,
                inputs,
            )
            inputs = tf.contrib.layers.layer_norm(inputs + outputs,
                                                  center=True,
                                                  scale=True,
                                                  begin_norm_axis=-1)
            outputs = self.fc_layer(inputs)
            inputs = tf.contrib.layers.layer_norm(inputs + outputs,
                                                  center=True,
                                                  scale=True,
                                                  begin_norm_axis=-1)
            return new_keys, new_values, inputs

    def attention_layer(self, timestep_idxs, keys, values, inputs, scope='attention'):
        """
        Apply masked attention for a single timestep.

        Args:
          timestep_idxs: a 1-D Tensor of indices.
          keys: the key history for the layer.
          values: the value history for the layer.
          inputs: the inputs for the current timesteps.
          scope: the scope name.

        Returns:
          A tuple (new_keys, new_values, outputs):
            new_keys: the new key history.
            new_values: the new value history.
            outputs: a [batch x N] Tensor from the layer.
        """
        batch_size = optimized_shape(timestep_idxs)[0]
        with tf.variable_scope(None, default_name=scope):
            projected = tf.layers.dense(inputs, self.output_size * 3, name='key_query_value')
            projected = tf.expand_dims(projected, axis=1)

            # Resulting shape: [batch x 1 x N]
            next_keys, next_queries, next_values = tf.split(projected, 3, axis=-1)

            keys = inject_at_timestep(timestep_idxs, keys, next_keys[:, 0])
            values = inject_at_timestep(timestep_idxs, values, next_values[:, 0])

            # Resulting shape: [batch x heads x timesteps x N/heads]
            split_keys = split_heads(keys, self.num_heads)
            split_values = split_heads(values, self.num_heads)

            # Resulting shape: [batch x heads x N/heads]
            split_queries = split_heads(next_queries, self.num_heads)[:, :, 0]

            attended = self.raw_attention(timestep_idxs, split_keys, split_values, split_queries)
            combined = tf.reshape(attended, [batch_size, self.output_size])
            mixed = tf.layers.dense(combined, self.output_size, name='mix_heads')

            return (keys, values, mixed)

    def raw_attention(self, timestep_idxs, keys, values, queries):
        """
        Apply attention for a single timestep given the
        latest keys, values, and queries.

        Args:
          timestep_idxs: a 1-D Tensor of indices.
          keys: the input keys for the layer, of shape
            [batch x heads x timesteps x N/heads].
          values: the input values for the layer, of shape
            [batch x heads x timesteps x N/heads].
          queries: queries for the latest timestep.
            Of shape [batch x heads x N/heads].

        Returns:
          A [batch x heads x N/heads] Tensor.
        """
        max_timestep = tf.reduce_max(timestep_idxs) + 1
        batch_size = optimized_shape(queries)[0]

        # Resulting shape: [batch x heads x 1 x N/heads]
        expanded_queries = tf.expand_dims(queries, axis=2)

        # Resulting shape: [batch x heads x 1 x max_timestep]
        logits = tf.matmul(expanded_queries, tf.transpose(keys[:, :, :max_timestep], [0, 1, 3, 2]))
        logits /= sqrt(keys.get_shape()[-1].value)
        logits += tf.reshape(sequence_masks(timestep_idxs, max_timestep, logits.dtype),
                             [batch_size, 1, 1, max_timestep])
        weights = tf.nn.softmax(logits)

        # Resulting shape: [batch x heads x 1 x N/heads]
        weighted_sum = tf.matmul(weights, values[:, :, :max_timestep])

        return weighted_sum[:, :, 0]

    def fc_layer(self, inputs):
        """
        Apply the fully-connected layer.

        Args:
          inputs: a [batch x N] input.

        Returns:
          A [batch x N] output.
        """
        outs = inputs
        outs = tf.layers.dense(outs, self.hidden, activation=self.fc_activation)
        outs = tf.layers.dense(outs, self.output_size, activation=self.fc_activation)
        return outs


def inject_at_timestep(timestep_idxs, sequence, new_values):
    """
    Inject a batch of values at respective timesteps of a
    sequence.

    Args:
      timestep_idxs: a 1-D Tensor of indices.
      sequence: a [batch x timesteps x N] sequence.
      new_values: a [batch x N] Tensor where each batch
        element should be injected into the given timestep
        index of the input sequence.

    Returns:
      A [batch x timestep x N] sequence.
    """
    # Create a [batch x timestep] Tensor of timesteps.
    ranges = tf.range(optimized_shape(sequence)[1], dtype=timestep_idxs.dtype)
    ranges = tf.tile(tf.expand_dims(ranges, axis=0), [tf.shape(sequence)[0], 1])

    # Create a mask of shape [batch x timestep x N].
    mask = tf.equal(ranges - tf.expand_dims(timestep_idxs, axis=-1), tf.zeros_like(ranges))
    mask = tf.tile(tf.expand_dims(mask, axis=-1), [1, 1, sequence.get_shape()[-1].value])

    new_seq = tf.zeros_like(sequence) + tf.expand_dims(new_values, axis=1)
    return tf.where(mask, new_seq, sequence)


def shift_overflows(timestep_idxs, sequence):
    """
    For sequences where the final timestep was just used,
    shift the sequence over by one for the next state.

    Args:
      timestep_idxs: a 1-D Tensor of indices.
      sequence: a [batch x timesteps x N] sequence.

    Returns:
      A new [batch x timesteps x N] sequence.
    """
    shifted = tf.concat([sequence[:, 1:], tf.zeros_like(sequence[:, :1])], axis=1)
    return tf.where(tf.equal(timestep_idxs, sequence.get_shape()[1].value - 1), shifted, sequence)


def sequence_masks(timestep_idxs, sequence_length, dtype):
    """
    Create a mask that can be added to a batch of
    sequences to make the sequences -inf after the latest
    timestep.

    Args:
      timestep_idxs: a 1-D Tensor of indices.
      sequence_length: the max sequence length.
      dtype: the resulting datatype.

    Returns:
      A mask of shape [batch x sequence_length].
    """
    batch_size = optimized_shape(timestep_idxs)[0]
    indices = tf.tile(tf.expand_dims(tf.range(sequence_length), axis=0), [batch_size, 1])
    greater = (tf.expand_dims(timestep_idxs, axis=-1) >= indices)
    zeros = tf.zeros([batch_size, sequence_length], dtype=dtype)
    return tf.where(greater, zeros, zeros - np.inf)
