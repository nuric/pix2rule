"""Flatten layer for tensors with features in last dimension."""
import tensorflow as tf


class SpacialFlatten(tf.keras.layers.Layer):
    """Flattens input spacial dimensions."""

    def call(self, inputs: tf.Tensor, **kwargs):
        """Flatten except batch and last dimension."""
        # tensor (B, ..., X)
        shape = tf.shape(inputs)  # [B, ..., X]
        new_shape = tf.concat([shape[0], -1, shape[-1]], 0)  # [B, -1, X]
        return tf.reshape(inputs, new_shape)  # (B, ., X)
