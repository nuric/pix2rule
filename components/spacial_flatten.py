"""Flatten layer for tensors with features in last dimension."""
import tensorflow as tf


class SpacialFlatten(tf.keras.layers.Layer):
    """Flattens input spacial dimensions."""

    def call(self, inputs: tf.Tensor, **kwargs):
        """Flatten except batch and last dimension."""
        # tensor (B, ..., X)
        batch_size = tf.shape(inputs)[0]  # B
        # We take this differently as above method gives None at compile time
        # which makes upstream layers unhappy. But it safe to assume the number
        # of features is known before hand.
        features_size = inputs.shape[-1]  # X
        assert features_size is not None, "Last dimension to spacial flatten is None."
        new_shape = tf.stack([batch_size, -1, features_size], 0)  # [B, -1, X]
        return tf.reshape(inputs, new_shape)  # (B, ., X)
