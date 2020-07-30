"""Shuffle layer."""
import tensorflow as tf


class Shuffle(tf.keras.layers.Layer):
    """Shuffle given tensor on a given axis.
    This layer applies the same shuffle to all elements along the axis."""

    def __init__(self, shuffle_axis: int = 1, seed: int = None, **kwargs):
        super(Shuffle, self).__init__(**kwargs)
        self.shuffle_axis = shuffle_axis
        self.seed = seed

    def call(self, inputs, **kwargs):
        """Perform forward pass."""
        # inputs (B, ..., X, ...)
        # ---------------------------
        # tf.random.shuffle only shuffles axis 0
        if self.shuffle_axis == 0:
            return tf.random.shuffle(inputs, seed=self.seed)
        # We need to swap axis with batch axis
        perm = list(range(len(inputs.shape)))
        perm[0], perm[self.shuffle_axis] = perm[self.shuffle_axis], perm[0]
        transposed = tf.transpose(inputs, perm)  # (X, ..., B, ...)
        shuffled = tf.random.shuffle(transposed, seed=self.seed)  # (X, ..., B, ...)
        return tf.transpose(shuffled, perm)  # (B, ..., X, ...)

    def get_config(self):
        """Serialisable configuration dictionary."""
        config = super(Shuffle, self).get_config()
        config.update({"shuffle_axis": self.shuffle_axis, "seed": self.seed})
        return config
