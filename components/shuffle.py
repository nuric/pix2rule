"""Shuffle layer."""
import tensorflow as tf


class Shuffle(tf.keras.layers.Layer):
    """Shuffle given tensor on a given axis.
    This layer applies the same shuffle to all elements along the axis."""

    def __init__(self, shuffle_axis: int = 1, seed: int = None, **kwargs):
        super().__init__(**kwargs)
        self.shuffle_axis = shuffle_axis
        self.seed = seed

    def call(self, inputs, **kwargs):
        """Perform forward pass."""
        # inputs (B, ..., X, ...)
        # ---------------------------
        # Generate random idxs along that axis
        ridxs = tf.random.shuffle(
            tf.range(tf.shape(inputs)[self.shuffle_axis]), seed=self.seed
        )  # (X,)
        return tf.gather(inputs, ridxs, axis=self.shuffle_axis)  # (B, ..., X, ...)

    def get_config(self):
        """Serialisable configuration dictionary."""
        config = super().get_config()
        config.update({"shuffle_axis": self.shuffle_axis, "seed": self.seed})
        return config
