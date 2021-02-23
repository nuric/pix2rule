"""Categorical input layer."""
import tensorflow as tf


class OneHotCategoricalInput(tf.keras.layers.Layer):
    """Process a categorical input as one-hot."""

    def __init__(self, num_categories: int, **kwargs):
        super().__init__(**kwargs)
        self.num_categories = num_categories

    def call(self, inputs: tf.Tensor, **kwargs):
        """Forward pass which encodes inputs as one-hot."""
        # inputs (B,)
        return tf.one_hot(inputs, self.num_categories, on_value=1.0, off_value=0.0)

    def get_config(self):
        """Serialisable configuration dictionary."""
        config = super().get_config()
        config.update(
            {
                "num_categories": self.num_categories,
            }
        )
        return config
