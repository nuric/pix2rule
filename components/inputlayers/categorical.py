"""Categorical input layer."""
import tensorflow as tf


class OneHotCategoricalInput(tf.keras.layers.Layer):
    """Process a categorical input as one-hot."""

    def __init__(self, num_categories: int, activation: str = "sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.num_categories = num_categories
        self.activation = activation

    def call(self, inputs: tf.Tensor, **kwargs):
        """Forward pass which encodes inputs as one-hot."""
        # inputs (B,)
        off_value = -1.0 if self.activation == "tanh" else 0.0
        return tf.one_hot(
            inputs, self.num_categories, on_value=1.0, off_value=off_value
        )

    def get_config(self):
        """Serialisable configuration dictionary."""
        config = super().get_config()
        config.update(
            {"num_categories": self.num_categories, "activation": self.activation}
        )
        return config
