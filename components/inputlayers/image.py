"""Image processing layers."""
import tensorflow as tf
import tensorflow.keras.layers as L


class BaseImageInput(L.Layer):
    """Base class for image input, often children class will be CNNs."""

    def __init__(self, hidden_size: int = 32, activation: str = "relu", **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.activation = activation

    def get_config(self):
        """Serialisable configuration dictionary."""
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "activation": self.activation,
            }
        )
        return config


class RelationsGameImageInput(BaseImageInput):
    """Relations Game CNN to process images."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Values taken from paper: https://arxiv.org/pdf/1905.10307.pdf
        # and downsized to save computing resources
        self.cnn_l1 = L.Conv2D(
            self.hidden_size, 4, strides=2, activation=self.activation
        )

    def call(self, inputs, **kwargs):
        """Process the provided image."""
        # inputs (B, 12, 12, 3)
        return self.cnn_l1(inputs)  # (B, 5, 5, 32)


class RelationsGamePixelImageInput(BaseImageInput):
    """Relations Game CNN to process images at pixel level."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder_cnn = tf.keras.Sequential(
            [
                L.Conv2D(
                    self.hidden_size,
                    kernel_size=5,
                    padding="SAME",
                    activation=self.activation,
                ),
                L.Conv2D(
                    self.hidden_size,
                    kernel_size=5,
                    padding="SAME",
                    activation=self.activation,
                ),
                L.Conv2D(
                    self.hidden_size,
                    kernel_size=5,
                    padding="SAME",
                    activation=self.activation,
                ),
            ],
            name="encoder_cnn",
        )

    def call(self, inputs, **kwargs):
        """Process the provided image."""
        # inputs (B, 12, 12, 3)
        return self.encoder_cnn(inputs)  # (B, 12, 12, H)