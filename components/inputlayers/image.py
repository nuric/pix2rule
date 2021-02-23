"""Image processing layers."""
import tensorflow as tf
import tensorflow.keras.layers as L

# Following dictionary defines configurable parameters
# so we can change them as hyperparameters later on.
# We follow a tell don't ask approach here and each
# module tells what can be configured when used.
configurable = {
    "layer_name": {
        "type": str,
        "default": "RelationsGameImageInput",
        "choices": ["RelationsGameImageInput", "RelationsGamePixelImageInput"],
        "help": "Image input layer to use.",
    },
    "hidden_size": {
        "type": int,
        "default": 32,
        "help": "Hidden size of image pipeline layers.",
    },
    "activation": {
        "type": str,
        "choices": ["relu", "sigmoid", "tanh"],
        "help": "Activation of hidden and final layer of image pipeline.",
    },
    "noise_stddev": {
        "type": float,
        "default": 0.0,
        "help": "Standard deviation of added noise to image input.",
    },
}


class BaseImageInput(L.Layer):
    """Base class for image input, often children class will be CNNs."""

    def __init__(
        self,
        hidden_size: int = 32,
        activation: str = "relu",
        noise_stddev: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.activation = activation
        assert (
            noise_stddev >= 0
        ), f"Noise stddev needs to be non-negative got {noise_stddev}."
        self.noise_stddev = noise_stddev
        self.noise_layer = L.GaussianNoise(noise_stddev)

    def call(self, inputs, **kwargs):
        """Process the provided image."""
        # inputs (B, W, H, C)
        return self.noise_layer(inputs) if self.noise_stddev > 0.0 else inputs

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
        inputs = super().call(inputs)  # (B, 12, 12, 3)
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
        inputs = super().call(inputs)  # (B, 12, 12, 3)
        return self.encoder_cnn(inputs)  # (B, 12, 12, H)
