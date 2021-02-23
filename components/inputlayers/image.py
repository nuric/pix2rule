"""Image processing layers."""
from typing import Dict, Any
import tensorflow as tf
import tensorflow.keras.layers as L

# Following dictionary defines configurable parameters
# so we can change them as hyperparameters later on.
# We follow a tell don't ask approach here and each
# module tells what can be configured when used.
configurable: Dict[str, Dict[str, Any]] = {
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
    "with_position": {
        "action": "store_true",
        "help": "Append position coordinates.",
    },
}


class BaseImageInput(L.Layer):
    """Base class for image input, often children class will be CNNs."""

    def __init__(
        self,
        hidden_size: int = 32,
        activation: str = "relu",
        noise_stddev: float = 0.0,
        with_position: bool = False,
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
        self.with_position = with_position
        self.encoder_cnn = lambda x: x  # ID layer for pass through

    def call(self, inputs, **kwargs):
        """Process the provided image."""
        # inputs (B, W, H, C)
        image_input = self.noise_layer(inputs) if self.noise_stddev > 0.0 else inputs
        encoded = self.encoder_cnn(image_input)
        return self.post_call(encoded)

    def post_call(self, inputs):
        """Perform post call operations."""
        # inputs (B, W, H, E)
        if not self.with_position:
            return inputs
        # Construct and append positions
        spacial_dims = inputs.shape[1:-1]  # [W, H]
        assert all(
            i > 0 for i in spacial_dims
        ), f"Spacial dimensions need to be non-zero, {spacial_dims}."
        ranges = [tf.linspace(0.0, 1.0, res) for res in spacial_dims]  # [(W,), (H,)]
        grid = tf.meshgrid(*ranges, indexing="ij")  # [(W, H), (W, H)]
        grid = tf.stack(grid, -1)  # (W, H, 2)
        grid = tf.concat([grid, 1 - grid], -1)  # (W, H, 4)
        grid = tf.repeat(grid[None], tf.shape(inputs)[0], axis=0)  # (B, W, H, 4)
        return tf.concat([inputs, grid], -1)  # (B, W, H, E+4)

    def get_config(self):
        """Serialisable configuration dictionary."""
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "activation": self.activation,
                "noise_stddev": self.noise_stddev,
                "with_position": self.with_position,
            }
        )
        return config


class RelationsGameImageInput(BaseImageInput):
    """Relations Game CNN to process images."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Values taken from paper: https://arxiv.org/pdf/1905.10307.pdf
        # and downsized to save computing resources
        self.encoder_cnn = L.Conv2D(
            self.hidden_size, 4, strides=2, activation=self.activation
        )


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
