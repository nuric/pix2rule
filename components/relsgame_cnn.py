"""Relations Game CNN layer."""
import tensorflow as tf
import tensorflow.keras.layers as L

from . import slot_attention


class RelationsGameCNN(L.Layer):
    """Relations Game CNN to process images."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Values taken from paper: https://arxiv.org/pdf/1905.10307.pdf
        # and downsized to save computing resources
        self.cnn_l1 = L.Conv2D(32, 4, strides=2, activation="relu")
        self.reshape = L.Reshape((25, 32))  # Flatten the image regions

    def call(self, inputs, **kwargs):
        """Process the provided image."""
        # inputs (B, 12, 12, 3)
        feats = self.cnn_l1(inputs)  # (B, 5, 5, 32)
        return self.reshape(feats)  # (B, 25, 32)


class RelationsGamePixelCNN(L.Layer):
    """Relations Game CNN to process images at pixel level."""

    def __init__(self, hidden_size: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = 32
        self.encoder_cnn = tf.keras.Sequential(
            [
                L.Conv2D(hidden_size, kernel_size=5, padding="SAME", activation="relu"),
                L.Conv2D(hidden_size, kernel_size=5, padding="SAME", activation="relu"),
                L.Conv2D(hidden_size, kernel_size=5, padding="SAME", activation="relu"),
            ],
            name="encoder_cnn",
        )
        self.pos_enc = slot_attention.SoftPositionEmbed(hidden_size, (12, 12))
        self.reshape = L.Reshape((144, hidden_size))  # Flatten the image regions

    def call(self, inputs, **kwargs):
        """Process the provided image."""
        # inputs (B, 12, 12, 3)
        feats = self.encoder_cnn(inputs)  # (B, 12, 12, H)
        feats = self.pos_enc(feats)  # (B, 12, 12, H)
        return self.reshape(feats)  # (B, 144, H)

    def get_config(self):
        """Serialisable configuration dictionary."""
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
            }
        )
        return config
