"""Relations Game CNN layer."""
import tensorflow.keras.layers as L


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
