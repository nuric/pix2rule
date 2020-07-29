"""Relations Game CNN layer."""
import tensorflow as tf
import tensorflow.keras.layers as L


class RelationsGameCNN(L.Layer):
    """Relations Game CNN to process images."""

    def __init__(self, **kwargs):
        super(RelationsGameCNN, self).__init__(**kwargs)
        # Values taken from paper: https://arxiv.org/pdf/1905.10307.pdf
        self.cnn_l1 = L.Conv2D(32, 12, strides=6, activation="relu")
        self.reshape = L.Reshape((25, 32))  # Flatten the image regions

    def call(self, inputs: tf.Tensor):
        """Process the provided image."""
        # inputs (B, 36, 36, 3)
        feats = self.cnn_l1(inputs)  # (B, 5, 5, 32)
        return self.reshape(feats)  # (B, 25, 32)
