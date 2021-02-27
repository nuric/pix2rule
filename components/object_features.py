"""Layers to compute object features for predicate grounding."""
from typing import Dict, Any
import tensorflow as tf
import tensorflow.keras.layers as L

configurable: Dict[str, Dict[str, Any]] = {
    "layer_name": {
        "type": str,
        "default": "LinearObjectFeatures",
        "choices": ["LinearObjectFeatures"],
        "help": "Selection layer to use.",
    },
    "unary_size": {
        "type": int,
        "default": 4,
        "help": "Number of unary predicates for objects.",
    },
    "binary_size": {
        "type": int,
        "default": 8,
        "help": "Number of binary predicates for objects.",
    },
}


class LinearObjectFeatures(L.Layer):
    """Computes linear object features."""

    def __init__(self, unary_size: int = 4, binary_size: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.unary_size = unary_size
        self.binary_size = binary_size
        self.unary_model = L.Dense(
            unary_size,
            activation="sigmoid",
            name="unary_model",
            # bias_initializer=tf.keras.initializers.Constant(4),
            # bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=4.0),
        )
        self.binary_model = L.Dense(
            binary_size,
            activation="sigmoid",
            name="binary_model",
            # bias_initializer=tf.keras.initializers.Constant(4),
            # bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=4.0),
        )

    def build(self, input_shape: tf.TensorShape):
        """Build layer weights."""
        # input_shape (B, num_objects N, embedding_size E)
        num_objects = input_shape[1]  # N
        # The following captures all the non-diagonal indices representing p(X,Y)
        # and omitting p(X,X). So every object compared to every other object
        binary_idxs = tf.where(1 - tf.eye(num_objects, dtype=tf.int8))  # (O*(O-1), 2)
        # pylint: disable=attribute-defined-outside-init
        self.binary_idxs = tf.reshape(
            binary_idxs, [num_objects, num_objects - 1, 2]
        )  # (O, O-1, 2)

    def call(self, inputs: tf.Tensor, **kwargs):
        """Perform forward pass."""
        # objects (B, num_objects N, embedding_size E)
        # ---------------------------
        # Compute unary features
        unary_preds = self.unary_model(inputs)  # (B, O, P1)
        # ---------------------------
        # Compute binary features
        arg1 = tf.gather(inputs, self.binary_idxs[..., 0], axis=1)  # (B, O, O-1, E)
        arg2 = tf.gather(inputs, self.binary_idxs[..., 1], axis=1)  # (B, O, O-1, E)
        paired_objects = arg1 - arg2  # (B, O, O-1, E)
        binary_preds = self.binary_model(paired_objects)  # (B, O, O-1, P2)
        # ---------------------------
        return {
            "unary": unary_preds,
            "binary": binary_preds,
        }

    def get_config(self):
        """Serialisable configuration dictionary."""
        config = super().get_config()
        config.update(
            {
                "unary_size": self.unary_size,
                "binary_size": self.binary_size,
            }
        )
        return config
