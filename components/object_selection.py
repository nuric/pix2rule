"""This module contains layers for object selection."""
from typing import Dict, Any
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_probability as tfp

# Following dictionary defines configurable parameters
# so we can change them as hyperparameters later on.
# We follow a tell don't ask approach here and each
# module tells what can be configured when used.
configurable: Dict[str, Dict[str, Any]] = {
    "layer_name": {
        "type": str,
        "default": "RelaxedObjectSelection",
        "choices": ["RelaxedObjectSelection", "TopKObjectSelection"],
        "help": "Selection layer to use.",
    },
    "num_select": {
        "type": int,
        "default": 2,
        "help": "Number of object to select.",
    },
    "initial_temperature": {
        "type": float,
        "default": 0.5,
        "help": "Initial selection temperature if layer uses it.",
    },
}


class RelaxedObjectSelection(L.Layer):
    """Select a subset of objects based on object score."""

    def __init__(self, num_select: int = 2, initial_temperature: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.num_select = num_select
        self.initial_temperature = initial_temperature
        # We use a higher starting bias value so that the score inversion is more stable
        self.object_score = L.Dense(
            1, bias_initializer=tf.keras.initializers.Constant(10)
        )
        self.temperature = self.add_weight(
            name="temperature",
            initializer=tf.keras.initializers.Constant(initial_temperature),
            trainable=False,
        )

    def call(self, inputs, **kwargs):
        """Perform forward pass."""
        # inputs (B, num_objects O, embedding_size E)
        # ---------------------------
        object_scores = self.object_score(inputs)  # (B, O, 1)
        object_scores = tf.squeeze(object_scores, -1)  # (B, O)
        # ---------------------------
        # TopK selection
        # _, idxs = tf.math.top_k(object_scores, k=self.num_select)  # (B, N)
        # return tf.gather(inputs, idxs, axis=1, batch_dims=1)  # (B, N, O)
        # ---------------------------
        # Do a left to right selection
        atts = list()
        last_select = object_scores
        for _ in range(self.num_select):
            sample = tfp.distributions.RelaxedOneHotCategorical(
                self.temperature, logits=last_select
            ).sample()  # (B, O)
            sample = tf.cast(sample, tf.float32)  # (B, O)
            atts.append(sample)
            last_select = sample * (-last_select) + (1 - sample) * last_select
        object_atts = tf.stack(atts, 1)  # (B, N, O)
        # ---------------------------
        # Select the objects based on the attention
        # (B, N, O) x (B, O, E) -> (B, N, E)
        selected_objects = tf.einsum("bno,boe->bne", object_atts, inputs)
        return {
            "object_scores": object_scores,
            "object_atts": object_atts,
            "objects": selected_objects,
        }

    def get_config(self):
        """Serialisable configuration dictionary."""
        config = super().get_config()
        config.update(
            {
                "num_select": self.num_select,
                "initial_temperature": self.initial_temperature,
            }
        )
        return config


class TopKObjectSelection(L.Layer):
    """Select a subset of objects based on top object score ."""

    def __init__(self, num_select: int = 2, **kwargs):
        # Remove unused configurable arguments
        if "initial_temperature" in kwargs:
            del kwargs["initial_temperature"]
        super().__init__(**kwargs)
        self.num_select = num_select
        # We use a higher starting bias value so that the score inversion is more stable
        self.object_score = L.Dense(1)

    def call(self, inputs, **kwargs):
        """Perform forward pass."""
        # inputs (B, num_objects O, embedding_size E)
        # ---------------------------
        object_scores = self.object_score(inputs)  # (B, O, 1)
        object_scores = tf.squeeze(object_scores, -1)  # (B, O)
        # ---------------------------
        # TopK selection
        _, idxs = tf.math.top_k(object_scores, k=self.num_select)  # (B, N)
        object_atts = tf.gather(tf.eye(tf.shape(inputs)[1]), idxs, axis=0)  # (B, N, O)
        selected_objects = tf.gather(inputs, idxs, axis=1, batch_dims=1)  # (B, N, E)
        # ---------------------------
        return {
            "object_scores": object_scores,
            "object_atts": object_atts,
            "objects": selected_objects,
        }

    def get_config(self):
        """Serialisable configuration dictionary."""
        config = super().get_config()
        config.update(
            {
                "num_select": self.num_select,
            }
        )
        return config
