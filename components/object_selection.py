"""This module contains layers for object selection."""
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_probability as tfp

from reportlib import report_tensor


class ObjectSelection(L.Layer):
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
        report_tensor("object_scores", object_scores)
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
        report_tensor("object_atts", object_atts)
        # ---------------------------
        # Select the objects based on the attention
        # (B, N, O) x (B, O, E) -> (B, N, E)
        selected_objects = tf.einsum("bno,boe->bne", object_atts, inputs)
        return selected_objects

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
