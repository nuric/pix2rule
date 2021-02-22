"""This module contains utility layers."""
from typing import List, Dict
import tensorflow as tf


class SpacialFlatten(tf.keras.layers.Layer):
    """Flattens input spacial dimensions."""

    def call(self, inputs: tf.Tensor, **kwargs):
        """Flatten except batch and last dimension."""
        # tensor (B, ..., X)
        batch_size = tf.shape(inputs)[0]  # B
        # We take this differently as above method gives None at compile time
        # which makes upstream layers unhappy. But it safe to assume the number
        # of features is known before hand.
        features_size = inputs.shape[-1]  # X
        assert features_size is not None, "Last dimension to spacial flatten is None."
        # Handle dimension erasure case, if we use -1 and all the dimensions are known
        # we just lose dimension size information, so let's check if we know the dimensions
        # to reshape, if so then we can take their product
        middle_dim = -1
        if all(i is not None for i in inputs.shape[1:-1]):
            # We know all the dimensions
            middle_dim = tf.reduce_prod(inputs.shape[1:-1])
        new_shape = tf.stack([batch_size, middle_dim, features_size], 0)  # [B, -1, X]
        return tf.reshape(inputs, new_shape)  # (B, ., X)


class MergeFacts(tf.keras.layers.Layer):
    """Merge dictionary based fact tensors."""

    def call(self, inputs: List[Dict[str, tf.Tensor]], **kwargs):
        """Merging facts based on arity by concatenating them."""
        # inputs [{'nullary': ..., 'binary': ...}, {'binary': ...}]
        # This layer needs to be used with care as it assumes that every feature of every
        # object is already computed and we just need to concatenate them. That is, all facts
        # contain all objects and their features are concatenated.
        facts = {**inputs[0]}  # Start with left most and reduce towards right
        for fact_dict in inputs[1:]:
            for key, tensor in fact_dict.items():
                facts[key] = (
                    tf.concat([facts[key], tensor], -1) if key in facts else tensor
                )
        # Optionally add missing nullary entry
        if "nullary" not in facts:
            facts["nullary"] = tf.zeros(
                (tf.shape(facts[next(iter(facts.keys()))])[0], 0), dtype=tf.float32
            )
        return facts


class Shuffle(tf.keras.layers.Layer):
    """Shuffle given tensor on a given axis.
    This layer applies the same shuffle to all elements along the axis."""

    def __init__(self, shuffle_axis: int = 1, seed: int = None, **kwargs):
        super().__init__(**kwargs)
        self.shuffle_axis = shuffle_axis
        self.seed = seed

    def call(self, inputs, **kwargs):
        """Perform forward pass."""
        # inputs (B, ..., X, ...)
        # ---------------------------
        # Generate random idxs along that axis
        ridxs = tf.random.shuffle(
            tf.range(tf.shape(inputs)[self.shuffle_axis]), seed=self.seed
        )  # (X,)
        return tf.gather(inputs, ridxs, axis=self.shuffle_axis)  # (B, ..., X, ...)

    def get_config(self):
        """Serialisable configuration dictionary."""
        config = super().get_config()
        config.update({"shuffle_axis": self.shuffle_axis, "seed": self.seed})
        return config
