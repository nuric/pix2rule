"""This module contains utility layers."""
from typing import List, Dict, Tuple
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


class SpacialBroadcast(tf.keras.layers.Layer):
    """Broadcast a set of objects into a given spacial resolution."""

    def __init__(self, resolution: List[int], **kwargs):
        super().__init__(**kwargs)
        self.resolution = resolution

    def call(self, inputs: tf.Tensor, **kwargs):
        """Broadcast given inputs."""
        # inputs (B, O, E)
        assert (
            len(inputs.shape) == 3
        ), f"Expected 3 dimensional tensor for spacial broadcast, got {len(inputs.shape)}"
        # ---------------------------
        objects = tf.reshape(
            inputs, [-1] + [1] * len(self.resolution) + [inputs.shape[-1]]
        )  # (B*O, 1, ..., E)
        grid = tf.tile(objects, [1] + self.resolution + [1])  # (B*O, D0, D1, ..., E)
        return grid

    def get_config(self):
        """Serialisable configuration dictionary."""
        config = super().get_config()
        config.update({"resolution": self.resolution})
        return config


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


class RecombineStackedImage(tf.keras.layers.Layer):
    """Recombines a broadcasted image reconstruction."""

    def __init__(self, num_channels: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = num_channels

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], **kwargs):
        """Unstack, split and recombine reconstructed image."""
        # inputs (B, S, E), (B*S, W, H, 4)
        assert len(inputs) == 2, "Recombination expects 2 inputs got {len(inputs)}."
        # We pass both inputs to reshape B*S into S
        new_shape = tf.concat([tf.shape(inputs[0])[:2], tf.shape(inputs[1])[1:]], 0)
        # (B, S, W, H, 4)
        unstacked = tf.reshape(inputs[1], new_shape)  # (B, S, W, H, 4)
        channels, masks = tf.split(
            unstacked, [self.num_channels, 1], -1
        )  # [(B, S, W, H, 3), (B, S, W, H, 1)]
        masks = tf.nn.softmax(masks, axis=1)  # (B, S, W, H, 1)
        reconstruction = tf.reduce_sum(channels * masks, axis=1)  # (B, W, H, 3)
        return {
            "combined": reconstruction,
            "recon_masks": masks,
            "reconstructions": channels,
        }


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
