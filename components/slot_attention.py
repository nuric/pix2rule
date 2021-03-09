"""Slot Attention model for object discovery and set prediction.

coding=utf-8
Copyright 2020 The Google Research Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

From the paper Object-Centric Learning with Slot Attention
URL: https://papers.nips.cc/paper/2020/file/8511df98c02ab60aea1b2356c013bc0f-Paper.pdf

---

Modified by nuric to adapt to project needs, linting and styling.
"""
from typing import List
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L


class SlotAttention(L.Layer):  # pylint: disable=too-many-instance-attributes
    """Slot Attention module."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        num_iterations: int,
        num_slots: int,
        slot_size: int,
        mlp_hidden_size: int,
        epsilon: float = 1e-8,
    ):
        """Builds the Slot Attention module.

        Args:
          num_iterations: Number of iterations.
          num_slots: Number of slots.
          slot_size: Dimensionality of slot feature vectors.
          mlp_hidden_size: Hidden layer size of MLP.
          epsilon: Offset for attention coefficients before normalization.
        """
        super().__init__()
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        self.norm_inputs = L.LayerNormalization()
        self.norm_slots = L.LayerNormalization()
        self.norm_mlp = L.LayerNormalization()

        # Parameters for Gaussian init (shared by all slots).
        self.slots_mu = self.add_weight(
            initializer="glorot_uniform",
            shape=[1, 1, self.slot_size],
            dtype=tf.float32,
            name="slots_mu",
        )
        self.slots_log_sigma = self.add_weight(
            initializer="glorot_uniform",
            shape=[1, 1, self.slot_size],
            dtype=tf.float32,
            name="slots_log_sigma",
        )

        # Linear maps for the attention module.
        self.project_q = L.Dense(self.slot_size, use_bias=False, name="q")
        self.project_k = L.Dense(self.slot_size, use_bias=False, name="k")
        self.project_v = L.Dense(self.slot_size, use_bias=False, name="v")

        # Slot update functions.
        self.gru = L.GRUCell(self.slot_size)
        self.mlp = tf.keras.Sequential(
            [
                L.Dense(self.mlp_hidden_size, activation="relu"),
                L.Dense(self.slot_size),
            ],
            name="mlp",
        )

    def call(self, inputs: tf.Tensor, **kwargs):
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        keys = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        values = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        slots = self.slots_mu + tf.exp(self.slots_log_sigma) * tf.random.normal(
            [tf.shape(inputs)[0], self.num_slots, self.slot_size]
        )

        # Multiple rounds of attention.
        attns = list()
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            queries = self.project_q(
                slots
            )  # Shape: [batch_size, num_slots, slot_size].
            queries *= self.slot_size ** -0.5  # Normalization.
            attn_logits = tf.keras.backend.batch_dot(keys, queries, axes=-1)
            attn = tf.nn.softmax(attn_logits, axis=-1)
            # `attn` has shape: [batch_size, num_inputs, num_slots].
            attns.append(attn)

            # Weigted mean.
            attn += self.epsilon
            attn /= tf.reduce_sum(attn, axis=-2, keepdims=True)
            updates = tf.keras.backend.batch_dot(attn, values, axes=-2)
            # `updates` has shape: [batch_size, num_slots, slot_size].

            # Slot update.
            slots, _ = self.gru(updates, [slots_prev])
            slots += self.mlp(self.norm_mlp(slots))

        attns = tf.stack(attns, axis=1)  # (B, iterations, num_inputs, num_slots)
        return {"slots": slots, "slot_attention": attns}

    def get_config(self):
        """Serialisable configuration dictionary."""
        config = super().get_config()
        config.update(
            {
                "num_iterations": self.num_iterations,
                "num_slots": self.num_slots,
                "slot_size": self.slot_size,
                "mlp_hidden_size": self.mlp_hidden_size,
                "epsilon": self.epsilon,
            }
        )
        return config


class SoftPositionEmbed(L.Layer):
    """Adds soft positional embedding with learnable projection."""

    def __init__(self, hidden_size: int, resolution: List[int]):
        """Builds the soft position embedding layer.

        Args:
          hidden_size: Size of input feature dimension.
          resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.resolution = resolution
        self.dense = L.Dense(hidden_size, use_bias=True)
        self.grid = self.build_grid(resolution)

    @staticmethod
    def build_grid(resolution: List[int]):
        """Build 2d grid for dense embedding of position in images."""
        ranges = [np.linspace(0.0, 1.0, num=res) for res in resolution]
        grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [resolution[0], resolution[1], -1])
        grid = np.expand_dims(grid, axis=0)
        grid = grid.astype(np.float32)
        return np.concatenate([grid, 1.0 - grid], axis=-1)

    def call(self, inputs: tf.Tensor, **kwargs):
        return inputs + self.dense(self.grid)

    def get_config(self):
        """Serialisable configuration dictionary."""
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "resolution": self.resolution,
            }
        )
        return config
