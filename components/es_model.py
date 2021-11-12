"""Evolution strategy based training model wrapper."""
from typing import List
import tensorflow as tf

# ---------------------------
# Adapted from
# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
def compute_ranks(flat_tensor: tf.Tensor) -> tf.Tensor:
    """Returns ranks in [0, len(x))"""
    # flat_tensor (X,)
    assert (
        len(flat_tensor.shape) == 1
    ), f"Expected flat tensor for ranking, got {flat_tensor.shape}"
    sorted_idxs = tf.argsort(flat_tensor)  # (X,)
    ranks = tf.scatter_nd(
        sorted_idxs[:, None], tf.range(tf.size(flat_tensor)), tf.shape(flat_tensor)
    )
    return ranks


def compute_centered_ranks(tensor: tf.Tensor) -> tf.Tensor:
    """Compute ranks centered around 0."""
    # tensor (...)
    flat_tensor = tf.reshape(tensor, [tf.size(tensor)])  # (X,)
    ranks = tf.cast(compute_ranks(flat_tensor), tf.float32)  # (X,)
    ranks = tf.reshape(ranks, tensor.shape)  # (...)
    ranks = ranks / (tf.cast(tf.size(tensor), tf.float32) - 1) - 0.5
    return ranks


# ---------------------------


class ESModel(tf.keras.Model):  # pylint: disable=abstract-method
    """Natural Evolution Strategy based training model."""

    def __init__(self, pop_size: int = 40, noise_stddev: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.pop_size = pop_size
        self.noise_stddev = noise_stddev

    @property
    def flat_trainable_weights(self) -> tf.Tensor:
        """Flattened trainable weights."""
        return tf.concat(
            [tf.reshape(w, [tf.size(w)]) for w in self.trainable_weights], 0
        )  # (X,)

    def expand_flat_trainable_weights(self, flat_weights: tf.Tensor) -> List[tf.Tensor]:
        """Expand flattened trainable weights."""
        # We have a list of tf.Variable here, we will slice through
        # the flat weights and get the values
        acc_size = tf.constant(0)
        expanded_weights: List[tf.Tensor] = list()
        for weight in self.trainable_weights:
            sliced_weight = flat_weights[acc_size : acc_size + tf.size(weight)]
            sliced_weight = tf.reshape(sliced_weight, weight.shape)
            expanded_weights.append(sliced_weight)
            acc_size += tf.size(weight)
        return expanded_weights

    def set_flat_trainable_weights(self, flat_weights: tf.Tensor):
        """Set trainable weights from flattened."""
        # flat_weights (X,)
        new_weights = self.expand_flat_trainable_weights(flat_weights)  # [(...), (...)]
        for weight, new_value in zip(self.trainable_weights, new_weights):
            weight.assign(new_value, read_value=False)

    def train_step(self, data):
        """Overwrite gradient based training step."""
        # data is a single batch
        inputs, expected = data
        # ---------------------------
        predicted = self(inputs, training=True)  # Forward pass
        loss = self.compiled_loss(
            expected, predicted, regularization_losses=self.losses
        )
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(expected, predicted)
        # ---------------------------
        # Generate noise for weights
        flat_weights = self.flat_trainable_weights  # (X,)
        noise = tf.random.normal(
            [self.pop_size, flat_weights.shape[0]], stddev=self.noise_stddev
        )  # (N, X)
        # ---------------------------
        # Evaluate points to gather losses
        losses = list()
        # We evaluate both +noise and -noise, hence pop_size*2
        for i in range(self.pop_size * 2):
            parity = ((i + 1) % 2) * 2 - 1  # 1, -1, 1, -1, etc
            self.set_flat_trainable_weights(flat_weights + noise[i // 2] * parity)
            predicted = self(inputs, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(
                expected, predicted, regularization_losses=self.losses
            )
            losses.append(loss)
        # ---------------------------
        # Compute gradients
        # stacked_losses = losses.stack()  # (N*2,)
        losses = tf.stack(losses)  # (N*2)
        losses = tf.reshape(losses, [self.pop_size, 2])  # (N, 2)
        proc_losses = compute_centered_ranks(losses)  # (N, 2)
        # Mirrored sampling process
        proc_losses = proc_losses[:, 0] - proc_losses[:, 1]  # (N,)
        gradients = tf.tensordot(noise, proc_losses, [[0], [0]])  # (X,)
        gradients = gradients / (self.pop_size * self.noise_stddev)  # (X,)
        gradients = self.expand_flat_trainable_weights(gradients)  # [(...), (...), ...]
        # ---------------------------
        # Restore original weights and update them
        self.set_flat_trainable_weights(flat_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        # ---------------------------
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        """Configuration dictionary for serialisation."""
        config = super().get_config()
        config.update({"pop_size": self.pop_size, "noise_stddev": self.noise_stddev})
        return config
