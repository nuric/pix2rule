"""Sequence features extration module."""
import tensorflow as tf
import tensorflow.keras.layers as L


class SequenceFeatures(L.Layer):
    """Compute sequence features of a given input."""

    def __init__(self, num_symbols: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.num_symbols = num_symbols
        self.embedding = L.Embedding(
            1 + num_symbols,
            1 + num_symbols,
            mask_zero=True,
            weights=[tf.eye(1 + num_symbols)],
            trainable=False,
        )

    def call(self, inputs, **kwargs):
        """Perform forward pass."""
        # inputs (batch_size B, length N)
        # ---------------------------
        # Compute unary predicates
        neye = tf.eye(inputs.shape[1])  # (N, N) N cannot be None
        unary_p_pos = tf.repeat(neye[None], tf.shape(inputs)[0], axis=0)  # (B, N, N)
        unary_p_sym = self.embedding(inputs)  # (B, N, S)
        unary_ps = tf.concat([unary_p_pos, unary_p_sym], -1)  # (B, N, P1)
        # ---------------------------
        # Compute binary predicates
        binary_eq_sym = tf.matmul(
            unary_p_sym, unary_p_sym, transpose_b=True
        )  # (B, N, N)
        # Collect binary predicates
        binary_ps = tf.expand_dims(binary_eq_sym, -1)  # (B, N, N, P2)
        # Remove self relations, i.e. p(X, X) = 0 always
        # p(X,X) should be modelled as p(X) (unary) instead
        binary_ps *= 1 - neye[:, :, None]  # (B, N, N, P2)
        # ---------------------------
        return unary_ps, binary_ps

    def get_config(self):
        """Serialisable configuration dictionary."""
        config = super().get_config()
        config.update({"num_symbols": self.num_symbols})
        return config
