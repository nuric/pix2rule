"""Sequence features extration module."""
import tensorflow as tf
import tensorflow.keras.layers as L

from configlib import config as C


class SequenceFeatures(L.Layer):
    """Compute sequence features of a given input."""

    def __init__(self, **kwargs):
        super(SequenceFeatures, self).__init__(**kwargs)
        # self.embedding = L.Embedding(C["seq_symbols"], C["embed"], mask_zero=True)
        self.embedding = L.Embedding(
            1 + C["seq_symbols"],
            1 + C["seq_symbols"],
            mask_zero=True,
            weights=[tf.eye(1 + C["seq_symbols"])],
            trainable=False,
        )

    def call(self, inputs, **kwargs):
        """Perform forward pass."""
        # inputs (batch_size B, length N)
        # ---------------------------
        # Compute unary predicates
        neye = tf.eye(1 + C["seq_length"])  # (N, N)
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
