"""Represents ways to encode categorical sequences."""
import tensorflow as tf


class OneHotCategoricalSequenceInput(tf.keras.layers.Layer):
    """Compute sequence features of a given input."""

    def __init__(self, num_symbols: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.num_symbols = num_symbols
        self.embedding = tf.keras.layers.Embedding(
            1 + num_symbols,
            1 + num_symbols,
            mask_zero=True,
            weights=[tf.eye(1 + num_symbols)],
            trainable=False,
        )

    def call(self, inputs, **kwargs):
        """Perform forward pass."""
        # inputs (batch_size B, length N)
        batch_size = tf.shape(inputs)[0]  # B
        length = tf.shape(inputs)[1]  # N
        # ---------------------------
        # Compute unary predicates
        neye = tf.eye(length)  # (N, N) N cannot be None
        unary_p_pos = tf.repeat(neye[None], batch_size, axis=0)  # (B, N, N)
        unary_p_sym = self.embedding(inputs)  # (B, N, S)
        unary_ps = tf.concat([unary_p_pos, unary_p_sym], -1)  # (B, N, P1)
        # ---------------------------
        # Compute binary predicates
        binary_eq_sym = tf.matmul(
            unary_p_sym, unary_p_sym, transpose_b=True
        )  # (B, N, N)
        # Remove self relations, i.e. p(X, X) = 0 always
        # p(X,X) should be modelled as p(X) (unary) instead
        # Here we take every element except the diagonal, 1-eye
        idxs = tf.where(1 - tf.eye(length, dtype=tf.int8))  # (N*(N-1), 2)
        idxs = tf.reshape(idxs, [length, length - 1, 2])  # (N, N-1, 2)
        idxs = tf.repeat(idxs[None], batch_size, axis=0)  # (B, N, N-1, 2)
        binary_ps = tf.gather_nd(
            binary_eq_sym[..., None], idxs, batch_dims=1
        )  # (B, N, N-1, P2)
        # ---------------------------
        facts = {"unary": unary_ps, "binary": binary_ps}
        return facts

    def get_config(self):
        """Serialisable configuration dictionary."""
        config = super().get_config()
        config.update({"num_symbols": self.num_symbols})
        return config
