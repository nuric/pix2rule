"""Utility operations."""
import tensorflow as tf


def leftright_cumprod(tensor: tf.Tensor, axis: int = -1) -> tf.Tensor:
    """Convert sigmoid outputs to categorical distribution."""
    # tensor (..., X)
    cumprod = tf.math.cumprod(1 - tensor, axis=axis, exclusive=True)  # (..., X)
    return tensor * cumprod  # (..., X)


def reduce_probsum(
    tensor: tf.Tensor, axis: int = -1, keepdims: bool = False
) -> tf.Tensor:
    """Reduce given axis using probabilistic sum."""
    # tensor (..., X, ...)
    return 1 - tf.reduce_prod(
        1 - tensor, axis=axis, keepdims=keepdims
    )  # (..., 1?, ...)
