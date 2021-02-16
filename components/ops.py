"""Utility operations."""
from typing import List, Union
import tensorflow as tf


def leftright_cumprod(tensor: tf.Tensor, axis: int = -1) -> tf.Tensor:
    """Convert sigmoid outputs to categorical distribution."""
    # tensor (..., X)
    cumprod = tf.math.cumprod(1 - tensor, axis=axis, exclusive=True)  # (..., X)
    return tensor * cumprod  # (..., X)


def reduce_probsum(
    tensor: tf.Tensor, axis: Union[int, List[int]] = -1, keepdims: bool = False
) -> tf.Tensor:
    """Reduce given axis using probabilistic sum."""
    # tensor (..., X, ...)
    return 1 - tf.reduce_prod(
        1 - tensor, axis=axis, keepdims=keepdims
    )  # (..., 1?, ...)


def scaled_softmax(
    tensor: tf.Tensor, axis: int = -1, alpha: float = 0.999
) -> tf.Tensor:
    """Compute scaled softmax which will output alpha as highest."""
    # tensor (..., X, ...)
    size = tf.shape(tensor)[axis]  # X
    # alpha is softmax target scale on one-hot vector
    sm_scale = tf.math.log(
        alpha * tf.cast(size - 1, tf.float32) / (1 - alpha)
    )  # k = log(alpha * (n-1) / (1-alpha)) derived from softmax(kx)
    sm_scale = tf.cond(size == 1, lambda: 1.0, lambda: sm_scale)
    return tf.nn.softmax(tensor * sm_scale, axis=axis)
