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


def soft_maximum(
    tensor: tf.Tensor, axis: int = -1, temperature: float = 1.0, keepdims: bool = False
) -> tf.Tensor:
    """Compute the soft maximum of a given tensor along an axis."""
    # tensor (..., X, ...)
    return tf.reduce_sum(
        tf.nn.softmax(tensor / temperature, axis) * tensor, axis, keepdims=keepdims
    )


def soft_minimum(
    tensor: tf.Tensor, axis: int = -1, temperature: float = 1.0, keepdims: bool = False
) -> tf.Tensor:
    """Compute the soft minimum of a given tensor along an axis."""
    # tensor (..., X, ...)
    return -soft_maximum(-tensor, axis=axis, temperature=temperature, keepdims=keepdims)


def flatten_concat(tensors: List[tf.Tensor], batch_dims: int = 1) -> tf.Tensor:
    """Flatten given inputs and concatenate them."""
    # tensors [(B, ...), (B, ...)]
    flattened: List[tf.Tensor] = list()  # [(B, X), (B, Y) ...]
    for tensor in tensors:
        final_dim = -1
        if all(i is not None for i in tensor.shape[batch_dims:]):
            # We know all the dimensions
            final_dim = tf.reduce_prod(tensor.shape[batch_dims:])
        flat_tensor = tf.reshape(
            tensor, tf.concat([tf.shape(tensor)[:batch_dims], [final_dim]], 0)
        )
        flattened.append(flat_tensor)
    return tf.concat(flattened, -1)  # (B, X+Y+...)


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
