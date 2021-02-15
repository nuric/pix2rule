"""Gaussian mixture based initialiser."""
from typing import List, Tuple

import tensorflow as tf
import tensorflow_probability as tfp


class CategoricalRandomNormal(tf.keras.initializers.Initializer):
    """Categorical mixture model based weight initialiser."""

    def __init__(self, probs: List[float], mean: float = 0.0, stddev: float = 1.0):
        self.probs = probs
        self.mean = mean
        self.stddev = stddev
        self.categorical = tfp.distributions.OneHotCategorical(probs=probs)

    def __call__(self, shape: Tuple[int, ...], dtype: tf.DType = None, **kwargs):
        # shape (..., C)
        assert shape[-1] == len(
            self.probs
        ), f"Expected {self.probs} categories, got {shape} as input."
        dtype = dtype if dtype is not None else tf.float32
        weights = tf.random.normal(
            shape[:-1], mean=self.mean, stddev=self.stddev, dtype=dtype
        )  # (...,)
        weights = tf.expand_dims(weights, -1)  # (..., 1)
        sample = self.categorical.sample(shape[:-1])  # (..., C)
        # Selected weights gets positive, others get negative
        weights = weights * tf.cast(sample * 2 - 1, tf.float32)  # (..., C)
        return weights

    def get_config(self):  # To support serialization
        return {"probs": self.probs, "mean": self.mean, "stddev": self.stddev}


class BernoulliRandomNormal(CategoricalRandomNormal):
    """Bernoulli mixture model based weight initialiser."""

    def __init__(self, prob: float, mean: float = 0.0, stddev: float = 1.0):
        super().__init__([prob, 1 - prob], mean=mean, stddev=stddev)
        self.prob = prob

    def __call__(self, shape: Tuple[int, ...], dtype: tf.DType = None, **kwargs):
        # shape (...,)
        shape = shape + (2,)  # (..., 2)
        weights = super().__call__(shape, dtype=dtype, **kwargs)  # (..., 2)
        return weights[..., 0]  # (...,)

    def get_config(self):  # To support serialization
        config = super().get_config()
        del config["probs"]
        config["prob"] = self.prob
        return config