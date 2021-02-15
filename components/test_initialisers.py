"""Test cases for categorical weight initialisers."""
import numpy as np
import tensorflow as tf
from . import initialisers


class TestCategoricalRandomNormal(tf.test.TestCase):
    """Unit test cases for categorical random normal weight initialiser."""

    def test_returns_the_correct_shape(self):
        """Returns the correct number of categorical classes."""
        shape = (4, 2, 4)
        probs = tf.random.normal(shape[-1:])  # (4,)
        probs = tf.nn.softmax(probs)
        init = initialisers.CategoricalRandomNormal(probs=probs)
        weights = init(shape)
        self.assertAllEqual(weights.shape, shape)

    def test_categorical_sampling(self):
        """Weights are assigned correctly to the sampling process."""
        shape = (4, 2, 4)
        ridx = np.random.randint(shape[-1])
        probs = np.eye(shape[-1])[ridx]  # (4,)
        init = initialisers.CategoricalRandomNormal(probs=probs, mean=10.0)
        weights = init(shape)
        match = tf.argmax(weights, -1) == ridx
        self.assertTrue(tf.reduce_all(match))


class TestBernoulliRandomNormal(tf.test.TestCase):
    """Unit test cases for bernoulli random normal weight initialiser."""

    def test_returns_the_correct_shape(self):
        """Correct number of weights are returned."""
        shape = (4, 2)
        init = initialisers.BernoulliRandomNormal(prob=0.42)
        weights = init(shape)
        self.assertAllEqual(weights.shape, shape)
