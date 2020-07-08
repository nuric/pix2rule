"""Utility operations test suites."""
import tensorflow as tf
from . import ops


class TestOps(tf.test.TestCase):
    """Unit test cases for utility operations."""

    def test_leftright_cumprod_dist(self):
        """Check if sigmoid summation is converted correctly."""
        tensor = tf.constant([0.5, 1.0, 0.5])
        res = ops.leftright_cumprod(tensor)
        self.assertEqual(res.shape, tensor.shape)
        self.assertAllEqual(res, [0.5, 0.5, 0.0])

    def test_leftright_cumprod_sums_to_one(self):
        """Sigmoid convertion produces a valid distribution."""
        tensor = tf.random.uniform([4, 2, 5])
        tensor = tf.concat([tensor, tf.ones([4, 2, 1])], -1)  # (4, 2, 6)
        res = ops.leftright_cumprod(tensor)
        self.assertEqual(res.shape, tensor.shape)
        sums = tf.reduce_sum(res, -1)  # (4, 2)
        self.assertAllClose(tf.ones(sums.shape), sums)
