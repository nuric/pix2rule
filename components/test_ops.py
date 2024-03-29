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
        tensor = tf.random.uniform([4, 2, 5], dtype=tf.float32)
        tensor = tf.concat([tensor, tf.ones([4, 2, 1])], -1)  # (4, 2, 6)
        res = ops.leftright_cumprod(tensor)
        self.assertEqual(res.shape, tensor.shape)
        sums = tf.reduce_sum(res, -1)  # (4, 2)
        self.assertAllClose(tf.ones(sums.shape), sums)

    def test_reduce_probsum_vector(self):
        """Reduce probsum of a single vector return logical or."""
        tensor = tf.constant([0.5, 0.5])
        res = ops.reduce_probsum(tensor)
        self.assertEqual(res, 0.75)

    def test_reduce_probsum_uniform(self):
        """Probsum of uniform distribution with 1 appended is always 1."""
        tensor = tf.random.uniform([4, 2, 5], dtype=tf.float32)
        tensor = tf.concat([tensor, tf.ones([4, 2, 1])], -1)  # (4, 2, 6)
        res = ops.reduce_probsum(tensor)
        self.assertEqual(res.shape, [4, 2])
        self.assertAllClose(tf.ones(res.shape), res)

    def test_soft_maximum_single_dim(self):
        """Soft maximum returns the maximum when low enough temperature is given."""
        tensor = tf.random.normal([4, 2], stddev=10)
        res = ops.soft_maximum(tensor, temperature=0.01)
        self.assertAllClose(res, tf.reduce_max(tensor, -1))

    def test_soft_minimum_single_dim(self):
        """Soft maximum returns the maximum when low enough temperature is given."""
        tensor = tf.random.normal([4, 2], stddev=10)
        res = ops.soft_minimum(tensor, temperature=0.01)
        self.assertAllClose(res, tf.reduce_min(tensor, -1))

    def test_scaled_softmax(self):
        """Scaled softmax returns a sharpened version of softmax."""
        tensor = tf.constant([0.2, 0.9])
        output = ops.scaled_softmax(tensor)
        self.assertAllClose(tf.math.round(output), [0.0, 1.00])

    def test_scaled_softmax_single_class(self):
        """Scaled softmax returns 1 when there is only one input."""
        tensor = tf.random.normal((4, 1, 2))
        output = ops.scaled_softmax(tensor, axis=1)
        self.assertAllEqual(output, tf.ones_like(output))

    def test_flattened_concat_single_batch_dim(self):
        """Flattens given tensors with a single batch dim."""
        tensor1 = tf.random.normal((4, 2, 4))
        tensor2 = tf.random.normal((4, 3, 2))
        output = ops.flatten_concat([tensor1, tensor2], batch_dims=1)
        self.assertEqual(output.shape, [4, 14])

    def test_flattened_concat_multiple_batch_dim(self):
        """Flattens given tensors with a single batch dim."""
        tensor1 = tf.random.normal((4, 3, 4))
        tensor2 = tf.random.normal((4, 3, 2))
        output = ops.flatten_concat([tensor1, tensor2], batch_dims=2)
        self.assertEqual(output.shape, [4, 3, 6])
