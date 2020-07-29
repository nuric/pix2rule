"""Unit tests for sequence features."""
import logging

import tensorflow as tf

from .sequence_features import SequenceFeatures


class TestSequenceFeatures(tf.test.TestCase):
    """Unit test cases sequence feature layer."""

    def test_variable_sequence_length4(self):
        """Sequences of length 4 return correct shapes."""
        seq = tf.constant([[1, 2, 3, 2], [3, 3, 2, 1]])  # (B, N)
        res_unary, res_binary = SequenceFeatures()(seq)  # (B, N, P1), (B, N, N, P2)
        self.assertEqual(res_unary.shape, [2, 4, 13])
        self.assertEqual(res_binary.shape, [2, 4, 4, 1])
        self.assertTrue(tf.reduce_all(tf.reduce_any(res_unary == 1.0, -1)))

    def test_variable_sequence_length3(self):
        """Sequences of length 3 return correct shapes."""
        seq = tf.constant([[3, 2, 1]])  # (B, N)
        res_unary, res_binary = SequenceFeatures()(seq)  # (B, N, P1), (B, N, N, P2)
        self.assertEqual(res_unary.shape, [1, 3, 12])
        self.assertEqual(res_binary.shape, [1, 3, 3, 1])
        self.assertTrue(tf.reduce_all(tf.reduce_any(res_unary == 1.0, -1)))

    def test_variable_symbol_numbers3(self):
        """Sequences with total number of 3 symbols return correct shapes."""
        seq = tf.constant([[1, 2], [2, 1]])
        res_unary, res_binary = SequenceFeatures(num_symbols=2)(
            seq
        )  # (B, N, P1), (B, N, N, P2)
        self.assertEqual(res_unary.shape, [2, 2, 5])
        self.assertEqual(res_binary.shape, [2, 2, 2, 1])
        self.assertTrue(tf.reduce_all(tf.reduce_any(res_unary == 1.0, -1)))
