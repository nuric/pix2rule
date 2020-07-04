"""Unit tests for graph based unification."""
import os
import unittest
from typing import Tuple

import tensorflow as tf

from unification import unify

# Calm down tensorflow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def create_batch() -> Tuple[tf.Tensor, tf.Tensor]:
    """Create batch examples for testing."""
    # Construct sample input graphs
    # 1232 and 3321, with pos, sym and leftof relations
    batch_eye = tf.eye(4)
    # ---------------------------
    # Unary predications, pos and sym
    batch_syms = tf.constant([[1, 2, 3, 2], [3, 3, 2, 1]])  # (B, M)
    batch_syms = tf.gather(batch_eye, batch_syms, axis=0)  # (B, M, M)
    batch_pos = tf.repeat(batch_eye[None], 2, axis=0)  # (B, M, M)
    batch_unary = tf.concat([batch_pos, batch_syms], -1)  # (B, M, P1)
    # ---------------------------
    # Binary relations
    # leftof relation
    batch_leftof = tf.linalg.diag(  # pylint: disable=unexpected-keyword-arg
        tf.ones([2, 3]), k=1
    )  # (B, M, M)
    batch_equals = tf.matmul(batch_syms, batch_syms, transpose_b=True)  # (B, M, M)
    batch_equals *= 1 - batch_eye  # (B, M, M)
    batch_binary = tf.concat(
        [batch_leftof[..., None], batch_equals[..., None]], -1
    )  # (B, M, M, P2)
    # ---------------------------
    return batch_unary, batch_binary


batch_examples = create_batch()
num_p1 = batch_examples[0].shape[-1]  # P1
num_p2 = batch_examples[1].shape[-1]  # P2


def batch_unify(inv_unary: tf.Tensor, inv_binary: tf.Tensor) -> tf.Tensor:
    """Unify given rules with test suite examples."""
    return unify(inv_unary, inv_binary, *batch_examples)


class TestUnification(unittest.TestCase):
    """Unit test cases for graph based unification."""

    def test_blank_rule(self):
        """Rule with no conditions unify with all nodes."""
        inv_unary = tf.zeros((2, 3, num_p1))  # (I, N, P1)
        inv_binary = tf.zeros((2, 3, 3, num_p2))  # (I, N, N, P2)
        res = batch_unify(inv_unary, inv_binary)  # (B, I, N, M)
        self.assertEqual(res.shape, [2, 2, 3, 4])
        all_uni = tf.reduce_all(res == tf.ones(res.shape))  # ()
        self.assertTrue(all_uni)
