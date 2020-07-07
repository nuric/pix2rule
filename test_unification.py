"""Unit tests for graph based unification."""
import os
import re
import unittest
from typing import Tuple, List, Dict

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


def get_rule_tensors(
    num_rules: int, num_nodes: int, coords: List[str] = None
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Create rule tensors based on given sparse coordinate patterns:
    i#n#pos# - i#n#sym# - i#n#left# - i#n#same#
    """
    # ---------------------------
    # Extract coordinates
    indices: Dict[str, List[List[int]]] = {"unary": list(), "binary": list()}
    for coord in coords or list():
        idxs = [int(x) for x in re.findall(r"\d+", coord)]  # coord numbers
        prefix = re.findall(r"\D+", coord)[-1]  # coord prefixes
        if prefix == "pos":
            indices["unary"].append(idxs)
        elif prefix == "sym":
            idxs[2] += 4  # offset pos predicate for sym
            indices["unary"].append(idxs)
        elif prefix == "left":
            idxs.append(0)  # binary predicate number
            indices["binary"].append(idxs)
        elif prefix == "same":
            idxs.append(1)  # binary predicate number
            indices["binary"].append(idxs)
            idxs = idxs.copy()
            # same is symmetrical
            idxs[1], idxs[2] = idxs[2], idxs[1]
            indices["binary"].append(idxs)
    # ---------------------------
    # Create tensors
    inv_unary = tf.zeros((num_rules, num_nodes, num_p1))
    if indices["unary"]:
        unary_inds = tf.constant(indices["unary"], dtype=tf.int64)  # (X, 3)
        inv_unary = tf.sparse.SparseTensor(
            unary_inds, tf.ones(len(indices["unary"])), (num_rules, num_nodes, num_p1)
        )
        inv_unary = tf.sparse.to_dense(inv_unary)
    inv_binary = tf.zeros((num_rules, num_nodes, num_nodes, num_p2))
    if indices["binary"]:
        binary_inds = tf.constant(indices["binary"], dtype=tf.int64)  # (X,
        inv_binary = tf.sparse.SparseTensor(
            binary_inds,
            tf.ones(len(indices["binary"])),
            (num_rules, num_nodes, num_nodes, num_p2),
        )
        inv_binary = tf.sparse.to_dense(inv_binary)
    # ---------------------------
    return inv_unary, inv_binary


class TestUnification(unittest.TestCase):
    """Unit test cases for graph based unification."""

    def assert_all(self, got: tf.Tensor, expected: tf.Tensor):
        """Assert all elements of tensors are equal."""
        self.assertTrue(tf.reduce_all(got == expected))

    def test_blank_rule(self):
        """Rule with no conditions unify with all nodes."""
        res = batch_unify(*get_rule_tensors(2, 3))  # (B, I, N, M)
        self.assertEqual(res.shape, [2, 2, 3, 4])
        self.assert_all(res, tf.ones(res.shape))

    def test_single_unary_rule(self):
        """Rule with a single node and 1 unary condition."""
        rule = get_rule_tensors(1, 1, ["i0n0pos1"])
        res = batch_unify(*rule)  # (B, I, N, M)
        # Only 1 node can unify
        expect = tf.constant([0, 1.0, 0, 0])
        self.assert_all(res, expect)

    def test_single_unary_double_binding(self):
        """Single unary condition with two possible assignments."""
        rule = get_rule_tensors(1, 1, ["i0n0sym2"])
        res = batch_unify(*rule)  # (B, I, N, M)
        self.assert_all(res[0, 0, 0], tf.constant([0, 1.0, 0, 1.0]))
        self.assert_all(res[1, 0, 0], tf.constant([0, 0.0, 1.0, 0]))
