"""Test cases for merging facts."""
import tensorflow as tf
from . import merge_facts


class TestMergeFacts(tf.test.TestCase):
    """Unit test cases merging dictionary based facts."""

    def test_matching_keys(self):
        """If both keys exist, they get concatenated."""
        facts1 = {"binary": tf.random.normal((4, 2))}
        facts2 = {"binary": tf.random.normal((4, 2))}
        res = merge_facts.MergeFacts()([facts1, facts2])
        self.assertEqual(res["binary"].shape, [4, 4])

    def test_missing_keys(self):
        """Missing keys get merged into a single dictionary."""
        facts1 = {"nullary": tf.random.normal((4, 2))}
        facts2 = {"unary": tf.random.normal((4, 3))}
        res = merge_facts.MergeFacts()([facts1, facts2])
        self.assertEqual(res["nullary"].shape, [4, 2])
        self.assertEqual(res["unary"].shape, [4, 3])
