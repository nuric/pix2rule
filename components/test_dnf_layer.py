"""Test cases for DNF layer."""
from typing import Dict
import tensorflow as tf

from . import dnf_layer


def gen_random_facts(num_objects: int = 2) -> Dict[str, tf.Tensor]:
    """Generate batch of random facts with given number of objects."""
    return {
        "nullary": tf.random.uniform((4, 2), dtype=tf.float32),
        "unary": tf.random.uniform((4, num_objects, 4), dtype=tf.float32),
        "binary": tf.random.uniform(
            (4, num_objects, num_objects - 1, 3), dtype=tf.float32
        ),
    }


class TestDNFLayer(tf.test.TestCase):
    """Test cases for disjunctive normal form rule learning layer."""

    def test_different_arities(self):
        """Returns correct number of rules."""
        facts = gen_random_facts()
        facts = dnf_layer.DNF(arities=[0, 1, 1, 2, 2, 2])(facts)
        self.assertEqual(facts["nullary"].shape, [4, 1])
        self.assertEqual(facts["unary"].shape, [4, 2, 2])
        self.assertEqual(facts["binary"].shape, [4, 2, 1, 3])

    def test_recursive_merge(self):
        """When run recursively, it merges the learnt rules."""
        facts = gen_random_facts()
        dnf = dnf_layer.DNF(arities=[0, 1, 1, 2, 2, 2], recursive=True)
        facts = dnf.pad_inputs(facts)
        facts = dnf(facts)
        self.assertEqual(facts["nullary"].shape, [4, 3])
        self.assertEqual(facts["unary"].shape, [4, 2, 6])
        self.assertEqual(facts["binary"].shape, [4, 2, 1, 6])

    def test_different_num_variables(self):
        """Works with different number of variables."""
        facts = gen_random_facts(4)
        facts = dnf_layer.DNF(arities=[2], num_total_variables=3)(facts)
        self.assertEqual(facts["binary"].shape, [4, 4, 3, 1])
