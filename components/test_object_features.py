"""Test object feature layers."""
import tensorflow as tf

from . import object_features


class TestLinearObjectFeatures(tf.test.TestCase):
    """Test cases for linear object features."""

    def test_correct_facts_shape(self):
        """Returns unary and binary predicates."""
        # Batch of 2 objects with 4 features
        objects = tf.random.normal((4, 3, 4))
        facts = object_features.LinearObjectFeatures()(objects)
        self.assertEqual(facts["unary"].shape[:2], [4, 3])
        self.assertEqual(facts["binary"].shape[:3], [4, 3, 2])
