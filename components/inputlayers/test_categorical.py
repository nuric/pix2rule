"""Unit tests for object selection layers."""
import tensorflow as tf

from . import categorical


class TestOneHotCategoricalInput(tf.test.TestCase):
    """Test categorical input encoding."""

    def test_categorical_encoding(self):
        """The input is one hot encoded."""
        tensor = tf.range(5)
        facts = categorical.OneHotCategoricalInput(5)(tensor)
        encoded = facts["nullary"]
        self.assertEqual(encoded.shape, [5, 5])
        self.assertAllEqual(tf.reduce_sum(encoded, -1), tf.ones_like(tensor))
