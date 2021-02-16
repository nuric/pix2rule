"""Test categorical sequence inputs."""
import tensorflow as tf

from . import categorical_sequence


class TestOneHotCategoricalSequenceInput(tf.test.TestCase):
    """Test categorical sequence input encoding."""

    def test_correct_facts_shape(self):
        """Returns unary and binary predicates."""
        sequence = tf.constant([[4, 2, 4], [4, 2, 0]], dtype=tf.int32)
        facts = categorical_sequence.OneHotCategoricalSequenceInput(num_symbols=4)(
            sequence
        )
        self.assertEqual(facts["unary"].shape[:2], [2, 3])
        self.assertEqual(facts["binary"].shape[:3], [2, 3, 2])
