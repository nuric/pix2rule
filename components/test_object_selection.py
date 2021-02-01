"""Unit tests for object selection layers."""
import inspect
import tensorflow as tf

from . import object_selection

# Collect all selector layers defined in object selection
all_selectors = {
    name: layer
    for name, layer in inspect.getmembers(object_selection)
    if inspect.isclass(layer) and name.endswith("ObjectSelection")
}


def attention_entropy(attention: tf.Tensor, axis: int = -1) -> tf.Tensor:
    """Computes entropy over given dimension."""
    # attention (..., A)
    return -1 * tf.reduce_sum(
        attention * tf.math.log(attention + 0.001), axis=axis
    )  # (...)


class TestCommonObjectSelection(tf.test.TestCase):
    """Test common properties of object selection in one go."""

    def test_number_of_objects_selected(self):
        """The object selection layer returns the specified number of objects."""
        for name, selector in all_selectors.items():
            with self.subTest(name):
                objects = tf.random.uniform(
                    (7, 5, 3), dtype=tf.float32
                )  # Batch of 5 objects
                selected = selector(num_select=2)(objects)
                self.assertEqual(selected["objects"].shape, [7, 2, 3])
                selected = selector(num_select=4)(objects)
                self.assertEqual(selected["objects"].shape, [7, 4, 3])


class TestRelaxedObjectSelection(tf.test.TestCase):
    """Unit test cases for relaxed object selection."""

    def test_temperature_sharpening(self):
        """As the temperature decreases, the attentions sharpen."""
        objects = tf.random.uniform((7, 5, 3), dtype=tf.float32)  # Batch of 5 objects
        selected = object_selection.RelaxedObjectSelection(
            initial_temperature=0.5, num_select=2
        )(objects)
        atts = selected["object_atts"]  # (B, S, O)
        high_entropy = tf.reduce_mean(attention_entropy(atts))  # (B, S)
        selected = object_selection.RelaxedObjectSelection(
            initial_temperature=0.2, num_select=2
        )(objects)
        atts = selected["object_atts"]  # (B, S, O)
        low_entropy = tf.reduce_mean(attention_entropy(atts))  # (B, S)
        self.assertGreater(high_entropy, low_entropy)