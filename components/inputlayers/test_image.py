"""Unit tests image input layers."""
import inspect
import tensorflow as tf

from . import image

# Collect all selector layers defined in object selection
all_image_layers = {
    name: layer
    for name, layer in inspect.getmembers(image)
    if inspect.isclass(layer) and name.endswith("ImageInput")
}


class TestCommonObjectSelection(tf.test.TestCase):
    """Test common properties of object selection in one go."""

    def test_correct_input_shape(self):
        """All image input layers process an image and give back an image."""
        for name, image_layer in all_image_layers.items():
            with self.subTest(name):
                input_images = tf.random.uniform(
                    (7, 5, 3, 3), dtype=tf.float32
                )  # Batch of 7 images
                processed_images = image_layer()(input_images)
                self.assertEqual(len(processed_images.shape), 4)

    def test_hidden_size(self):
        """All image input layers return the correct hidden size output."""
        for name, image_layer in all_image_layers.items():
            with self.subTest(name):
                input_images = tf.random.uniform(
                    (7, 5, 3, 3), dtype=tf.float32
                )  # Batch of 7 images
                processed_images = image_layer(hidden_size=42)(input_images)
                self.assertEqual(processed_images.shape[0], 7)
                self.assertEqual(processed_images.shape[-1], 42)
