"""Test suites for utility layers."""
import tensorflow as tf

from . import util_layers


class TestSpacialFlatten(tf.test.TestCase):
    """Unit test cases for spacial flatten layer."""

    def test_already_flat_input(self):
        """Does not affect already flat input."""
        tensor = tf.random.normal((4, 2, 4))
        res = util_layers.SpacialFlatten()(tensor)
        self.assertEqual(res.shape, [4, 2, 4])

    def test_spacial_flatten(self):
        """Flattens spacial dimensions."""
        tensor = tf.random.normal((4, 4, 2, 4))
        res = util_layers.SpacialFlatten()(tensor)
        self.assertEqual(res.shape, [4, 8, 4])


class TestSpacialBroadcast(tf.test.TestCase):
    """Test cases for spacial broadcasting."""

    def test_2d_broadcasting(self):
        """Successfully broadcasts into a 2d grid."""
        tensor = tf.random.normal((4, 2, 4))
        res = util_layers.SpacialBroadcast([8, 7])(tensor)
        self.assertEqual(res.shape, [8, 8, 7, 4])

    def test_3d_broadcasting(self):
        """Broadcasts objects into a 3d grid."""
        tensor = tf.random.normal((4, 2, 4))
        res = util_layers.SpacialBroadcast([7, 9, 8])(tensor)
        self.assertEqual(res.shape, [8, 7, 9, 8, 4])


class TestMergeFacts(tf.test.TestCase):
    """Unit test cases merging dictionary based facts."""

    def test_single_tensor_passthrough(self):
        """If there is one tensor, it returns that tensor."""
        facts1 = {"binary": tf.random.normal((4, 2))}
        res = util_layers.MergeFacts()([facts1])
        self.assertEqual(res["binary"].shape, [4, 2])

    def test_missing_nullary_key(self):
        """Adds empty tensor for missing nullary key."""
        facts1 = {"binary": tf.random.normal((4, 2))}
        res = util_layers.MergeFacts()([facts1])
        self.assertIn("nullary", res)
        self.assertEqual(res["nullary"].shape, (4, 0))

    def test_matching_keys(self):
        """If both keys exist, they get concatenated."""
        facts1 = {"binary": tf.random.normal((4, 2))}
        facts2 = {"binary": tf.random.normal((4, 2))}
        res = util_layers.MergeFacts()([facts1, facts2])
        self.assertEqual(res["binary"].shape, [4, 4])

    def test_missing_keys(self):
        """Missing keys get merged into a single dictionary."""
        facts1 = {"nullary": tf.random.normal((4, 2))}
        facts2 = {"unary": tf.random.normal((4, 3))}
        res = util_layers.MergeFacts()([facts1, facts2])
        self.assertEqual(res["nullary"].shape, [4, 2])
        self.assertEqual(res["unary"].shape, [4, 3])


class TestShuffle(tf.test.TestCase):
    """Test unit cases for the shuffle layer."""

    def test_shuffle_batch_axis(self):
        """It can shuffle the batch axis."""
        inputs = tf.constant([[1, 2], [3, 4]])  # (2, 2)
        tf.random.set_seed(42)
        res = util_layers.Shuffle(shuffle_axis=0, seed=42)(inputs)
        expected = tf.constant([[3, 4], [1, 2]])
        self.assertAllEqual(res, expected)

    def test_shuffle_first_axis(self):
        """It can shuffle the first axis."""
        inputs = tf.constant([[1, 2, 3], [4, 5, 6]])  # (2, 3)
        tf.random.set_seed(42)
        res = util_layers.Shuffle(shuffle_axis=1, seed=41)(inputs)
        expected = tf.constant([[2, 3, 1], [5, 6, 4]])
        self.assertAllEqual(res, expected)
