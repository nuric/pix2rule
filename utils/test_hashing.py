"""Test cases for the hashing library."""
import unittest

from . import hashing


class TestListHash(unittest.TestCase):
    """Test cases for the list hashing function."""

    def test_can_produce_hash_single_type(self):
        """The list hashing function can produce a string hash."""
        lhash = hashing.list_hash([1, 2, 3])
        self.assertTrue(lhash)

    def test_can_produce_hash_var_type(self):
        """List with varying types can be hashed."""
        lhash = hashing.list_hash([1, "test", 2, 3.0])
        self.assertTrue(lhash)

    def test_different_types_produce_different_hashes(self):
        """Lists with different types produce different hashes."""
        lhash1 = hashing.list_hash([1, 2, 3])
        lhash2 = hashing.list_hash(["1", 2, 3])
        self.assertNotEqual(lhash1, lhash2)

    def test_ordering_changes_hash(self):
        """Lists with different orders produce different hashes."""
        lhash1 = hashing.list_hash([1, 2, 3])
        lhash2 = hashing.list_hash([1, 3, 2])
        self.assertNotEqual(lhash1, lhash2)


class TestSetHash(unittest.TestCase):
    """Test cases for set hashing function."""

    def test_can_hash_single_type(self):
        """Can hash a set of single typed values."""
        shash = hashing.set_hash({1, 2})
        self.assertTrue(shash)

    def test_cannot_hash_var_type(self):
        """Cannot hash a set of varying typed values."""
        self.assertRaises(ValueError, hashing.set_hash, {"1", 2})

    def test_order_produces_same_hash(self):
        """Different ordered elements produce same hash."""
        shash1 = hashing.set_hash([1, 2, 3])
        shash2 = hashing.set_hash([2, 1, 3])
        self.assertEqual(shash1, shash2)


class TestDictHash(unittest.TestCase):
    """Test cases for dictionary hashing function."""

    def test_can_hash_single_type(self):
        """Can hash dictionary with single type."""
        dhash = hashing.dict_hash({"a": 1, "b": 2})
        self.assertTrue(dhash)

    def test_can_hash_var_type(self):
        """Can hash with variable type key values."""
        dhash = hashing.dict_hash({"a": "c", "b": [2, 2, 3]})
        self.assertTrue(dhash)

    def test_key_ordering_produces_same_hash(self):
        """Change in order of keys produces same hash."""
        dhash1 = hashing.dict_hash({"a": "test", "b": 2})
        dhash2 = hashing.dict_hash({"b": 2, "a": "test"})
        self.assertEqual(dhash1, dhash2)
