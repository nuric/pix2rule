"""Test cases for the hyper run configuration library."""
from typing import Dict, Any
import unittest

from . import hyperrun


class TestHyperRun(unittest.TestCase):
    """Test cases for the hyper configuration generation function."""

    def test_empty_dict_gives_empty_dict(self):
        """The configuration generator gives an empty dictionary on an empty dictionary."""
        hconfig: Dict[str, Any] = dict()
        configs = list(hyperrun.generate_configs(hconfig))
        self.assertEqual(len(configs), 1)
        self.assertFalse(configs[0])

    def test_constant_dict_gives_single_dict(self):
        """A single dictionary with constant values give a single dictionary."""
        hconfig = {"foo": "bar"}
        configs = list(hyperrun.generate_configs(hconfig))
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0], hconfig)

    def test_single_iterative_field(self):
        """A single iterative field gives back multiple configurations."""
        hconfig = {"foo": ["bar", "car"]}
        configs = list(hyperrun.generate_configs(hconfig))
        self.assertEqual(len(configs), 2)
        self.assertEqual(configs[0], {"foo": "bar"})
        self.assertEqual(configs[1], {"foo": "car"})

    def test_iterative_and_constant_together(self):
        """Constant fields are added to iterative fields."""
        hconfig = {"foo": ["bar", "car"], "joe": "bloggs"}
        configs = list(hyperrun.generate_configs(hconfig))
        self.assertEqual(len(configs), 2)
        self.assertEqual(configs[0], {"foo": "bar", "joe": "bloggs"})
        self.assertEqual(configs[1], {"foo": "car", "joe": "bloggs"})

    def test_multiple_iterative_fields_together(self):
        """Multiple iterative fields produce cartesian product."""
        hconfig = {"foo": ["bar", "car"], "joe": ["mason", "bloggs"]}
        configs = list(hyperrun.generate_configs(hconfig))
        self.assertEqual(len(configs), 4)
        self.assertEqual(configs[0], {"foo": "bar", "joe": "mason"})
        self.assertEqual(configs[-1], {"foo": "car", "joe": "bloggs"})

    def test_addition_of_hashes_configurations(self):
        """Each generated config gets a unique hash."""
        hconfig = {"foo": ["bar", "car"], "joe": ["mason", "bloggs"]}
        configs = list(hyperrun.generate_configs(hconfig, with_hash=True))
        self.assertIn("hash", configs[0])
        self.assertEqual(len({c["hash"] for c in configs}), 4)
