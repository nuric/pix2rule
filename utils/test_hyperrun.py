"""Test cases for the hyper run configuration library."""
from typing import Dict, Any
import unittest

from . import hyperrun


class TestHyperRunGeneration(unittest.TestCase):
    """Test cases for the hyper configuration generation function."""

    def test_empty_dict_gives_empty_dict(self):
        """The configuration generator gives an empty dictionary on an empty dictionary."""
        hconfig: Dict[str, Any] = dict()
        configs = hyperrun.generate_configs(hconfig)
        self.assertEqual(len(configs), 1)
        self.assertFalse(configs[0])

    def test_constant_dict_gives_single_dict(self):
        """A single dictionary with constant values give a single dictionary."""
        hconfig = {"foo": "bar"}
        configs = hyperrun.generate_configs(hconfig)
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0], hconfig)

    def test_single_iterative_field(self):
        """A single iterative field gives back multiple configurations."""
        hconfig = {"foo": ["bar", "car"]}
        configs = hyperrun.generate_configs(hconfig)
        self.assertEqual(len(configs), 2)
        self.assertEqual(configs[0], {"foo": "bar"})
        self.assertEqual(configs[1], {"foo": "car"})

    def test_iterative_and_constant_together(self):
        """Constant fields are added to iterative fields."""
        hconfig = {"foo": ["bar", "car"], "joe": "bloggs"}
        configs = hyperrun.generate_configs(hconfig)
        self.assertEqual(len(configs), 2)
        self.assertEqual(configs[0], {"foo": "bar", "joe": "bloggs"})
        self.assertEqual(configs[1], {"foo": "car", "joe": "bloggs"})

    def test_multiple_iterative_fields_together(self):
        """Multiple iterative fields produce cartesian product."""
        hconfig = {"foo": ["bar", "car"], "joe": ["mason", "bloggs"]}
        configs = hyperrun.generate_configs(hconfig)
        self.assertEqual(len(configs), 4)
        self.assertEqual(configs[0], {"foo": "bar", "joe": "mason"})
        self.assertEqual(configs[-1], {"foo": "car", "joe": "bloggs"})

    def test_addition_of_hashes_configurations(self):
        """Each generated config gets a unique hash."""
        hconfig = {"foo": ["bar", "car"], "joe": ["mason", "bloggs"]}
        configs = hyperrun.generate_configs(hconfig, with_hash=True)
        self.assertIn("hash", configs[0])
        self.assertEqual(len({c["hash"] for c in configs}), 4)


class TestHyperRunChain(unittest.TestCase):
    """Test cases for chaining configuration dictionaries."""

    def test_single_configuration(self):
        """Single configuration gives back a single configuration."""
        config = {"foo": "bar"}
        configs = hyperrun.chain_configs([config])
        self.assertEqual(len(configs), 1)

    def test_no_chain_two_configurations(self):
        """Two configurations from same list do not get chained."""
        configs1 = [{"foo": "bar"}, {"bar": "foo"}]
        configs = hyperrun.chain_configs(configs1)
        self.assertEqual(len(configs), 2)

    def test_chain_two_configurations(self):
        """Two different configuration lists get chained."""
        configs1 = [{"foo": "bar"}, {"bar": "foo"}]
        configs2 = [{"thomas": "edison"}, {"life": "42"}]
        configs = hyperrun.chain_configs(configs1, configs2)
        self.assertEqual(len(configs), 4)
