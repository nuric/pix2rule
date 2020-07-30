"""Configuration for experiments."""
from typing import Dict, Any
import logging
import pprint
import sys
import argparse
import hashlib
import json

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description=__doc__,
    fromfile_prefix_chars="@",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

config: Dict[str, Any] = {}


def add_parser(title: str, description: str = ""):
    """Create a new context for arguments and return a handle."""
    return parser.add_argument_group(title, description)


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def parse(save_fname: str = "") -> str:
    """Clean configuration and parse given arguments."""
    # Start from clean configuration
    config.clear()
    config.update(vars(parser.parse_args()))
    logging.info("Parsed %i arguments.", len(config))
    # Save passed arguments
    if save_fname:
        with open(save_fname, "w") as fout:
            fout.write("\n".join(sys.argv[1:]))
        logging.info("Saving arguments to %s.", save_fname)
    return dict_hash(config)


def print_config():
    """Print the current config to stdout."""
    pprint.pprint(config)
