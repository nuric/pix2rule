"""Configuration for experiments."""
from typing import Dict, Any
import logging
import pprint
import sys
import argparse

import utils.hashing

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


def add_arguments_dict(
    existing_parser: argparse.ArgumentParser,
    arguments: Dict[str, Dict[str, Any]],
    prefix: str = "",
):
    """Add arguments from a dictionary into the parser with given prefix."""
    if not prefix.startswith("--"):
        prefix = "--" + prefix
    for argname, conf in arguments.items():
        existing_parser.add_argument(prefix + argname, **conf)


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
    return utils.hashing.dict_hash(config)


def print_config():
    """Print the current config to stdout."""
    pprint.pprint(config)
