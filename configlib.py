"""Configuration for experiments."""
from typing import Any, Dict, Callable
import argparse
import json
import logging
import pprint

import utils.hashing

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--config_json", help="Configuration json and index to merge.")

config: Dict[str, Any] = {}


def add_group(
    title: str, prefix: str = "", description: str = ""
) -> Callable[..., Any]:
    """Create a new context for arguments and return a handle."""
    arg_group = parser.add_argument_group(title, description)
    prefix = prefix + "_" if prefix else prefix

    def add_argument_wrapper(name: str, **kwargs: Any):
        """Wrapper for adding arguments into the group."""
        # Strip -- if exists, refactor to removeprefix in 3.9+
        if name.startswith("--"):
            name = name[2:]
        # Add splitting character
        arg_group.add_argument("--" + prefix + name, **kwargs)

    return add_argument_wrapper


def add_arguments_dict(
    add_function: Callable[..., Any],
    arguments: Dict[str, Dict[str, Any]],
    prefix: str = "",
):
    """Add arguments from a dictionary into the parser with given prefix."""
    prefix = prefix + "_" if prefix else prefix
    for argname, conf in arguments.items():
        add_function(prefix + argname, **conf)


def parse(save_fname: str = "") -> str:
    """Clean configuration and parse given arguments."""
    # Start from clean configuration
    config.clear()
    config.update(vars(parser.parse_args()))
    logger.info("Parsed %i arguments.", len(config))
    # Save passed arguments
    if save_fname:
        save_config(save_fname)
    # Optionally merge in json configuration
    if config["config_json"]:
        # This is a filepath and index
        # myconfigs.json.2347825
        *json_path, index = config["config_json"].split(".")
        # [myconfigs, json], 2348725
        with open(".".join(json_path)) as json_file:
            json_index = json.load(json_file)
            # Warn against missing keys, this is not all bad as
            # the user might want to add extra labels and tags
            diff = set(json_index[index].keys()) - set(config.keys())
            if diff:
                logger.warning("Unspecified configuration keys: %s", str(diff))
            # Update the global configuration
            config.update(json_index[index])
            logger.info(
                "Loaded %d parameters from %s",
                len(json_index[index]),
                config["config_json"],
            )
    return utils.hashing.dict_hash(config)


def save_config(save_fname: str = "config.json"):
    """Save config file as a json."""
    assert save_fname.endswith(
        ".json"
    ), f"Config file needs end with json, got {save_fname}."
    with open(save_fname, "w") as config_file:
        json.dump(config, config_file, indent=4)
    logger.info("Saved configuration to %s.", save_fname)


def print_config():
    """Print the current config to stdout."""
    pprint.pprint(config)
