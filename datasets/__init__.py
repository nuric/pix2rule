"""Combined data module."""
from typing import Dict, Tuple, Any
import tensorflow as tf
import configlib
from configlib import config as C

# Data modules
from . import relsgame
from . import gendnf

# ---------------------------
# Dataset registry
# registry = {d.__name__.split(".")[-1]: d for d in [sequences, relsgame]}
# type checker seems to not recognise what is going above
registry = {"relsgame": relsgame, "gendnf": gendnf}
# ---------------------------

# ---------------------------
# Configuration arguments
add_argument = configlib.add_group("Data config", prefix="dataset")
add_argument(
    "--name",
    default="relsgame",
    choices=registry.keys(),
    help="Dataset name to train / evaluate.",
)
# ---------------------------


def get_dataset(name: str = ""):
    """Get dataset module by name, defaults to argument parameter."""
    assert "data_dir" in C, "No data_dir specified for datasets."
    return registry[name or C["dataset_name"]]
