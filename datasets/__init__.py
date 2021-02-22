"""Combined data module."""
from typing import Dict, Tuple, Any
import tensorflow as tf
import configlib
from configlib import config as C

# Data modules
from . import sequences
from . import relsgame

# ---------------------------
# Dataset registry
# registry = {d.__name__.split(".")[-1]: d for d in [sequences, relsgame]}
# type checker seems to not recognise what is going above
registry = {"sequences": sequences.load_data, "relsgame": relsgame.load_data}
# ---------------------------

# ---------------------------
# Configuration arguments
add_argument = configlib.add_group("Data config", prefix="dataset")
add_argument(
    "--name",
    default="sequences",
    choices=registry.keys(),
    help="Dataset name to train / evaluate.",
)
# ---------------------------


def load_data(name: str = None) -> Tuple[Dict[str, Any], Dict[str, tf.data.Dataset]]:
    """Load dataset by given name."""
    assert "data_dir" in C, "No data_dir specified for datasets."
    return registry[name or C["dataset_name"]]()
