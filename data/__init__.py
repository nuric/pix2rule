"""Combined data module."""
from typing import Dict
import tensorflow as tf
import configlib

# Data modules
# from . import mnist
from . import sequences

# ---------------------------
# Configuration arguments
parser = configlib.add_parser("Data config")
parser.add_argument(
    "--tfds_data_dir", default="~/tensorflow_datasets", help="Data folder."
)
# ---------------------------
