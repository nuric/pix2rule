"""Combined data module."""
from typing import Dict
import tensorflow as tf
import configlib

# Data modules
# from . import mnist
from . import sequences
from . import relsgame

# ---------------------------
# Configuration arguments
parser = configlib.add_parser("Data config")
parser.add_argument("--data_dir", default="data", help="Data folder.")
# ---------------------------
