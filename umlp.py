"""Unification MLP."""
import os
import logging
import numpy as np

import configlib
import data

# Calm down tensorflow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Setup logging
logging.basicConfig(level=logging.INFO)

# Disable scientific printing
np.set_printoptions(suppress=True, precision=5, linewidth=180)

# Arguments
parser = configlib.add_parser("UMLP options.")
parser.add_argument(
    "--invariants", default=1, type=int, help="Number of invariants per task."
)
parser.add_argument("--embed", default=16, type=int, help="Embedding size.")
parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
parser.add_argument("--nouni", action="store_true", help="Disable unification.")


configlib.parse()
print("Running with configuration:")
configlib.print_config()
dsets = data.sequences.load_data()
print(dsets)
