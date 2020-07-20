"""Sequences dataset."""
from typing import Dict, List
import logging
import numpy as np
import tensorflow as tf

import configlib
from configlib import config as C

logger = logging.getLogger(__name__)

# ---------------------------
# Configuration arguments
parser = configlib.add_parser("Sequences Dataset config")
parser.add_argument(
    "--seq_length", default=4, type=int, help="Fixed length of symbol sequences."
)
parser.add_argument("--seq_symbols", default=8, type=int, help="Number of symbols.")
parser.add_argument(
    "--seq_tasks",
    nargs="*",
    default=[],
    type=int,
    help="Tasks to generate, empty list for all.",
)
parser.add_argument(
    "--seq_train_size",
    default=1000,
    type=int,
    help="Training size per task, 0 to use everything.",
)
parser.add_argument(
    "--seq_test_split",
    default=0.1,
    type=float,
    help="Test split of generated data before training size.",
)
parser.add_argument(
    "--seq_gen_size", default=1000, type=int, help="Random data tries per task."
)
parser.add_argument("--seq_batch_size", default=64, type=int, help="Data batch size.")

# ---------------------------


def rand_syms(replace: bool = True) -> np.ndarray:
    """Return unique random symbols."""
    # We'll add 1, reserve 0 for padding
    return np.random.choice(C["seq_symbols"], size=C["seq_length"], replace=replace) + 1


# Generate random data for tasks
def gen_task1() -> np.ndarray:
    """Task 1: return constant symbol."""
    seq = rand_syms()
    return np.concatenate(([1], seq, [2]))  # (1+L+1,)


def gen_task2() -> np.ndarray:
    """Task 2: head of random sequence."""
    seq = rand_syms()
    return np.concatenate(([2], seq, [seq[0]]))  # (1+L+1,)


def gen_task3() -> np.ndarray:
    """Task 3: tail of random sequence."""
    seq = rand_syms()
    return np.concatenate(([3], seq, [seq[-1]]))  # (1+L+1,)


def gen_task4() -> np.ndarray:
    """Task 4: item that is repeated twice."""
    seq = rand_syms(replace=False)
    # select two random locations and make them equal
    xid, yid = np.random.choice(len(seq), size=2, replace=False)
    seq[xid] = seq[yid]  # item is repeated
    return np.concatenate(([4], seq, [seq[xid]]))  # (1+L+1,)


def gen_all() -> Dict[str, np.ndarray]:
    """Generate all tasks."""
    gdata: Dict[str, List[np.ndarray]] = {"train": list(), "test": list()}
    # Get the number of available tasks
    tasks = [int(x[8:]) for x in globals().keys() if x.startswith("gen_task")]
    tasks = C["seq_tasks"] or tasks
    # Generate each one of them
    for i in tasks:
        func = globals()["gen_task" + str(i)]
        task = [func() for _ in range(C["seq_gen_size"])]  # (S, 1+L+1)
        task = np.unique(np.stack(task), axis=0)  # (<S, 1+L+1)
        np.random.shuffle(task)  # (<S, 1+L+1)
        # Split test set
        split = np.ceil(len(task) * C["seq_test_split"]).astype(int)
        gdata["test"].append(task[:split])
        gdata["train"].append(task[split : split + C["seq_train_size"]])
    # Merge tasks into a single dataset
    def concat_shuffle(ldata: List[np.ndarray]) -> np.ndarray:
        """Concatenate and shuffle given list of datasets."""
        concat = np.concatenate(ldata)  # (S, 1+L+1)
        np.random.shuffle(concat)
        return concat

    return {k: concat_shuffle(v) for k, v in gdata.items()}


# ---------------------------

# Data loading
def load_data() -> Dict[str, tf.data.Dataset]:
    """Load and prepare Sequences data."""
    # Generate the dataset, we get a single numpy array
    data = gen_all()  # {'train': (train, 1+L+1), 'test': (test, 1+L+1)}
    # Extract some information about the data
    metadata = dict()
    for k, arr in data.items():
        metadata[k] = arr.shape
        metadata[k + "_tasks"] = np.unique(arr[:, 0], return_counts=True)
        metadata[k + "_labels"] = np.unique(arr[:, -1], return_counts=True)
    logger.info("Generated sequence data %s", str(metadata))
    # ---------------------------
    # Convert to tf.data.Dataset
    # tfdata = tf.data.Dataset.from_tensor_slices({"input": data[:, :-1], "label": data[:, -1]})
    dsets: Dict[str, tf.data.Dataset] = {
        k: tf.data.Dataset.from_tensor_slices(({"input": v[:, :-1]}, v[:, -1]))
        for k, v in data.items()
    }
    # Shuffle and batch
    dsets["train"] = dsets["train"].shuffle(1000).batch(C["seq_batch_size"]).repeat()
    dsets["test"] = dsets["test"].batch(C["seq_batch_size"])
    # ---------------------------
    return dsets
