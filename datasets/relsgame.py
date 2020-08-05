"""Relations game dataset from PrediNet paper.

Source of data: https://arxiv.org/pdf/1905.10307.pdf"""
from typing import Dict, List
import logging
from pathlib import Path

import tqdm
import numpy as np
import tensorflow as tf

import configlib
from configlib import config as C
import utils.hashing

logger = logging.getLogger(__name__)

all_tasks = ["same", "between", "occurs", "xoccurs", "colour_and_or_shape"]
all_types = ["pentos", "hexos", "stripes"]

# ---------------------------
# Configuration arguments
parser = configlib.add_parser("Sequences Dataset config")
parser.add_argument(
    "--relsgame_tasks",
    nargs="*",
    default=[],
    choices=all_tasks,
    help="Tasks to generate, empty list for all.",
)
parser.add_argument(
    "--relsgame_train_size",
    default=1000,
    type=int,
    help="Training size per task, 0 to use everything.",
)
parser.add_argument(
    "--relsgame_test_size",
    default=1000,
    type=int,
    help="Test size per task, 0 to use everything.",
)
parser.add_argument(
    "--relsgame_batch_size", default=64, type=int, help="Data batch size."
)


def get_file(fname: str) -> str:
    """Get or download relations game data."""
    assert fname.endswith(".npz"), "Can only download npz files for relsgame."
    url = "https://storage.googleapis.com/storage/v1/b/relations-game-datasets/o/{}?alt=media"
    try:
        fpath = tf.keras.utils.get_file(
            fname, url.format(fname), cache_subdir="relsgame", cache_dir=C["data_dir"]
        )
        return str(fpath)
    except Exception as exception:  # pylint: disable=broad-except
        logger.warning("Could not download %s: %s", fname, exception)
        return ""


def get_compressed_path() -> Path:
    """Return compressed dataset npz path for given training and test sizes."""
    task_hash = utils.hashing.set_hash(C["relsgame_tasks"] or all_tasks)
    fname = (
        f"train{C['relsgame_train_size']}_test{C['relsgame_test_size']}_{task_hash}.npz"
    )
    return Path(C["data_dir"]) / "relsgame" / fname


def create_compressed_files():
    """Load compressed shortened versions of data files."""
    # ---------------------------
    # Collect all the files
    dfiles = [
        get_file("{}_{}.npz".format(tname, dname))
        for tname in C["relsgame_tasks"] or all_tasks
        for dname in all_types
    ]
    # Remove empty ones, some datasets don't have stripes etc.
    dfiles = [f for f in dfiles if f]
    logger.info("Found %i relsgame data files.", len(dfiles))
    # ---------------------------
    # Curate the files into a single dataset
    all_arrs: Dict[str, Dict[str, List[np.array]]] = {
        "test_" + k: {"images": list(), "task_ids": list(), "labels": list()}
        for k in all_types
    }
    all_arrs["train"] = {"images": list(), "task_ids": list(), "labels": list()}
    logger.info("Loading relsgame data npz files.")
    for fname in tqdm.tqdm(dfiles):
        # Parse filename
        dname = fname.split("/")[-1].split(".")[0]  # colour_and_or_shape_pentos
        *task_name, dname = dname.split("_")  # [colour, and, or, shape] pentos
        task_name = "_".join(task_name)  # colour_and_or_shape
        # ---
        # Load npz file
        dnpz = np.load(fname)
        # dnpz {'images': (N, 36, 36, 3), 'task_ids': (N,), 'labels': (N,)}
        imgs, labels = dnpz["images"], dnpz["labels"].squeeze()
        assert len(imgs.shape) == 4, "Image data not in expected dimensions."
        assert (
            imgs.shape[0] == labels.shape[0]
        ), "Different number of images and labels."
        logger.debug("Relsgame file %s has %i many images.", fname, imgs.shape[0])
        if dname == "pentos":
            sel_idxs = np.random.choice(
                imgs.shape[0], size=C["relsgame_train_size"] + C["relsgame_test_size"]
            )
            train_idxs, test_idxs = (
                sel_idxs[: C["relsgame_train_size"]],
                sel_idxs[C["relsgame_train_size"] :],
            )
            all_arrs["train"]["images"].append(imgs[train_idxs])
            all_arrs["train"]["labels"].append(labels[train_idxs])
            task_ids = np.full(
                len(train_idxs), all_tasks.index(task_name), dtype=np.int32
            )
            all_arrs["train"]["task_ids"].append(task_ids)
        else:
            test_idxs = np.random.choice(imgs.shape[0], size=C["relsgame_test_size"])
        all_arrs["test_" + dname]["images"].append(imgs[test_idxs])
        all_arrs["test_" + dname]["labels"].append(labels[test_idxs])
        task_ids = np.full(len(test_idxs), all_tasks.index(task_name), dtype=np.int32)
        all_arrs["test_" + dname]["task_ids"].append(task_ids)
    # ---------------------------
    # Compress and save
    compressed_arrs: Dict[str, np.array] = {
        dsetname + "_" + iname: np.concatenate(arrs).astype(np.uint8)
        for dsetname, dset in all_arrs.items()
        for iname, arrs in dset.items()
    }
    cpath = str(get_compressed_path())
    logger.info("Creating %s with keys: %s", cpath, str(compressed_arrs.keys()))
    np.savez_compressed(cpath, **compressed_arrs)


def load_data() -> Dict[str, tf.data.Dataset]:
    """Load and process relations game dataset."""
    cpath = get_compressed_path()
    if not cpath.exists():
        logger.warning("Given compressed file does not exist: %s", str(cpath))
        create_compressed_files()
    # ---------------------------
    # Load the compressed data file
    dnpz = np.load(str(cpath))  # {'train_images': .., 'test_stripes_images': ...
    dsetnames = {
        "_".join(k.split("_")[:2]) for k in dnpz.files if k.startswith("test")
    }  # [test_stripes, ...]
    dsetnames.add("train")
    # ---------------------------
    # Curate datasets
    dsets: Dict[str, tf.data.Dataset] = dict()
    for dname in dsetnames:
        # Shuffle in unison
        ridxs = np.random.permutation(dnpz[dname + "_labels"].shape[0])
        imgs = tf.image.convert_image_dtype(dnpz[dname + "_images"][ridxs], tf.float32)
        # we expand types for tensorflow
        task_ids = dnpz[dname + "_task_ids"][ridxs].astype(np.int32)
        labels = dnpz[dname + "_labels"][ridxs].astype(np.int32)
        tfdata = tf.data.Dataset.from_tensor_slices(
            ({"image": imgs, "task_id": task_ids}, labels)
        )
        if dname == "train":
            tfdata = tfdata.shuffle(1000).batch(C["relsgame_batch_size"]).repeat()
        else:
            tfdata = tfdata.batch(C["relsgame_batch_size"])
        dsets[dname] = tfdata
    # ---------------------------
    # Convert to tf.data.Dataset
    return dsets
