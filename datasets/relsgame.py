"""Relations game dataset from PrediNet paper.

Source of data: https://arxiv.org/pdf/1905.10307.pdf"""
from typing import Dict, List, Tuple, Any
import logging
from pathlib import Path
import json

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
add_argument = configlib.add_group("Relsgame Dataset config", prefix="relsgame")
add_argument(
    "--tasks",
    nargs="*",
    default=[],
    choices=all_tasks,
    help="Tasks to generate, empty list for all.",
)
add_argument(
    "--train_size",
    default=1000,
    type=int,
    help="Training size per task, 0 to use everything.",
)
add_argument(
    "--validation_size",
    default=1000,
    type=int,
    help="Validation size per task, 0 to use everything.",
)
add_argument(
    "--test_size",
    default=1000,
    type=int,
    help="Test size per task, 0 to use everything.",
)
add_argument("--batch_size", default=64, type=int, help="Data batch size.")
add_argument("--one_hot_labels", action="store_true", help="One-hot encode labels.")


def get_file(fname: str) -> str:
    """Get or download relations game data."""
    assert fname.endswith(".npz"), "Can only download npz files for relsgame."
    # url = "https://storage.googleapis.com/storage/v1/b/relations-game-datasets/o/{}?alt=media"
    # fpath = tf.keras.utils.get_file(
    #     fname, url.format(fname), cache_subdir="relsgame", cache_dir=C["data_dir"]
    # )
    # 2021-02-18 (nuric): Direct download from the above url no longer works as the files
    # are not publicly exposed anymore. Using a Google Account we need to download them from
    # https://console.cloud.google.com/storage/browser/relations-game-datasets
    fpath = Path(C["data_dir"]) / "relsgame" / "original" / fname
    logger.info("Looking for file %s -> %s", str(fpath), fpath.exists())
    return str(fpath) if fpath.exists() else ""


def get_compressed_path() -> Path:
    """Return compressed dataset npz path for given training and test sizes."""
    # ---------------------------
    # Generate file description hash
    tasks = C["relsgame_tasks"] or all_tasks
    desc = {
        "name": "relsgame",
        "train_size": C["relsgame_train_size"],
        "validation_size": C["relsgame_validation_size"],
        "test_size": C["relsgame_test_size"],
        "tasks": sorted(tasks),
    }
    desc_hash = utils.hashing.dict_hash(desc)
    # ---------------------------
    # Create compressed data folder and log the index.json
    compressed_dir = Path(C["data_dir"]) / "relsgame" / "compressed"
    compressed_dir.mkdir(parents=True, exist_ok=True)
    index_json_filepath = compressed_dir / "index.json"
    # ---------------------------
    # Update json with new generated data file
    index_json = dict()
    if index_json_filepath.exists():
        with index_json_filepath.open() as filep:
            index_json = json.load(filep)
    index_json[desc_hash] = desc
    with index_json_filepath.open("w") as filep:
        json.dump(index_json, filep, indent=4)
    # ---------------------------
    return compressed_dir / f"{desc_hash}.npz"


def downsize_images(images: np.ndarray) -> np.ndarray:
    """Downsize images removing unnecessary pixels."""
    # images (N, 36, 36, 3)
    # images are formed by 1 pad, 3x3 pixel blocks, 2 pixel pad
    # each section is 12 pixels, and set in a 3x3 grid giving 36x36
    # ---------------------------
    # We will remove extra padding on the top and left,
    # and strip 1 padding on right and bottom
    # as well as reduce 3x3 pixel blocks down to 1x1
    # The images will look identical afterwards so no semantics is lost
    cidxs = np.arange(1, 12, 3)
    cidxs = np.concatenate([cidxs, cidxs + 12, cidxs + 12 * 2])
    return images[:, cidxs[:, None], cidxs]  # (N, 12, 12, 3)


def generate_data():  # pylint: disable=too-many-locals
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
    dataset_types = [
        fn.split(".")[0].split("_")[-1] for fn in dfiles
    ]  # [pentos, hexos, ...]
    logger.info("Compressing dataset types: %s", dataset_types)
    all_arrs: Dict[str, Dict[str, List[np.array]]] = {
        "test_" + k: {"images": list(), "task_ids": list(), "labels": list()}
        for k in dataset_types
    }
    all_arrs["train"] = {"images": list(), "task_ids": list(), "labels": list()}
    all_arrs["validation"] = {"images": list(), "task_ids": list(), "labels": list()}
    # ---------------------------
    # Iterate through data files and generate merged arrays
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
        imgs, labels = downsize_images(dnpz["images"]), dnpz["labels"].squeeze()
        assert len(imgs.shape) == 4, "Image data not in expected dimensions."
        assert (
            imgs.shape[0] == labels.shape[0]
        ), "Different number of images and labels."
        logger.debug("Relsgame file %s has %i many images.", fname, imgs.shape[0])
        if dname == "pentos":
            # ---------------------------
            # Check if we have enough data points
            total_required = (
                C["relsgame_train_size"]
                + C["relsgame_validation_size"]
                + C["relsgame_test_size"]
            )
            assert (
                total_required <= imgs.shape[0]
            ), f"Have {imgs.shape[0]} data points, need {total_required}."
            # ---
            # Generate random indices for training, validation and test
            sel_idxs = np.random.choice(imgs.shape[0], size=total_required)
            train_idxs, validation_idxs, test_idxs = (
                sel_idxs[: C["relsgame_train_size"]],
                sel_idxs[
                    C["relsgame_train_size"] : C["relsgame_train_size"]
                    + C["relsgame_validation_size"]
                ],
                sel_idxs[-C["relsgame_test_size"] :],
            )
            for key, indices in [
                ("train", train_idxs),
                ("validation", validation_idxs),
            ]:
                all_arrs[key]["images"].append(imgs[indices])
                all_arrs[key]["labels"].append(labels[indices])
                task_ids = np.full(
                    len(indices), all_tasks.index(task_name), dtype=np.int32
                )
                all_arrs[key]["task_ids"].append(task_ids)
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


def load_data() -> Tuple[  # pylint: disable=too-many-locals
    Dict[str, Any], Dict[str, tf.data.Dataset]
]:
    """Load and process relations game dataset."""
    cpath = get_compressed_path()
    if not cpath.exists():
        logger.warning("Given compressed file does not exist: %s", str(cpath))
        generate_data()
    # ---------------------------
    # Load the compressed data file
    dnpz = np.load(str(cpath))  # {'train_images': .., 'test_stripes_images': ...
    dsetnames = {
        "_".join(k.split("_")[:2]) for k in dnpz.files if k.startswith("test")
    }  # [test_stripes, ...]
    dsetnames.update(["train", "validation"])
    # ---------------------------
    # Sanity check
    for dataname, array in dnpz.items():
        # dataname is test_pentos, train etc.
        data_type = dataname.split("_")[0]  # train, validation or test
        tasks = C["relsgame_tasks"] or all_tasks
        multiplier = len(tasks)
        if "stripes" in dataname and "colour_and_or_shape" in tasks:
            multiplier -= 1  # one task does not have stripes
        expected = C["relsgame_" + data_type + "_size"] * multiplier
        assert (
            array.shape[0] == expected
        ), f"Expected {dataname} to have {expected} examples, got {array.shape}"
    # ---------------------------
    # Compute max labels for one-hot encoding
    max_label = max([v.max() for k, v in dnpz.items() if k.endswith("labels")])
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
        if C["relsgame_one_hot_labels"]:
            labels = np.eye(max_label + 1, dtype=np.int32)[labels]
        input_dict = {"image": imgs}
        # Optionally add task_ids if there are multiple tasks
        if len(C["relsgame_tasks"] or all_tasks) > 1:
            input_dict["task_id"] = task_ids
        tfdata = tf.data.Dataset.from_tensor_slices((input_dict, labels))
        if dname == "train":
            tfdata = tfdata.shuffle(1000).batch(C["relsgame_batch_size"]).repeat()
        else:
            tfdata = tfdata.batch(C["relsgame_batch_size"])
        dsets[dname] = tfdata
    # ---------------------------
    # Generate description
    inputs = {
        k: {"shape": tuple(v.shape), "dtype": v.dtype}
        for k, v in dsets["train"].element_spec[0].items()
    }
    inputs["image"]["type"] = "image"
    if "task_id" in inputs:
        inputs["task_id"]["type"] = "categorical"
        inputs["task_id"]["num_categories"] = len(all_tasks)
    # ---
    output_spec = dsets["train"].element_spec[1]
    output = {
        "shape": tuple(output_spec.shape),
        "dtype": output_spec.dtype,
        "num_categories": max_label + 1,
        "type": "multilabel" if len(output_spec.shape) > 1 else "multiclass",
        # We are learning a nullary predicate for each label
        "target_rules": [0] * (max_label + 1),
    }
    description = {
        "name": "relsgame",
        "inputs": inputs,
        "output": output,
        "datasets": list(dsets.keys()),
    }
    # ---------------------------
    return description, dsets
