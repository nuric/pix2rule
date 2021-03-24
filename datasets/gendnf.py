"""Synthetic DNF data for evaluation rule learning."""
from typing import Dict, Tuple, Any
import logging
from pathlib import Path
import json
import itertools

import numpy as np
import tensorflow as tf

import configlib
from configlib import config as C
import utils.hashing

logger = logging.getLogger(__name__)

# ---------------------------
# Configuration arguments
add_argument = configlib.add_group("DNF dataset config", prefix="gendnf")
add_argument(
    "--num_objects",
    type=int,
    default=4,
    help="Number of objects / constants.",
)
add_argument(
    "--num_nullary",
    default=6,
    type=int,
    help="Number of nullary predicates.",
)
add_argument(
    "--num_unary",
    default=7,
    type=int,
    help="Number of unary predicates in the language.",
)
add_argument(
    "--num_binary",
    default=8,
    type=int,
    help="Number of binary predicates in the language.",
)
add_argument(
    "--num_variables", type=int, default=3, help="Number of variables in rule body."
)
add_argument(
    "--num_conjuncts", type=int, default=4, help="Number of ways a rule can be defined."
)
add_argument(
    "--target_arity", type=int, default=0, help="Arity of target rule to learn."
)
add_argument(
    "--gen_size",
    type=int,
    default=10000,
    help="Number of examples per label to generate.",
)
add_argument(
    "--train_size",
    default=1000,
    type=int,
    help="DNF training size, number of positive + negative examples upper bound.",
)
add_argument(
    "--validation_size",
    default=1000,
    type=int,
    help="Validation size positive + negative examples upper bound.",
)
add_argument(
    "--test_size",
    default=1000,
    type=int,
    help="Test size upper bound.",
)
add_argument("--batch_size", default=64, type=int, help="Data batch size.")
add_argument(
    "--noise_stddev",
    default=0.0,
    type=float,
    help="Added label noise.",
)
add_argument(
    "--rng_seed",
    default=42,
    type=int,
    help="Random number generator seed.",
)
# ---------------------------


def get_build_path() -> Path:
    """Return a build path based on configuration parameters."""
    # We do this to ensure that models train on the same generated data
    # rather than regenerating a new set for each model run
    # ---------------------------
    # Generate file description hash
    desc = {
        k: C["gendnf_" + k]
        for k in [
            "num_objects",
            "num_variables",
            "num_conjuncts",
            "num_nullary",
            "num_unary",
            "num_binary",
            "rng_seed",
        ]
    }
    desc["name"] = "gendnf"
    desc_hash = utils.hashing.dict_hash(desc)
    # ---------------------------
    # Create the build path
    output_dir = Path(C["data_dir"]) / "gendnf"
    bpath = output_dir / f"{desc_hash}.npz"
    # ---------------------------
    # Check for new generated file
    if not bpath.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        index_json_filepath = output_dir / "index.json"
        # ---------------------------
        index_json = dict()
        if index_json_filepath.exists():
            with index_json_filepath.open() as filep:
                index_json = json.load(filep)
        index_json[desc_hash] = desc
        with index_json_filepath.open("w") as filep:
            json.dump(index_json, filep, indent=4)
    # ---------------------------
    return bpath


def flatten_interpretation(
    nullary: np.ndarray, unary: np.ndarray, binary: np.ndarray
) -> np.ndarray:
    """Flatten given interpretation with different arities."""
    # nullary (..., numNullary), unary (..., O, numUnary), binary (..., O, O-1, numBinary)
    flat_unary = unary.reshape(unary.shape[:-2] + (-1,))  # (..., O*numUnary)
    flat_binary = binary.reshape(binary.shape[:-3] + (-1,))  # (..., O*(O-1)*numUnary)
    return np.concatenate([nullary, flat_unary, flat_binary], -1)  # (..., IN)


def evaluate_dnf(
    num_objects: int,
    num_vars: int,
    nullary: np.ndarray,
    unary: np.ndarray,
    binary: np.ndarray,
    and_kernel: np.ndarray,
    or_kernel: np.ndarray,
    target_arity: int,
) -> np.ndarray:
    """Evaluate given batch of interpretations."""
    # nullary (B, numNullary)
    # unary (B, O, numUnary)
    # binary (B, O, O-1, numBinary)
    # and_kernel (H, IN)
    # or_kernel (H,)
    # ---------------------------
    # We need a binding / permutation matrix that binds every object to every
    # variable, so we can evaluate the rule. The following list of tuples,
    # tells us which constant each variable is for each permutation
    perm_idxs = np.array(
        list(itertools.permutations(range(num_objects), num_vars))
    )  # (K, V)
    # ---
    # Binary comparison indices for variables, XY XZ YX YZ ...
    var_bidxs = np.stack(np.nonzero(1 - np.eye(num_vars))).T  # (V*(V-1), 2)
    perm_bidxs = perm_idxs[:, var_bidxs]  # (K, V*(V-1), 2)
    obj_idxs = np.stack(np.nonzero(1 - np.eye(num_objects))).T  # (O*(O-1), 2)
    # The following matrix tells with variable binding pair is actually the
    # object pair we're looking for
    var_obj_pairs = (perm_bidxs[..., None, :] == obj_idxs).all(-1)
    # (K, V*(V-1), O*(O-1))
    # We are guaranteed to have 1 matching pair due to unique bindings, so the
    # non-zero elements in the last dimension encode the index we want
    var_obj_pairs = np.reshape(np.nonzero(var_obj_pairs)[-1], var_obj_pairs.shape[:2])
    # (K, V*(V-1))
    # ---------------------------
    batch_size = nullary.shape[0]  # B
    # Take the permutations
    perm_unary = unary[:, perm_idxs]  # (B, K, V, numUnary)
    perm_binary = binary.reshape(
        (batch_size, -1, binary.shape[-1])
    )  # (B, O*(O-1), numBinary)
    perm_binary = perm_binary[:, var_obj_pairs]  # (B, K, V*(V-1), numBinary)
    perm_binary = perm_binary.reshape(
        (
            batch_size,
            var_obj_pairs.shape[0],
            num_vars,
            num_vars - 1,
            perm_binary.shape[-1],
        )
    )
    # (B, K, V, V-1, numBinary)
    # ---------------------------
    # Merge different arities
    flat_nullary = np.repeat(
        nullary[:, None], perm_unary.shape[1], axis=1
    )  # (B, K, numNullary)
    interpretation = flatten_interpretation(flat_nullary, perm_unary, perm_binary)
    # (B, K, IN)
    # ---------------------------
    # Evaluate
    and_eval = np.min(
        interpretation[:, :, None] * and_kernel + (and_kernel == 0), -1
    )  # (B, K, H)
    # ---
    # Reduction of existential variables if any, K actually expands to O, O-1 etc numVars many times
    # If the arity of the target predicate is 0, then we can reduce over K. If
    # it is 1, then expand once then reduce over remaining variables, i.e. O, K//O, H -> (O, H)
    shape_range = num_objects - np.arange(num_objects)  # [numObjs, numObjs-1, ...]
    new_shape = np.concatenate(
        [[batch_size], shape_range[:target_arity], [-1, and_eval.shape[-1]]]
    )  # [B, O, K//O,, H]
    and_eval = np.reshape(and_eval, new_shape)
    # (B, O, K//0, H)
    perm_eval = np.max(and_eval, -2)  # (B, H,) if arity 0, (B, O, H) if 1 etc.
    # ---
    or_eval = np.max(
        or_kernel * perm_eval - (or_kernel == 0), -1
    )  # (B,) if arity 0, (B, O) if 1 etc.
    # ---------------------------
    return or_eval


def generate_data() -> str:  # pylint: disable=too-many-locals
    """Generate and return path to requested data."""
    # Check if this already exists
    fpath = get_build_path()
    if fpath.exists():
        return str(fpath)
    # ---------------------------
    # Generate and save DNF data
    logger.info("Generating DNF data file: %s", str(fpath))
    rng = np.random.default_rng(seed=C["gendnf_rng_seed"])
    max_rng_tries = 100  # Number of tries before we give up generating
    # We will follow a monte carlo method and gamble with random interpretations
    # until we get enough examples. You can call it, generate and test approach
    # rather than generating a rule that we know works. Random samples are easier to implement
    # and more efficient when things scale.
    # ---
    # I'm changing the style here to make things easier to follow, be quiet pylint
    # pylint: disable=invalid-name
    numVars, numObjs = C["gendnf_num_variables"], C["gendnf_num_objects"]
    numNullary, numUnary, numBinary = (
        C["gendnf_num_nullary"],
        C["gendnf_num_unary"],
        C["gendnf_num_binary"],
    )
    numConjuncts = C["gendnf_num_conjuncts"]
    targetArity = C["gendnf_target_arity"]
    assert (
        targetArity == 0
    ), "Only arity 0, propositional rules are implemented for now."
    gen_size = C["gendnf_gen_size"]
    # ---------------------------
    # Let's generate a random rule first, by looking at the size of input interpretation
    # interpretation is the grounding of every variable on every predicate in the language
    assert numVars <= numObjs, f"Got more variables {numVars} than objects {numObjs}."
    # This is the size of the truth value of variable, any mapping over this
    # will define a conjunction Suppose we have [0, 1, 1], and 2 nullary and 1
    # unary predicate, it could mean u1 AND p1(X)
    in_size = numNullary + numVars * numUnary + numVars * (numVars - 1) * numBinary
    var_bidxs = np.stack(np.nonzero(1 - np.eye(numVars))).T  # (V*(V-1), 2)
    pred_names = (
        [f"n{i}" for i in range(numNullary)]
        + [f"p{j}(V{i})" for i, j in itertools.product(range(numVars), range(numUnary))]
        + [
            f"q{k}(V{i},V{j})"  # There is this modulus business because we omit p(X,X)
            for (i, j), k in itertools.product(var_bidxs, range(numBinary))
        ]
    )  # This is a mess that produces "n0..n, p1..n(V1), p1...n(V2), etc"
    assert len(pred_names) == in_size, "Predicate names did not match number of inputs."
    # Here is a full dump for our sanity:
    # ['n0', 'n1', 'n2', 'n3', 'p0(V0)', 'p1(V0)', 'p2(V0)', 'p3(V0)',
    # 'p0(V1)', 'p1(V1)', 'p2(V1)', 'p3(V1)', 'p0(V2)', 'p1(V2)', 'p2(V2)',
    # 'p3(V2)', 'q0(V0,V1)', 'q1(V0,V1)', 'q2(V0,V1)', 'q3(V0,V1)',
    # 'q0(V0,V2)', 'q1(V0,V2)', 'q2(V0,V2)', 'q3(V0,V2)', 'q0(V1,V0)',
    # 'q1(V1,V0)', 'q2(V1,V0)', 'q3(V1,V0)', 'q0(V1,V2)', 'q1(V1,V2)',
    # 'q2(V1,V2)', 'q3(V1,V2)', 'q0(V2,V0)', 'q1(V2,V0)', 'q2(V2,V0)',
    # 'q3(V2,V0)', 'q0(V2,V1)', 'q1(V2,V1)', 'q2(V2,V1)', 'q3(V2,V1)']
    # ---
    # But we want the conjuncts to be unique, here is when we start to gamble
    for i in range(max_rng_tries):
        # -1, 0, 1 is negation, not in, positive in the conjunct
        and_kernel = rng.choice([-1, 0, 1], size=(numConjuncts, in_size))  # (H, IN)
        if np.unique(and_kernel, axis=0).shape[0] == numConjuncts:
            # We're done, we found unique conjuncts
            break
        if i == (max_rng_tries - 1):
            raise RuntimeError(
                "Could not generate unique conjuncts, try increasing the language size."
            )
    # ---------------------------
    # Now let's generate the final disjunction, luckily it's a one off
    # We only generate either in disjunction or not since having negation of conjuncts
    # does not conform to normal logic programs. That is, you never get p <- not (q, r)
    or_kernel = rng.choice([0, 1], size=(numConjuncts,))  # (H,)
    while not or_kernel.any():  # We want at least one conjunction
        or_kernel = rng.choice([0, 1], size=(numConjuncts,))  # (H,)
    # That's the rule done, we have a kernel for conjunctions and an outer kernel for disjunctions
    # ---------------------------
    # Generate positive examples, we will reverse engineer the or_kernel and and_kernel
    # if we randomly try, as we increase the language size, it becomes increasingly unlikely
    # to get a positive example. So we take a more principled approach
    # We'll pick one conjunction to satisfy
    conjunct_idx = np.flatnonzero(or_kernel)  # (<H,)
    conjunct_idx = rng.choice(conjunct_idx, size=gen_size)  # (B,)
    ckernel = and_kernel[conjunct_idx]  # (B, IN)
    # Now we generate an interpretation that will satisfy ckernel
    # if the predicates are in the rule then take their value
    # otherwise it could be random, we don't care
    cmask = ckernel == 0  # (B, IN)
    pos_example = (
        cmask * rng.choice([-1, 1], size=(gen_size, in_size)) + (1 - cmask) * ckernel
    )  # (B, IN)
    # The above is in the space of variables, post binding if you will. We need to add missing
    # objects if any. That is, if O > V.
    pos_nullary = pos_example[:, :numNullary]  # (B, numNullary)
    pos_unary = pos_example[:, numNullary : numNullary + numUnary * numVars].reshape(
        (gen_size, numVars, numUnary)
    )  # (B, V, numUnary)
    pos_binary = pos_example[:, numNullary + numUnary * numVars :].reshape(
        (gen_size, numVars, numVars - 1, numBinary)
    )  # (B, V, V-1, numBinary)
    if numVars < numObjs:
        # Fill in remaining objects with random groundings
        rng_unary = rng.choice([-1, 1], size=(gen_size, numObjs - numVars, numUnary))
        # (B, O-V, numUnary)
        pos_unary = np.concatenate([pos_unary, rng_unary], 1)  # (B, O, numUnary)
        # ---
        # Here are essentially adding new objects in.
        # Existing objects V many, gets compared to new objects O-V many
        rng_binary = rng.choice(
            [-1, 1], size=(gen_size, numVars, numObjs - numVars, numBinary)
        )  # (B, V, O-V, numBinary)
        pos_binary = np.concatenate([pos_binary, rng_binary], axis=2)
        # (B, V, O-1, numBinary)
        # New objects O-V many, gets compared to everything O-1 many
        rng_binary = rng.choice(
            [-1, 1], size=(gen_size, numObjs - numVars, numObjs - 1, numBinary)
        )  # (B, O-V, O-1, numBinary)
        pos_binary = np.concatenate([pos_binary, rng_binary], axis=1)
        # (B, O, O-1, numBinary)
    res = evaluate_dnf(
        numObjs,
        numVars,
        pos_nullary,
        pos_unary,
        pos_binary,
        and_kernel,
        or_kernel,
        targetArity,
    )
    # Sanity check
    assert np.all(res == 1), "Expected positive examples to return 1"
    # ---------------------------
    # Generate negative examples, it's more likely if we generate random examples
    # they will be false, so we let all the random number generator do it's magic
    neg_nullary = rng.choice([-1, 1], size=(gen_size, numNullary))  # (B, numNullary,)
    neg_unary = rng.choice(
        [-1, 1], size=(gen_size, numObjs, numUnary)
    )  # (B, O, numUnary)
    neg_binary = rng.choice(
        [-1, 1], size=(gen_size, numObjs, numObjs - 1, numBinary)
    )  # (B, O*(O-1), numBinary)
    # ---
    # Check if they are actually false
    res = evaluate_dnf(
        numObjs,
        numVars,
        neg_nullary,
        neg_unary,
        neg_binary,
        and_kernel,
        or_kernel,
        targetArity,
    )  # (B,) if arity 0, (B, O) if arity 1 etc.
    # ---
    neg_idxs = np.flatnonzero(res == -1)  # (<B)
    neg_nullary = neg_nullary[neg_idxs]  # (<B, numNullary)
    neg_unary = neg_unary[neg_idxs]  # (<B, O, numUnary)
    neg_binary = neg_binary[neg_idxs]  # (<B, O, O-1, numBinary)
    # ---------------------------
    # Let's merge positive and negative examples
    nullary = np.concatenate([pos_nullary, neg_nullary], 0)  # (B', numNullary)
    unary = np.concatenate([pos_unary, neg_unary], 0)  # (B', O, numUnary)
    binary = np.concatenate([pos_binary, neg_binary], 0)  # (B', O, O-1, numBinary)
    target = np.concatenate(
        [np.ones(len(pos_nullary)), np.zeros(len(neg_nullary)) - 1], 0
    )  # (B',)
    data = {"nullary": nullary, "unary": unary, "binary": binary, "target": target}
    # ---------------------------
    # Remove any duplicates in the data
    flat_in = flatten_interpretation(nullary, unary, binary)  # (B', IN)
    _, unique_idxs = np.unique(flat_in, return_index=True, axis=0)  # (<B',)
    data = {k: v[unique_idxs] for k, v in data.items()}
    data["and_kernel"] = and_kernel
    data["or_kernel"] = or_kernel
    data["num_vars"] = numVars
    data["num_objects"] = numObjs
    logger.info("Managed to generate %i many unique examples", data["nullary"].shape[0])
    # ---------------------------
    # We're done, save the file
    logger.info("Creating %s with keys: %s", str(fpath), str(data.keys()))
    np.savez_compressed(fpath, **data)
    return str(fpath)


def load_data() -> Tuple[  # pylint: disable=too-many-locals
    Dict[str, Any], Dict[str, tf.data.Dataset]
]:
    """Load and process relations game dataset."""
    fpath = generate_data()
    # ---------------------------
    # Load the dataset file
    dnpz = np.load(fpath)  # {'nullary': ..., ...}
    # ---------------------------
    # Generate train, validation and test splits
    rng = np.random.default_rng(seed=C["gendnf_rng_seed"])
    data_size = dnpz["nullary"].shape[0]  # D
    idxs = np.arange(data_size)  # (D,)
    rng.shuffle(idxs)  # (D,)
    train_size, val_size, test_size = (
        C["gendnf_train_size"],
        C["gendnf_validation_size"],
        C["gendnf_test_size"],
    )
    required_size = train_size + val_size + test_size
    assert (
        required_size <= data_size
    ), f"Need {required_size} examples but have {data_size}."
    dsetidxs = {
        "train": idxs[:train_size],
        "validation": idxs[train_size : train_size + val_size],
        "test": idxs[train_size + val_size : required_size],
    }
    # ---------------------------
    # Curate the tf.data.Dataset
    dsets: Dict[str, tf.data.Dataset] = dict()
    for dname, didxs in dsetidxs.items():
        input_dict = {
            a: dnpz[a][didxs].astype(np.float32) for a in ("nullary", "unary", "binary")
        }
        # {'nullary': (tsize,), 'binary': ...}
        label = dnpz["target"][didxs]
        # Optionally add noise to labels by flipping them
        if C["gendnf_noise_stddev"]:
            noise_mask = rng.choice(
                [-1, 1],
                size=label.size,
                p=[C["gendnf_noise_stddev"], 1 - C["gendnf_noise_stddev"]],
            )
            label *= noise_mask
        label = (label == 1).astype(np.int32)  # convert back to 0, 1 labels
        output_dict = {"label": label}
        tfdata = tf.data.Dataset.from_tensor_slices((input_dict, output_dict))
        if dname == "train":
            tfdata = tfdata.shuffle(1000).batch(C["gendnf_batch_size"]).repeat()
        else:
            tfdata = tfdata.batch(C["gendnf_batch_size"])
        dsets[dname] = tfdata
    # ---------------------------
    # Generate description
    inputs = {
        k: {"shape": tuple(v.shape), "dtype": v.dtype}
        for k, v in dsets["train"].element_spec[0].items()
    }
    output_spec = dsets["train"].element_spec[1]
    outputs = {
        "label": {
            "shape": tuple(output_spec["label"].shape),
            "dtype": output_spec["label"].dtype,
            "num_categories": 1,
            "type": "multilabel",
            "target_rules": [0],  # 1 predicate to learn with arity 0
        }
    }
    description = {
        "name": "gendnf",
        "inputs": inputs,
        "outputs": outputs,
        "datasets": list(dsets.keys()),
    }
    # ---------------------------
    return description, dsets
