"""Thin wrapper around command line clingo."""
from typing import Dict, List
import itertools
import subprocess
import os

import numpy as np


def clingo_check(
    interpretation: Dict[str, np.ndarray], rule: Dict[str, np.ndarray]
) -> np.ndarray:
    """Generates logic program files and runs them with clingo to check satisfiability."""
    # interpretation {'nullary': (B, P0), 'unary': (B, O, P1), 'binary': (B, O, O-1, P2)}
    # rule {'and_kernel': (H, IN), 'or_kernel': (H,), 'num_variables': int}
    # ---------------------------
    num_nullary = interpretation["nullary"].shape[-1]
    num_unary = interpretation["unary"].shape[-1]
    num_binary = interpretation["binary"].shape[-1]
    num_variables = rule["num_variables"]
    # ---------------------------
    var_bidxs = np.stack(np.nonzero(1 - np.eye(num_variables))).T  # (V*(V-1), 2)
    pred_names = (
        [f"n{i}" for i in range(num_nullary)]
        + [
            f"p{j}(V{i})"
            for i, j in itertools.product(range(num_variables), range(num_unary))
        ]
        + [
            f"q{k}(V{i},V{j})"  # There is this modulus business because we omit p(X,X)
            for (i, j), k in itertools.product(var_bidxs, range(num_binary))
        ]
    )  # This is a mess that produces "n0..n, p1..n(V1), p1...n(V2), etc"
    # Here is a full dump for our sanity:
    # ['n0', 'n1', 'n2', 'n3', 'p0(V0)', 'p1(V0)', 'p2(V0)', 'p3(V0)',
    # 'p0(V1)', 'p1(V1)', 'p2(V1)', 'p3(V1)', 'p0(V2)', 'p1(V2)', 'p2(V2)',
    # 'p3(V2)', 'q0(V0,V1)', 'q1(V0,V1)', 'q2(V0,V1)', 'q3(V0,V1)',
    # 'q0(V0,V2)', 'q1(V0,V2)', 'q2(V0,V2)', 'q3(V0,V2)', 'q0(V1,V0)',
    # 'q1(V1,V0)', 'q2(V1,V0)', 'q3(V1,V0)', 'q0(V1,V2)', 'q1(V1,V2)',
    # 'q2(V1,V2)', 'q3(V1,V2)', 'q0(V2,V0)', 'q1(V2,V0)', 'q2(V2,V0)',
    # 'q3(V2,V0)', 'q0(V2,V1)', 'q1(V2,V1)', 'q2(V2,V1)', 'q3(V2,V1)']
    # ---------------------------
    # target rule must be satisfied, to get satisfiable output
    rule_lines: List[str] = [":- not t."]
    # ---------------------------
    # Construct the rule
    conjuncts: List[str] = list()
    for conjunct in rule["and_kernel"]:
        cstr = ", ".join(
            [
                n if i == 1 else "not " + n
                for n, i in zip(pred_names, conjunct)
                if i != 0
            ]
        )  # ['n0', '-p0(V1)', ...]
        conjuncts.append(cstr)
    # Target rule t is now defined according to which conjuncts are in the rule
    assert (
        rule["or_kernel"].min() == 0
    ), "Clingo cannot work with negated conjunctions inside a disjunction."
    # or_kernel [0, 1, 0, 0, ...] x H
    disjuncts = [
        "t :- {}.".format(conjuncts[i])
        for i in range(len(conjuncts))
        if rule["or_kernel"][i] != 0
    ]
    rule_lines.extend(disjuncts)
    # ---------------------------
    # That's the rule done, now add the interpretation
    batch_size = interpretation["nullary"].shape[0]  # B
    num_objects = interpretation["unary"].shape[1]  # O
    results: List[bool] = list()
    for bidx in range(batch_size):
        logic_program = rule_lines.copy()
        nullary = interpretation["nullary"][bidx]  # (P0,)
        unary = interpretation["unary"][bidx]  # (O, P1)
        binary = interpretation["binary"][bidx]  # (O, O-1, P2)
        # ---------------------------
        # Add nullary facts.
        logic_program.extend(
            [f"n{i}." if nullary[i] == 1 else "" for i in range(num_nullary)]
        )
        # ---------------------------
        # Add unary facts.
        logic_program.extend(
            [
                f"p{j}(o{i})." if unary[i, j] == 1 else ""
                for i, j in itertools.product(range(num_objects), range(num_unary))
            ]
        )
        # ---------------------------
        # Add binary facts.
        obj_bidxs = np.stack(np.nonzero(1 - np.eye(num_objects))).T  # (O*(O-1), 2)
        logic_program.extend(
            [
                f"q{k}(o{i},o{j})." if binary[i, j - (j >= i), k] == 1 else ""
                for (i, j), k in itertools.product(obj_bidxs, range(num_binary))
            ]
        )
        # ---------------------------
        # Write the program file.
        with open("clingo_temp.lp", "w") as fout:
            fout.write("\n".join(logic_program))
        # ---------------------------
        # Ask clingo the result
        # Clingo for some reason returns exit code 10 on success, we won't have return check
        res = subprocess.run(
            ["clingo", "clingo_temp.lp"], capture_output=True, check=False, text=True
        )
        results.append("UNSATISFIABLE" not in res.stdout)
        # ---------------------------
        # Clean up
        os.remove("clingo_temp.lp")
    # ---------------------------
    return np.array(results)  # (B,)
