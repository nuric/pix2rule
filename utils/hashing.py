"""Hashing functions used through the project."""
from typing import Dict, Any, List
import hashlib
import json


def list_hash(hlist: List[Any]) -> str:
    """MD5 hash of list."""
    lhash = hashlib.md5()
    encoded = json.dumps(hlist, sort_keys=True).encode()
    lhash.update(encoded)
    return lhash.hexdigest()


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()
