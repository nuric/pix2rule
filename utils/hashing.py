"""Hashing functions used through the project."""
from typing import Dict, Any, List, Union, Set
import hashlib
import json


def json_encode_hash(obj: Union[List[Any], Dict[str, Any]]) -> str:
    """Json encode the object and take MD5 hash with sorted keys."""
    obj_hash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(obj, sort_keys=True).encode()
    obj_hash.update(encoded)
    return obj_hash.hexdigest()


def list_hash(hlist: List[Any]) -> str:
    """MD5 hash of list."""
    return json_encode_hash(hlist)


def set_hash(hset: Union[List[Any], Set[Any]]) -> str:
    """MD5 hash of unique list or set."""
    # Set type is not serialasable,
    # convert everything to a sorted list
    # but we can only sort of same comparable types
    ttype = type(next(iter(hset)))
    if not all([isinstance(i, ttype) for i in hset]):
        raise ValueError("Cannot hash set with different types.")
    sorted_list = sorted(hset)
    return list_hash(sorted_list)


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    return json_encode_hash(dictionary)
