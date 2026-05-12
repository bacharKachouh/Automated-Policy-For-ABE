"""
Charm-Crypto serialisation helpers.

Round-trips group elements / nested dicts of group elements through bytes
using Charm's native objectToBytes / bytesToObject. Replaces the broken
str(...) JSON dumps that previously lived in src/abe/authority.py.
"""
from charm.core.engine.util import objectToBytes, bytesToObject


def serialize_element(group, element):
    """Serialise a single Charm group element to bytes."""
    return group.serialize(element)


def deserialize_element(group, blob):
    """Deserialise a single Charm group element from bytes."""
    return group.deserialize(blob)


def serialize_obj(group, obj):
    """Serialise an arbitrary Charm-aware object (dict of elements, ciphertext, etc.)."""
    return objectToBytes(obj, group)


def deserialize_obj(group, blob):
    """Inverse of serialize_obj."""
    return bytesToObject(blob, group)
