"""Bounded MessagePack with a single, explicitly specified ndarray extension."""

import math
from typing import Any

import msgpack
import numpy as np

from .errors import ProtocolError

MAX_MESSAGE_BYTES = 64 * 1024 * 1024
MAX_DEPTH = 32
MAX_NODES = 100_000
WIRE_DTYPES = frozenset(
    ("|b1", "|i1", "|u1", "<i2", "<u2", "<i4", "<u4", "<i8", "<u8", "<f2", "<f4", "<f8")
)


def message_limit(value: int) -> int:
    if type(value) is not int or value < 128:
        raise ValueError("max_message_bytes must be an integer of at least 128")
    return value


def _shape(shape: Any) -> None:
    if not isinstance(shape, list) or len(shape) > 32:
        raise ValueError("array shape must be a list with at most 32 dimensions")
    if any(type(n) is not int or not 0 <= n <= 2**31 - 1 for n in shape):
        raise ValueError("invalid array dimension")


def _pairs(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if not isinstance(key, str) or key in result:
            raise ValueError("map keys must be unique strings")
        result[key] = value
    return result


def _unpack(raw: bytes, limit: int, ext_hook):
    return msgpack.unpackb(
        raw,
        raw=False,
        strict_map_key=True,
        object_pairs_hook=_pairs,
        ext_hook=ext_hook,
        max_str_len=limit,
        max_bin_len=limit,
        max_ext_len=limit,
        max_array_len=MAX_NODES,
        max_map_len=MAX_NODES,
    )


def _walk(value, *, encode: bool, limit: int, depth: int = 0, count=None):
    if count is None:
        count = [0]
    count[0] += 1
    if count[0] > MAX_NODES or depth > MAX_DEPTH:
        raise ValueError("payload exceeds structural limits")
    if isinstance(value, np.ndarray):
        if not encode:
            return value
        dtype = value.dtype.newbyteorder("<").str
        if dtype not in WIRE_DTYPES:
            raise TypeError("unsupported array dtype")
        shape = list(value.shape)
        _shape(shape)
        if value.nbytes > limit:
            raise ValueError("array exceeds message limit")
        raw = value.astype(np.dtype(dtype), copy=False).tobytes(order="C")
        return msgpack.ExtType(1, msgpack.packb([dtype, shape, raw], use_bin_type=True))
    if isinstance(value, np.generic):
        if value.dtype.kind not in "biuf" or value.dtype.itemsize > 8:
            raise TypeError("unsupported NumPy scalar")
        value = value.item()
    if value is None or type(value) in (bool, float):
        return value
    if type(value) is int:
        if not -(2**63) <= value < 2**64:
            raise ValueError("integer outside MessagePack range")
        return value
    if isinstance(value, (str, bytes)):
        if len(value) > limit:
            raise ValueError("value exceeds message limit")
        return value
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("map keys must be strings")
        return {
            key: _walk(item, encode=encode, limit=limit, depth=depth + 1, count=count)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_walk(v, encode=encode, limit=limit, depth=depth + 1, count=count) for v in value]
    raise TypeError("unsupported payload value")


def encode(message: dict, max_bytes: int = MAX_MESSAGE_BYTES) -> bytes:
    """Encode a string-keyed dict; reject unsupported local values before sending."""
    if not isinstance(message, dict):
        raise TypeError("payload root must be a dictionary")
    data = msgpack.packb(_walk(message, encode=True, limit=max_bytes), use_bin_type=True)
    if len(data) > max_bytes:
        raise ValueError("encoded message exceeds message limit")
    return data


def decode(data: bytes, max_bytes: int = MAX_MESSAGE_BYTES) -> dict:
    """Return native-endian, writable C arrays; never reconstruct Python objects."""

    def reject_ext(code, raw):
        raise ValueError("nested array extension is invalid")

    def array_ext(code, raw):
        if code != 1:
            raise ValueError("unknown extension type")
        parts = _unpack(raw, max_bytes, reject_ext)
        if not isinstance(parts, list) or len(parts) != 3:
            raise ValueError("invalid array extension")
        dtype, shape, body = parts
        if not isinstance(dtype, str) or dtype not in WIRE_DTYPES:
            raise ValueError("invalid wire dtype")
        _shape(shape)
        dt = np.dtype(dtype)
        if not isinstance(body, bytes) or len(body) != math.prod(shape) * dt.itemsize:
            raise ValueError("array shape and byte length disagree")
        return np.frombuffer(body, dtype=dt).reshape(shape).astype(dt.newbyteorder("="), copy=True)

    try:
        if not isinstance(data, bytes) or len(data) > max_bytes:
            raise ValueError("expected a binary message within size limit")
        result = _unpack(data, max_bytes, array_ext)
        if not isinstance(result, dict):
            raise ValueError("message root must be a dictionary")
        return _walk(result, encode=False, limit=max_bytes)
    except (ValueError, TypeError, OverflowError, RecursionError, msgpack.UnpackException) as exc:
        raise ProtocolError("Invalid MessagePack payload") from exc
