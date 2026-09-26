import msgpack
import numpy as np
import pytest

from inferport import ProtocolError
from inferport.codec import MAX_NODES, WIRE_DTYPES, decode, encode


@pytest.mark.parametrize("dtype", sorted(WIRE_DTYPES))
@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (2, 3)])
def test_all_arrays(dtype, shape):
    original = np.ones(shape, dtype=dtype)
    result = decode(encode({"x": original}))["x"]
    np.testing.assert_array_equal(result, original)
    assert result.dtype == original.dtype.newbyteorder("=")
    assert result.flags.writeable and result.flags.c_contiguous and result.dtype.isnative
    assert type(result) is np.ndarray


def test_endian_slicing_and_scalars():
    array = np.arange(120, dtype=">f4").reshape(10, 12)[::-2, ::3]
    values = [None, True, "中文", b"bytes", -(2**63), 2**64 - 1, np.int32(2), np.float32(3)]
    result = decode(encode({"x": array, "values": tuple(values), "bool": np.bool_(True)}))
    np.testing.assert_array_equal(result["x"], array)
    assert result["x"].flags.writeable and result["x"].flags.c_contiguous
    assert result["values"] == values
    assert type(result["values"][-1]) is float
    assert type(result["values"][-2]) is int
    assert result["bool"] is True
    special = np.array([np.nan, np.inf, -np.inf])
    np.testing.assert_array_equal(decode(encode({"x": special}))["x"], special)


@pytest.mark.parametrize(
    "value", [object(), 2**64, -(2**63) - 1, {1: 2}, complex(1), np.complex64(1)]
)
def test_reject_unsupported_values(value):
    with pytest.raises((ValueError, TypeError)):
        encode({"x": value})


@pytest.mark.parametrize("dtype", ["O", "U1", "S1", "c8", "m8[s]", "M8[s]", [("x", "i4")]])
def test_reject_unsupported_arrays(dtype):
    with pytest.raises(TypeError):
        encode({"x": np.zeros(1, dtype=dtype)})


def extension(parts, code=1):
    return msgpack.packb({"x": msgpack.ExtType(code, msgpack.packb(parts, use_bin_type=True))})


@pytest.mark.parametrize(
    "parts",
    [
        ["<f4", [-1], b""],
        ["<f4", [True], b"1234"],
        ["<f4", [1], b"12345"],
        ["<f4", [1], b"123"],
        ["f4", [1], b"1234"],
        ["<c8", [0], b""],
        ["<f4", [], b""],
        ["<f4", [2**31], b""],
        ["<f4", [0] * 33, b""],
        ["<f4", [0], ""],
        ["<f4", {}, b""],
        [[], [], b""],
        [],
    ],
)
def test_reject_bad_array_extensions(parts):
    with pytest.raises(ProtocolError):
        decode(extension(parts))


@pytest.mark.parametrize(
    "raw",
    [
        b"\x82\xa1x\x01\xa1x\x02",
        b"\x81\x01\x02",
        b"\xc1",
        b"\x80\x80",
        msgpack.packb({"x": msgpack.ExtType(3, b"data")}),
        msgpack.packb({"x": msgpack.Timestamp(1)}),
        b"\x81\xa1x\xdb\xff\xff\xff\xff",
        "text frame",
        b"\x90",
    ],
)
def test_reject_bad_messages(raw):
    with pytest.raises(ProtocolError):
        decode(raw)


def test_limits_and_no_marker_collision():
    payload = {"__ndarray__": True, "dtype": "f4", "shape": [1], "data": b"1234"}
    assert decode(encode(payload)) == payload
    with pytest.raises(ValueError):
        encode({"x": np.zeros(512)}, 128)
    with pytest.raises(ProtocolError):
        decode(encode({"x": "x" * 128}), 128)
    with pytest.raises(ValueError):
        encode({"x": [None] * MAX_NODES})
    with pytest.raises(ProtocolError):
        decode(msgpack.packb({"x": [None] * MAX_NODES}))
    nested = {}
    for _ in range(33):
        nested = {"x": nested}
    with pytest.raises(ValueError):
        encode(nested)
    with pytest.raises(ProtocolError):
        decode(msgpack.packb(nested))
