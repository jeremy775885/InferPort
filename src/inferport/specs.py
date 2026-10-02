"""Portable data contracts. Validation never converts or normalizes payloads."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from .errors import Error


class SpecError(ValueError, Error):
    """A specification, payload, or pair of contracts is incompatible."""


def _require(condition, message):
    if not condition:
        raise SpecError(message)


@dataclass(frozen=True)
class Dimension:
    minimum: int = 1
    maximum: int = 4096

    def __post_init__(self):
        _require(
            type(self.minimum) is int
            and type(self.maximum) is int
            and 0 <= self.minimum <= self.maximum,
            "Invalid dimension bounds",
        )


@dataclass(frozen=True)
class Channel:
    name: str
    semantic: str
    unit: str

    def __post_init__(self):
        _require(
            all(isinstance(v, str) and v for v in (self.name, self.semantic, self.unit)),
            "Channel name, semantic and unit must be nonempty strings",
        )


@dataclass(frozen=True)
class TensorSpec:
    dtype: str
    shape: tuple[int | Dimension, ...]
    semantic: str = ""
    axes: tuple[str, ...] = ()
    channels: tuple[Channel, ...] = ()
    finite: bool = True

    def __post_init__(self):
        try:
            dtype = np.dtype(self.dtype)
        except (TypeError, ValueError) as exc:
            raise SpecError("Invalid tensor dtype") from exc
        _require(
            isinstance(self.dtype, str)
            and dtype.name == self.dtype
            and dtype.kind in "biuf"
            and dtype.itemsize in (1, 2, 4, 8),
            "Use a supported canonical numeric dtype",
        )
        _require(
            isinstance(self.shape, (tuple, list)) and len(self.shape) <= 32, "Invalid tensor rank"
        )
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "axes", tuple(self.axes))
        object.__setattr__(self, "channels", tuple(self.channels))
        _require(
            all(isinstance(d, Dimension) or type(d) is int and d >= 0 for d in self.shape),
            "Dimensions must be nonnegative integers or bounded Dimension values",
        )
        _require(
            isinstance(self.semantic, str) and type(self.finite) is bool,
            "Invalid tensor semantic/finite fields",
        )
        _require(
            not self.axes
            or len(self.axes) == len(self.shape)
            and all(isinstance(a, str) and a for a in self.axes),
            "Invalid tensor axes",
        )
        _require(all(isinstance(c, Channel) for c in self.channels), "Invalid channels")
        _require(
            not self.channels or self.shape and self.shape[-1] == len(self.channels),
            "Channels must describe the fixed final tensor dimension",
        )
        _require(
            len({c.name for c in self.channels}) == len(self.channels), "Duplicate channel names"
        )

    def validate(self, value, path="value"):
        _require(
            isinstance(value, np.ndarray) and value.dtype.name == self.dtype,
            f"{path}: expected ndarray {self.dtype}",
        )
        _require(value.ndim == len(self.shape), f"{path}: incorrect rank")
        for actual, declared in zip(value.shape, self.shape, strict=True):
            low, high = _bounds(declared)
            _require(low <= actual <= high, f"{path}: shape {value.shape} violates {self.shape}")
        _require(not self.finite or np.isfinite(value).all(), f"{path}: nonfinite values")

    def to_dict(self):
        return dict(
            kind="tensor",
            dtype=self.dtype,
            shape=[
                dict(minimum=d.minimum, maximum=d.maximum) if isinstance(d, Dimension) else d
                for d in self.shape
            ],
            semantic=self.semantic,
            axes=list(self.axes),
            channels=[dict(name=c.name, semantic=c.semantic, unit=c.unit) for c in self.channels],
            finite=self.finite,
        )


@dataclass(frozen=True)
class ScalarSpec:
    kind: str
    minimum: float | int | None = None
    maximum: float | int | None = None
    nonempty: bool = False

    def __post_init__(self):
        _require(self.kind in ("string", "integer", "number", "boolean"), "Invalid scalar kind")
        _require(
            type(self.nonempty) is bool and (not self.nonempty or self.kind == "string"),
            "nonempty applies only to strings",
        )
        for bound in (self.minimum, self.maximum):
            _require(
                bound is None
                or self.kind in ("integer", "number")
                and type(bound) in (float, int)
                and math.isfinite(bound),
                "Invalid scalar bounds",
            )
        _require(
            self.minimum is None or self.maximum is None or self.minimum <= self.maximum,
            "Reversed scalar bounds",
        )

    def validate(self, value, path="value"):
        if isinstance(value, np.generic):
            _require(
                value.dtype.kind in "biuf" and value.dtype.itemsize <= 8,
                f"{path}: unsupported NumPy scalar",
            )
            # Check the codec's scalar representation without replacing payload values.
            value = value.item()
        valid = {
            "string": isinstance(value, str),
            "integer": type(value) is int,
            "number": type(value) in (int, float),
            "boolean": type(value) is bool,
        }[self.kind]
        _require(valid, f"{path}: expected {self.kind}")
        if self.kind in ("integer", "number"):
            _require(math.isfinite(value), f"{path}: nonfinite number")
            _require(self.minimum is None or value >= self.minimum, f"{path}: below minimum")
            _require(self.maximum is None or value <= self.maximum, f"{path}: above maximum")
        if self.nonempty:
            _require(bool(value.strip()), f"{path}: empty string")

    def to_dict(self):
        return dict(
            kind=self.kind, minimum=self.minimum, maximum=self.maximum, nonempty=self.nonempty
        )


@dataclass(frozen=True)
class ObjectSpec:
    fields: Mapping[str, TensorSpec | ScalarSpec | ObjectSpec] = field(default_factory=dict)
    optional: tuple[str, ...] = ()

    def __post_init__(self):
        _require(
            isinstance(self.fields, Mapping) and len(self.fields) <= 256, "Invalid object fields"
        )
        _require(
            all(
                isinstance(k, str) and k and isinstance(v, (TensorSpec, ScalarSpec, ObjectSpec))
                for k, v in self.fields.items()
            ),
            "Invalid object field",
        )
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))
        object.__setattr__(self, "optional", tuple(self.optional))
        _require(
            all(isinstance(k, str) for k in self.optional)
            and len(set(self.optional)) == len(self.optional)
            and set(self.optional) <= self.fields.keys(),
            "Invalid optional fields",
        )

    def validate(self, value, path="value"):
        _require(isinstance(value, dict), f"{path}: expected dictionary")
        _require(
            self.fields.keys() - set(self.optional) <= value.keys()
            and value.keys() <= self.fields.keys(),
            f"{path}: missing or unknown fields",
        )
        for key, item in value.items():
            self.fields[key].validate(item, f"{path}.{key}")

    def to_dict(self):
        return dict(
            kind="object",
            fields={k: v.to_dict() for k, v in self.fields.items()},
            optional=list(self.optional),
        )


def _bounds(dimension):
    return (
        (dimension.minimum, dimension.maximum)
        if isinstance(dimension, Dimension)
        else (dimension, dimension)
    )


def _parse(data, depth=0):
    _require(depth <= 16 and isinstance(data, dict), "Invalid/deep specification")
    kind = data.get("kind")
    if kind == "tensor":
        _require(
            set(data) == {"kind", "dtype", "shape", "semantic", "axes", "channels", "finite"},
            "Invalid tensor spec fields",
        )
        _require(
            isinstance(data["shape"], list)
            and isinstance(data["axes"], list)
            and isinstance(data["channels"], list),
            "Invalid tensor spec lists",
        )
        return TensorSpec(
            data["dtype"],
            tuple(Dimension(**d) if isinstance(d, dict) else d for d in data["shape"]),
            data["semantic"],
            tuple(data["axes"]),
            tuple(Channel(**c) for c in data["channels"]),
            data["finite"],
        )
    if kind == "object":
        _require(
            set(data) == {"kind", "fields", "optional"}
            and isinstance(data["fields"], dict)
            and isinstance(data["optional"], list),
            "Invalid object spec fields",
        )
        return ObjectSpec(
            {k: _parse(v, depth + 1) for k, v in data["fields"].items()}, tuple(data["optional"])
        )
    _require(set(data) == {"kind", "minimum", "maximum", "nonempty"}, "Invalid scalar spec fields")
    return ScalarSpec(**data)


@dataclass(frozen=True)
class InferenceSpec:
    inputs: ObjectSpec
    outputs: ObjectSpec
    context: ObjectSpec = field(default_factory=ObjectSpec)
    profile: str = ""

    def __post_init__(self):
        _require(
            all(isinstance(v, ObjectSpec) for v in (self.inputs, self.outputs, self.context))
            and isinstance(self.profile, str),
            "Invalid inference specification",
        )

    def to_dict(self):
        return dict(
            version=1,
            profile=self.profile,
            inputs=self.inputs.to_dict(),
            outputs=self.outputs.to_dict(),
            context=self.context.to_dict(),
        )

    @classmethod
    def from_dict(cls, data):
        try:
            _require(
                isinstance(data, dict)
                and set(data) == {"version", "profile", "inputs", "outputs", "context"}
                and type(data["version"]) is int
                and data["version"] == 1,
                "Unsupported inference specification",
            )
            return cls(
                _parse(data["inputs"]),
                _parse(data["outputs"]),
                _parse(data["context"]),
                data["profile"],
            )
        except (TypeError, KeyError, AttributeError, RecursionError) as exc:
            raise SpecError("Malformed inference specification") from exc


def _accepts(receiver, sender, path):
    """Every value allowed by sender must be accepted by receiver."""
    _require(type(receiver) is type(sender), f"{path}: incompatible field types")
    if isinstance(receiver, ObjectSpec):
        _require(
            receiver.fields.keys() - set(receiver.optional)
            <= sender.fields.keys() - set(sender.optional),
            f"{path}: required fields aren't guaranteed",
        )
        _require(sender.fields.keys() <= receiver.fields.keys(), f"{path}: unknown fields")
        for key in sender.fields:
            _accepts(receiver.fields[key], sender.fields[key], f"{path}.{key}")
    elif isinstance(receiver, TensorSpec):
        _require(
            (receiver.dtype, receiver.semantic, receiver.axes, receiver.channels)
            == (sender.dtype, sender.semantic, sender.axes, sender.channels),
            f"{path}: dtype, semantic, axes or channel ordering/units differ",
        )
        _require(not receiver.finite or sender.finite, f"{path}: finite values aren't guaranteed")
        _require(len(receiver.shape) == len(sender.shape), f"{path}: tensor ranks differ")
        for accepted, offered in zip(receiver.shape, sender.shape, strict=True):
            a, b = _bounds(accepted)
            c, d = _bounds(offered)
            _require(a <= c <= d <= b, f"{path}: incompatible dimension bounds")
    else:
        _require(
            receiver.kind == sender.kind and (not receiver.nonempty or sender.nonempty),
            f"{path}: scalar kinds/constraints differ",
        )
        _require(
            receiver.minimum is None
            or sender.minimum is not None
            and sender.minimum >= receiver.minimum,
            f"{path}: incompatible lower bound",
        )
        _require(
            receiver.maximum is None
            or sender.maximum is not None
            and sender.maximum <= receiver.maximum,
            f"{path}: incompatible upper bound",
        )


def check_compatibility(service: InferenceSpec, execution: InferenceSpec) -> None:
    """Execution inputs/context are produced locally; service outputs arrive remotely."""
    _require(
        isinstance(service, InferenceSpec) and isinstance(execution, InferenceSpec),
        "Both endpoints must declare an inference specification",
    )
    _require(service.profile == execution.profile, "Incompatible inference profiles")
    _accepts(service.inputs, execution.inputs, "inputs")
    _accepts(service.context, execution.context, "context")
    _accepts(execution.outputs, service.outputs, "outputs")
