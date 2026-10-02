"""Small, synchronous API for model-independent inference exchange."""

from .backend import Backend, Payload
from .client import Client
from .errors import Error, InvalidInput, ProtocolError, RemoteError, RequestTimeout, TransportError
from .server import serve
from .specs import (
    Channel,
    Dimension,
    InferenceSpec,
    ObjectSpec,
    ScalarSpec,
    SpecError,
    TensorSpec,
    check_compatibility,
)

__all__ = [
    "Backend",
    "Channel",
    "Dimension",
    "InferenceSpec",
    "ObjectSpec",
    "ScalarSpec",
    "SpecError",
    "TensorSpec",
    "check_compatibility",
    "Client",
    "Error",
    "InvalidInput",
    "Payload",
    "ProtocolError",
    "RemoteError",
    "RequestTimeout",
    "TransportError",
    "serve",
]
__version__ = "0.2.0"
