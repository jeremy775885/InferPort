"""Small, synchronous API for model-independent inference exchange."""

from .backend import Backend, Payload
from .client import Client
from .errors import Error, InvalidInput, ProtocolError, RemoteError, RequestTimeout, TransportError
from .server import serve

__all__ = [
    "Backend",
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
__version__ = "0.1.0"
