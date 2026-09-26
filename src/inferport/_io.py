"""Private transport lifetime helpers using public asyncio/websockets hooks."""

import asyncio
import math

from websockets.asyncio.client import ClientConnection
from websockets.asyncio.server import ServerConnection

CLOSE_TIMEOUT = 5.0


def positive_timeout(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def validate_token(token: str | None) -> None:
    if token is not None and (
        not isinstance(token, str) or not token or any(not 33 <= ord(c) <= 126 for c in token)
    ):
        raise ValueError("token must be a nonempty printable ASCII string without whitespace")


class _BoundedClose:
    """Include drain in the deadline, including close calls made by websockets."""

    def connection_made(self, transport):
        self._owned_transport = transport
        super().connection_made(transport)

    def abort(self):
        transport = getattr(self, "_owned_transport", None)
        if transport is not None:
            transport.abort()

    async def close(self, code=1000, reason=""):
        task = asyncio.create_task(super().close(code, reason))
        try:
            done, _ = await asyncio.wait({task}, timeout=CLOSE_TIMEOUT)
            if not done:
                self.abort()
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        except BaseException:
            self.abort()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise


class ClientSocket(_BoundedClose, ClientConnection):
    pass


class ServerSocket(_BoundedClose, ServerConnection):
    pass
