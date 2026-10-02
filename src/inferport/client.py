"""Synchronous facade with a dedicated, bounded-lifetime network loop."""

import asyncio
import concurrent.futures
import threading
from ssl import SSLContext

from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus, NegotiationError, WebSocketException

from . import codec, protocol
from ._io import CLOSE_TIMEOUT, ClientSocket, positive_timeout, validate_token
from .backend import Payload
from .errors import ProtocolError, RemoteError, RequestTimeout, TransportError
from .specs import InferenceSpec, SpecError


class Client:
    def __init__(
        self,
        uri: str,
        *,
        timeout: float = 30.0,
        open_timeout: float = 10.0,
        max_message_bytes: int = codec.MAX_MESSAGE_BYTES,
        token: str | None = None,
        ssl: SSLContext | None = None,
    ):
        self._uri = uri
        self._timeout = positive_timeout(timeout, "timeout")
        self._open_timeout = positive_timeout(open_timeout, "open_timeout")
        self._limit = codec.message_limit(max_message_bytes)
        validate_token(token)
        self._token, self._ssl = token, ssl
        self._call_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._connected = self._closed = False
        self._loop = self._thread = self._ws = self._active = None
        self._id = 0
        self._spec: InferenceSpec

    def _start(self):
        self._loop = asyncio.new_event_loop()

        def run():
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_forever()
            finally:
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
                self._loop.close()

        self._thread = threading.Thread(target=run, name="inferport-client", daemon=True)
        self._thread.start()

    def _submit(self, coroutine):
        with self._state_lock:
            if self._closed:
                coroutine.close()
                raise TransportError("Client was closed")
            self._active = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
            future = self._active
        try:
            return future.result()
        except concurrent.futures.CancelledError as exc:
            raise TransportError("Client was closed during exchange") from exc
        finally:
            with self._state_lock:
                if self._active is future:
                    self._active = None

    async def _open(self):
        stage = "open"

        async def opening():
            nonlocal stage
            kwargs = {"ssl": self._ssl} if self._ssl is not None else {}
            self._ws = await connect(
                self._uri,
                subprotocols=[protocol.SUBPROTOCOL],
                additional_headers={"Authorization": f"Bearer {self._token}"}
                if self._token
                else None,
                compression=None,
                proxy=None,
                max_size=self._limit,
                max_queue=1,
                ping_interval=20,
                ping_timeout=20,
                open_timeout=self._open_timeout,
                close_timeout=CLOSE_TIMEOUT,
                create_connection=ClientSocket,
                **kwargs,
            )
            if self._ws.subprotocol != protocol.SUBPROTOCOL:
                raise ProtocolError("Server didn't negotiate inferport.v2")
            stage = "ready"
            raw = await self._ws.recv()
            protocol.response(codec.decode(raw, self._limit), 0, empty=True)
            stage = "describe"
            await self._ws.send(codec.encode({"id": 1, "op": "describe", "data": {}}, self._limit))
            result = protocol.response(codec.decode(await self._ws.recv(), self._limit), 1)
            try:
                if set(result) != {"spec"}:
                    raise SpecError("Malformed description response")
                return InferenceSpec.from_dict(result["spec"])
            except SpecError as exc:
                raise ProtocolError("Invalid inference specification") from exc

        try:
            return await asyncio.wait_for(opening(), self._open_timeout)
        except asyncio.TimeoutError as exc:
            raise RequestTimeout(stage) from exc
        except InvalidStatus as exc:
            if exc.response.status_code == 401:
                raise RemoteError("unauthorized", "Authentication failed", None, True) from exc
            if exc.response.headers.get_all("InferPort-Error") == ["protocol_error"]:
                raise ProtocolError("Server rejected the protocol version") from exc
            raise TransportError("WebSocket handshake rejected") from exc
        except NegotiationError as exc:
            raise ProtocolError("WebSocket subprotocol negotiation failed") from exc
        except (OSError, WebSocketException, ValueError) as exc:
            raise TransportError("Unable to connect to server") from exc

    def connect(self) -> "Client":
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError("Concurrent Client calls aren't supported")
        try:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("Create a new Client after closing or failure")
                if self._connected:
                    return self
                self._start()
            try:
                self._spec = self._submit(self._open())
                with self._state_lock:
                    if self._closed:
                        raise TransportError("Client was closed while connecting")
                    self._id = 1
                    self._connected = True
                return self
            except BaseException:
                self._dispose(abort=True)
                raise
        finally:
            self._call_lock.release()

    async def _exchange(self, data: bytes, timeout: float):
        stage = "send"

        async def exchanging():
            nonlocal stage
            await self._ws.send(data)
            stage = "response"
            return await self._ws.recv()

        try:
            return await asyncio.wait_for(exchanging(), timeout)
        except asyncio.TimeoutError as exc:
            self._ws.abort()
            raise RequestTimeout(stage) from exc
        except (OSError, WebSocketException) as exc:
            raise TransportError("Connection interrupted during exchange") from exc

    def _call(self, op: str, data: Payload, timeout: float | None) -> Payload:
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError("Concurrent Client calls aren't supported")
        try:
            with self._state_lock:
                if not self._connected or self._closed:
                    raise RuntimeError("Client must be connected and usable")
            deadline = self._timeout if timeout is None else positive_timeout(timeout, "timeout")
            if not isinstance(data, dict):
                raise TypeError("payload root must be a dictionary")
            if self._id == protocol.MAX_ID:
                self._dispose(abort=False)
                raise RuntimeError("Request IDs exhausted; create a new Client")
            request_id = self._id + 1
            # Local validation failures leave both the connection and ID unchanged.
            try:
                encoded = codec.encode({"id": request_id, "op": op, "data": data}, self._limit)
            except KeyboardInterrupt:
                self._dispose(abort=True)
                raise
            self._id = request_id
            try:
                raw = self._submit(self._exchange(encoded, deadline))
                result = protocol.response(
                    codec.decode(raw, self._limit), request_id, empty=op == "reset"
                )
                if op == "infer":
                    try:
                        self._spec.outputs.validate(result, "outputs")
                    except SpecError as exc:
                        raise ProtocolError(
                            f"Response violates the output contract: {exc}"
                        ) from exc
                return result
            except RemoteError as exc:
                if exc.fatal:
                    self._dispose(abort=True)
                raise
            except BaseException:
                self._dispose(abort=True)
                raise
        finally:
            self._call_lock.release()

    def describe(self) -> InferenceSpec:
        """Read the immutable contract cached by connect; no network request."""
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError("Concurrent Client calls aren't supported")
        try:
            with self._state_lock:
                if not self._connected or self._closed:
                    raise RuntimeError("Client must be connected and usable")
                return self._spec
        finally:
            self._call_lock.release()

    def infer(self, inputs: Payload, *, timeout: float | None = None) -> Payload:
        return self._call("infer", inputs, timeout)

    def reset(self, context: Payload | None = None, *, timeout: float | None = None) -> None:
        self._call("reset", {} if context is None else context, timeout)

    async def _shutdown(self, abort: bool):
        tasks = asyncio.all_tasks() - {asyncio.current_task()}
        # Cancellation also interrupts an unfinished HTTP handshake.
        for task in tasks:
            task.cancel()
        if self._ws is not None:
            if abort:
                self._ws.abort()
            else:
                await self._ws.close()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _dispose(self, *, abort: bool):
        with self._close_lock:
            with self._state_lock:
                if self._closed:
                    return
                self._closed = True
                self._connected = False
                loop, thread = self._loop, self._thread
                abort = abort or (self._active is not None and not self._active.done())
            if loop is not None:
                try:
                    asyncio.run_coroutine_threadsafe(self._shutdown(abort), loop).result()
                finally:
                    loop.call_soon_threadsafe(loop.stop)
                    thread.join()

    def close(self) -> None:
        self._dispose(abort=False)

    def __enter__(self) -> "Client":
        return self.connect()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
