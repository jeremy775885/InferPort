"""One connection owns one backend, including during initialization and cleanup."""

import asyncio
import hmac
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from ssl import SSLContext

from websockets.asyncio.server import serve as websocket_serve
from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

from . import codec, protocol
from ._io import CLOSE_TIMEOUT, ServerSocket, positive_timeout, validate_token
from .backend import Backend
from .errors import Error, InvalidInput, ProtocolError
from .specs import InferenceSpec, SpecError

logger = logging.getLogger("inferport.server")


class _Service:
    def __init__(self, backend, executor, limit, send_timeout, token):
        self.backend, self.executor = backend, executor
        self.limit, self.send_timeout, self.token = limit, send_timeout, token
        self.owner = None
        self.stopping = asyncio.Event()
        self.failure = None
        self.spec: InferenceSpec  # Initialized on the backend worker before READY.

    async def work(self, function, *args):
        future = asyncio.get_running_loop().run_in_executor(self.executor, function, *args)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # Never release ownership while a native/GPU operation is still running.
            await asyncio.shield(future)
            raise

    def handshake(self, ws, request):
        if request.path != "/":
            return ws.respond(HTTPStatus.NOT_FOUND, "Unknown endpoint\n")
        if request.headers.get_all("Origin"):
            return ws.respond(HTTPStatus.FORBIDDEN, "Browser origins aren't supported\n")
        authorization = request.headers.get_all("Authorization")
        if self.token is not None and (
            len(authorization) != 1
            or not hmac.compare_digest(authorization[0].encode(), f"Bearer {self.token}".encode())
        ):
            return ws.respond(HTTPStatus.UNAUTHORIZED, "Authentication failed\n")
        offered = [
            part.strip()
            for header in request.headers.get_all("Sec-WebSocket-Protocol")
            for part in header.split(",")
        ]
        if protocol.SUBPROTOCOL not in offered:
            response = ws.respond(HTTPStatus.BAD_REQUEST, "inferport subprotocol is required\n")
            response.headers["InferPort-Error"] = "protocol_error"
            return response
        return None

    async def send(self, ws, payload):
        # Only fixed, small control envelopes come here. In particular, busy must
        # not queue behind the current owner's model work.
        encoded = codec.encode(payload, self.limit)
        await self.send_bytes(ws, encoded)

    async def send_bytes(self, ws, encoded):
        try:
            await asyncio.wait_for(ws.send(encoded), self.send_timeout)
        except asyncio.TimeoutError:
            ws.abort()
            raise

    def failed(self, message):
        logger.exception(message)
        self.failure = Error(message)
        self.stopping.set()

    def snapshot_spec(self):
        declared = self.backend.describe()
        if not isinstance(declared, InferenceSpec):
            raise TypeError("Backend.describe must return InferenceSpec")
        self.spec = InferenceSpec.from_dict(declared.to_dict())

    def execute(self, op, inputs, request_id):
        try:
            if op == "describe":
                if inputs:
                    raise InvalidInput("describe requires an empty dictionary")
                result = {"spec": self.spec.to_dict()}
            elif op == "infer":
                try:
                    self.spec.inputs.validate(inputs, "inputs")
                except SpecError as exc:
                    raise InvalidInput(str(exc)) from exc
                result = self.backend.infer(inputs)
            else:
                if inputs:
                    try:
                        self.spec.context.validate(inputs, "context")
                    except SpecError as exc:
                        raise InvalidInput(str(exc)) from exc
                result = self.backend.reset(inputs)
                if result is not None:
                    return None, "invalid_output"
                result = {}
        except InvalidInput as exc:
            # This is an explicitly public validation error. Bound diagnostic size;
            # tiny message limits and unencodable exception text still get a reply.
            try:
                encoded = codec.encode(
                    protocol.failure(
                        request_id, "invalid_input", str(exc)[:512] or "Input rejected"
                    ),
                    self.limit,
                )
            except ValueError:
                encoded = codec.encode(
                    protocol.failure(request_id, "invalid_input", "Input rejected"), self.limit
                )
            return encoded, "invalid_input"
        except Exception:
            logger.exception("backend_error id=%s op=%s", request_id, op)
            return None, "backend_error"
        try:
            if not isinstance(result, dict):
                raise TypeError("backend output must be a dictionary")
            if op == "infer":
                self.spec.outputs.validate(result, "outputs")
            return codec.encode(protocol.success(request_id, result), self.limit), None
        except (TypeError, ValueError, OverflowError, RecursionError):
            logger.exception("invalid_output id=%s op=%s", request_id, op)
            return None, "invalid_output"

    async def handle(self, ws):
        if self.stopping.is_set():
            await ws.close(1001)
            return
        if self.owner is not None:
            try:
                await self.send(ws, protocol.failure(0, "busy", "Backend is already in use"))
                await ws.close(1013)
            except (ConnectionClosed, asyncio.TimeoutError):
                pass
            return
        # No await between testing and acquiring: atomic in this event loop.
        self.owner = ws
        logger.info("connection acquired")
        initialized = False
        try:
            try:
                result = await self.work(self.backend.reset, {})
                if result is not None:
                    raise TypeError("Backend.reset must return None")
                initialized = True
                await self.work(self.snapshot_spec)
            except Exception:
                self.failed("Backend initialization failed; stopping service")
                await self.send(
                    ws, protocol.failure(0, "backend_error", "Backend initialization failed")
                )
                await ws.close(1011)
                return
            if ws.state != State.OPEN or self.stopping.is_set():
                return
            await self.send(ws, protocol.success(0, {}))
            previous = 0
            while ws.state == State.OPEN and not self.stopping.is_set():
                raw = await ws.recv()
                if ws.state != State.OPEN or self.stopping.is_set():
                    break
                message = None
                try:
                    message = await self.work(codec.decode, raw, self.limit)
                    request_id, op, inputs = protocol.request(message, previous)
                except ProtocolError:
                    if message is not None and protocol.valid_id(message.get("id")):
                        await self.send(
                            ws, protocol.failure(message["id"], "protocol_error", "Invalid request")
                        )
                    await ws.close(1008)
                    break
                previous = request_id
                if ws.state != State.OPEN or self.stopping.is_set():
                    break
                if op not in ("infer", "reset", "describe"):
                    await self.send(
                        ws,
                        protocol.failure(request_id, "unsupported_operation", "Unknown operation"),
                    )
                    await ws.close(1008)
                    break
                started = time.monotonic()
                encoded, error = await self.work(self.execute, op, inputs, request_id)
                logger.info(
                    "request id=%s op=%s duration=%.6f error=%s",
                    request_id,
                    op,
                    time.monotonic() - started,
                    error,
                )
                if ws.state != State.OPEN or self.stopping.is_set():
                    break
                if error and error != "invalid_input":
                    await self.send(
                        ws, protocol.failure(request_id, error, "Backend request failed")
                    )
                    await ws.close(1011)
                    break
                await self.send_bytes(ws, encoded)
        except (ConnectionClosed, asyncio.TimeoutError):
            pass
        finally:
            try:
                if initialized:
                    result = await self.work(self.backend.reset, {})
                    if result is not None:
                        raise TypeError("Backend.reset must return None")
            except Exception:
                self.failed("Backend cleanup failed; stopping service")
            finally:
                self.owner = None
                logger.info("connection released")


def serve(
    backend: Backend,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    token: str | None = None,
    ssl: SSLContext | None = None,
    max_message_bytes: int = codec.MAX_MESSAGE_BYTES,
    send_timeout: float = 30.0,
    stop_event: threading.Event | None = None,
) -> None:
    """Block until stopped; take ownership of backend and close it exactly once.

    Backend hooks run on one worker. A permanently blocked hook requires external
    process termination; disconnect/timeout cannot safely cancel model execution.
    """
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="inferport-backend") as executor:

        async def run():
            limit = codec.message_limit(max_message_bytes)
            timeout = positive_timeout(send_timeout, "send_timeout")
            validate_token(token)
            service = _Service(backend, executor, limit, timeout, token)
            async with websocket_serve(
                service.handle,
                host,
                port,
                ssl=ssl,
                subprotocols=[protocol.SUBPROTOCOL],
                process_request=service.handshake,
                compression=None,
                max_size=limit,
                max_queue=1,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=CLOSE_TIMEOUT,
                create_connection=ServerSocket,
            ):
                try:
                    while not service.stopping.is_set():
                        if stop_event is not None and stop_event.is_set():
                            break
                        try:
                            await asyncio.wait_for(service.stopping.wait(), 0.05)
                        except asyncio.TimeoutError:
                            pass
                finally:
                    service.stopping.set()
            if service.failure is not None:
                raise service.failure

        try:
            asyncio.run(run())
        except KeyboardInterrupt:
            pass
        finally:
            executor.submit(backend.close).result()
