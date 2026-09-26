# InferPort

A small Python library for exchanging inference inputs and results between a model
repository and a robot, simulator, or evaluation program. Supports policy, value,
reward, and other backends without importing a model framework or robot SDK.

The public API is `Backend`, `serve`, and `Client`. The execution loop belongs to the
calling application. Transport is WebSocket with binary MessagePack and NumPy arrays.
One connection owns one backend; batch is simply part of your array shapes.

## Install and run

Python **3.10+**, NumPy **>=1.21.3,<3**. Runtime dependencies are NumPy, msgpack,
and websockets. The package is currently developed locally; no PyPI publication is implied.

```bash
# From this repository; uv is a development convenience, not a runtime requirement.
uv sync --group dev
uv run examples/serve_value.py

# In another terminal, from this repository:
uv run examples/call_value.py
# [7. 7. 7. 7.]
```

Install into either consuming repository's environment with `uv pip install /path/to/InferPort`.
The repository is named `InferPort`; the installed distribution and import are `inferport`.
Existing projects can retain a compatible NumPy version, including `1.21.3` on Python 3.10,
`1.23.5` on Python 3.11, and `1.26.4` on Python 3.10–3.12. The selected NumPy version must
also support the environment's Python version; newer Python versions need newer NumPy releases.

Model repository:

```python
import numpy as np
from inferport import Backend, InvalidInput, serve


class ValueBackend(Backend):
    def infer(self, inputs):
        state = inputs.get("state")
        if not isinstance(state, np.ndarray) or state.ndim != 2:
            raise InvalidInput("expected state with shape [B, D]")
        return {"value": np.sum(state * state, axis=-1)}


serve(ValueBackend())  # localhost:8000; blocks until Ctrl+C
```

Execution repository:

```python
import numpy as np
from inferport import Client

with Client("ws://127.0.0.1:8000") as client:
    result = client.infer({"state": np.ones((4, 7), dtype=np.float32)})
    print(result["value"])
```

For a robot or simulator, call `infer()` inside your own loop and consume the returned
actions there. Define image layout, joint order, units, action chunk handling, and
normalization in your adapters. InferPort does not resize images or control hardware.

## State and errors

Only `Backend.infer(inputs) -> dict` is required. Stateful backends implement
`reset(context) -> None` to clear **all** model/processor/cache state and replace the
context. An empty context must always work. See [the counter example](examples/stateful_counter.py).
`close() -> None` releases backend resources at service shutdown.

For engines that must be created and used on the same thread, see
[the thread-bound model example](examples/thread_bound_backend.py). It initializes the
model on the backend worker before READY; later resets retain the loaded model.
Allow sufficient `open_timeout` for first-time model loading.

The server calls `reset({})` before READY and after disconnect, then `close()` once
when it exits. All hooks run serially on one worker thread; the Backend constructor
runs in the caller. `serve()` owns the backend even if startup fails. A second connection
receives `RemoteError(code="busy")`, including while old work is finishing or cleaning up.
If initialization or disconnect cleanup fails, the service stops.

`Client()` performs no I/O. Use `with` or explicitly call `connect()` and `close()`.
Client calls must be sequential; concurrent calls raise `RuntimeError`. Cross-thread
`close()` interrupts network waits. Closing is idempotent. Create a new Client after
closing or a fatal failure; there is no reconnect, retry, or automatic request replay.

- Local invalid input types raise `TypeError`/`ValueError` before sending; the connection stays usable.
- A backend raises `InvalidInput` **before changing state** for an expected input problem.
  The caller gets nonfatal `RemoteError(code="invalid_input")` and may continue.
  Its message is public validation guidance (bounded to 512 characters).
- Unexpected backend exceptions and unsupported outputs produce fatal `RemoteError`.
  These unexpected errors use generic client messages; server tracebacks are logged locally.
- `ProtocolError` rejects malformed messages or mismatched responses.
- `TransportError` covers connection failures; `RequestTimeout` is its subclass and
  provides `.stage` (`open`, `ready`, `send`, or `response`).
- All library exceptions inherit `Error`. `RemoteError` exposes `code`, `message`,
  `request_id`, and `fatal`.

## Deadlines and limits

`Client(uri, timeout=30, open_timeout=10, max_message_bytes=64*1024*1024)`:

- `open_timeout` covers connection, handshake, and READY together.
- `infer(data, timeout=...)` / `reset(context, timeout=...)` cover send and receive,
  including model execution. Local encoding and decoding are outside this deadline.
- Timeout makes the Client unusable. Network cleanup has a separate five-second budget.
  A timeout cannot cancel an already running backend operation. Its result is dropped;
  ownership remains held until that work and cleanup finish.
- A permanently stuck backend requires external process termination. InferPort cannot
  safely kill GPU/Python worker operations or promise hard realtime behavior.

Payloads are string-keyed dictionaries containing nested dictionaries/lists, basic
scalars, bytes, and real numeric/bool NumPy arrays. Tuples become lists; NumPy scalars
become Python scalars. Arrays preserve values, shape, and type width and arrive as
writable, C-contiguous, native-endian arrays. Object, structured, complex, text,
datetime, and custom array dtypes are rejected. Convert Torch/JAX tensors explicitly.

The default **64 MiB complete message** limit applies on both sides, including the
envelope. Configure the same limit on Client and serve (minimum 128 bytes). Arrays
have at most 32 dimensions; messages at most 32 nesting levels and 100,000 nodes.
This is not a total process memory limit: serialization, buffers, and writable arrays
require additional memory. No automatic chunking or compression is performed.

## Deployment

`serve(backend, host="127.0.0.1", port=8000, token=None, ssl=None,
max_message_bytes=64*1024*1024, send_timeout=30, stop_event=None)`.
Use `threading.Event` for an embedded service's stop request; Ctrl+C works in a script.
`send_timeout` bounds network writes, not backend execution.

To listen across machines, explicitly select the listening address. Both sides accept
`token="..."` for bearer authentication and an `ssl.SSLContext` for TLS. The server
context must load its certificate; `wss://` clients verify certificates and hostnames
by default. Use a client context for a private CA. Tokens must be nonempty printable
ASCII without whitespace; keep them out of URLs. Use TLS or a trusted encrypted tunnel
across untrusted networks; a token does not encrypt `ws://` traffic.

The only endpoint is `/`, with required subprotocol `inferport.v1`. Browser Origin
requests are rejected. Compression and system proxy discovery are disabled. Heartbeats
check connection liveness, not model progress. Standard Python logging provides request
IDs, operations, durations, and errors without automatically logging payloads.

## Development and scope

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv build
uv run benchmarks/roundtrip.py --iterations 100
```

See [the full protocol and design](docs/inferport-design.md) and
[verification results and remaining gaps](docs/validation.md).
CI covers Python 3.10–3.14, minimum dependencies (NumPy 1.21.3), NumPy 1.23.5 / 1.26.4,
and newer combinations.
It includes Windows/macOS smoke jobs; configured CI jobs aren't evidence of a successful run.

InferPort replaces `policy_runtime` with no compatibility alias or TCP/JSON fallback.
This repository doesn't implement RTC, robot/environment base classes, action scheduling,
automatic batching, multiple sessions/models, or a schema/description framework.
Real model and robot integrations remain in their owning repositories and require
separate validation.
