# InferPort

A small Python library for exchanging inference inputs and results between a model
repository and a robot, simulator, or evaluation program. Supports policy, value,
reward, and other backends without importing a model framework or robot SDK.

The public API is `Backend`, `serve`, and `Client`. The execution loop belongs to the
calling application. Transport is WebSocket with binary MessagePack and NumPy arrays.
One connection owns one backend; batch is simply part of your array shapes.

## Install and run

Python **3.10+**, NumPy **>=1.21.3,<3**. Runtime dependencies are NumPy, msgpack,
and websockets. Install from PyPI inside your project's virtual environment:

```bash
python -m pip install inferport
```

If your configured mirror doesn't have a new release yet, use the official index:
`python -m pip install --index-url https://pypi.org/simple inferport`.

For installation from source and local development:

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
from inferport import Backend, Dimension, InferenceSpec, ObjectSpec, TensorSpec, serve


class ValueBackend(Backend):
    def describe(self):
        return InferenceSpec(
            ObjectSpec({"state": TensorSpec("float32", (Dimension(), Dimension()))}),
            ObjectSpec({"value": TensorSpec("float32", (Dimension(),))}),
        )

    def infer(self, inputs):
        state = inputs["state"]
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
actions there. Declare image layout, joint order, units and chunk shapes through an
`InferenceSpec`; both adapters implement that contract. InferPort validates data but
does not resize images, normalize model features or control hardware.

## Input and output contracts

Every backend implements `Backend.describe() -> InferenceSpec`. The
server snapshots it once per connection, after the initial `reset({})`, on the same
backend worker. `Client.connect()` fetches and validates the snapshot within
`open_timeout`; `client.describe()` reads its immutable local cache without I/O.
Missing or invalid contracts prevent initialization;
there is no untyped backend mode.

```python
from inferport import Dimension, InferenceSpec, ObjectSpec, TensorSpec

spec = InferenceSpec(
    inputs=ObjectSpec({"state": TensorSpec("float32", (Dimension(1, 128), 7))}),
    outputs=ObjectSpec({"value": TensorSpec("float32", (Dimension(1, 128),))}),
)
```

`ScalarSpec` describes strings, integers, numbers and booleans; `ObjectSpec` describes
required/optional fields. Scalar validation accepts the codec's supported NumPy
bool/integer/float scalars without changing the payload; received scalars are Python
values. Booleans remain distinct from numbers, and finite/range checks still apply.
`TensorSpec` describes dtype, fixed/bounded dimensions,
axes, semantics and ordered `Channel(name, semantic, unit)` values. Contracts serialize
through `to_dict()` / `InferenceSpec.from_dict()`; no Python objects cross the wire.
Dimensions are independent bounds, not symbolic relations across input/output fields.
Applications choose field names and compose these specs; the core has no policy-specific
payload keys. Local dataclasses/custom types can map to dictionaries and arrays at the
application boundary; InferPort doesn't transmit or import their Python classes.

The server rejects invalid input/context **before** calling
the backend, and rejects invalid output before transmission. The client also validates
received outputs before `infer()` returns; violations close the connection and raise
`ProtocolError`. Applications needn't repeat those format checks. Backends still check
business rules before changing state, such as agreement with the current instruction.
Direct Python calls to Backend bypass serve and must meet its declared contract.
Empty reset always remains valid. `check_compatibility(service_spec, execution_spec)` checks both data directions,
including channel ordering and units; applications call it before executing actions.
The specification is stable for a connection. It does not identify loaded weights.

`inferport.robotics.joint_target_spec()` provides the versioned
`robotics.joint-targets.v1` profile: `images.<camera-role>` (uint8 RGB HWC), `state`
(float32 ordered channels), `instruction`, and float32 `action[steps, channels]`.
It distinguishes measured positions from drive targets; actions are sequential absolute
joint targets with executor-defined duration. Revolute joints use rad, prismatic joints
use m, and grippers use fraction (0 closed, 1 open). Nonempty reset context supplies
`instruction`. Diagnostics, task metadata and model seed options belong to applications;
extend the returned dataclass with `dataclasses.replace` and additional `ObjectSpec`
fields on each endpoint as needed. No extension registry is required.
Each endpoint supplies its own camera sizes, channel names and output horizon; no model
or robot shape is built into InferPort. See the [contract reference](docs/inferport-design.md#input-output-contracts).

This release uses the single WebSocket subprotocol `inferport`, without a version
suffix. Both endpoints must use the current SDK; earlier versioned subprotocols are
not accepted. Backends must implement `describe()` and declare their data contract.
The 0.2.0 release includes the required contract API; old backends need updating.
READY/reset acknowledgments remain empty.

## State and errors

Both `Backend.describe() -> InferenceSpec` and `Backend.infer(inputs) -> dict` are
required. Stateful backends implement
`reset(context) -> None` to clear **all** model/processor/cache state and replace the
context. An empty context must always work. See [the counter example](https://github.com/jeremy775885/InferPort/blob/main/examples/stateful_counter.py).
`close() -> None` releases backend resources at service shutdown.

For engines that must be created and used on the same thread, see
[the thread-bound model example](https://github.com/jeremy775885/InferPort/blob/main/examples/thread_bound_backend.py). It initializes the
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
- `ProtocolError` rejects malformed messages, mismatched responses and results violating the contract.
- `TransportError` covers connection failures; `RequestTimeout` is its subclass and
  provides `.stage` (`open`, `ready`, `describe`, `send`, or `response`).
- All library exceptions inherit `Error`. `RemoteError` exposes `code`, `message`,
  `request_id`, and `fatal`.

## Deadlines and limits

`Client(uri, timeout=30, open_timeout=10, max_message_bytes=64*1024*1024)`:

- `open_timeout` covers connection, handshake, READY and contract retrieval together.
- `infer(data, timeout=...)` / `reset(context, timeout=...)` cover send and receive,
  including model execution. Local encoding and decoding are outside this deadline.
- Timeout makes the Client unusable. Network cleanup has a separate five-second budget.
  A timeout cannot cancel an already running backend operation. Its result is dropped;
  ownership remains held until that work and cleanup finish.
- A permanently stuck backend requires external process termination. InferPort cannot
  safely kill GPU/Python worker operations or promise hard realtime behavior.

The codec supports string-keyed dictionaries containing nested dictionaries/lists, basic
scalars, bytes, and real numeric/bool NumPy arrays. Tuples become lists; NumPy scalars
become Python scalars. Arrays preserve values, shape, and type width and arrive as
writable, C-contiguous, native-endian arrays. Object, structured, complex, text,
datetime, and custom array dtypes are rejected. Convert Torch/JAX tensors explicitly.
Business payloads must also fit the declared contracts: current specs describe
scalars, arrays and objects, with no null/bytes/list field specs.

The default **64 MiB complete message** limit applies on both sides, including the
envelope and contract response. Configure the same limit on Client and serve
(minimum setting 128 bytes; the chosen contract may require more). Arrays
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

The only endpoint is `/`, with required subprotocol `inferport`. Browser Origin
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

CI covers Python 3.10–3.14, minimum dependencies (NumPy 1.21.3), NumPy 1.23.5 / 1.26.4,
and newer combinations.
It includes Windows/macOS smoke jobs; completed runs and their scope are recorded in
[the validation record](https://github.com/jeremy775885/InferPort/blob/main/docs/validation.md#github-actions).

This repository doesn't implement RTC, robot/environment base classes, action scheduling,
automatic batching or multiple sessions/models. Contracts describe and validate data;
model and robot conversions remain in the consuming repositories.
Real model and robot integrations remain in their owning repositories and require
separate validation.

## Documentation

Version 0.2.0 is an unpublished candidate in this checkout. Build/install locally until
it is released; 0.1.0 remains the published historical baseline.

| Document | Purpose |
|---|---|
| [Protocol and API reference](https://github.com/jeremy775885/InferPort/blob/main/docs/inferport-design.md) | Data format, public interfaces, lifecycle, and limits |
| [Validation overview](https://github.com/jeremy775885/InferPort/blob/main/docs/validation.md) | Completed checks, evidence, and remaining validation gaps |
| [Real inference validation plan](https://github.com/jeremy775885/InferPort/blob/main/docs/integration-validation.md) | Model adapters, remote inference comparison, and closed-loop validation |
| [Roadmap](https://github.com/jeremy775885/InferPort/blob/main/docs/roadmap.md) | Remaining priorities and deferred features |
| [Release guide](https://github.com/jeremy775885/InferPort/blob/main/docs/releasing.md) | Publishing subsequent versions |
| [Changelog](https://github.com/jeremy775885/InferPort/blob/main/CHANGELOG.md) | Changes by released version |

Design research and detailed first-release experiments are kept in
[the archive](https://github.com/jeremy775885/InferPort/tree/main/docs/archive), linked from the relevant reference documents.
