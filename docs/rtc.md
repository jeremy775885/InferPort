# Trained-prefix RTC client

`inferport.rtc.RTCClient` adds one asynchronous action stream without importing
LeRobot, Torch or a robot SDK. The model backend must already implement trained
prefix conditioning; this client does not turn an ordinary model into an RTC model.

Install InferPort 0.3.0 from its source/wheel until published. The communication
protocol is unchanged: a compatible existing server receives ordinary observations
with an optional `action_prefix` and returns `action`.

## Contract and timing

The service declares float32 `action[H,A]` with `absolute_joint_targets` semantics,
fixed positive dimensions and an
optional float32 `action_prefix[0..D_train,A]`. Units, channels and axes must match.
The client reads the maximum prefix length from the contract; it does not infer
model support from a name. Applications still check compatibility with their robot.

- `prefix_steps=d`: the deployment's committed old-action prefix, not measured delay.
- `period_steps=p`: request every p committed policy actions.
- `action_interval`: seconds per policy action, independent of video/servo FPS.
- The checkpoint bound D_train, d and actual end-to-end latency are separate.
- Admission requires `0 < d <= D_train`, `d < p` and `p+d <= H`.

The first request has no prefix and begins motion only when its chunk is ready.
Later requests snapshot the next d targets at the observation's action position.
An early result waits for that prefix to execute. Only the unexecuted suffix is
replaced at the boundary; committed targets cannot be rewritten. The completion
deadline covers capture preparation, upload, model execution, download and parsing.
`captured_at` must be the start of capture on the same monotonic clock as the client.
Capture when requested at the current control tick; captures from earlier action
steps are rejected rather than silently rebased.

## Control loop

```python
import time
from inferport import Client
from inferport.rtc import RTCClient

client = Client("ws://127.0.0.1:9234").connect()
with RTCClient(client, prefix_steps=5, period_steps=10, action_interval=0.1) as rtc:
    rtc.reset({"instruction": "pick up the object"})  # while the executor is stopped
    try:
        while running():
            if rtc.needs_observation:
                captured = time.monotonic()
                rtc.submit(read_observation(), captured_at=captured)
            target = rtc.next_action()  # never waits for inference
            if target is None:  # bootstrap or before the next scheduled tick
                time.sleep(0.001)
                continue
            send_target(target.values)  # execute immediately and unchanged
    finally:
        stop_executor()
```

The three application functions and stop behavior belong to your robot/simulator.
Capture/submit/next_action belong to one control thread. Returning `RTCAction`
commits the target; its read-only values, epoch, step, request ID, chunk offset
and monotonic `scheduled_at` describe that commitment. Request IDs here are local
RTC diagnostic sequences, not server or wire-envelope counters. Do not clip, blend, skip,
queue beyond the declared cadence or substitute targets without stopping/resetting.
Physical tracking error is different from rewriting a command.

After successful construction, `RTCClient` exclusively owns its connected `Client`.
If validation in the constructor fails, the caller retains ownership and must close
or reuse that Client; an outer Client context manager handles this case. Do not call the wrapped client
concurrently. A worker performs serialized RPCs. `reset(context)` immediately
invalidates local actions and waits for an ordered server reset, bounded by
`control_timeout`. Stop execution before reset, including instruction changes.
A failed reset closes the connection; construct a new session. `close()` revokes
motion, interrupts network waits and joins the worker. It cannot interrupt a
server's running GPU call or stop physical hardware for you.

## Preparing executable targets

Applications may provide `prepare_actions(array)` to apply a deterministic
platform conversion before a predicted chunk enters the queue. It runs on the
RPC worker and returns a finite float32 array of the same shape. For example,
move an existing gripper-bound clip here instead of silently clipping after a
target has been committed. Future prefixes then contain the actual targets.

The SDK checks the service's raw prefix before this callback and checks it again
afterward. Preparation must leave all already committed prefix values unchanged;
it cannot conceal a corrupt service prefix. The callback must be bounded and
nonblocking; network cancellation cannot interrupt arbitrary application Python code.
Result diagnostics include
`adjusted_values`. Targets returned by `next_action()` must still be executed
unchanged. This is an application hook, not a model normalization or robot SDK.

## Failure and diagnostics

Late/corrupt results, missing observations, queue exhaustion and transport failures
raise terminal `RTCError`. A control call more than one action interval behind its
scheduled tick also faults; this is a bounded best-effort loop, not hard realtime.
There is no automatic retry, reconnect, repeated-last-target or delay expansion.
The application must execute its own stop/hold policy when an error occurs.

`drain_events()` returns up to 128 recent reset/request/result/switch/fault events.
Drain regularly for a complete application trace. Payload arrays are not retained
in the diagnostic buffer. `step` is the next uncommitted index, and `failed` latches
an inference/control fault. Two action chunks at most are retained (active and
pending replacement), in addition to the single in-flight observation/prefix.

Result events include `request_id`, `anchor`, `captured_at`, `deadline`,
`completed_at`, `latency_s` and `deadline_margin_s = deadline - completed_at`.
Completion means the RPC, decoding, validation and `prepare_actions` have finished
on the client. All timestamps use the client's monotonic clock; a server completion
timestamp is not interchangeable with them. A positive margin measures result
readiness before the deadline, not the time hardware actually applied a target.

Fault events include `detected_at` and, once motion has started, `scheduled_at`
for the next target. When an in-flight or ready request is associated with the fault,
they also include `request_id`, `anchor`, `captured_at` and `deadline`.
`completed_at` is present only when a valid prepared result's completion is known.
A request still in transit or one failing validation has no such timestamp.
Faults before a request is created may have no request fields. Faults latch: a late
result neither clears the fault nor appends a successful result event afterward.

For a request anchored at step `n`, old commands may be committed at
`n .. n+d-1`. At step `n+d`, the replacement must be available to the control
thread; otherwise `next_action()` raises before returning that target. `step=n+d`
means this is the next uncommitted target, not one already executed. A completed
result at exactly the deadline may be accepted if published before the boundary
check; an unfinished result at that check faults. Full chunks remain anchored at
`n`, so the first new target is at chunk offset `d`, never offset zero.

These checks run when the application calls the RTC control API. They are not an
independent hardware watchdog. The executor must not prequeue commands beyond the
commitment boundary and must implement its stop/hold behavior on failure. A blocked
control thread needs a device-side timeout. Capture and `submit()` (which copies
the observation) also run on the control thread; measure their cost with real
cameras before changing how snapshots are acquired.

Explicit client prefixes keep server lifecycle unchanged. Future same-observation
batch candidates can send one longest prefix plus candidate lengths; this version
does not implement candidate batching or cross-request server action references.

## Validation without a model or simulator

The RTC tests use known action sequences, a controllable clock and local WebSocket
connections. They need only the development dependencies and CPU:

```bash
uv run pytest -q tests/test_rtc.py
```

They cover result completion before/at/after a deadline, partially delivered
responses, wrong response IDs, disconnects, delayed client action preparation,
fault latching and reset isolation. Event gates control ordering without relying
on millisecond sleeps for deadline assertions. The recorded targets check command
ordering; these tests do not validate robot motion, model quality or hard realtime.

For image transport measurements, use the existing loopback benchmark's policy
mode. This is a synthetic backend that returns a small action array; it does not
run a learned policy or the RTC control loop:

```bash
uv run benchmarks/roundtrip.py --mode policy \
  --cameras 3 --height 480 --width 640 \
  --horizon 50 --action-width 14 --prefix-steps 5 \
  --iterations 1000 --warmup 20 --budget-ms 100 \
  --output /tmp/inferport-policy-load.json
```

This Linux benchmark reports request/response sizes, client/server CPU time,
snapshot-copy and RPC P50/P95/P99, and paired raw timing samples in the JSON file.
`--budget-ms` is an optional comparison budget for **snapshot + RPC** only;
exceedances are not RTC deadline violations. The example 100 ms is not a recommended
robot budget. Camera acquisition, model inference, action preparation and execution
must be measured separately on the target platform. Peak RSS includes earlier
codec/JSON comparisons and is not a per-request memory footprint. The default
`--mode echo` retains the original image roundtrip workload; historical results
are not overwritten unless their path is explicitly supplied.

Use the model-free baseline to decide whether a same-machine image shared-memory
prototype is worth measuring. It does not establish a zero-copy or reliability
advantage. Batch candidate latency still requires the actual model and GPU, and
executor timing and stop behavior still require the actual execution environment.
