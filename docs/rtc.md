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

Explicit client prefixes keep server lifecycle unchanged. Future same-observation
batch candidates can send one longest prefix plus candidate lengths; this version
does not implement candidate batching or cross-request server action references.
