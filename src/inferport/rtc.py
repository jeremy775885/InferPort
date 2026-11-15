"""Optional, client-owned trained-prefix RTC; no model or robot dependencies.

The caller captures observations and executes returned targets unchanged. One
worker serializes RPCs while the control thread commits targets on a fixed clock.
This module does not control hardware, cancel model computation, or reconnect.
"""

from __future__ import annotations

import copy
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .client import Client
from .errors import Error
from .specs import Dimension, TensorSpec


class RTCError(Error):
    """Motion is revoked; stop the executor before resetting or closing."""


@dataclass(frozen=True)
class RTCAction:
    """Returning this immutable target commits its step to the executor."""

    values: np.ndarray
    step: int
    epoch: int
    request_id: int
    chunk_offset: int
    scheduled_at: float


@dataclass(frozen=True)
class _Prediction:
    actions: np.ndarray
    completed_at: float
    adjusted_values: int = 0


@dataclass(frozen=True)
class _Request:
    future: Future
    epoch: int
    identity: int
    anchor: int
    captured_at: float
    deadline: float
    prefix: np.ndarray | None


class RTCClient:
    """Wrap a connected Client and take exclusive ownership of its calls/lifetime.

    Call reset(context) while stopped, then poll needs_observation/next_action.
    When observation is needed, capture its start time with the same monotonic
    clock and call submit(inputs, captured_at=...). Execute each returned target
    immediately and unchanged. None means bootstrap/not yet the next control tick.
    Once motion starts, missed inference or control deadlines raise RTCError.
    Call close() on execution failure; this class cannot stop physical hardware.

    Only reset/close may interrupt another control call. Capture, submit and
    next_action belong to one control thread. Clock injection is for deterministic
    tests, not substitution of recording FPS for physical time.
    """

    def __init__(
        self,
        client: Client,
        *,
        prefix_steps: int,
        period_steps: int,
        action_interval: float,
        control_timeout: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        prepare_actions: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        for name, value in (("prefix_steps", prefix_steps), ("period_steps", period_steps)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("action_interval", action_interval),
            ("control_timeout", control_timeout),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be positive and finite")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        spec = client.describe()
        action = spec.outputs.fields.get("action")
        prefix = spec.inputs.fields.get("action_prefix")
        if (
            not isinstance(action, TensorSpec)
            or action.dtype != "float32"
            or action.semantic != "absolute_joint_targets"
            or len(action.shape) != 2
            or any(type(n) is not int or n <= 0 for n in action.shape)
        ):
            raise ValueError("RTC requires fixed float32 absolute joint targets [horizon, width]")
        if (
            not isinstance(prefix, TensorSpec)
            or "action_prefix" not in spec.inputs.optional
            or prefix.dtype != "float32"
            or len(prefix.shape) != 2
            or prefix.shape[1] != action.shape[1]
            or not isinstance(prefix.shape[0], Dimension)
            or prefix.shape[0].minimum != 0
            or prefix.shape[0].maximum is None
            or prefix.semantic != action.semantic
            or prefix.axes != action.axes
            or prefix.channels != action.channels
        ):
            raise ValueError("Service must declare a compatible optional bounded action_prefix")
        self.horizon, self.width = action.shape
        self.max_prefix_steps = prefix.shape[0].maximum
        if not prefix_steps <= self.max_prefix_steps:
            raise ValueError("prefix_steps exceeds the checkpoint's trained bound")
        if not prefix_steps < period_steps or period_steps + prefix_steps > self.horizon:
            raise ValueError("RTC requires prefix_steps < period_steps and their sum <= horizon")
        self.prefix_steps, self.period_steps = prefix_steps, period_steps
        self.action_interval, self.control_timeout = float(action_interval), float(control_timeout)
        self._client, self._clock = client, clock
        self._prepare_actions = prepare_actions
        self._lock = threading.RLock()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inferport-rtc")
        self._closed = self._initialized = False
        self._epoch = self._sequence = 0
        self._fault: RTCError | None = None
        self._events: deque[dict] = deque(maxlen=128)
        self._clear()

    def _clear(self) -> None:
        self._pending: _Request | None = None
        self._ready: tuple[_Request, _Prediction] | None = None
        self._active: np.ndarray | None = None
        self._anchor = self._cursor = self._active_id = self._next_request = 0
        self._next_tick: float | None = None

    def _check(self) -> None:
        if self._closed:
            raise RTCError("RTC client is closed")
        if self._fault is not None:
            raise self._fault

    def _fail(
        self,
        message: str,
        *,
        request: _Request | None = None,
        completed_at: float | None = None,
    ) -> None:
        self._fault = RTCError(message)
        event = {
            "event": "fault",
            "epoch": self._epoch,
            "step": self._cursor,
            "detail": message,
            "detected_at": self._clock(),
        }
        if self._next_tick is not None:
            event["scheduled_at"] = self._next_tick
        if request is None:
            request = self._pending
            if request is None and self._ready is not None:
                request, result = self._ready
                completed_at = result.completed_at
        if request is not None:
            event.update(
                request_id=request.identity,
                anchor=request.anchor,
                captured_at=request.captured_at,
                deadline=request.deadline,
            )
        if completed_at is not None:
            event["completed_at"] = completed_at
        self._events.append(event)
        raise self._fault

    @property
    def step(self) -> int:
        """Next uncommitted policy action index in this epoch."""
        with self._lock:
            return self._cursor

    @property
    def failed(self) -> bool:
        with self._lock:
            return self._fault is not None

    @property
    def needs_observation(self) -> bool:
        with self._lock:
            self._check()
            self._poll()
            return (
                self._initialized
                and self._pending is None
                and self._ready is None
                and self._cursor >= self._next_request
                and (self._next_tick is None or self._clock() >= self._next_tick)
            )

    def reset(self, context: dict | None = None) -> None:
        """Invalidate immediately; wait boundedly for the serialized server reset.

        The caller must stop physical execution first. A timeout closes the
        underlying connection; a new Client/RTCClient is then required.
        """
        with self._lock:
            if self._closed:
                raise RTCError("RTC client is closed")
            self._epoch += 1
            epoch = self._epoch
            self._initialized = False
            self._fault = None
            self._clear()
            future = self._worker.submit(
                self._client.reset, copy.deepcopy(context), timeout=self.control_timeout
            )
        try:
            future.result(timeout=self.control_timeout)
        except Exception as exc:
            self.close()
            raise RTCError("RTC reset failed; create a new session") from exc
        with self._lock:
            if self._closed or epoch != self._epoch:
                raise RTCError("RTC reset was superseded")
            self._initialized = True
            self._events.append({"event": "reset", "epoch": epoch})

    def submit(self, inputs: dict, *, captured_at: float | None = None) -> None:
        """Snapshot one fresh observation at the current uncommitted action step."""
        with self._lock:
            self._check()
            if not self.needs_observation:
                raise RuntimeError("RTC is not requesting an observation at this step")
            now = self._clock()
            captured = now if captured_at is None else captured_at
            if isinstance(captured, bool) or not isinstance(captured, (int, float)):
                raise ValueError("captured_at must use the client's monotonic clock")
            if not math.isfinite(captured) or captured > now:
                raise ValueError("captured_at must be finite and not in the future")
            if not isinstance(inputs, dict) or "action_prefix" in inputs:
                raise ValueError("Supply observations only; RTCClient owns action_prefix")
            if self._next_tick is not None and captured < self._next_tick:
                self._fail("Observation preparation used a capture before the current control step")
            payload = copy.deepcopy(inputs)
            prefix = None
            if self._active is not None:
                offset = self._cursor - self._anchor
                prefix = self._active[offset : offset + self.prefix_steps].copy()
                if len(prefix) != self.prefix_steps:
                    self._fail("Insufficient continuation for the trained prefix")
                payload["action_prefix"] = prefix.copy()
                deadline = min(captured, self._next_tick) + self.prefix_steps * self.action_interval
            else:
                deadline = captured + self.control_timeout
            if self._clock() > deadline:
                self._fail("Observation preparation exceeded the RTC deadline")
            identity = self._sequence
            self._sequence += 1
            future = self._worker.submit(self._infer, payload)
            self._pending = _Request(
                future, self._epoch, identity, self._cursor, captured, deadline, prefix
            )
            self._next_request = self._cursor + self.period_steps
            self._events.append(
                {
                    "event": "request",
                    "epoch": self._epoch,
                    "request_id": identity,
                    "step": self._cursor,
                    "captured_at": captured,
                    "deadline": deadline,
                    "prefix_steps": 0 if prefix is None else len(prefix),
                }
            )

    def _infer(self, payload: dict) -> _Prediction:
        response = self._client.infer(payload, timeout=self.control_timeout)
        actions = response.get("action")
        if (
            not isinstance(actions, np.ndarray)
            or actions.dtype != np.float32
            or actions.shape != (self.horizon, self.width)
            or not np.isfinite(actions).all()
        ):
            raise RTCError("Invalid action chunk from service")
        prefix = payload.get("action_prefix")
        if prefix is not None and not np.array_equal(actions[: len(prefix)], prefix):
            raise RTCError("Service changed the committed action_prefix")
        owned = actions.copy()
        adjusted = 0
        if self._prepare_actions is not None:
            owned = self._prepare_actions(owned)
            if (
                not isinstance(owned, np.ndarray)
                or owned.dtype != np.float32
                or owned.shape != actions.shape
                or not np.isfinite(owned).all()
            ):
                raise RTCError("prepare_actions must preserve finite float32 chunk shape")
            if prefix is not None and not np.array_equal(owned[: len(prefix)], prefix):
                raise RTCError("prepare_actions changed an already committed prefix")
            adjusted = int(np.count_nonzero(owned != actions))
            owned = owned.copy()
        owned.setflags(write=False)
        return _Prediction(owned, self._clock(), adjusted)

    def _poll(self) -> None:
        request = self._pending
        if request is None:
            return
        if request.future.done():
            self._pending = None
            try:
                result = request.future.result()
            except Exception as exc:
                try:
                    self._fail(
                        f"RTC inference failed: {type(exc).__name__}: {exc}", request=request
                    )
                except RTCError as error:
                    raise error from exc
            if request.epoch != self._epoch:
                return
            if result.completed_at > request.deadline:
                self._fail(
                    "RTC result missed its prefix deadline",
                    request=request,
                    completed_at=result.completed_at,
                )
            if request.prefix is not None and not np.array_equal(
                result.actions[: self.prefix_steps], request.prefix
            ):
                self._fail(
                    "Service changed the committed action_prefix",
                    request=request,
                    completed_at=result.completed_at,
                )
            self._events.append(
                {
                    "event": "result",
                    "epoch": request.epoch,
                    "request_id": request.identity,
                    "step": self._cursor,
                    "anchor": request.anchor,
                    "captured_at": request.captured_at,
                    "deadline": request.deadline,
                    "completed_at": result.completed_at,
                    "latency_s": result.completed_at - request.captured_at,
                    "deadline_margin_s": request.deadline - result.completed_at,
                    "adjusted_values": result.adjusted_values,
                }
            )
            if request.prefix is None:
                self._active = result.actions
                self._anchor, self._active_id = request.anchor, request.identity
                self._next_tick = self._clock()
            else:
                self._ready = (request, result)
        elif self._clock() > request.deadline:
            self._fail("RTC request missed its prefix deadline")

    def next_action(self) -> RTCAction | None:
        """Commit one due target without waiting for network/model execution."""
        with self._lock:
            self._check()
            self._poll()
            if not self._initialized or self._active is None:
                return None
            now = self._clock()
            if now < self._next_tick:
                return None
            if now > self._next_tick + self.action_interval:
                self._fail("Control loop missed an action interval; reset while stopped")
            if self._cursor >= self._next_request:
                self._fail("Fresh observation was not submitted at the request period")
            if self._ready is not None:
                request, result = self._ready
                if self._cursor >= request.anchor + self.prefix_steps:
                    self._active, self._anchor = result.actions, request.anchor
                    self._active_id, self._ready = request.identity, None
                    self._events.append(
                        {
                            "event": "switch",
                            "epoch": self._epoch,
                            "request_id": request.identity,
                            "step": self._cursor,
                        }
                    )
            if self._pending is not None and self._pending.prefix is not None:
                if self._cursor >= self._pending.anchor + self.prefix_steps:
                    self._fail("No prediction ready at the prefix boundary")
            offset = self._cursor - self._anchor
            if not 0 <= offset < len(self._active):
                self._fail("RTC action queue exhausted")
            values = self._active[offset].copy()
            values.setflags(write=False)
            target = RTCAction(
                values, self._cursor, self._epoch, self._active_id, offset, self._next_tick
            )
            self._cursor += 1
            self._next_tick += self.action_interval
            return target

    def drain_events(self) -> list[dict]:
        """Return bounded diagnostics; no image or action payloads are retained."""
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return events

    def close(self) -> None:
        """Revoke motion, interrupt the network wait and join the RPC worker."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._initialized = False
            self._epoch += 1
            self._clear()
        try:
            self._client.close()
        finally:
            self._worker.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> RTCClient:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
