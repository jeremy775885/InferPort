"""Trained-prefix scheduling, real transport, and lifecycle regressions."""

import copy
import threading
import time
from dataclasses import replace
from queue import Queue

import numpy as np
import pytest

from inferport import (
    Backend,
    Client,
    Dimension,
    Error,
    InferenceSpec,
    ObjectSpec,
    TensorSpec,
    codec,
    protocol,
)
from inferport.rtc import RTCClient, RTCError
from tests.peers import frame, peer, recv_frame


def contract(horizon=12, maximum=4):
    action = TensorSpec("float32", (horizon, 2), "absolute_joint_targets", ("step", "channel"))
    return InferenceSpec(
        ObjectSpec(
            {
                "state": TensorSpec("float32", (2,)),
                "action_prefix": replace(action, shape=(Dimension(0, maximum), 2)),
            },
            optional=("action_prefix",),
        ),
        ObjectSpec({"action": action}),
    )


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class ControlledClient:
    def __init__(self):
        self.started = Queue()
        self.release = threading.Event()
        self.release.set()
        self.calls = []
        self.resets = 0
        self.corrupt = False
        self.closed = False

    def describe(self):
        return contract()

    def reset(self, context=None, *, timeout=None):
        self.resets += 1

    def infer(self, inputs, *, timeout=None):
        self.calls.append(copy.deepcopy(inputs))
        number = len(self.calls) - 1
        self.started.put(inputs)
        assert self.release.wait(2), "test worker was not released"
        if self.closed:
            raise RuntimeError("closed")
        actions = np.repeat(np.arange(12, dtype=np.float32)[:, None], 2, axis=1)
        actions += number * 100
        prefix = inputs.get("action_prefix")
        if prefix is not None:
            actions[: len(prefix)] = prefix + (1 if self.corrupt else 0)
        return {"action": actions}

    def close(self):
        self.closed = True
        self.release.set()


@pytest.fixture
def runtime():
    clock, client = Clock(), ControlledClient()
    rtc = RTCClient(client, prefix_steps=3, period_steps=6, action_interval=1, clock=clock)
    rtc.reset()
    yield rtc, client, clock
    rtc.close()


def wait_result(rtc):
    until = time.monotonic() + 2
    while time.monotonic() < until:
        _ = rtc.needs_observation
        events = rtc.drain_events()
        if any(event["event"] == "result" for event in events):
            return
        time.sleep(0.001)
    raise AssertionError("no inference result")


def bootstrap(rtc):
    rtc.submit({"state": np.zeros(2, np.float32)})
    wait_result(rtc)


def to_request(rtc, clock, *, constant=None):
    bootstrap(rtc)
    for index in range(6):
        clock.now = 100 + index
        target = rtc.next_action()
        assert target.step == index
        expected = index if constant is None else constant
        np.testing.assert_array_equal(target.values, [expected, expected])
    clock.now = 106
    assert rtc.needs_observation


@pytest.mark.parametrize("prefix,period", [(0, 6), (True, 6), (5, 6), (3, 3), (3, 10)])
def test_configuration_rejected_before_rpc(prefix, period):
    client = ControlledClient()
    with pytest.raises(ValueError):
        RTCClient(client, prefix_steps=prefix, period_steps=period, action_interval=1)
    assert client.resets == 0 and client.calls == []


@pytest.mark.parametrize("completed_at", [108.999, 109.0, 109.001])
def test_result_completion_around_deadline(runtime, completed_at):
    rtc, client, clock = runtime
    to_request(rtc, clock)
    client.release.clear()
    rtc.submit({"state": np.zeros(2, np.float32)})
    # Wait for worker publication without polling or adopting the result yet.
    future = rtc._pending.future
    committed = []
    for step in (6, 7, 8):
        clock.now = 100 + step
        committed.append(rtc.next_action())
    clock.now = completed_at
    client.release.set()
    future.result(timeout=2)
    clock.now = max(109.0, completed_at)
    if completed_at > 109:
        with pytest.raises(RTCError, match="result missed"):
            rtc.next_action()
        assert rtc.step == 9
        fault = rtc.drain_events()[-1]
        assert fault["event"] == "fault"
        assert fault["request_id"] == 1 and fault["anchor"] == 6
        assert fault["captured_at"] == 106 and fault["deadline"] == 109
        assert fault["completed_at"] == fault["detected_at"] == completed_at
        assert fault["scheduled_at"] == 109
    else:
        for step in (9, 10, 11):
            clock.now = 100 + step
            committed.append(rtc.next_action())
        assert [a.step for a in committed] == list(range(6, 12))
        assert [a.values[0] for a in committed] == [6, 7, 8, 103, 104, 105]
        assert [a.chunk_offset for a in committed[3:]] == [3, 4, 5]
        assert all(a.request_id == 1 and a.epoch == 1 for a in committed[3:])
        events = rtc.drain_events()
        result = next(e for e in events if e["event"] == "result")
        assert result["anchor"] == 6 and result["deadline"] == 109
        assert result["deadline_margin_s"] == pytest.approx(109 - completed_at)
        assert sum(e["event"] == "switch" for e in events) == 1


@pytest.mark.parametrize("failure", ["partial_response", "wrong_id", "disconnect"])
def test_real_wire_failure_never_commits_unconditioned_action(failure):
    stalled, resume = threading.Event(), threading.Event()

    def handler(sock, release):
        count = 0
        while True:
            try:
                opcode, raw = recv_frame(sock)
            except EOFError:
                return
            if opcode == 8:
                sock.sendall(frame(raw, opcode=8))
                return
            message = codec.decode(raw)
            request_id = message["id"]
            if message["op"] == "reset":
                output = {}
            else:
                count += 1
                actions = np.full((12, 2), count, dtype=np.float32)
                prefix = message["data"].get("action_prefix")
                if prefix is not None:
                    actions[: len(prefix)] = prefix
                output = {"action": actions}
                if count == 2 and failure == "disconnect":
                    return
                if count == 2 and failure == "wrong_id":
                    request_id += 1
            packet = frame(codec.encode(protocol.success(request_id, output)))
            if count == 2 and message["op"] == "infer" and failure == "partial_response":
                midpoint = len(packet) // 2
                sock.sendall(packet[:midpoint])
                stalled.set()
                assert resume.wait(5), "test didn't release partial response"
                sock.sendall(packet[midpoint:])
            else:
                sock.sendall(packet)

    clock = Clock()
    with peer(handler, spec=contract()) as uri:
        with RTCClient(
            Client(uri).connect(), prefix_steps=3, period_steps=6, action_interval=1, clock=clock
        ) as rtc:
            try:
                rtc.reset()
                to_request(rtc, clock, constant=1)
                rtc.submit({"state": np.zeros(2, np.float32)})
                future = rtc._pending.future
                if failure == "partial_response":
                    assert stalled.wait(2)
                    committed = []
                    for step in (6, 7, 8):
                        clock.now = 100 + step
                        committed.append(rtc.next_action().step)
                    clock.now = 109
                    with pytest.raises(RTCError, match="prefix boundary"):
                        rtc.next_action()
                    assert committed == [6, 7, 8] and rtc.step == 9
                else:
                    with pytest.raises(Error):
                        future.result(timeout=2)
                    with pytest.raises(RTCError, match="RTC inference failed"):
                        rtc.next_action()
                    assert rtc.step == 6
                fault = rtc.drain_events()[-1]
                assert fault["event"] == "fault" and fault["request_id"] == 1
                assert fault["anchor"] == 6 and fault["deadline"] == 109
                assert "completed_at" not in fault
                if failure == "partial_response":
                    # A completion at the deadline cannot undo an earlier boundary fault.
                    resume.set()
                    future.result(timeout=2)
                    with pytest.raises(RTCError):
                        rtc.next_action()
                    assert rtc.step == 9 and rtc.drain_events() == []
                    rtc.reset()
                    bootstrap(rtc)
                    action = rtc.next_action()
                    assert action.step == 0 and action.epoch == 2
                    np.testing.assert_array_equal(action.values, [3, 3])
            finally:
                resume.set()


def test_ready_result_does_not_hide_control_thread_overrun(runtime):
    rtc, _, clock = runtime
    to_request(rtc, clock)
    rtc.submit({"state": np.zeros(2, np.float32)})
    wait_result(rtc)
    # Inference was on time, but the control thread missed a whole action interval.
    clock.now = 108
    with pytest.raises(RTCError, match="Control loop"):
        rtc.next_action()
    assert rtc.step == 6
    fault = rtc.drain_events()[-1]
    assert fault["request_id"] == 1 and fault["anchor"] == 6
    assert fault["completed_at"] == fault["scheduled_at"] == 106
    assert fault["detected_at"] == 108 and fault["deadline"] == 109


def test_client_preparation_after_real_rpc_is_inside_deadline(server):
    entered, resume = threading.Event(), threading.Event()
    calls = 0

    def prepare(actions):
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            assert resume.wait(5)
        return actions

    clock = Clock()
    with server(PrefixBackend()) as (uri, *_):
        with RTCClient(
            Client(uri).connect(),
            prefix_steps=3,
            period_steps=6,
            action_interval=1,
            clock=clock,
            prepare_actions=prepare,
        ) as rtc:
            try:
                rtc.reset()
                to_request(rtc, clock, constant=1)
                rtc.submit({"state": np.zeros(2, np.float32)})
                future = rtc._pending.future
                assert entered.wait(2), "response didn't reach client preparation"
                for step in (6, 7, 8):
                    clock.now = 100 + step
                    assert rtc.next_action().step == step
                clock.now = 109.001
                resume.set()
                future.result(timeout=2)
                with pytest.raises(RTCError, match="result missed"):
                    rtc.next_action()
                assert rtc.step == 9
                fault = rtc.drain_events()[-1]
                assert fault["completed_at"] == 109.001 and fault["deadline"] == 109
            finally:
                resume.set()


def test_untrained_or_different_units_rejected():
    client = ControlledClient()
    original = client.describe()
    client.describe = lambda: replace(
        original, inputs=ObjectSpec({"state": TensorSpec("float32", (2,))})
    )
    with pytest.raises(ValueError, match="action_prefix"):
        RTCClient(client, prefix_steps=3, period_steps=6, action_interval=1)
    wrong = replace(original.inputs.fields["action_prefix"], semantic="normalized")
    client.describe = lambda: replace(
        original, inputs=replace(original.inputs, fields={"action_prefix": wrong})
    )
    with pytest.raises(ValueError, match="compatible"):
        RTCClient(client, prefix_steps=3, period_steps=6, action_interval=1)


def test_early_result_keeps_prefix_and_switches_once(runtime):
    rtc, client, clock = runtime
    to_request(rtc, clock)
    rtc.submit({"state": np.zeros(2, np.float32)})
    wait_result(rtc)
    np.testing.assert_array_equal(client.calls[1]["action_prefix"][:, 0], [6, 7, 8])
    for index, expected in enumerate([6, 7, 8, 103, 104, 105], start=6):
        clock.now = 100 + index
        target = rtc.next_action()
        assert target.step == index
        np.testing.assert_array_equal(target.values, [expected, expected])
        with pytest.raises(ValueError):
            target.values[0] = 999
    clock.now = 112
    assert rtc.needs_observation
    rtc.submit({"state": np.zeros(2, np.float32)})
    wait_result(rtc)
    np.testing.assert_array_equal(client.calls[2]["action_prefix"][:, 0], [106, 107, 108])


def test_inference_does_not_block_control_and_observation_is_owned(runtime):
    rtc, client, clock = runtime
    to_request(rtc, clock)
    client.release.clear()
    state = np.zeros(2, np.float32)
    rtc.submit({"state": state})
    state[:] = 99
    for index in (6, 7, 8):
        clock.now = 100 + index
        started = time.monotonic()
        target = rtc.next_action()
        assert time.monotonic() - started < 0.1
        assert target.step == index
    client.release.set()
    wait_result(rtc)
    assert client.calls[1]["state"].tolist() == [0, 0]
    clock.now = 109
    assert rtc.next_action().values.tolist() == [103, 103]


def test_boundary_without_result_faults_before_unconditioned_action(runtime):
    rtc, client, clock = runtime
    to_request(rtc, clock)
    client.release.clear()
    rtc.submit({"state": np.zeros(2, np.float32)})
    for index in (6, 7, 8):
        clock.now = 100 + index
        assert rtc.next_action().step == index
    clock.now = 109
    with pytest.raises(RTCError, match="prefix boundary"):
        rtc.next_action()
    assert rtc.step == 9
    client.release.set()
    with pytest.raises(RTCError):
        rtc.next_action()


def test_elapsed_latency_faults_even_when_control_did_not_advance(runtime):
    rtc, client, clock = runtime
    to_request(rtc, clock)
    client.release.clear()
    rtc.submit({"state": np.zeros(2, np.float32)})
    clock.now = 109.1
    with pytest.raises(RTCError, match="deadline"):
        rtc.next_action()
    assert rtc.step == 6


def test_corrupt_prefix_revokes_motion(runtime):
    rtc, client, clock = runtime
    to_request(rtc, clock)
    client.corrupt = True
    rtc.submit({"state": np.zeros(2, np.float32)})
    with pytest.raises(RTCError, match="changed"):
        wait_result(rtc)
    assert rtc.step == 6


def test_reset_while_inference_pending_discards_old_epoch(runtime):
    rtc, client, clock = runtime
    to_request(rtc, clock)
    client.release.clear()
    rtc.submit({"state": np.zeros(2, np.float32)})
    errors = []

    def reset():
        try:
            rtc.reset()
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=reset)
    thread.start()
    deadline = time.monotonic() + 1
    while rtc.step != 0 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert rtc.next_action() is None
    client.release.set()
    thread.join(2)
    assert not thread.is_alive() and not errors
    bootstrap(rtc)
    target = rtc.next_action()
    assert target.step == 0 and target.epoch == 2
    assert "action_prefix" not in client.calls[-1]
    assert client.resets == 2


def test_cadence_and_missing_observation(runtime):
    rtc, _, clock = runtime
    bootstrap(rtc)
    assert rtc.next_action().step == 0
    assert rtc.next_action() is None
    for index in range(1, 6):
        clock.now = 100 + index
        rtc.next_action()
    clock.now = 106
    with pytest.raises(RTCError, match="Fresh observation"):
        rtc.next_action()


def test_stale_capture_and_control_overrun(runtime):
    rtc, _, clock = runtime
    to_request(rtc, clock)
    with pytest.raises(RTCError, match="preparation"):
        rtc.submit({"state": np.zeros(2, np.float32)}, captured_at=102)
    rtc.reset()
    bootstrap(rtc)
    rtc.next_action()
    clock.now += 3
    with pytest.raises(RTCError, match="Control loop"):
        rtc.next_action()


class PrefixBackend(Backend):
    def describe(self):
        return contract()

    def infer(self, inputs):
        time.sleep(0.005)
        actions = np.ones((12, 2), np.float32)
        if "action_prefix" in inputs:
            prefix = inputs["action_prefix"]
            actions[: len(prefix)] = prefix
        return {"action": actions}


def test_real_socket_multiple_chunks_and_close(server):
    with server(PrefixBackend()) as (uri, *_):
        client = Client(uri).connect()
        with RTCClient(client, prefix_steps=3, period_steps=6, action_interval=0.03) as rtc:
            rtc.reset()
            actual = []
            deadline = time.monotonic() + 4
            while len(actual) < 22 and time.monotonic() < deadline:
                if rtc.needs_observation:
                    rtc.submit({"state": np.zeros(2, np.float32)})
                action = rtc.next_action()
                if action is not None:
                    actual.append(action.step)
                time.sleep(0.001)
            assert actual == list(range(22))
            assert sum(e["event"] == "switch" for e in rtc.drain_events()) >= 3
        with pytest.raises(RTCError, match="closed"):
            rtc.next_action()


def test_close_interrupts_pending_call(runtime):
    rtc, client, _ = runtime
    client.release.clear()
    rtc.submit({"state": np.zeros(2, np.float32)})
    client.started.get(timeout=1)
    rtc.close()
    assert client.closed
    with pytest.raises(RTCError):
        rtc.next_action()


def test_preparation_is_before_commit_and_prefix_uses_actual_targets():
    client, clock = ControlledClient(), Clock()

    def prepare(actions):
        return np.clip(actions, 0, 5).astype(np.float32)

    with RTCClient(
        client,
        prefix_steps=3,
        period_steps=6,
        action_interval=1,
        clock=clock,
        prepare_actions=prepare,
    ) as rtc:
        rtc.reset()
        bootstrap(rtc)
        for step in range(6):
            clock.now = 100 + step
            np.testing.assert_array_equal(rtc.next_action().values, [min(step, 5)] * 2)
        clock.now = 106
        rtc.submit({"state": np.zeros(2, np.float32)})
        wait_result(rtc)
        np.testing.assert_array_equal(client.calls[-1]["action_prefix"], np.full((3, 2), 5))
        for step in range(6, 10):
            clock.now = 100 + step
            np.testing.assert_array_equal(rtc.next_action().values, [5, 5])


def test_preparation_cannot_hide_corrupt_server_prefix():
    client, clock = ControlledClient(), Clock()
    with RTCClient(
        client,
        prefix_steps=3,
        period_steps=6,
        action_interval=1,
        clock=clock,
        prepare_actions=lambda a: np.clip(a, 0, 0),
    ) as rtc:
        rtc.reset()
        bootstrap(rtc)
        for step in range(6):
            clock.now = 100 + step
            rtc.next_action()
        clock.now = 106
        client.corrupt = True
        rtc.submit({"state": np.zeros(2, np.float32)})
        with pytest.raises(RTCError, match="Service changed"):
            wait_result(rtc)


def test_preparation_must_preserve_previously_committed_prefix():
    client, clock = ControlledClient(), Clock()
    with RTCClient(
        client,
        prefix_steps=3,
        period_steps=6,
        action_interval=1,
        clock=clock,
        prepare_actions=lambda a: a + 1,
    ) as rtc:
        rtc.reset()
        bootstrap(rtc)
        for step in range(6):
            clock.now = 100 + step
            rtc.next_action()
        clock.now = 106
        rtc.submit({"state": np.zeros(2, np.float32)})
        with pytest.raises(RTCError, match="already committed"):
            wait_result(rtc)


def test_capture_from_previous_step_is_not_a_current_observation(runtime):
    rtc, client, clock = runtime
    to_request(rtc, clock)
    with pytest.raises(RTCError, match="before the current"):
        rtc.submit({"state": np.zeros(2, np.float32)}, captured_at=105.5)
    assert len(client.calls) == 1 and rtc.step == 6


@pytest.mark.parametrize(
    "prepare", [lambda a: a.astype(np.float64), lambda a: a[:1], lambda a: a * np.nan]
)
def test_invalid_preparation_never_commits_motion(prepare):
    client, clock = ControlledClient(), Clock()
    with RTCClient(
        client,
        prefix_steps=3,
        period_steps=6,
        action_interval=1,
        clock=clock,
        prepare_actions=prepare,
    ) as rtc:
        rtc.reset()
        rtc.submit({"state": np.zeros(2, np.float32)})
        with pytest.raises(RTCError, match="preserve finite"):
            wait_result(rtc)
        assert rtc.step == 0


def test_relative_actions_rejected_even_when_both_spaces_match():
    client = ControlledClient()
    declared = client.describe()
    action = replace(declared.outputs.fields["action"], semantic="relative_joint_deltas")
    prefix = replace(declared.inputs.fields["action_prefix"], semantic="relative_joint_deltas")
    client.describe = lambda: replace(
        declared,
        outputs=ObjectSpec({"action": action}),
        inputs=replace(declared.inputs, fields={**declared.inputs.fields, "action_prefix": prefix}),
    )
    with pytest.raises(ValueError, match="absolute joint"):
        RTCClient(client, prefix_steps=3, period_steps=6, action_interval=1)
    assert client.resets == 0 and client.calls == []
