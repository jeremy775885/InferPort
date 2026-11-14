"""Trained-prefix scheduling, real transport, and lifecycle regressions."""

import copy
import threading
import time
from dataclasses import replace
from queue import Queue

import numpy as np
import pytest
from inferport import Backend, Client, Dimension, InferenceSpec, ObjectSpec, TensorSpec
from inferport.rtc import RTCClient, RTCError


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


def to_request(rtc, clock):
    bootstrap(rtc)
    for index in range(6):
        clock.now = 100 + index
        target = rtc.next_action()
        assert target.step == index
        np.testing.assert_array_equal(target.values, [index, index])
    clock.now = 106
    assert rtc.needs_observation


@pytest.mark.parametrize("prefix,period", [(0, 6), (True, 6), (5, 6), (3, 3), (3, 10)])
def test_configuration_rejected_before_rpc(prefix, period):
    client = ControlledClient()
    with pytest.raises(ValueError):
        RTCClient(client, prefix_steps=prefix, period_steps=period, action_interval=1)
    assert client.resets == 0 and client.calls == []


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
