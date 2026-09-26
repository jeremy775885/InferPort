import concurrent.futures
import threading

import numpy as np
import pytest

from inferport import (
    Backend,
    Client,
    Error,
    InvalidInput,
    RemoteError,
    RequestTimeout,
    TransportError,
    serve,
)


class Counter(Backend):
    def __init__(self):
        self.condition = threading.Condition()
        self.calls = []
        self.threads = set()
        self.value = 0
        self.context = {}
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()
        self.fail_reset_at = None
        self.reset_count = 0

    def record(self, kind):
        with self.condition:
            self.calls.append(kind)
            self.threads.add(threading.get_ident())
            self.condition.notify_all()

    def reset(self, context):
        self.reset_count += 1
        self.record("reset")
        if self.reset_count == self.fail_reset_at:
            raise RuntimeError("reset failed")
        self.value = 0
        self.context = context.copy()

    def infer(self, inputs):
        self.record("infer")
        if inputs.get("invalid"):
            raise InvalidInput("input must be valid")
        if inputs.get("crash"):
            raise RuntimeError("private internal path")
        if inputs.get("bad_output"):
            return object()
        if inputs.get("block"):
            self.entered.set()
            assert self.release.wait(10)
        self.value += 1
        return {"value": self.value, "context": self.context}

    def close(self):
        self.record("close")
        self.closed.set()

    def wait_calls(self, kind, count):
        with self.condition:
            assert self.condition.wait_for(lambda: self.calls.count(kind) >= count, timeout=5)


def test_roundtrip_and_local_validation(server):
    with server() as (uri, _, _, errors), Client(uri) as client:
        assert client.connect() is client
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        batch = np.ones((8, 16), dtype=np.float32)
        result = client.infer({"image": image, "batch": batch})
        np.testing.assert_array_equal(result["image"], image)
        np.testing.assert_array_equal(result["batch"], batch)
        with pytest.raises(TypeError):
            client.infer({"bad": object()})
        with pytest.raises(TypeError):
            client.infer([])
        with pytest.raises(ValueError):
            client.infer({}, timeout=0)
        assert client.infer({"ok": 1}) == {"ok": 1}
        client.reset()
    assert not errors


def test_state_reset_invalid_input_busy_and_close(server):
    backend = Counter()
    with server(backend) as (uri, _, _, errors):
        with Client(uri) as client:
            client.reset({"instruction": "first"})
            assert client.infer({}) == {"value": 1, "context": {"instruction": "first"}}
            with pytest.raises(RemoteError) as busy:
                Client(uri).connect()
            assert busy.value.code == "busy" and busy.value.request_id == 0
            with pytest.raises(RemoteError) as bad:
                client.infer({"invalid": True})
            assert not bad.value.fatal and bad.value.message == "input must be valid"
            assert client.infer({})["value"] == 2
            client.reset({"new": 1})
            assert client.infer({}) == {"value": 1, "context": {"new": 1}}
        backend.wait_calls("reset", 4)
        with Client(uri) as client:
            assert client.infer({}) == {"value": 1, "context": {}}
    assert not errors
    assert backend.calls.count("close") == 1
    assert len(backend.threads) == 1


@pytest.mark.parametrize(
    "input_key,code", [("crash", "backend_error"), ("bad_output", "invalid_output")]
)
def test_fatal_backend_error_recovers_after_cleanup(server, input_key, code):
    backend = Counter()
    with server(backend) as (uri, _, _, errors):
        client = Client(uri).connect()
        with pytest.raises(RemoteError) as error:
            client.infer({input_key: True})
        assert error.value.code == code and error.value.fatal
        assert "private" not in str(error.value)
        with pytest.raises(RuntimeError):
            client.infer({})
        backend.wait_calls("reset", 2)
        with Client(uri) as new:
            assert new.infer({})["value"] == 1
    assert not errors


def test_timeout_holds_ownership_until_actual_work_and_cleanup(server):
    backend = Counter()
    with server(backend) as (uri, _, _, errors):
        client = Client(uri).connect()
        try:
            with pytest.raises(RequestTimeout) as error:
                client.infer({"block": True}, timeout=0.1)
            assert error.value.stage == "response" and backend.entered.is_set()
            assert backend.calls == ["reset", "infer"]
            with pytest.raises(RemoteError) as busy:
                Client(uri, open_timeout=1).connect()
            assert busy.value.code == "busy"
            with pytest.raises(RuntimeError):
                client.connect()
        finally:
            backend.release.set()
        backend.wait_calls("reset", 2)
        with Client(uri) as new:
            assert new.infer({})["value"] == 1
    assert not errors


def test_concurrent_call_rejected_close_interrupts_wait(server):
    backend = Counter()
    with server(backend) as (uri, _, _, errors), concurrent.futures.ThreadPoolExecutor() as pool:
        client = Client(uri).connect()
        future = pool.submit(client.infer, {"block": True})
        try:
            assert backend.entered.wait(5)
            with pytest.raises(RuntimeError):
                client.infer({})
            client.close()
            with pytest.raises(TransportError):
                future.result(timeout=2)
            assert backend.calls == ["reset", "infer"]
        finally:
            backend.release.set()
    assert not errors


@pytest.mark.parametrize("fail_at", [1, 2])
def test_failed_reset_stops_service_and_closes_once(server, fail_at):
    backend = Counter()
    backend.fail_reset_at = fail_at
    with server(backend) as (uri, _, thread, errors):
        if fail_at == 1:
            with pytest.raises((RemoteError, TransportError)):
                Client(uri).connect()
        else:
            Client(uri).connect().close()
        thread.join(5)
        assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], Error)
    assert backend.calls.count("close") == 1


def test_startup_failure_owns_backend():
    backend = Counter()
    with pytest.raises(ValueError):
        serve(backend, send_timeout=-1)
    assert backend.calls == ["close"]


def test_constructor_and_closed_client():
    client = Client("ws://127.0.0.1:1")
    with pytest.raises(RuntimeError):
        client.infer({})
    client.close()
    client.close()
    with pytest.raises(RuntimeError):
        client.connect()


def test_stop_waits_for_backend(server):
    backend = Counter()
    with (
        server(backend) as (uri, stop, thread, errors),
        concurrent.futures.ThreadPoolExecutor() as pool,
    ):
        client = Client(uri).connect()
        future = pool.submit(client.infer, {"block": True})
        try:
            assert backend.entered.wait(5)
            stop.set()
            with pytest.raises(TransportError):
                future.result(timeout=3)
            assert thread.is_alive() and not backend.closed.is_set()
        finally:
            backend.release.set()
    assert not errors
    assert backend.calls == ["reset", "infer", "reset", "close"]


def test_disconnect_during_initialization_keeps_owner(server):
    class SlowInit(Counter):
        def reset(self, context):
            if self.reset_count == 0:
                self.entered.set()
                assert self.release.wait(5)
            super().reset(context)

    backend = SlowInit()
    with server(backend) as (uri, _, _, errors):
        try:
            with pytest.raises(RequestTimeout) as timeout:
                Client(uri, open_timeout=0.15).connect()
            assert timeout.value.stage == "ready"
            assert backend.entered.is_set()
            with pytest.raises(RemoteError) as busy:
                Client(uri).connect()
            assert busy.value.code == "busy"
        finally:
            backend.release.set()
        backend.wait_calls("reset", 2)
    assert not errors and "infer" not in backend.calls


def test_keyboard_interrupt_invalidates_connection(server, monkeypatch):
    with server() as (uri, _, _, _):
        client = Client(uri).connect()

        def interrupt(coroutine):
            coroutine.close()
            raise KeyboardInterrupt

        monkeypatch.setattr(client, "_submit", interrupt)
        with pytest.raises(KeyboardInterrupt):
            client.infer({})
        with pytest.raises(RuntimeError):
            client.infer({})


def test_request_count_does_not_grow_threads_or_descriptors(server):
    import os

    with server() as (uri, _, _, _), Client(uri) as client:
        thread_count = len(threading.enumerate())
        descriptor_path = "/proc/self/fd"
        fd_count = len(os.listdir(descriptor_path)) if os.path.isdir(descriptor_path) else None
        for i in range(500):
            assert client.infer({"id": i}) == {"id": i}
        assert len(threading.enumerate()) == thread_count
        if fd_count is not None:
            assert len(os.listdir(descriptor_path)) <= fd_count


def test_cleanup_still_rejects_new_owners(server):
    class SlowCleanup(Counter):
        def reset(self, context):
            if self.reset_count == 1:
                self.entered.set()
                assert self.release.wait(5)
            super().reset(context)

    backend = SlowCleanup()
    with server(backend) as (uri, _, _, errors):
        try:
            Client(uri).connect().close()
            assert backend.entered.wait(5)
            with pytest.raises(RemoteError) as busy:
                Client(uri).connect()
            assert busy.value.code == "busy"
        finally:
            backend.release.set()
    assert not errors


@pytest.mark.parametrize("message", ["x" * 1024, "\ud800"])
def test_invalid_input_diagnostic_cannot_break_small_message_budget(server, message):
    class Validation(Backend):
        def infer(self, inputs):
            if inputs:
                raise InvalidInput(message)
            return {}

    with server(Validation(), max_message_bytes=128) as (uri, _, _, errors):
        with Client(uri, max_message_bytes=128) as client:
            with pytest.raises(RemoteError) as error:
                client.infer({"invalid": True})
            assert not error.value.fatal and error.value.message == "Input rejected"
            assert client.infer({}) == {}
    assert not errors
