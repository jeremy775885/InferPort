import contextlib
import socket
import threading
import time

import pytest

from inferport import Backend, InferenceSpec, ObjectSpec, ScalarSpec, TensorSpec, serve


class Echo(Backend):
    def describe(self):
        fields = {
            "image": TensorSpec("uint8", (480, 640, 3)),
            "batch": TensorSpec("float32", (8, 16)),
            "ok": ScalarSpec("integer"),
            "id": ScalarSpec("integer"),
        }
        payload = ObjectSpec(fields, optional=tuple(fields))
        return InferenceSpec(payload, payload)

    def infer(self, inputs):
        return inputs


@contextlib.contextmanager
def running(backend=None, **kwargs):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    stop = threading.Event()
    errors = []

    def run():
        try:
            serve(backend or Echo(), port=port, stop_event=stop, **kwargs)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, name="test-server", daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    try:
        while thread.is_alive():
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise AssertionError("server didn't start") from None
                stop.wait(0.01)
        if errors:
            raise errors[0]
        yield f"ws://127.0.0.1:{port}", stop, thread, errors
    finally:
        stop.set()
        thread.join(8)
        assert not thread.is_alive(), "server leaked a worker or connection"


@pytest.fixture
def server():
    return running


@pytest.fixture(autouse=True)
def no_client_threads_leaked():
    before = {t.ident for t in threading.enumerate() if t.name.startswith("inferport-")}
    yield
    after = {t.ident for t in threading.enumerate() if t.name.startswith("inferport-")}
    assert after <= before, "InferPort left background threads alive"
