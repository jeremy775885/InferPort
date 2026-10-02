"""Regression coverage for model adapter data ownership and thread affinity."""

import runpy
import threading
from pathlib import Path

import numpy as np

from inferport import Backend, Client, InferenceSpec, ObjectSpec, TensorSpec

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "thread_bound_backend.py"


def test_reused_backend_buffer_does_not_change_previous_results(server):
    class ReusingBackend(Backend):
        def describe(self):
            return InferenceSpec(
                ObjectSpec(),
                ObjectSpec(
                    {
                        "values": TensorSpec("float32", (2, 4)),
                        "nested": ObjectSpec({"slice": TensorSpec("float32", (2, 2))}),
                    }
                ),
            )

        def __init__(self):
            self.buffer = np.zeros((2, 4), dtype=np.float32)
            self.calls = 0

        def reset(self, context):
            self.calls = 0
            self.buffer.fill(0)

        def infer(self, inputs):
            self.calls += 1
            self.buffer.fill(self.calls)
            return {"values": self.buffer, "nested": {"slice": self.buffer[:, ::2]}}

    with server(ReusingBackend()) as (uri, _, _, errors):
        with Client(uri) as client:
            first = client.infer({})
            second = client.infer({})
            client.reset()
            restarted = client.infer({})
        # Disconnect resets and overwrites the backend buffer again at shutdown.
    assert not errors
    np.testing.assert_array_equal(first["values"], np.ones((2, 4), dtype=np.float32))
    np.testing.assert_array_equal(first["nested"]["slice"], np.ones((2, 2), dtype=np.float32))
    np.testing.assert_array_equal(second["values"], np.full((2, 4), 2, dtype=np.float32))
    np.testing.assert_array_equal(second["nested"]["slice"], np.full((2, 2), 2, dtype=np.float32))
    np.testing.assert_array_equal(restarted["values"], np.ones((2, 4), dtype=np.float32))
    assert first["values"].flags.writeable


def test_lazy_model_creation_and_all_hooks_share_one_worker(server):
    example = runpy.run_path(str(EXAMPLE))
    events = []
    caller_thread = threading.get_ident()

    def record(name):
        events.append((name, threading.get_ident()))

    class TrackedModel(example["ThreadBoundModel"]):
        def __init__(self):
            super().__init__()
            record("load")

        def reset(self):
            super().reset()
            record("reset")

        def predict(self, state):
            result = super().predict(state)
            record("predict")
            return result

        def close(self):
            super().close()
            record("close")

    backend = example["ThreadBoundBackend"](model_factory=TrackedModel)
    assert events == []  # Construction must not load a thread-bound model.
    inputs = {"state": np.ones((4, 7), dtype=np.float32)}
    with server(backend) as (uri, _, _, errors):
        with Client(uri) as client:
            np.testing.assert_array_equal(client.infer(inputs)["value"], np.full(4, 7))
            client.reset()
            np.testing.assert_array_equal(client.infer(inputs)["value"], np.full(4, 7))
    assert not errors
    assert [name for name, _ in events] == [
        "load",
        "reset",
        "predict",
        "reset",
        "predict",
        "reset",
        "close",
    ]
    worker_threads = {ident for _, ident in events}
    assert len(worker_threads) == 1
    assert caller_thread not in worker_threads


def test_stopping_without_a_client_does_not_load_a_model(server):
    example = runpy.run_path(str(EXAMPLE))

    def unexpected_load():
        raise AssertionError("Stopping an unused backend must not initialize its model")

    backend = example["ThreadBoundBackend"](model_factory=unexpected_load)
    with server(backend) as (_, _, _, errors):
        pass
    assert not errors
