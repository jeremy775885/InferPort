"""Initialize thread-bound model resources on InferPort's backend worker.

Run: uv run examples/thread_bound_backend.py
Call: uv run examples/call_value.py

The small NumPy model stands in for an engine whose creation, inference, reset,
and release must use the same thread.
"""

import threading
from collections.abc import Callable

import numpy as np

from inferport import Backend, Dimension, InferenceSpec, ObjectSpec, Payload, TensorSpec, serve


class ThreadBoundModel:
    """A CPU-only example that enforces a real thread ownership constraint."""

    def __init__(self):
        self._owner = threading.get_ident()
        self._closed = False

    def _check_thread(self) -> None:
        if threading.get_ident() != self._owner:
            raise RuntimeError("Model resource used from a different thread")
        if self._closed:
            raise RuntimeError("Model resource has been closed")

    def reset(self) -> None:
        self._check_thread()
        # Clear engine/processor session state here when adapting a stateful model.

    def predict(self, state: np.ndarray) -> np.ndarray:
        self._check_thread()
        return np.sum(state * state, axis=-1)

    def close(self) -> None:
        self._check_thread()
        self._closed = True


class ThreadBoundBackend(Backend):
    def describe(self) -> InferenceSpec:
        return InferenceSpec(
            ObjectSpec({"state": TensorSpec("float32", (Dimension(), Dimension()))}),
            ObjectSpec({"value": TensorSpec("float32", (Dimension(),))}),
        )

    def __init__(self, model_factory: Callable[[], ThreadBoundModel] = ThreadBoundModel):
        # Store configuration/factories here; don't construct thread-bound handles.
        self._model_factory = model_factory
        self._model: ThreadBoundModel | None = None

    def reset(self, context: Payload) -> None:
        # serve calls reset({}) on its worker before sending READY. Load once there.
        # Subsequent resets clear session state while retaining the model.
        if self._model is None:
            self._model = self._model_factory()
        self._model.reset()

    def infer(self, inputs: Payload) -> Payload:
        state = inputs["state"]
        if self._model is None:
            raise RuntimeError("Backend must be initialized by serve before inference")
        return {"value": self._model.predict(state)}

    def close(self) -> None:
        # A service may stop before any client connects: don't initialize on close.
        if self._model is not None:
            try:
                self._model.close()
            finally:
                self._model = None


if __name__ == "__main__":
    serve(ThreadBoundBackend())
