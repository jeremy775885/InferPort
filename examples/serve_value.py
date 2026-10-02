"""Run with: uv run examples/serve_value.py [--port 8000]."""

import argparse

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(ValueBackend(), port=args.port)
