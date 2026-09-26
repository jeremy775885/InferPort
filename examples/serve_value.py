"""Run with: uv run examples/serve_value.py [--port 8000]."""

import argparse

import numpy as np

from inferport import Backend, InvalidInput, serve


class ValueBackend(Backend):
    def infer(self, inputs):
        state = inputs.get("state")
        if not isinstance(state, np.ndarray) or state.ndim != 2 or state.dtype.kind != "f":
            raise InvalidInput("state must be a floating point array with shape [B, D]")
        return {"value": np.sum(state * state, axis=-1)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(ValueBackend(), port=args.port)
