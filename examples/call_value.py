"""Run after serve_value.py: uv run examples/call_value.py [--port 8000]."""

import argparse

import numpy as np

from inferport import Client

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    with Client(f"ws://127.0.0.1:{args.port}") as client:
        result = client.infer({"state": np.ones((4, 7), dtype=np.float32)})
        print(result["value"])  # [7. 7. 7. 7.]
