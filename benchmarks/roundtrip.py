"""Linux loopback benchmark, separate server process, no model or socket baseline.

uv run benchmarks/roundtrip.py --iterations 100 --output docs/benchmark-results.json
"""

import argparse
import base64
import json
import logging
import multiprocessing as mp
import platform
import resource
import socket
import sys
import time
from pathlib import Path

import msgpack
import numpy as np
import websockets

from inferport import Backend, Client, InferenceSpec, ObjectSpec, ScalarSpec, TensorSpec, serve
from inferport.codec import MAX_MESSAGE_BYTES, decode, encode


class Echo(Backend):
    def describe(self):
        fields = {
            "state": TensorSpec("float32", (32,)),
            "image": TensorSpec("uint8", (224, 224, 3)),
            "images": TensorSpec("uint8", (3, 480, 640, 3)),
            "batch_images": TensorSpec("uint8", (8, 3, 224, 224, 3)),
            "batch_state": TensorSpec("float32", (8, 32)),
            "array": TensorSpec("uint8", (MAX_MESSAGE_BYTES - 1024,)),
            "stats": ScalarSpec("boolean"),
        }
        outputs = {
            **fields,
            "cpu": ScalarSpec("number"),
            "peak_rss_kib": ScalarSpec("integer"),
        }
        return InferenceSpec(
            ObjectSpec(fields, optional=tuple(fields)),
            ObjectSpec(outputs, optional=tuple(outputs)),
        )

    def infer(self, inputs):
        if inputs.get("stats") is True:
            return {
                "cpu": time.process_time(),
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
        return inputs


def run_server(port, stop):
    logging.getLogger("websockets.server").setLevel(logging.CRITICAL)
    serve(Echo(), port=port, stop_event=stop)


def old_json(value):
    if isinstance(value, np.ndarray):
        return {
            "__numpy_array__": True,
            "dtype": str(value.dtype),
            "shape": value.shape,
            "data": base64.b64encode(value.tobytes()).decode("ascii"),
        }
    raise TypeError


def old_hook(value):
    if value.get("__numpy_array__"):
        return np.frombuffer(base64.b64decode(value["data"]), dtype=value["dtype"]).reshape(
            value["shape"]
        )
    return value


def timed(function, iterations):
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000)
    return float(np.median(samples))


def cases():
    yield "state", {"state": np.ones(32, dtype=np.float32)}
    yield "image224", {"image": np.zeros((224, 224, 3), dtype=np.uint8)}
    yield "three_images640", {"images": np.zeros((3, 480, 640, 3), dtype=np.uint8)}
    yield (
        "batch8",
        {
            "batch_images": np.zeros((8, 3, 224, 224, 3), dtype=np.uint8),
            "batch_state": np.ones((8, 32), dtype=np.float32),
        },
    )
    yield "near64MiB", {"array": np.zeros(MAX_MESSAGE_BYTES - 1024, dtype=np.uint8)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 10:
        parser.error("use at least 10 iterations")
    if sys.platform != "linux":
        parser.error("this RSS measurement currently targets Linux")
    context = mp.get_context("spawn")
    stop = context.Event()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = context.Process(target=run_server, args=(port, stop))
    process.start()
    results = []
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                if time.monotonic() > deadline or not process.is_alive():
                    raise RuntimeError("server didn't start") from None
                time.sleep(0.01)
        with Client(f"ws://127.0.0.1:{port}") as client:
            for name, payload in cases():
                envelope = {"id": 1, "op": "infer", "data": payload}
                encoded = encode(envelope)
                old = json.dumps(envelope, default=old_json).encode()
                # Limit the expensive, separate codec samples; network samples use N.
                codec_n = min(args.iterations, 10)
                metrics = {
                    "case": name,
                    "iterations": args.iterations,
                    "message_bytes": len(encoded),
                    "json_bytes": len(old),
                    "encode_ms": timed(lambda value=envelope: encode(value), codec_n),
                    "decode_ms": timed(lambda value=encoded: decode(value), codec_n),
                    "json_encode_ms": timed(
                        lambda value=envelope: json.dumps(value, default=old_json).encode(), codec_n
                    ),
                    "json_decode_ms": timed(
                        lambda value=old: json.loads(value.decode(), object_hook=old_hook), codec_n
                    ),
                }
                del old, encoded
                for _ in range(3):
                    client.infer(payload)
                before = client.infer({"stats": True})
                cpu_before, wall_before = time.process_time(), time.perf_counter()
                samples = []
                for _ in range(args.iterations):
                    start = time.perf_counter()
                    result = client.infer(payload)
                    samples.append((time.perf_counter() - start) * 1000)
                    del result
                elapsed, cpu = time.perf_counter() - wall_before, time.process_time() - cpu_before
                after = client.infer({"stats": True})
                metrics.update(
                    {
                        "roundtrip_p50_ms": float(np.percentile(samples, 50)),
                        "roundtrip_p95_ms": float(np.percentile(samples, 95)),
                        "roundtrip_p99_ms": float(np.percentile(samples, 99)),
                        "roundtrips_per_second": args.iterations / elapsed,
                        "payload_MiB_per_second_both_directions": 2
                        * metrics["message_bytes"]
                        * args.iterations
                        / elapsed
                        / 2**20,
                        "client_cpu_seconds": cpu,
                        "server_cpu_seconds": after["cpu"] - before["cpu"],
                        "client_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        "server_peak_rss_kib": after["peak_rss_kib"],
                    }
                )
                results.append(metrics)
                print(json.dumps(metrics), flush=True)
    finally:
        stop.set()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join()
            raise RuntimeError("server failed to stop")
    report = {
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "msgpack": msgpack.__version__,
            "websockets": websockets.__version__,
            "platform": platform.platform(),
        },
        "method": (
            "Two processes, Linux loopback, echo backend, no TLS/compression. "
            "RSS is cumulative process high-water including separate JSON codec samples; "
            "no TCP baseline. Network percentiles include local codec, 3 warmups per case."
        ),
        "results": results,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
