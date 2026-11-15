"""Linux loopback benchmark, separate server process, no model or socket baseline.

uv run benchmarks/roundtrip.py --iterations 100 --output docs/benchmark-results.json
uv run benchmarks/roundtrip.py --mode policy --iterations 1000 --budget-ms 100
"""

import argparse
import base64
import copy
import json
import logging
import math
import multiprocessing as mp
import platform
import resource
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import msgpack
import numpy as np
import websockets

from inferport import (
    Backend,
    Client,
    InferenceSpec,
    ObjectSpec,
    ScalarSpec,
    TensorSpec,
    __version__,
    serve,
)
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


class SyntheticPolicy(Backend):
    """Fixed-size image input and small action output; no learned computation."""

    def __init__(self, image_shape, horizon, action_width, prefix_steps):
        self.image_shape = image_shape
        self.horizon, self.action_width, self.prefix_steps = horizon, action_width, prefix_steps

    def describe(self):
        inputs = {
            "images": TensorSpec("uint8", self.image_shape),
            "state": TensorSpec("float32", (self.action_width,)),
            "action_prefix": TensorSpec("float32", (self.prefix_steps, self.action_width)),
            "stats": ScalarSpec("boolean"),
        }
        outputs = {
            "action": TensorSpec("float32", (self.horizon, self.action_width)),
            "cpu": ScalarSpec("number"),
            "peak_rss_kib": ScalarSpec("integer"),
        }
        return InferenceSpec(
            ObjectSpec(inputs, optional=tuple(inputs)),
            ObjectSpec(outputs, optional=tuple(outputs)),
        )

    def infer(self, inputs):
        if inputs.get("stats") is True:
            return {
                "cpu": time.process_time(),
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
        actions = np.zeros((self.horizon, self.action_width), dtype=np.float32)
        actions[: self.prefix_steps] = inputs["action_prefix"]
        return {"action": actions}


def run_server(port, stop, backend):
    logging.getLogger("websockets.server").setLevel(logging.CRITICAL)
    serve(backend, port=port, stop_event=stop)


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
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("echo", "policy"), default="echo")
    parser.add_argument("--cameras", type=int, default=3)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--action-width", type=int, default=14)
    parser.add_argument("--prefix-steps", type=int, default=5)
    parser.add_argument(
        "--budget-ms", type=float, help="policy snapshot + RPC budget; not an RTC control deadline"
    )
    args = parser.parse_args()
    if args.iterations < 10:
        parser.error("use at least 10 iterations")
    if sys.platform != "linux":
        parser.error("this RSS measurement currently targets Linux")
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")
    if any(
        n <= 0 for n in (args.cameras, args.height, args.width, args.horizon, args.action_width)
    ):
        parser.error("image and action dimensions must be positive")
    if not 0 <= args.prefix_steps < args.horizon:
        parser.error("prefix-steps must be nonnegative and less than horizon")
    if args.budget_ms is not None and (
        args.mode != "policy" or not math.isfinite(args.budget_ms) or args.budget_ms <= 0
    ):
        parser.error("budget-ms requires policy mode and must be positive and finite")
    image_shape = (args.cameras, args.height, args.width, 3)
    if args.mode == "policy":
        # Bound allocations before constructing a user-sized synthetic payload.
        input_bytes = math.prod(image_shape) + 4 * args.action_width * (1 + args.prefix_steps)
        output_bytes = 4 * args.horizon * args.action_width
        if max(input_bytes, output_bytes) > MAX_MESSAGE_BYTES - 4096:
            parser.error("policy payload exceeds the message limit (including metadata reserve)")
        backend = SyntheticPolicy(image_shape, args.horizon, args.action_width, args.prefix_steps)
        payloads = [
            (
                "images_to_actions",
                {
                    "images": np.zeros(image_shape, dtype=np.uint8),
                    "state": np.ones(args.action_width, dtype=np.float32),
                    "action_prefix": np.ones(
                        (args.prefix_steps, args.action_width), dtype=np.float32
                    ),
                },
            )
        ]
    else:
        backend, payloads = Echo(), cases()
    context = mp.get_context("spawn")
    stop = context.Event()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = context.Process(target=run_server, args=(port, stop, backend))
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
            for name, payload in payloads:
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
                for _ in range(args.warmup):
                    client.infer(copy.deepcopy(payload) if args.mode == "policy" else payload)
                response = client.infer(payload)
                response_envelope = {"id": 1, "ok": True, "data": response}
                metrics["response_message_bytes"] = len(encode(response_envelope))
                if args.mode == "policy":
                    assert response["action"].shape == (args.horizon, args.action_width)
                    np.testing.assert_array_equal(
                        response["action"][: args.prefix_steps], payload["action_prefix"]
                    )
                del response, response_envelope
                before = client.infer({"stats": True})
                cpu_before, wall_before = time.process_time(), time.perf_counter()
                samples = []
                snapshot_samples, total_samples = [], []
                for _ in range(args.iterations):
                    total_start = time.perf_counter()
                    sent = copy.deepcopy(payload) if args.mode == "policy" else payload
                    start = time.perf_counter()
                    result = client.infer(sent)
                    finished = time.perf_counter()
                    samples.append((finished - start) * 1000)
                    snapshot_samples.append((start - total_start) * 1000)
                    total_samples.append((finished - total_start) * 1000)
                    del result, sent
                elapsed, cpu = time.perf_counter() - wall_before, time.process_time() - cpu_before
                after = client.infer({"stats": True})
                metrics.update(
                    {
                        "roundtrip_p50_ms": float(np.percentile(samples, 50)),
                        "roundtrip_p95_ms": float(np.percentile(samples, 95)),
                        "roundtrip_p99_ms": float(np.percentile(samples, 99)),
                        "roundtrips_per_second": args.iterations / elapsed,
                        "payload_MiB_per_second_both_directions": (
                            metrics["message_bytes"] + metrics["response_message_bytes"]
                        )
                        * args.iterations
                        / elapsed
                        / 2**20,
                        "client_cpu_seconds": cpu,
                        "server_cpu_seconds": after["cpu"] - before["cpu"],
                        "client_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        "server_peak_rss_kib": after["peak_rss_kib"],
                    }
                )
                if args.mode == "policy":
                    for label, values in (
                        ("snapshot", snapshot_samples),
                        ("snapshot_plus_rpc", total_samples),
                    ):
                        for percentile in (50, 95, 99):
                            metrics[f"{label}_p{percentile}_ms"] = float(
                                np.percentile(values, percentile)
                            )
                    metrics["snapshot_plus_rpc_max_ms"] = max(total_samples)
                    metrics["budget_ms"] = args.budget_ms
                    if args.budget_ms is not None:
                        margins = args.budget_ms - np.asarray(total_samples)
                        metrics["budget_exceeded_count"] = int(np.count_nonzero(margins < 0))
                        metrics["budget_exceeded_fraction"] = float(np.mean(margins < 0))
                        metrics["budget_margin_min_ms"] = float(np.min(margins))
                    # Retain paired samples so tails and later comparisons are auditable.
                    metrics["rpc_samples_ms"] = samples
                    metrics["snapshot_samples_ms"] = snapshot_samples
                    metrics["snapshot_plus_rpc_samples_ms"] = total_samples
                results.append(metrics)
                print(
                    json.dumps({k: v for k, v in metrics.items() if not isinstance(v, list)}),
                    flush=True,
                )
    finally:
        stop.set()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join()
            raise RuntimeError("server failed to stop")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "environment": {
            "inferport": __version__,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "msgpack": msgpack.__version__,
            "websockets": websockets.__version__,
            "platform": platform.platform(),
        },
        "method": (
            f"Two processes, Linux loopback, {args.mode} backend, no TLS/compression. "
            "RSS is cumulative process high-water including separate JSON codec samples; "
            "No TCP baseline. RPC percentiles include local codec; "
            f"{args.warmup} warmups plus one untimed response validation per case. "
            "Policy mode measures synthetic uint8 images in, float32 actions out, with a client "
            "deep-copy snapshot before each RPC. It excludes camera capture, learned inference, "
            "RTC scheduling, action preparation and hardware execution. Budget exceedances "
            "refer only to snapshot + RPC, not real RTC deadline violations."
        ),
        "results": results,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
