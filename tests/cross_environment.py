"""Real two-process interchange; swap interpreters to test both directions.

uv run tests/cross_environment.py --python-a .venv-310/bin/python --python-b .venv/bin/python
"""

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import msgpack
import numpy as np
import websockets

from inferport import (
    Backend,
    Channel,
    Client,
    Dimension,
    InferenceSpec,
    ObjectSpec,
    ScalarSpec,
    TensorSpec,
    check_compatibility,
    serve,
)
from inferport.robotics import joint_target_spec


class Echo(Backend):
    def describe(self):
        payload = ObjectSpec(
            {
                "image": TensorSpec("uint8", (224, 224, 3)),
                "state": TensorSpec("float32", (4, 8)),
                "batch": TensorSpec("float64", (8, 4, 7)),
                "context": ObjectSpec(
                    {
                        "text": ScalarSpec("string"),
                        "scalar": ScalarSpec("integer"),
                        "score": ScalarSpec("number"),
                        "done": ScalarSpec("boolean"),
                    }
                ),
            }
        )
        return InferenceSpec(payload, payload, ObjectSpec({"instruction": ScalarSpec("string")}))

    def infer(self, inputs):
        return {
            **inputs,
            "context": {
                **inputs["context"],
                "scalar": np.int64(inputs["context"]["scalar"]),
                "score": np.float32(inputs["context"]["score"]),
                "done": np.bool_(inputs["context"]["done"]),
            },
        }


def policy_spec(horizon):
    return joint_target_spec(
        cameras={"head": (2, 3)},
        channels=(Channel("joint1", "joint_position", "rad"),),
        horizon=horizon,
    )


class Policy(Backend):
    def describe(self):
        return policy_spec(16)

    def infer(self, inputs):
        return {"action": np.repeat(inputs["state"][None], 16, axis=0)}


def versions():
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "msgpack": msgpack.__version__,
        "websockets": websockets.__version__,
    }


def wait_listening(port, process):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited: {process.communicate()}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.01)
    raise TimeoutError("server startup")


def interchange(server_python, client_python, backend):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory() as temp:
        marker = Path(temp) / "stop"
        script = str(Path(__file__).resolve())
        command = [
            server_python,
            script,
            "--role",
            "server",
            "--port",
            str(port),
            "--stop-file",
            str(marker),
        ]
        extra = ["--backend", backend]
        command.extend(extra)
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=temp
        )
        try:
            wait_listening(port, process)
            result = subprocess.run(
                [client_python, script, "--role", "client", "--port", str(port), *extra],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
                cwd=temp,
            )
        finally:
            marker.touch()
            try:
                out, err = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise
        if process.returncode:
            raise RuntimeError(err)
        print(
            json.dumps(
                {
                    "server": json.loads(out),
                    "client": json.loads(result.stdout),
                    "backend": backend,
                    "result": "passed",
                }
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-a", default=sys.executable)
    parser.add_argument("--python-b", default=sys.executable)
    parser.add_argument("--role", choices=["server", "client"])
    parser.add_argument("--port", type=int)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--backend", choices=["echo", "policy"], default="echo")
    args = parser.parse_args()
    if args.role == "server":
        logging.getLogger("websockets.server").setLevel(logging.CRITICAL)
        stop = threading.Event()

        def watch():
            while not stop.wait(0.02):
                if args.stop_file.exists():
                    stop.set()

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            serve(Policy() if args.backend == "policy" else Echo(), port=args.port, stop_event=stop)
        finally:
            stop.set()
            watcher.join()
        print(json.dumps(versions()))
    elif args.role == "client":
        payload = {
            "image": np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3),
            "state": np.arange(64, dtype=">f4").reshape(4, 16)[:, ::2],
            "batch": np.ones((8, 4, 7), dtype=np.float64),
            "context": {
                "text": "测试",
                "scalar": np.int32(3),
                "score": np.float32(0.5),
                "done": np.bool_(True),
            },
        }
        with Client(f"ws://127.0.0.1:{args.port}") as client:
            if args.backend == "policy":
                declared = client.describe()
                check_compatibility(declared, policy_spec(Dimension(1, 100)))
                client.reset({"instruction": "move"})
                result = client.infer(
                    {
                        "images.head": np.zeros((2, 3, 3), np.uint8),
                        "state": np.array([0.5], np.float32),
                        "instruction": "move",
                    }
                )
                declared.outputs.validate(result)
                np.testing.assert_array_equal(result["action"], np.full((16, 1), 0.5, np.float32))
                print(json.dumps(versions()))
                return
            assert client.describe() == Echo().describe()
            client.reset({"instruction": "trial"})
            for _ in range(3):
                result = client.infer(payload)
                for name in ("image", "state", "batch"):
                    np.testing.assert_array_equal(result[name], payload[name])
                    assert result[name].flags.writeable and result[name].dtype.isnative
                assert result["context"] == payload["context"]
                for name, scalar_type in (("scalar", int), ("score", float), ("done", bool)):
                    assert type(result["context"][name]) is scalar_type
        print(json.dumps(versions()))
    else:
        a, b = os.path.abspath(args.python_a), os.path.abspath(args.python_b)
        for backend in ("echo", "policy"):
            interchange(a, b, backend)
            interchange(b, a, backend)


if __name__ == "__main__":
    main()
