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

from inferport import Backend, Client, serve


class Echo(Backend):
    def infer(self, inputs):
        return inputs


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


def interchange(server_python, client_python):
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
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=temp
        )
        try:
            wait_listening(port, process)
            result = subprocess.run(
                [client_python, script, "--role", "client", "--port", str(port)],
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
                {"server": json.loads(out), "client": json.loads(result.stdout), "result": "passed"}
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-a", default=sys.executable)
    parser.add_argument("--python-b", default=sys.executable)
    parser.add_argument("--role", choices=["server", "client"])
    parser.add_argument("--port", type=int)
    parser.add_argument("--stop-file", type=Path)
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
            serve(Echo(), port=args.port, stop_event=stop)
        finally:
            stop.set()
            watcher.join()
        print(json.dumps(versions()))
    elif args.role == "client":
        payload = {
            "image": np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3),
            "state": np.arange(64, dtype=">f4").reshape(4, 16)[:, ::2],
            "batch": np.ones((8, 4, 7), dtype=np.float64),
            "context": {"text": "测试", "raw": b"\x00\xff", "scalar": np.int32(3)},
        }
        with Client(f"ws://127.0.0.1:{args.port}") as client:
            client.reset({"instruction": "trial"})
            for _ in range(3):
                result = client.infer(payload)
                for name in ("image", "state", "batch"):
                    np.testing.assert_array_equal(result[name], payload[name])
                    assert result[name].flags.writeable and result[name].dtype.isnative
                assert result["context"] == payload["context"]
        print(json.dumps(versions()))
    else:
        a, b = os.path.abspath(args.python_a), os.path.abspath(args.python_b)
        interchange(a, b)
        interchange(b, a)


if __name__ == "__main__":
    main()
