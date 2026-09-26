import importlib.metadata
import json
import subprocess
import sys

import inferport


def test_public_import_has_no_model_framework_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import inferport, sys, json; "
            "assert all(hasattr(inferport, n) for n in ('Client', 'Backend', 'serve')); "
            "print(json.dumps(list(sys.modules)))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    imported = {name.split(".")[0] for name in json.loads(result.stdout)}
    assert not imported.intersection({"torch", "jax", "lerobot", "gymnasium", "rclpy"})
    metadata = importlib.metadata.metadata("inferport")
    assert metadata["Version"] == inferport.__version__
    assert metadata["License-Expression"] == "MIT"
    assert metadata.get_all("License-File") == ["LICENSE"]
    assert metadata["Requires-Python"] == ">=3.10"
    requirements = metadata.get_all("Requires-Dist")
    assert len(requirements) == 3
    assert {item.split("<")[0].split(">=")[0] for item in requirements} == {
        "numpy",
        "msgpack",
        "websockets",
    }


def test_two_independent_processes():
    from pathlib import Path

    script = Path(__file__).with_name("cross_environment.py")
    subprocess.run([sys.executable, str(script)], check=True, capture_output=True, timeout=30)


def test_ctrl_c_closes_backend(tmp_path):
    import os
    import signal
    import socket

    import pytest

    from .cross_environment import wait_listening

    if os.name != "posix":
        pytest.skip("SIGINT process test uses POSIX signal delivery")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    marker = tmp_path / "closed"
    script = tmp_path / "server.py"
    script.write_text(
        "from pathlib import Path\n"
        "from inferport import Backend, serve\n"
        "class B(Backend):\n"
        "    def infer(self, inputs): return inputs\n"
        f"    def close(self): Path({str(marker)!r}).write_text('closed')\n"
        f"serve(B(), port={port})\n"
    )
    process = subprocess.Popen(
        [sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        wait_listening(port, process)
        process.send_signal(signal.SIGINT)
        _, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, stderr.decode()
        assert marker.read_text() == "closed"
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
