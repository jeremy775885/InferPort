import asyncio
import socket
import ssl
import subprocess
import time

import numpy as np
import pytest
from websockets.asyncio.client import connect as async_connect
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.sync.client import connect

from inferport import (
    Client,
    InferenceSpec,
    ObjectSpec,
    ProtocolError,
    RemoteError,
    RequestTimeout,
    TensorSpec,
    TransportError,
    _io,
    codec,
    protocol,
)

from .peers import frame, peer, recv_frame
from .test_lifecycle import Counter


def quiet(sock, release):
    assert release.wait(6)


@pytest.mark.parametrize("upgrade,stage", [(False, "open"), (True, "ready")])
def test_open_deadline_and_cleanup(upgrade, stage):
    with peer(quiet, upgrade=upgrade, ready=False) as uri:
        client = Client(uri, open_timeout=0.15)
        with pytest.raises(RequestTimeout) as error:
            client.connect()
        assert error.value.stage == stage
        with pytest.raises(RuntimeError):
            client.connect()


def test_connection_refused():
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        with pytest.raises(TransportError):
            Client(f"ws://127.0.0.1:{reserved.getsockname()[1]}").connect()


def test_send_deadline_is_real_socket_backpressure():
    with peer(quiet) as uri:
        client = Client(uri).connect()
        start = time.monotonic()
        with pytest.raises(RequestTimeout) as error:
            client.infer({"image": np.zeros(16 * 1024 * 1024, dtype=np.uint8)}, timeout=0.15)
        assert error.value.stage == "send"
        assert time.monotonic() - start < 2


@pytest.mark.parametrize("explicit_close", [True, False])
def test_close_is_bounded_when_write_buffer_is_blocked(monkeypatch, explicit_close):
    monkeypatch.setattr(_io, "CLOSE_TIMEOUT", 0.1)

    async def check(uri):
        async with async_connect(
            uri,
            subprotocols=[protocol.SUBPROTOCOL],
            proxy=None,
            compression=None,
            create_connection=_io.ClientSocket,
            close_timeout=0.1,
        ) as ws:
            await ws.recv()
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(ws.send(bytes(16 * 1024 * 1024)), 0.1)
            start = time.monotonic()
            if explicit_close:
                await ws.close()
        await asyncio.wait_for(ws.wait_closed(), 1)
        assert time.monotonic() - start < 1

    with peer(quiet, describe=False) as uri:
        asyncio.run(check(uri))


@pytest.mark.parametrize(
    "reply",
    [
        codec.encode(protocol.success(3, {})),
        codec.encode({"id": 2, "ok": True, "data": []}),
        b"\xc1",
        "text",
    ],
)
def test_bad_reply_invalidates_client(reply):
    def respond(sock, release):
        recv_frame(sock)
        sock.sendall(
            frame(
                reply.encode() if isinstance(reply, str) else reply,
                1 if isinstance(reply, str) else 2,
            )
        )
        release.wait(5)

    with peer(respond) as uri:
        client = Client(uri).connect()
        with pytest.raises(ProtocolError):
            client.infer({})
        with pytest.raises(RuntimeError):
            client.infer({})


def test_response_interrupted():
    def respond(sock, release):
        recv_frame(sock)
        sock.sendall(b"\x82\x7e\x10\x00truncated")

    with peer(respond) as uri, Client(uri) as client:
        with pytest.raises(TransportError):
            client.infer({})


def test_auth_origin_and_protocol_do_not_touch_backend(server):
    backend = Counter()
    with server(backend, token="secret") as (uri, _, _, errors):
        with pytest.raises(RemoteError) as error:
            Client(uri, token="wrong").connect()
        assert error.value.code == "unauthorized" and error.value.request_id is None
        for options in (
            {},
            {"subprotocols": ["inferport.v99"]},
            {"subprotocols": [protocol.SUBPROTOCOL], "origin": "https://example.com"},
        ):
            with pytest.raises(InvalidStatus):
                connect(
                    uri,
                    additional_headers={"Authorization": "Bearer secret"},
                    proxy=None,
                    **options,
                )
        assert backend.calls == []
        with Client(uri, token="secret") as client:
            assert client.infer({})["value"] == 1
            with pytest.raises(RemoteError):
                Client(uri).connect()
            assert client.infer({})["value"] == 2
    assert not errors


@pytest.mark.parametrize(
    "raw,code",
    [
        (codec.encode({"id": 1, "op": "eval", "data": {}}), "unsupported_operation"),
        (codec.encode({"id": 1, "op": 7, "data": {}}), "protocol_error"),
        (codec.encode({"id": True, "op": "infer", "data": {}}), None),
        (b"\xc1", None),
        ("text", None),
    ],
)
def test_malformed_client_is_isolated(server, raw, code):
    backend = Counter()
    with server(backend) as (uri, _, _, errors):
        with connect(uri, subprotocols=[protocol.SUBPROTOCOL], proxy=None) as ws:
            ws.recv()
            ws.send(raw)
            if code:
                assert codec.decode(ws.recv())["error"]["code"] == code
            with pytest.raises(ConnectionClosed):
                ws.recv()
        backend.wait_calls("reset", 2)
        with Client(uri) as client:
            assert client.infer({})["value"] == 1
    assert not errors and backend.calls.count("infer") == 1


def test_duplicate_request_id_and_size_limit(server):
    with server(max_message_bytes=256) as (uri, _, _, _):
        with connect(uri, subprotocols=[protocol.SUBPROTOCOL], proxy=None) as ws:
            ws.recv()
            message = codec.encode({"id": 1, "op": "infer", "data": {}})
            ws.send(message)
            assert codec.decode(ws.recv())["ok"]
            ws.send(message)
            assert codec.decode(ws.recv())["error"]["code"] == "protocol_error"
            with pytest.raises(ConnectionClosed):
                ws.recv()
    with server(max_message_bytes=256) as (uri, _, _, _):
        with connect(uri, subprotocols=[protocol.SUBPROTOCOL], proxy=None) as ws:
            ws.recv()
            ws.send(bytes(1024))
            with pytest.raises(ConnectionClosed) as error:
                ws.recv()
            assert error.value.rcvd.code == 1009


def test_server_send_timeout_cleans_up(server):
    class Large(Counter):
        def describe(self):
            return InferenceSpec(
                ObjectSpec(), ObjectSpec({"image": TensorSpec("uint8", (16 * 1024 * 1024,))})
            )

        def infer(self, inputs):
            self.record("infer")
            return {"image": np.zeros(16 * 1024 * 1024, dtype=np.uint8)}

    backend = Large()
    with server(backend, send_timeout=0.15) as (uri, _, _, errors):
        # A raw TCP socket with a small receive window and no reader after READY.
        port = int(uri.rsplit(":", 1)[1])
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
            sock.connect(("127.0.0.1", port))
            # sync websockets starts a reader, so use a manually masked wire request.
            from .peers import headers

            sock.sendall(
                b"GET / HTTP/1.1\r\nHost: localhost\r\n"
                b"Upgrade: websocket\r\nConnection: Upgrade\r\n"
                b"Sec-WebSocket-Version: 13\r\nSec-WebSocket-Key: MDEyMzQ1Njc4OWFiY2RlZg==\r\n"
                b"Sec-WebSocket-Protocol: inferport\r\n\r\n"
            )
            headers(sock)
            recv_frame(sock)
            request = codec.encode({"id": 1, "op": "infer", "data": {}})
            sock.sendall(bytes([0x82, 0x80 | len(request)]) + bytes(4) + request)
            backend.wait_calls("reset", 2)
    assert not errors


def test_shutdown_bounds_library_initiated_close(server, monkeypatch):
    import threading

    monkeypatch.setattr(_io, "CLOSE_TIMEOUT", 0.1)
    sending = threading.Event()
    original_send = _io.ServerSocket.send

    async def observed_send(ws, message, *args, **kwargs):
        if isinstance(message, bytes) and len(message) > 1024 * 1024:
            sending.set()
        return await original_send(ws, message, *args, **kwargs)

    monkeypatch.setattr(_io.ServerSocket, "send", observed_send)

    class Large(Counter):
        def describe(self):
            return InferenceSpec(
                ObjectSpec(), ObjectSpec({"image": TensorSpec("uint8", (16 * 1024 * 1024,))})
            )

        def infer(self, inputs):
            return {"image": np.zeros(16 * 1024 * 1024, dtype=np.uint8)}

    from .peers import headers

    with server(Large()) as (uri, stop, thread, errors), socket.socket() as sock:
        sock.settimeout(5)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        sock.connect(("127.0.0.1", int(uri.rsplit(":", 1)[1])))
        sock.sendall(
            b"GET / HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n"
            b"Connection: Upgrade\r\nSec-WebSocket-Version: 13\r\n"
            b"Sec-WebSocket-Key: MDEyMzQ1Njc4OWFiY2RlZg==\r\n"
            b"Sec-WebSocket-Protocol: inferport\r\n\r\n"
        )
        headers(sock)
        recv_frame(sock)
        request = codec.encode({"id": 1, "op": "infer", "data": {}})
        sock.sendall(bytes([0x82, 0x80 | len(request)]) + bytes(4) + request)
        assert sending.wait(3)
        stop.set()
        thread.join(2)
        assert not thread.is_alive()
    assert not errors


def test_tls_verification_and_private_ca(server, tmp_path):
    key, cert = tmp_path / "key.pem", tmp_path / "cert.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost",
        ],
        check=True,
        capture_output=True,
    )
    server_ssl = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ssl.load_cert_chain(cert, key)
    client_ssl = ssl.create_default_context(cafile=str(cert))
    with server(ssl=server_ssl, token="secret") as (uri, _, _, errors):
        secure_uri = uri.replace("ws://127.0.0.1", "wss://localhost")
        with pytest.raises(TransportError):
            Client(secure_uri, token="secret").connect()
        with pytest.raises(TransportError):
            Client(uri.replace("ws:", "wss:"), token="secret", ssl=client_ssl).connect()
        with Client(secure_uri, token="secret", ssl=client_ssl) as client:
            assert client.infer({"ok": 1}) == {"ok": 1}
    assert not errors
