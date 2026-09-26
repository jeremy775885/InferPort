"""Controlled wire peers for failures a normal WebSocket endpoint won't emit."""

import base64
import contextlib
import hashlib
import socket
import struct
import threading

from inferport import codec, protocol


def exact(sock, count):
    parts = bytearray()
    while len(parts) < count:
        data = sock.recv(count - len(parts))
        if not data:
            raise EOFError
        parts.extend(data)
    return bytes(parts)


def headers(sock):
    data = bytearray()
    while not data.endswith(b"\r\n\r\n"):
        data.extend(exact(sock, 1))
        if len(data) > 8192:
            raise ValueError("oversized handshake")
    return bytes(data)


def frame(data, opcode=2):
    if len(data) < 126:
        prefix = bytes([128 | opcode, len(data)])
    elif len(data) < 65536:
        prefix = bytes([128 | opcode, 126]) + struct.pack("!H", len(data))
    else:
        prefix = bytes([128 | opcode, 127]) + struct.pack("!Q", len(data))
    return prefix + data


def recv_frame(sock):
    first, second = exact(sock, 2)
    count = second & 127
    if count == 126:
        count = struct.unpack("!H", exact(sock, 2))[0]
    elif count == 127:
        count = struct.unpack("!Q", exact(sock, 8))[0]
    mask = exact(sock, 4) if second & 128 else None
    body = exact(sock, count)
    if mask:
        body = bytes(b ^ mask[i % 4] for i, b in enumerate(body))
    return first & 15, body


def handshake(sock):
    request = headers(sock).decode("ascii")
    key = next(
        line.split(":", 1)[1].strip()
        for line in request.splitlines()
        if line.lower().startswith("sec-websocket-key:")
    )
    accept = base64.b64encode(
        hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
    )
    sock.sendall(
        b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
        b"Sec-WebSocket-Protocol: inferport.v1\r\nSec-WebSocket-Accept: " + accept + b"\r\n\r\n"
    )


@contextlib.contextmanager
def peer(handler, *, upgrade=True, ready=True):
    release = threading.Event()
    errors = []
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(5)

        def run():
            try:
                with listener.accept()[0] as sock:
                    sock.settimeout(5)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
                    if upgrade:
                        handshake(sock)
                    if ready:
                        sock.sendall(frame(codec.encode(protocol.success(0, {}))))
                    handler(sock, release)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=run, name="controlled-peer", daemon=True)
        thread.start()
        try:
            yield f"ws://127.0.0.1:{listener.getsockname()[1]}"
        finally:
            release.set()
            thread.join(6)
            assert not thread.is_alive()
            assert not errors, errors
