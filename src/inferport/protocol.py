"""Strict InferPort v2 envelopes, independent of transport and model semantics."""

from .errors import ProtocolError, RemoteError

SUBPROTOCOL = "inferport.v2"
MAX_ID = 2**63 - 1
ERROR_CODES = frozenset(
    (
        "busy",
        "invalid_input",
        "unsupported_operation",
        "protocol_error",
        "backend_error",
        "invalid_output",
    )
)


def valid_id(value, *, ready=False):
    return type(value) is int and (0 if ready else 1) <= value <= MAX_ID


def request(message: dict, previous_id: int) -> tuple[int, str, dict]:
    if set(message) != {"id", "op", "data"}:
        raise ProtocolError("Invalid request fields")
    request_id, op, data = message["id"], message["op"], message["data"]
    if not valid_id(request_id) or request_id <= previous_id:
        raise ProtocolError("Request IDs must increase")
    if not isinstance(op, str) or not isinstance(data, dict):
        raise ProtocolError("Invalid request types")
    return request_id, op, data


def success(request_id: int, data: dict) -> dict:
    return {"id": request_id, "ok": True, "data": data}


def failure(request_id: int, code: str, message: str) -> dict:
    return {
        "id": request_id,
        "ok": False,
        "error": {"code": code, "message": message, "fatal": code != "invalid_input"},
    }


def response(message: dict, expected_id: int, *, empty=False) -> dict:
    request_id = message.get("id")
    if not valid_id(request_id, ready=True) or request_id != expected_id:
        raise ProtocolError("Response ID doesn't match request")
    if type(message.get("ok")) is not bool:
        raise ProtocolError("Invalid response status")
    if message["ok"]:
        if set(message) != {"id", "ok", "data"} or not isinstance(message["data"], dict):
            raise ProtocolError("Invalid success response")
        if empty and message["data"]:
            raise ProtocolError("Expected empty acknowledgment")
        return message["data"]
    error = message.get("error")
    if set(message) != {"id", "ok", "error"} or not isinstance(error, dict):
        raise ProtocolError("Invalid error response")
    if set(error) != {"code", "message", "fatal"}:
        raise ProtocolError("Invalid error fields")
    code, text, fatal = error["code"], error["message"], error["fatal"]
    if not isinstance(code, str) or code not in ERROR_CODES or not isinstance(text, str):
        raise ProtocolError("Unknown or malformed remote error")
    if type(fatal) is not bool or fatal != (code != "invalid_input"):
        raise ProtocolError("Invalid error severity")
    if (request_id == 0 and code not in ("busy", "backend_error")) or (
        code == "busy" and request_id != 0
    ):
        raise ProtocolError("Invalid connection error")
    raise RemoteError(code, text, request_id, fatal)
