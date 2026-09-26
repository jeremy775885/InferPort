import pytest

from inferport import ProtocolError, RemoteError
from inferport.protocol import failure, request, response, success


@pytest.mark.parametrize(
    "patch",
    [{"id": True}, {"id": 0}, {"id": 2**63}, {"id": 1}, {"op": 1}, {"data": []}, {"extra": 1}],
)
def test_bad_requests(patch):
    message = {"id": 2, "op": "infer", "data": {}}
    message.update(patch)
    with pytest.raises(ProtocolError):
        request(message, 1)


@pytest.mark.parametrize(
    "message",
    [
        {"id": True, "ok": True, "data": {}},
        success(2, {}),
        {"id": 1, "ok": 1, "data": {}},
        {"id": 1, "ok": True, "data": [], "extra": 1},
        {"id": 1, "ok": False, "error": {"code": "made_up", "message": "", "fatal": True}},
        {"id": 1, "ok": False, "error": {"code": "invalid_input", "message": "", "fatal": True}},
        failure(1, "busy", "busy"),
    ],
)
def test_bad_responses(message):
    with pytest.raises(ProtocolError):
        response(message, 1)


def test_error_and_empty_acknowledgment():
    with pytest.raises(RemoteError) as error:
        response(failure(4, "invalid_input", "bad input"), 4)
    assert error.value.request_id == 4 and not error.value.fatal
    with pytest.raises(ProtocolError):
        response(success(0, {"extra": 1}), 0, empty=True)
