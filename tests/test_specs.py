import threading
from dataclasses import replace

import numpy as np
import pytest

from inferport import (
    Backend,
    Channel,
    Client,
    Dimension,
    Error,
    InferenceSpec,
    ObjectSpec,
    ProtocolError,
    RemoteError,
    RequestTimeout,
    ScalarSpec,
    SpecError,
    TensorSpec,
    check_compatibility,
    codec,
    protocol,
)
from inferport.robotics import joint_target_spec

from .peers import frame, peer, recv_frame


def contract(horizon=50, channels=None):
    return joint_target_spec(
        cameras={"head": (2, 3)},
        channels=channels or (Channel("arm.joint1", "joint_position", "rad"),),
        horizon=horizon,
    )


def inputs():
    return {
        "images.head": np.zeros((2, 3, 3), np.uint8),
        "state": np.zeros(1, np.float32),
        "instruction": "move",
    }


class Policy(Backend):
    def __init__(self):
        self.calls = 0
        self.descriptions = 0
        self.threads = []
        self.bad_output = False

    def reset(self, context):
        self.threads.append(threading.get_ident())

    def describe(self):
        self.threads.append(threading.get_ident())
        self.descriptions += 1
        return contract()

    def infer(self, observation):
        self.calls += 1
        return {"action": np.zeros((1 if self.bad_output else 50, 1), np.float32)}


def test_round_trip_and_dynamic_horizons():
    for horizon in (1, 16, 50, Dimension(1, 80)):
        spec = contract(horizon)
        decoded = InferenceSpec.from_dict(codec.decode(codec.encode(spec.to_dict())))
        assert decoded == spec
        check_compatibility(decoded, contract(Dimension(1, 100)))
        spec.inputs.validate(inputs())
    with pytest.raises(SpecError, match="dimension"):
        check_compatibility(contract(50), contract(Dimension(1, 20)))


def test_semantic_mismatch_cannot_pass_by_shape():
    spec = contract()
    for channel in (
        Channel("other", "joint_position", "rad"),
        Channel("arm.joint1", "joint_position", "m"),
    ):
        with pytest.raises(SpecError, match="semantic"):
            check_compatibility(spec, contract(channels=(channel,)))
    with pytest.raises(SpecError, match="semantic"):
        check_compatibility(
            spec,
            replace(
                spec,
                inputs=ObjectSpec(
                    {
                        **spec.inputs.fields,
                        "state": replace(spec.inputs.fields["state"], semantic="joint_positions"),
                    }
                ),
            ),
        )
    with pytest.raises(SpecError, match="profiles"):
        check_compatibility(spec, replace(spec, profile="unknown.v1"))


def test_declared_fields_are_immutable_and_validate_payloads():
    spec = contract()
    with pytest.raises(TypeError):
        spec.inputs.fields["state"] = ScalarSpec("string")
    spec.outputs.validate({"action": np.zeros((50, 1), np.float32)})
    with pytest.raises(SpecError):
        spec.outputs.validate({"action": np.zeros((50, 1), np.float64)})
    with pytest.raises(SpecError):
        spec.inputs.validate({**inputs(), "extra": 1})


@pytest.mark.parametrize(
    "patch",
    [
        {"version": True},
        {"version": 2},
        {"unknown": 0},
        {"inputs": []},
        {"outputs": {"kind": "mystery"}},
    ],
)
def test_malformed_specs(patch):
    with pytest.raises(SpecError):
        InferenceSpec.from_dict({**contract().to_dict(), **patch})


@pytest.mark.parametrize("shape", [(True,), (-1,), ("H",), ({"minimum": 2, "maximum": 1},)])
def test_invalid_dimensions(shape):
    raw = contract().to_dict()
    raw["outputs"]["fields"]["action"]["shape"] = list(shape)
    with pytest.raises(SpecError):
        InferenceSpec.from_dict(raw)


def test_describe_is_snapshot_and_validation_precedes_backend_state(server):
    backend = Policy()
    with server(backend) as (url, *_), Client(url) as client:
        assert client.describe() == client.describe() == contract()
        assert backend.descriptions == 1 and backend.calls == 0
        assert len(set(backend.threads)) == 1
        assert backend.threads[0] != threading.get_ident()
        for bad in (
            {**inputs(), "state": np.ones(2, np.float32)},
            {**inputs(), "state": np.full(1, np.nan, np.float32)},
        ):
            with pytest.raises(RemoteError) as error:
                client.infer(bad)
            assert error.value.code == "invalid_input" and not error.value.fatal
            assert backend.calls == 0
        client.reset({"instruction": "move"})
        assert client.infer(inputs())["action"].shape == (50, 1)
        backend.bad_output = True
        with pytest.raises(RemoteError) as error:
            client.infer(inputs())
        assert error.value.code == "invalid_output" and error.value.fatal


def test_backend_requires_description():
    class MissingDescription(Backend):
        def infer(self, inputs):
            return inputs

    with pytest.raises(TypeError, match="describe"):
        MissingDescription()


@pytest.mark.parametrize("declaration", [None, {}])
def test_invalid_backend_description_fails_before_ready(server, declaration):
    class InvalidDescription(Policy):
        def describe(self):
            return declaration

        def close(self):
            self.closed = True

    backend = InvalidDescription()
    with server(backend) as (url, _, thread, errors):
        with pytest.raises(RemoteError) as error:
            Client(url).connect()
        assert error.value.code == "backend_error" and error.value.fatal
        thread.join(5)
        assert not thread.is_alive()
    assert backend.calls == 0 and backend.closed
    assert len(errors) == 1 and isinstance(errors[0], Error)


def test_invalid_context_and_describe_payload_do_not_touch_backend(server):
    backend = Policy()
    with server(backend) as (url, *_), Client(url) as client:
        before = list(backend.threads)
        for operation, payload in (("reset", {"instruction": 1}), ("describe", {"extra": 1})):
            with pytest.raises(RemoteError) as error:
                client._call(operation, payload, None)
            assert error.value.code == "invalid_input" and not error.value.fatal
            assert backend.threads == before
        assert client.describe() == contract()
        assert backend.calls == 0


def test_description_is_copied_before_backend_changes(server):
    backend = Policy()
    with server(backend) as (url, *_), Client(url) as client:
        backend.describe = lambda: contract(16)
        assert client.describe() == contract(50)
        client.reset({})
        assert client.describe() == contract(50)


@pytest.mark.parametrize("declaration", [None, {"version": 99}])
def test_malformed_remote_description_closes_client(declaration):
    def handler(sock, release):
        recv_frame(sock)
        sock.sendall(frame(codec.encode(protocol.success(1, {"spec": declaration}))))
        release.wait(2)

    with peer(handler, describe=False) as url:
        client = Client(url)
        with pytest.raises(ProtocolError, match="specification"):
            client.connect()
        with pytest.raises(RuntimeError, match="usable"):
            client.infer({})


@pytest.mark.parametrize(
    "output",
    [
        {"action": np.zeros((16, 1), np.float32)},
        {"action": np.zeros((50, 1), np.float64)},
        {"action": np.full((50, 1), np.nan, np.float32)},
        {},
        {"action": np.zeros((50, 1), np.float32), "unknown": 0},
    ],
)
def test_client_rejects_invalid_remote_output_without_application_validation(output):
    def handler(sock, release):
        request = codec.decode(recv_frame(sock)[1])
        assert request["id"] == 2 and request["op"] == "infer"
        sock.sendall(frame(codec.encode(protocol.success(2, output))))
        release.wait(2)

    # This wire peer bypasses serve's own output check: the client must reject it.
    with peer(handler, spec=contract()) as url, Client(url) as client:
        with pytest.raises(ProtocolError, match="output contract"):
            client.infer(inputs())  # No explicit describe() or validate() is needed.
        with pytest.raises(RuntimeError, match="usable"):
            client.infer(inputs())
        with pytest.raises(RuntimeError, match="usable"):
            client.describe()


def test_describe_cache_uses_no_rpc_and_survives_reset():
    def handler(sock, release):
        # The peer helper handled the single describe call during connect.
        for request_id, op in ((2, "infer"), (3, "reset"), (4, "infer")):
            request = codec.decode(recv_frame(sock)[1])
            assert request["id"] == request_id and request["op"] == op
            output = {"action": np.zeros((50, 1), np.float32)} if op == "infer" else {}
            sock.sendall(frame(codec.encode(protocol.success(request_id, output))))
        release.wait(2)

    with peer(handler, spec=contract()) as url, Client(url) as client:
        spec = client.describe()
        assert client.describe() is spec
        client.infer(inputs())
        client.reset({"instruction": "move"})
        assert client.describe() is spec
        client.infer(inputs())


def test_contract_fetch_uses_open_deadline_and_closes_client():
    def handler(sock, release):
        request = codec.decode(recv_frame(sock)[1])
        assert request["op"] == "describe"
        release.wait(2)

    with peer(handler, describe=False) as url:
        client = Client(url, open_timeout=0.1)
        with pytest.raises(RequestTimeout) as error:
            client.connect()
        assert error.value.stage == "describe"
        with pytest.raises(RuntimeError):
            client.connect()


def test_application_declares_extensions_without_changing_robot_profile(server):
    base = contract()
    assert set(base.outputs.fields) == {"action"}
    assert set(base.context.fields) == {"instruction"}
    extra = ObjectSpec({"custom_score": ScalarSpec("number", minimum=0)})
    spec = replace(
        base,
        outputs=ObjectSpec({**base.outputs.fields, "report": extra}, optional=("report",)),
    )

    class ReportingPolicy(Policy):
        def describe(self):
            return spec

        def infer(self, observation):
            return {**super().infer(observation), "report": {"custom_score": 0.5}}

    with server(ReportingPolicy()) as (url, *_), Client(url) as client:
        assert client.describe() == spec
        assert client.infer(inputs())["report"] == {"custom_score": 0.5}
    with pytest.raises(SpecError):
        spec.outputs.validate(
            {"action": np.zeros((50, 1), np.float32), "report": {"custom_score": -1}}
        )


def test_v1_handshake_rejected(server):
    import asyncio

    from websockets.asyncio.client import connect
    from websockets.exceptions import InvalidStatus

    async def old_client(url):
        with pytest.raises(InvalidStatus) as error:
            async with connect(url, subprotocols=["inferport.v1"]):
                pass
        assert error.value.response.status_code == 400

    with server() as (url, *_):
        asyncio.run(old_client(url))


def test_generic_value_backend_contract():
    spec = InferenceSpec(
        ObjectSpec({"state": TensorSpec("float32", (Dimension(1, 8), 3))}),
        ObjectSpec({"value": TensorSpec("float32", (Dimension(1, 8),))}),
    )
    assert InferenceSpec.from_dict(spec.to_dict()) == spec
    spec.outputs.validate({"value": np.ones(2, np.float32)})


class ScalarBackend(Backend):
    def __init__(self, scalar, value):
        payload = ObjectSpec({"value": scalar})
        self.spec = InferenceSpec(payload, payload)
        self.output = {"value": value}
        self.calls = 0
        self.received = None

    def describe(self):
        return self.spec

    def infer(self, inputs):
        self.calls += 1
        self.received = inputs["value"]
        return self.output


@pytest.mark.parametrize(
    "kind,value,expected",
    [
        ("boolean", np.bool_(True), True),
        ("integer", np.int8(-128), -128),
        ("integer", np.int16(-32768), -32768),
        ("integer", np.int32(-(2**31)), -(2**31)),
        ("integer", np.int64(-(2**63)), -(2**63)),
        ("integer", np.uint8(255), 255),
        ("integer", np.uint16(65535), 65535),
        ("integer", np.uint32(2**32 - 1), 2**32 - 1),
        ("integer", np.uint64(2**64 - 1), 2**64 - 1),
        ("number", np.float16(0.5), 0.5),
        ("number", np.float32(0.5), 0.5),
        ("number", np.float64(0.5), 0.5),
        ("number", np.int64(7), 7),
    ],
)
def test_numpy_scalars_round_trip_without_mutating_backend_output(server, kind, value, expected):
    bounds = {} if kind == "boolean" else {"minimum": expected, "maximum": expected}
    backend = ScalarBackend(ScalarSpec(kind, **bounds), value)
    with server(backend) as (url, *_), Client(url) as client:
        for _ in range(2):
            result = client.infer({"value": value})
            assert result == {"value": expected}
            assert type(result["value"]) is type(expected)
            assert type(backend.received) is type(expected)
            assert backend.output["value"] is value
    assert backend.calls == 2


@pytest.mark.parametrize(
    "scalar,value",
    [
        (ScalarSpec("integer"), np.bool_(True)),
        (ScalarSpec("number"), np.bool_(False)),
        (ScalarSpec("boolean"), np.int8(1)),
        (ScalarSpec("integer"), np.float32(1.0)),
        (ScalarSpec("number"), np.float16(np.nan)),
        (ScalarSpec("number"), np.float32(np.inf)),
        (ScalarSpec("number"), np.float64(-np.inf)),
        (ScalarSpec("integer", minimum=0), np.int64(-1)),
        (ScalarSpec("integer", maximum=2**64 - 2), np.uint64(2**64 - 1)),
        (ScalarSpec("number", minimum=0), np.float32(-0.5)),
        (ScalarSpec("number", maximum=1), np.float64(1.5)),
        (ScalarSpec("number"), np.array(0.5, np.float32)),
    ],
)
def test_invalid_scalar_constraints_preserve_rpc_error_semantics(server, scalar, value):
    backend = ScalarBackend(scalar, value)
    with server(backend) as (url, *_), Client(url) as client:
        with pytest.raises(RemoteError) as error:
            client.infer({"value": value})
        assert error.value.code == "invalid_input" and not error.value.fatal
        assert backend.calls == 0

        valid = True if scalar.kind == "boolean" else 0
        with pytest.raises(RemoteError) as error:
            client.infer({"value": valid})
        assert backend.calls == 1  # Invalid input left the connection usable.
        assert error.value.code == "invalid_output" and error.value.fatal
        with pytest.raises(RuntimeError, match="usable"):
            client.infer({"value": valid})


@pytest.mark.parametrize(
    "scalar,value",
    [
        (ScalarSpec("number"), np.complex64(1)),
        (ScalarSpec("integer"), np.datetime64("2026-10-02")),
        (ScalarSpec("string"), np.str_("unsupported by codec")),
    ],
)
def test_scalar_spec_rejects_unsupported_numpy_kinds(scalar, value):
    with pytest.raises(SpecError):
        scalar.validate(value)
