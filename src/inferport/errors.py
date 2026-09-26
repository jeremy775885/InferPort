"""Public exceptions; callers needn't depend on the network implementation."""


class Error(Exception):
    """Base InferPort error."""


class TransportError(Error):
    """Connection establishment or exchange failed."""


class RequestTimeout(TransportError):
    """The network deadline expired (local encoding/decoding is excluded)."""

    def __init__(self, stage: str):
        self.stage = stage
        super().__init__(f"Timed out during {stage}")


class ProtocolError(Error):
    """The peer violated InferPort v1."""


class RemoteError(Error):
    """An explicitly reported server error."""

    def __init__(self, code: str, message: str, request_id: int | None, fatal: bool):
        self.code = code
        self.message = message
        self.request_id = request_id
        self.fatal = fatal
        super().__init__(f"{code}: {message}")


class InvalidInput(Error):
    """Backend validation failed before changing any state."""
