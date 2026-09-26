"""The model-side interface. Execution environments don't need a base class."""

from abc import ABC, abstractmethod
from typing import Any

Payload = dict[str, Any]


class Backend(ABC):
    @abstractmethod
    def infer(self, inputs: Payload) -> Payload:
        """Compute a result; validate inputs before mutating state."""

    def reset(self, context: Payload) -> None:
        """Clear all session state and replace context. An empty dict must work."""
        return None

    def close(self) -> None:
        """Release resources at service shutdown, not on each disconnect."""
        return None
