"""The model-side interface. Execution environments don't need a base class."""

from abc import ABC, abstractmethod
from typing import Any

from .specs import InferenceSpec

Payload = dict[str, Any]


class Backend(ABC):
    @abstractmethod
    def infer(self, inputs: Payload) -> Payload:
        """Compute from contract-validated inputs; check business rules before mutation.

        serve validates the declared input/context contracts. Direct Python callers
        are responsible for meeting those contracts themselves.
        """

    @abstractmethod
    def describe(self) -> InferenceSpec:
        """Declare the required input, output and reset-context contract.

        Called once per connection on the backend worker, after reset({}).
        Must not mutate model state, processors or RNG.
        """

    def reset(self, context: Payload) -> None:
        """Clear all session state and replace context. An empty dict must work."""
        return None

    def close(self) -> None:
        """Release resources at service shutdown, not on each disconnect."""
        return None
