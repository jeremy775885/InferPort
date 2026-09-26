"""A stateful backend: uv run examples/stateful_counter.py.

From another process:
    with Client("ws://127.0.0.1:8000") as client:
        client.reset({"label": "trial"})
        assert client.infer({}) == {"count": 1, "label": "trial"}
        client.reset()
        assert client.infer({}) == {"count": 1, "label": None}
"""

from inferport import Backend, InvalidInput, serve


class Counter(Backend):
    def __init__(self):
        self.count = 0
        self.label = None

    def reset(self, context):
        label = context.get("label")
        if label is not None and not isinstance(label, str):
            raise InvalidInput("label must be a string")
        self.count = 0
        self.label = label

    def infer(self, inputs):
        self.count += 1
        return {"count": self.count, "label": self.label}


if __name__ == "__main__":
    serve(Counter())
