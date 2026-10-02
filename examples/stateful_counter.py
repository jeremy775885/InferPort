"""A stateful backend: uv run examples/stateful_counter.py.

From another process:
    with Client("ws://127.0.0.1:8000") as client:
        client.reset({"label": "trial"})
        assert client.infer({}) == {"count": 1, "label": "trial"}
        client.reset()
        assert client.infer({}) == {"count": 1}
"""

from inferport import Backend, InferenceSpec, ObjectSpec, ScalarSpec, serve


class Counter(Backend):
    def describe(self):
        return InferenceSpec(
            ObjectSpec(),
            ObjectSpec(
                {"count": ScalarSpec("integer"), "label": ScalarSpec("string")},
                optional=("label",),
            ),
            ObjectSpec({"label": ScalarSpec("string")}, optional=("label",)),
        )

    def __init__(self):
        self.count = 0
        self.label = None

    def reset(self, context):
        self.count = 0
        self.label = context.get("label")

    def infer(self, inputs):
        self.count += 1
        result = {"count": self.count}
        if self.label is not None:
            result["label"] = self.label
        return result


if __name__ == "__main__":
    serve(Counter())
