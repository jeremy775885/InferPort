"""Joint-target policy profile v1; no robot SDK or model framework dependencies.

Arrays carry RGB HWC images, an ordered joint/gripper state and sequential absolute
joint targets. Targets have executor-defined duration (not a fixed-rate trajectory).
Gripper positions use 0=closed, 1=open. No normalization or clipping is performed.
"""

from .specs import Channel, Dimension, InferenceSpec, ObjectSpec, ScalarSpec, SpecError, TensorSpec

JOINT_TARGET_PROFILE = "robotics.joint-targets.v1"


def joint_target_spec(*, cameras, channels, horizon, state_semantic="joint_targets"):
    """Build a shared contract; camera keys are physical roles, not model features.

    ``channels`` contains names/units/meanings in actual vector order. Supported
    meanings are joint_position (rad or m) and gripper_position (fraction).
    ``state_semantic`` distinguishes drive targets from measured positions.
    Application-specific diagnostics and reset options can be declared by replacing
    outputs/context on the returned InferenceSpec with extended ObjectSpec values.
    """
    channels = tuple(channels)
    if not channels or not cameras or state_semantic not in ("joint_targets", "joint_positions"):
        raise SpecError("Cameras, channels and a supported state semantic are required")
    for c in channels:
        if not isinstance(c, Channel) or (c.semantic, c.unit) not in (
            ("joint_position", "rad"),
            ("joint_position", "m"),
            ("gripper_position", "fraction"),
        ):
            raise SpecError("Unsupported joint/gripper channel semantics or units")
    low = horizon.minimum if isinstance(horizon, Dimension) else horizon
    if type(low) is not int or low < 1:
        raise SpecError("Action horizons must be positive")
    images = {}
    for role, size in cameras.items():
        if (
            not isinstance(role, str)
            or not role
            or len(size) != 2
            or any(type(n) is not int or n <= 0 for n in size)
        ):
            raise SpecError("Each camera role requires a positive height/width")
        images[f"images.{role}"] = TensorSpec(
            "uint8", (*size, 3), "rgb", ("height", "width", "channel")
        )
    inputs = ObjectSpec(
        {
            **images,
            "state": TensorSpec(
                "float32", (len(channels),), state_semantic, ("channel",), channels
            ),
            "instruction": ScalarSpec("string", nonempty=True),
        }
    )
    outputs = ObjectSpec(
        {
            "action": TensorSpec(
                "float32",
                (horizon, len(channels)),
                "absolute_joint_targets",
                ("step", "channel"),
                channels,
            ),
        },
    )
    context = ObjectSpec({"instruction": ScalarSpec("string", nonempty=True)})
    return InferenceSpec(inputs, outputs, context, JOINT_TARGET_PROFILE)
