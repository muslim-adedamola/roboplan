import math

import numpy as np
import pinocchio as pin

from roboplan.core import Scene


def computeStepsPerSegment(policy_dt: float, control_dt: float) -> int:
    """Compute the number of interpolation steps per policy segment."""
    if policy_dt <= 0.0:
        raise ValueError("policy_dt must be positive.")
    if control_dt <= 0.0:
        raise ValueError("control_dt must be positive.")

    return max(1, int(math.ceil(policy_dt / control_dt)))


def interpolateConfigurationWaypoints(
    scene: Scene,
    waypoints: list[np.ndarray] | np.ndarray,
    policy_dt: float,
    control_dt: float,
) -> list[np.ndarray]:
    """
    Interpolate configuration waypoints using the scene interpolation method.
    """
    if len(waypoints) < 2:
        return list(waypoints)

    steps_per_segment = computeStepsPerSegment(policy_dt, control_dt)
    dense_waypoints = []

    for idx in range(len(waypoints) - 1):
        start = np.asarray(waypoints[idx])
        end = np.asarray(waypoints[idx + 1])

        for step in range(steps_per_segment):
            alpha = step / steps_per_segment
            dense_waypoints.append(scene.interpolate(start, end, alpha))

    dense_waypoints.append(np.asarray(waypoints[-1]))
    return dense_waypoints


def interpolateSE3Waypoints(
    transforms: list[np.ndarray],
    policy_dt: float,
    control_dt: float,
) -> list[np.ndarray]:
    """
    Interpolate SE(3) waypoints using Pinocchio SE(3) interpolation.
    """
    if len(transforms) < 2:
        return transforms

    steps_per_segment = computeStepsPerSegment(policy_dt, control_dt)
    dense_transforms = []

    for idx in range(len(transforms) - 1):
        start = pin.SE3(np.asarray(transforms[idx]))
        end = pin.SE3(np.asarray(transforms[idx + 1]))

        for step in range(steps_per_segment):
            alpha = step / steps_per_segment
            dense_transforms.append(pin.SE3.Interpolate(start, end, alpha).homogeneous)

    dense_transforms.append(np.asarray(transforms[-1]))
    return dense_transforms
