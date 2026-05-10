import math

import numpy as np
import pinocchio as pin

from roboplan.core import Scene


def computeStepsPerSegment(segment_time: float, control_dt: float) -> int:
    """Compute the number of interpolation intervals in one segment.

    Args:
        segment_time: Duration of one waypoint-to-waypoint segment, in seconds.
        control_dt: Desired interpolation sample period, in seconds.

    Returns:
        Number of interpolation intervals for the segment.
    """
    if segment_time <= 0.0:
        raise ValueError("segment_time must be positive.")
    if control_dt <= 0.0:
        raise ValueError("control_dt must be positive.")

    return max(1, int(math.ceil(segment_time / control_dt)))


def interpolateConfigurationWaypoints(
    scene: Scene,
    waypoints: list[np.ndarray],
    segment_time: float,
    control_dt: float,
) -> list[np.ndarray]:
    """Interpolate configuration waypoints using Scene.interpolate().

    Args:
        scene: RoboPlan scene used to interpolate between configurations.
        waypoints: Sparse configuration waypoints.
        segment_time: Duration of each waypoint-to-waypoint segment, in seconds.
        control_dt: Desired interpolation sample period, in seconds.

    Returns:
        Dense configuration waypoints sampled approximately every control_dt seconds.
    """
    if len(waypoints) < 2:
        return waypoints.copy()

    steps_per_segment = computeStepsPerSegment(segment_time, control_dt)
    dense_waypoints = []

    for idx in range(len(waypoints) - 1):
        start = waypoints[idx]
        end = waypoints[idx + 1]

        for step in range(steps_per_segment + 1):
            if idx > 0 and step == 0:
                continue

            alpha = step / steps_per_segment
            dense_waypoints.append(scene.interpolate(start, end, alpha))

    return dense_waypoints


def interpolateSE3Waypoints(
    transforms: list[np.ndarray],
    segment_time: float,
    control_dt: float,
) -> list[np.ndarray]:
    """Interpolate SE(3) waypoints using Pinocchio SE(3) interpolation.

    Args:
        transforms: Sparse SE(3) waypoints as 4x4 homogeneous matrices.
        segment_time: Duration of each waypoint-to-waypoint segment, in seconds.
        control_dt: Desired interpolation sample period, in seconds.

    Returns:
        Dense SE(3) waypoints as 4x4 homogeneous matrices sampled approximately
        every control_dt seconds.
    """
    if len(transforms) < 2:
        return transforms.copy()

    steps_per_segment = computeStepsPerSegment(segment_time, control_dt)
    dense_transforms = []

    for idx in range(len(transforms) - 1):
        start = pin.SE3(transforms[idx])
        end = pin.SE3(transforms[idx + 1])

        for step in range(steps_per_segment + 1):
            if idx > 0 and step == 0:
                continue

            alpha = step / steps_per_segment
            dense_transforms.append(pin.SE3.Interpolate(start, end, alpha).homogeneous)

    return dense_transforms
