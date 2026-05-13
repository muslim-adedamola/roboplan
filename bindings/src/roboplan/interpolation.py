import math

import numpy as np
import pinocchio as pin

from roboplan.core import CartesianPath, CartesianTrajectory, JointTrajectory, Scene


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
        # Deep-copy to match the behaviour of the interpolated path (fresh arrays).
        return [np.asarray(w).copy() for w in waypoints]

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


def interpolateJointTrajectory(
    scene: Scene,
    trajectory: JointTrajectory,
    control_dt: float,
) -> list[np.ndarray]:
    """Interpolate a JointTrajectory using its waypoint times.

    Args:
        scene: RoboPlan scene used to interpolate between configurations.
        trajectory: Sparse joint trajectory with positions and waypoint times.
        control_dt: Desired interpolation sample period, in seconds.

    Returns:
        Dense configuration waypoints sampled according to the trajectory times.
    """
    if control_dt <= 0.0:
        raise ValueError("control_dt must be positive.")

    if len(trajectory.positions) != len(trajectory.times):
        raise ValueError(
            "JointTrajectory positions and times must have the same length."
        )

    if len(trajectory.positions) < 2:
        return [np.asarray(position).copy() for position in trajectory.positions]

    # Validate all timestamps upfront before doing any interpolation work.
    for i in range(len(trajectory.times) - 1):
        if trajectory.times[i + 1] - trajectory.times[i] <= 0.0:
            raise ValueError("JointTrajectory times must be strictly increasing.")

    dense_waypoints = []

    for idx in range(len(trajectory.positions) - 1):
        # JointTrajectory.positions comes from nanobind/Eigen vectors, so convert
        # each position to a NumPy array before using Scene.interpolate().
        start = np.asarray(trajectory.positions[idx])
        end = np.asarray(trajectory.positions[idx + 1])
        segment_time = trajectory.times[idx + 1] - trajectory.times[idx]

        steps_per_segment = computeStepsPerSegment(segment_time, control_dt)

        for step in range(steps_per_segment + 1):
            if idx > 0 and step == 0:
                continue

            alpha = step / steps_per_segment
            dense_waypoints.append(scene.interpolate(start, end, alpha))

    return dense_waypoints


def _validate_cartesian_frames_and_tforms(
    base_frames: list[str],
    tip_frames: list[str],
    tforms: list[list[np.ndarray]],
    expected_tform_length: int | None = None,
) -> None:
    """Validate multi-frame Cartesian path or trajectory data.

    Args:
        base_frames: Reference frame names, one per end-effector.
        tip_frames: Tip frame names, one per end-effector.
        tforms: Transform sequences, one per end-effector.
        expected_tform_length: If provided, each per-frame transform sequence
            must have exactly this many entries.
    """
    if len(base_frames) != len(tip_frames):
        raise ValueError("base_frames and tip_frames must have the same length.")

    if len(tforms) != len(tip_frames):
        raise ValueError("tforms must have one transform sequence per tip frame.")

    if expected_tform_length is not None:
        for frame_tforms in tforms:
            if len(frame_tforms) != expected_tform_length:
                raise ValueError(
                    "Each transform sequence must have the same length as times."
                )


def interpolateCartesianTrajectoryUnchecked(
    trajectory: CartesianTrajectory,
    control_dt: float,
) -> CartesianTrajectory:
    """Interpolate a CartesianTrajectory without re-running validation.

    This is an internal helper. Call interpolateCartesianTrajectory() directly
    unless validation has already been performed by the caller.

    Args:
        trajectory: Pre-validated sparse Cartesian trajectory.
        control_dt: Desired interpolation sample period, in seconds.

    Returns:
        Dense Cartesian trajectory sampled according to the sparse waypoint times.
    """
    # Pre-convert all transforms to SE3 once, avoiding repeated conversions in the loop.
    se3_tforms_by_frame = [
        [pin.SE3(np.asarray(tform)) for tform in frame_tforms]
        for frame_tforms in trajectory.tforms
    ]

    dense_times = []
    dense_tforms_by_frame = [[] for _ in trajectory.tip_frames]

    for idx in range(len(trajectory.times) - 1):
        segment_time = trajectory.times[idx + 1] - trajectory.times[idx]
        if segment_time <= 0.0:
            raise ValueError("CartesianTrajectory times must be strictly increasing.")

        steps_per_segment = computeStepsPerSegment(segment_time, control_dt)

        for step in range(steps_per_segment + 1):
            if idx > 0 and step == 0:
                continue

            alpha = step / steps_per_segment
            dense_times.append(trajectory.times[idx] + alpha * segment_time)

            for frame_idx, frame_se3s in enumerate(se3_tforms_by_frame):
                dense_tforms_by_frame[frame_idx].append(
                    pin.SE3.Interpolate(
                        frame_se3s[idx], frame_se3s[idx + 1], alpha
                    ).homogeneous
                )

    return CartesianTrajectory(
        base_frames=trajectory.base_frames,
        tip_frames=trajectory.tip_frames,
        times=dense_times,
        tforms=dense_tforms_by_frame,
    )


def interpolateCartesianTrajectory(
    trajectory: CartesianTrajectory,
    control_dt: float,
) -> CartesianTrajectory:
    """Interpolate a CartesianTrajectory using its waypoint times.

    Args:
        trajectory: Sparse Cartesian trajectory with transforms and waypoint times.
        control_dt: Desired interpolation sample period, in seconds.

    Returns:
        Dense Cartesian trajectory sampled according to the sparse waypoint times.
    """
    if control_dt <= 0.0:
        raise ValueError("control_dt must be positive.")

    _validate_cartesian_frames_and_tforms(
        trajectory.base_frames,
        trajectory.tip_frames,
        trajectory.tforms,
        expected_tform_length=len(trajectory.times),
    )

    if len(trajectory.times) < 2:
        return CartesianTrajectory(
            base_frames=trajectory.base_frames,
            tip_frames=trajectory.tip_frames,
            times=trajectory.times,
            tforms=[
                [np.asarray(tform).copy() for tform in frame_tforms]
                for frame_tforms in trajectory.tforms
            ],
        )

    return interpolateCartesianTrajectoryUnchecked(trajectory, control_dt)


def interpolateCartesianPath(
    path: CartesianPath,
    waypoint_times: list[float],
    control_dt: float,
) -> CartesianTrajectory:
    """Interpolate a CartesianPath using waypoint times.

    Prefer interpolateCartesianTrajectory() when the input already includes times.

    Args:
        path: Cartesian path containing one transform sequence per end-effector frame.
        waypoint_times: Times corresponding to the path waypoints.
        control_dt: Desired dense interpolation sample period, in seconds.

    Returns:
        Dense Cartesian trajectory sampled according to the waypoint times.
    """

    sparse_trajectory = CartesianTrajectory(
        base_frames=path.base_frames,
        tip_frames=path.tip_frames,
        times=waypoint_times,
        tforms=path.tforms,
    )

    return interpolateCartesianTrajectory(sparse_trajectory, control_dt)


def interpolateSE3Waypoints(
    transforms: list[np.ndarray],
    waypoint_times: list[float],
    control_dt: float,
    base_frame: str,
    tip_frame: str,
) -> list[np.ndarray]:
    """Interpolate single-frame SE(3) waypoints.

    Prefer interpolateCartesianPath() or interpolateCartesianTrajectory() for
    multi-frame Cartesian interpolation.

    Args:
        transforms: Sparse SE(3) waypoints as 4x4 homogeneous matrices.
        waypoint_times: Timestamp for each waypoint, in seconds. Must be strictly increasing.
        control_dt: Desired interpolation sample period, in seconds.
        base_frame: Name of the reference frame.
        tip_frame: Name of the tip (target) frame.

    Returns:
        Dense list of interpolated SE(3) transforms as 4x4 homogeneous matrices.
    """
    path = CartesianPath(
        base_frames=[base_frame],
        tip_frames=[tip_frame],
        tforms=[transforms],
    )

    dense_trajectory = interpolateCartesianPath(path, waypoint_times, control_dt)
    return dense_trajectory.tforms[0]