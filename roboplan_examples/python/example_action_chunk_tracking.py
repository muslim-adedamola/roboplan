"""
Track mock learned policy action chunks with RoboPlan OInK.
This example demonstrates how a learned policy action chunk can be treated as a
short-horizon command sequence, interpolated at a smaller control timestep, and
tracked through OInK while respecting robot position and velocity limits.

Supported chunk types:
  1. Cartesian/end-effector deltas: [dx, dy, dz, droll, dpitch, dyaw]
  2. Joint-space deltas:          [dq_1, dq_2, ..., dq_n]

The main idea is:
  sparse policy action chunk
      -> sparse target poses/configurations
      -> dense interpolated targets at control_dt
      -> OInK tracking with PositionLimit + VelocityLimit
      -> integrated constrained trajectory
      -> visualization
"""

import sys
import time
from typing import Literal

import numpy as np
import pinocchio as pin
import tyro
import xacro
from pinocchio.visualize import ViserVisualizer

from common import MODELS
from roboplan.core import CartesianConfiguration, Scene
from roboplan.example_models import get_package_share_dir
from roboplan.optimal_ik import (
    ConfigurationTask,
    ConfigurationTaskOptions,
    FrameTask,
    FrameTaskOptions,
    Oink,
    PositionLimit,
    VelocityLimit,
)

from roboplan.interpolation import (
    interpolateConfigurationWaypoints,
    interpolateSE3Waypoints,
)
from roboplan.visualization import visualizePositionTrace

ActionSpace = Literal["cartesian", "joint"]


# Convert a 6D Cartesian delta into an SE(3) transform.
# delta = [dx, dy, dz, droll, dpitch, dyaw]. Rotation is represented as small XYZ Euler increments.
def se3_from_delta(delta: np.ndarray) -> pin.SE3:

    translation = np.asarray(delta[:3], dtype=float)
    roll, pitch, yaw = np.asarray(delta[3:6], dtype=float)

    rotation = (
        pin.rpy.rpyToMatrix(np.array([roll, pitch, yaw], dtype=float))
        if np.linalg.norm(delta[3:6]) > 0.0
        else np.eye(3)
    )
    return pin.SE3(rotation, translation)


# Create a mock Cartesian action chunk of shape [horizon, 6]
def make_mock_cartesian_action_chunk(
    horizon: int,
    translation_scale: float = 0.015,
    rotation_scale: float = 0.03,
    action_scale: float = 1.0,
) -> np.ndarray:

    chunk = np.zeros((horizon, 6), dtype=float)

    # Move forward/up slightly, with a small yaw/roll change.
    for i in range(horizon):
        phase = (i + 1) / float(horizon)
        chunk[i, 0] = translation_scale
        chunk[i, 1] = 0.003 * np.sin(np.pi * phase)
        chunk[i, 2] = 0.006 * np.sin(0.5 * np.pi * phase)
        chunk[i, 3] = rotation_scale * 0.2
        chunk[i, 4] = 0.0
        chunk[i, 5] = rotation_scale * 0.3

    return action_scale * chunk


# Create a mock joint-space action chunk of shape [horizon, num_joints].
def make_mock_joint_action_chunk(
    horizon: int,
    num_joints: int,
    joint_delta_scale: float = 0.015,
    action_scale: float = 1.0,
) -> np.ndarray:

    chunk = np.zeros((horizon, num_joints), dtype=float)

    for i in range(horizon):
        phase = (i + 1) / float(horizon)
        direction = np.sin(np.linspace(0.0, np.pi, num_joints) + phase)
        chunk[i] = joint_delta_scale * direction

    return action_scale * chunk


# Accumulate Cartesian deltas into sparse end-effector target poses.
def cartesian_chunk_to_sparse_targets(
    scene: Scene,
    q_start: np.ndarray,
    ee_frame_name: str,
    action_chunk: np.ndarray,
) -> list[np.ndarray]:

    current_target = scene.forwardKinematics(q_start, ee_frame_name)
    targets = [current_target.copy()]

    target_se3 = pin.SE3(current_target)
    for delta in action_chunk:
        target_se3 = target_se3 * se3_from_delta(delta)
        targets.append(target_se3.homogeneous.copy())

    return targets


def joint_chunk_to_sparse_targets(
    scene: Scene,
    q_full_start: np.ndarray,
    v_indices: np.ndarray,
    num_velocity_variables: int,
    action_chunk: np.ndarray,
) -> list[np.ndarray]:
    """Accumulate joint-space deltas into sparse full-configuration targets.

    The joint-space action chunk lives in the robot tangent/velocity space, not
    directly in configuration space. Therefore, each delta is lifted into the full
    velocity vector and applied using scene.integrate().
    """
    targets = [np.asarray(q_full_start, dtype=float).copy()]
    q_target = np.asarray(q_full_start, dtype=float).copy()

    for delta_group in action_chunk:
        delta_full = np.zeros(num_velocity_variables)
        delta_full[v_indices] = np.asarray(delta_group, dtype=float)
        q_target = scene.integrate(q_target, delta_full)
        targets.append(q_target.copy())

    return targets


# Compute end-effector xyz positions for a sequence of full configurations
def compute_end_effector_positions(
    scene: Scene,
    configurations: list[np.ndarray],
    ee_frame_name: str,
) -> np.ndarray:

    positions = []

    for q in configurations:
        scene.setJointPositions(q)
        ee_tform = scene.forwardKinematics(q, ee_frame_name)
        positions.append(ee_tform[:3, 3].copy())

    return np.asarray(positions)


# Extract xyz positions from Cartesian SE(3) target transforms
def cartesian_target_positions(target_transforms: list[np.ndarray]) -> np.ndarray:
    return np.asarray([np.asarray(tform)[:3, 3].copy() for tform in target_transforms])


# Return the starting configuration for the selected model.
def get_starting_configuration(
    scene: Scene,
    model_data,
) -> np.ndarray:

    q_full = scene.getCurrentJointPositions()
    q_start_full = np.array(model_data.starting_joint_config)

    if len(q_start_full) == len(q_full):
        return q_start_full.copy()

    print(
        f"Warning: starting_joint_config size ({len(q_start_full)}) does not match "
        f"model configuration size ({len(q_full)}). Using scene default instead."
    )
    return q_full


# Create an OInK solver from a fixed starting configuration
def create_oink_solver(
    scene: Scene,
    joint_group: str,
    q_start: np.ndarray,
) -> Oink:

    scene.setJointPositions(q_start)
    return Oink(scene, joint_group)


def main(
    model: str = "kinova",
    action_space: ActionSpace = "cartesian",
    chunk_horizon: int = 6,
    action_scale: float = 1.0,
    policy_dt: float = 0.2,
    control_freq: float = 100.0,
    task_gain: float = 1.0,
    lm_damping: float = 0.01,
    regularization: float = 1e-6,
    sleep: bool = False,
    animation_dt: float = 0.03,
    host: str = "localhost",
    port: str = "8000",
):
    """Track a mock policy action chunk with OInK and velocity limits.

    Args:
        model: Robot model name from roboplan_examples/python/common.py.
        action_space: Whether to use Cartesian or joint-space mock action chunks.
        chunk_horizon: Number of sparse actions produced by the mock policy.
        action_scale: Scale applied to the mock action chunk to make the motion shorter or longer.
        policy_dt: Time between sparse policy actions, in seconds.
        control_freq: Dense OInK tracking frequency, in Hz.
        task_gain: OInK task gain.
        lm_damping: Levenberg-Marquardt damping for the frame task.
        regularization: Tikhonov regularization passed to OInK.
        sleep: If true, sleep between dense tracking steps while initially generating the trajectory.
        animation_dt: Delay between displayed configurations when using the GUI animation button.
        host: Viser host.
        port: Viser port.
    """
    if model not in MODELS:
        print(f"Invalid model requested: {model}")
        print(f"Available models: {list(MODELS.keys())}")
        sys.exit(1)

    model_data = MODELS[model]
    package_paths = [get_package_share_dir()]

    urdf_xml = xacro.process_file(model_data.urdf_path).toxml()
    srdf_xml = xacro.process_file(model_data.srdf_path).toxml()

    scene = Scene(
        "policy_action_chunk_oink_scene",
        urdf=urdf_xml,
        srdf=srdf_xml,
        package_paths=package_paths,
        yaml_config_path=model_data.yaml_config_path,
    )

    joint_group = model_data.default_joint_group
    joint_names = scene.getJointGroupInfo(joint_group).joint_names

    print(f"\n=== Model: {model} ===")
    print(f"Joint group: {joint_group}")
    print(f"Joint names: {joint_names}")
    print(f"Action space: {action_space}")
    print(f"Action scale: {action_scale}")

    # Create a redundant Pinocchio model for visualization and for obtaining
    # the full velocity-space size.
    model_pin = pin.buildModelFromXML(urdf_xml)
    q_start = get_starting_configuration(scene, model_data)

    # Fix scene at the selected model start configuration
    scene.setJointPositions(q_start)

    # Build geometry models for visualization.
    collision_model = pin.buildGeomFromUrdfString(
        model_pin,
        urdf_xml,
        pin.GeometryType.COLLISION,
        package_dirs=package_paths,
    )
    visual_model = pin.buildGeomFromUrdfString(
        model_pin,
        urdf_xml,
        pin.GeometryType.VISUAL,
        package_dirs=package_paths,
    )

    viz = ViserVisualizer(model_pin, collision_model, visual_model)
    viz.initViewer(open=True, loadModel=True, host=host, port=port)
    viz.display(q_start)

    # Set up OInK after fixing the scene at the selected start configuration.
    oink = create_oink_solver(scene, joint_group, q_start)
    num_variables = len(oink.v_indices)
    dt = 1.0 / control_freq

    v_max = np.hstack(
        [scene.getJointInfo(name).limits.max_velocity for name in joint_names]
    )

    constraints = [
        PositionLimit(oink, gain=1.0),
        VelocityLimit(oink, dt, v_max),
    ]

    print(f"Velocity variables: {num_variables}")
    print(f"Dense control dt: {dt:.4f} s")
    print(f"Sparse policy dt: {policy_dt:.4f} s")
    print(
        f"Interpolation substeps per policy action: {max(1, int(round(policy_dt / dt)))}"
    )

    # Configuration regularization task.
    joint_weights = np.full(num_variables, 0.05)
    config_options = ConfigurationTaskOptions(task_gain=0.1, lm_damping=0.0)
    config_task = ConfigurationTask(
        oink,
        q_start[oink.q_indices],
        joint_weights,
        config_options,
    )

    # Frame task for Cartesian tracking.
    ee_frame_name = model_data.ee_names[0]
    task_options = FrameTaskOptions(
        position_cost=1.0,
        orientation_cost=0.1,
        task_gain=task_gain,
        lm_damping=lm_damping,
    )

    goal = CartesianConfiguration()
    goal.base_frame = model_data.base_link
    goal.tip_frame = ee_frame_name
    frame_task = FrameTask(oink, scene, goal, task_options)

    if action_space == "cartesian":
        action_chunk = make_mock_cartesian_action_chunk(
            chunk_horizon,
            action_scale=action_scale,
        )
        sparse_targets = cartesian_chunk_to_sparse_targets(
            scene,
            q_start,
            ee_frame_name,
            action_chunk,
        )
        dense_targets = interpolateSE3Waypoints(sparse_targets, policy_dt, dt)
        sparse_target_positions = cartesian_target_positions(sparse_targets)
        dense_target_positions = cartesian_target_positions(dense_targets)
        tasks = [frame_task, config_task]

    else:
        # Joint-space chunks are tangent-space increments, so generate them in
        # velocity space and apply them with scene.integrate().
        action_chunk = make_mock_joint_action_chunk(
            chunk_horizon,
            num_joints=len(oink.v_indices),
            action_scale=action_scale,
        )
        sparse_full_targets = joint_chunk_to_sparse_targets(
            scene,
            q_start,
            oink.v_indices,
            model_pin.nv,
            action_chunk,
        )
        dense_full_targets = interpolateConfigurationWaypoints(
            scene,
            sparse_full_targets,
            policy_dt,
            dt,
        )

        # OInK's ConfigurationTask target lives in the active configuration
        # coordinates, while the visualization uses the full robot configurations.
        dense_targets = [
            q_full_target[oink.q_indices] for q_full_target in dense_full_targets
        ]
        sparse_targets = sparse_full_targets

        sparse_target_positions = compute_end_effector_positions(
            scene, sparse_full_targets, ee_frame_name
        )
        dense_target_positions = compute_end_effector_positions(
            scene, dense_full_targets, ee_frame_name
        )

    print(f"Sparse targets: {len(sparse_targets)}")
    print(f"Dense targets:  {len(dense_targets)}")

    # Trajectory rollout starts from the fixed start configuration.
    scene.setJointPositions(q_start)
    q_current = q_start.copy()
    delta_q = np.zeros(num_variables, dtype=float)
    delta_q_full = np.zeros(model_pin.nv, dtype=float)
    trajectory = [q_current.copy()]

    for idx, target in enumerate(dense_targets):
        loop_start = time.time()

        if action_space == "cartesian":
            frame_task.setTargetFrameTransform(target)
            active_tasks = tasks
        else:
            # ConfigurationTask currently has no Python-side target setter, so
            # recreate the task for each dense target. This is cheap enough for
            # an example and keeps joint-space tracking compatible with the
            # current RoboPlan Python bindings.
            target_config_task = ConfigurationTask(
                oink,
                np.asarray(target, dtype=float),
                joint_weights,
                config_options,
            )
            active_tasks = [target_config_task]

        try:
            oink.solveIk(scene, active_tasks, constraints, delta_q, regularization)
        except RuntimeError as exc:
            print(f"Warning: OInK failed at dense step {idx}: {exc}")
            delta_q[:] = 0.0

        delta_q_full[:] = 0.0
        delta_q_full[oink.v_indices] = delta_q

        q_current = scene.integrate(q_current, delta_q_full)
        scene.setJointPositions(q_current)

        # Refresh FK for the frame task when using Cartesian tracking.
        if action_space == "cartesian":
            scene.forwardKinematics(q_current, ee_frame_name)

        trajectory.append(q_current.copy())

        if sleep:
            viz.display(q_current)
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, dt - elapsed))

    print("Finished tracking action chunk.")
    print(f"Generated constrained trajectory with {len(trajectory)} configurations.")

    executed_ee_positions = compute_end_effector_positions(
        scene,
        trajectory,
        ee_frame_name,
    )

    # Visualize 1. sparse policy waypoints,
    # 2. dense interpolated references,
    # 3. final OInK-constrained executed trajectory.
    visualizePositionTrace(
        viz,
        sparse_target_positions,
        trace_name="/action_chunk/sparse_policy_waypoints/trace",
        waypoint_root="/action_chunk/sparse_policy_waypoints/markers",
        trace_color=(255, 160, 0),
        waypoint_color=(255, 160, 0),
        line_width=1.0,
        waypoint_radius=0.009,
        draw_trace=False,
        draw_waypoints=True,
    )

    visualizePositionTrace(
        viz,
        dense_target_positions,
        trace_name="/action_chunk/dense_interpolated_targets/trace",
        waypoint_root="/action_chunk/dense_interpolated_targets/markers",
        trace_color=(90, 90, 90),
        waypoint_color=(90, 90, 90),
        line_width=3.0,
        waypoint_radius=0.004,
        draw_trace=True,
        draw_waypoints=True,
        waypoint_stride=10,
    )

    visualizePositionTrace(
        viz,
        executed_ee_positions,
        trace_name="/action_chunk/executed_oink_trace/trace",
        waypoint_root="/action_chunk/executed_oink_trace/markers",
        trace_color=(0, 80, 255),
        waypoint_color=(0, 80, 255),
        line_width=10.0,
        waypoint_radius=0.01,
        draw_trace=True,
        draw_waypoints=False,
    )

    current_ee_marker = None
    try:
        current_ee_marker = viz.viewer.scene.add_icosphere(
            "/action_chunk/current_ee",
            radius=0.025,
            position=executed_ee_positions[0],
            color=(255, 0, 0),
        )
    except Exception as exc:
        print(f"Warning: could not draw current end-effector marker: {exc}")

    print("Visualization added:")
    print("  orange: sparse policy waypoints shown as small markers only")
    print("  gray:   dense interpolated targets")
    print("  blue:   executed OInK-constrained trajectory")
    print("  red:    current end-effector position")
    print(
        "Use the Viser GUI buttons to animate, step through, or reset the trajectory."
    )

    state = {
        "step_idx": 0,
        "animating": False,
    }

    animate_button = viz.viewer.gui.add_button("Animate action chunk")
    step_button = viz.viewer.gui.add_button("Step once")
    reset_button = viz.viewer.gui.add_button("Reset")

    def display_step(step_idx: int):
        """Display one tracked configuration by index."""
        step_idx = max(0, min(step_idx, len(trajectory) - 1))

        viz.display(trajectory[step_idx])

        if current_ee_marker is not None:
            current_ee_marker.position = executed_ee_positions[step_idx]

        state["step_idx"] = step_idx

    @animate_button.on_click
    def animate_action_chunk(_):
        if state["animating"]:
            return

        state["animating"] = True
        animate_button.disabled = True
        step_button.disabled = True
        reset_button.disabled = True

        if state["step_idx"] >= len(trajectory) - 1:
            state["step_idx"] = 0

        try:
            for idx in range(state["step_idx"], len(trajectory)):
                display_step(idx)
                time.sleep(animation_dt)
        finally:
            state["animating"] = False
            animate_button.disabled = False
            step_button.disabled = False
            reset_button.disabled = False

    @step_button.on_click
    def step_once(_):
        if state["animating"]:
            return

        next_idx = min(state["step_idx"] + 1, len(trajectory) - 1)
        display_step(next_idx)

    @reset_button.on_click
    def reset(_):
        if state["animating"]:
            return

        display_step(0)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    tyro.cli(main)
