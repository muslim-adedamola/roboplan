#!/usr/bin/env python3

"""
Demo: tracking a target relative to a *movable* base frame.

Most IK examples define their end-effector targets in the world frame. This demo
shows what the FrameTask `base_frame` feature buys us: tracking a pose defined
relative to another frame that is itself moving.

Setup (--model dual, the dual FR3 arms):
  - The LEFT arm tracks an interactive marker in the world frame (base_frame
    "universe", so the relative Jacobian degenerates to the absolute one).
  - The RIGHT arm's task uses the LEFT hand as its base_frame, and holds a fixed
    pose relative to it. As you drag the left hand around, the left hand (the
    right arm's base frame) moves, and the right arm tracks it to keep the
    relative offset constant.

Coordinate axes are drawn at the left hand so the moving base frame is visible.
"""

import sys
import threading
import time
import tyro
import xacro

import numpy as np
import pinocchio as pin
from pinocchio.visualize import ViserVisualizer

from common import get_model_data
from roboplan.filters import SE3LowPassFilter
from roboplan.core import Scene, CartesianConfiguration
from roboplan.example_models import get_package_share_dir
from roboplan.optimal_ik import (
    ConfigurationTask,
    ConfigurationTaskOptions,
    FrameTask,
    FrameTaskOptions,
    Oink,
    PositionLimit,
    SelfCollisionBarrier,
    VelocityLimit,
)


def _tform_to_viser(tform):
    """Split a 4x4 homogeneous transform into Viser (wxyz, position)."""
    wxyz = pin.Quaternion(tform[:3, :3]).coeffs()[[3, 0, 1, 2]]
    return wxyz, tform[:3, 3]


def main(
    model: str = "dual",
    task_gain: float = 1.0,
    lm_damping: float = 0.01,
    regularization: float = 1e-6,
    control_freq: float = 100.0,
    reference_filter_tau: float = 0.1,
    self_collision_num_pairs: int = 0,
    self_collision_d_min: float = 0.02,
    self_collision_gain: float = 1.0,
    host: str = "localhost",
    port: str = "8000",
):
    """
    Run the movable base-frame IK demo.

    Parameters:
        model: Model to use. Must expose at least two end-effector frames; the
            default "dual" model (dual FR3 arms) is what this demo is designed around.
        task_gain: Task gain (alpha) for the IK solver (0-1).
        lm_damping: Levenberg-Marquardt damping for regularization.
        regularization: Tikhonov regularization weight for the QP Hessian.
        control_freq: Control loop frequency in Hz.
        reference_filter_tau: Time constant (s) for smoothing the marker target.
            Set to 0 to disable.
        self_collision_num_pairs: Number of collision pairs to use for the solver's
            self-collision barrier. If zero, no collision barrier will be used.
            Useful here since the two arms can be dragged into each other.
        self_collision_d_min: Minimum distance (meters) the IK solver will try to keep
            between every pair of self-collision bodies declared by the SRDF.
        self_collision_gain: Barrier gain (gamma) for the self-collision barrier. Higher
            values produce stronger pushback as bodies approach `self_collision_d_min`.
        host: The host for the ViserVisualizer.
        port: The port for the ViserVisualizer.
    """
    model_data = get_model_data().get(model)
    if model_data is None:
        print(f"Invalid model requested: {model}")
        sys.exit(1)
    if len(model_data.ee_names) < 2:
        print(
            f"Model '{model}' exposes {len(model_data.ee_names)} end-effector(s); "
            "this demo needs at least two (try --model dual)."
        )
        sys.exit(1)

    # The first end-effector is driven by the world marker; the second tracks a
    # pose relative to the first (which therefore acts as its movable base frame).
    tracked_frame = model_data.ee_names[0]
    follower_frame = model_data.ee_names[1]

    package_paths = [get_package_share_dir()]

    # Pre-process with xacro. This is not necessary for raw URDFs.
    urdf_xml = xacro.process_file(model_data.urdf_path).toxml()
    srdf_xml = xacro.process_file(model_data.srdf_path).toxml()

    scene = Scene(
        "oink_base_frame_demo",
        urdf=urdf_xml,
        srdf=srdf_xml,
        package_paths=package_paths,
        yaml_config_path=model_data.yaml_config_path,
    )

    print(f"\n=== Model: {model} ===")
    print(f"Tracked frame (world marker):   {tracked_frame}")
    print(f"Follower frame (base = above):  {follower_frame}")
    joint_names = scene.getJointGroupInfo(model_data.default_joint_group).joint_names

    q_full = scene.getCurrentJointPositions()

    # Redundant Pinocchio model for visualization (mimic joints).
    model_pin = pin.buildModelFromXML(urdf_xml, mimic=True)
    collision_model = pin.buildGeomFromUrdfString(
        model_pin, urdf_xml, pin.GeometryType.COLLISION, package_dirs=package_paths
    )
    visual_model = pin.buildGeomFromUrdfString(
        model_pin, urdf_xml, pin.GeometryType.VISUAL, package_dirs=package_paths
    )

    viz = ViserVisualizer(model_pin, collision_model, visual_model)
    viz.initViewer(open=True, loadModel=True, host=host, port=port)

    # Set up the Oink solver
    oink = Oink(scene, model_data.default_joint_group)
    num_variables = len(oink.v_indices)

    scene_lock = threading.Lock()
    dt = 1.0 / control_freq

    constraints = [
        PositionLimit(oink, gain=1.0),
        VelocityLimit(
            oink,
            dt,
            np.hstack(
                [scene.getJointInfo(name).limits.max_velocity for name in joint_names]
            ),
        ),
    ]

    # Self-collision barrier: keep every declared collision pair at least
    # `self_collision_d_min` meters apart. The two arms share a workspace here, so
    # this is worth enabling to stop them interpenetrating when dragged together.
    if self_collision_num_pairs > 0:
        print(
            f"Self-collision barrier enabled with {self_collision_num_pairs} collision pair(s)."
        )
        barriers = [
            SelfCollisionBarrier(
                oink,
                scene,
                n_collision_pairs=self_collision_num_pairs,
                dt=dt,
                gain=self_collision_gain,
                safe_displacement_gain=0.01,
                d_min=self_collision_d_min,
            )
        ]
    else:
        barriers = []

    # Regularize toward the starting pose in the nullspace of the frame tasks.
    q_canonical = np.array(model_data.starting_joint_config)
    if len(q_canonical) != len(q_full):
        with scene_lock:
            q_canonical = scene.getCurrentJointPositions()
    config_options = ConfigurationTaskOptions(task_gain=1.0, lm_damping=0.0, priority=2)
    config_task = ConfigurationTask(
        oink,
        q_canonical[oink.q_indices],
        np.full(num_variables, 0.05),
        config_options,
    )

    task_options = FrameTaskOptions(
        position_cost=1.0,
        orientation_cost=0.1,
        task_gain=task_gain,
        lm_damping=lm_damping,
    )

    # Tracked task: target lives in the world frame.
    tracked_goal = CartesianConfiguration()
    tracked_goal.base_frame = "universe"
    tracked_goal.tip_frame = tracked_frame
    tracked_task = FrameTask(oink, scene, tracked_goal, task_options)

    # Follower task: target lives in the tracked frame, i.e. the tracked hand is
    # this task's (moving) base frame.
    follower_goal = CartesianConfiguration()
    follower_goal.base_frame = tracked_frame
    follower_goal.tip_frame = follower_frame
    follower_task = FrameTask(oink, scene, follower_goal, task_options)

    tasks = [tracked_task, follower_task, config_task]

    # Establish initial poses, the world marker, and the constant relative offset.
    with scene_lock:
        q_full = q_canonical.copy()
        scene.setJointPositions(q_full)
        world_T_tracked0 = scene.forwardKinematics(q_full, tracked_frame)
        world_T_follower0 = scene.forwardKinematics(q_full, follower_frame)

    # Pose of the follower hand expressed in the tracked hand's frame. Held fixed,
    # so the follower rigidly maintains this offset as the tracked hand moves.
    tracked_T_follower = np.linalg.inv(world_T_tracked0) @ world_T_follower0

    ref_filter = SE3LowPassFilter(tau=reference_filter_tau)
    ref_filter.reset(world_T_tracked0)
    raw_target = world_T_tracked0.copy()

    # Interactive marker for the tracked hand (world frame).
    controls = viz.viewer.scene.add_transform_controls(
        "/base_frame_demo/tracked_target",
        depth_test=False,
        scale=0.2,
        disable_sliders=True,
    )
    wxyz0, pos0 = _tform_to_viser(world_T_tracked0)
    controls.wxyz, controls.position = wxyz0, pos0

    # Live axes drawn at the moving base frame (the tracked hand).
    base_frame_axes = viz.viewer.scene.add_frame(
        "/base_frame_demo/base_frame",
        wxyz=wxyz0,
        position=pos0,
        axes_length=0.15,
        axes_radius=0.006,
    )

    paused = [True]

    def on_marker_update(_):
        with scene_lock:
            raw_target[:] = pin.SE3(
                pin.Quaternion(controls.wxyz[[1, 2, 3, 0]]), controls.position
            ).homogeneous
        paused[0] = False

    controls.on_update(on_marker_update)

    running = [True]

    def control_loop():
        delta_q = np.zeros(num_variables)
        delta_q_full = np.zeros(model_pin.nv)
        while running[0]:
            loop_start = time.time()
            if not paused[0]:
                with scene_lock:
                    q_current = scene.getCurrentJointPositions()

                    # Tracked task: marker pose is in the world frame and the base
                    # frame is "universe", so it passes through unchanged.
                    filtered_target = (
                        ref_filter.update(raw_target, dt)
                        if reference_filter_tau > 0
                        else raw_target
                    )
                    tracked_task.setTargetFrameTransform(filtered_target)

                    # Follower task: target is already expressed in its base frame
                    # (the tracked hand), so it is set directly and stays constant.
                    follower_task.setTargetFrameTransform(tracked_T_follower)

                    try:
                        oink.solveIk(
                            scene, tasks, constraints, barriers, delta_q, regularization
                        )
                    except RuntimeError as exc:
                        delta_q[:] = 0.0
                        print(f"Warning: IK solver failed: {exc}")

                    delta_q_full[oink.v_indices] = delta_q

                    # Validate barrier feasibility post-solve and zero delta_q on violation.
                    if barriers:
                        oink.enforceBarriers(
                            scene, barriers, delta_q_full, tolerance=0.0
                        )

                    q_current = scene.integrate(q_current, delta_q_full)
                    scene.setJointPositions(q_current)

                    # Refresh FK so oMf is current for the next iteration. This also
                    # keeps the tracked hand (the follower's base frame) up to date.
                    world_T_tracked = scene.forwardKinematics(q_current, tracked_frame)
                    scene.forwardKinematics(q_current, follower_frame)

                    viz.display(q_current)

                # Move the visualized axes to the current base-frame pose.
                base_frame_axes.wxyz, base_frame_axes.position = _tform_to_viser(
                    world_T_tracked
                )

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, dt - elapsed))

    control_thread = threading.Thread(target=control_loop, daemon=True)
    control_thread.start()

    reset_button = viz.viewer.gui.add_button("Reset")

    @reset_button.on_click
    def _reset(_):
        nonlocal tracked_T_follower
        paused[0] = True
        with scene_lock:
            q_reset = q_canonical.copy()
            scene.setJointPositions(q_reset)
            world_T_tracked = scene.forwardKinematics(q_reset, tracked_frame)
            world_T_follower = scene.forwardKinematics(q_reset, follower_frame)
            tracked_T_follower = np.linalg.inv(world_T_tracked) @ world_T_follower
            raw_target[:] = world_T_tracked
            ref_filter.reset(world_T_tracked)
            wxyz, pos = _tform_to_viser(world_T_tracked)
            controls.wxyz, controls.position = wxyz, pos
            base_frame_axes.wxyz, base_frame_axes.position = wxyz, pos
            viz.display(q_reset)

    viz.display(q_full)
    print("\nDrag the marker on the tracked hand; the follower hand holds its")
    print("pose relative to it. Press Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(10.0)
    except KeyboardInterrupt:
        running[0] = False
        control_thread.join(timeout=1.0)


if __name__ == "__main__":
    tyro.cli(main)
