#!/usr/bin/env python3
import argparse
import asyncio
import math
import os
import time

from isaacsim import SimulationApp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--usd",
        default="/home/market2/market2_0709/market2_shelf_background_bundle_portable/new_0713_3camera.usd",
    )
    parser.add_argument(
        "--camera",
        default="/World/Robot/lifting_link/head_view_camera",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/market2/graspnet_input",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--warmup-frames", type=int, default=80)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--disable-gravity",
        action="store_true",
        help="Disable gravity while positioning a robot for an offline RGB-D snapshot.",
    )
    parser.add_argument("--robot-base", nargs=4, type=float, metavar=("X", "Y", "Z", "YAW"))
    parser.add_argument("--lift", type=float)
    parser.add_argument(
        "--arm-joints",
        nargs=6,
        type=float,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Optional six revolute joint positions in radians.",
    )
    parser.add_argument(
        "--arm-joint-root",
        help="USD path containing joint_1 through joint_6.",
    )
    parser.add_argument(
        "--reference-frame",
        help=(
            "Optional USD prim whose local frame is used as the grasp-planning "
            "frame (for Bobac, pass the base_link_arm prim)."
        ),
    )
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
import omni
import omni.replicator.core as rep
import scipy.io as scio
from PIL import Image
from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage
from isaacsim.sensors.camera import Camera
from pxr import Gf, Usd, UsdGeom


def camera_intrinsic_matrix(camera_prim, width, height):
    usd_cam = UsdGeom.Camera(camera_prim)
    focal_length = float(usd_cam.GetFocalLengthAttr().Get())
    horiz_aperture = float(usd_cam.GetHorizontalApertureAttr().Get())
    vert_aperture_attr = usd_cam.GetVerticalApertureAttr().Get()
    if vert_aperture_attr is None or float(vert_aperture_attr) == 0.0:
        vert_aperture = horiz_aperture * float(height) / float(width)
    else:
        vert_aperture = float(vert_aperture_attr)

    fx = width * focal_length / horiz_aperture
    fy = height * focal_length / vert_aperture
    cx = width * 0.5
    cy = height * 0.5
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def prim_to_world_matrix(prim):
    local_to_world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    origin = local_to_world.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
    axis_x = local_to_world.Transform(Gf.Vec3d(1.0, 0.0, 0.0)) - origin
    axis_y = local_to_world.Transform(Gf.Vec3d(0.0, 1.0, 0.0)) - origin
    axis_z = local_to_world.Transform(Gf.Vec3d(0.0, 0.0, 1.0)) - origin

    rotation_with_scale = np.column_stack(
        [
            np.asarray(axis_x, dtype=np.float64),
            np.asarray(axis_y, dtype=np.float64),
            np.asarray(axis_z, dtype=np.float64),
        ]
    )
    # Imported robot assets may carry a 0.01 unit scale in their hierarchy.
    # A pose transform for metric RGB-D points must contain a unit rotation,
    # not that authored geometry scale.
    u, _, vt = np.linalg.svd(rotation_with_scale)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(origin, dtype=np.float64)
    return transform


def camera_cv_to_world_matrix(camera_prim):
    camera_usd_to_world = prim_to_world_matrix(camera_prim)

    # GraspNet point clouds use the common CV camera frame:
    # +X right, +Y down, +Z forward. USD cameras use +X right, +Y up, -Z forward.
    camera_cv_to_usd = np.diag([1.0, -1.0, -1.0, 1.0])
    return camera_usd_to_world @ camera_cv_to_usd


def get_frame_value(frame, *names):
    for name in names:
        if name in frame:
            value = frame[name]
            if isinstance(value, dict) and "data" in value:
                value = value["data"]
            return value
    raise KeyError(f"none of {names} are present; available keys={list(frame.keys())}")


def set_robot_pose(stage, x, y, z, yaw):
    robot = stage.GetPrimAtPath("/World/Robot")
    if not robot.IsValid():
        raise RuntimeError("robot prim not found: /World/Robot")
    xform = UsdGeom.Xformable(robot)
    ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}
    (ops.get("xformOp:translate") or xform.AddTranslateOp()).Set(Gf.Vec3d(x, y, z))
    (ops.get("xformOp:rotateZ") or xform.AddRotateZOp()).Set(np.degrees(yaw))


def set_lift(stage, lift):
    joint = stage.GetPrimAtPath("/World/Robot/joints/lifting_joint")
    if not joint.IsValid():
        return
    for attr_name in ("drive:linear:physics:targetPosition", "state:linear:physics:position"):
        attr = joint.GetAttribute(attr_name)
        if attr and attr.IsValid():
            attr.Set(float(lift))


def set_arm_joints(stage, joint_root, positions):
    if not joint_root:
        raise ValueError("--arm-joint-root is required with --arm-joints")
    for index, position in enumerate(positions, 1):
        path = f"{joint_root.rstrip('/')}/joint_{index}"
        joint = stage.GetPrimAtPath(path)
        if not joint.IsValid():
            raise RuntimeError(f"arm joint prim not found: {path}")
        position_degrees = math.degrees(float(position))
        changed = False
        for attr_name in (
            "drive:angular:physics:targetPosition",
            "state:angular:physics:position",
        ):
            attr = joint.GetAttribute(attr_name)
            if attr and attr.IsValid():
                attr.Set(position_degrees)
                changed = True
        if not changed:
            raise RuntimeError(f"arm joint has no writable state/drive position: {path}")


async def capture_replicator_rgbd(camera_path, width, height, warmup_frames):
    render_product = rep.create.render_product(camera_path, (width, height))
    rgb_annot = rep.annotators.get("rgb")
    depth_annot = rep.annotators.get("distance_to_image_plane")
    rgb_annot.attach(render_product)
    depth_annot.attach(render_product)

    for frame_idx in range(max(warmup_frames, 1)):
        await rep.orchestrator.step_async()
        if frame_idx % 25 == 0:
            print(f"[export_rgbd] replicator warmup {frame_idx}", flush=True)

    rgb = rgb_annot.get_data()
    depth = depth_annot.get_data()
    print(
        f"[export_rgbd] replicator rgb type={type(rgb)} shape={getattr(rgb, 'shape', None)} dtype={getattr(rgb, 'dtype', None)}",
        flush=True,
    )
    print(
        f"[export_rgbd] replicator depth type={type(depth)} shape={getattr(depth, 'shape', None)} dtype={getattr(depth, 'dtype', None)}",
        flush=True,
    )
    rgb_annot.detach()
    depth_annot.detach()
    render_product.destroy()
    return rgb, depth


def main():
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[export_rgbd] opening stage: {args.usd}", flush=True)
    open_stage(args.usd)
    print("[export_rgbd] stage open requested", flush=True)
    for _ in range(20):
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    print(f"[export_rgbd] stage object: {stage}", flush=True)
    if args.robot_base is not None:
        print(f"[export_rgbd] setting robot base: {args.robot_base}", flush=True)
        set_robot_pose(stage, *args.robot_base)
    if args.lift is not None:
        print(f"[export_rgbd] setting lift: {args.lift}", flush=True)
        set_lift(stage, args.lift)
    prim = stage.GetPrimAtPath(args.camera)
    print(f"[export_rgbd] camera prim valid={prim.IsValid()} path={args.camera}", flush=True)
    if not prim.IsValid():
        raise RuntimeError(f"camera prim not found: {args.camera}")

    print("[export_rgbd] creating world/camera wrapper", flush=True)
    world = World(stage_units_in_meters=1.0)
    if args.disable_gravity:
        world.get_physics_context().set_gravity(0.0)
        print("[export_rgbd] disabled gravity", flush=True)
    camera = Camera(
        prim_path=args.camera,
        name="graspnet_export_camera",
        resolution=(args.width, args.height),
        frequency=30,
    )
    print("[export_rgbd] world reset begin", flush=True)
    world.reset()
    print("[export_rgbd] world reset done", flush=True)
    if args.arm_joints is not None:
        print(f"[export_rgbd] setting arm joints (rad): {args.arm_joints}", flush=True)
        set_arm_joints(stage, args.arm_joint_root, args.arm_joints)
        for _ in range(30):
            world.step(render=False)
    print("[export_rgbd] camera initialize begin", flush=True)
    camera.initialize()
    print("[export_rgbd] camera initialize done", flush=True)
    print("[export_rgbd] attach rgb/depth begin", flush=True)
    camera.add_rgb_to_frame()
    camera.add_distance_to_image_plane_to_frame()
    camera.resume()
    print("[export_rgbd] attach rgb/depth done", flush=True)

    print("[export_rgbd] waiting for camera current_frame rgb/depth", flush=True)
    rgb_value = None
    depth_value = None
    for frame_idx in range(1000):
        world.step(render=True)
        if frame_idx < args.warmup_frames:
            time.sleep(0.02)
            continue
        current_frame = camera.get_current_frame(clone=True)
        rgb_value = current_frame.get("rgb")
        depth_value = current_frame.get("distance_to_image_plane")
        if frame_idx % 50 == 0:
            print(
                f"[export_rgbd] wait {frame_idx}: rgb={getattr(rgb_value, 'shape', None)} depth={getattr(depth_value, 'shape', None)}",
                flush=True,
            )
        if (
            rgb_value is not None
            and depth_value is not None
            and getattr(rgb_value, "size", 0) > 0
            and getattr(depth_value, "size", 0) > 0
        ):
            break
        time.sleep(0.02)
    if rgb_value is None or depth_value is None:
        raise RuntimeError("Camera did not produce rgb/depth data")
    print(
        f"[export_rgbd] rgb raw type={type(rgb_value)} shape={getattr(rgb_value, 'shape', None)} dtype={getattr(rgb_value, 'dtype', None)}",
        flush=True,
    )
    print(
        f"[export_rgbd] depth raw type={type(depth_value)} shape={getattr(depth_value, 'shape', None)} dtype={getattr(depth_value, 'dtype', None)}",
        flush=True,
    )
    rgb = np.array(rgb_value)
    print(f"[export_rgbd] rgb np shape={rgb.shape} dtype={rgb.dtype}", flush=True)
    depth = np.array(depth_value)
    print(f"[export_rgbd] depth np shape={depth.shape} dtype={depth.dtype}", flush=True)

    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    print(f"[export_rgbd] rgb trimmed shape={rgb.shape} dtype={rgb.dtype}", flush=True)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255.0 if rgb.max() <= 1.0 else rgb, 0, 255).astype(np.uint8)
    print("[export_rgbd] rgb converted", flush=True)

    depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    depth[depth < 0.0] = 0.0
    depth_mm = np.clip(depth * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    workspace_mask = (depth_mm > 0).astype(np.uint8) * 255
    print("[export_rgbd] depth/mask converted", flush=True)
    intrinsic = camera_intrinsic_matrix(prim, args.width, args.height)
    cv_to_world = camera_cv_to_world_matrix(prim)
    reference_to_world = None
    cv_to_reference = None
    reference_prim = None
    if args.reference_frame:
        reference_prim = stage.GetPrimAtPath(args.reference_frame)
        if not reference_prim.IsValid():
            raise RuntimeError(f"reference-frame prim not found: {args.reference_frame}")
        reference_to_world = prim_to_world_matrix(reference_prim)
        cv_to_reference = np.linalg.inv(reference_to_world) @ cv_to_world
    print(f"[export_rgbd] intrinsic={intrinsic}", flush=True)
    print(f"[export_rgbd] camera_cv_to_world={cv_to_world}", flush=True)
    if cv_to_reference is not None:
        print(
            f"[export_rgbd] camera_cv_to_reference ({args.reference_frame})={cv_to_reference}",
            flush=True,
        )

    print("[export_rgbd] saving images", flush=True)
    Image.fromarray(rgb).save(os.path.join(args.output_dir, "color.png"))
    Image.fromarray(depth_mm).save(os.path.join(args.output_dir, "depth.png"))
    Image.fromarray(workspace_mask).save(os.path.join(args.output_dir, "workspace_mask.png"))
    print("[export_rgbd] saving meta.mat", flush=True)
    metadata = {
        "intrinsic_matrix": intrinsic,
        "camera_cv_to_world": cv_to_world,
        "factor_depth": np.array([[1000.0]], dtype=np.float32),
        "camera_prim_path": np.array([args.camera], dtype=object),
        "capture_time": np.array([[time.time()]], dtype=np.float64),
    }
    if cv_to_reference is not None:
        metadata.update(
            {
                "camera_cv_to_reference": cv_to_reference,
                "reference_to_world": reference_to_world,
                "reference_frame_path": np.array([args.reference_frame], dtype=object),
            }
        )
    scio.savemat(os.path.join(args.output_dir, "meta.mat"), metadata)
    print(f"[export_rgbd] rgb shape={rgb.shape} depth_m range=({depth[depth > 0].min() if np.any(depth > 0) else 0:.4f}, {depth.max():.4f})")
    print(f"[export_rgbd] saved GraspNet input to {args.output_dir}", flush=True)


try:
    main()
finally:
    simulation_app.close()
