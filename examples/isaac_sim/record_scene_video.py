#!/usr/bin/env python3
"""Record a short camera sweep of the Isaac Sim supermarket scene.

Run inside an Isaac Sim 5.1.0 environment. The output is a scene recording,
not a demonstration of YOLO-World, SAM, or GraspNet inference.
"""

import argparse
import math
import os
import time

from isaacsim import SimulationApp


parser = argparse.ArgumentParser()
parser.add_argument("--usd", default="/home/market2/market2_0709/market2_shelf_background_bundle_portable/new_0713_3camera.usd")
parser.add_argument("--camera", default="/World/Robot/lifting_link/head_view_camera")
parser.add_argument("--output", required=True)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--seconds", type=int, default=6)
parser.add_argument("--yaw-degrees", type=float, default=12.0)
args = parser.parse_args()

app = SimulationApp({"headless": True})

import cv2
import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage
from isaacsim.sensors.camera import Camera


def quat_multiply(left, right):
    aw, ax, ay, az = left
    bw, bx, by, bz = right
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


try:
    print("Opening stage", flush=True)
    open_stage(args.usd)
    for _ in range(20):
        app.update()
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(args.camera).IsValid():
        raise RuntimeError(f"Camera does not exist: {args.camera}")
    print("Stage and camera ready", flush=True)

    world = World(stage_units_in_meters=1.0)
    camera = Camera(prim_path=args.camera, name="scene_video_camera",
                    resolution=(args.width, args.height), frequency=args.fps)
    world.reset()
    print("World reset", flush=True)
    camera.initialize()
    camera.add_rgb_to_frame()
    camera.add_distance_to_image_plane_to_frame()
    camera.resume()
    ready = False
    for warmup in range(500):
        world.step(render=True)
        frame = camera.get_current_frame(clone=True)
        rgb = frame.get("rgb")
        if rgb is not None and getattr(rgb, "size", 0) > 0:
            print(f"First RGB frame ready after {warmup + 1} steps", flush=True)
            ready = True
            break
        if warmup % 50 == 0:
            print(f"Waiting for first RGB frame: step {warmup}", flush=True)
        time.sleep(0.02)
    if not ready:
        raise RuntimeError("Camera did not produce an initial RGB frame")
    # Let materials and lighting settle before writing the first video frame.
    for _ in range(60):
        world.step(render=True)
        time.sleep(0.02)
    print("Camera warmed up", flush=True)

    position, base_orientation = camera.get_world_pose()
    print(f"Camera pose: {position} {base_orientation}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {args.output}")
    print("Video writer open", flush=True)

    first = None
    last = None
    frame_count = args.fps * args.seconds
    for i in range(frame_count):
        phase = i / max(frame_count - 1, 1)
        yaw = math.radians(args.yaw_degrees * math.sin(2 * math.pi * phase))
        turn = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        camera.set_world_pose(position=position,
                              orientation=quat_multiply(turn, base_orientation))
        rgb = None
        for _ in range(30):
            world.step(render=True)
            rgb = camera.get_current_frame(clone=True).get("rgb")
            if rgb is not None and getattr(rgb, "size", 0) > 0:
                break
            time.sleep(0.02)
        if rgb is None:
            raise RuntimeError(f"No RGB frame at index {i}")
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise RuntimeError(f"Unexpected RGB frame shape: {rgb.shape}")
        rgb = rgb[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * 255 if rgb.max() <= 1.0 else rgb, 0, 255).astype(np.uint8)
        if i == 0:
            first = rgb.copy()
        last = rgb.copy()
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if i % args.fps == 0:
            print(f"Recorded {i}/{frame_count} frames", flush=True)
    writer.release()
    print(f"Saved {frame_count} frames at {args.fps} fps: {args.output}", flush=True)
    print(f"First/last mean RGB difference: {np.abs(first.astype(np.int16) - last.astype(np.int16)).mean():.2f}", flush=True)
except BaseException as error:
    print(f"Recording failed: {type(error).__name__}: {error}", flush=True)
    raise
finally:
    app.close()
