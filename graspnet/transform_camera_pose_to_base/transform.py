from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict


def transform(robot_type: str, pose: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified router for camera->base pose transform.

    Expected robot module:
    - file: transform_camera_pose_to_base/<robot_type>.py
    - function: transform(pose: dict) -> dict
    """
    if not isinstance(pose, dict):
        raise ValueError("pose must be a dict with position/orientation")
    if not isinstance(robot_type, str) or not robot_type.strip():
        raise ValueError("robot_type is required")

    module_path = Path(__file__).resolve().parent / f"{robot_type.strip()}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"robot transform file not found: {module_path}")

    module_name = f"graspnet_transform_{robot_type.strip()}"
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load transform module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fn = getattr(module, "transform", None)
    if not callable(fn):
        raise AttributeError(f"{module_name}.transform is not callable")
    return fn(pose)
