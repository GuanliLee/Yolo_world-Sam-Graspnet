from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Tuple, Union


EE_TO_CAMERA_TRANSLATION = (-0.00808418, -0.12728741, -0.03898737)
EE_TO_CAMERA_RPY = (-0.00168291, -0.01074171, 0.02452491)
BASE_TO_EE_STARTUP_POSITION = (
    -0.3570150136947632,
    0.009452000260353088,
    0.5360159873962402,
)
BASE_TO_EE_STARTUP_ORIENTATION = (
    -0.7480990290641785,
    -0.6633099913597107,
    -0.009437999688088894,
    0.01663300022482872,
)
TOOL_OFFSET_IN_TOOL = (0.0, 0.0, -0.17)

Number = Union[int, float]


def _normalize_quaternion(q: Iterable[Number]) -> Tuple[float, float, float, float]:
    x, y, z, w = (float(v) for v in q)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / norm, y / norm, z / norm, w / norm)


def _quat_mul(
    q1: Tuple[float, float, float, float], q2: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _quat_conj(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, z, w = q
    return (-x, -y, -z, w)


def _rotate_vec(
    q: Tuple[float, float, float, float], v: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    qv = (v[0], v[1], v[2], 0.0)
    qr = _quat_mul(_quat_mul(q, qv), _quat_conj(q))
    return (qr[0], qr[1], qr[2])


def _rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return _normalize_quaternion(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )
    )


def _to_xyz(position: Union[Dict[str, Number], Iterable[Number]]) -> Tuple[float, float, float]:
    if isinstance(position, dict):
        return (float(position["x"]), float(position["y"]), float(position["z"]))
    x, y, z = position
    return (float(x), float(y), float(z))


def _to_xyzw(orientation: Union[Dict[str, Number], Iterable[Number]]) -> Tuple[float, float, float, float]:
    if isinstance(orientation, dict):
        return _normalize_quaternion(
            (
                float(orientation["x"]),
                float(orientation["y"]),
                float(orientation["z"]),
                float(orientation["w"]),
            )
        )
    x, y, z, w = orientation
    return _normalize_quaternion((x, y, z, w))


def _pose_dict(p: Tuple[float, float, float], q: Tuple[float, float, float, float]) -> Dict[str, Dict[str, float]]:
    return {
        "position": {"x": p[0], "y": p[1], "z": p[2]},
        "orientation": {"x": q[0], "y": q[1], "z": q[2], "w": q[3]},
    }


def transform(pose: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Transform camera-frame pose to base-frame pose.

    Input pose format:
    {"position": {"x","y","z"}, "orientation": {"x","y","z","w"}}
    """
    if not isinstance(pose, dict):
        raise ValueError("pose must be dict")
    position = pose.get("position")
    orientation = pose.get("orientation")
    if position is None or orientation is None:
        raise ValueError("pose requires position and orientation")

    p_cam = _to_xyz(position)
    q_cam = _to_xyzw(orientation)

    q_ee_cam = _rpy_to_quaternion(*EE_TO_CAMERA_RPY)
    p_ee_cam = EE_TO_CAMERA_TRANSLATION

    p_ee_raw_rot = _rotate_vec(q_ee_cam, p_cam)
    p_ee_raw = (
        p_ee_cam[0] + p_ee_raw_rot[0],
        p_ee_cam[1] + p_ee_raw_rot[1],
        p_ee_cam[2] + p_ee_raw_rot[2],
    )
    q_ee_raw = _normalize_quaternion(_quat_mul(q_ee_cam, q_cam))

    offset_in_ee = _rotate_vec(q_ee_raw, TOOL_OFFSET_IN_TOOL)
    p_ee_offset = (
        p_ee_raw[0] + offset_in_ee[0],
        p_ee_raw[1] + offset_in_ee[1],
        p_ee_raw[2] + offset_in_ee[2],
    )
    q_ee_offset = q_ee_raw

    q_base_ee = _normalize_quaternion(BASE_TO_EE_STARTUP_ORIENTATION)
    p_base_ee = BASE_TO_EE_STARTUP_POSITION
    p_base_from_ee = _rotate_vec(q_base_ee, p_ee_offset)
    p_base = (
        p_base_ee[0] + p_base_from_ee[0],
        p_base_ee[1] + p_base_from_ee[1],
        p_base_ee[2] + p_base_from_ee[2],
    )
    q_base = _normalize_quaternion(_quat_mul(q_base_ee, q_ee_offset))

    return {
        "camera_pose": _pose_dict(p_cam, q_cam),
        "ee_pose_before_offset": _pose_dict(p_ee_raw, q_ee_raw),
        "ee_pose_after_offset": _pose_dict(p_ee_offset, q_ee_offset),
        "base_pose": _pose_dict(p_base, q_base),
    }
