from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from utils.image import get_image_instance
from utils.logger import get_logger
from utils.redis_pool import get_redis_pool


logger = get_logger("graspnet", class_name="Model")


class Model:
    """GraspNet model implementation with baseline network inference."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config if isinstance(config, dict) else {}
        self._loaded = False
        self._gpu_id = -1
        self._device = "cpu"
        self._model = None
        self._model_path = ""

        self._net = None
        self._pred_decode = None
        self._GraspGroup = None
        self._CameraInfo = None
        self._create_point_cloud_from_depth_image = None
        self._ModelFreeCollisionDetector = None

        if not self.config:
            self.load_config()

    def load_config(self) -> Dict[str, Any]:
        yaml_path = Path(__file__).resolve().parent / "graspnet.yaml"
        self.config = {}
        if not yaml_path.exists():
            return self.config
        try:
            with yaml_path.open("r", encoding="utf-8") as f:
                payload = yaml.safe_load(f) or {}
            if isinstance(payload, dict):
                self.config = payload
        except Exception:
            self.config = {}
        return self.config

    @staticmethod
    def _parse_gpu_id(payload: Dict[str, Any]) -> int:
        try:
            return int(payload.get("gpu_id", 0))
        except Exception:
            return 0

    @staticmethod
    def _load_module_by_path(module_name: str, file_path: Path):
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load spec for {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _resolve_model_path(self) -> Optional[Path]:
        model_dir = Path(__file__).resolve().parent
        if not self.config:
            self.load_config()

        # 统一从 models/{model_id}/model/ 目录加载模型
        models_dir = model_dir / "model"
        if not models_dir.exists():
            return None

        # 优先：从 YAML 配置的 block.model_filename 获取
        block_cfg = self.config.get("block", {}) if isinstance(self.config, dict) else {}
        model_filename = ""
        if isinstance(block_cfg, dict):
            model_filename = str(block_cfg.get("model_filename", "")).strip()

        if model_filename:
            candidate = models_dir / model_filename
            if candidate.exists():
                return candidate

        # 回退：搜索 models/{model_id}/model/ 下任意后缀文件
        for ext in (".tar", ".pt", ".pth", ".ckpt", ".onnx", ".engine", ".trt"):
            found = list(models_dir.glob(f"*{ext}"))
            if found:
                return found[0]
        return None

    def _resolve_baseline_root(self) -> Optional[Path]:
        model_dir = Path(__file__).resolve().parent
        baseline_path = model_dir / "graspnet-baseline"
        if baseline_path.exists() and baseline_path.is_dir():
            return baseline_path
        return None

    def _setup_import_paths(self, baseline_root: Path) -> None:
        roots = [
            baseline_root,
            baseline_root / "models",
            baseline_root / "dataset",
            baseline_root / "utils",
            baseline_root / "pointnet2",
            baseline_root / "pointnet2" / "pointnet2",
            baseline_root / "graspnetAPI",
        ]
        for p in roots:
            p_str = str(p)
            if p.exists() and p_str not in sys.path:
                sys.path.insert(0, p_str)

    def _load_baseline_components(self, baseline_root: Path) -> None:
        self._setup_import_paths(baseline_root)

        graspnet_module = self._load_module_by_path(
            "graspnet_baseline_model", baseline_root / "models" / "graspnet.py"
        )
        data_utils_module = self._load_module_by_path(
            "graspnet_baseline_data_utils", baseline_root / "utils" / "data_utils.py"
        )

        self._pred_decode = graspnet_module.pred_decode
        self._CameraInfo = data_utils_module.CameraInfo
        self._create_point_cloud_from_depth_image = data_utils_module.create_point_cloud_from_depth_image

        try:
            from graspnetAPI import GraspGroup  # type: ignore

            self._GraspGroup = GraspGroup
        except Exception as e:
            raise ImportError(f"failed to import graspnetAPI.GraspGroup: {e}")

        try:
            collision_module = self._load_module_by_path(
                "graspnet_baseline_collision", baseline_root / "utils" / "collision_detector.py"
            )
            self._ModelFreeCollisionDetector = collision_module.ModelFreeCollisionDetector
        except Exception:
            self._ModelFreeCollisionDetector = None

        self._model = graspnet_module.GraspNet(
            input_feature_dim=self._cfg("network", "input_feature_dim", 0),
            num_view=self._cfg("network", "num_view", 300),
            num_angle=self._cfg("network", "num_angle", 12),
            num_depth=self._cfg("network", "num_depth", 4),
            cylinder_radius=self._cfg("network", "cylinder_radius", 0.05),
            hmin=self._cfg("network", "hmin", -0.02),
            hmax_list=self._cfg("network", "hmax_list", [0.01, 0.02, 0.03, 0.04]),
            is_training=False,
        )

    def _cfg(self, section: str, key: str, default: Any) -> Any:
        sec = self.config.get(section, {}) if isinstance(self.config, dict) else {}
        if isinstance(sec, dict):
            return sec.get(key, default)
        return default

    def load_model(self) -> Dict[str, Any]:
        if self._model is not None and self._pred_decode is not None:
            return {
                "loaded": True,
                "model": "graspnet",
                "model_path": self._model_path,
            }

        baseline_root = self._resolve_baseline_root()
        if baseline_root is None:
            return {
                "loaded": False,
                "model": "graspnet",
                "error": "graspnet-baseline not found",
            }

        model_path = self._resolve_model_path()
        if model_path is None:
            self._model_path = ""
            return {
                "loaded": False,
                "model": "graspnet",
                "error": "model file not found in model directory",
            }

        self._model_path = str(model_path)

        try:
            self._load_baseline_components(baseline_root)

            ckpt = torch.load(self._model_path, map_location="cpu")
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif isinstance(ckpt, dict):
                state_dict = ckpt
            else:
                return {
                    "loaded": False,
                    "model": "graspnet",
                    "error": "invalid checkpoint format",
                }
            self._model.load_state_dict(state_dict)
            self._model.eval()
        except Exception as e:
            self._model = None
            self._pred_decode = None
            return {
                "loaded": False,
                "model": "graspnet",
                "error": f"failed to load baseline model: {e}",
            }

        return {
            "loaded": True,
            "model": "graspnet",
            "model_path": self._model_path,
            "baseline_root": str(baseline_root),
        }

    def load(self, param: Dict[str, Any]) -> Dict[str, Any]:
        payload = param if isinstance(param, dict) else {}
        gpu_id = self._parse_gpu_id(payload)
        self._gpu_id = gpu_id

        if self._model is None:
            load_result = self.load_model()
            if not load_result.get("loaded"):
                self._loaded = False
                load_result["gpu_id"] = gpu_id
                load_result["input"] = payload
                return load_result

        # Prefer cuda when valid, fallback to cpu.
        if torch.cuda.is_available() and 0 <= gpu_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")

        try:
            self._device = str(device)
            self._model.to(device)
            _ = torch.empty((1,), device=device)
            self._loaded = True
        except Exception as e:
            self._loaded = False
            return {
                "loaded": False,
                "model": "graspnet",
                "gpu_id": gpu_id,
                "model_path": self._model_path,
                "error": f"load to device failed: {e}",
                "input": payload,
            }

        # Forward segmentation-related fields to the next step
        result = {
            "loaded": self._loaded,
            "model": "graspnet",
            "gpu_id": self._gpu_id,
            "device": self._device,
            "model_path": self._model_path,
            "input": payload,
        }
        # Pass through masks and related metadata if present
        if "masks" in payload:
            result["masks"] = payload["masks"]
        if "segmenter" in payload:
            result["segmenter"] = payload["segmenter"]
        if "mask_key" in payload:
            result["mask_key"] = payload["mask_key"]
        return result

    def _get_cam_k_from_config(self) -> List[List[float]]:
        camera_cfg = self.config.get("camera", {}) if isinstance(self.config, dict) else {}
        intr = camera_cfg.get("intrinsics", {}) if isinstance(camera_cfg, dict) else {}
        fx = float(intr.get("fx", 391.0689697265625))
        fy = float(intr.get("fy", 390.7862548828125))
        cx = float(intr.get("cx", 321.19244384765625))
        cy = float(intr.get("cy", 205.70883178710938))
        return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]

    @staticmethod
    def _to_mask(mask_value: Any, height: int, width: int) -> Optional[np.ndarray]:
        if isinstance(mask_value, np.ndarray):
            mask = mask_value
        elif isinstance(mask_value, list):
            try:
                mask = np.array(mask, dtype=np.uint8)
            except Exception:
                return None
        else:
            return None
        if mask.ndim != 2:
            return None
        if mask.shape != (height, width):
            return None
        return mask

    @staticmethod
    def _bbox_to_mask(bbox: List[float], height: int, width: int) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        x1_i = max(0, min(width - 1, int(round(x1))))
        y1_i = max(0, min(height - 1, int(round(y1))))
        x2_i = max(0, min(width - 1, int(round(x2))))
        y2_i = max(0, min(height - 1, int(round(y2))))
        if x2_i < x1_i:
            x1_i, x2_i = x2_i, x1_i
        if y2_i < y1_i:
            y1_i, y2_i = y2_i, y1_i
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y1_i : y2_i + 1, x1_i : x2_i + 1] = 1
        return mask

    @staticmethod
    def _extract_bbox_from_param(payload: Dict[str, Any]) -> Optional[List[float]]:
        raw_bbox = payload.get("input_bbox", payload.get("bbox"))
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            return None
        out: List[float] = []
        for v in raw_bbox:
            try:
                out.append(float(v))
            except Exception:
                return None
        return out

    @staticmethod
    def _normalize_depth_to_m(depth: np.ndarray) -> np.ndarray:
        depth_f = depth.astype(np.float32)
        if np.nanmax(depth_f) > 20.0:
            depth_f = depth_f / 1000.0
        return depth_f

    def _mask_from_payload(self, payload: Dict[str, Any], h: int, w: int) -> Optional[np.ndarray]:
        # 优先从 Redis 读取 mask（通过 mask_key）
        mask_key = payload.get("mask_key")
        if mask_key:
            try:
                redis_client = get_redis_pool().redis_client
                mask_raw = redis_client.get(mask_key)
                if mask_raw:
                    mask_array = np.frombuffer(mask_raw, dtype=np.uint8)
                    shape_raw = redis_client.get(f"{mask_key}:shape")
                    shape: Optional[Tuple[int, ...]] = None
                    if shape_raw:
                        try:
                            if len(shape_raw) in (8, 12, 16):
                                shape = tuple(np.frombuffer(shape_raw, dtype=np.int32))
                            else:
                                import json
                                shape = tuple(json.loads(shape_raw.decode("utf-8")))
                        except Exception:
                            shape = None

                    if shape is not None and int(np.prod(shape)) == mask_array.size:
                        return mask_array.reshape(shape)

                    # 如果没有 shape 信息，尝试用图像尺寸
                    if mask_array.size == h * w:
                        return mask_array.reshape((h, w))
            except Exception:
                pass  # Redis 读取失败，继续使用其他方法

        # 回原逻辑：直接使用 mask 参数
        mask = self._to_mask(payload.get("mask"), h, w)
        if mask is not None:
            return mask

        # 回原逻辑：使用 bbox 生成 mask
        bbox = self._extract_bbox_from_param(payload)
        if bbox is None:
            return None
        return self._bbox_to_mask(bbox, h, w)

    @staticmethod
    def _filter_vertical(grasps: List[Any], vertical_threshold_deg: float) -> List[Tuple[Any, float]]:
        vertical = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        angle_threshold = np.deg2rad(vertical_threshold_deg)
        filtered: List[Tuple[Any, float]] = []

        for g in grasps:
            approach_dir = g.rotation_matrix[:, 0]
            cos_angle = np.clip(np.dot(approach_dir, vertical), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            if angle < angle_threshold:
                filtered.append((g, float(np.degrees(angle))))

        filtered.sort(key=lambda x: x[0].score, reverse=True)
        return filtered

    def _pose_to_dict(self, pose_matrix: np.ndarray) -> Optional[Dict[str, Any]]:
        try:
            from scipy.spatial.transform import Rotation as R

            quat = R.from_matrix(pose_matrix[:3, :3]).as_quat()  # [x,y,z,w]
            return {
                "position": {
                    "x": float(pose_matrix[0, 3]),
                    "y": float(pose_matrix[1, 3]),
                    "z": float(pose_matrix[2, 3]),
                },
                "orientation": {
                    "x": float(quat[0]),
                    "y": float(quat[1]),
                    "z": float(quat[2]),
                    "w": float(quat[3]),
                },
            }
        except Exception:
            return None

    @staticmethod
    def _normalize_robot_type(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        robot_type = value.strip()
        if robot_type.endswith(".py"):
            robot_type = robot_type[:-3]
        return robot_type.strip()

    def _resolve_robot_type(self, preferred_robot_type: Any = None) -> str:
        """Resolve robot_type with fallback: input param -> config -> default."""
        default_robot_type = "bobac3_horizontal"
        transform_dir = Path(__file__).resolve().parent / "transform_camera_pose_to_base"

        preferred = self._normalize_robot_type(preferred_robot_type)
        if preferred:
            preferred_path = transform_dir / f"{preferred}.py"
            if preferred_path.exists():
                return preferred
            logger.warning(
                "graspnet transform robot_type from input not found, fallback to config: robot_type=%s, file=%s",
                preferred,
                preferred_path,
            )

        transform_cfg = self.config.get("transform", {}) if isinstance(self.config, dict) else {}
        config_robot_type = default_robot_type
        if isinstance(transform_cfg, dict):
            config_candidate = self._normalize_robot_type(transform_cfg.get("robot_type", default_robot_type))
            if config_candidate:
                config_robot_type = config_candidate

        config_path = transform_dir / f"{config_robot_type}.py"
        if config_path.exists():
            return config_robot_type

        logger.warning(
            "graspnet transform robot_type from config not found, fallback to default: robot_type=%s, file=%s",
            config_robot_type,
            config_path,
        )
        return default_robot_type

    def transform_pose_to_base(
        self, pose: Dict[str, Any], preferred_robot_type: Any = None
    ) -> Optional[Dict[str, Any]]:
        """Transform camera-frame pose to base-frame pose by robot_type routing."""
        robot_type = self._resolve_robot_type(preferred_robot_type)
        try:
            module_path = Path(__file__).resolve().parent / "transform_camera_pose_to_base" / "transform.py"
            module = self._load_module_by_path("graspnet_transform_router", module_path)
            transform_fn = getattr(module, "transform", None)
            if not callable(transform_fn):
                logger.error(
                    "graspnet transform router missing callable transform: module_path=%s",
                    module_path,
                )
                return None
            return transform_fn(robot_type=robot_type, pose=pose)
        except Exception:
            logger.exception("graspnet camera->base transform failed: robot_type=%s", robot_type)
            return None

    def postprocess_result(self, param: Dict[str, Any]) -> Dict[str, Any]:
        """Transform camera-frame pose to base-frame pose and return success flag."""
        payload = param if isinstance(param, dict) else {}
        result = dict(payload)
        if not result.get("success"):
            return {"success": False, "pose": None}

        pose = result.get("pose")
        if not isinstance(pose, dict):
            logger.error("graspnet postprocess missing camera pose dict")
            return {"success": False, "pose": None}

        requested_robot_type = payload.get("robot_type")
        converted = self.transform_pose_to_base(pose, preferred_robot_type=requested_robot_type)
        if isinstance(converted, dict):
            base_pose = converted.get("base_pose")
            if isinstance(base_pose, dict):
                # 只返回机器人基座坐标系的 pose
                return {"success": True, "pose": base_pose}

        logger.error("graspnet postprocess failed to produce base_pose")
        return {"success": False, "pose": None}

    def execute_inference(self, param: Dict[str, Any]) -> Dict[str, Any]:
        payload = param if isinstance(param, dict) else {}
        if not self._loaded or self._model is None or self._pred_decode is None:
            return {
                "success": False,
                "error": "model is not loaded",
                "planner": "graspnet",
            }

        data_id = str(payload.get("image", "")).strip()
        if not data_id:
            return {
                "success": False,
                "error": "image is required",
                "planner": "graspnet",
            }

        rgbd = get_image_instance().get_rgbd(data_id)
        rgb = rgbd.get("rgb")
        depth = rgbd.get("depth")
        if rgb is None or depth is None:
            return {
                "success": False,
                "error": f"rgb/depth image not found for image={data_id}",
                "planner": "graspnet",
            }

        depth_m = self._normalize_depth_to_m(depth)
        h, w = depth_m.shape[:2]
        mask = self._mask_from_payload(payload, h, w)
        if mask is None:
            return {
                "success": False,
                "error": "mask or bbox(input_bbox) is required",
                "planner": "graspnet",
            }

        cam_k = payload.get("cam_K")
        if not isinstance(cam_k, list) or len(cam_k) != 3:
            cam_k = self._get_cam_k_from_config()

        # 1) point cloud generation
        camera = self._CameraInfo(
            w,
            h,
            float(cam_k[0][0]),
            float(cam_k[1][1]),
            float(cam_k[0][2]),
            float(cam_k[1][2]),
            1.0,
        )
        cloud = self._create_point_cloud_from_depth_image(depth_m, camera, organized=True)

        min_depth = float(self._cfg("detection", "min_depth", 0.01))
        max_depth = float(self._cfg("detection", "max_depth", 2.0))
        valid_mask = (mask > 0) & (depth_m > min_depth) & (depth_m < max_depth)

        cloud_masked = cloud[valid_mask]
        if len(cloud_masked) < 50:
            return {
                "success": False,
                "error": "insufficient_points",
                "planner": "graspnet",
            }

        color_float = rgb.astype(np.float32) / 255.0
        color_masked = color_float[valid_mask]

        # 2) sampling
        num_point = int(self._cfg("detection", "num_point", 3000))
        if len(cloud_masked) >= num_point:
            idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = cloud_masked[idxs]

        # 3) network inference
        device = torch.device(self._device)
        cloud_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
        end_points = {
            "point_clouds": cloud_tensor,
            "cloud_colors": color_masked[idxs],
        }
        with torch.no_grad():
            end_points = self._model(end_points)
            grasp_preds = self._pred_decode(end_points)

        gg = self._GraspGroup(grasp_preds[0].detach().cpu().numpy())
        if len(gg) == 0:
            return {
                "success": False,
                "error": "no_grasp_candidates",
                "planner": "graspnet",
            }

        # 4) optional collision detection (requires open3d)
        if self._ModelFreeCollisionDetector is not None:
            try:
                mfcdetector = self._ModelFreeCollisionDetector(
                    cloud_masked.astype(np.float32),
                    voxel_size=float(self._cfg("collision", "voxel_size", 0.01)),
                )
                collision_mask = mfcdetector.detect(
                    gg,
                    approach_dist=float(self._cfg("collision", "approach_dist", 0.05)),
                    collision_thresh=float(self._cfg("collision", "collision_thresh", 0.01)),
                )
                gg = gg[~collision_mask]
            except Exception:
                pass

        if len(gg) == 0:
            return {
                "success": False,
                "error": "no_valid_grasp_after_collision",
                "planner": "graspnet",
            }

        # 5) nms + ranking + vertical filter
        gg.nms().sort_by_score()
        all_grasps = list(gg)

        vertical_threshold = float(self._cfg("detection", "vertical_threshold", 30))
        filtered = self._filter_vertical(all_grasps, vertical_threshold)
        if len(filtered) == 0:
            filtered = self._filter_vertical(all_grasps, 45.0)
        best_grasp = filtered[0][0] if len(filtered) > 0 else all_grasps[0]

        # 6) pose transform: GraspNet frame -> robot gripper frame
        pose_matrix = np.eye(4, dtype=np.float32)
        r_grasp = best_grasp.rotation_matrix
        logger.info("原始 r_grasp:\n%s", r_grasp)
        logger.info("r_grasp 行列式: %s", np.linalg.det(r_grasp))
        logger.info("best_grasp.translation: %s", best_grasp.translation)
        r_g1_to_g2 = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        r_transformed = r_grasp @ r_g1_to_g2
        logger.info("转换后 r_transformed:\n%s", r_transformed)
        if np.linalg.det(r_transformed) < 0:
            r_transformed[:, 1] *= -1

        pose_matrix[:3, :3] = r_transformed
        pose_matrix[:3, 3] = best_grasp.translation
        pose_matrix[2, 3] -= 0.02

        pose_dict = self._pose_to_dict(pose_matrix)

        result = {
            "success": True,
            "planner": "graspnet",
            "pose_matrix": pose_matrix.tolist(),
            "translation": [
                float(best_grasp.translation[0]),
                float(best_grasp.translation[1]),
                float(best_grasp.translation[2]),
            ],
            "rotation_matrix": r_transformed.tolist(),
            "grasp_width": float(best_grasp.width),
            "grasp_score": float(best_grasp.score),
            "gpu_id": self._gpu_id,
            "device": self._device,
            "model_path": self._model_path,
            "data_id": data_id,
        }
        if pose_dict is not None:
            result["pose"] = pose_dict
        # 直接返回结果，不调用 postprocess_result（由 DAG 中的 postprocess_result 节点处理）
        return result

    def inference_self_with_offset(self, param: Dict[str, Any]) -> Dict[str, Any]:
        payload = param if isinstance(param, dict) else {}
        if not self._loaded or self._model is None or self._pred_decode is None:
            return {
                "success": False,
                "error": "model is not loaded",
                "planner": "graspnet",
            }

        data_id = str(payload.get("image", "")).strip()
        if not data_id:
            return {
                "success": False,
                "error": "image is required",
                "planner": "graspnet",
            }

        rgbd = get_image_instance().get_rgbd(data_id)
        rgb = rgbd.get("rgb")
        depth = rgbd.get("depth")
        if rgb is None or depth is None:
            return {
                "success": False,
                "error": f"rgb/depth image not found for image={data_id}",
                "planner": "graspnet",
            }

        depth_m = self._normalize_depth_to_m(depth)
        h, w = depth_m.shape[:2]
        mask = self._mask_from_payload(payload, h, w)
        if mask is None:
            return {
                "success": False,
                "error": "mask or bbox(input_bbox) is required",
                "planner": "graspnet",
            }

        cam_k = payload.get("cam_K")
        if not isinstance(cam_k, list) or len(cam_k) != 3:
            cam_k = self._get_cam_k_from_config()

        # 1) point cloud generation
        camera = self._CameraInfo(
            w,
            h,
            float(cam_k[0][0]),
            float(cam_k[1][1]),
            float(cam_k[0][2]),
            float(cam_k[1][2]),
            1.0,
        )
        cloud = self._create_point_cloud_from_depth_image(depth_m, camera, organized=True)

        min_depth = float(self._cfg("detection", "min_depth", 0.01))
        max_depth = float(self._cfg("detection", "max_depth", 2.0))
        valid_mask = (mask > 0) & (depth_m > min_depth) & (depth_m < max_depth)

        cloud_masked = cloud[valid_mask]
        if len(cloud_masked) < 50:
            return {
                "success": False,
                "error": "insufficient_points",
                "planner": "graspnet",
            }

        color_float = rgb.astype(np.float32) / 255.0
        color_masked = color_float[valid_mask]

        # 2) sampling
        num_point = int(self._cfg("detection", "num_point", 3000))
        if len(cloud_masked) >= num_point:
            idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = cloud_masked[idxs]

        # 3) network inference
        device = torch.device(self._device)
        cloud_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
        end_points = {
            "point_clouds": cloud_tensor,
            "cloud_colors": color_masked[idxs],
        }
        with torch.no_grad():
            end_points = self._model(end_points)
            grasp_preds = self._pred_decode(end_points)

        gg = self._GraspGroup(grasp_preds[0].detach().cpu().numpy())
        if len(gg) == 0:
            return {
                "success": False,
                "error": "no_grasp_candidates",
                "planner": "graspnet",
            }

        # 4) optional collision detection (requires open3d)
        if self._ModelFreeCollisionDetector is not None:
            try:
                mfcdetector = self._ModelFreeCollisionDetector(
                    cloud_masked.astype(np.float32),
                    voxel_size=float(self._cfg("collision", "voxel_size", 0.01)),
                )
                collision_mask = mfcdetector.detect(
                    gg,
                    approach_dist=float(self._cfg("collision", "approach_dist", 0.05)),
                    collision_thresh=float(self._cfg("collision", "collision_thresh", 0.01)),
                )
                gg = gg[~collision_mask]
            except Exception:
                pass

        if len(gg) == 0:
            return {
                "success": False,
                "error": "no_valid_grasp_after_collision",
                "planner": "graspnet",
            }

        # 5) nms + ranking + vertical filter
        gg.nms().sort_by_score()
        all_grasps = list(gg)

        vertical_threshold = float(self._cfg("detection", "vertical_threshold", 30))
        filtered = self._filter_vertical(all_grasps, vertical_threshold)
        if len(filtered) == 0:
            filtered = self._filter_vertical(all_grasps, 45.0)
        best_grasp = filtered[0][0] if len(filtered) > 0 else all_grasps[0]

        # 6) pose transform: GraspNet frame -> robot gripper frame
        pose_matrix = np.eye(4, dtype=np.float32)
        r_grasp = best_grasp.rotation_matrix
        r_g1_to_g2 = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        r_transformed = r_grasp @ r_g1_to_g2
        if np.linalg.det(r_transformed) < 0:
            r_transformed[:, 1] *= -1

        grasp_depth = best_grasp.depth
        tcp_offset = float(self._cfg("detection", "tcp_offset", 0.0))
        safety_margin = float(self._cfg("detection", "safety_margin", 0.01))
        fixed_offset = float(self._cfg("detection", "fixed_offset", -0.05))

        total_retreat = grasp_depth + tcp_offset + safety_margin + fixed_offset
        approach_dir = r_transformed[:, 2]
        adjusted_position = best_grasp.translation - approach_dir * total_retreat

        pose_matrix[:3, :3] = r_transformed
        pose_matrix[:3, 3] = adjusted_position

        pose_dict = self._pose_to_dict(pose_matrix)

        result = {
            "success": True,
            "planner": "graspnet",
            "pose_matrix": pose_matrix.tolist(),
            "translation": [
                float(adjusted_position[0]),
                float(adjusted_position[1]),
                float(adjusted_position[2]),
            ],
            "rotation_matrix": r_transformed.tolist(),
            "grasp_width": float(best_grasp.width),
            "grasp_score": float(best_grasp.score),
            "gpu_id": self._gpu_id,
            "device": self._device,
            "model_path": self._model_path,
            "data_id": data_id,
        }
        if pose_dict is not None:
            result["pose"] = pose_dict
        return result


def create_model() -> Model:
    return Model()
