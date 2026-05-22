from pathlib import Path
from typing import Any, Dict, List, Optional

import shutil
import time

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO, YOLOWorld

from utils.image import get_image_instance
from utils.logger import get_logger


logger = get_logger("yolo_world", class_name="Model")


class Model:
    """YOLO-World model implementation."""

    def __init__(self) -> None:
        self.config: Dict[str, Any] = {}
        self._loaded = False
        self._gpu_id = -1
        self._device = "cpu"
        self._model = None
        self._model_path = ""
        self.load_config()
        self._clear_debug_dir()

    def load_config(self) -> Dict[str, Any]:
        model_dir = Path(__file__).resolve().parent
        yaml_path = model_dir / "yolo_world.yaml"
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

    def _clear_debug_dir(self) -> None:
        debug_dir = Path(__file__).resolve().parent / "debug"
        if not debug_dir.exists() or not debug_dir.is_dir():
            return
        try:
            for item in debug_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
        except Exception:
            # Debug artifacts cleanup should not block model initialization.
            pass

    @staticmethod
    def _parse_gpu_id(payload: Dict[str, Any]) -> int:
        try:
            return int(payload.get("gpu_id", 0))
        except Exception:
            return 0

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
        for ext in (".pt", ".pth", ".ckpt", ".onnx", ".engine", ".trt", ".tar"):
            found = list(models_dir.glob(f"*{ext}"))
            if found:
                return found[0]
        return None

    def load_model(self) -> Dict[str, Any]:
        if self._model is not None:
            return {
                "loaded": True,
                "model": "yolo_world",
                "model_path": self._model_path,
            }

        model_path = self._resolve_model_path()
        if model_path is None:
            self._model = None
            self._model_path = ""
            return {
                "loaded": False,
                "model": "yolo_world",
                "error": "model file not found in model directory",
            }

        self._model_path = str(model_path)
        try:
            self._model = YOLOWorld(self._model_path)
        except Exception:
            try:
                self._model = YOLO(self._model_path)
            except Exception as e:
                self._model = None
                return {
                    "loaded": False,
                    "model": "yolo_world",
                    "model_path": self._model_path,
                    "error": f"load model failed: {e}",
                }

        return {
            "loaded": True,
            "model": "yolo_world",
            "model_path": self._model_path,
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

        try:
            if not torch.cuda.is_available():
                self._loaded = False
                return {
                    "loaded": False,
                    "model": "yolo_world",
                    "gpu_id": gpu_id,
                    "model_path": self._model_path,
                    "error": "cuda not available",
                    "input": payload,
                }
            if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
                self._loaded = False
                return {
                    "loaded": False,
                    "model": "yolo_world",
                    "gpu_id": gpu_id,
                    "model_path": self._model_path,
                    "error": "invalid gpu_id",
                    "input": payload,
                }

            device = torch.device(f"cuda:{gpu_id}")
            self._device = str(device)
            if hasattr(self._model, "to"):
                self._model.to(device)
            _ = torch.empty((1,), device=device)
            self._loaded = True
        except Exception as e:
            self._loaded = False
            return {
                "loaded": False,
                "model": "yolo_world",
                "gpu_id": gpu_id,
                "model_path": self._model_path,
                "error": f"load to gpu failed: {e}",
                "input": payload,
            }

        return {
            "loaded": self._loaded,
            "model": "yolo_world",
            "gpu_id": self._gpu_id,
            "device": self._device,
            "model_path": self._model_path,
            "input": payload,
        }

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        value_str = str(value).strip().lower()
        if value_str in {"1", "true", "yes", "y", "on"}:
            return True
        if value_str in {"0", "false", "no", "n", "off"}:
            return False
        return default

    @staticmethod
    def _to_float_list(values: Any) -> List[float]:
        result: List[float] = []
        if not isinstance(values, (list, tuple)):
            return result
        for item in values:
            try:
                result.append(float(item))
            except Exception:
                result.append(0.0)
        return result

    @staticmethod
    def _extract_text_prompts(payload: Dict[str, Any]) -> List[str]:
        raw_prompt = payload.get("text_query")

        if isinstance(raw_prompt, str):
            prompt = raw_prompt.strip()
            if not prompt:
                return []
            # Support both "cup" and "cup, bottle" inputs.
            prompt_list = [item.strip() for item in prompt.split(",") if item.strip()]
            return prompt_list if prompt_list else [prompt]

        if isinstance(raw_prompt, (list, tuple)):
            prompt_list: List[str] = []
            for item in raw_prompt:
                item_str = str(item).strip()
                if item_str:
                    prompt_list.append(item_str)
            return prompt_list

        return []

    @staticmethod
    def _safe_name(value: Any) -> str:
        text = str(value).strip()
        if not text:
            return "image"
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
        return cleaned[:128] if cleaned else "image"

    def _save_detection_visualization(
        self, rgb: Any, detections: List[Dict[str, Any]], data_id: str
    ) -> Optional[str]:
        if not isinstance(rgb, np.ndarray) or rgb.size == 0:
            return None

        try:
            vis_img = rgb.copy()
            if vis_img.dtype != np.uint8:
                vis_img = np.clip(vis_img, 0, 255).astype(np.uint8)

            if vis_img.ndim == 2:
                vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
            elif vis_img.ndim == 3 and vis_img.shape[2] == 4:
                vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGRA2BGR)

            if vis_img.ndim != 3 or vis_img.shape[2] != 3:
                return None

            img_h, img_w = vis_img.shape[:2]
            line_thickness = max(2, int(round(min(img_h, img_w) * 0.003)))
            font_scale = max(0.5, min(img_h, img_w) / 1000.0)
            box_color = (0, 255, 0)
            text_color = (255, 255, 255)

            for det in detections:
                if not isinstance(det, dict):
                    continue
                bbox = det.get("bbox")
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue
                try:
                    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
                except Exception:
                    continue

                x1 = max(0, min(img_w - 1, x1))
                y1 = max(0, min(img_h - 1, y1))
                x2 = max(0, min(img_w - 1, x2))
                y2 = max(0, min(img_h - 1, y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                score = det.get("score", 0.0)
                label_text = str(det.get("label", "object"))
                label = f"{label_text}: {score:.2f}"

                cv2.rectangle(vis_img, (x1, y1), (x2, y2), box_color, line_thickness)

                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                text_top = max(0, y1 - text_h - baseline - 6)
                text_bottom = text_top + text_h + baseline + 6
                text_right = min(img_w - 1, x1 + text_w + 8)
                cv2.rectangle(
                    vis_img,
                    (x1, text_top),
                    (text_right, text_bottom),
                    box_color,
                    thickness=-1,
                )
                cv2.putText(
                    vis_img,
                    label,
                    (x1 + 4, text_bottom - baseline - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )

            debug_dir = Path(__file__).resolve().parent / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            file_stem = self._safe_name(data_id)
            file_name = f"{file_stem}_{int(time.time() * 1000)}.jpg"
            file_path = debug_dir / file_name

            if not cv2.imwrite(str(file_path), vis_img):
                return None
            return str(file_path)
        except Exception:
            return None

    def execute_detect(self, param: Dict[str, Any]) -> Dict[str, Any]:
        payload = param if isinstance(param, dict) else {}
        if not self._loaded or self._model is None:
            return {
                "detections": [],
                "detector": "yolo_world",
                "loaded": False,
                "error": "model is not loaded",
            }

        data_id = str(payload.get("image", "")).strip()
        if not data_id:
            return {
                "detections": [],
                "detector": "yolo_world",
                "loaded": self._loaded,
                "error": "image is required",
            }

        rgbd = get_image_instance().get_rgbd(data_id)
        rgb = rgbd.get("rgb")
        if rgb is None:
            return {
                "detections": [],
                "detector": "yolo_world",
                "loaded": self._loaded,
                "error": f"rgb image not found for image={data_id}",
            }

        detection_cfg = self.config.get("detection", {}) if isinstance(self.config, dict) else {}
        conf = self._as_float(detection_cfg.get("conf_threshold", 0.1), 0.1)
        iou = self._as_float(detection_cfg.get("iou_threshold", 0.45), 0.45)
        max_det = self._as_int(detection_cfg.get("max_detections", 10), 10)
        imgsz = self._as_int(detection_cfg.get("imgsz", 640), 640)

        prompt_list = self._extract_text_prompts(payload)
        if prompt_list and hasattr(self._model, "set_classes"):
            try:
                self._model.set_classes(prompt_list)
            except Exception:
                # Keep detection available even if open-vocabulary class update fails.
                pass

        # Probe with a low threshold to observe raw confidence distribution.
        probe_conf = min(conf, 0.001)
        results = self._model.predict(
            source=rgb,
            conf=probe_conf,
            iou=iou,
            max_det=max_det,
            imgsz=imgsz,
            device=self._device,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        pre_filter_candidate_count = 0
        pre_filter_max_score: Optional[float] = None
        if results:
            result0 = results[0]
            boxes = getattr(result0, "boxes", None)
            names = getattr(result0, "names", {}) or {}
            if boxes is not None and hasattr(boxes, "xyxy"):
                xyxy_list = boxes.xyxy.tolist() if hasattr(boxes.xyxy, "tolist") else []
                conf_list = boxes.conf.tolist() if hasattr(boxes, "conf") else []
                cls_list = boxes.cls.tolist() if hasattr(boxes, "cls") else []

                pre_filter_candidate_count = len(conf_list)
                if conf_list:
                    pre_filter_max_score = max(float(v) for v in conf_list)

                for idx, box in enumerate(xyxy_list):
                    cls_id = int(cls_list[idx]) if idx < len(cls_list) else -1
                    score = float(conf_list[idx]) if idx < len(conf_list) else 0.0
                    if score < conf:
                        continue
                    if isinstance(names, dict):
                        class_name = str(names.get(cls_id, cls_id))
                    elif isinstance(names, (list, tuple)):
                        class_name = str(names[cls_id]) if 0 <= cls_id < len(names) else str(cls_id)
                    else:
                        class_name = str(cls_id)
                    detections.append(
                        {
                            "label": class_name,
                            "score": round(float(score), 4),
                            "bbox": [round(float(value), 2) for value in box],
                        }
                    )

        response = {
            "detections": detections,
            "detector": "yolo_world",
            "gpu_id": self._gpu_id,
            "device": self._device,
            "model_path": self._model_path,
            "data_id": data_id,
            "conf_threshold": conf,
            "pre_filter_candidate_count": pre_filter_candidate_count,
            "pre_filter_max_score": pre_filter_max_score,
        }

        debug_cfg = self.config.get("debug", {}) if isinstance(self.config, dict) else {}
        save_vis = self._as_bool(debug_cfg.get("save_visualization", False), default=False)
        if save_vis:
            vis_path = self._save_detection_visualization(rgb, detections, data_id)
            if vis_path:
                response["visualization_path"] = vis_path

        return response

    def execute(self, param: Dict[str, Any]) -> Dict[str, Any]:
        payload = param if isinstance(param, dict) else {}
        detections: List[Dict[str, Any]] = payload.get("detections", [])
        if not isinstance(detections, list):
            detections = []
        return {
            "detections": detections,
            "detector": "yolo_world",
        }


def create_model() -> Model:
    return Model()
