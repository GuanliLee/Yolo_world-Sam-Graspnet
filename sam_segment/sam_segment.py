import cv2
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from ultralytics import SAM

from utils.image import get_image_instance
from utils.logger import get_logger
from utils.redis_pool import get_redis_pool


logger = get_logger("sam_segment", class_name="Model")


class Model:
    """SAM segmentation model implementation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config if isinstance(config, dict) else {}
        self._loaded = False
        self._gpu_id = -1
        self._device = "cpu"
        self._model: Optional[SAM] = None
        self._model_path = ""
        try:
            self._redis_pool = get_redis_pool()
        except Exception as e:
            self._redis_pool = None
            logger.warning("init redis pool failed in sam_segment: %s", e)
        if not self.config:
            self.load_config()
        self._clear_debug_dir()

    def load_config(self) -> Dict[str, Any]:
        model_dir = Path(__file__).resolve().parent
        yaml_path = model_dir / "sam_segment.yaml"
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
        """清理 debug 目录"""
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
                "model": "sam_segment",
                "model_path": self._model_path,
            }

        model_path = self._resolve_model_path()
        if model_path is None:
            self._model = None
            self._model_path = ""
            return {
                "loaded": False,
                "model": "sam_segment",
                "error": "model file not found in model directory",
            }

        self._model_path = str(model_path)
        try:
            self._model = SAM(self._model_path)
        except Exception as e:
            self._model = None
            return {
                "loaded": False,
                "model": "sam_segment",
                "model_path": self._model_path,
                "error": f"load model failed: {e}",
            }

        return {
            "loaded": True,
            "model": "sam_segment",
            "model_path": self._model_path,
        }

    def load(self, param: Dict[str, Any]) -> Dict[str, Any]:
        payload = param if isinstance(param, dict) else {}
        gpu_id = self._parse_gpu_id(payload)
        if self._loaded and self._model is not None and self._gpu_id == gpu_id:
            # Forward detection-related fields to the next step
            result = {
                "loaded": True,
                "model": "sam_segment",
                "gpu_id": self._gpu_id,
                "device": self._device,
                "model_path": self._model_path,
                "config": self.config,
                "input": payload,
            }
            # Pass through detections and related metadata if present
            if "detections" in payload:
                result["detections"] = payload["detections"]
            if "detector" in payload:
                result["detector"] = payload["detector"]
            if "data_id" in payload:
                result["data_id"] = payload["data_id"]
            if "conf_threshold" in payload:
                result["conf_threshold"] = payload["conf_threshold"]
            return result

        if self._model is None:
            load_result = self.load_model()
            if not load_result.get("loaded"):
                self._loaded = False
                load_result["gpu_id"] = gpu_id
                load_result["input"] = payload
                return load_result

        if not torch.cuda.is_available():
            self._loaded = False
            return {
                "loaded": False,
                "model": "sam_segment",
                "gpu_id": gpu_id,
                "model_path": self._model_path,
                "error": "cuda not available",
                "input": payload,
            }
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            self._loaded = False
            return {
                "loaded": False,
                "model": "sam_segment",
                "gpu_id": gpu_id,
                "model_path": self._model_path,
                "error": "invalid gpu_id",
                "input": payload,
            }

        try:
            device = torch.device(f"cuda:{gpu_id}")
            self._device = str(device)
            if self._model is not None and hasattr(self._model, "to"):
                self._model.to(device)
            _ = torch.empty((1,), device=device)
            self._loaded = True
            self._gpu_id = gpu_id
        except Exception as e:
            self._loaded = False
            return {
                "loaded": False,
                "model": "sam_segment",
                "gpu_id": gpu_id,
                "model_path": self._model_path,
                "error": f"load to gpu failed: {e}",
                "input": payload,
            }

        # Forward detection-related fields to the next step
        result = {
            "loaded": self._loaded,
            "model": "sam_segment",
            "gpu_id": self._gpu_id,
            "device": self._device,
            "model_path": self._model_path,
            "config": self.config,
            "input": payload,
        }
        # Pass through detections and related metadata if present
        if "detections" in payload:
            result["detections"] = payload["detections"]
        if "detector" in payload:
            result["detector"] = payload["detector"]
        if "data_id" in payload:
            result["data_id"] = payload["data_id"]
        if "conf_threshold" in payload:
            result["conf_threshold"] = payload["conf_threshold"]
        return result

    @staticmethod
    def _to_bbox(value: Any) -> Optional[List[float]]:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        bbox: List[float] = []
        for item in value:
            try:
                bbox.append(float(item))
            except Exception:
                return None
        return bbox

    @staticmethod
    def _extract_det_bbox_and_score(item: Dict[str, Any]) -> Optional[Tuple[List[float], float]]:
        if not isinstance(item, dict):
            return None
        bbox = Model._to_bbox(item.get("bbox"))
        if bbox is None:
            return None
        score = item.get("score", item.get("conf", item.get("confidence", 0.0)))
        try:
            score_f = float(score)
        except Exception:
            score_f = 0.0
        return bbox, score_f

    def _pick_best_bbox(self, payload: Dict[str, Any]) -> Optional[List[float]]:
        best_bbox: Optional[List[float]] = None
        best_score = float("-inf")

        detections = payload.get("detections", [])
        if isinstance(detections, list):
            for item in detections:
                parsed = self._extract_det_bbox_and_score(item)
                if parsed is None:
                    continue
                bbox, score = parsed
                if score > best_score:
                    best_bbox, best_score = bbox, score

        bboxes = payload.get("bboxes", [])
        scores = payload.get("scores", payload.get("confidences", []))
        if isinstance(bboxes, list) and bboxes:
            for idx, raw_bbox in enumerate(bboxes):
                bbox = self._to_bbox(raw_bbox)
                if bbox is None:
                    continue
                score_f = 0.0
                if isinstance(scores, list) and idx < len(scores):
                    try:
                        score_f = float(scores[idx])
                    except Exception:
                        score_f = 0.0
                if score_f > best_score:
                    best_bbox, best_score = bbox, score_f

        if best_bbox is not None:
            return best_bbox

        return self._to_bbox(payload.get("bbox"))

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        """解析布尔值"""
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
    def _safe_name(value: Any) -> str:
        """生成安全的文件名"""
        text = str(value).strip()
        if not text:
            return "image"
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
        return cleaned[:128] if cleaned else "image"

    def _save_segmentation_visualization(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        bbox: List[float],
        data_id: str
    ) -> Optional[str]:
        """保存 SAM 分割可视化结果"""
        if not isinstance(rgb, np.ndarray) or rgb.size == 0:
            return None
        if not isinstance(mask, np.ndarray) or mask.size == 0:
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

            # 确保_mask 尺寸匹配
            if mask.shape[:2] != (img_h, img_w):
                # 调整 mask 尺寸
                mask_resized = cv2.resize(mask.astype(np.uint8), (img_w, img_h),
                                         interpolation=cv2.INTER_NEAREST)
                mask = mask_resized > 0

            # 创建彩色 mask（半透明绿色叠加）
            mask_colored = np.zeros_like(vis_img)
            mask_colored[mask > 0] = [0, 255, 0]  # 绿色

            # 叠加到原图
            vis_img = cv2.addWeighted(vis_img, 0.7, mask_colored, 0.3, 0)

            # 绘制 mask 轮廓
            if mask.ndim == 2:
                mask_uint8 = mask.astype(np.uint8)
            else:
                mask_uint8 = (mask.max(axis=-1) > 0).astype(np.uint8)

            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_img, contours, -1, (0, 255, 255), 2)

            # 绘制输入 bbox（蓝色框）
            if len(bbox) == 4:
                x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
                x1 = max(0, min(img_w - 1, x1))
                y1 = max(0, min(img_h - 1, y1))
                x2 = max(0, min(img_w - 1, x2))
                y2 = max(0, min(img_h - 1, y2))
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 100, 0), 2)

            # 添加信息文本
            mask_area = np.sum(mask > 0)
            mask_ratio = mask_area / (img_h * img_w) * 100
            text_lines = [
                f"Area: {mask_area} px ({mask_ratio:.1f}%)",
                f"Masks: 1"
            ]

            y_offset = 30
            for line in text_lines:
                cv2.putText(vis_img, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

            # 保存到 debug 目录
            debug_dir = Path(__file__).resolve().parent / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            file_stem = self._safe_name(data_id)
            file_name = f"{file_stem}_{int(time.time() * 1000)}.jpg"
            file_path = debug_dir / file_name

            if cv2.imwrite(str(file_path), vis_img):
                logger.info("SAM 分割可视化保存: %s", file_path)
                return str(file_path)
            return None
        except Exception as e:
            logger.warning("保存 SAM 可视化失败: %s", e)
            return None

    def execute_inference(self, param: Dict[str, Any]) -> Dict[str, Any]:
        payload = param if isinstance(param, dict) else {}
        if not self._loaded or self._model is None:
            return {
                "masks": [],
                "segmenter": "sam_segment",
                "loaded": False,
                "error": "model is not loaded",
            }

        data_id = str(payload.get("image", "")).strip()
        if not data_id:
            return {
                "masks": [],
                "segmenter": "sam_segment",
                "loaded": self._loaded,
                "error": "image is required",
            }

        best_bbox = self._pick_best_bbox(payload)
        if best_bbox is None:
            return {
                "masks": [],
                "segmenter": "sam_segment",
                "loaded": self._loaded,
                "error": "no valid bbox in detections",
            }

        rgbd = get_image_instance().get_rgbd(data_id)
        rgb = rgbd.get("rgb")
        if rgb is None:
            return {
                "masks": [],
                "segmenter": "sam_segment",
                "loaded": self._loaded,
                "error": f"rgb image not found for image={data_id}",
            }

        results = self._model.predict(
            source=rgb,
            bboxes=[best_bbox],
            device=self._device,
            verbose=False,
        )

        masks: List[Dict[str, Any]] = []
        mask_key = None
        if results:
            result0 = results[0]
            mask_obj = getattr(result0, "masks", None)
            logger.debug("SAM 推理结果: result0 type=%s, mask_obj type=%s",
                        type(result0), type(mask_obj) if mask_obj else None)
            if mask_obj is not None and hasattr(mask_obj, "data"):
                data = mask_obj.data
                try:
                    mask_array = data.detach().cpu().numpy()
                except Exception:
                    mask_array = data.cpu().numpy() if hasattr(data, "cpu") else []

                logger.debug("mask_array shape=%s, dtype=%s, len=%s",
                            mask_array.shape if hasattr(mask_array, 'shape') else 'N/A',
                            mask_array.dtype if hasattr(mask_array, 'dtype') else 'N/A',
                            len(mask_array) if hasattr(mask_array, '__len__') else 'N/A')

                # Save first mask to Redis with 5-minute TTL.
                if len(mask_array) > 0:
                    timestamp = int(time.time() * 1000)
                    candidate_key = f"{timestamp}_mask"
                    try:
                        mask_first = mask_array[0].astype(np.uint8)
                        logger.debug("准备保存 mask 到 Redis: key=%s, shape=%s, size=%d bytes",
                                    candidate_key, mask_first.shape, mask_first.tobytes().__sizeof__())
                        if self._redis_pool is None:
                            self._redis_pool = get_redis_pool()
                        # 使用 set_binary 方法存储二进制数据
                        self._redis_pool.set_binary(candidate_key, mask_first.tobytes(), ex=300)
                        # 保存 shape 信息
                        self._redis_pool.set(
                            f"{candidate_key}:shape",
                            json.dumps(list(mask_first.shape)),
                            ex=300
                        )
                        mask_key = candidate_key
                        logger.info("成功保存 mask 到 Redis: key=%s, shape=%s", candidate_key, mask_first.shape)
                    except Exception as e:
                        logger.warning(
                            "save mask to redis failed: key=%s, error=%s",
                            candidate_key,
                            e,
                        )
                else:
                    logger.warning("mask_array 为空，无法保存到 Redis")

                for idx, mask in enumerate(mask_array):
                    area = int((mask > 0.5).sum())
                    masks.append(
                        {
                            "index": idx,
                            "area": area,
                            "height": int(mask.shape[0]) if len(mask.shape) >= 2 else 0,
                            "width": int(mask.shape[1]) if len(mask.shape) >= 2 else 0,
                        }
                    )
            else:
                logger.warning("SAM 模型未返回 mask 数据: mask_obj=%s", mask_obj)
        else:
            logger.warning("SAM 推理未返回任何结果")

        response = {
            "masks": masks,
            "mask_key": mask_key,  # 添加 mask_key 供下游使用
            "segmenter": "sam_segment",
            "gpu_id": self._gpu_id,
            "device": self._device,
            "model_path": self._model_path,
            "data_id": data_id,
            "input_bbox": best_bbox,
        }

        # 保存可视化结果（如果启用）
        debug_cfg = self.config.get("debug", {}) if isinstance(self.config, dict) else {}
        save_vis = self._as_bool(debug_cfg.get("save_visualization", False), default=False)
        if save_vis and len(mask_array) > 0:
            vis_path = self._save_segmentation_visualization(rgb, mask_array[0], best_bbox, data_id)
            if vis_path:
                response["visualization_path"] = vis_path

        return response


def create_model() -> Model:
    return Model()
