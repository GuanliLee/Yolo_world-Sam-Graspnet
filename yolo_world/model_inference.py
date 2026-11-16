"""
YOLO-World 开放词汇检测推理引擎
基于 ultralytics YOLOWorld 框架实现，适配 invoke 框架
"""

import os
import sys
import json
import cv2
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple
import threading
import time

try:
    from utils.utils import get_logger
    logger = get_logger("yolo_world")
except ImportError:
    import logging
    logger = logging.getLogger("yolo_world")
    logger.setLevel(logging.INFO)

try:
    from base.base_model_service import IModelService
except ImportError:
    IModelService = object


class YoloWorldPredictor:
    """
    YOLO-World 开放词汇检测预测器

    提供：
    - 模型加载和预热
    - 单张图像推理
    - 批量图像推理
    - 结果解析（Boxes → Dict）
    """

    def __init__(self,
                 model_path: str,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 max_detections: int = 100,
                 imgsz: int = 640,
                 device: str = None,
                 half: bool = False):
        """
        初始化预测器

        Args:
            model_path: 模型文件路径 (.pt)
            conf_threshold: 置信度阈值
            iou_threshold: NMS IOU 阈值
            max_detections: 每张图最大检测数量
            imgsz: 推理图像尺寸
            device: 计算设备 (None=自动, 'cuda', 'cpu')
            half: 是否使用 FP16
        """
        self.model_path = model_path
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.max_det = max_detections
        self.imgsz = imgsz
        self.half = half

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = None
        self.class_names = []
        self._done_warmup = False

        logger.info(f"[YOLO-World] 初始化预测器")
        logger.info(f"   模型路径: {model_path}")
        logger.info(f"   置信度: {conf_threshold}, IOU: {iou_threshold}")
        logger.info(f"   图像尺寸: {imgsz}, 最大检测: {max_detections}")
        logger.info(f"   设备: {self.device}, FP16: {half}")

    def load_model(self) -> bool:
        """
        加载 YOLO-World 模型

        Returns:
            是否加载成功
        """
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"[YOLO-World] 模型文件不存在: {self.model_path}")
                return False

            from ultralytics import YOLOWorld

            self.model = YOLOWorld(self.model_path)

            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = list(self.model.names.values())
                logger.info(f"[YOLO-World] ✅ 模型加载成功")
                logger.info(f"[YOLO-World] 类别数量: {len(self.class_names)}")
                logger.info(f"[YOLO-World] 类别列表: {self.class_names}")
            else:
                self.class_names = ['object']
                logger.warning(f"[YOLO-World] 无法获取类别，使用默认: {self.class_names}")

            if hasattr(self.model, 'to'):
                self.model.to(self.device)

            return True

        except Exception as e:
            logger.error(f"[YOLO-World] ❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def set_classes(self, classes: List[str]):
        """
        设置检测类别（开放词汇特性）

        Args:
            classes: 类别列表
        """
        try:
            if self.model is not None:
                import ssl
                import urllib.request
                old_context = ssl._create_default_https_context
                ssl._create_default_https_context = ssl._create_unverified_context
                try:
                    self.model.set_classes(classes)
                    self.class_names = classes
                    logger.info(f"[YOLO-World] ✅ 设置检测类别: {classes}")
                finally:
                    ssl._create_default_https_context = old_context
        except Exception as e:
            logger.error(f"[YOLO-World] ❌ 设置类别失败: {e}")

    def warmup(self, img_size: Tuple[int, int] = (640, 640)):
        """
        模型预热

        Args:
            img_size: 预热图像尺寸
        """
        if self._done_warmup:
            return

        try:
            logger.info(f"[YOLO-World] 预热模型...")

            warmup_img = torch.zeros((1, 3, *img_size), dtype=torch.float32).to(self.device)

            if self.half and self.device == 'cuda':
                warmup_img = warmup_img.half()

            with torch.no_grad():
                _ = self.model(warmup_img, verbose=False)

            self._done_warmup = True
            logger.info(f"[YOLO-World] ✅ 预热完成")

        except Exception as e:
            logger.warning(f"[YOLO-World] 预热失败（不影响使用）: {e}")

    def predict(self,
                image: np.ndarray,
                text_prompt: str = None,
                save: bool = False,
                verbose: bool = False) -> List[Dict[str, Any]]:
        """
        单张图像推理

        Args:
            image: 输入图像 (BGR, HWC)
            text_prompt: 文本提示（可选）
            save: 是否保存结果
            verbose: 是否打印详细信息

        Returns:
            检测结果列表
        """
        if self.model is None:
            logger.error("[YOLO-World] 模型未加载")
            return []

        try:
            # 测量真正的 GPU 推理时间（排除类别设置）
            if text_prompt:
                self.set_classes([text_prompt])

            start_time = time.time() * 1000

            results = self.model.predict(
                image,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                imgsz=self.imgsz,
                save=save,
                verbose=verbose,
                half=self.half
            )

            inference_time = int(time.time() * 1000 - start_time)
            logger.info(f"[YOLO-World] GPU 推理时间: {inference_time}ms")

            detections = self._parse_results(results[0] if results else None)

            return detections

        except Exception as e:
            logger.error(f"[YOLO-World] 推理失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def predict_batch(self,
                      images: List[np.ndarray],
                      text_prompt: str = None,
                      save: bool = False,
                      verbose: bool = False) -> List[List[Dict[str, Any]]]:
        """
        批量推理

        Args:
            images: 图像列表 (BGR, HWC)
            text_prompt: 文本提示（可选）
            save: 是否保存结果
            verbose: 是否打印详细信息

        Returns:
            检测结果列表（每张图像的检测结果）
        """
        if self.model is None:
            logger.error("[YOLO-World] 模型未加载")
            return [[] for _ in images]

        try:
            if text_prompt:
                self.set_classes([text_prompt])

            results = self.model.predict(
                images,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                imgsz=self.imgsz,
                save=save,
                verbose=verbose,
                half=self.half
            )

            all_detections = []
            for result in results:
                detections = self._parse_results(result)
                all_detections.append(detections)

            logger.info(f"[YOLO-World] 批量推理完成: {len(images)} 张图, "
                       f"总计 {sum(len(d) for d in all_detections)} 个检测")

            return all_detections

        except Exception as e:
            logger.error(f"[YOLO-World] 批量推理失败: {e}")
            import traceback
            traceback.print_exc()
            return [[] for _ in images]

    def _parse_results(self, result) -> List[Dict[str, Any]]:
        """
        解析 ultralytics Results 对象

        Args:
            result: ultralytics.engine.results.Results 对象

        Returns:
            检测结果列表
        """
        detections = []

        if result is None or result.boxes is None:
            return detections

        try:
            boxes = result.boxes

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(float, xyxy)

                conf = float(boxes.conf[i].cpu().numpy())

                cls_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"

                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': class_name
                }

                detections.append(detection)

        except Exception as e:
            logger.warning(f"[YOLO-World] 解析结果失败: {e}")

        return detections


class YoloWorldService(IModelService):
    """
    YOLO-World 开放词汇检测服务（单例模式）

    负责：
    - 模型生命周期管理
    - 配置管理（从 YAML 加载）
    - 推理接口封装
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, model_id=None, model_dir=None, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_id=None, model_dir=None, **kwargs):
        if self._initialized:
            return

        self.predictor: Optional[YoloWorldPredictor] = None
        self.model_id = None

        self._config = None
        self._config_path = None

        self._initialized = False
        logger.info("[YOLO-World] 服务实例创建")

    def _load_config_from_yaml(self) -> Dict[str, Any]:
        """
        加载 yolo_world.yaml 配置文件

        Returns:
            配置字典
        """
        if self._config is not None:
            return self._config

        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, 'yolo_world.yaml')

            if not os.path.exists(config_path):
                logger.warning(f"[YOLO-World] 配置文件不存在: {config_path}")
                return {}

            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)

            self._config_path = config_path
            logger.info(f"[YOLO-World] ✅ 配置加载成功: {config_path}")

            return self._config

        except Exception as e:
            logger.warning(f"[YOLO-World] 加载配置失败: {e}")
            return {}

    def load_model(self, model_id: str, model_dir: str) -> bool:
        """
        加载模型

        Args:
            model_id: 模型 ID（如 'yolo_world'）
            model_dir: 模型目录路径

        Returns:
            是否加载成功
        """
        if self._initialized:
            logger.info(f"[YOLO-World] 模型已加载: {self.model_id}")
            return True

        try:
            config = self._load_config_from_yaml()
            detection_cfg = config.get('detection', {})
            block_cfg = config.get('block', {})

            model_filename = block_cfg.get('model_filename', 'yolov8s-world.pt')
            model_path = os.path.join(model_dir, model_filename)

            if not os.path.exists(model_path):
                logger.warning(f"[YOLO-World] 配置的模型文件不存在: {model_path}")
                alternative_names = [
                    f"{model_id}.pt",
                    'yolov8s-world.pt', 'yolov8m-world.pt', 'yolov8l-world.pt', 'yolov8x-world.pt',
                    'yolov8s-worldv2.pt', 'yolov8m-worldv2.pt', 'yolov8l-worldv2.pt', 'yolov8x-worldv2.pt',
                    'yolov8-world.pt'
                ]

                for name in alternative_names:
                    test_path = os.path.join(model_dir, name)
                    if os.path.exists(test_path):
                        model_path = test_path
                        logger.info(f"[YOLO-World] 找到模型文件: {name}")
                        break
                else:
                    logger.error(f"[YOLO-World] 在 {model_dir} 中找不到模型文件")
                    return False

            self.predictor = YoloWorldPredictor(
                model_path=model_path,
                conf_threshold=detection_cfg.get('conf_threshold', 0.25),
                iou_threshold=detection_cfg.get('iou_threshold', 0.45),
                max_detections=detection_cfg.get('max_detections', 100),
                imgsz=detection_cfg.get('imgsz', 640),
                device='cuda' if torch.cuda.is_available() else 'cpu',
                half=config.get('optimization', {}).get('use_fp16', False)
            )

            success = self.predictor.load_model()
            if not success:
                return False

            self.predictor.warmup(img_size=(detection_cfg.get('imgsz', 640), detection_cfg.get('imgsz', 640)))

            self.model_id = model_id
            self._initialized = True

            logger.info(f"[YOLO-World] ✅ 服务初始化完成 (model_id={model_id})")
            return True

        except Exception as e:
            logger.error(f"[YOLO-World] ❌ 服务初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def detect(self, prep_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行检测

        Args:
            prep_result: prepareInput 的返回结果

        Returns:
            检测结果
        """
        if self.predictor is None:
            logger.error("[YOLO-World] 预测器未初始化")
            return {"detections": []}

        try:
            image = self._get_image_from_prep_result(prep_result)
            if image is None:
                logger.error("[YOLO-World] 无法获取图像")
                return {"detections": []}

            text_prompt = prep_result.get('text_prompt', 'object')
            detections = self.predictor.predict(image, text_prompt=text_prompt, save=False, verbose=False)

            logger.info(f"[YOLO-World] 推理完成: 检测到 {len(detections)} 个目标")

            if len(detections) > 0:
                logger.info(f"[YOLO-World] 🔍 详细检测结果:")
                for i, det in enumerate(detections):
                    bbox = det.get('bbox', [])
                    conf = det.get('confidence', 0)
                    cls_name = det.get('class_name', 'unknown')
                    logger.info(f"  [{i+1}] {cls_name}: 置信度={conf:.3f}, bbox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
            else:
                logger.info(f"[YOLO-World] ✅ 未检测到目标")

            return {"detections": detections}

        except Exception as e:
            logger.error(f"[YOLO-World] 检测失败: {e}")
            import traceback
            traceback.print_exc()
            return {"detections": []}

    def inference(self, param: Dict[str, Any], cur_result_set: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        框架推理接口（适配 invoke 框架）

        Args:
            param: 输入参数
            cur_result_set: 当前结果集（可选）

        Returns:
            推理结果
        """
        if cur_result_set is None:
            cur_result_set = {}

        if self.predictor is None or not self._initialized:
            logger.info("[YOLO-World] 模型未加载，尝试自动加载...")
            model_dir = param.get('model_dir', os.path.dirname(os.path.abspath(__file__)))
            model_id = param.get('model_id', 'yolo_world')

            success = self.load_model(model_id, model_dir)
            if not success:
                logger.error("[YOLO-World] 自动加载模型失败")
                return {"detections": [], "error": "model_load_failed"}

        try:
            prep_result = self._prepare_input_for_inference(param)

            if not prep_result.get('success', False):
                logger.error(f"[YOLO-World] 准备输入失败: {prep_result.get('error', 'unknown')}")
                return {"detections": [], "error": prep_result.get('error')}

            detect_result = self.detect(prep_result)

            result = {
                **detect_result,
                "detection_count": len(detect_result.get("detections", []))
            }

            logger.info("=" * 60)
            logger.info(f"[YOLO-World] 🎯 推理结果汇总:")
            logger.info(f"  - detection_count: {result.get('detection_count', 0)}")
            if 'error' in result:
                logger.error(f"  - error: {result.get('error')}")
            logger.info("=" * 60)

            return result

        except Exception as e:
            logger.error(f"[YOLO-World] inference 失败: {e}")
            import traceback
            traceback.print_exc()
            return {"detections": [], "error": str(e)}

    def _prepare_input_for_inference(self, param: Dict[str, Any]) -> Dict[str, Any]:
        """
        为 inference 准备输入数据

        Args:
            param: 输入参数

        Returns:
            准备结果
        """
        try:
            if 'image' not in param:
                logger.warning(f"[YOLO-World] param 中没有 image 字段，尝试从其他字段推断")

                data_id = param.get('dataId')
                if data_id:
                    logger.info(f"[YOLO-World] 使用 dataId 作为 image key: {data_id}")
                    param = {**param, 'image': data_id}
                else:
                    logger.error(f"[YOLO-World] 无法推断 image key，param keys: {list(param.keys())}")
                    return {'success': False, 'error': 'missing_image_param'}

            from model_config.yolo_world.config_yolo_world import prepareInput
            return prepareInput(param)
        except Exception as e:
            logger.error(f"[YOLO-World] 准备输入失败: {e}")
            return {'success': False, 'error': str(e)}

    def _get_image_from_prep_result(self, prep_result: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        从 prepareInput 结果中获取图像

        Args:
            prep_result: prepareInput 的返回结果

        Returns:
            图像数组 (BGR, HWC)
        """
        try:
            if prep_result.get('use_redis'):
                from utils.utils import connect_remote_redis
                redis_client = connect_remote_redis()

                image_key = prep_result.get('redis_key')
                if not image_key:
                    return None

                img_bytes = redis_client.get(image_key)
                if not img_bytes:
                    return None

                shape_bytes = redis_client.get(f"{image_key}:shape")
                if shape_bytes:
                    shape = tuple(json.loads(shape_bytes))
                    image = np.frombuffer(img_bytes, dtype=np.uint8).reshape(shape)
                else:
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                return image
            else:
                return prep_result.get('image')

        except Exception as e:
            logger.error(f"[YOLO-World] 获取图像失败: {e}")
            return None


def get_service() -> YoloWorldService:
    """
    获取 YoloWorldService 单例实例

    Returns:
        YoloWorldService 实例
    """
    return YoloWorldService()
