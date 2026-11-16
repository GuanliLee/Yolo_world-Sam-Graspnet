"""
SAM 分割推理引擎
基于 ultralytics SAM 框架实现，适配 invoke 框架
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

# 添加 base 路径
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

try:
    from utils.utils import get_logger
    logger = get_logger("sam_segment")
except ImportError:
    import logging
    logger = logging.getLogger("sam_segment")
    logger.setLevel(logging.INFO)

try:
    from base.base_model_service import IModelService
except ImportError:
    IModelService = object





class SAMSegmentService(IModelService):
    """
    SAM 分割服务（单例模式）

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

        self.sam = None
        self.predictor = None
        self.device = None
        self.cfg = {}
        self.model_id = None
        self._inference_lock = threading.Lock()  # 🔒 添加锁以防止并发调用干扰 predictor 状态

        self._config = None
        self._config_path = None

        self._initialized = False
        logger.info("[SAM] 服务实例创建")

    def _load_config_from_yaml(self) -> Dict[str, Any]:
        """
        加载 sam_segment.yaml 配置文件

        Returns:
            配置字典
        """
        if self._config is not None:
            return self._config

        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, 'sam_segment.yaml')

            if not os.path.exists(config_path):
                logger.warning(f"[SAM] 配置文件不存在: {config_path}")
                return {}

            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)

            self._config_path = config_path
            logger.info(f"[SAM] ✅ 配置加载成功: {config_path}")

            return self._config

        except Exception as e:
            logger.warning(f"[SAM] 加载配置失败: {e}")
            return {}

    def load_model(self, model_id: str, model_dir: str) -> bool:
        """
        加载模型

        Args:
            model_id: 模型 ID（如 'sam_segment'）
            model_dir: 模型目录路径

        Returns:
            是否加载成功
        """
        if self._initialized:
            logger.info(f"[SAM] 模型已加载: {self.model_id}")
            return True

        try:
            config = self._load_config_from_yaml()
            block_cfg = config.get('block', {})
            segment_config = config.get('segmentation', {})

            self.cfg = {
                'model_type': block_cfg.get('model_type', 'vit_b'),
                'sort_by': segment_config.get('sort_by', 'area'),
                'num_of_clicks_for_detection': segment_config.get('num_of_clicks_for_detection', 3),
                'min_area_size': segment_config.get('min_area_size', 200),
                'desired_height': segment_config.get('desired_height', 240),
                'desired_width': segment_config.get('desired_width', 320),
            }

            # 🚀 动态GPU选择：根据Redis限流状态选择空闲GPU
            hardware = block_cfg.get('hardware', 'gpu').lower()
            preferred_gpu_id = block_cfg.get('gpu_id', None)

            if hardware == 'gpu' and torch.cuda.is_available():
                try:
                    from utils.gpu_selector import select_gpu_for_model
                    from utils import utils

                    # 动态选择空闲GPU
                    self.device = select_gpu_for_model(
                        redis_client=utils.connect_remote_redis(),
                        config_gpu_id=preferred_gpu_id
                    )
                    logger.info(f"  - SAM动态GPU选择: {self.device} (配置gpu_id={preferred_gpu_id})")
                except Exception as e:
                    logger.warning(f"⚠️ GPU选择失败: {e}，使用默认cuda")
                    self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')

            # 处理模型文件路径
            model_filename = block_cfg.get('model_filename', 'sam_b.pt')
            self.cfg['checkpoint_path'] = os.path.join(model_dir, model_filename)

            if not os.path.exists(self.cfg['checkpoint_path']):
                logger.warning(f"[SAM] 配置的模型文件不存在: {self.cfg['checkpoint_path']}")
                alternative_names = [
                    f"{model_id}.pth",
                    'sam_vit_b_01ec64.pth', 'sam_vit_l_0b3195.pth', 'sam_vit_h_4b8939.pth',
                    'mobile_sam.pt',
                    'sam_b.pt'
                ]

                for name in alternative_names:
                    test_path = os.path.join(model_dir, name)
                    if os.path.exists(test_path):
                        self.cfg['checkpoint_path'] = test_path
                        logger.info(f"[SAM] 找到模型文件: {name}")
                        break
                else:
                    logger.error(f"[SAM] 在 {model_dir} 中找不到模型文件")
                    return False

            # 加载 SAM 模型
            try:
                # 首先尝试从本地 sam 目录导入
                sam_path = os.path.join(model_dir, 'sam')
                if os.path.exists(sam_path) and sam_path not in sys.path:
                    sys.path.insert(0, sam_path)
                from segment_anything import sam_model_registry, SamPredictor

                self.sam = sam_model_registry[self.cfg['model_type']](checkpoint=self.cfg['checkpoint_path'])
                self.sam.to(device=self.device)
                self.predictor = SamPredictor(self.sam)

                logger.info(f"✓ SAM 模型加载成功: {self.cfg['model_type']}")
            except ImportError as e:
                logger.error(f"[SAM] 导入 segment_anything 失败: {e}")
                logger.error(f"[SAM] 检查 sam 目录是否存在: {os.path.exists(sam_path)}")
                logger.error(f"[SAM] sam 目录路径: {sam_path}")
                return False
            except Exception as e:
                logger.error(f"[SAM] 加载 SAM 模型失败: {e}")
                import traceback
                traceback.print_exc()
                return False

            self.model_id = model_id
            self._initialized = True

            logger.info(f"[SAM] ✅ 服务初始化完成 (model_id={model_id})")
            return True

        except Exception as e:
            logger.error(f"[SAM] ❌ 服务初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def segment(self, prep_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行分割

        Args:
            prep_result: prepareInput 的返回结果

        Returns:
            分割结果
        """
        if self.predictor is None:
            logger.error("[SAM] 预测器未初始化")
            return {"masks": []}

        try:
            image = self._get_image_from_prep_result(prep_result)
            if image is None:
                logger.error("[SAM] 无法获取图像")
                return {"masks": []}

            bboxes = prep_result.get('bboxes', [])
            points = prep_result.get('points', [])
            detections = prep_result.get('detections', [])

            # 转换为 RGB（SAM 需要 RGB 格式）
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 设置图像
            logger.info(f"[SAM] 准备调用 set_image, image_rgb.shape={image_rgb.shape}")
            self.predictor.set_image(image_rgb)
            logger.info(f"[SAM] ✅ set_image 完成, features={self.predictor.features is not None}, original_size={self.predictor.original_size}")

            masks = []
            scores = []
            result_bboxes = []

            # 解析 FastJSON 格式的 bbox 数据
            def parse_fastjson_bbox(data):
                """
                解析 FastJSON 格式的 bbox 数据，返回扁平的数值列表
                格式: ['com.alibaba.fastjson.JSONArray', [...]]
                或: ['java.math.BigDecimal', 112.04]
                """
                if isinstance(data, (int, float)):
                    return [data]
                if isinstance(data, str):
                    if data.startswith('com.') or data.startswith('java.'):
                        return []
                    try:
                        return [float(data)]
                    except ValueError:
                        return []
                if isinstance(data, (list, tuple)):
                    if len(data) == 2 and isinstance(data[0], str):
                        if data[0].startswith('com.') or data[0].startswith('java.'):
                            return parse_fastjson_bbox(data[1])
                    result = []
                    for item in data:
                        result.extend(parse_fastjson_bbox(item))
                    return result
                return []

            # 解析 bboxes（处理 FastJSON 格式）
            parsed_bboxes = parse_fastjson_bbox(bboxes)
            logger.info(f"[SAM] 解析后的 bboxes: {parsed_bboxes}")

            # 将扁平列表重新组织为 bbox 列表（每 4 个值一个 bbox）
            if parsed_bboxes and len(parsed_bboxes) >= 4:
                # 如果是扁平的坐标列表，按每 4 个值分组
                if all(isinstance(x, (int, float)) for x in parsed_bboxes):
                    num_bboxes = len(parsed_bboxes) // 4
                    bboxes = [parsed_bboxes[i*4:(i+1)*4] for i in range(num_bboxes)]
                    logger.info(f"[SAM] 重组后的 bboxes: {bboxes}")
                else:
                    bboxes = parsed_bboxes

            # 如果有 bbox，使用 bbox 提示进行分割
            if bboxes and len(bboxes) > 0:
                logger.info(f"[SAM] 使用 {len(bboxes)} 个 bbox 进行分割")

                for idx, bbox in enumerate(bboxes):
                    try:
                        logger.info(f"[SAM] bbox {idx} 原始值: {bbox}, 类型: {type(bbox)}")

                        # 确保 bbox 是列表格式
                        if isinstance(bbox, (int, float)):
                            logger.warning(f"[SAM] bbox {idx} 是单个数值，跳过")
                            continue

                        # 处理 JSON Reference 格式
                        if isinstance(bbox, dict) and '$ref' in bbox:
                            logger.warning(f"[SAM] bbox {idx} 是 JSON Reference 格式，跳过: {bbox}")
                            continue

                        # 转换为浮点数列表
                        if isinstance(bbox, (list, tuple)):
                            flat_bbox = [float(x) for x in bbox if isinstance(x, (int, float))]
                        elif isinstance(bbox, dict):
                            # 尝试从字典中提取坐标
                            if all(key in bbox for key in ['x1', 'y1', 'x2', 'y2']):
                                flat_bbox = [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']]
                            elif all(key in bbox for key in ['xmin', 'ymin', 'xmax', 'ymax']):
                                flat_bbox = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
                            else:
                                logger.warning(f"[SAM] 无效的bbox字典格式: {bbox}")
                                continue
                        else:
                            logger.warning(f"[SAM] bbox {idx} 格式不支持: {type(bbox)}")
                            continue

                        logger.info(f"[SAM] bbox {idx} 处理后: {flat_bbox}")

                        # 确保有 4 个坐标值
                        if len(flat_bbox) < 4:
                            logger.warning(f"[SAM] bbox {idx} 格式错误，元素不足: {flat_bbox}")
                            continue

                        box = np.array(flat_bbox[:4], dtype=np.float32)
                        logger.info(f"[SAM] bbox {idx} 开始执行 SAM 分割, box={box}")

                        # 检查 predictor 状态
                        logger.info(f"[SAM] bbox {idx} predictor 状态: features={self.predictor.features is not None}, "
                                   f"original_size={getattr(self.predictor, 'original_size', None)}")

                        # 执行分割
                        mask_results, score_results, _ = self.predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=box,
                            multimask_output=False
                        )

                        logger.info(f"[SAM] bbox {idx} SAM 分割完成, mask_results={len(mask_results)}")

                        if len(mask_results) > 0:
                            mask = mask_results[0]
                            score = float(score_results[0])

                            # 过滤小区域
                            area = np.sum(mask)
                            logger.info(f"[SAM] bbox {idx} 掩码面积={area}, 阈值={self.cfg['min_area_size']}")
                            if area >= self.cfg['min_area_size']:
                                masks.append(mask.astype(np.uint8))
                                scores.append(score)
                                result_bboxes.append(bbox)
                                logger.info(f"[SAM] bbox {idx} ✅ 掩码已添加")
                            else:
                                logger.warning(f"[SAM] bbox {idx} ❌ 分割区域太小: {area} < {self.cfg['min_area_size']}")
                    except Exception as e:
                        logger.error(f"[SAM] bbox {idx} ❌ 分割失败: {e}")
                        import traceback
                        traceback.print_exc()

            # 如果有点提示，使用点提示进行分割
            elif points and len(points) > 0:
                logger.info(f"[SAM] 使用 {len(points)} 个点进行分割")

                point_coords = np.array([[p['x'], p['y']] for p in points])
                point_labels = np.array([p.get('label', 1) for p in points])

                mask_results, score_results, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True
                )

                # 选择最佳掩码
                best_idx = np.argmax(score_results)
                mask = mask_results[best_idx]
                score = float(score_results[best_idx])

                if np.sum(mask) >= self.cfg['min_area_size']:
                    masks.append(mask.astype(np.uint8))
                    scores.append(score)
                    # 从掩码计算 bbox
                    y_indices, x_indices = np.where(mask)
                    if len(x_indices) > 0:
                        x1, x2 = x_indices.min(), x_indices.max()
                        y1, y2 = y_indices.min(), y_indices.max()
                        result_bboxes.append([int(x1), int(y1), int(x2), int(y2)])

            logger.info(f"[SAM] 分割完成, 生成了 {len(masks)} 个掩码")

            # 🎯 新方案：将二进制掩码保存到 Redis，避免 RLE 编码/解码问题
            # 从 cur_result_set 获取 redis_key 来生成 mask_key
            redis_client = None
            image_key = prep_result.get('redis_key') or prep_result.get('image', '')
            mask_key = f"{image_key}:sam_mask"

            try:
                from utils.utils import connect_remote_redis
                redis_client = connect_remote_redis()

                if masks and len(masks) > 0:
                    # 将掩码转换为 uint8 并保存到 Redis
                    mask_binary = masks[0].astype(np.uint8)
                    redis_client.setex(mask_key, 300, mask_binary.tobytes())
                    redis_client.setex(f"{mask_key}:shape", 300, json.dumps(list(mask_binary.shape)))
                    logger.info(f"[SAM] ✅ 二进制掩码已保存到 Redis: {mask_key}, shape={mask_binary.shape}, 前景像素={mask_binary.sum()}")
                else:
                    logger.warning("[SAM] ⚠️ 没有掩码需要保存")
            except Exception as e:
                logger.warning(f"[SAM] 保存掩码到 Redis 失败: {e}")

            return {
                "masks": mask_key if mask_key else [],
                "scores": scores,
                "bboxes": result_bboxes,
                "num_masks": len(masks),
                "mask_key": mask_key  # 添加 mask_key 方便 GraspNet 使用
            }

        except Exception as e:
            logger.error(f"[SAM] 分割失败: {e}")
            import traceback
            traceback.print_exc()
            return {"masks": []}

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
            logger.info("[SAM] 模型未加载，尝试自动加载...")
            model_dir = param.get('model_dir', os.path.dirname(os.path.abspath(__file__)))
            model_id = param.get('model_id', 'sam_segment')

            success = self.load_model(model_id, model_dir)
            if not success:
                logger.error("[SAM] 自动加载模型失败")
                return {"masks": [], "error": "model_load_failed"}

        # 🔒 使用锁保护整个推理过程，防止并发调用干扰 predictor 的 set_image 状态
        with self._inference_lock:
            try:
                start_time = time.time() * 1000
                prep_result = self._prepare_input_for_inference(param, cur_result_set)

                if not prep_result.get('success', False):
                    logger.error(f"[SAM] 准备输入失败: {prep_result.get('error', 'unknown')}")
                    return {"masks": [], "error": prep_result.get('error')}

                segment_result = self.segment(prep_result)

                total_time = int(time.time() * 1000 - start_time)
                result = {
                    **segment_result,
                    "success": True,
                    "mask_count": len(segment_result.get("masks", [])),
                    "total_time": total_time
                }

                logger.info("=" * 60)
                logger.info(f"[SAM] 🎯 推理结果汇总:")
                logger.info(f"  - mask_count: {result.get('mask_count', 0)}")
                logger.info(f"  - 推理时间: {total_time}ms")
                if 'error' in result:
                    logger.error(f"  - error: {result.get('error')}")
                logger.info("=" * 60)

                return result

            except Exception as e:
                logger.error(f"[SAM] inference 失败: {e}")
                import traceback
                traceback.print_exc()
                return {"masks": [], "error": str(e), "success": False}

    def _prepare_input_for_inference(self, param: Dict[str, Any], cur_result_set: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        为 inference 准备输入数据

        Args:
            param: 输入参数
            cur_result_set: 当前结果集

        Returns:
            准备结果
        """
        try:
            # 确保参数中包含image字段
            if 'image' not in param:
                logger.warning(f"[SAM] param 中没有 image 字段，尝试从其他字段推断")

                # 尝试从redis_key字段推断
                redis_key = param.get('redis_key')
                if redis_key:
                    logger.info(f"[SAM] 使用 redis_key 作为 image key: {redis_key}")
                    param = {**param, 'image': redis_key}
                else:
                    # 尝试从dataId字段推断
                    data_id = param.get('dataId')
                    if data_id:
                        logger.info(f"[SAM] 使用 dataId 作为 image key: {data_id}")
                        param = {**param, 'image': data_id}
                    else:
                        logger.error(f"[SAM] 无法推断 image key，param keys: {list(param.keys())}")
                        return {'success': False, 'error': 'missing_image_param'}

            from model_config.sam_segment.config_sam_segment import prepareInput
            return prepareInput(param, cur_result_set)
        except Exception as e:
            logger.error(f"[SAM] 准备输入失败: {e}")
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
            logger.error(f"[SAM] 获取图像失败: {e}")
            return None

    def _mask_to_rle(self, mask: np.ndarray) -> Dict:
        """
        将二值掩码转换为 RLE 编码

        Args:
            mask: 二值掩码数组

        Returns:
            RLE 编码的掩码字典
        """
        pixels = mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return {
            'counts': runs.tolist(),
            'size': list(mask.shape)
        }


def get_service() -> SAMSegmentService:
    """
    获取 SAMSegmentService 单例实例

    Returns:
        SAMSegmentService 实例
    """
    return SAMSegmentService()
