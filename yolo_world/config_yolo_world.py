"""
YOLO-World 开放词汇检测模型配置文件
提供 prepareInput, postAction 等钩子函数
供 invoke 框架调度时调用

V1.0 - YOLO-World 开放词汇检测
"""

import sys
import os
import json
import time
import cv2
import numpy as np
import yaml
from typing import Dict, Any, Optional, List

try:
    from utils.utils import get_logger
    logger = get_logger("yoloseries-yolo_world")
except ImportError:
    import logging
    logger = logging.getLogger("yoloseries-yolo_world")
    logger.setLevel(logging.INFO)

# 组件名称
component = "yoloseries-yolo_world"

# 全局模块缓存
_model_inference_module = None


def _get_model_inference_module():
    """获取 model_inference 模块（延迟加载）"""
    global _model_inference_module

    if _model_inference_module is None:
        import importlib.util

        module_name = 'yoloseries_yolo_world_model_inference'

        if module_name not in sys.modules:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_inference_path = os.path.join(current_dir, 'model_inference.py')

            spec = importlib.util.spec_from_file_location(module_name, model_inference_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            _model_inference_module = module
        else:
            _model_inference_module = sys.modules[module_name]

    return _model_inference_module


def get_execution_model(input_event: Dict[str, Any]) -> str:
    """确定当前应该使用哪个模型实例"""
    return "yoloseries-yolo_world"


_config_cache = None


def _load_config() -> Dict[str, Any]:
    """加载 yolo_world.yaml 配置"""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'yolo_world.yaml')

    default_config = {
        'detection': {
            'conf_threshold': 0.25,
            'iou_threshold': 0.45,
            'max_detections': 100,
            'imgsz': 640
        }
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config:
                    def merge_dict(base, update):
                        for key, value in update.items():
                            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                                merge_dict(base[key], value)
                            else:
                                base[key] = value
                    merge_dict(default_config, loaded_config)
        except Exception as e:
            logger.warning(f'[YOLO-World] 加载配置失败: {e}')

    _config_cache = default_config
    return _config_cache


def prepareInput(param: Dict[str, Any] = None,
                 cur_result_set: Dict[str, Any] = None,
                 invoke_callback=None) -> Dict[str, Any]:
    """
    推理前数据预处理

    Args:
        param: 输入参数
            - image: Redis key 或 base64/文件路径/numpy数组
            - use_redis_key: 是否使用 Redis key 模式
            - text_prompt: 文本提示（用于 YOLO-World）
        cur_result_set: 当前结果集（可选）
        invoke_callback: 回调函数（可选）

    Returns:
        处理后的输入数据
    """
    start_time = time.time() * 1000
    if param is None:
        param = {}
    if cur_result_set is None:
        cur_result_set = {}

    logger.info('=' * 50)
    logger.info('[YOLO-World] prepareInput 开始')
    logger.info(f'[YOLO-World] 📥 param keys: {list(param.keys())}')

    image_key = param.get('image')
    use_redis_key = param.get('use_redis_key', False)
    text_prompt = param.get('text_query') or param.get('text_prompt', 'apple')

    if not image_key:
        logger.error('[YOLO-World] ❌ 缺少 image 参数')
        return {'success': False, 'error': 'missing_image_param'}

    # Redis 模式（推荐）
    if use_redis_key:
        try:
            from utils.utils import connect_remote_redis
            redis_client = connect_remote_redis()

            if not redis_client.exists(image_key):
                logger.error(f'[YOLO-World] ❌ Redis 中找不到图像: {image_key}')
                return {'success': False, 'error': 'redis_key_not_found'}

            shape_bytes = redis_client.get(f"{image_key}:shape")
            image_shape = None
            if shape_bytes:
                image_shape = tuple(json.loads(shape_bytes))
                logger.info(f'[YOLO-World] ✅ 图像key验证通过, shape: {image_shape}')
            else:
                logger.info(f'[YOLO-World] ✅ 图像key验证通过')

            total_time = int(time.time() * 1000 - start_time)
            result = {
                'success': True,
                'redis_key': image_key,
                'use_redis': True,
                'image_shape': list(image_shape) if image_shape else None,
                'text_prompt': text_prompt,
                'frame_id': param.get('frame_id'),
                'frame_timestamp': param.get('frame_timestamp'),
                'depth_map': param.get('depth_map'),
                'right_image': param.get('right_image'),
                'total_time': total_time
            }

            logger.info(f'[YOLO-World] ✅ prepareInput 完成')
            logger.info(f'[YOLO-World] 前处理时间: {total_time}ms')
            logger.info('=' * 50)
            return result

        except Exception as e:
            logger.error(f'[YOLO-World] ❌ prepareInput 错误: {e}')
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    # Base64 模式
    elif isinstance(image_key, str) and image_key.startswith('data:image'):
        try:
            import base64
            _, encoded = image_key.split(',', 1)
            np_arr = np.frombuffer(base64.b64decode(encoded), np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if image is None:
                logger.error('[YOLO-World] ❌ Base64 图像解码失败')
                return {'success': False, 'error': 'base64_decode_failed'}

            logger.info(f'[YOLO-World] ✅ Base64 图像解码成功, shape: {image.shape}')

            total_time = int(time.time() * 1000 - start_time)
            result = {
                'success': True,
                'image': image,
                'use_redis': False,
                'image_shape': list(image.shape),
                'text_prompt': text_prompt,
                'total_time': total_time
            }

            logger.info(f'[YOLO-World] ✅ prepareInput 完成')
            logger.info(f'[YOLO-World] 前处理时间: {total_time}ms')
            logger.info('=' * 50)
            return result

        except Exception as e:
            logger.error(f'[YOLO-World] ❌ Base64 解码错误: {e}')
            return {'success': False, 'error': str(e)}

    # 文件路径模式
    elif isinstance(image_key, str) and os.path.exists(image_key):
        try:
            image = cv2.imread(image_key)
            if image is None:
                logger.error(f'[YOLO-World] ❌ 无法读取图像文件: {image_key}')
                return {'success': False, 'error': 'file_read_failed'}

            logger.info(f'[YOLO-World] ✅ 从文件读取图像成功, shape: {image.shape}')

            total_time = int(time.time() * 1000 - start_time)
            result = {
                'success': True,
                'image': image,
                'use_redis': False,
                'image_shape': list(image.shape),
                'text_prompt': text_prompt,
                'total_time': total_time
            }

            logger.info(f'[YOLO-World] ✅ prepareInput 完成')
            logger.info(f'[YOLO-World] 前处理时间: {total_time}ms')
            logger.info('=' * 50)
            return result

        except Exception as e:
            logger.error(f'[YOLO-World] ❌ 文件读取错误: {e}')
            return {'success': False, 'error': str(e)}

    # Numpy 数组模式
    elif isinstance(image_key, np.ndarray):
        logger.info(f'[YOLO-World] ✅ 接收到 numpy 数组, shape: {image_key.shape}')
        total_time = int(time.time() * 1000 - start_time)
        result = {
            'success': True,
            'image': image_key,
            'use_redis': False,
            'image_shape': list(image_key.shape),
            'text_prompt': text_prompt,
            'total_time': total_time
        }
        logger.info(f'[YOLO-World] ✅ prepareInput 完成')
        logger.info(f'[YOLO-World] 前处理时间: {total_time}ms')
        logger.info('=' * 50)
        return result

    else:
        logger.error(f'[YOLO-World] ❌ 不支持的输入类型: {type(image_key)}')
        return {'success': False, 'error': 'unsupported_input_type'}


def postAction(param: Dict[str, Any] = None,
               cur_result_set: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    推理后后处理

    Args:
        param: 输入参数
        cur_result_set: 推理结果
            - inference_result: 推理输出

    Returns:
        后处理结果
    """
    import time

    if param is None:
        param = {}
    if cur_result_set is None:
        cur_result_set = {}

    start_time = time.time() * 1000

    logger.info('=' * 50)
    logger.info('[YOLO-World] postAction 开始')
    logger.info(f'[YOLO-World] 📥 param keys: {list(param.keys())}')
    logger.info(f'[YOLO-World] 📥 cur_result_set keys: {list(cur_result_set.keys())}')

    inference_result = cur_result_set.get('inference_result', {})

    logger.info(f'[YOLO-World] 推理结果: {inference_result}')

    detections = inference_result.get('detections', [])
    bboxes = [det.get('bbox', []) for det in detections]

    total_time = int(time.time() * 1000 - start_time)

    result = {
        'success': True,
        'detections': detections,
        'bboxes': bboxes,
        'num_detections': len(detections),
        'total_time': total_time,
        # 传递原始参数给下游节点
        'image': param.get('image'),
        'use_redis_key': param.get('use_redis_key', False),
        'right_image': param.get('right_image', ''),
        'text_query': param.get('text_query', ''),
        'text_prompt': param.get('text_prompt', ''),
        'desired_feature': param.get('desired_feature', []),
        'init_mode': param.get('init_mode', 'text'),
        'frame_id': param.get('frame_id'),
        'frame_timestamp': param.get('frame_timestamp'),
        'dataId': param.get('dataId'),
        'robot_id': param.get('robot_id'),
    }

    # 保存可视化结果
    _save_visualization(param, detections)

    logger.info(f'[YOLO-World] ✅ postAction 完成')
    logger.info(f'[YOLO-World] 总耗时: {total_time}ms')
    logger.info('=' * 50)

    return result


def _save_visualization(param: Dict[str, Any], detections: List[Dict[str, Any]]):
    """
    保存检测结果的可视化图像

    Args:
        param: 输入参数
        detections: 检测结果列表
    """
    try:
        config = _load_config()
        debug_cfg = config.get('debug', {})
        save_vis = debug_cfg.get('save_visualization', False)

        if not save_vis:
            return

        logger.info(f'[YOLO-World] 开始保存可视化结果...')

        # 获取图像
        image = _get_image_for_visualization(param)
        if image is None:
            logger.warning(f'[YOLO-World] 无法获取图像，跳过可视化保存')
            return

        # 创建可视化图像
        vis_image = image.copy()

        # 绘制检测框
        for i, det in enumerate(detections):
            bbox = det.get('bbox', [])
            confidence = det.get('confidence', 0)
            class_name = det.get('class_name', 'unknown')

            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])

                # 绘制边界框（绿色）
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 绘制标签
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(vis_image, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                logger.info(f'[YOLO-World] 绘制检测框: {class_name}, conf={confidence:.2f}, bbox={bbox}')

        # 创建 debug 目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        debug_dir = os.path.join(current_dir, 'debug')
        os.makedirs(debug_dir, exist_ok=True)

        # 生成文件名
        frame_id = param.get('frame_id', 0)
        timestamp = int(time.time() * 1000)
        filename = f"yolo_world_det_{frame_id}_{timestamp}.jpg"
        save_path = os.path.join(debug_dir, filename)

        # 保存图像
        cv2.imwrite(save_path, vis_image)
        logger.info(f'[YOLO-World] ✅ 可视化结果已保存: {save_path}')

    except Exception as e:
        logger.error(f'[YOLO-World] 保存可视化失败: {e}')
        import traceback
        traceback.print_exc()


def _get_image_for_visualization(param: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    从参数中获取图像用于可视化

    Args:
        param: 输入参数

    Returns:
        图像数组 (BGR, HWC)
    """
    try:
        image_key = param.get('image')
        use_redis_key = param.get('use_redis_key', False)

        # Redis 模式
        if use_redis_key and image_key:
            from utils.utils import connect_remote_redis
            redis_client = connect_remote_redis()

            if not redis_client.exists(image_key):
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

        # Numpy 数组模式
        elif isinstance(image_key, np.ndarray):
            return image_key

        # 文件路径模式
        elif isinstance(image_key, str) and os.path.exists(image_key):
            return cv2.imread(image_key)

        return None

    except Exception as e:
        logger.error(f'[YOLO-World] 获取图像失败: {e}')
        return None


def prepareChildInput(param: Dict[str, Any] = None,
                      cur_result_set: Dict[str, Any] = None,
                      callback=None) -> Dict[str, Any]:
    """
    为子节点准备输入数据（传递检测框给 SAM 和 GraspNet）

    Args:
        param: 输入参数
        cur_result_set: 当前结果集（包含 postAction_result）
        callback: 回调函数（可选）

    Returns:
        子节点输入数据
    """
    if param is None:
        param = {}
    if cur_result_set is None:
        cur_result_set = {}

    logger.info('=' * 50)
    logger.info('[YOLO-World] prepareChildInput 开始处理')

    post_result = cur_result_set.get('postAction_result', {})

    detections = post_result.get('detections', [])
    bboxes = [det.get('bbox', []) for det in detections]

    image = param.get('image')
    use_redis_key = param.get('use_redis_key', False)

    # 传递所有必要参数给下游节点（SAM、GraspNet）
    child_input = {
        'detections': detections,
        'bboxes': bboxes,
        'num_detections': len(detections),
        'image': image,
        'use_redis_key': use_redis_key,
        # 传递查询参数（用于目标选择）
        'text_query': param.get('text_query', ''),
        'text_prompt': param.get('text_prompt', ''),
        'desired_feature': param.get('desired_feature', []),
        'init_mode': param.get('init_mode', 'text'),
        'frame_id': param.get('frame_id'),
        'frame_timestamp': param.get('frame_timestamp'),
        # 传递右图（用于深度图计算）
        'right_image': param.get('right_image', ''),
        # 传递其他可能的参数
        'dataId': param.get('dataId'),
        'robot_id': param.get('robot_id'),
    }

    logger.info(f'[YOLO-World] prepareChildInput 完成, detections={len(detections)}')
    logger.info(f'[YOLO-World]   - image: {image}')
    logger.info(f'[YOLO-World]   - right_image: {child_input.get("right_image", "None")}')
    logger.info(f'[YOLO-World]   - text_query: {child_input.get("text_query", "")}')
    logger.info('=' * 50)

    return child_input
