"""
SAM 分割模型配置文件
提供 prepareInput, postAction 等钩子函数
供 invoke 框架调度时调用

V1.0 - SAM 分割
"""

import sys
import os
import json
import cv2
import numpy as np
import yaml
from typing import Dict, Any, Optional, List

try:
    from utils.utils import get_logger
    logger = get_logger("yoloseries-sam_segment")
except ImportError:
    import logging
    logger = logging.getLogger("yoloseries-sam_segment")
    logger.setLevel(logging.INFO)

# 组件名称
component = "yoloseries_sam_segment"

# 全局模块缓存
_model_inference_module = None


def _get_model_inference_module():
    """获取 model_inference 模块（延迟加载）"""
    global _model_inference_module

    if _model_inference_module is None:
        import importlib.util

        module_name = 'yoloseries-sam_segment_model_inference'

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
    return "yoloseries-sam_segment"


_config_cache = None


def _load_config() -> Dict[str, Any]:
    """加载 sam_segment.yaml 配置"""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'sam_segment.yaml')

    default_config = {
        'segmentation': {
            'sort_by': 'area',
            'num_of_clicks_for_detection': 3,
            'min_area_size': 200,
            'desired_height': 240,
            'desired_width': 320
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
            logger.warning(f'[SAM] 加载配置失败: {e}')

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
            - bboxes: 检测框列表（来自 YOLO-World）
        cur_result_set: 当前结果集（可选）
        invoke_callback: 回调函数（可选）

    Returns:
        处理后的输入数据
    """
    if param is None:
        param = {}
    if cur_result_set is None:
        cur_result_set = {}

    logger.info('=' * 50)
    logger.info('[SAM] prepareInput 开始')
    logger.info(f'[SAM] 📥 param keys: {list(param.keys())}')

    image_key = param.get('image')
    use_redis_key = param.get('use_redis_key', False)

    # 首先尝试从cur_result_set中获取bboxes
    bboxes = []
    if cur_result_set:
        prepare_result = cur_result_set.get('prepareInput_result', {})
        if prepare_result:
            bboxes = prepare_result.get('bboxes', [])
            logger.info(f'[SAM] 从 cur_result_set 获取 bboxes: {bboxes}')

    # 如果cur_result_set中没有，再从param中获取
    if not bboxes:
        bboxes = param.get('bboxes', [])
        logger.info(f'[SAM] 从 param 获取 bboxes: {bboxes}')

    if not image_key:
        logger.error('[SAM] ❌ 缺少 image 参数')
        return {'success': False, 'error': 'missing_image_param'}

    # Redis 模式（推荐）
    if use_redis_key:
        try:
            from utils.utils import connect_remote_redis
            redis_client = connect_remote_redis()

            if not redis_client.exists(image_key):
                logger.error(f'[SAM] ❌ Redis 中找不到图像: {image_key}')
                return {'success': False, 'error': 'redis_key_not_found'}

            shape_bytes = redis_client.get(f"{image_key}:shape")
            image_shape = None
            if shape_bytes:
                image_shape = tuple(json.loads(shape_bytes))
                logger.info(f'[SAM] ✅ 图像key验证通过, shape: {image_shape}')
            else:
                logger.info(f'[SAM] ✅ 图像key验证通过')

            result = {
                'success': True,
                'redis_key': image_key,
                'use_redis': True,
                'image_shape': list(image_shape) if image_shape else None,
                'bboxes': bboxes,
                'frame_id': param.get('frame_id'),
                'frame_timestamp': param.get('frame_timestamp'),
                'depth_map': param.get('depth_map'),
                'right_image': param.get('right_image')
            }

            logger.info(f'[SAM] ✅ prepareInput 完成')
            logger.info(f'[SAM] 检测框数量: {len(bboxes)}')
            logger.info('=' * 50)
            return result

        except Exception as e:
            logger.error(f'[SAM] ❌ prepareInput 错误: {e}')
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
                logger.error('[SAM] ❌ Base64 图像解码失败')
                return {'success': False, 'error': 'base64_decode_failed'}

            logger.info(f'[SAM] ✅ Base64 图像解码成功, shape: {image.shape}')

            result = {
                'success': True,
                'image': image,
                'use_redis': False,
                'image_shape': list(image.shape),
                'bboxes': bboxes
            }

            logger.info(f'[SAM] ✅ prepareInput 完成')
            logger.info('=' * 50)
            return result

        except Exception as e:
            logger.error(f'[SAM] ❌ Base64 解码错误: {e}')
            return {'success': False, 'error': str(e)}

    # 文件路径模式
    elif isinstance(image_key, str) and os.path.exists(image_key):
        try:
            image = cv2.imread(image_key)
            if image is None:
                logger.error(f'[SAM] ❌ 无法读取图像文件: {image_key}')
                return {'success': False, 'error': 'file_read_failed'}

            logger.info(f'[SAM] ✅ 从文件读取图像成功, shape: {image.shape}')

            result = {
                'success': True,
                'image': image,
                'use_redis': False,
                'image_shape': list(image.shape),
                'bboxes': bboxes
            }

            logger.info(f'[SAM] ✅ prepareInput 完成')
            logger.info('=' * 50)
            return result

        except Exception as e:
            logger.error(f'[SAM] ❌ 文件读取错误: {e}')
            return {'success': False, 'error': str(e)}

    # Numpy 数组模式
    elif isinstance(image_key, np.ndarray):
        logger.info(f'[SAM] ✅ 接收到 numpy 数组, shape: {image_key.shape}')
        result = {
            'success': True,
            'image': image_key,
            'use_redis': False,
            'image_shape': list(image_key.shape),
            'bboxes': bboxes
        }
        logger.info(f'[SAM] ✅ prepareInput 完成')
        logger.info('=' * 50)
        return result

    else:
        logger.error(f'[SAM] ❌ 不支持的输入类型: {type(image_key)}')
        return {'success': False, 'error': 'unsupported_input_type'}


def _rle_to_mask(rle: Dict) -> np.ndarray:
    """
    将 SAM 模型的 RLE 编码转换为二值掩码

    Args:
        rle: RLE 编码字典，包含 'counts' 和 'size' 字段

    Returns:
        二值掩码数组
    """
    counts = np.array(rle['counts'])
    size = rle['size']

    mask = np.zeros(size[0] * size[1], dtype=np.uint8)

    # 实现与 SAM 中 mask_to_rle 完全对应的解码算法
    # SAM 的编码是：
    # runs = [start1, length1, start2, length2, start3, length3, ...]
    # 其中 start 是连续 1 段的起始索引，length 是该段的长度
    for i in range(0, len(counts), 2):
        if i + 1 < len(counts):
            start = counts[i]
            length = counts[i + 1]
            mask[start:start + length] = 1

    return mask.reshape(size)


def _save_visualization(param: Dict[str, Any], masks: List[Dict], cur_result_set: Dict[str, Any]):
    """
    保存分割结果的可视化图像

    Args:
        param: 输入参数
        masks: RLE 编码的掩码列表
        cur_result_set: 当前结果集
    """
    import time
    try:
        # 加载配置
        config = _load_config()
        debug_cfg = config.get('debug', {})
        save_vis = debug_cfg.get('save_visualization', True)

        if not save_vis:
            return

        # 获取输入图像
        image = None
        if param.get('use_redis_key'):
            image_key = param.get('image')
            if image_key:
                try:
                    from utils.utils import connect_remote_redis
                    redis_client = connect_remote_redis()

                    if not redis_client.exists(image_key):
                        logger.warning(f'[SAM] 图像key {image_key} 在 Redis 中不存在')
                    else:
                        img_bytes = redis_client.get(image_key)
                        if img_bytes:
                            # 首先尝试获取 shape 信息，使用与 YOLO-World 相同的逻辑
                            shape_bytes = redis_client.get(f"{image_key}:shape")
                            if shape_bytes:
                                shape = tuple(json.loads(shape_bytes))
                                image = np.frombuffer(img_bytes, dtype=np.uint8).reshape(shape)
                            else:
                                np_arr = np.frombuffer(img_bytes, np.uint8)
                                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                except Exception as e:
                    logger.warning(f'[SAM] 从 Redis 读取图像失败: {e}')
        else:
            image = param.get('image')

        # 如果 param 中没有图像，尝试从 prepareInput_result 中获取
        if image is None and cur_result_set:
            prep_result = cur_result_set.get('prepareInput_result', {})
            if prep_result.get('success', False):
                if prep_result.get('use_redis'):
                    image_key = prep_result.get('redis_key')
                    logger.info(f'SAM  的 image_key 结果: {image_key}')
                    if image_key:
                        try:
                            from utils.utils import connect_remote_redis
                            redis_client = connect_remote_redis()

                            if not redis_client.exists(image_key):
                                logger.warning(f'[SAM] 图像key {image_key} 在 Redis 中不存在')
                            else:
                                img_bytes = redis_client.get(image_key)
                                if img_bytes:
                                    shape_bytes = redis_client.get(f"{image_key}:shape")
                                    if shape_bytes:
                                        shape = tuple(json.loads(shape_bytes))
                                        image = np.frombuffer(img_bytes, dtype=np.uint8).reshape(shape)
                                    else:
                                        np_arr = np.frombuffer(img_bytes, np.uint8)
                                        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        except Exception as e:
                            logger.warning(f'[SAM] 从 Redis 读取图像失败: {e}')
                else:
                    image = prep_result.get('image')

        if image is None:
            logger.warning(f'[SAM] 无法获取图像，跳过可视化保存')
            return

        # 创建可视化图像
        vis_image = image.copy()

        # 为每个掩码添加颜色和边界
        colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 黄色
            (255, 0, 255),  # 洋红色
            (0, 255, 255),  # 青色
        ]

        # 获取目标类别（用于文件名）
        target_class = param.get('text_query', 'object')
        if not target_class:
            # 从 cur_result_set 中获取
            prepare_result = cur_result_set.get('prepareInput_result', {})
            target_class = prepare_result.get('text_prompt', 'object')

        for i, mask_rle in enumerate(masks):
            mask = _rle_to_mask(mask_rle)

            # 为掩码创建彩色叠加
            color = colors[i % len(colors)]
            alpha = 0.5  # 透明度

            # 创建彩色掩码
            colored_mask = np.zeros_like(vis_image)
            colored_mask[mask > 0] = color

            # 叠加到原图
            vis_image = cv2.addWeighted(colored_mask, alpha, vis_image, 1 - alpha, 0)

            # 在掩码上添加边界
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_image, contours, -1, (255, 255, 255), 2)

            # 添加掩码编号和目标类别
            mask_center = np.where(mask > 0)
            if mask_center[0].size > 0:
                center_y = int(np.mean(mask_center[0]))
                center_x = int(np.mean(mask_center[1]))
                label = f'{target_class} {i+1}'
                cv2.putText(vis_image, label, (center_x, center_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 创建 debug 目录并保存图像
        current_dir = os.path.dirname(os.path.abspath(__file__))
        debug_dir = os.path.join(current_dir, 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)

        # 使用目标类别作为文件名的一部分，方便查找
        filename = f"sam_{target_class}_{timestamp}.jpg"
        save_path = os.path.join(debug_dir, filename)
        cv2.imwrite(save_path, vis_image)
        logger.info(f'[SAM] ✅ 可视化结果已保存: {save_path}')

        # 同时保存单独的掩码图像，参考用户提供的代码
        for i, mask_rle in enumerate(masks):
            mask = _rle_to_mask(mask_rle)
            mask_uint8 = (mask * 255).astype(np.uint8)
            mask_filename = f"mask_{target_class}_{i+1}_{timestamp}.png"
            mask_save_path = os.path.join(debug_dir, mask_filename)
            cv2.imwrite(mask_save_path, mask_uint8)
            logger.info(f'[SAM] ✅ 掩码图像已保存: {mask_save_path}')

    except Exception as e:
        logger.error(f'[SAM] 保存可视化失败: {e}')


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
    logger.info('[SAM] postAction 开始')
    logger.info(f'[SAM] 📥 param keys: {list(param.keys())}')
    logger.info(f'[SAM] 📥 cur_result_set keys: {list(cur_result_set.keys())}')

    inference_result = cur_result_set.get('inference_result', {})

    logger.info(f'[SAM] 推理结果: {inference_result}')

    # 保存可视化结果
    masks = inference_result.get('masks', [])
    if masks:
        _save_visualization(param, masks, cur_result_set)

    total_time = int(time.time() * 1000 - start_time)

    result = {
        'success': True,
        'masks': masks,
        'total_time': total_time
    }

    logger.info(f'[SAM] ✅ postAction 完成')
    logger.info(f'[SAM] 总耗时: {total_time}ms')
    logger.info('=' * 50)

    return result
