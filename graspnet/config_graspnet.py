"""
GraspNet 抓取预测模型配置文件
提供 prepareInput, postAction 钩子函数
返回4x4抓取位姿矩阵

V1.0 - 简化版，只返回4x4矩阵
V1.1 - 添加立体匹配计算深度图
"""

import sys
import os
import json
import cv2
import numpy as np
import yaml
from typing import Dict, Any, Optional

# 添加项目路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))

try:
    from utils.utils import get_logger
    logger = get_logger("graspnet")
except ImportError:
    import logging
    logger = logging.getLogger("graspnet")
    logger.setLevel(logging.INFO)

# 组件名称
component = "yoloseries_graspnet"

# 全局模块缓存
_model_inference_module = None


def compute_depth_from_stereo(left_image: np.ndarray,
                           right_image: np.ndarray,
                           baseline: float = 0.06,
                           focal_length: float = 2452.0,
                           min_disparity: int = 0,
                           num_disparities: int = 64,
                           block_size: int = 15) -> np.ndarray:
    """
    从左右立体图像计算深度图（使用 SGBM 算法）

    Args:
        left_image: 左图像 (H, W, 3) 或 (H, W)
        right_image: 右图像 (H, W, 3) 或 (H, W)
        baseline: 双目相机基线距离（米），默认 60mm
        focal_length: 焦距（像素），默认根据 IMX219 相机计算 (2452px)
        min_disparity: 最小视差
        num_disparities: 视差搜索范围（必须是16的倍数）
        block_size: 匹配块大小（必须是奇数）

    Returns:
        深度图 (H, W), dtype=float32, 单位：米
    """
    logger.info("[GraspNet] 📐 开始立体匹配计算深度图")
    logger.info(f"   左图形状: {left_image.shape}")
    logger.info(f"   右图形状: {right_image.shape}")
    logger.info(f"   基线: {baseline}m ({baseline*1000:.1f}mm)")
    logger.info(f"   焦距: {focal_length}px ({focal_length*0.00112:.2f}mm)")

    # 转换为灰度图
    if len(left_image.shape) == 3:
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = left_image

    if len(right_image.shape) == 3:
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    else:
        right_gray = right_image

    # 创建 SGBM 匹配器
    window_size = block_size
    min_disp = min_disparity
    num_disp = num_disparities

    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # 计算视差图
    disparity = sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0

    # 计算深度图: depth = (focal_length * baseline) / disparity
    # 避免除以零
    disparity[disparity <= 0] = 0.1
    depth = (focal_length * baseline) / disparity

    # 限制深度范围（0.1m - 5.0m）
    depth = np.clip(depth, 0.1, 5.0)

    logger.info(f"[GraspNet] ✅ 深度图计算完成")
    logger.info(f"   深度范围: {depth.min():.3f}m - {depth.max():.3f}m")
    logger.info(f"   深度图形状: {depth.shape}")

    return depth.astype(np.float32)


def _get_model_inference_module():
    """获取 model_inference 模块（延迟加载）"""
    global _model_inference_module

    if _model_inference_module is None:
        import importlib.util

        module_name = 'graspnet_model_inference'

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
    return "graspnet"


_config_cache = None


def _load_config() -> Dict[str, Any]:
    """加载 graspnet.yaml 配置"""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'graspnet.yaml')

    default_config = {
        'detection': {
            'top_k': 20,
            'vertical_threshold': 30,
            'num_point': 3000
        },
        'network': {
            'input_feature_dim': 0,
            'num_view': 300,
            'num_angle': 12,
            'num_depth': 4,
            'cylinder_radius': 0.05,
            'hmin': -0.02,
            'hmax_list': [0.01, 0.02, 0.03, 0.04]
        },
        'collision': {
            'voxel_size': 0.01,
            'approach_dist': 0.05,
            'collision_thresh': 0.01
        }
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config:
                    # 递归合并配置
                    def merge_dict(base, update):
                        for key, value in update.items():
                            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                                merge_dict(base[key], value)
                            else:
                                base[key] = value
                    merge_dict(default_config, loaded_config)
        except Exception as e:
            logger.warning(f'[GraspNet] 加载配置失败: {e}')

    _config_cache = default_config
    return default_config


def prepareInput(param: Dict[str, Any] = None,
                 cur_result_set: Dict[str, Any] = None,
                 invoke_callback=None) -> Dict[str, Any]:
    """
    推理前数据预处理

    从 Redis 读取 RGB-D 图像和 mask
    """
    if param is None:
        param = {}
    if cur_result_set is None:
        cur_result_set = {}

    logger.info('=' * 60)
    logger.info('[GraspNet] prepareInput 开始')
    logger.info(f'[GraspNet] 📥 param keys: {list(param.keys())}')

    # 处理不同的参数名称
    use_redis_key = param.get('use_redis_key', False)
    # 兼容 use_redis 参数
    if not use_redis_key:
        use_redis_key = param.get('use_redis', False)

    # ============================================
    # Redis 模式（推荐）
    # ============================================
    if use_redis_key:
        try:
            from utils.utils import connect_remote_redis
            redis_client = connect_remote_redis()

            # 1. 获取 RGB 图像
            image_key = param.get('image')
            # 兼容 redis_key 参数
            if not image_key:
                image_key = param.get('redis_key')
            if not image_key:
                logger.error('[GraspNet] ❌ 缺少 image 或 redis_key 参数')
                return {'success': False, 'error': 'missing_image_param'}

            if not redis_client.exists(image_key):
                logger.error(f'[GraspNet] ❌ Redis 中找不到图像: {image_key}')
                return {'success': False, 'error': 'redis_key_not_found'}

            # 获取图像 shape
            shape_bytes = redis_client.get(f"{image_key}:shape")
            image_shape = None
            if shape_bytes:
                image_shape = tuple(json.loads(shape_bytes))
                logger.info(f'[GraspNet] ✅ RGB图像key验证通过, shape: {image_shape}')

            # 1.5. 获取左右图像（用于立体匹配计算深度图）
            left_image_key = param.get('left_image')
            right_image_key = param.get('right_image')
            left_image = None
            right_image = None

            if left_image_key and right_image_key:
                if redis_client.exists(left_image_key) and redis_client.exists(right_image_key):
                    left_shape_bytes = redis_client.get(f"{left_image_key}:shape")
                    right_shape_bytes = redis_client.get(f"{right_image_key}:shape")

                    left_shape = tuple(json.loads(left_shape_bytes)) if left_shape_bytes else None
                    right_shape = tuple(json.loads(right_shape_bytes)) if right_shape_bytes else None

                    left_data = redis_client.get(left_image_key)
                    right_data = redis_client.get(right_image_key)

                    left_image = np.frombuffer(left_data, dtype=np.uint8).reshape(left_shape)
                    right_image = np.frombuffer(right_data, dtype=np.uint8).reshape(right_shape)

                    logger.info(f'[GraspNet] ✅ 左右图像加载成功')
                    logger.info(f'   左图: {left_image.shape}')
                    logger.info(f'   右图: {right_image.shape}')

            # 2. 获取深度图（优先使用现成的深度图）
            depth_key = param.get('depth_map')
            depth_shape = None
            computed_depth = False

            # 优先从 Redis 获取现成的深度图
            if depth_key and redis_client.exists(depth_key):
                depth_shape_bytes = redis_client.get(f"{depth_key}:shape")
                if depth_shape_bytes:
                    depth_shape = tuple(json.loads(depth_shape_bytes))
                    logger.info(f'[GraspNet] ✅ 深度图key验证通过, shape: {depth_shape}')
                else:
                    logger.info(f'[GraspNet] ✅ 深度图key验证通过')
            # 如果没有现成的深度图，且提供了左右图像，才计算深度图
            elif left_image is not None and right_image is not None:
                logger.info('[GraspNet] 📐 使用左右图像计算深度图')

                # 获取立体匹配参数
                baseline = param.get('stereo_baseline', 0.06)
                focal_length = param.get('stereo_focal', 525.0)

                # 计算深度图
                depth_array = compute_depth_from_stereo(
                    left_image, right_image,
                    baseline=baseline,
                    focal_length=focal_length
                )

                # 将计算出的深度图保存到 Redis
                computed_depth_key = f"{image_key}:computed_depth"
                redis_client.setex(computed_depth_key, 300, depth_array.tobytes())
                redis_client.setex(f"{computed_depth_key}:shape", 300, json.dumps(list(depth_array.shape)))

                depth_key = computed_depth_key
                depth_shape = depth_array.shape
                computed_depth = True
                logger.info(f'[GraspNet] ✅ 深度图计算并保存: {computed_depth_key}')

            # 3. 获取 mask
            mask_key = param.get('mask')
            mask_shape = None
            if mask_key and redis_client.exists(mask_key):
                mask_shape_bytes = redis_client.get(f"{mask_key}:shape")
                if mask_shape_bytes:
                    mask_shape = tuple(json.loads(mask_shape_bytes))
                    logger.info(f'[GraspNet] ✅ Mask key验证通过, shape: {mask_shape}')

            # 4. 获取相机内参（可选）
            cam_K = param.get('cam_K')
            if cam_K:
                logger.info(f'[GraspNet] ✅ 相机内参K: {cam_K}')

            result = {
                'success': True,
                'use_redis': True,
                'image_key': image_key,
                'image_shape': list(image_shape) if image_shape else None,
                'depth_key': depth_key,
                'depth_shape': list(depth_shape) if depth_shape else None,
                'computed_depth': computed_depth,
                'mask_key': mask_key,
                'mask_shape': list(mask_shape) if mask_shape else None,
                'cam_K': cam_K,
                'frame_id': param.get('frame_id'),
                'frame_timestamp': param.get('frame_timestamp')
            }

            logger.info(f'[GraspNet] ✅ prepareInput 完成')
            logger.info('=' * 60)
            return result

        except Exception as e:
            logger.error(f'[GraspNet] ❌ prepareInput 错误: {e}')
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    # ============================================
    # Base64 模式
    # ============================================
    elif isinstance(param.get('image'), str) and param.get('image', '').startswith('data:image'):
        try:
            import base64
            image_key = param.get('image')
            _, encoded = image_key.split(',', 1)
            np_arr = np.frombuffer(base64.b64decode(encoded), np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if image is None:
                logger.error('[GraspNet] ❌ Base64 图像解码失败')
                return {'success': False, 'error': 'base64_decode_failed'}

            logger.info(f'[GraspNet] ✅ Base64 图像解码成功, shape: {image.shape}')

            result = {
                'success': True,
                'image': image,
                'use_redis': False,
                'image_shape': list(image.shape)
            }

            logger.info(f'[GraspNet] ✅ prepareInput 完成')
            logger.info('=' * 60)
            return result

        except Exception as e:
            logger.error(f'[GraspNet] ❌ Base64 解码错误: {e}')
            return {'success': False, 'error': str(e)}

    # ============================================
    # 文件路径模式
    # ============================================
    elif isinstance(param.get('image'), str) and os.path.exists(param.get('image')):
        try:
            image_path = param.get('image')
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f'[GraspNet] ❌ 无法读取图像文件: {image_path}')
                return {'success': False, 'error': 'file_read_failed'}

            logger.info(f'[GraspNet] ✅ 从文件读取图像成功, shape: {image.shape}')

            result = {
                'success': True,
                'image': image,
                'use_redis': False,
                'image_shape': list(image.shape)
            }

            logger.info(f'[GraspNet] ✅ prepareInput 完成')
            logger.info('=' * 60)
            return result

        except Exception as e:
            logger.error(f'[GraspNet] ❌ 文件读取错误: {e}')
            return {'success': False, 'error': str(e)}

    # ============================================
    # Numpy 数组模式
    # ============================================
    elif isinstance(param.get('image'), np.ndarray):
        image = param.get('image')
        logger.info(f'[GraspNet] ✅ 接收到 numpy 数组, shape: {image.shape}')
        result = {
            'success': True,
            'image': image,
            'use_redis': False,
            'image_shape': list(image.shape)
        }
        logger.info(f'[GraspNet] ✅ prepareInput 完成')
        logger.info('=' * 60)
        return result

    else:
        logger.error(f'[GraspNet] ❌ 不支持的输入类型')
        return {'success': False, 'error': 'unsupported_input_type'}


def postAction(param: Dict[str, Any] = None,
               cur_result_set: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    推理后处理 - 提取并返回4x4矩阵

    Args:
        param: 输入参数
        cur_result_set: 推理结果
            - inference_result: 推理输出 {"pose_matrix": [[...]], ...}

    Returns:
        简化格式输出，只返回4x4矩阵
        {
            "success": True,
            "pose_matrix": [[...]],  # 4x4 numpy array (list格式)
            ...
        }
    """
    import time

    if param is None:
        param = {}
    if cur_result_set is None:
        cur_result_set = {}

    # 记录开始时间（毫秒）
    start_time = time.time() * 1000

    logger.info('=' * 60)
    logger.info('[GraspNet] postAction 开始')
    logger.info(f'[GraspNet] 📥 param keys: {list(param.keys())}')
    logger.info(f'[GraspNet] 📥 cur_result_set keys: {list(cur_result_set.keys())}')

    # ============================================
    # 从推理结果中提取4x4矩阵
    # ============================================
    inference_result = cur_result_set.get('inference_result', {})

    if not inference_result.get('success'):
        logger.error(f'[GraspNet] ❌ 推理失败: {inference_result.get("error")}')
        return {
            'success': False,
            'error': inference_result.get('error', 'inference_failed')
        }

    # 提取4x4位姿矩阵（核心输出）
    pose_matrix = inference_result.get('pose_matrix')

    if pose_matrix is None:
        logger.error('[GraspNet] ❌ 推理结果中没有 pose_matrix')
        return {'success': False, 'error': 'missing_pose_matrix'}

    # 转换为列表格式（可JSON序列化）
    if isinstance(pose_matrix, np.ndarray):
        pose_matrix_list = pose_matrix.tolist()
    else:
        pose_matrix_list = pose_matrix

    logger.info(f'[GraspNet] ✅ 提取到4x4位姿矩阵')
    logger.info(f'[GraspNet]   矩阵形状: {len(pose_matrix_list)}x{len(pose_matrix_list[0])}')
    logger.info(f'[GraspNet]   位置 (第4列): {pose_matrix_list[0][:3]}')
    logger.info(f'[GraspNet]   旋转 (左上3x3): 已提取')

    # 计算总耗时
    total_time = int(time.time() * 1000 - start_time)

    # 构造最终结果（简化版，只返回4x4矩阵）
    result = {
        'success': True,
        'pose_matrix': pose_matrix_list,  # 4x4抓取位姿矩阵（主要输出）
        'translation': pose_matrix_list[0][:3] if pose_matrix_list else None,
        'rotation_matrix': [row[:3] for row in pose_matrix_list[:3]] if pose_matrix_list else None,
        'total_time': total_time
    }

    logger.info(f'[GraspNet] ✅ postAction 完成')
    logger.info(f'[GraspNet]   总耗时: {total_time}ms')
    logger.info('=' * 60)

    return result
