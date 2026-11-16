#!/usr/bin/env python3
"""
单模型测试脚本
用于单独测试 YOLO-World、SAM、GraspNet 每个模型

使用方法:
    python test_single_models.py --model yolo_world
    python test_single_models.py --model sam_segment
    python test_single_models.py --model graspnet
"""

import os
import sys
import json
import cv2
import numpy as np
import argparse
import time
from typing import Dict, Any, Optional

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, '..'))

try:
    from utils.utils import get_logger, connect_remote_redis
    logger = get_logger("test_single_models")
except ImportError:
    import logging
    logger = logging.getLogger("test_single_models")
    logger.setLevel(logging.INFO)


def create_test_image() -> np.ndarray:
    """
    创建测试图像

    Returns:
        RGB 图像 (H, W, 3)
    """
    logger.info("🎨 创建测试图像...")

    rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(rgb_image, (200, 150), (440, 330), (0, 0, 255), -1)
    cv2.putText(rgb_image, "Test Object", (220, 240),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    logger.info(f"   图像形状: {rgb_image.shape}")
    return rgb_image


def create_test_depth_image() -> np.ndarray:
    """
    创建测试深度图

    Returns:
        深度图 (H, W)
    """
    logger.info("🎨 创建测试深度图...")

    depth_image = np.full((480, 640), 0.6, dtype=np.float32)
    depth_image[150:330, 200:440] = 0.5

    logger.info(f"   深度图形状: {depth_image.shape}")
    return depth_image


def upload_image_to_redis(redis_client, image: np.ndarray, key: str) -> bool:
    """
    上传图像到 Redis

    Args:
        redis_client: Redis 客户端
        image: 图像数组
        key: Redis key

    Returns:
        是否上传成功
    """
    try:
        img_bytes = image.tobytes()
        redis_client.setex(key, 300, img_bytes)
        redis_client.setex(f"{key}:shape", 300, json.dumps(list(image.shape)))
        logger.info(f"   ✅ 图像上传成功: {key}")
        return True
    except Exception as e:
        logger.error(f"   ❌ 上传图像失败: {e}")
        return False


def test_yolo_world(redis_client, image_key: str) -> bool:
    """
    测试 YOLO-World 模型

    Args:
        redis_client: Redis 客户端
        image_key: 图像 Redis key

    Returns:
        是否测试成功
    """
    try:
        logger.info("=" * 60)
        logger.info("🔍 测试 YOLO-World 模型")
        logger.info("=" * 60)

        sys.path.insert(0, os.path.join(current_dir, 'yolo_world'))
        from config_yolo_world import prepareInput
        from model_inference import get_service

        param = {
            'image': image_key,
            'use_redis_key': True,
            'text_prompt': 'object'
        }

        prep_result = prepareInput(param)
        if not prep_result.get('success', False):
            logger.error(f"❌ prepareInput 失败: {prep_result.get('error')}")
            return False

        service = get_service()
        model_dir = os.path.join(current_dir, 'yolo_world', 'models')
        load_success = service.load_model('yolo_world', model_dir)

        if not load_success:
            logger.error("❌ 模型加载失败")
            return False

        detect_result = service.detect(prep_result)

        detections = detect_result.get('detections', [])
        logger.info(f"✅ YOLO-World 测试完成")
        logger.info(f"   检测到 {len(detections)} 个目标")

        for i, det in enumerate(detections):
            bbox = det.get('bbox', [])
            conf = det.get('confidence', 0)
            cls_name = det.get('class_name', 'unknown')
            logger.info(f"   [{i+1}] {cls_name}: 置信度={conf:.3f}, bbox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")

        return True

    except Exception as e:
        logger.error(f"❌ YOLO-World 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sam_segment(redis_client, image_key: str) -> bool:
    """
    测试 SAM 分割模型

    Args:
        redis_client: Redis 客户端
        image_key: 图像 Redis key

    Returns:
        是否测试成功
    """
    try:
        logger.info("=" * 60)
        logger.info("✂️ 测试 SAM 分割模型")
        logger.info("=" * 60)

        sys.path.insert(0, os.path.join(current_dir, 'sam_segment'))
        from config_sam_segment import prepareInput
        from model_inference import get_service

        # 使用一个测试检测框
        bboxes = [[200, 150, 440, 330]]

        param = {
            'image': image_key,
            'use_redis_key': True,
            'bboxes': bboxes
        }

        prep_result = prepareInput(param)
        if not prep_result.get('success', False):
            logger.error(f"❌ prepareInput 失败: {prep_result.get('error')}")
            return False

        service = get_service()
        model_dir = os.path.join(current_dir, 'sam_segment', 'models')
        load_success = service.load_model('sam_segment', model_dir)

        if not load_success:
            logger.error("❌ 模型加载失败")
            return False

        segment_result = service.segment(prep_result)

        masks = segment_result.get('masks', [])
        logger.info(f"✅ SAM 测试完成")
        logger.info(f"   分割出 {len(masks)} 个 mask")

        for i, mask in enumerate(masks):
            mask_data = mask.get('mask')
            if mask_data is not None:
                logger.info(f"   [{i+1}] mask shape: {mask_data.shape}")

        return True

    except Exception as e:
        logger.error(f"❌ SAM 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graspnet(redis_client, rgb_key: str, depth_key: str) -> bool:
    """
    测试 GraspNet 模型

    Args:
        redis_client: Redis 客户端
        rgb_key: RGB 图像 Redis key
        depth_key: 深度图 Redis key

    Returns:
        是否测试成功
    """
    try:
        logger.info("=" * 60)
        logger.info("🤖 测试 GraspNet 模型")
        logger.info("=" * 60)

        sys.path.insert(0, os.path.join(current_dir, 'graspnet'))
        from config_graspnet import prepareInput
        from model_inference import get_service

        # 创建测试 mask
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[150:330, 200:440] = 1
        mask_key = f"test_graspnet_mask"

        upload_image_to_redis(redis_client, mask, mask_key)

        # 相机内参
        H, W = 480, 640
        fovy = np.pi / 4
        focal = H / (2.0 * np.tan(fovy / 2.0))
        cam_K = [
            [focal, 0.0, W / 2.0],
            [0.0, focal, H / 2.0],
            [0.0, 0.0, 1.0]
        ]

        param = {
            'image': rgb_key,
            'depth_map': depth_key,
            'mask': mask_key,
            'use_redis_key': True,
            'cam_K': cam_K
        }

        prep_result = prepareInput(param)
        if not prep_result.get('success', False):
            logger.error(f"❌ prepareInput 失败: {prep_result.get('error')}")
            return False

        service = get_service()
        model_dir = os.path.join(current_dir, 'graspnet', 'models')
        load_success = service.load_model('graspnet', model_dir)

        if not load_success:
            logger.error("❌ 模型加载失败")
            return False

        inference_result = service.inference(param)

        logger.info(f"✅ GraspNet 测试完成")

        if inference_result.get('success'):
            pose_matrix = inference_result.get('pose_matrix')
            if pose_matrix is not None:
                logger.info(f"   ✅ 成功返回 4x4 抓取位姿矩阵:")
                for i, row in enumerate(pose_matrix):
                    logger.info(f"      [{i}] {row}")
                return True
            else:
                logger.error("   ❌ 推理结果中没有 pose_matrix")
                return False
        else:
            logger.error(f"   ❌ 推理失败: {inference_result.get('error')}")
            return False

    except Exception as e:
        logger.error(f"❌ GraspNet 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='测试单个模型')
    parser.add_argument('--model', type=str, required=True,
                       choices=['yolo_world', 'sam_segment', 'graspnet'],
                       help='要测试的模型')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info(f"🚀 开始测试模型: {args.model}")
    logger.info("=" * 80)

    try:
        # 连接 Redis
        logger.info("📡 连接 Redis...")
        redis_client = connect_remote_redis()
        logger.info("✅ Redis 连接成功")

        # 创建测试数据
        rgb_image = create_test_image()
        depth_image = create_test_depth_image()

        data_id = f"test_{args.model}_{int(time.time())}"
        rgb_key = f"{data_id}:rgb"
        depth_key = f"{data_id}:depth"

        # 上传图像到 Redis
        logger.info("📤 上传测试图像到 Redis...")
        upload_success = (
            upload_image_to_redis(redis_client, rgb_image, rgb_key) and
            upload_image_to_redis(redis_client, depth_image, depth_key)
        )

        if not upload_success:
            logger.error("❌ 上传图像失败，测试终止")
            return

        # 根据模型类型执行测试
        success = False
        if args.model == 'yolo_world':
            success = test_yolo_world(redis_client, rgb_key)
        elif args.model == 'sam_segment':
            success = test_sam_segment(redis_client, rgb_key)
        elif args.model == 'graspnet':
            success = test_graspnet(redis_client, rgb_key, depth_key)

        # 清理 Redis 数据
        redis_client.delete(rgb_key)
        redis_client.delete(f"{rgb_key}:shape")
        redis_client.delete(depth_key)
        redis_client.delete(f"{depth_key}:shape")
        redis_client.delete(f"test_graspnet_mask")
        logger.info("🧹 清理 Redis 数据完成")

        # 输出测试结果
        logger.info("=" * 80)
        if success:
            logger.info(f"✅ {args.model} 测试成功！")
        else:
            logger.info(f"❌ {args.model} 测试失败！")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
