#!/usr/bin/env python3
"""
GraspNet 完整流程测试脚本
测试流程: Redis RGB图 + 深度图 → YOLO-World检测 → SAM分割 → GraspNet推理 → 返回4x4矩阵

使用方法:
    python test_graspnet_pipeline.py
"""

import os
import sys
import json
import cv2
import numpy as np
import time
from typing import Dict, Any, Optional

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, '..'))

try:
    from utils.utils import get_logger, connect_remote_redis
    logger = get_logger("test_graspnet_pipeline")
except ImportError:
    import logging
    logger = logging.getLogger("test_graspnet_pipeline")
    logger.setLevel(logging.INFO)


def upload_test_images_to_redis(redis_client,
                               rgb_image: np.ndarray,
                               depth_image: np.ndarray,
                               data_id: str,
                               cam_K: Optional[list] = None) -> bool:
    """
    上传测试图像到 Redis

    Args:
        redis_client: Redis 客户端
        rgb_image: RGB 图像 (H, W, 3)
        depth_image: 深度图 (H, W)
        data_id: 数据 ID
        cam_K: 相机内参矩阵 (3, 3)

    Returns:
        是否上传成功
    """
    try:
        logger.info("=" * 60)
        logger.info("📤 上传测试图像到 Redis")
        logger.info(f"   Data ID: {data_id}")
        logger.info(f"   RGB Shape: {rgb_image.shape}")
        logger.info(f"   Depth Shape: {depth_image.shape}")

        # 1. 上传 RGB 图像
        rgb_key = f"{data_id}:rgb"
        rgb_bytes = rgb_image.tobytes()
        redis_client.setex(rgb_key, 300, rgb_bytes)
        redis_client.setex(f"{rgb_key}:shape", 300, json.dumps(list(rgb_image.shape)))
        redis_client.setex(f"{rgb_key}:name", 300, data_id)
        logger.info(f"   ✅ RGB 图像上传成功: {rgb_key}")

        # 2. 上传深度图
        depth_key = f"{data_id}:depth"
        depth_bytes = depth_image.astype(np.float32).tobytes()
        redis_client.setex(depth_key, 300, depth_bytes)
        redis_client.setex(f"{depth_key}:shape", 300, json.dumps(list(depth_image.shape)))
        logger.info(f"   ✅ 深度图上传成功: {depth_key}")

        # 3. 上传相机内参（可选）
        if cam_K is not None:
            cam_K_key = f"{data_id}:cam_K"
            redis_client.setex(cam_K_key, 300, json.dumps(cam_K))
            logger.info(f"   ✅ 相机内参上传成功: {cam_K_key}")

        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"❌ 上传图像到 Redis 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_test_images() -> tuple:
    """
    创建测试图像

    Returns:
        (rgb_image, depth_image, cam_K)
    """
    logger.info("🎨 创建测试图像...")

    # 创建 RGB 图像 (480, 640, 3)
    rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # 绘制一个红色矩形（模拟目标物体）
    cv2.rectangle(rgb_image, (200, 150), (440, 330), (0, 0, 255), -1)
    cv2.putText(rgb_image, "Test Object", (220, 240),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 创建深度图 (480, 640)
    depth_image = np.full((480, 640), 0.6, dtype=np.float32)

    # 在目标区域设置不同的深度
    depth_image[150:330, 200:440] = 0.5

    # 相机内参 (3x3)
    H, W = rgb_image.shape[:2]
    fovy = np.pi / 4
    focal = H / (2.0 * np.tan(fovy / 2.0))
    cam_K = [
        [focal, 0.0, W / 2.0],
        [0.0, focal, H / 2.0],
        [0.0, 0.0, 1.0]
    ]

    logger.info(f"   RGB Shape: {rgb_image.shape}")
    logger.info(f"   Depth Shape: {depth_image.shape}")
    logger.info(f"   Camera K: {cam_K}")

    return rgb_image, depth_image, cam_K


def test_yolo_world_detection(redis_client, data_id: str) -> Optional[Dict[str, Any]]:
    """
    测试 YOLO-World 检测

    Args:
        redis_client: Redis 客户端
        data_id: 数据 ID

    Returns:
        检测结果
    """
    try:
        logger.info("=" * 60)
        logger.info("🔍 步骤 1: YOLO-World 检测")
        logger.info("=" * 60)

        # 导入 YOLO-World 配置
        sys.path.insert(0, os.path.join(current_dir, 'yolo_world'))
        from config_yolo_world import prepareInput
        from model_inference import get_service

        # 准备输入
        param = {
            'image': f"{data_id}:rgb",
            'use_redis_key': True,
            'text_prompt': 'object'
        }

        prep_result = prepareInput(param)
        if not prep_result.get('success', False):
            logger.error(f"❌ YOLO-World prepareInput 失败: {prep_result.get('error')}")
            return None

        # 执行检测
        service = get_service()
        model_dir = os.path.join(current_dir, 'yolo_world', 'models')
        service.load_model('yolo_world', model_dir)

        detect_result = service.detect(prep_result)

        logger.info(f"✅ YOLO-World 检测完成")
        logger.info(f"   检测到 {len(detect_result.get('detections', []))} 个目标")

        return detect_result

    except Exception as e:
        logger.error(f"❌ YOLO-World 检测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_sam_segmentation(redis_client, data_id: str, detections: list) -> Optional[Dict[str, Any]]:
    """
    测试 SAM 分割

    Args:
        redis_client: Redis 客户端
        data_id: 数据 ID
        detections: YOLO-World 检测结果

    Returns:
        分割结果
    """
    try:
        logger.info("=" * 60)
        logger.info("✂️  步骤 2: SAM 分割")
        logger.info("=" * 60)

        # 导入 SAM 配置
        sys.path.insert(0, os.path.join(current_dir, 'sam_segment'))
        from config_sam_segment import prepareInput
        from model_inference import get_service

        # 提取检测框
        bboxes = [det.get('bbox', []) for det in detections.get('detections', [])]
        logger.info(f"   检测框数量: {len(bboxes)}")

        if len(bboxes) == 0:
            logger.warning("⚠️  没有检测框，跳过 SAM 分割")
            return {'masks': []}

        # 准备输入
        param = {
            'image': f"{data_id}:rgb",
            'use_redis_key': True,
            'bboxes': bboxes
        }

        prep_result = prepareInput(param)
        if not prep_result.get('success', False):
            logger.error(f"❌ SAM prepareInput 失败: {prep_result.get('error')}")
            return None

        # 执行分割
        service = get_service()
        model_dir = os.path.join(current_dir, 'sam_segment', 'models')
        service.load_model('sam_segment', model_dir)

        segment_result = service.segment(prep_result)

        logger.info(f"✅ SAM 分割完成")
        logger.info(f"   分割出 {len(segment_result.get('masks', []))} 个 mask")

        return segment_result

    except Exception as e:
        logger.error(f"❌ SAM 分割失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_graspnet_inference(redis_client, data_id: str, masks: list, cam_K: list) -> Optional[Dict[str, Any]]:
    """
    测试 GraspNet 推理

    Args:
        redis_client: Redis 客户端
        data_id: 数据 ID
        masks: SAM 分割结果
        cam_K: 相机内参

    Returns:
        抓取位姿结果（4x4 矩阵）
    """
    try:
        logger.info("=" * 60)
        logger.info("🤖 步骤 3: GraspNet 抓取预测")
        logger.info("=" * 60)

        # 导入 GraspNet 配置
        sys.path.insert(0, os.path.join(current_dir, 'graspnet'))
        from config_graspnet import prepareInput
        from model_inference import get_service

        # 准备输入
        param = {
            'image': f"{data_id}:rgb",
            'depth_map': f"{data_id}:depth",
            'mask': f"{data_id}:mask",
            'use_redis_key': True,
            'cam_K': cam_K
        }

        prep_result = prepareInput(param)
        if not prep_result.get('success', False):
            logger.error(f"❌ GraspNet prepareInput 失败: {prep_result.get('error')}")
            return None

        # 执行推理
        service = get_service()
        model_dir = os.path.join(current_dir, 'graspnet', 'models')
        service.load_model('graspnet', model_dir)

        inference_result = service.inference(param)

        logger.info(f"✅ GraspNet 推理完成")

        # 检查返回结果
        if inference_result.get('success'):
            pose_matrix = inference_result.get('pose_matrix')
            if pose_matrix is not None:
                logger.info("=" * 60)
                logger.info("🎯 GraspNet 推理结果:")
                logger.info("=" * 60)
                logger.info(f"✅ 成功返回 4x4 抓取位姿矩阵:")
                for i, row in enumerate(pose_matrix):
                    logger.info(f"   [{i}] {row}")
                logger.info("=" * 60)
                return inference_result
            else:
                logger.error("❌ 推理结果中没有 pose_matrix")
                return None
        else:
            logger.error(f"❌ GraspNet 推理失败: {inference_result.get('error')}")
            return None

    except Exception as e:
        logger.error(f"❌ GraspNet 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_full_pipeline():
    """
    测试完整流程: Redis RGB图 + 深度图 → YOLO-World检测 → SAM分割 → GraspNet推理 → 返回4x4矩阵
    """
    logger.info("=" * 80)
    logger.info("🚀 开始测试 GraspNet 完整流程")
    logger.info("=" * 80)

    try:
        # 1. 连接 Redis
        logger.info("📡 连接 Redis...")
        redis_client = connect_remote_redis()
        logger.info("✅ Redis 连接成功")

        # 2. 创建测试图像
        rgb_image, depth_image, cam_K = create_test_images()

        # 3. 上传图像到 Redis
        data_id = f"graspnet_test_{int(time.time())}"
        upload_success = upload_test_images_to_redis(
            redis_client, rgb_image, depth_image, data_id, cam_K
        )

        if not upload_success:
            logger.error("❌ 上传图像失败，测试终止")
            return

        # 4. 执行 YOLO-World 检测
        detect_result = test_yolo_world_detection(redis_client, data_id)
        if detect_result is None:
            logger.error("❌ YOLO-World 检测失败，测试终止")
            return

        # 5. 执行 SAM 分割
        segment_result = test_sam_segmentation(redis_client, data_id, detect_result)
        if segment_result is None:
            logger.error("❌ SAM 分割失败，测试终止")
            return

        # 6. 执行 GraspNet 推理
        graspnet_result = test_graspnet_inference(
            redis_client, data_id,
            segment_result.get('masks', []),
            cam_K
        )

        # 7. 总结测试结果
        logger.info("=" * 80)
        logger.info("📊 测试结果总结:")
        logger.info("=" * 80)
        logger.info(f"   Data ID: {data_id}")
        logger.info(f"   YOLO-World 检测: {len(detect_result.get('detections', []))} 个目标")
        logger.info(f"   SAM 分割: {len(segment_result.get('masks', []))} 个 mask")

        if graspnet_result and graspnet_result.get('success'):
            logger.info(f"   GraspNet 推理: ✅ 成功")
            logger.info(f"   返回 4x4 矩阵: ✅ 是")
        else:
            logger.info(f"   GraspNet 推理: ❌ 失败")
            logger.info(f"   返回 4x4 矩阵: ❌ 否")

        logger.info("=" * 80)

        # 8. 清理 Redis 数据
        redis_client.delete(f"{data_id}:rgb")
        redis_client.delete(f"{data_id}:rgb:shape")
        redis_client.delete(f"{data_id}:rgb:name")
        redis_client.delete(f"{data_id}:depth")
        redis_client.delete(f"{data_id}:depth:shape")
        redis_client.delete(f"{data_id}:cam_K")
        logger.info("🧹 清理 Redis 数据完成")

        logger.info("=" * 80)
        logger.info("✅ 测试完成！")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_full_pipeline()
