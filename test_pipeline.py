#!/usr/bin/env python3
"""
完整流水线测试脚本
测试 YOLO-World + SAM + GraspNet 完整流程

流程:
1. 上传测试图像到 Redis
2. YOLO-World: 目标检测
3. SAM: 分割检测到的目标
4. GraspNet: 预测抓取位姿
"""

import os
import sys
import json
import cv2
import numpy as np
import time
from typing import Dict, Any, Optional, List

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, '..'))

try:
    from utils.utils import get_logger, connect_remote_redis
    logger = get_logger("test_pipeline")
except ImportError:
    import logging
    logger = logging.getLogger("test_pipeline")
    logger.setLevel(logging.INFO)


def create_test_image() -> np.ndarray:
    """
    创建测试图像

    Returns:
        RGB 图像 (H, W, 3)
    """
    logger.info("🎨 创建测试图像...")

    rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # 绘制一个红色矩形作为测试对象
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


def step1_yolo_world_detection(redis_client, rgb_key: str) -> Optional[List[Dict]]:
    """
    步骤1: YOLO-World 目标检测

    Args:
        redis_client: Redis 客户端
        rgb_key: RGB 图像 Redis key

    Returns:
        检测结果列表，失败返回 None
    """
    try:
        logger.info("=" * 80)
        logger.info("📌 步骤 1/3: YOLO-World 目标检测")
        logger.info("=" * 80)

        sys.path.insert(0, os.path.join(current_dir, 'yolo_world'))
        from config_yolo_world import prepareInput
        from model_inference import YoloWorldService

        param = {
            'image': rgb_key,
            'use_redis_key': True,
            'text_prompt': 'object'
        }

        prep_result = prepareInput(param)
        if not prep_result.get('success', False):
            logger.error(f"❌ prepareInput 失败: {prep_result.get('error')}")
            return None

        service = YoloWorldService()
        model_dir = os.path.join(current_dir, 'yolo_world')
        load_success = service.load_model('yolo_world', model_dir)

        if not load_success:
            logger.error("❌ 模型加载失败")
            return None

        detect_result = service.detect(prep_result)

        detections = detect_result.get('detections', [])
        logger.info(f"✅ YOLO-World 检测完成")
        logger.info(f"   检测到 {len(detections)} 个目标")

        for i, det in enumerate(detections):
            bbox = det.get('bbox', [])
            conf = det.get('confidence', 0)
            cls_name = det.get('class_name', 'unknown')
            logger.info(f"   [{i+1}] {cls_name}: 置信度={conf:.3f}, bbox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")

        return detections

    except Exception as e:
        logger.error(f"❌ YOLO-World 检测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def step2_sam_segmentation(redis_client, rgb_key: str, bboxes: List[List]) -> Optional[List[Dict]]:
    """
    步骤2: SAM 分割

    Args:
        redis_client: Redis 客户端
        rgb_key: RGB 图像 Redis key
        bboxes: 边界框列表

    Returns:
        分割结果列表，失败返回 None
    """
    try:
        logger.info("=" * 80)
        logger.info("📌 步骤 2/3: SAM 分割")
        logger.info("=" * 80)

        sys.path.insert(0, os.path.join(current_dir, 'sam_segment'))
        from config_sam_segment import prepareInput
        from model_inference import SAMSegmentService

        param = {
            'image': rgb_key,
            'use_redis_key': True,
            'bboxes': bboxes
        }

        prep_result = prepareInput(param)
        if not prep_result.get('success', False):
            logger.error(f"❌ prepareInput 失败: {prep_result.get('error')}")
            return None

        service = SAMSegmentService()
        model_dir = os.path.join(current_dir, 'sam_segment')
        load_success = service.load_model('sam_segment', model_dir)

        if not load_success:
            logger.error("❌ 模型加载失败")
            return None

        segment_result = service.segment(prep_result)

        masks = segment_result.get('masks', [])
        logger.info(f"✅ SAM 分割完成")
        logger.info(f"   分割出 {len(masks)} 个 mask")

        for i, mask in enumerate(masks):
            mask_data = mask.get('mask')
            if mask_data is not None:
                logger.info(f"   [{i+1}] mask shape: {mask_data.shape}")

        return masks

    except Exception as e:
        logger.error(f"❌ SAM 分割失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def step3_graspnet_prediction(redis_client, rgb_key: str, depth_key: str, mask: np.ndarray) -> Optional[Dict]:
    """
    步骤3: GraspNet 抓取预测

    Args:
        redis_client: Redis 客户端
        rgb_key: RGB 图像 Redis key
        depth_key: 深度图 Redis key
        mask: 分割 mask

    Returns:
        抓取位姿结果，失败返回 None
    """
    try:
        logger.info("=" * 80)
        logger.info("📌 步骤 3/3: GraspNet 抓取预测")
        logger.info("=" * 80)

        sys.path.insert(0, os.path.join(current_dir, 'graspnet'))
        from config_graspnet import prepareInput
        from model_inference import GraspNetService

        # 上传 mask 到 Redis
        mask_key = f"test_graspnet_mask_{int(time.time())}"
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
            return None

        service = GraspNetService()
        model_dir = os.path.join(current_dir, 'graspnet')
        load_success = service.load_model('graspnet', model_dir)

        if not load_success:
            logger.error("❌ 模型加载失败")
            return None

        inference_result = service.inference(param)

        if inference_result.get('success'):
            pose_matrix = inference_result.get('pose_matrix')
            if pose_matrix is not None:
                logger.info(f"✅ GraspNet 预测完成")
                logger.info(f"   ✅ 成功返回 4x4 抓取位姿矩阵:")
                for i, row in enumerate(pose_matrix):
                    logger.info(f"      [{i}] {row}")
                return inference_result
            else:
                logger.error("   ❌ 推理结果中没有 pose_matrix")
                return None
        else:
            logger.error(f"   ❌ 推理失败: {inference_result.get('error')}")
            return None

    except Exception as e:
        logger.error(f"❌ GraspNet 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    主函数
    """
    logger.info("=" * 80)
    logger.info("🚀 开始完整流水线测试: YOLO-World + SAM + GraspNet")
    logger.info("=" * 80)

    try:
        # 连接 Redis
        logger.info("📡 连接 Redis...")
        redis_client = connect_remote_redis()
        logger.info("✅ Redis 连接成功")

        # 创建测试数据
        rgb_image = create_test_image()
        depth_image = create_test_depth_image()

        data_id = f"test_pipeline_{int(time.time())}"
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

        # 步骤1: YOLO-World 检测
        detections = step1_yolo_world_detection(redis_client, rgb_key)
        if detections is None or len(detections) == 0:
            logger.error("❌ YOLO-World 检测失败或未检测到目标")
            return

        # 步骤2: SAM 分割
        bboxes = [det.get('bbox') for det in detections]
        masks = step2_sam_segmentation(redis_client, rgb_key, bboxes)
        if masks is None or len(masks) == 0:
            logger.error("❌ SAM 分割失败")
            return

        # 步骤3: GraspNet 预测
        mask = masks[0].get('mask')
        if mask is None:
            logger.error("❌ SAM 没有返回有效的 mask")
            return

        grasp_result = step3_graspnet_prediction(redis_client, rgb_key, depth_key, mask)
        if grasp_result is None:
            logger.error("❌ GraspNet 预测失败")
            return

        # 输出最终结果
        logger.info("=" * 80)
        logger.info("🎉 完整流水线测试成功！")
        logger.info("=" * 80)
        logger.info("📊 结果汇总:")
        logger.info(f"   1️⃣ YOLO-World: 检测到 {len(detections)} 个目标")
        logger.info(f"   2️⃣ SAM: 分割出 {len(masks)} 个 mask")
        logger.info(f"   3️⃣ GraspNet: 成功预测抓取位姿")
        logger.info("=" * 80)

        # 清理 Redis 数据
        redis_client.delete(rgb_key)
        redis_client.delete(f"{rgb_key}:shape")
        redis_client.delete(depth_key)
        redis_client.delete(f"{depth_key}:shape")
        logger.info("🧹 清理 Redis 数据完成")

    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
