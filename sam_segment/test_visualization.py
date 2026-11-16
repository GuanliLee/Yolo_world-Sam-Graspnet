#!/usr/bin/env python3
"""
测试 SAM 分割可视化功能的脚本
"""

import sys
import os
import yaml
import cv2
import numpy as np
import time

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from model_config.sam_segment.config_sam_segment import _save_visualization

def test_visualization():
    print("=" * 50)
    print("测试 SAM 分割可视化功能")
    print("=" * 50)

    try:
        # 模拟参数
        param = {
            "text_query": "mouse",
            "use_redis_key": False,
            "image": None  # 我们将直接创建一个测试图像
        }

        # 创建一个简单的测试图像（黑色背景，红色矩形模拟目标）
        test_image = np.zeros((400, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (285, 123), (351, 200), (0, 0, 255), 2)
        cv2.putText(test_image, "mouse", (285, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        param["image"] = test_image

        # 模拟掩码（与真实推理结果中的掩码类似）
        masks = [
            {
                "counts": [79032, 20, 79665, 30, 80304, 32, 80943, 35, 81582, 38, 82222, 41, 82861, 44, 83500, 47, 84140, 49, 84779, 51, 85419, 52, 86059, 52, 86698, 53, 87338, 53, 87978, 53, 88617, 53, 89257, 53, 89897, 53, 90537, 53, 91177, 52, 91816, 53, 92456, 52, 93096, 52, 93735, 52, 94375, 52, 95015, 52, 95654, 52, 96294, 52, 96934, 52, 97574, 52, 98214, 52, 98854, 52, 99493, 53, 100133, 53, 100773, 53, 101413, 53, 102052, 54, 102692, 54, 103332, 54, 103971, 55, 104611, 55, 105251, 55, 105890, 56, 106530, 56, 107170, 56, 107809, 57, 108449, 57, 109089, 57, 109729, 57, 110368, 58, 111008, 57, 111648, 57, 112287, 58, 112927, 58, 113567, 58, 114207, 57, 114847, 57, 115487, 56, 116127, 56, 116767, 56, 117407, 55, 118047, 54, 118688, 52, 119328, 51, 119968, 50, 120609, 49, 121249, 48, 121890, 46, 122531, 44, 123173, 40, 123814, 38, 124456, 34, 125099, 29, 125742, 24, 126386, 15],
                "size": [400, 640]
            }
        ]

        # 模拟 cur_result_set
        cur_result_set = {
            "prepareInput_result": {
                "success": True,
                "text_prompt": "mouse"
            }
        }

        print("✅ 创建测试数据成功")

        # 调用可视化函数
        _save_visualization(param, masks, cur_result_set)

        print("=" * 50)
        print("✅ 测试完成！")
        print("=" * 50)

        # 检查 debug 目录
        debug_dir = os.path.join(os.path.dirname(__file__), "debug")
        if os.path.exists(debug_dir):
            print(f"\n📁 debug 目录内容:")
            files = os.listdir(debug_dir)
            for file in sorted(files):
                if "sam_" in file or "mask_" in file:
                    file_path = os.path.join(debug_dir, file)
                    file_size = os.path.getsize(file_path)
                    print(f"  - {file} ({file_size} bytes)")
        else:
            print("\n⚠️ debug 目录不存在")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    test_visualization()
