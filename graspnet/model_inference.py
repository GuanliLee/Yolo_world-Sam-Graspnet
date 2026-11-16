"""
GraspNet 抓取预测推理引擎
返回4x4抓取位姿矩阵

参考: main_with_foundationpose.py, main_yoloWorld_sam.py
"""

import os
import sys
import json
import cv2
import numpy as np
import torch
import open3d as o3d
from typing import Dict, Any, Optional, List
import time

# 先定义 logger
try:
    from utils.utils import get_logger, connect_remote_redis
    logger = get_logger("graspnet")
except ImportError:
    import logging
    logger = logging.getLogger("graspnet")
    logger.setLevel(logging.INFO)

try:
    from pycocotools import mask as mask_utils
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False
    logger.warning("[GraspNet] pycocotools 未安装，将使用备用 RLE 解码方法")

try:
    from base.base_model_service import IModelService
except ImportError:
    IModelService = object


class GraspNetPredictor:
    """
    GraspNet 抓取预测器

    返回4x4抓取位姿矩阵
    """

    def init(self,
                 model_path: str,
                 network_config: Dict = None,
                 sampling_config: Dict = None,
                 collision_config: Dict = None,
                 detection_config: Dict = None,
                 debug_config: Dict = None,
                 device: str = None):
        """
        初始化预测器

        Args:
            model_path: 模型文件路径 (.pt)
            network_config: 网络配置
            sampling_config: 采样配置
            collision_config: 碰撞检测配置
            detection_config: 检测配置
            debug_config: 调试配置
            device: 计算设备 (None=自动, 'cuda', 'cpu')
        """
        self.model_path = model_path
        self.network_config = network_config or {}
        self.sampling_config = sampling_config or {}
        self.collision_config = collision_config or {}
        self.detection_config = detection_config or {}
        self.debug_config = debug_config or {}

        # 自动选择设备
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 模型和类别
        self.net = None
        self._done_warmup = False

        logger.info(f"[GraspNet] 初始化预测器")
        logger.info(f"   模型路径: {model_path}")
        logger.info(f"   设备: {self.device}")

    def load_model(self) -> bool:
        """
        加载 GraspNet 模型

        Returns:
            是否加载成功
        """
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"[GraspNet] 模型文件不存在: {self.model_path}")
                return False

            # 添加项目路径
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            graspnet_baseline = os.environ.get(
                'GRASPNET_BASELINE_DIR',
                os.path.join(ROOT_DIR, '..', 'graspnet-baseline'),
            )
            if os.path.exists(graspnet_baseline):
                # 首先添加我们创建的 pointnet2 目录，这样可以找到 pointnet2._ext
                invoke_pointnet2_dir = os.path.join(ROOT_DIR, '..', '..', 'pointnet2')
                if os.path.exists(invoke_pointnet2_dir):
                    sys.path.insert(0, invoke_pointnet2_dir)
                    logger.info(f"[GraspNet] ✅ 添加 invoke3/pointnet2 到 sys.path 最前面")

                # 然后添加 pointnet2/pointnet2 目录，这样可以直接导入 _ext
                sys.path.append(os.path.join(graspnet_baseline, 'pointnet2', 'pointnet2'))
                # 然后添加其他路径
                sys.path.append(graspnet_baseline)
                sys.path.append(os.path.join(graspnet_baseline, 'models'))
                sys.path.append(os.path.join(graspnet_baseline, 'dataset'))
                sys.path.append(os.path.join(graspnet_baseline, 'utils'))
                sys.path.append(os.path.join(graspnet_baseline, 'pointnet2'))
                # 添加 graspnetAPI 目录，这样可以找到 graspnetAPI 模块
                sys.path.append(os.path.join(graspnet_baseline, 'graspnetAPI'))
                logger.info(f"[GraspNet] ✅ 找到 graspnet-baseline: {graspnet_baseline}")
                logger.info(f"[GraspNet] ✅ 添加 pointnet2/pointnet2 到 sys.path")
                logger.info(f"[GraspNet] ✅ 添加 graspnetAPI 到 sys.path")
            else:
                logger.error(f"[GraspNet] ❌ graspnet-baseline 不存在: {graspnet_baseline}")
                return False

            try:
                # 注册自定义导入处理器，用于处理 pointnet2._ext 的导入
                class ExtImporter:
                    def find_module(self, fullname, path=None):
                        if fullname == 'pointnet2._ext':
                            return self
                        return None

                    def load_module(self, fullname):
                        if fullname == 'pointnet2._ext':
                            # 尝试直接导入 _ext 模块
                            import _ext
                            # 将 _ext 模块添加到 pointnet2._ext 命名空间
                            sys.modules[fullname] = _ext
                            logger.info(f"[GraspNet] ✅ 成功将 _ext 模块重定向到 pointnet2._ext")
                            return _ext
                        raise ImportError(f"无法导入模块 {fullname}")

                # 添加自定义导入处理器到 sys.meta_path
                sys.meta_path.insert(0, ExtImporter())
                logger.info(f"[GraspNet] ✅ 注册自定义导入处理器")

                # 现在导入其他模块
                from models.graspnet import GraspNet, pred_decode
                from graspnetAPI import GraspGroup
                from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image
                from utils.collision_detector import ModelFreeCollisionDetector
                logger.info(f"[GraspNet] ✅ 成功导入所有 graspnet 模块")
            except ImportError as e:
                logger.error(f"[GraspNet] ❌ graspnet 模块导入失败: {e}")
                logger.error(f"[GraspNet] 请确保 graspnet-baseline 代码库已安装")
                logger.error(f"[GraspNet] 期望路径: {graspnet_baseline}")
                return False

            # 创建网络
            self.net = GraspNet(
                input_feature_dim=self.network_config.get('input_feature_dim', 0),
                num_view=self.network_config.get('num_view', 300),
                num_angle=self.network_config.get('num_angle', 12),
                num_depth=self.network_config.get('num_depth', 4),
                cylinder_radius=self.network_config.get('cylinder_radius', 0.05),
                hmin=self.network_config.get('hmin', -0.02),
                hmax_list=self.network_config.get('hmax_list', [0.01, 0.02, 0.03, 0.04]),
                is_training=False
            )

            self.net.to(self.device)

            # 加载权重
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.net.eval()

            logger.info("[GraspNet] ✅ 模型加载成功")
            return True

        except Exception as e:
            logger.error(f"[GraspNet] ❌ 加载模型失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def predict(self, rgb, depth, mask, cam_K) -> Dict[str, Any]:
        """
        执行抓取预测推理

        Args:
            rgb: RGB 图像 (H, W, 3), dtype=uint8
            depth: 深度图 (H, W), dtype=float32
            mask: 物体 mask (H, W), dtype=uint8
            cam_K: 相机内参矩阵 (3, 3)

        Returns:
            dict: 推理结果
                {
                    'success': bool,
                    'pose_matrix': (4, 4) 抓取位姿矩阵  # 核心输出
                }
        """
        logger.info("[GraspNet] 开始抓取预测推理...")
        start_time = time.time()
        times = {}

        try:
            # 添加 graspnet-baseline 路径（完整路径设置）
            graspnet_baseline = os.environ.get(
                'GRASPNET_BASELINE_DIR',
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'graspnet-baseline'),
            )
            if os.path.exists(graspnet_baseline):
                # 首先添加 pointnet2/pointnet2 目录到最前面，这样可以直接导入 _ext
                pointnet2_pointnet2_dir = os.path.join(graspnet_baseline, 'pointnet2', 'pointnet2')
                if pointnet2_pointnet2_dir not in sys.path:
                    sys.path.insert(0, pointnet2_pointnet2_dir)

                # 添加其他路径
                if graspnet_baseline not in sys.path:
                    sys.path.append(graspnet_baseline)
                models_dir = os.path.join(graspnet_baseline, 'models')
                if models_dir not in sys.path:
                    sys.path.append(models_dir)
                utils_dir = os.path.join(graspnet_baseline, 'utils')
                if utils_dir not in sys.path:
                    sys.path.append(utils_dir)
                pointnet2_dir = os.path.join(graspnet_baseline, 'pointnet2')
                if pointnet2_dir not in sys.path:
                    sys.path.append(pointnet2_dir)
                graspnet_api_dir = os.path.join(graspnet_baseline, 'graspnetAPI')
                if graspnet_api_dir not in sys.path:
                    sys.path.append(graspnet_api_dir)

                logger.info(f"[GraspNet] ✅ 添加 graspnet-baseline 路径到 sys.path")

                # 注册自定义导入处理器（如果尚未注册）
                ext_importer_exists = any(
                    hasattr(importer, 'load_module') and
                    'ExtImporter' in type(importer).__name__
                    for importer in sys.meta_path
                )
                if not ext_importer_exists:
                    class ExtImporter:
                        def find_module(self, fullname, path=None):
                            if fullname == 'pointnet2._ext':
                                return self
                            return None

                        def load_module(self, fullname):
                            if fullname == 'pointnet2._ext':
                                import _ext
                                sys.modules[fullname] = _ext
                                logger.info(f"[GraspNet] ✅ 成功将 _ext 模块重定向到 pointnet2._ext")
                                return _ext
                            raise ImportError(f"无法导入模块 {fullname}")

                    sys.meta_path.insert(0, ExtImporter())
                    logger.info(f"[GraspNet] ✅ 注册自定义导入处理器")
            else:
                logger.error(f"[GraspNet] ❌ graspnet-baseline 不存在: {graspnet_baseline}")
                return {'success': False, 'error': 'graspnet_baseline_not_found'}

            try:
                # 直接从文件导入，避免包导入问题
                import importlib.util

                # 导入 GraspNet
                graspnet_path = os.path.join(graspnet_baseline, 'models', 'graspnet.py')
                spec = importlib.util.spec_from_file_location('graspnet', graspnet_path)
                graspnet_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(graspnet_module)
                GraspNet = graspnet_module.GraspNet
                pred_decode = graspnet_module.pred_decode

                # 导入 GraspGroup
                from graspnetAPI import GraspGroup

                # 导入 data_utils
                data_utils_path = os.path.join(graspnet_baseline, 'utils', 'data_utils.py')
                spec = importlib.util.spec_from_file_location('data_utils', data_utils_path)
                data_utils_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(data_utils_module)
                CameraInfo = data_utils_module.CameraInfo
                create_point_cloud_from_depth_image = data_utils_module.create_point_cloud_from_depth_image

                # 导入 collision_detector
                collision_detector_path = os.path.join(graspnet_baseline, 'utils', 'collision_detector.py')
                spec = importlib.util.spec_from_file_location('collision_detector', collision_detector_path)
                collision_detector_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(collision_detector_module)
                ModelFreeCollisionDetector = collision_detector_module.ModelFreeCollisionDetector

                logger.info(f"[GraspNet] ✅ 成功导入所有 graspnet 模块")

                # 确保 self.net 被正确初始化
                if self.net is None:
                    # 创建网络
                    self.net = GraspNet(
                        input_feature_dim=self.network_config.get('input_feature_dim', 0),
                        num_view=self.network_config.get('num_view', 300),
                        num_angle=self.network_config.get('num_angle', 12),
                        num_depth=self.network_config.get('num_depth', 4),
                        cylinder_radius=self.network_config.get('cylinder_radius', 0.05),
                        hmin=self.network_config.get('hmin', -0.02),
                        hmax_list=self.network_config.get('hmax_list', [0.01, 0.02, 0.03, 0.04]),
                        is_training=False
                    )

                    self.net.to(self.device)

                    # 加载权重
                    checkpoint = torch.load(self.model_path, map_location=self.device)
                    self.net.load_state_dict(checkpoint['model_state_dict'])
                    self.net.eval()

                    logger.info("[GraspNet] ✅ 模型重新加载成功")
            except Exception as e:
                logger.error(f"[GraspNet] ❌ graspnet 模块导入失败: {e}")
                logger.error(f"[GraspNet] 请确保 graspnet-baseline 代码库已安装")
                logger.error(f"[GraspNet] 当前 sys.path: {sys.path[:5]}...")
                import traceback
                logger.error(traceback.format_exc())
                return {'success': False, 'error': 'graspnet_module_not_found'}

            # ============================================
            # 1. 准备点云数据（前处理）
            # ============================================
            preprocess_start = time.time()
            H, W = depth.shape

            # 转换RGB格式
            if isinstance(rgb, np.ndarray):
                color_float = rgb.astype(np.float32) / 255.0
            else:
                color_float = np.array(rgb, dtype=np.float32) / 255.0

            depth_float = depth.astype(np.float32)

            # 生成点云
            camera = CameraInfo(
                W, H,
                cam_K[0][0], cam_K[1][1],
                cam_K[0][2], cam_K[1][2],
                1.0
            )
            cloud = create_point_cloud_from_depth_image(depth_float, camera, organized=True)

            # 应用掩码
            min_depth = 0.01
            max_depth = 2.0
            valid_mask = (mask > 0) & (depth_float > min_depth) & (depth_float < max_depth)

            cloud_masked = cloud[valid_mask]
            color_masked = color_float[valid_mask]

            # 降低点云点数阈值，提高容错性
            min_points_threshold = 50
            if len(cloud_masked) < min_points_threshold:
                logger.error(f"[GraspNet] 点云点数太少: {len(cloud_masked)}")
                return {'success': False, 'error': 'insufficient_points'}
            elif len(cloud_masked) < 100:
                logger.warning(f"[GraspNet] 点云点数较少: {len(cloud_masked)}, 但仍尝试处理")

            # ============================================
            # 2. 采样点云
            # ============================================
            num_point = self.sampling_config.get('num_point', 3000)
            if len(cloud_masked) >= num_point:
                idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
            else:
                idxs1 = np.arange(len(cloud_masked))
                idxs2 = np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True)
                idxs = np.concatenate([idxs1, idxs2], axis=0)

            cloud_sampled = cloud_masked[idxs]

            # 创建Open3D点云（用于碰撞检测）
            cloud_o3d = o3d.geometry.PointCloud()
            cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
            cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
            times['preprocess'] = time.time() - preprocess_start
            logger.info(f"[GraspNet] ⏱️ 前处理时间: {times['preprocess']:.4f}s")

            # ============================================
            # 3. GraspNet推理
            # ============================================
            inference_start = time.time()
            cloud_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(self.device)
            end_points = {
                'point_clouds': cloud_tensor,
                'cloud_colors': color_masked[idxs]
            }

            with torch.no_grad():
                end_points = self.net(end_points)
                grasp_preds = pred_decode(end_points)

            gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())
            times['inference'] = time.time() - inference_start
            logger.info(f"[GraspNet] ⏱️ 推理时间: {times['inference']:.4f}s")
            logger.info(f"[GraspNet] 生成 {len(gg)} 个抓取候选")

            if len(gg) == 0:
                logger.error("[GraspNet] ❌ 没有生成任何抓取候选")
                return {'success': False, 'error': 'no_grasp_candidates'}

            # ============================================
            # 4. 碰撞检测
            # ============================================
            voxel_size = self.collision_config.get('voxel_size', 0.01)
            approach_dist = self.collision_config.get('approach_dist', 0.05)
            collision_thresh = self.collision_config.get('collision_thresh', 0.01)

            mfcdetector = ModelFreeCollisionDetector(
                np.asarray(cloud_o3d.points),
                voxel_size=voxel_size
            )
            collision_mask = mfcdetector.detect(gg, approach_dist=approach_dist, collision_thresh=collision_thresh)
            gg = gg[~collision_mask]
            logger.info(f"[GraspNet] 碰撞检测后剩余 {len(gg)} 个候选")

            if len(gg) == 0:
                logger.error("[GraspNet] ❌ 碰撞检测后没有有效的抓取候选")
                return {'success': False, 'error': 'no_valid_grasp_after_collision'}

            # ============================================
            # 5. NMS + 排序
            # ============================================
            gg.nms().sort_by_score()

            # ============================================
            # 6. 垂直角度筛选
            # ============================================
            vertical_threshold = self.detection_config.get('vertical_threshold', 30)
            angle_threshold = np.deg2rad(vertical_threshold)
            vertical = np.array([0, 0, 1])

            filtered = []
            for grasp in gg:
                approach_dir = grasp.rotation_matrix[:, 0]
                cos_angle = np.dot(approach_dir, vertical)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                if angle < angle_threshold:
                    filtered.append(grasp)

            if len(filtered) == 0:
                logger.error("[GraspNet] ❌ 没有满足垂直角度条件的抓取，且无备用候选")
                return {'success': False, 'error': 'no_valid_grasp_after_filter'}
            else:
                logger.info(f"[GraspNet] 垂直角度筛选后剩余 {len(filtered)} 个候选")

            # ============================================
            # 7. 选择最佳抓取（得分最高的）
            # ============================================
            if len(filtered) == 0:
                logger.error("[GraspNet] ❌ 筛选后没有有效的抓取候选")
                return {'success': False, 'error': 'no_valid_grasp_candidates'}

            filtered.sort(key=lambda g: g.score, reverse=True)
            best_grasp = filtered[0]

            logger.info(f"[GraspNet] ✅ 抓取预测完成")
            logger.info(f"   位置: {best_grasp.translation}")
            logger.info(f"   得分: {best_grasp.score:.4f}")
            logger.info(f"   宽度: {best_grasp.width:.4f}")

            # ============================================
            # 8. 构造4x4位姿矩阵（核心输出）
            # ============================================
            postprocess_start = time.time()
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = best_grasp.rotation_matrix
            pose_matrix[:3, 3] = best_grasp.translation

            logger.info(f"[GraspNet] ✅ 生成4x4位姿矩阵")
            logger.info(f"   矩阵形状: {pose_matrix.shape}")
            logger.info(f"   完整矩阵:")
            logger.info(f"   [[{pose_matrix[0,0]:.4f}, {pose_matrix[0,1]:.4f}, {pose_matrix[0,2]:.4f}, {pose_matrix[0,3]:.4f}]")
            logger.info(f"    [{pose_matrix[1,0]:.4f}, {pose_matrix[1,1]:.4f}, {pose_matrix[1,2]:.4f}, {pose_matrix[1,3]:.4f}]")
            logger.info(f"    [{pose_matrix[2,0]:.4f}, {pose_matrix[2,1]:.4f}, {pose_matrix[2,2]:.4f}, {pose_matrix[2,3]:.4f}]")
            logger.info(f"    [{pose_matrix[3,0]:.4f}, {pose_matrix[3,1]:.4f}, {pose_matrix[3,2]:.4f}, {pose_matrix[3,3]:.4f}]]")
            times['postprocess'] = time.time() - postprocess_start
            times['total'] = time.time() - start_time
            logger.info(f"[GraspNet] ⏱️ 后处理时间: {times['postprocess']:.4f}s")
            logger.info(f"[GraspNet] ⏱️ 总时间: {times['total']:.4f}s")

            # ============================================
            # 9. 保存点云结果
            # ============================================
            # 检查 debug 配置中的 save_visualization 设置
            save_visualization = False
            if hasattr(self, 'debug_config'):
                save_visualization = self.debug_config.get('save_visualization', False)
            elif hasattr(self, 'config'):
                save_visualization = self.config.get('debug', {}).get('save_visualization', False)
            # 同时检查 detection_config 以保持兼容性
            save_visualization = save_visualization or self.detection_config.get('save_visualization', False)

            if save_visualization:
                # 创建 debug 目录
                debug_dir = os.path.join(os.path.dirname(__file__), 'debug')
                os.makedirs(debug_dir, exist_ok=True)

                # 生成时间戳文件名
                timestamp = time.strftime('%Y%m%d_%H%M%S')

                # 保存点云
                try:
                    cloud_o3d = o3d.geometry.PointCloud()
                    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
                    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
                    cloud_path = os.path.join(debug_dir, f'cloud_{timestamp}.ply')
                    o3d.io.write_point_cloud(cloud_path, cloud_o3d)
                    logger.info(f"[GraspNet] ✅ 保存点云到: {cloud_path}")
                except Exception as e:
                    logger.warning(f"[GraspNet] ⚠️ 点云保存失败: {e}")

            result = {
                'success': True,
                'pose_matrix': pose_matrix,  # 4x4位姿矩阵（核心输出）
                'translation': best_grasp.translation,
                'rotation_matrix': best_grasp.rotation_matrix,
                'grasp_width': float(best_grasp.width),
                'grasp_score': float(best_grasp.score)
            }

            return result

        except Exception as e:
            logger.error(f"[GraspNet] ❌ 推理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}

    def predict_from_redis(self,
                          rgb_key: str,
                          depth_key: Optional[str] = None,
                          mask_key: Optional[str] = None,
                          cam_K: Optional[List] = None,
                          redis_client=None,
                          **kwargs) -> Dict[str, Any]:
        """
        从 Redis 读取数据并预测

        Args:
            rgb_key: RGB图像的Redis key
            depth_key: 深度图的Redis key（可选）
            mask_key: Mask的Redis key（可选）
            cam_K: 相机内参矩阵（可选）
            redis_client: Redis客户端

        Returns:
            推理结果
        """
        try:
            if redis_client is None:
                from utils.utils import connect_remote_redis
                redis_client = connect_remote_redis()

            # ============================================
            # 1. 读取RGB图像
            # ============================================
            rgb_data = redis_client.get(rgb_key)
            if not rgb_data:
                return {'success': False, 'error': 'rgb_not_found'}

            rgb_shape_key = f"{rgb_key}:shape"
            rgb_shape_bytes = redis_client.get(rgb_shape_key)
            if rgb_shape_bytes:
                rgb_shape = tuple(json.loads(rgb_shape_bytes))
                rgb_array = np.frombuffer(rgb_data, dtype=np.uint8).reshape(rgb_shape)
                logger.info(f"[GraspNet] ✅ RGB图像加载: {rgb_array.shape}")
            else:
                # 尝试使用cv2解码（处理编码的图像格式）
                try:
                    np_arr = np.frombuffer(rgb_data, np.uint8)
                    rgb_array = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if rgb_array is None:
                        logger.error("[GraspNet] ❌ 图像解码失败")
                        return {'success': False, 'error': 'image_decode_failed'}
                    logger.info(f"[GraspNet] ✅ RGB图像解码: {rgb_array.shape}")
                except Exception as e:
                    logger.error(f"[GraspNet] ❌ 图像处理失败: {e}")
                    return {'success': False, 'error': str(e)}

            # ============================================
            # 2. 读取深度图
            # ============================================
            if depth_key:
                depth_data = redis_client.get(depth_key)
                if depth_data:
                    depth_shape_key = f"{depth_key}:shape"
                    logger.info(f"[GraspNet] 深度图形状key: {depth_shape_key}")
                    depth_shape_bytes = redis_client.get(depth_shape_key)
                    if depth_shape_bytes:
                        depth_shape = tuple(json.loads(depth_shape_bytes))
                    else:
                        depth_shape = rgb_array.shape[:2]

                    # 尝试使用 uint16 格式（mm），如果失败再使用 float32
                    try:
                        depth_array = np.frombuffer(depth_data, dtype=np.uint16).reshape(depth_shape)
                        # 转换为米（除以 1000）
                        depth_array = depth_array.astype(np.float32) / 1000.0
                        logger.info(f"[GraspNet] ✅ 深度图加载 (uint16, mm): {depth_array.shape}")
                        logger.info(f"   深度范围: {depth_array.min():.3f}m - {depth_array.max():.3f}m")
                    except:
                        # 回退到 float32 格式
                        depth_array = np.frombuffer(depth_data, dtype=np.float32).reshape(depth_shape)
                        logger.info(f"[GraspNet] ✅ 深度图加载 (float32): {depth_array.shape}")
                else:
                    # 深度图不存在，直接报错
                    logger.error("[GraspNet] ❌ 深度图不存在")
                    return {'success': False, 'error': 'depth_not_found'}
            else:
                # 未提供深度图，直接报错
                logger.error("[GraspNet] ❌ 未提供深度图")
                return {'success': False, 'error': 'depth_not_provided'}

            # ============================================
            # 3. 读取mask
            # ============================================
            mask_array = None
            mask_source = None

            # 优先级1: kwargs 中的 masks 参数（来自 SAM）
            if 'masks' in kwargs:
                masks = kwargs.get('masks', [])
                logger.info(f"[GraspNet] Mask来源检测: kwargs.masks, 长度={len(masks) if isinstance(masks, list) else 1}, 类型={type(masks)}")
                if masks:
                    if isinstance(masks, str):
                        # SAM 返回的是 Redis mask_key，不是 RLE
                        logger.info(f"[GraspNet] ✅ 检测到 SAM 返回的 mask_key: {masks}")
                        mask_key = masks
                    else:
                        # 传统 RLE 格式处理
                        first_mask = masks[0] if isinstance(masks, list) and len(masks) > 0 else masks
                        if isinstance(first_mask, dict):
                            counts = first_mask.get('counts', [])
                            logger.info(f"[GraspNet] 第一个mask是字典, counts长度={len(counts)}, counts前10个={counts[:10]}")
                        else:
                            logger.info(f"[GraspNet] 第一个mask类型={type(first_mask)}")

                        if isinstance(masks, list) and len(masks) > 0:
                            try:
                                rle_mask = masks[0]

                                if HAS_PYCOCOTOOLS:
                                    rle = rle_mask.copy()
                                    if 'counts' in rle and isinstance(rle['counts'], str):
                                        pass
                                    if 'size' in rle:
                                        h, w = rle['size'][:2]
                                    else:
                                        h, w = rgb_array.shape[:2]
                                    rle = mask_utils.frPyObjects(rle, h, w)
                                    mask_array = mask_utils.decode(rle)
                                    logger.info(f"[GraspNet] ✅ 使用 pycocotools 解码 Mask: {mask_array.shape}, 前景像素: {mask_array.sum()}")
                                else:
                                    def rle_decode_fallback(rle, shape):
                                        if isinstance(rle, dict):
                                            counts = rle.get('counts', [])
                                            size = rle.get('size', shape)
                                            if isinstance(size, list) and len(size) == 2:
                                                shape = (size[0], size[1])

                                            if isinstance(counts, int):
                                                counts = [counts]

                                            mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)

                                            logger.info(f"[GraspNet] RLE解码(备用): counts前20个={counts[:20]}, total_elements={len(counts)}")

                                            current_pos = 0
                                            for i in range(0, len(counts), 2):
                                                if i >= len(counts):
                                                    break

                                                bg_count = counts[i]
                                                current_pos += bg_count

                                                if i + 1 < len(counts):
                                                    fg_count = counts[i + 1]
                                                    end = min(current_pos + fg_count, len(mask))
                                                    if current_pos < end:
                                                        mask[current_pos:end] = 1
                                                    current_pos = end

                                            return mask.reshape(shape)
                                        else:
                                            # 如果 rle 不是字典，返回全1 mask
                                            logger.warning(f"[GraspNet] ⚠️ RLE 格式错误，使用全1 mask")
                                            return np.ones(shape, dtype=np.uint8)

                                    mask_array = rle_decode_fallback(rle_mask, rgb_array.shape[:2])
                                    logger.info(f"[GraspNet] ✅ 使用备用方法解码 Mask: {mask_array.shape}, 前景像素: {mask_array.sum()}")
                                mask_source = "kwargs.masks (RLE)"
                            except Exception as e:
                                logger.error(f"[GraspNet] ❌ 解码masks失败: {e}")
                                import traceback
                                logger.error(traceback.format_exc())
                                mask_array = None

            # 优先级2: 从 mask_key 获取
            if mask_array is None and mask_key:
                mask_data = redis_client.get(mask_key)
                if mask_data:
                    mask_shape_key = f"{mask_key}:shape"
                    mask_shape_bytes = redis_client.get(mask_shape_key)
                    if mask_shape_bytes:
                        mask_shape = tuple(json.loads(mask_shape_bytes))
                    else:
                        mask_shape = rgb_array.shape[:2]

                    mask_array = np.frombuffer(mask_data, dtype=np.uint8).reshape(mask_shape)
                    logger.info(f"[GraspNet] ✅ Mask加载: {mask_array.shape}, 前景像素: {mask_array.sum()}")
                    mask_source = "redis.mask_key"
                else:
                    logger.warning("[GraspNet] ⚠️ Mask不存在")

            # 优先级3: 全1 mask
            if mask_array is None:
                logger.warning("[GraspNet] ⚠️ 未提供mask，使用全1 mask")
                mask_array = np.ones(rgb_array.shape[:2], dtype=np.uint8)
                mask_source = "fallback (全1)"

            logger.info(f"[GraspNet] 📌 最终使用Mask来源: {mask_source}")

            # ============================================
            # 4. 相机内参
            # ============================================
            if cam_K is None:
                logger.warning("[GraspNet] ⚠️ 未提供相机内参，使用默认值")
                H, W = rgb_array.shape[:2]
                fovy = np.pi / 4
                focal = H / (2.0 * np.tan(fovy / 2.0))
                cam_K = [
                    [focal, 0.0, W / 2.0],
                    [0.0, focal, H / 2.0],
                    [0.0, 0.0, 1.0]
                ]

            # ============================================
            # 5. 执行预测
            # ============================================
            result = self.predict(rgb_array, depth_array, mask_array, cam_K)

            return result

        except Exception as e:
            logger.error(f"[GraspNet] ❌ predict_from_redis 失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}


class GraspNetService(IModelService):
    """
    GraspNet 模型服务适配器

    实现 IModelService 接口，适配 GraspNetPredictor
    """

    def __init__(self):
        super().__init__()
        self.predictor = None

    def load_model(self, model_id: str, model_dir: str) -> bool:
        """
        加载 GraspNet 模型

        Args:
            model_id: 模型ID
            model_dir: 模型目录

        Returns:
            是否成功
        """
        try:
            config_path = os.path.join(model_dir, 'graspnet.yaml')
            if not os.path.exists(config_path):
                logger.error(f"[GraspNetService] 配置文件不存在: {config_path}")
                return False

            config = self.load_config(model_id, config_path)
            block_config = config.get('block', {})
            model_filename = block_config.get('model_filename', 'checkpoint-rs.tar')
            # 检查 models 子目录中是否存在模型文件
            model_path = os.path.join(model_dir, 'models', model_filename)
            if not os.path.exists(model_path):
                # 如果不存在，尝试直接在模型目录中查找
                model_path = os.path.join(model_dir, model_filename)
                if not os.path.exists(model_path):
                    logger.error(f"[GraspNetService] 模型文件不存在: {model_path}")
                    return False

            # 转换硬件类型为PyTorch可识别的设备类型
            hardware = block_config.get('hardware', 'cpu')
            device = hardware
            if hardware == 'gpu':
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            elif hardware == 'cpu':
                device = 'cpu'

            self.predictor = GraspNetPredictor()
            self.predictor.init(
                model_path=model_path,
                network_config=config.get('network', {}),
                sampling_config=config.get('sampling', {}),
                collision_config=config.get('collision', {}),
                detection_config=config.get('detection', {}),
                debug_config=config.get('debug', {}),
                device=device
            )

            success = self.predictor.load_model()
            if success:
                self._models[model_id] = self.predictor
                self._devices[model_id] = block_config.get('hardware', 'cpu')
                logger.info(f"[GraspNetService] ✅ 模型加载成功: {model_id}")
            return success

        except Exception as e:
            logger.error(f"[GraspNetService] ❌ 加载模型失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def inference(self, param: Dict[str, Any], cur_result_set: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行推理

        Args:
            param: 输入参数
            cur_result_set: 当前结果集（可选）

        Returns:
            推理结果
        """
        try:
            if self.predictor is None:
                return {'success': False, 'error': 'model_not_loaded'}

            input_data = param.copy()

            # 打印输入参数，以便调试
            logger.info(f"[GraspNetService] 输入参数: {input_data}")
            logger.info(f"[GraspNetService] cur_result_set: {cur_result_set}")

            # 兼容不同的参数名称
            if 'rgb_key' not in input_data:
                # 尝试从redis_key获取
                if 'redis_key' in input_data:
                    input_data['rgb_key'] = input_data['redis_key']
                # 尝试从image获取
                elif 'image' in input_data:
                    input_data['rgb_key'] = input_data['image']
                # 尝试从image_key获取
                elif 'image_key' in input_data:
                    input_data['rgb_key'] = input_data['image_key']
                # 尝试从cur_result_set获取
                elif cur_result_set:
                    prepare_input_result = cur_result_set.get('prepareInput_result', {})
                    if 'image_key' in prepare_input_result:
                        input_data['rgb_key'] = prepare_input_result['image_key']
                    elif 'redis_key' in prepare_input_result:
                        input_data['rgb_key'] = prepare_input_result['redis_key']

            if 'depth_key' not in input_data:
                if 'depth_map' in input_data:
                    input_data['depth_key'] = input_data['depth_map']
                elif cur_result_set:
                    prepare_input_result = cur_result_set.get('prepareInput_result', {})
                    if 'depth_key' in prepare_input_result:
                        input_data['depth_key'] = prepare_input_result['depth_key']

            logger.info(f"[GraspNetService] 最终输入参数: rgb_key={input_data.get('rgb_key')}, depth_key={input_data.get('depth_key')}")

            if isinstance(input_data, dict):
                return self.predictor.predict_from_redis(**input_data)
            elif isinstance(input_data, (list, tuple)) and len(input_data) >= 3:
                rgb, depth, mask = input_data[:3]
                cam_K = input_data[3] if len(input_data) > 3 else None
                return self.predictor.predict(rgb, depth, mask, cam_K)
            else:
                return {'success': False, 'error': 'invalid_input_format'}

        except Exception as e:
            logger.error(f"[GraspNetService] ❌ 推理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}
