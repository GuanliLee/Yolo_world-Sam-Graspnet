# GraspNet 抓取预测流程测试指南

## 📋 目录结构

```
/home/ligl/ligl/invoke3/invoke/model_config/
├── yolo_world/                    # YOLO-World 开放词汇检测
│   ├── config_yolo_world.py
│   ├── model_inference.py
│   └── yolo_world.yaml
│
├── sam_segment/                   # SAM 分割模型
│   ├── config_sam_segment.py
│   ├── model_inference.py
│   └── sam_segment.yaml
│
├── graspnet/                      # GraspNet 抓取预测
│   ├── config_graspnet.py
│   ├── model_inference.py
│   └── graspnet.yaml
│
├── graspnet_pipeline/              # 模型编排配置
│   └── model_compose.yaml
│
├── test_graspnet_pipeline.py       # 完整流程测试脚本
├── test_single_models.py          # 单模型测试脚本
└── README.md                      # 本文档
```

## 🎯 测试流程

### 完整流程：Redis RGB图 + 深度图 → YOLO-World检测 → SAM分割 → GraspNet推理 → 返回4x4矩阵

```
┌─────────────┐
│  Redis     │
│  RGB+Depth │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ YOLO-World │  检测目标 bbox
└─────┬───────┘
      │
      ▼
┌─────────────┐
│    SAM     │  根据 bbox 生成 mask
└─────┬───────┘
      │
      ▼
┌─────────────┐
│  GraspNet  │  使用 RGB-D + mask → 4x4 矩阵
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ 4x4 矩阵  │  抓取位姿
└─────────────┘
```

## 🚀 使用方法

### 1. 测试完整流程

```bash
cd /home/ligl/ligl/invoke3/invoke/model_config
python test_graspnet_pipeline.py
```

**测试步骤：**
1. 创建测试 RGB 图像和深度图
2. 上传图像到 Redis
3. 执行 YOLO-World 检测
4. 执行 SAM 分割
5. 执行 GraspNet 推理
6. 返回 4x4 抓取位姿矩阵

### 2. 测试单个模型

#### 测试 YOLO-World
```bash
python test_single_models.py --model yolo_world
```

#### 测试 SAM 分割
```bash
python test_single_models.py --model sam_segment
```

#### 测试 GraspNet
```bash
python test_single_models.py --model graspnet
```

## 📝 配置说明

### YOLO-World 配置 (yolo_world.yaml)

```yaml
detection:
  conf_threshold: 0.25    # 置信度阈值
  iou_threshold: 0.45     # NMS IOU 阈值
  max_detections: 100     # 最大检测数量
  imgsz: 640              # 推理图像尺寸

block:
  model_filename: "models/yolov8s-world.pt"  # 模型文件路径
```

### SAM 配置 (sam_segment.yaml)

```yaml
segmentation:
  sort_by: area              # 排序方式
  min_area_size: 200         # 最小区域面积

block:
  model_filename: "models/sam_vit_b_01ec64.pth"  # 模型文件路径
```

### GraspNet 配置 (graspnet.yaml)

```yaml
detection:
  top_k: 20                    # 保留前K个候选
  vertical_threshold: 30       # 垂直角度阈值(度)
  num_point: 3000              # 点云采样数量

block:
  model_filename: "models/checkpoint-rs.tar"  # 模型文件路径
```

## 🔧 准备模型文件

在运行测试之前，需要准备以下模型文件：

### 1. YOLO-World 模型

```bash
cd /home/ligl/ligl/invoke3/invoke/model_config/yolo_world
mkdir -p models
# 下载模型文件到 models 目录
# 例如: wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-world.pt
```

### 2. SAM 模型

```bash
cd /home/ligl/ligl/invoke3/invoke/model_config/sam_segment
mkdir -p models
# 下载模型文件到 models 目录
# 例如: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### 3. GraspNet 模型

```bash
cd /home/ligl/ligl/invoke3/invoke/model_config/graspnet
mkdir -p models
# 下载模型文件到 models 目录
# 例如: cp /path/to/checkpoint-rs.tar models/
```

## 📊 测试输出示例

### 完整流程测试输出

```
================================================================================
🚀 开始测试 GraspNet 完整流程
================================================================================
📡 连接 Redis...
✅ Redis 连接成功
🎨 创建测试图像...
   RGB Shape: (480, 640, 3)
   Depth Shape: (480, 640)
   Camera K: [[480.0, 0.0, 320.0], [0.0, 480.0, 240.0], [0.0, 0.0, 1.0]]
================================================================================
📤 上传测试图像到 Redis
   Data ID: graspnet_test_1765791212696
   RGB Shape: (480, 640, 3)
   Depth Shape: (480, 640)
   ✅ RGB 图像上传成功: graspnet_test_1765791212696:rgb
   ✅ 深度图上传成功: graspnet_test_1765791212696:depth
   ✅ 相机内参上传成功: graspnet_test_1765791212696:cam_K
================================================================================
🔍 步骤 1: YOLO-World 检测
================================================================================
✅ YOLO-World 检测完成
   检测到 1 个目标
================================================================================
✂️ 步骤 2: SAM 分割
================================================================================
   检测框数量: 1
✅ SAM 分割完成
   分割出 1 个 mask
================================================================================
🤖 步骤 3: GraspNet 抓取预测
================================================================================
✅ GraspNet 推理完成
================================================================================
🎯 GraspNet 推理结果:
================================================================================
✅ 成功返回 4x4 抓取位姿矩阵:
   [0.999, 0.001, 0.000, 0.320]
   [0.001, 0.999, 0.000, 0.240]
   [0.000, 0.000, 1.000, 0.500]
   [0.000, 0.000, 0.000, 1.000]
================================================================================
📊 测试结果总结:
================================================================================
   Data ID: graspnet_test_1765791212696
   YOLO-World 检测: 1 个目标
   SAM 分割: 1 个 mask
   GraspNet 推理: ✅ 成功
   返回 4x4 矩阵: ✅ 是
================================================================================
🧹 清理 Redis 数据完成
================================================================================
✅ 测试完成！
================================================================================
```

## ⚠️ 注意事项

1. **Redis 连接**: 确保 Redis 服务正在运行，并且配置正确
2. **模型文件**: 确保所有模型文件都已下载到正确的目录
3. **GPU 支持**: 建议使用 GPU 加速推理
4. **内存需求**: GraspNet 需要较多内存，建议至少 8GB RAM
5. **深度图格式**: 深度图应该是 float32 格式，单位为米

## 🐛 故障排除

### 问题 1: 模型加载失败

**错误信息**: `模型文件不存在`

**解决方案**:
- 检查模型文件路径是否正确
- 确保模型文件已下载到 `models/` 目录
- 检查文件权限

### 问题 2: Redis 连接失败

**错误信息**: `Redis 连接失败`

**解决方案**:
- 检查 Redis 服务是否运行: `redis-cli ping`
- 检查 Redis 配置文件
- 检查网络连接

### 问题 3: GraspNet 推理返回空结果

**错误信息**: `推理结果中没有 pose_matrix`

**解决方案**:
- 检查深度图是否正确
- 检查 mask 是否有效
- 检查相机内参是否正确
- 检查点云数量是否足够

## 📚 参考资料

- [YOLO-World 文档](https://docs.ultralytics.com/models/yolo-world)
- [SAM 文档](https://docs.ultralytics.com/models/sam)
- [GraspNet 论文](https://arxiv.org/abs/1909.07110)

## 📞 联系方式

如有问题，请联系开发团队。
