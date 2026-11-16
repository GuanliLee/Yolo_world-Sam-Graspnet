# YOLO-World + SAM + GraspNet in Isaac Sim

开放词汇目标检测、提示式分割和 RGB-D 抓取位姿估计的仿真项目代码归档。代码来自 5090 上的 `model_config` 和 Isaac Sim 场景脚本。

![Isaac Sim 超市货架与购物车场景的相机截图](docs/images/isaac_sim_capture_20261116.png)

上图于 2026-11-16 在 5090 上使用 Isaac Sim 5.1.0、项目原始 USD 场景和 `/World/Robot/lifting_link/head_view_camera` 重新渲染，分辨率为 1280×720。这是未经算法叠加的 RGB 场景图；它不表示目标检测、分割或抓取成功。

## 流程

```mermaid
flowchart LR
    A[Isaac Sim RGB-D 与相机内参] --> B[YOLO-World 开放词汇检测]
    B -->|检测框| C[SAM 分割]
    C -->|目标 mask| D[GraspNet 抓取估计]
    A --> D
    D --> E[相机系抓取位姿 / 世界系转换]
```

`graspnet_pipeline/model_compose.yaml` 描述 YOLO-World → SAM → GraspNet 的串行 DAG。模型适配代码在 `yolo_world/`、`sam_segment/`、`graspnet/`。`examples/isaac_sim/` 包含 RGB-D 导出、SAM 框提示、结果绘图和基线推理脚本。原始环境笔记保存在 `docs/original_model_config_README.md`，其中的机器路径是历史记录。

## 运行前提

这是一份研究环境代码归档，**尚未整理成一键运行的独立发行版**。模型服务依赖原项目的 `invoke`/`base.base_model_service` 编排接口、Redis 连接工具，以及 PyTorch、Ultralytics、Segment Anything、GraspNet baseline、graspnetAPI、Open3D、OpenCV 和 Isaac Sim。请按各上游项目的安装说明配置与版本匹配的环境。

模型权重、第三方源码、仿真资产、点云和运行结果未纳入仓库。运行前需要自行获取对应权重并放到 YAML 指定的位置；GraspNet baseline 路径可通过 `GRASPNET_BASELINE_DIR` 设置。Redis 连接仍需由原编排环境的 `utils.utils.connect_remote_redis` 提供。`examples/isaac_sim/` 中的默认路径来自原 5090 环境，换机器运行时请调整路径、容器名和脚本参数。

## 验证范围

本仓库的 Python 文件通过语法编译检查，YAML 可解析。仿真场景 RGB-D 截图已在 5090 的 Isaac Sim 5.1.0 中重新生成。没有在干净环境中完成 YOLO-World、SAM、GraspNet 联合推理或机器人抓取验证。
