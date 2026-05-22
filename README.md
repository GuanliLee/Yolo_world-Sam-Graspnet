# YOLO-World + SAM + GraspNet

A vision-driven robotic grasping pipeline that combines open-vocabulary object detection, instance segmentation, and grasp pose estimation.

## Pipeline Overview

```
RGB-D Image
    │
    ▼
┌───────────┐    ┌─────────────┐    ┌──────────┐
│ YOLO-World │───▶│ SAM Segment │───▶│ GraspNet │───▶ Grasp Pose
│ Detection  │    │ Segmentation│    │ Inference│
└───────────┘    └─────────────┘    └──────────┘
```

1. **YOLO-World** - Open-vocabulary object detection based on text prompts (e.g., `cup, box`)
2. **SAM Segment** - Segment Anything Model for precise object mask extraction
3. **GraspNet** - Grasp pose estimation from point cloud with collision detection

## Project Structure

```
├── yolo_world/               # YOLO-World open-vocabulary detection
│   ├── yolo_world.py         # Detection & 3D localization model
│   ├── yolo_world.yaml       # Detection config (confidence, NMS, etc.)
│   └── model/                # Model weights (yolov8x-worldv2.pt)
│
├── sam_segment/              # SAM instance segmentation
│   ├── sam_segment.py        # Segmentation model
│   ├── sam_segment.yaml      # Segmentation config (model type, area filter)
│   └── model/                # Model weights (sam_b.pt)
│
└── graspnet/                 # GraspNet grasp pose estimation
    ├── graspnet.py           # Grasp prediction model
    ├── graspnet.yaml         # Grasp config (top_k, point cloud, collision)
    ├── graspnet-baseline/    # GraspNet baseline network
    │   └── demo.py           # Inference demo
    ├── transform_camera_pose_to_base/  # Camera-to-base coordinate transform
    └── model/                # Model checkpoint (checkpoint-rs.tar)
```

## Dependencies

- Python 3.10
- PyTorch (CUDA)
- ultralytics
- OpenCV
- Open3D
- NumPy

## Configuration

Each module has its own YAML config file for tuning parameters:

| Module | Config | Key Parameters |
|--------|--------|---------------|
| YOLO-World | `yolo_world/yolo_world.yaml` | `conf_threshold`, `iou_threshold`, `imgsz` |
| SAM Segment | `sam_segment/sam_segment.yaml` | `model_type`, `min_area_size` |
| GraspNet | `graspnet/graspnet.yaml` | `top_k`, `num_point`, `collision_thresh` |

## Usage

Each module exposes a standard interface:

```python
model = Model()

# Step 1: Load model to GPU
model.load({"gpu_id": 0})

# Step 2: Run inference
result = model.execute_detect({
    "image": "image_id",
    "text_query": "cup,box",
    ...
})
```

## License

This project is for research and educational purposes.
