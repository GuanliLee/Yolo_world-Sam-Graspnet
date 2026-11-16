#!/usr/bin/env bash
set -euo pipefail

ISAAC_CONTAINER="${ISAAC_CONTAINER:-isaac-sim-5.1.0}"
ISAAC_INPUT_DIR="${ISAAC_INPUT_DIR:-/home/market2/graspnet_input}"
HOST_INPUT_DIR="${HOST_INPUT_DIR:-/home/ligl/lgl/market2/graspnet_input}"
RESULT_DIR="${RESULT_DIR:-/home/ligl/manipulator_grasp/results}"
GRASPNET_DIR="${GRASPNET_DIR:-/home/ligl/manipulator_grasp/graspnet-baseline}"
GRASPNET_PY="${GRASPNET_PY:-/home/ligl/miniconda3/envs/graspnet/bin/python}"
CHECKPOINT="${CHECKPOINT:-logs/log_rs/checkpoint-rs.tar}"

mkdir -p "${RESULT_DIR}"

docker exec "${ISAAC_CONTAINER}" bash -lc \
  "/isaac-sim/python.sh /home/market2/scripts/export_isaac_rgbd_for_graspnet.py \
    --headless \
    --width 640 \
    --height 360 \
    --warmup-frames 20 \
    --output-dir ${ISAAC_INPUT_DIR}"

cd "${GRASPNET_DIR}"
"${GRASPNET_PY}" demo.py \
  --checkpoint_path "${CHECKPOINT}" \
  --data_dir "${HOST_INPUT_DIR}" \
  --no_vis \
  --save_grasps "${RESULT_DIR}/isaac_scene_grasps.npy"

"${GRASPNET_PY}" - <<'PY'
import json
import numpy as np
import scipy.io as scio

grasps_path = "/home/ligl/manipulator_grasp/results/isaac_scene_grasps.npy"
meta_path = "/home/ligl/lgl/market2/graspnet_input/meta.mat"
summary_path = "/home/ligl/manipulator_grasp/results/isaac_scene_top_grasp.json"

grasps = np.load(grasps_path)
meta = scio.loadmat(meta_path)
top = grasps[0]
camera_cv_to_world = meta["camera_cv_to_world"].astype(float)
rotation_camera = top[4:13].reshape(3, 3)
translation_camera = top[13:16]
rotation_world = camera_cv_to_world[:3, :3] @ rotation_camera
translation_world = camera_cv_to_world[:3, :3] @ translation_camera + camera_cv_to_world[:3, 3]

summary = {
    "num_grasps": int(len(grasps)),
    "score": float(top[0]),
    "width": float(top[1]),
    "height": float(top[2]),
    "depth": float(top[3]),
    "translation_camera_xyz_m": translation_camera.tolist(),
    "translation_world_xyz_m": translation_world.tolist(),
    "rotation_camera_3x3": rotation_camera.tolist(),
    "rotation_world_3x3": rotation_world.tolist(),
    "camera_cv_to_world": camera_cv_to_world.tolist(),
}

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps({
    "saved": summary_path,
    "num_grasps": summary["num_grasps"],
    "score": summary["score"],
    "translation_world_xyz_m": summary["translation_world_xyz_m"],
}, indent=2))
PY
