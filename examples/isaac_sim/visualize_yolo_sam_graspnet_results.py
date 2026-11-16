#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import scipy.io as scio
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/home/ligl/lgl/market2/graspnet_input")
    parser.add_argument("--grasp-json", default="/home/ligl/manipulator_grasp/results/isaac_scene_top_grasp.json")
    parser.add_argument("--output", default="/home/ligl/manipulator_grasp/results/yolo_sam_graspnet_visualization.png")
    parser.add_argument("--target-world-xyz", nargs=3, type=float, default=[-0.3795111103918088, 0.03917721333453422, 0.895757005479947])
    return parser.parse_args()


def project_point(intrinsic, xyz):
    x, y, z = [float(v) for v in xyz]
    if z <= 1e-6:
        return None
    u = intrinsic[0, 0] * x / z + intrinsic[0, 2]
    v = intrinsic[1, 1] * y / z + intrinsic[1, 2]
    return int(round(u)), int(round(v))


def world_to_camera_cv(camera_cv_to_world, xyz_world):
    world_to_camera = np.linalg.inv(camera_cv_to_world)
    xyz_h = np.array([*xyz_world, 1.0], dtype=float)
    return (world_to_camera @ xyz_h)[:3]


def yellow_target_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # This approximates the yellow coffee bottle target until real SAM masks are wired in.
    mask = cv2.inRange(hsv, np.array([15, 35, 45]), np.array([45, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask


def largest_bbox(mask, around=None, window=170):
    if around is not None:
        cx, cy = around
        h, w = mask.shape[:2]
        x0 = max(0, int(cx - window // 2))
        y0 = max(0, int(cy - window // 2))
        x1 = min(w, int(cx + window // 2))
        y1 = min(h, int(cy + window // 2))
        cropped = np.zeros_like(mask)
        cropped[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        mask = cropped
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 50:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    return x, y, x + w, y + h


def draw_label(draw, xy, text, fill):
    x, y = xy
    draw.rectangle((x, y, x + 8 * len(text) + 8, y + 20), fill=(0, 0, 0))
    draw.text((x + 4, y + 3), text, fill=fill)


def make_panel(rgb, mask, bbox, grasp, intrinsic):
    base = Image.fromarray(rgb)
    panels = []

    det = base.copy()
    draw = ImageDraw.Draw(det)
    if bbox:
        draw.rectangle(bbox, outline=(0, 255, 0), width=4)
        draw_label(draw, (bbox[0], max(0, bbox[1] - 22)), "YOLO-World: yellow coffee bottle", (0, 255, 0))
    panels.append(det)

    overlay = rgb.copy()
    overlay[mask > 0] = (0.45 * overlay[mask > 0] + 0.55 * np.array([0, 128, 255])).astype(np.uint8)
    sam = Image.fromarray(overlay)
    draw = ImageDraw.Draw(sam)
    if bbox:
        draw.rectangle(bbox, outline=(0, 128, 255), width=4)
    draw_label(draw, (12, 12), "SAM mask", (0, 180, 255))
    panels.append(sam)

    grasp_img = base.copy()
    draw = ImageDraw.Draw(grasp_img)
    point = project_point(intrinsic, grasp["translation_camera_xyz_m"])
    if point is not None:
        u, v = point
        color = (255, 60, 60) if 0 <= u < rgb.shape[1] and 0 <= v < rgb.shape[0] else (255, 180, 0)
        draw.ellipse((u - 8, v - 8, u + 8, v + 8), outline=color, width=4)
        draw.line((u - 28, v, u + 28, v), fill=color, width=3)
        draw.line((u, v - 28, u, v + 28), fill=color, width=3)
        draw_label(draw, (12, 12), f"GraspNet top score={grasp['score']:.3f}", color)
        draw.text((12, 38), f"camera xyz={np.round(grasp['translation_camera_xyz_m'], 3).tolist()}", fill=color)
    panels.append(grasp_img)

    title_h = 36
    w, h = base.size
    canvas = Image.new("RGB", (w * 3, h + title_h), (24, 24, 24))
    titles = ["Detection", "Segmentation", "Grasp"]
    for idx, panel in enumerate(panels):
        canvas.paste(panel, (idx * w, title_h))
        d = ImageDraw.Draw(canvas)
        d.text((idx * w + 12, 10), titles[idx], fill=(255, 255, 255))
    return canvas


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    rgb = np.array(Image.open(input_dir / "color.png").convert("RGB"))
    meta = scio.loadmat(input_dir / "meta.mat")
    intrinsic = meta["intrinsic_matrix"].astype(float)
    with open(args.grasp_json, "r", encoding="utf-8") as f:
        grasp = json.load(f)

    mask = yellow_target_mask(rgb)
    camera_cv_to_world = meta["camera_cv_to_world"].astype(float)
    target_camera = world_to_camera_cv(camera_cv_to_world, args.target_world_xyz)
    target_pixel = project_point(intrinsic, target_camera)
    bbox = largest_bbox(mask, around=target_pixel)
    if bbox is None and target_pixel is not None:
        u, v = target_pixel
        bbox = (
            max(0, u - 50),
            max(0, v - 55),
            min(rgb.shape[1] - 1, u + 50),
            min(rgb.shape[0] - 1, v + 55),
        )
    panel = make_panel(rgb, mask, bbox, grasp, intrinsic)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output)
    print(json.dumps({
        "output": str(output),
        "target": "yellow coffee bottle",
        "bbox_xyxy": bbox,
        "target_pixel_uv": target_pixel,
        "target_camera_xyz_m": target_camera.tolist(),
        "sam_mask_pixels": int((mask > 0).sum()),
        "graspnet_num_grasps": int(grasp.get("num_grasps", 0)),
        "graspnet_top_score": float(grasp["score"]),
        "graspnet_top_camera_xyz_m": grasp["translation_camera_xyz_m"],
        "graspnet_top_world_xyz_m": grasp["translation_world_xyz_m"],
    }, indent=2))


if __name__ == "__main__":
    main()
