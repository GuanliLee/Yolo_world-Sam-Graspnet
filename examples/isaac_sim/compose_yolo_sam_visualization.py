#!/usr/bin/env python3
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compose the RGB, YOLOE and SAM outputs into one review panel."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--yolo", required=True)
    parser.add_argument("--sam-overlay", required=True)
    parser.add_argument("--sam-mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-label", default="AD Calcium Milk")
    parser.add_argument("--detector-label", default="YOLOE")
    parser.add_argument("--camera-label", default="Left wrist")
    return parser.parse_args()


def fit_image(path, size, binary=False):
    image = Image.open(path).convert("L" if binary else "RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS if not binary else Image.Resampling.NEAREST)
    canvas = Image.new("RGB", size, (24, 24, 24))
    if binary:
        image = image.convert("RGB")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    tile_size = (640, 360)
    title_height = 42
    margin = 12
    panel = Image.new(
        "RGB",
        (tile_size[0] * 2 + margin * 3, (tile_size[1] + title_height) * 2 + margin * 3),
        (16, 16, 16),
    )
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    tiles = (
        (args.image, f"1. {args.camera_label} RGB after navigation", False),
        (args.yolo, f"2. {args.detector_label} selected {args.target_label}", False),
        (args.sam_overlay, f"3. SAM segmentation | {args.target_label}", False),
        (args.sam_mask, "4. SAM binary mask used by GraspNet", True),
    )

    for index, (path, title, binary) in enumerate(tiles):
        row, col = divmod(index, 2)
        x = margin + col * (tile_size[0] + margin)
        y = margin + row * (tile_size[1] + title_height + margin)
        draw.text((x + 8, y + 14), title, fill=(238, 238, 238), font=font)
        panel.paste(fit_image(path, tile_size, binary=binary), (x, y + title_height))

    panel.save(output)
    print(f"saved YOLOE/SAM visualization: {output}")


if __name__ == "__main__":
    main()
