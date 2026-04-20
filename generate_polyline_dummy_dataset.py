import os
import cv2
import json
import numpy as np
from pathlib import Path

# ── Config (must match your model's CONFIG) ──────────────────────────────────
IMG_H = 512
IMG_W = 512
NUM_LANES = 8
NUM_TRAIN = 32   # images for training split
NUM_TEST  = 8    # images for test/val split

OUTPUT_DIR = Path(__file__).parent / "dummy_dataset"
TRAIN_IMG_DIR   = OUTPUT_DIR / "train_image"
TEST_IMG_DIR    = OUTPUT_DIR / "test_image"
TRAIN_LABEL     = OUTPUT_DIR / "train_image_label.json"
TEST_LABEL      = OUTPUT_DIR / "test_image_label.json"

# ────────────────────────────────────────────────────────────────────────────

def make_dirs():
    TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
    TEST_IMG_DIR.mkdir(parents=True, exist_ok=True)


def random_lane_points(lane_idx: int, num_lanes: int) -> list[list[int]]:
    """
    Generate a realistic curved lane as a list of [x, y] points.
    Each lane is spaced evenly across the image width with some random curve.
    Returns ~20 anchor points from bottom to top of the image.
    """
    # Base x position at the bottom of the image for this lane
    section_w = IMG_W / num_lanes
    base_x = section_w * lane_idx + section_w / 2

    # Random lateral drift and curvature
    drift = np.random.uniform(-section_w * 0.3, section_w * 0.3)
    curve = np.random.uniform(-0.0003, 0.0003)   # quadratic bend

    num_pts = 20
    ys = np.linspace(IMG_H - 1, IMG_H // 4, num_pts).astype(int)
    points = []
    for y in ys:
        # Quadratic curve: x shifts as y goes up
        dy = IMG_H - y
        x = base_x + drift + curve * (dy ** 2)
        x = int(np.clip(x, 5, IMG_W - 5))
        points.append([x, int(y)])
    return points


def draw_lane_on_image(img: np.ndarray, points: list[list[int]]) -> np.ndarray:
    """Draw a lane line on the image so it looks non-trivial."""
    pts = np.array(points, dtype=np.int32)
    color = (
        np.random.randint(180, 255),
        np.random.randint(180, 255),
        np.random.randint(180, 255),
    )
    for i in range(len(pts) - 1):
        cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), color, thickness=3)
    return img


def make_background() -> np.ndarray:
    """Create a simple road-like background (dark grey + noise)."""
    img = np.full((IMG_H, IMG_W, 3), fill_value=60, dtype=np.uint8)
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def generate_sample(image_name: str, img_dir: Path) -> dict:
    """Generate one image + label dict."""
    img = make_background()

    # Randomly pick which lanes are present (at least 1, at most NUM_LANES)
    num_active = np.random.randint(1, NUM_LANES + 1)
    active_lane_ids = np.random.choice(NUM_LANES, size=num_active, replace=False)

    lanes = {}
    for lane_idx in active_lane_ids:
        points = random_lane_points(int(lane_idx), NUM_LANES)
        lanes[f"lane_{lane_idx}"] = points
        img = draw_lane_on_image(img, points)

    cv2.imwrite(str(img_dir / image_name), img)

    return {
        "image": image_name,
        "lanes": lanes
    }


def generate_split(num_samples: int, img_dir: Path, label_path: Path, prefix: str):
    records = []
    for i in range(num_samples):
        name = f"{prefix}_{i:04d}.jpg"
        record = generate_sample(name, img_dir)
        records.append(record)
        if (i + 1) % 10 == 0 or (i + 1) == num_samples:
            print(f"  [{prefix}] {i+1}/{num_samples} done")

    with open(label_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  Saved labels → {label_path}")


def main():
    print("Generating dummy LaneNet dataset...")
    make_dirs()

    print(f"\nTraining split  ({NUM_TRAIN} images):")
    generate_split(NUM_TRAIN, TRAIN_IMG_DIR, TRAIN_LABEL, "train")

    print(f"\nTest/Val split  ({NUM_TEST} images):")
    generate_split(NUM_TEST, TEST_IMG_DIR, TEST_LABEL, "test")

    print(f"""
Done!  Dataset written to: {OUTPUT_DIR}

Directory layout
────────────────
dummy_dataset/
  train_image/           ← {NUM_TRAIN} .jpg images
  train_image_label.json ← label file for training
  test_image/            ← {NUM_TEST}  .jpg images
  test_image_label.json  ← label file for val/test

Label format (per entry)
────────────────────────
{{
  "image": "train_0000.jpg",
  "lanes": {{
    "lane_0": [[x0,y0], [x1,y1], ...],   # 20 points per lane
    "lane_3": [[x0,y0], ...],
    ...   (only active lanes present)
  }}
}}

Update your CONFIG to point here:
  "train_image_folder": "dummy_dataset/train_image",
  "train_label_json":   "dummy_dataset/train_image_label.json",
  "val_image_folder":   "dummy_dataset/test_image",
  "val_label_json":     "dummy_dataset/test_image_label.json",
""")


if __name__ == "__main__":
    main()
