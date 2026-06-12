"""
Your LiDAR gives you a cloud of (X, Y, Z) points — just dots floating in 3D space. PointPillars converts these dots into vertical 
columns (pillars) standing on the ground, turns them into a 2D image-like grid, then uses a standard image detection network to 
find objects. Think of it like looking down at the scene from above.

  Top-down view of the ground plane:
  ┌──────────────────────────────────┐
  │  · · ·  [car pillar]  · · · · ·· │
  │  · · · ┌─────┐ · · · · · · · · · │
  │  · · · │█████│ · · [tree]· · · · │
  │  · · · │█████│ · · ┌──┐ · · · ·  │
  │  · · · └─────┘ · · │██│ · · · ·  │
  │  · · · · · · · · · └──┘ · · · ·  │
  └──────────────────────────────────┘

  Each pillar = all LiDAR points that fall
  within a (0.16m × 0.16m) grid cell.
  The model learns features from each pillar.

You have 921,600 points → after pillaring, you'll have ~10,000–16,000 non-empty pillars, each with up to 32 points. This is how 
PointPillars handles your huge point cloud efficiently.


Runs in real-time (~60Hz) because it avoids 3D convolutions
State-of-the-art on KITTI benchmark when it was published
3 parts: pillar encoder → backbone → detection head
Outputs 2D BEV boxes, great for self-driving/robotics   (Bird's Eye View)


We need to organise our files, parse the annotations into standard 7-DOF bounding boxes [cx, cy, cz, length, width, height, yaw], 
and write a PyTorch Dataset class that returns one sample at a time.

  dataset/
  ├── pointclouds/
  │   ├── scene_001.pcd
  │   ├── scene_002.pcd
  │   └── ...
  ├── annotations/
  │   ├── scene_001.json     ← list of objects in scene
  │   ├── scene_002.json
  │   └── ...
  └── splits/
      ├── train.txt          ← "scene_001\nscene_002\n..."
      └── val.txt
    
"""


import json
import numpy as np

def parse_annotation_file(json_path):
    """
    Load a JSON annotation file and return a list of 7-DOF boxes.
    Each box = [cx, cy, cz, length, width, height, yaw]
    """
    with open(json_path) as f:
        # JSON file is a LIST of objects in the scene
        objects = json.load(f)

    boxes  = []  # shape will be (N, 7)
    labels = []  # shape will be (N,)  — integer class id

    # Map class names to integers
    CLASS_MAP = {"car": 0, "pedestrian": 1, "cyclist": 2}

    for obj in objects:
        c = obj["contour"]

        cx  = c["center3D"]["x"]
        cy  = c["center3D"]["y"]
        cz  = c["center3D"]["z"]
        l   = c["size3D"]["x"]   # length (along x)
        w   = c["size3D"]["y"]   # width  (along y)
        h   = c["size3D"]["z"]   # height (along z)
        yaw = c["rotation3D"]["z"] # yaw = rotation around Z axis

        boxes.append([cx, cy, cz, l, w, h, yaw])
        labels.append(CLASS_MAP.get(obj.get("class", "car"), 0))

    return {
        "boxes":  np.array(boxes,  dtype=np.float32),   # (N, 7)
        "labels": np.array(labels, dtype=np.int64),     # (N,)
    }

