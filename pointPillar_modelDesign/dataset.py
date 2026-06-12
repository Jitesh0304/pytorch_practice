import open3d as o3d
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from annotation_parser import parse_annotation_file

class PointCloudDataset(Dataset):
    def __init__(self, root_dir, split="train", max_points=16384):
        """
        root_dir  : path to dataset/ folder
        split     : "train" or "val"
        max_points: subsample to this many points per scene
        """
        self.root      = Path(root_dir)
        self.max_pts   = max_points

        # Read the list of scene IDs (e.g. "scene_001")
        split_file = self.root / "splits" / f"{split}.txt"
        self.scene_ids = split_file.read_text().strip().splitlines()

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, idx):
        scene_id = self.scene_ids[idx]

        # ── 1. Load point cloud ──────────────────────────────────
        pcd_path = self.root / "pointclouds" / f"{scene_id}.pcd"
        pcd      = o3d.io.read_point_cloud(str(pcd_path))
        points   = np.asarray(pcd.points, dtype=np.float32)  # (921600, 3)

        # ── 2. Subsample (random, for speed) ─────────────────────
        if len(points) > self.max_pts:
            idx_pts = np.random.choice(len(points), self.max_pts, replace=False)
            points  = points[idx_pts]

        # ── 3. Load annotation ───────────────────────────────────
        ann_path = self.root / "annotations" / f"{scene_id}.json"
        ann      = parse_annotation_file(ann_path)

        # ── 4. Return as tensors ─────────────────────────────────
        return {
            "points":  torch.from_numpy(points),        # (N, 3)
            "boxes":   torch.from_numpy(ann["boxes"]),  # (num_obj, 7)
            "labels":  torch.from_numpy(ann["labels"]), # (num_obj,)
            "scene_id": scene_id
        }
    


