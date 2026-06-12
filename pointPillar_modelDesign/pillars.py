"""
What is a DataLoader?
PyTorch's DataLoader wraps your Dataset and feeds it to the model in batches. It also handles shuffling, parallel loading with 
multiple CPU workers, and stacking samples into batched tensors. The tricky part here is that each scene has a different 
number of objects — we need a custom collate function to handle that.

  Point Cloud → Pillars
  Raw Points (N, 3)                          Pillar Features (P, 32, 9)
  ┌─────────────┐                            ┌─────────────────────────┐
  │ x  y  z     │                            │  For each of P pillars: │
  │ 1.2 3.4 0.1 │ ──→  Voxelize Grid ──→     │  up to 32 points, each  │
  │ 1.3 3.4 0.2 │       (0.16m cells)        │  with 9 features:       │
  │ ...         │                            │  x, y, z,               │
  └─────────────┘                            │  x-xc, y-yc, z-zc,      │
                                             │  xc, yc, zc             │
                                             └─────────────────────────┘
  (x-xc = offset from pillar center)

"""

# convert raw points into pillar format
import numpy as np

def create_pillars(points,
                    voxel_size=(0.16, 0.16),
                    point_range=(-50, -50, -5, 50, 50, 3),
                    max_pillars=12000,
                    max_pts_per_pillar=32):
    """
    points: numpy array (N, 3)  [x, y, z]
    Returns:
        pillars      : (max_pillars, max_pts_per_pillar, 9)
        pillar_coords: (max_pillars, 2)  [grid_x, grid_y]
        num_pillars  : int  (how many non-empty pillars found)
    """
    x_min, y_min, z_min, x_max, y_max, z_max = point_range
    vx, vy = voxel_size

    # ── 1. Filter points to range ────────────────────────────
    mask = (
        (points[:,0] >= x_min) & (points[:,0] < x_max) &
        (points[:,1] >= y_min) & (points[:,1] < y_max) &
        (points[:,2] >= z_min) & (points[:,2] < z_max)
    )
    points = points[mask]

    # ── 2. Assign each point to a grid cell ──────────────────
    grid_x = ((points[:,0] - x_min) / vx).astype(int)
    grid_y = ((points[:,1] - y_min) / vy).astype(int)

    # ── 3. Group points by pillar (grid_x, grid_y) ───────────
    pillar_dict = {}
    for i, (gx, gy) in enumerate(zip(grid_x, grid_y)):
        key = (gx, gy)
        if key not in pillar_dict:
            pillar_dict[key] = []
        if len(pillar_dict[key]) < max_pts_per_pillar:
            pillar_dict[key].append(i)

    # ── 4. Build pillar feature tensors ──────────────────────
    num_pillars = min(len(pillar_dict), max_pillars)
    pillars       = np.zeros((max_pillars, max_pts_per_pillar, 9), dtype=np.float32)
    pillar_coords = np.zeros((max_pillars, 2), dtype=np.int32)

    for p_idx, ((gx, gy), pt_indices) in enumerate(pillar_dict.items()):
        if p_idx >= max_pillars:
            break

        pts = points[pt_indices]      # (M, 3) where M ≤ 32
        xc  = pts[:,0].mean()         # pillar centroid
        yc  = pts[:,1].mean()
        zc  = pts[:,2].mean()

        M = len(pts)
        pillars[p_idx, :M, :3]  = pts            # x, y, z
        pillars[p_idx, :M, 3]   = pts[:,0] - xc  # x offset from centroid
        pillars[p_idx, :M, 4]   = pts[:,1] - yc  # y offset from centroid
        pillars[p_idx, :M, 5]   = pts[:,2] - zc  # z offset from centroid
        pillars[p_idx, :M, 6]   = xc              # pillar center x
        pillars[p_idx, :M, 7]   = yc              # pillar center y
        pillars[p_idx, :M, 8]   = zc              # pillar center z
        pillar_coords[p_idx]     = [gx, gy]

    return pillars, pillar_coords, num_pillars


