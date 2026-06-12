"""
BEV Grid (625 × 625 cells)
Each cell has 2 anchors (one at 0°, one at 90°)
Total anchors = 625 × 625 × 2 = ~781,250 anchors per scene

Each anchor = [cx, cy, cz, l, w, h, yaw]
              fixed prior sizes based on object class

"""



import numpy as np

# def generate_anchors(
#     grid_h=625, grid_w=625,
#     voxel_size=(0.16, 0.16),
#     point_range=(-50, -50, -5, 50, 50, 3),
#     anchor_sizes=None   # [l, w, h] per class
# ):
#     """
#     Generate all anchor boxes on the BEV grid.
#     Returns: (grid_h, grid_w, 2, 7)
#                2 rotations per cell, 7 = [cx,cy,cz,l,w,h,yaw]
#     """
#     if anchor_sizes is None:
#         # Typical sizes for cars (tune these to your data!)
#         anchor_sizes = [[3.9, 1.6, 1.56]]   # [length, width, height]

#     x_min, y_min, z_min, x_max, y_max, z_max = point_range
#     vx, vy = voxel_size

#     # Center of each grid cell
#     x_centers = np.arange(grid_w) * vx + x_min + vx / 2   # (W,)
#     y_centers = np.arange(grid_h) * vy + y_min + vy / 2   # (H,)

#     # Meshgrid → all (x, y) combinations
#     xx, yy = np.meshgrid(x_centers, y_centers)   # (H, W)

#     # Two rotations: 0° and 90°
#     rotations = [0, np.pi / 2]

#     l, w, h = anchor_sizes[0]
#     cz = z_min + h / 2          # anchor sits on ground

#     anchors = np.zeros((grid_h, grid_w, 2, 7), dtype=np.float32)
#     anchors[:, :, 0] = [xx, yy, np.full_like(xx, cz),       # rotation 0°
#                          np.full_like(xx, l),
#                          np.full_like(xx, w),
#                          np.full_like(xx, h),
#                          np.full_like(xx, 0.0)]
#     # Simpler explicit fill:
#     for r_idx, rot in enumerate(rotations):
#         anchors[:, :, r_idx, 0] = xx
#         anchors[:, :, r_idx, 1] = yy
#         anchors[:, :, r_idx, 2] = cz
#         anchors[:, :, r_idx, 3] = l
#         anchors[:, :, r_idx, 4] = w
#         anchors[:, :, r_idx, 5] = h
#         anchors[:, :, r_idx, 6] = rot

#     return anchors   # (H, W, 2, 7)


# def bev_iou(anchors_flat, gt_box):
#     """
#     IoU Between Anchor and GT Box
    
#     Compute 2D BEV IoU between N anchors and 1 GT box.
#     anchors_flat : (N, 7)  [cx,cy,cz,l,w,h,yaw]
#     gt_box       : (7,)
#     Returns      : (N,)  iou scores
    
#     Simplified axis-aligned version (ignores rotation).
#     For rotated IoU, use iou3d_nms from spconv/mmdet3d.
#     """
#     def to_corners(boxes):
#         # boxes: (N, 7)
#         cx, cy = boxes[:, 0], boxes[:, 1]
#         l,  w  = boxes[:, 3], boxes[:, 4]
#         x1 = cx - l / 2;  x2 = cx + l / 2
#         y1 = cy - w / 2;  y2 = cy + w / 2
#         return x1, y1, x2, y2

#     N = len(anchors_flat)
#     gt = gt_box[None].repeat(N, 0) if isinstance(gt_box, np.ndarray) \
#          else gt_box.unsqueeze(0).expand(N, -1)

#     ax1, ay1, ax2, ay2 = to_corners(anchors_flat)
#     gx1, gy1, gx2, gy2 = to_corners(gt)

#     # Intersection
#     ix1 = np.maximum(ax1, gx1);  ix2 = np.minimum(ax2, gx2)
#     iy1 = np.maximum(ay1, gy1);  iy2 = np.minimum(ay2, gy2)
#     inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)

#     # Union
#     area_a = (ax2 - ax1) * (ay2 - ay1)
#     area_g = (gx2 - gx1) * (gy2 - gy1)
#     union  = area_a + area_g - inter + 1e-6

#     return inter / union   # (N,)


def generate_anchors(
    grid_h=625, grid_w=625,
    voxel_size=(0.16, 0.16),
    point_range=(-50, -50, -5, 50, 50, 3),
    anchor_sizes=None
):
    """
    Returns: np.ndarray (grid_h, grid_w, 2, 7)
             2 rotations (0° and 90°) per cell
             7 = [cx, cy, cz, l, w, h, yaw]
    """
    if anchor_sizes is None:
        anchor_sizes = [[3.9, 1.6, 1.56]]   # [l, w, h] for cars

    x_min, y_min, z_min = point_range[0], point_range[1], point_range[2]
    vx, vy = voxel_size
    l, w, h = anchor_sizes[0]
    cz = z_min + h / 2

    # Cell centers
    x_centers = np.arange(grid_w) * vx + x_min + vx / 2   # (W,)
    y_centers = np.arange(grid_h) * vy + y_min + vy / 2   # (H,)
    xx, yy = np.meshgrid(x_centers, y_centers)             # (H, W)

    anchors = np.zeros((grid_h, grid_w, 2, 7), dtype=np.float32)

    # ✅ rotation 0 — l along x, w along y
    anchors[:, :, 0, 0] = xx
    anchors[:, :, 0, 1] = yy
    anchors[:, :, 0, 2] = cz
    anchors[:, :, 0, 3] = l
    anchors[:, :, 0, 4] = w
    anchors[:, :, 0, 5] = h
    anchors[:, :, 0, 6] = 0.0

    # ✅ rotation 1 — 90°, swap l and w
    anchors[:, :, 1, 0] = xx
    anchors[:, :, 1, 1] = yy
    anchors[:, :, 1, 2] = cz
    anchors[:, :, 1, 3] = w    # swapped
    anchors[:, :, 1, 4] = l    # swapped
    anchors[:, :, 1, 5] = h
    anchors[:, :, 1, 6] = np.pi / 2

    return anchors   # (H, W, 2, 7)



def bev_iou(anchors_flat: np.ndarray, gt_box: np.ndarray) -> np.ndarray:
    """
    Pure numpy. Axis-aligned BEV IoU (ignores yaw for speed).
    anchors_flat : (N, 7)
    gt_box       : (7,)
    Returns      : (N,) iou scores in [0, 1]
    """
    N  = len(anchors_flat)
    gt = np.tile(gt_box[None], (N, 1))   # ✅ (N, 7)

    def corners(b):
        cx, cy, l, w = b[:, 0], b[:, 1], b[:, 3], b[:, 4]
        return cx - l/2, cy - w/2, cx + l/2, cy + w/2

    ax1, ay1, ax2, ay2 = corners(anchors_flat)
    gx1, gy1, gx2, gy2 = corners(gt)

    ix1 = np.maximum(ax1, gx1);  ix2 = np.minimum(ax2, gx2)
    iy1 = np.maximum(ay1, gy1);  iy2 = np.minimum(ay2, gy2)
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_g = float((gt_box[3] * gt_box[4]))
    union  = area_a + area_g - inter + 1e-6

    return inter / union   # (N,)


def encode_box(anchor: np.ndarray, gt_box: np.ndarray) -> np.ndarray:
    """
    Encode GT box as offsets relative to anchor.
    anchor, gt_box : (7,)  [cx,cy,cz,l,w,h,yaw]
    Returns        : (7,)  offsets
    """
    da = np.sqrt(anchor[3]**2 + anchor[4]**2)   # anchor diagonal

    dx = (gt_box[0] - anchor[0]) / da
    dy = (gt_box[1] - anchor[1]) / da
    dz = (gt_box[2] - anchor[2]) / anchor[5]
    dl = np.log(gt_box[3] / anchor[3] + 1e-6)
    dw = np.log(gt_box[4] / anchor[4] + 1e-6)
    dh = np.log(gt_box[5] / anchor[5] + 1e-6)
    dt = gt_box[6] - anchor[6]

    return np.array([dx, dy, dz, dl, dw, dh, dt], dtype=np.float32)






"""
Encode GT Box as Regression Target
The model doesn't predict absolute coordinates — it predicts offsets relative to the anchor. This makes learning much easier.
# Why offsets? 
# Raw target: cx=45.3m  → hard to learn from scratch
# Offset:     Δcx = (45.3 - anchor_cx) / anchor_l  → small number near 0
"""

def encode_box(anchor, gt_box):
    """
    Convert absolute GT box into regression offsets.
    anchor, gt_box : (7,)  [cx, cy, cz, l, w, h, yaw]
    Returns        : (7,)  offsets the model should predict
    """
    da = np.sqrt(anchor[3]**2 + anchor[4]**2)   # diagonal of anchor

    dx  = (gt_box[0] - anchor[0]) / da
    dy  = (gt_box[1] - anchor[1]) / da
    dz  = (gt_box[2] - anchor[2]) / anchor[5]   # divide by anchor height
    dl  = np.log(gt_box[3] / anchor[3])          # log-scale for sizes
    dw  = np.log(gt_box[4] / anchor[4])
    dh  = np.log(gt_box[5] / anchor[5])
    dθ  = gt_box[6] - anchor[6]                  # raw angle difference

    return np.array([dx, dy, dz, dl, dw, dh, dθ], dtype=np.float32)


