
"""
GT boxes  ──┐
            ├──► bev_iou() ──► IoU scores per anchor
Anchors   ──┘                        │
                          ┌──────────┴──────────┐
                     IoU ≥ 0.6            IoU < 0.45
                     POSITIVE             NEGATIVE
                          │                    │
                    encode_box()         cls target = 0
                    (learn offsets)      (learn background)
                          │
                    box_targets, cls_targets, pos_mask
"""

import numpy as np
import torch
from anchors import generate_anchors, bev_iou, encode_box



# def build_targets(preds, gt_boxes_list, gt_labels_list, device,
#                   pos_iou_thresh=0.6, neg_iou_thresh=0.45):
#     """
#     For every anchor in the grid, decide:
#       - Is it positive (matched to a GT box)?   → pos_mask = True
#       - Is it negative (background)?            → neg_mask = True
#       - What regression offsets should it learn?

#     preds         : dict of model outputs (we use shape to know grid size)
#     gt_boxes_list : list of tensors, one per batch item [(N1,7), (N2,7),...]
#     gt_labels_list: list of tensors, one per batch item [(N1,),  (N2,), ...]
#     device        : torch device

#     Returns: dict of target tensors matching pred shapes
#     """
#     B   = preds["cls"].shape[0]
#     H   = preds["cls"].shape[2]
#     W   = preds["cls"].shape[3]
#     num_anchors    = 2
#     num_classes    = preds["cls"].shape[1] // num_anchors

#     # ── Generate anchors (same grid as predictions) ──────────────
#     anchors_np = generate_anchors(grid_h=H, grid_w=W)   # (H, W, 2, 7)
#     anchors_flat = anchors_np.reshape(-1, 7)             # (H*W*2, 7)
#     A = len(anchors_flat)

#     # ── Allocate target tensors ───────────────────────────────────
#     # Shape mirrors prediction output: (B, num_anchors*C, H, W)
#     cls_targets = torch.zeros(B, num_anchors * num_classes, H, W)
#     box_targets = torch.zeros(B, num_anchors * 7,           H, W)
#     dir_targets = torch.zeros(B, num_anchors * 2,           H, W, dtype=torch.long)
#     pos_mask    = torch.zeros(B, num_anchors,               H, W, dtype=torch.bool)
#     neg_mask    = torch.ones (B, num_anchors,               H, W, dtype=torch.bool)  # all negative by default

#     # ── Process each batch item separately ───────────────────────
#     for b in range(B):
#         gt_boxes  = gt_boxes_list[b].numpy()    # (num_gt, 7)
#         gt_labels = gt_labels_list[b].numpy()   # (num_gt,)

#         if len(gt_boxes) == 0:
#             continue   # no objects in this scene

#         # ── For each GT box, compute IoU with ALL anchors ─────────
#         for g_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):

#             ious = bev_iou(anchors_flat, gt_box)    # (A,) iou scores

#             # Positive: IoU > 0.6  → this anchor matches the GT box
#             pos_indices = np.where(ious >= pos_iou_thresh)[0]

#             # Also always assign the single best anchor per GT
#             # (guarantees every GT box has at least one positive)
#             best_anchor = np.argmax(ious)
#             pos_indices = np.unique(np.append(pos_indices, best_anchor))

#             # Negative: IoU < 0.45 → definitely background (ignore 0.45–0.6)
#             neg_indices = np.where(ious < neg_iou_thresh)[0]

#             # ── Encode regression targets for positive anchors ────
#             for a_flat in pos_indices:
#                 # Convert flat index → (grid_y, grid_x, rotation)
#                 gy   = a_flat // (W * num_anchors)
#                 rest = a_flat  % (W * num_anchors)
#                 gx   = rest   // num_anchors
#                 r    = rest    % num_anchors

#                 # Classification target
#                 cls_targets[b, r * num_classes + gt_label, gy, gx] = 1.0

#                 # Box regression target (encode as offsets)
#                 anchor = anchors_np[gy, gx, r]
#                 offsets = encode_box(anchor, gt_box)       # (7,)
#                 box_targets[b, r*7:(r+1)*7, gy, gx] = torch.from_numpy(offsets)

#                 # Direction target (0 = forward, 1 = backward)
#                 dir_targets[b, r, gy, gx] = 0 if gt_box[6] >= 0 else 1

#                 # Mark as positive
#                 pos_mask[b, r, gy, gx] = True
#                 neg_mask[b, r, gy, gx] = False   # not negative anymore

#             # Clear ignore-zone from neg_mask (0.45–0.6 IoU = ignore)
#             for a_flat in np.where((ious >= neg_iou_thresh) & (ious < pos_iou_thresh))[0]:
#                 gy   = a_flat // (W * num_anchors)
#                 rest = a_flat  % (W * num_anchors)
#                 gx   = rest   // num_anchors
#                 r    = rest    % num_anchors
#                 neg_mask[b, r, gy, gx] = False   # ignore, not negative

#     return {
#         "cls":      cls_targets.to(device),
#         "box":      box_targets.to(device),
#         "dir":      dir_targets.to(device),
#         "pos_mask": pos_mask.to(device),
#         "neg_mask": neg_mask.to(device),
#     }







def build_targets(preds, gt_boxes_list, gt_labels_list, device,
                  pos_iou_thresh=0.6, neg_iou_thresh=0.45):
    """
    preds          : dict {cls:(B,A*C,H,W), box:(B,A*7,H,W), dir:(B,A*2,H,W)}
    gt_boxes_list  : list of B tensors, each (num_gt, 7)
    gt_labels_list : list of B tensors, each (num_gt,)
    """
    B           = preds["cls"].shape[0]
    num_classes = preds["cls"].shape[1] // 2   # A=2 rotations
    H           = preds["cls"].shape[2]
    W           = preds["cls"].shape[3]
    A           = 2   # rotations per cell

    # ── Generate anchors ──────────────────────────────────────────
    anchors_np   = generate_anchors(grid_h=H, grid_w=W)   # (H, W, 2, 7)
    anchors_flat = anchors_np.reshape(-1, 7)               # (H*W*2, 7)

    # ── Allocate targets (same spatial layout as predictions) ─────
    cls_targets = torch.zeros(B, A * num_classes, H, W, dtype=torch.float32)
    box_targets = torch.zeros(B, A * 7,           H, W, dtype=torch.float32)
    dir_targets = torch.zeros(B, A,               H, W, dtype=torch.long)
    pos_mask    = torch.zeros(B, A,               H, W, dtype=torch.bool)
    neg_mask    = torch.ones (B, A,               H, W, dtype=torch.bool)

    for b in range(B):
        gt_boxes  = gt_boxes_list[b].cpu().numpy()    # (G, 7)
        gt_labels = gt_labels_list[b].cpu().numpy()   # (G,)

        if len(gt_boxes) == 0:
            continue

        for gt_box, gt_label in zip(gt_boxes, gt_labels):

            ious = bev_iou(anchors_flat, gt_box)   # (H*W*2,)

            # Best anchor always gets assigned (covers cases where
            # no anchor clears the threshold)
            best_idx  = int(np.argmax(ious))
            pos_idx   = np.where(ious >= pos_iou_thresh)[0]
            pos_idx   = np.unique(np.append(pos_idx, best_idx))
            ign_idx   = np.where(
                            (ious >= neg_iou_thresh) & (ious < pos_iou_thresh)
                        )[0]

            # ── Helper: flat index → (gy, gx, r) ─────────────────
            def flat_to_gyr(a_flat):
                gy = int(a_flat) // (W * A)
                rx = int(a_flat)  % (W * A)
                gx = rx // A
                r  = rx  % A
                return gy, gx, r

            # ── Fill positive anchors ─────────────────────────────
            for a_flat in pos_idx:
                gy, gx, r = flat_to_gyr(a_flat)
                anchor    = anchors_np[gy, gx, r]          # (7,)
                offsets   = encode_box(anchor, gt_box)      # (7,)

                # cls: one-hot at the gt class channel for this rotation
                cls_targets[b, r * num_classes + int(gt_label), gy, gx] = 1.0

                # box: 7 channels starting at r*7
                box_targets[b, r*7 : r*7+7, gy, gx] = torch.from_numpy(offsets)

                # direction: 0 if yaw≥0, else 1
                dir_targets[b, r, gy, gx] = 0 if gt_box[6] >= 0 else 1

                pos_mask[b, r, gy, gx] = True
                neg_mask[b, r, gy, gx] = False

            # ── Clear ignore zone from neg_mask ───────────────────
            for a_flat in ign_idx:
                gy, gx, r = flat_to_gyr(a_flat)
                neg_mask[b, r, gy, gx] = False

    return {
        "cls":      cls_targets.to(device),
        "box":      box_targets.to(device),
        "dir":      dir_targets.to(device),
        "pos_mask": pos_mask.to(device),    # (B, 2, H, W)
        "neg_mask": neg_mask.to(device),    # (B, 2, H, W)
    }



