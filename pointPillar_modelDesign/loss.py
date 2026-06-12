"""
What the model is learning
The model makes predictions on thousands of anchor boxes. We compare these predictions to the ground-truth boxes and calculate 
3 losses simultaneously. The total loss guides the optimizer to improve all three aspects at once.


3 Loss Components
Loss	        What it measures	                                    Function
cls_loss	    Is there an object here? Which class?	                Focal Loss (handles class imbalance)
box_loss	    How wrong are the predicted box coordinates?	        SmoothL1 Loss
dir_loss	    Is the object facing left or right?	                    CrossEntropy (2-class)

Why Focal Loss? 
In a BEV grid with 300×300 cells, maybe only 10 contain actual objects. If you use standard cross-entropy, the model just learns 
to predict "no object" everywhere and gets 99% accuracy but detects nothing. Focal Loss down-weights easy negatives so the model 
focuses on hard positives.

"""



import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss — penalises hard/wrong predictions more.
    alpha: balance pos/neg  (default 0.25)
    gamma: focus parameter  (default 2.0)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        # preds  : raw logits  (any shape)
        # targets: 0/1 labels  (same shape as preds)
        bce    = F.binary_cross_entropy_with_logits(preds, targets, reduction="none")
        pt     = torch.exp(-bce)                 # probability of correct class
        focal  = self.alpha * ((1 - pt) ** self.gamma) * bce
        return focal.mean()


# class PointPillarsLoss(nn.Module):
#     """
#     Combined loss for PointPillars.
#     Takes model predictions and matched ground-truth targets.
#     """
#     def __init__(self):
#         super().__init__()
#         self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
#         self.smooth_l1  = nn.SmoothL1Loss()
#         self.ce_loss    = nn.CrossEntropyLoss()

#     def forward(self, preds, targets):
#         """
#         preds   : dict from model.forward()  {cls, box, dir}
#         targets : dict of matched GT targets  {cls, box, dir, pos_mask}
#         """
#         pos_mask = targets["pos_mask"]  # (B, A, H, W) — which anchors match GT

#         # ── Classification loss ───────────────────────────────────
#         cls_loss = self.focal_loss(preds["cls"], targets["cls"])

#         # ── Box regression loss (only on positive anchors) ────────
#         pos_pred_box = preds["box"][pos_mask]    # only matched anchors
#         pos_tgt_box  = targets["box"][pos_mask]
#         box_loss     = self.smooth_l1(pos_pred_box, pos_tgt_box)  if pos_pred_box.numel() > 0 else torch.tensor(0.0)

#         # ── Direction loss (only on positive anchors) ─────────────
#         pos_pred_dir = preds["dir"][pos_mask]
#         pos_tgt_dir  = targets["dir"][pos_mask]
#         dir_loss     = self.ce_loss(pos_pred_dir, pos_tgt_dir)    if pos_pred_dir.numel() > 0 else torch.tensor(0.0)

#         # ── Combine with weights ──────────────────────────────────
#         total = 1.0 * cls_loss + 2.0 * box_loss + 0.2 * dir_loss

#         return {
#             "total": total,
#             "cls":   cls_loss.item(),
#             "box":   box_loss.item(),
#             "dir":   dir_loss.item(),
#         }




class PointPillarsLoss(nn.Module):
    def forward(self, preds, targets):
        pos_mask = targets["pos_mask"]   # (B, 2, H, W)
        B, _, H, W = pos_mask.shape

        # ── cls loss (focal, on pos+neg only) ─────────────────────
        cls_loss = self.focal_loss(preds["cls"], targets["cls"])

        # ── box loss — expand (B,2,H,W) → (B,14,H,W) ✅ ──────────
        pos_box = pos_mask.unsqueeze(2).expand(B, 2, 7, H, W) \
                          .reshape(B, 14, H, W)

        pred_box = preds["box"][pos_box]
        tgt_box  = targets["box"][pos_box]
        box_loss = self.smooth_l1(pred_box, tgt_box) \
                   if pred_box.numel() > 0 \
                   else torch.tensor(0.0, device=pred_box.device if pred_box.numel()>0
                                     else preds["box"].device)

        # ── dir loss — expand (B,2,H,W) → flat on positives ──────
        pred_dir = preds["dir"].reshape(B, 2, 2, H, W)[pos_mask]  # (P, 2)
        tgt_dir  = targets["dir"][pos_mask]                        # (P,)
        dir_loss = self.ce_loss(pred_dir, tgt_dir) \
                   if pred_dir.numel() > 0 \
                   else torch.tensor(0.0, device=preds["dir"].device)

        total = 1.0 * cls_loss + 2.0 * box_loss + 0.2 * dir_loss
        return {"total": total,
                "cls":   cls_loss.item(),
                "box":   box_loss.item(),
                "dir":   dir_loss.item()}
    



    
# ── Anchor matching (GT box → nearest anchor) ─────────────────
def match_anchors_to_gt(anchors, gt_boxes, gt_labels,
                          pos_iou_thresh=0.6, neg_iou_thresh=0.45):
    """
    For each GT box, find the best-matching anchor.
    Returns target tensors with the same shape as predictions.
    (Simplified version — production code uses vectorised 3D IoU)
    """
    # In practice, use a library like spconv or mmdet3d
    # that provides optimised 3D IoU + anchor assignment
    pass  # see training.py for integration pattern














