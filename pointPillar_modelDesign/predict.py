"""
Inference Pipeline
At test time there are no labels — you just load a raw .pcd file, run it through the model, and get back a list of predicted 
bounding boxes. The extra step is NMS (Non-Maximum Suppression) — the model predicts hundreds of overlapping boxes, NMS keeps 
only the best one per object.
"""



# predict.py
import torch
import open3d as o3d
import numpy as np
from model   import PointPillars
from pillars import create_pillars

CLASS_NAMES = ["car", "pedestrian", "cyclist"]

def load_model(checkpoint_path, num_classes=3, device="cpu"):
    model = PointPillars(num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()        # switch to inference mode
    model.to(device)
    return model


def nms_bev(boxes, scores, iou_threshold=0.1):
    """
    Simple 2D NMS on BEV boxes.
    boxes  : (N, 7)  [cx, cy, cz, l, w, h, yaw]
    scores : (N,)
    Returns indices of kept boxes.
    """
    if len(boxes) == 0:
        return []

    order  = scores.argsort()[::-1]  # sort by confidence
    keep   = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        # Compute BEV IoU with remaining boxes
        ious = bev_iou(boxes[i], boxes[order[1:]])
        mask = ious < iou_threshold       # keep boxes with low overlap
        order = order[1:][mask]

    return keep


def predict(pcd_path, model, device="cpu",
             conf_threshold=0.3, nms_threshold=0.1):
    """
    Run detection on a single .pcd file.
    Returns list of dicts: {box, class_name, score}
    """
    # ── 1. Load + preprocess ─────────────────────────────────
    pcd    = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points, dtype=np.float32)

    pillars, coords, n_pil = create_pillars(points)

    pillars_t = torch.from_numpy(pillars).unsqueeze(0).to(device)  # (1, P, 32, 9)
    coords_t  = torch.from_numpy(coords).unsqueeze(0).to(device)   # (1, P, 2)

    # ── 2. Forward pass ──────────────────────────────────────
    with torch.no_grad():
        preds = model(pillars_t, coords_t, [n_pil])

    # ── 3. Decode predictions ────────────────────────────────
    cls_logits = preds["cls"][0]        # (A*num_classes, H, W)
    box_preds  = preds["box"][0]        # (A*7, H, W)

    cls_scores = torch.sigmoid(cls_logits)  # convert logits → probabilities
    scores, cls_ids = cls_scores.max(dim=0)

    # Flatten spatial dims
    scores  = scores.flatten().cpu().numpy()
    cls_ids = cls_ids.flatten().cpu().numpy()
    boxes   = decode_boxes(box_preds)    # convert offsets → absolute coords

    # ── 4. Filter by confidence ──────────────────────────────
    mask   = scores > conf_threshold
    boxes  = boxes[mask]
    scores = scores[mask]
    cls_ids= cls_ids[mask]

    # ── 5. NMS — remove duplicate boxes ─────────────────────
    keep   = nms_bev(boxes, scores, iou_threshold=nms_threshold)
    boxes  = boxes[keep]
    scores = scores[keep]
    cls_ids= cls_ids[keep]

    # ── 6. Format results ────────────────────────────────────
    results = []
    for i in range(len(boxes)):
        results.append({
            "box":        boxes[i].tolist(),      # [cx,cy,cz,l,w,h,yaw]
            "class_name": CLASS_NAMES[cls_ids[i]],
            "score":      float(scores[i])
        })

    return results


# ── Run it ────────────────────────────────────────────────────
if __name__ == "__main__":
    model   = load_model("best_model.pth")
    results = predict("test_scene.pcd", model)

    print(f"Detected {len(results)} objects:")
    for r in results:
        print(f"  {r['class_name']:12s}  score={r['score']:.2f}"
              f"  box={[round(v,2) for v in r['box']]}")
        

        