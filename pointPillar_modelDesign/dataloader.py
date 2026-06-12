"""
PyTorch's DataLoader wraps your Dataset and feeds it to the model in batches. It also handles shuffling, parallel loading with 
multiple CPU workers, and stacking samples into batched tensors. The tricky part here is that each scene has a different 
number of objects — we need a custom collate function to handle that.
"""
import torch
from torch.utils.data import DataLoader
from dataset import PointCloudDataset
from pillars import create_pillars




def collate_fn(batch):
    """
    Each sample has:
      - points:  (N, 3)      — N may differ between scenes
      - boxes:   (num_obj, 7)— num_obj differs per scene
      - labels:  (num_obj,)

    We convert points → pillars here (in the worker process).
    boxes/labels are kept as a LIST (not stacked) because sizes differ.
    """
    all_pillars, all_coords, all_num = [], [], []
    all_boxes, all_labels = [], []

    for sample in batch:
        pts = sample["points"].numpy()
        pillars, coords, n = create_pillars(pts)
        all_pillars.append(torch.from_numpy(pillars))
        all_coords.append(torch.from_numpy(coords))
        all_num.append(n)
        all_boxes.append(sample["boxes"])
        all_labels.append(sample["labels"])

    return {
        "pillars":       torch.stack(all_pillars),  # (B, 12000, 32, 9)
        "coords":        torch.stack(all_coords),   # (B, 12000, 2)
        "num_pillars":   all_num,                   # list of ints
        "gt_boxes":      all_boxes,                 # list of tensors
        "gt_labels":     all_labels,                # list of tensors
    }

# ── Create DataLoaders ────────────────────────────────────────
def get_loaders(root_dir, batch_size=4):
    train_ds = PointCloudDataset(root_dir, split="train")
    val_ds   = PointCloudDataset(root_dir, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,          # randomise order each epoch
        num_workers=4,          # parallel data loading
        collate_fn=collate_fn,
        pin_memory=True         # faster GPU transfer
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,          # no need to shuffle for validation
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader

# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, _ = get_loaders("./dataset")
    batch = next(iter(train_loader))
    print("pillars shape:", batch["pillars"].shape)   # (4, 12000, 32, 9)
    print("coords shape :", batch["coords"].shape)    # (4, 12000, 2)


