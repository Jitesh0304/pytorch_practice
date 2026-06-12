"""
The Training Cycle
Training is an iterative loop. In each iteration we: load a batch → forward pass → compute loss → backpropagate gradients → update weights. We repeat this for many epochs until the loss converges (stops decreasing significantly).

🔁 One training step
  batch → model.forward() → predictions → loss_fn() → loss.backward() → optimizer.step()
    ↑                                                                          |
    └───────────────────── repeat for next batch ──────────────────────────────┘

Raw PCD         Pillars         PillarNet           BEV Map         Backbone        Boxes
(921600, 3)     (P, N, 9)       (P, 64)             (64, H, W)      CNN             [x,y,z,l,w,h,θ]

"""

# train.py
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from model      import PointPillars
from loss        import PointPillarsLoss
from dataloader  import get_loaders
from buildTargets import build_targets




def train(root_dir, num_epochs=80, batch_size=4, lr=3e-4):

    # ── Setup ─────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model     = PointPillars(num_classes=3).to(device)
    loss_fn   = PointPillarsLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    train_loader, val_loader = get_loaders(root_dir, batch_size)

    # OneCycleLR: warms up then decays lr — very effective
    scheduler = OneCycleLR(optimizer, max_lr=lr,
                             total_steps=num_epochs * len(train_loader))

    best_val_loss = float("inf")

    # ── Epoch loop ────────────────────────────────────────────
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # ── Batch loop ────────────────────────────────────────
        for step, batch in enumerate(train_loader):

            # 1. Move data to GPU
            pillars = batch["pillars"].to(device)      # (B, P, 32, 9)
            coords  = batch["coords"].to(device)       # (B, P, 2)
            n_pil   = batch["num_pillars"]             # list of ints

            # 2. Forward pass — get predictions
            preds = model(pillars, coords, n_pil)       # dict: cls, box, dir

            # 3. Build targets (anchor matching)
            targets = build_targets(preds, batch["gt_boxes"],
                                      batch["gt_labels"], device)

            # 4. Compute loss
            losses = loss_fn(preds, targets)
            loss   = losses["total"]

            # 5. Backprop
            optimizer.zero_grad()   # clear old gradients
            loss.backward()          # compute new gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 35)  # prevent explosion
            optimizer.step()         # update weights
            scheduler.step()         # update learning rate

            total_loss += loss.item()

            if step % 10 == 0:
                print(f"  [Ep {epoch+1}/{num_epochs} | step {step}]"
                      f" loss={loss.item():.4f}"
                      f" cls={losses['cls']:.3f}"
                      f" box={losses['box']:.3f}")

        # ── Validation ────────────────────────────────────────
        val_loss = evaluate(model, val_loader, loss_fn, device)
        avg_train = total_loss / len(train_loader)

        print(f"Epoch {epoch+1}: train={avg_train:.4f} | val={val_loss:.4f}")

        # ── Save best model ───────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("  ✅ Saved best model")


def evaluate(model, loader, loss_fn, device):
    """Run one epoch on validation set, return average loss."""
    model.eval()
    total = 0.0
    with torch.no_grad():  # no gradients needed for validation
        for batch in loader:
            pillars = batch["pillars"].to(device)
            coords  = batch["coords"].to(device)
            preds   = model(pillars, coords, batch["num_pillars"])
            targets = build_targets(preds, batch["gt_boxes"],
                                      batch["gt_labels"], device)
            total += loss_fn(preds, targets)["total"].item()
    return total / len(loader)


if __name__ == "__main__":
    train(root_dir="./dataset", num_epochs=80, batch_size=4)



"""
What to watch during training
Metric	    Good sign	                    Bad sign
train_loss	Steadily decreasing	            Spikes or stays flat → reduce LR
val_loss	Follows train_loss	            Val > train by big margin → overfitting
cls_loss	Drops below 0.3	                Stuck at ~0.7 → check class imbalance
box_loss	Drops below 0.2	                Not decreasing → check anchor sizes
"""


