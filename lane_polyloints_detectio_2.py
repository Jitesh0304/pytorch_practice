import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp import autocast
from torchvision import models
from pathlib import Path
from datetime import datetime
from scipy.optimize import linear_sum_assignment


# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "img_h": 512,
    "img_w": 512,
    "num_queries": 16,      # max lanes the model can detect per image (replaces fixed num_lanes)
    "num_points": 32,       # how many (x,y) points describe each lane polyline
    "feat_dim": 256,        # feature / query embedding dimension
    "num_decoder_layers": 3,# transformer decoder depth
    "batch_size": 8,
    "epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "exist_loss_weight": 2.0,   # up-weight existence loss (class imbalance: few active queries)
    "train_image_folder": os.path.join(Path(__file__).parent, 'dummy_dataset', 'train_image'),
    "train_label_json":   os.path.join(Path(__file__).parent, 'dummy_dataset', 'train_image_label.json'),
    "val_image_folder":   os.path.join(Path(__file__).parent, 'dummy_dataset', 'test_image'),
    "val_label_json":     os.path.join(Path(__file__).parent, 'dummy_dataset', 'test_image_label.json'),
    "model_folder":       os.path.join(Path(__file__).parent, 'models'),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

LANE_COLORS = [
    (0,   0,   255),  (0,   165, 255),  (0,   255, 255),  (0,   255,   0),
    (255, 255,   0),  (255,   0,   0),  (255,   0, 255),  (128,   0, 128),
    (0,   128, 255),  (128, 255,   0),  (255, 128,   0),  (0,   255, 128),
    (200,  50, 200),  (50,  200, 200),  (200, 200,  50),  (100, 100, 255),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DATASET
#     Label format (updated from v1):
#       "lanes": [ [[x0,y0],[x1,y1],...], [...], ... ]   ← plain list of lanes,
#     each lane is a list of [x, y] pixel coords.
#     No lane_0 / lane_3 naming needed — Hungarian matching handles ordering.
# ═══════════════════════════════════════════════════════════════════════════════

def normalise_lane(pts: np.ndarray) -> np.ndarray:
    """
    Resample a variable-length lane (N×2 pixel coords) to exactly num_points
    evenly-spaced points, then normalise to [0, 1].
    """
    P = CONFIG['num_points']
    pts = pts.astype(np.float32)

    # Compute cumulative arc-length

    # axis=0 → operate row-wise (down the rows) => diffs[i] = arr[i+1] − arr[i] 
    # dim => if actual dim is (r, c) then diffs dim will be (r-1, c)
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))         # sum values of all columns.  dim will be (r-1,)
    cum = np.concatenate([[0], np.cumsum(seg_lengths)])     # cumsum = [arr[0], arr[0]+arr[1], arr[0]+arr[1]+arr[2], ...]
    total = cum[-1]         # last element of the array

    if total < 1e-6:
        # Degenerate lane — replicate the single point
        # (P, 1) => P → repeat rows (axis 0)  &   1 → repeat columns (axis 1)   & shape (P, 2)
        return np.tile(pts[0] / [CONFIG['img_w'], CONFIG['img_h']], (P, 1))

    # Uniformly sample P points along the arc & shape (P,)
    sample_at = np.linspace(0, total, P)
    
    # 1D linear interpolation   (shape = (P, 2))
    resampled = np.stack([
        np.interp(sample_at, cum, pts[:, 0]),
        np.interp(sample_at, cum, pts[:, 1]),
    ], axis=1)

    # Normalise to [0, 1]
    resampled[:, 0] /= CONFIG['img_w']      # shape = (P, 2)
    resampled[:, 1] /= CONFIG['img_h']      # shape = (P, 2)
    return resampled.astype(np.float32)


class LaneDataset(Dataset):
    """
    Expects label JSON as a list of dicts:
      { "image": "xxx.jpg",
        "lanes": [ [[x,y], ...], [[x,y], ...] ]  }

    Also supports the OLD v1 format:
      { "lanes": { "lane_0": [[x,y],...], "lane_3": [[x,y],...] } }
    so you can reuse the dummy dataset without regenerating it.
    """

    def __init__(self, img_dir: str, label_file: str):
        self.img_dir = img_dir
        with open(label_file) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item   = self.data[index]
        img    = cv2.imread(os.path.join(self.img_dir, item['image']))
        img    = cv2.resize(img, (CONFIG['img_w'], CONFIG['img_h']))
        img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(
            (img / 255.0).astype(np.float32)
        ).permute(2, 0, 1)   # (3, H, W) pytorch format

        # ── Parse lanes (support both v1 dict and v2 list format) ──────────
        raw = item['lanes']
        if isinstance(raw, dict):
            # v1 format: {"lane_0": [...], "lane_3": [...]}
            lane_list = list(raw.values())
        else:
            # v2 format: [ [...], [...] ]
            lane_list = raw

        lanes = []
        for pts in lane_list:
            pts_np = np.array(pts, dtype=np.float32)
            if len(pts_np) >= 2:
                lanes.append(normalise_lane(pts_np))   # (P, 2)

        # lanes is a list of at most Q arrays of shape (P, 2)
        # We return it as a padded tensor + a count so collate_fn can batch
        G = len(lanes)
        P = CONFIG['num_points']
        gt_tensor = torch.zeros(CONFIG['num_queries'], P, 2)    # shape (num_queries, P, 2)
        for i, lane in enumerate(lanes[:CONFIG['num_queries']]):
            gt_tensor[i] = torch.from_numpy(lane)       # replace the zeros matrix of each lane with actual values 

        return tensor, gt_tensor, torch.tensor(G, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class FPNBackbone(nn.Module):
    """
    ResNet-18 backbone with a lightweight 2-level FPN.
    Outputs a single (B, feat_dim, H/16, W/16) feature map.
    """

    def __init__(self, feat_dim: int = 256):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)   # shape = (batch_size, 1000) = 1000 classes

        # Stage extractors (we keep up to layer3 for speed)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        # conv1
        # k = 7     S = 2       p = 3
        # ((W - K + 2p)/S) + 1    =>      ((512 - 7 + 2*3)/2) + 1
        # result (256)      (B, 64, 256, 256)

        # maxpol
        # K = 3     S = 2       p = 1
        # ((W - K + 2p)/S) + 1    =>      ((256 - 3 + 2*1)/2) + 1
        # result (128)      (B, 64, 128, 128)

        self.layer1 = resnet.layer1   # stride 4,  64 ch    
        # K = 3     S = 1       p = 1   (B, 64, 128, 128)

        self.layer2 = resnet.layer2   # stride 8,  128 ch   
        # K = 3     S = 2       p = 1   (B, 128, 64, 64)

        self.layer3 = resnet.layer3   # stride 16, 256 ch   
        # K = 3     S = 2       p = 1   (B, 256, 32, 32)

        # Lateral projections → common feat_dim
        self.lat2 = nn.Conv2d(128, feat_dim, 1)
        self.lat3 = nn.Conv2d(256, feat_dim, 1)     # (B, feat_dim, 32, 32)

        # Output projection
        self.out_conv = nn.Conv2d(feat_dim, feat_dim, 3, padding=1)

    def forward(self, x):
        # input (B, 3, 516, 516)
        x  = self.layer0(x)     # output  (B, 64, 128, 128)
        c1 = self.layer1(x)    # stride 4   (B, 64, 128, 128)
        c2 = self.layer2(c1)   # stride 8   (B, 128, 64, 64)
        c3 = self.layer3(c2)   # stride 16  (B, 256, 32, 32)

        # FPN top-down merge
        p3 = self.lat3(c3)      # output (B, feat_dim, 32, 32)

        # p2 shape (B, 256, 64, 64)
        # F.interpolate => It resizes p3 feature map to match the spatial size of c2 using nearest-neighbor upsampling.
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')      # output (B, 256, 64, 64)

        return self.out_conv(p2)   # (B, feat_dim, H/8, W/8)    (B, feat_dim, 64, 64)


class LaneQueryDecoder(nn.Module):
    """
    Transformer decoder that refines Q learned lane queries by attending
    to the FPN feature map.  Each query then regresses:
      • an existence score   (1 value)
      • num_points (x,y)     (num_points × 2 values)
    """

    def __init__(self):
        super().__init__()
        Q   = CONFIG['num_queries']
        P   = CONFIG['num_points']
        D   = CONFIG['feat_dim']
        L   = CONFIG['num_decoder_layers']

        # Learned lane query embeddings     shape (16, 256)
        self.query_embed = nn.Embedding(Q, D)

        # Positional encoding for flattened feature map tokens      shape (B, 256)
        self.pos_proj = nn.Linear(D, D)

        # Stack of transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=D, nhead=8, dim_feedforward=D * 4,
                dropout=0.1, batch_first=True
            )
            for _ in range(L)
        ])

        # Output heads
        self.exist_head = nn.Linear(D, 1)       # last dimension → 1
        self.point_head = nn.Sequential(
            nn.Linear(D, D),        # last dimension → D (256)
            nn.ReLU(),
            nn.Linear(D, P * 2),    # last dimension → 2P (2*32 = 64)
            nn.Sigmoid()           # normalised coords already in [0,1]
        )

    def forward(self, feature_map):
        """
        feature_map : (B, D, Hf, Wf)
        returns:
            exist_logits : (B, Q)      — raw (pre-sigmoid) existence scores
            pred_points  : (B, Q, P, 2)— predicted lane polypoints in [0,1]
        """
        # input (B, feat_dim, 64, 64)
        B  = feature_map.size(0)
        D  = CONFIG['feat_dim']
        P  = CONFIG['num_points']
        Q  = CONFIG['num_queries']

        # Flatten spatial → sequence: (B, Hf*Wf, D)
        memory = feature_map.flatten(2).permute(0, 2, 1)    # output = (B, 64*64, feat_dim)

        # Expand learned queries to batch: (B, Q, D)        (B, 16, 256)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        # Passing -1 as the size for a dimension means not changing the size of that dimension.

        # Iteratively refine through decoder layers
        out = queries
        for layer in self.decoder_layers:
            # A decoder layer takes two inputs:
            # T = target sequence length (e.g., number of queries)
            # S = source sequence length (from encoder | backbone)
            out = layer(out, memory)   # (B, Q, D)      (B, 16, 256)

        exist_logits = self.exist_head(out).squeeze(-1)             # (B, Q)
        # exist_head output = (B, 16, 256)
        # after squeeze (B, 16)
        pred_points  = self.point_head(out).view(B, Q, P, 2)        # (B, Q, P, 2)
        # point_head output = (B, 16, 64)
        # afte view = (B, 16, 32, 2)

        return exist_logits, pred_points


class LaneDetector(nn.Module):
    """Full model: FPN backbone + query decoder."""

    def __init__(self):
        super().__init__()
        self.backbone = FPNBackbone(feat_dim=CONFIG['feat_dim'])
        self.decoder  = LaneQueryDecoder()

    def forward(self, x):
        # input (B, 3, 512, 512)
        feat = self.backbone(x)     # output => (B, feat_dim, 64, 64)
        return self.decoder(feat)   # output => (B, 16) for lane exist      (B, 16, 32, 2) for lane points


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  HUNGARIAN MATCHING LOSS
#     For each image in the batch:
#       - Build a cost matrix (Q × G) comparing each query to each gt lane
#       - Run Hungarian algorithm to find the optimal 1-to-1 assignment
#       - Matched queries → L1 regression loss
#       - All queries → BCE existence loss (matched=1, unmatched=0)
# ═══════════════════════════════════════════════════════════════════════════════

def hungarian_loss(exist_logits, pred_points, gt_points, gt_counts):
    """
    exist_logits : (B, Q)
    pred_points  : (B, Q, P, 2)
    gt_points    : (B, Q, P, 2)   padded; first gt_counts[b] rows are real
    gt_counts    : (B,)           number of real lanes per image (B, G)
    """
    B, Q, P, _ = pred_points.shape
    total_exist = 0.0       # classification loss (lane exists or not)
    total_reg   = 0.0       # regression loss (lane shape)
    n_matched   = 0         # number of matched pairs (for averaging)

    for b in range(B):
        G = gt_counts[b].item()         # number of real lanes in this image

        exist_target = torch.zeros(Q, device=device)

        if G == 0:
            # No lanes → all queries should be inactive
            total_exist += F.binary_cross_entropy_with_logits(
                exist_logits[b], exist_target
            )
            continue

        gt_b = gt_points[b, :G]   # (G, P, 2)
        # G is the iteration count of the batch. So gt_points[b, :G] means in the batch[i] how many actual lanes are present

        # ── Cost matrix: L1 distance between each query and each gt lane ──
        pred_flat = pred_points[b].view(Q, P * 2)    # (Q, P*2)
        gt_flat   = gt_b.view(G, P * 2)              # (G, P*2)

        # (Q, G) pairwise L1
        cost = torch.cdist(pred_flat, gt_flat, p=1).detach().cpu().numpy()      # output (Q, G)

        # ── Hungarian matching ────────────────────────────────────────────
        q_idx, g_idx = linear_sum_assignment(cost)   # both are arrays of length min(Q,G)
            # output = (M, ) and (M, )          M = min(Q, G)   Q = pred lanes |  G = actual lanes
            # len(q_idx) == len(g_idx) == M
            # prediction q_idx[i]  <--->  ground truth g_idx[i]

        # ── Regression loss on matched pairs ─────────────────────────────
        matched_pred = pred_points[b][q_idx]          # (M, P, 2)
        matched_gt   = gt_b[g_idx]                    # (M, P, 2)
        total_reg   += F.l1_loss(matched_pred, matched_gt)      # how close predicted lanes are to ground truth lanes
        n_matched   += len(q_idx)                               # 

        # ── Existence loss ────────────────────────────────────────────────
        exist_target[q_idx] = 1.0     # make the lane exist
        total_exist += F.binary_cross_entropy_with_logits(
            exist_logits[b], exist_target                   # match predicted exist and actual exist 
        )

    exist_loss = total_exist / B
    reg_loss   = total_reg / max(n_matched, 1)

    return reg_loss + CONFIG['exist_loss_weight'] * exist_loss, reg_loss, exist_loss


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def build_loader(img_dir, label_file, shuffle=True):
    ds = LaneDataset(img_dir, label_file)
    return DataLoader(ds, batch_size=CONFIG['batch_size'],
                      shuffle=shuffle, num_workers=0, pin_memory=True)


def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    total_loss = 0.0
    last_loss  = 0.0
    running    = 0.0

    for i, (imgs, gt_points, gt_counts) in enumerate(loader):
        imgs      = imgs.to(device)
        gt_points = gt_points.to(device)
        gt_counts = gt_counts.to(device)

        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=device.type == 'cuda'):
            exist_logits, pred_points = model(imgs)
            loss, reg_l, exist_l = hungarian_loss(
                exist_logits, pred_points, gt_points, gt_counts
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running    += loss.item()
        total_loss += loss.item()

        if i % 10 == 9:
            last_loss = running / 10
            print(f"  batch {i+1:4d} | loss {last_loss:.4f} "
                  f"(reg {reg_l:.4f}, exist {exist_l:.4f})")
            running = 0.0

    avg = total_loss / len(loader)
    return avg


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total = 0.0
    for imgs, gt_points, gt_counts in loader:
        imgs      = imgs.to(device)
        gt_points = gt_points.to(device)
        gt_counts = gt_counts.to(device)

        with autocast(device_type=device.type, enabled=device.type == 'cuda'):
            exist_logits, pred_points = model(imgs)
            loss, _, _ = hungarian_loss(exist_logits, pred_points, gt_points, gt_counts)
        total += loss.item()
    return total / len(loader)


def start_training():
    os.makedirs(CONFIG['model_folder'], exist_ok=True)

    model     = LaneDetector().to(device)
    optimizer = optim.AdamW(model.parameters(),
                            lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scaler    = GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs'], eta_min=1e-6
    )

    train_loader = build_loader(CONFIG['train_image_folder'], CONFIG['train_label_json'])
    val_loader   = build_loader(CONFIG['val_image_folder'],   CONFIG['val_label_json'], shuffle=False)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_loss  = float('inf')
    best_path  = None

    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler)
        val_loss   = evaluate(model, val_loader)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"  train {train_loss:.4f} | val {val_loss:.4f} | lr {lr:.2e}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_path = os.path.join(
                CONFIG['model_folder'],
                f"lanev2_{timestamp}_ep{epoch+1}_val{val_loss:.4f}.pt"
            )
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ saved best model → {best_path}")

    return best_path


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  INFERENCE & VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: str) -> LaneDetector:
    m = LaneDetector().to(device)
    m.load_state_dict(torch.load(model_path, map_location=device))
    m.eval()
    print(f"Loaded model: {model_path}")
    return m


def decode_predictions(exist_logits, pred_points, exist_threshold=0.5):
    """
    exist_logits : (1, Q)
    pred_points  : (1, Q, P, 2)  — values in [0,1]

    Returns list of active lane point arrays, each (P, 2) in pixel coords.
    """
    exist_prob = torch.sigmoid(exist_logits[0]).cpu().numpy()   # (Q,)
    points     = pred_points[0].cpu().numpy()                   # (Q, P, 2)

    lanes = []
    for q in range(CONFIG['num_queries']):
        if exist_prob[q] >= exist_threshold:
            pts_pixel = points[q].copy()
            pts_pixel[:, 0] *= CONFIG['img_w']
            pts_pixel[:, 1] *= CONFIG['img_h']
            lanes.append((q, exist_prob[q], pts_pixel.astype(int)))
    return lanes


def draw_lanes(image: np.ndarray, lanes, orig_h: int, orig_w: int) -> np.ndarray:
    vis     = image.copy()
    scale_x = orig_w / CONFIG['img_w']
    scale_y = orig_h / CONFIG['img_h']

    for (q_idx, score, pts) in lanes:
        color = LANE_COLORS[q_idx % len(LANE_COLORS)]

        scaled = [(int(x * scale_x), int(y * scale_y)) for x, y in pts]

        # Polyline
        for j in range(len(scaled) - 1):
            cv2.line(vis, scaled[j], scaled[j+1], color, 2, cv2.LINE_AA)

        # Dots
        for (x, y) in scaled:
            cv2.circle(vis, (x, y), 3, color, -1)

        # Label at topmost point
        tx, ty = scaled[0]
        cv2.putText(vis, f"L{q_idx} {score:.2f}", (tx + 5, ty - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return vis


def add_legend(image: np.ndarray, lanes) -> np.ndarray:
    if not lanes:
        return image
    pad, lh = 8, 22
    bh = pad * 2 + lh * len(lanes)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (170, bh), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    for i, (q_idx, score, _) in enumerate(lanes):
        color = LANE_COLORS[q_idx % len(LANE_COLORS)]
        y = pad + i * lh + lh // 2
        cv2.rectangle(image, (pad, y-6), (pad+16, y+6), color, -1)
        cv2.putText(image, f"Lane {q_idx}  {score:.2f}",
                    (pad+22, y+5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return image


def run_inference(model_path: str, input_path: str,
                  output_dir: str = "inference_output",
                  exist_threshold: float = 0.5):
    os.makedirs(output_dir, exist_ok=True)
    model = load_model(model_path)

    p = Path(input_path)
    paths = sorted(p.iterdir()) if p.is_dir() else [p]
    paths = [f for f in paths if f.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}]

    print(f"Running inference on {len(paths)} image(s)…\n")

    for img_path in paths:
        print(f"  {img_path.name}")

        orig    = cv2.imread(str(img_path))
        orig_h, orig_w = orig.shape[:2]
        resized = cv2.resize(orig, (CONFIG['img_w'], CONFIG['img_h']))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor  = torch.from_numpy(
            (rgb / 255.0).astype(np.float32)
        ).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            with autocast(device_type=device.type, enabled=device.type == 'cuda'):
                exist_logits, pred_points = model(tensor)

        lanes = decode_predictions(exist_logits, pred_points, exist_threshold)
        vis   = draw_lanes(orig, lanes, orig_h, orig_w)
        vis   = add_legend(vis, lanes)

        out = Path(output_dir) / f"pred_{img_path.name}"
        cv2.imwrite(str(out), vis)
        print(f"    {len(lanes)} lane(s) detected → {out}")

    print(f"\nDone. Results in: {output_dir}/")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  ENTRY POINT  — swap comments to switch between training and inference
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── TRAINING ──────────────────────────────────────────────────────────────
    # model_path = start_training()

    # ── INFERENCE (comment out start_training above and uncomment below) ──────
    run_inference(
        model_path      = r"C:\Users\jites\Desktop\Personal_projects\Pytorch_code\models\lanev2_20260319_000210_ep19_val0.2185.pt",
        input_path      = r"C:\Users\jites\Desktop\Personal_projects\Pytorch_code\dummy_dataset\test_image\test_0003.jpg",
        output_dir      = CONFIG['model_folder'],
        exist_threshold = 0.5,
    )



