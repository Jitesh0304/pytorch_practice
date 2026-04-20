import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.amp import autocast
from pathlib import Path
from datetime import datetime





CONFIG = {
    "img_h": 512,
    "img_w": 512,
    "num_lanes": 8,
    "num_rows": 32,
    "num_grids": 32,
    "batch_size": 8,
    "epochs": 100,
    "lr": 1e-3,
    "train_image_folder": os.path.join(Path(__file__).parent, 'dummy_dataset', 'train_image'),
    "train_label_json": os.path.join(Path(__file__).parent, 'dummy_dataset', 'train_image_label.json'),
    "val_image_folder": os.path.join(Path(__file__).parent, 'dummy_dataset', 'test_image'),
    "val_label_json": os.path.join(Path(__file__).parent, 'dummy_dataset', 'test_image_label.json'),
    "model_folder": os.path.join(Path(__file__).parent, 'models'),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# it will create an array from start value to end value with equal gap
row_anchors = np.linspace(
    CONFIG['img_h']-1,
    0,
    CONFIG['num_rows']
).astype(int)



def generate_targets(points_dict: dict):
    num_lanes = CONFIG['num_lanes']
    num_rows = CONFIG['num_rows']
    num_grids = CONFIG['num_grids']
    img_w = CONFIG['img_w']

        # create a matrix of given dim with given value
    lane_targets = np.full((num_lanes, num_rows), -1)
        # create an array of 0 of given dim
    existance = np.zeros(num_lanes)

    for lane_name, pts in points_dict.items():
        lane_idx = int(lane_name.split("_")[1])

        if lane_idx >= num_lanes or len(pts) < 2:
            continue

        existance[lane_idx] = 1     # make the lane = 1 (exist)

        pts = np.array(pts)
            # all points of (x)
        xs = pts[:, 0]
            # all points of (y)
        ys = pts[:, 1]

        for r, y in enumerate(row_anchors):
                # it will subtract the y with the array of all y point and return the minimum value index of the result array
            idx = np.argmin(np.abs(ys - y))

            if abs(ys[idx]-y) < 15:
                x = xs[idx]
                grid = int(x / img_w * num_grids)
                lane_targets[lane_idx, r] = grid
    return lane_targets, existance



class LaneDataset(Dataset):

    def __init__(self, img_dir, label_file):
        self.img_dir = img_dir

        with open(label_file) as f:
            self.data = json.load(f)

        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        img_path = os.path.join(self.img_dir, item['image'])
        img = cv2.imread(img_path)

        img = cv2.resize(
            img, (CONFIG['img_w'], CONFIG['img_h'])
        )
        points_dict = item['lanes']

        lane_targets, existance = generate_targets(points_dict)
        img = (img / 255.0).astype(np.float32)
        img = self.transform(img)

        return (
            img, 
            torch.tensor(lane_targets).long(),
            torch.tensor(existance).float()
        )



class LaneNet(nn.Module):

    def __init__(self):
        super(LaneNet, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
                # ((W - K + 2p)/S) + 1
                # ((512 - 3+2*1)/1) + 1
                # 512
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),      # (B, 32, 256, 256)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),      # (B, 64, 128, 128)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),      # (B, 128, 64, 64)

            nn.Dropout(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size = 2)       # (B, 256, 32, 32)
            nn.AdaptiveAvgPool2d(output_size=(1,1))
        )

        feat_dim = 256

        self.lane_head = nn.Linear(
            in_features = feat_dim,
            out_features = CONFIG['num_lanes'] * CONFIG['num_rows'] * CONFIG['num_grids']
        )

        self.exist_head = nn.Linear(
            feat_dim,
            CONFIG['num_lanes']
        )


    def forward(self, x):
            # (B, 3, 512, 512)
        f = self.backbone(x)        # (B, 256, 32, 32)

        f = f.view(x.size(0), -1)

        lane_out = self.lane_head(f)
        lane_out = lane_out.view(
            x.size(0),
            CONFIG['num_lanes'],
            CONFIG['num_rows'],
            CONFIG['num_grids']
        )

        exist_out = self.exist_head(f)
        return lane_out, exist_out
    

model = LaneNet().to(device)
optimizer = optim.Adam(params = model.parameters(), lr = 1e-4,  weight_decay= 1e-4)   # Weight Decay (L2 Regularization)

scaler = GradScaler()

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.5
)


training_img_dataset = LaneDataset(img_dir= CONFIG['train_image_folder'], 
                                   label_file=CONFIG['train_label_json'])

training_loader = DataLoader(dataset = training_img_dataset, shuffle = True, batch_size = CONFIG['batch_size'])




def calculate_loss(pred_lane, pred_exist, lane_target, existance):
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    bce = nn.BCEWithLogitsLoss()

    lane_loss = ce(
        pred_lane.view(-1, CONFIG['num_grids']),
        lane_target.view(-1)
    )

    exist_loss = bce(
        pred_exist, existance
    )
    return lane_loss + exist_loss



def train_one_epoch():
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        img, lane_target, existance = data
        img = img.to(device)
        lane_target = lane_target.to(device)
        existance = existance.to(device)
    
        optimizer.zero_grad()  # IMPORTANT

        # Mixed Precision Forward Pass
        with autocast(device_type= 'cuda'):
            pred_lane, pred_exist = model(img)
            loss = calculate_loss(pred_lane, pred_exist, lane_target, existance)

        # Scaled Backward Pass
        scaler.scale(loss).backward()

        # Optimizer Step (scaled)
        scaler.step(optimizer)
        scaler.update()


        running_loss += loss.item()

        if i % 10 == 9:
            last_loss = running_loss / 10   # loss per batch
            print(" batch {} loss {}".format(i + 1, last_loss))
            running_loss = 0.
    return last_loss


def start_training_process():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    epoch_num = 0

    best_val_loss = 1_000_00.

    val_img_dataset = LaneDataset(img_dir= CONFIG['val_image_folder'], 
                                    label_file=CONFIG['val_label_json'])

    val_loader = DataLoader(dataset = val_img_dataset, shuffle = True, batch_size = CONFIG['batch_size'])


    for epoch in range(CONFIG['epochs']):
        print("Epoch {}".format(epoch+1))

        # make sure gradient tracking is on
        model.train()
        avg_loss = train_one_epoch()

        running_loss = 0.

        # set model to eval mode
        model.eval()

        with torch.no_grad():
            for i, val_data in enumerate(val_loader):
                img, lane_target, existance = val_data
                img = img.to(device)
                lane_target = lane_target.to(device)
                existance = existance.to(device)

                with autocast(device_type = 'cuda'):
                    pred_lane, pred_exist = model(img)
                    val_loss = calculate_loss(pred_lane, pred_exist, lane_target, existance)
                running_loss += val_loss.item()

        scheduler.step()
        print("Current LR:", optimizer.param_groups[0]['lr'])

        avg_val_loss = running_loss / (i + 1)
        print("Loss train {} valid {}".format(avg_loss, avg_val_loss))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(
                CONFIG['model_folder'],
                f'model_{timestamp}_{epoch_num+1}.pt'
            )
            torch.save(model.state_dict(), model_path)
        
        epoch_num += 1
    return model_path



LANE_COLORS = [
    (0,   0,   255),   # lane_0 → red
    (0,   165, 255),   # lane_1 → orange
    (0,   255, 255),   # lane_2 → yellow
    (0,   255, 0  ),   # lane_3 → green
    (255, 255, 0  ),   # lane_4 → cyan
    (255, 0,   0  ),   # lane_5 → blue
    (255, 0,   255),   # lane_6 → magenta
    (128, 0,   128),   # lane_7 → purple
]



def load_model(model_path: str) -> LaneNet:
    model = LaneNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from: {model_path}")
    return model


def preprocess(image_path: str):
    """Read image, resize, normalise → (1, 3, H, W) float32 tensor."""
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    resized = cv2.resize(orig, (CONFIG['img_w'], CONFIG['img_h']))
    tensor = (resized / 255.0).astype(np.float32)           # float32, not float64
    tensor = torch.from_numpy(tensor).permute(2, 0, 1)      # (3, H, W)
    tensor = tensor.unsqueeze(0).to(device)                  # (1, 3, H, W)
    return orig, resized, tensor


def decode_predictions(lane_out, exist_out, exist_threshold: float = 0.5):
    """
    lane_out  : (1, num_lanes, num_rows, num_grids) — raw logits
    exist_out : (1, num_lanes)                      — raw logits

    Returns:
        lane_points : list of num_lanes lists; each inner list is [(x, y), ...]
                      empty list means lane not detected
    """
    num_lanes  = CONFIG['num_lanes']
    num_rows   = CONFIG['num_rows']
    num_grids  = CONFIG['num_grids']
    img_w      = CONFIG['img_w']
    img_h      = CONFIG['img_h']

    # Existence: sigmoid → threshold
    exist_prob = torch.sigmoid(exist_out[0]).cpu().numpy()   # (num_lanes,)

    # Lane grid: argmax over grid dimension → grid index per row
    grid_idx = lane_out[0].argmax(dim=-1).cpu().numpy()      # (num_lanes, num_rows)

    lane_points = []
    for lane_i in range(num_lanes):
        if exist_prob[lane_i] < exist_threshold:
            lane_points.append([])
            continue

        pts = []
        for row_i in range(num_rows):
            g = grid_idx[lane_i, row_i]
            # Convert grid index → pixel x; row_anchor → pixel y
            x = int((g + 0.5) / num_grids * img_w)
            y = int(row_anchors[row_i])
            pts.append((x, y))
        lane_points.append(pts)

    return lane_points, exist_prob


def draw_lanes(image: np.ndarray, lane_points, exist_prob,
               orig_h: int, orig_w: int,
               dot_radius: int = 4,
               line_thickness: int = 2) -> np.ndarray:
    """
    Draw each active lane on a copy of `image` (already at model resolution).
    Then resize back to original dimensions.
    """
    vis = image.copy()

    scale_x = orig_w / CONFIG['img_w']
    scale_y = orig_h / CONFIG['img_h']

    for lane_i, pts in enumerate(lane_points):
        if not pts:
            continue

        color = LANE_COLORS[lane_i % len(LANE_COLORS)]

        # Scale points back to original image size
        scaled = [(int(x * scale_x), int(y * scale_y)) for x, y in pts]

        # Draw filled dots at each anchor row
        for (x, y) in scaled:
            cv2.circle(vis, (x, y), dot_radius, color, -1)

        # Draw connecting lines between consecutive points
        for j in range(len(scaled) - 1):
            cv2.line(vis, scaled[j], scaled[j + 1], color, line_thickness)

        # Label the lane
        lx, ly = scaled[-1]        # topmost point (row_anchor goes top→bottom in index order)
        label = f"L{lane_i} {exist_prob[lane_i]:.2f}"
        cv2.putText(vis, label, (lx + 6, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return vis


def add_legend(image: np.ndarray, lane_points, exist_prob) -> np.ndarray:
    """Overlay a small legend in the top-left corner."""
    active = [(i, exist_prob[i]) for i, pts in enumerate(lane_points) if pts]
    if not active:
        return image

    pad    = 8
    line_h = 22
    box_h  = pad * 2 + line_h * len(active)
    box_w  = 160

    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    for idx, (lane_i, prob) in enumerate(active):
        color = LANE_COLORS[lane_i % len(LANE_COLORS)]
        y = pad + idx * line_h + line_h // 2
        cv2.rectangle(image, (pad, y - 6), (pad + 16, y + 6), color, -1)
        cv2.putText(image, f"Lane {lane_i}  {prob:.2f}",
                    (pad + 22, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return image


def run_inference(model_path: str,
                  input_path: str,
                  output_dir: str = "inference_output",
                  exist_threshold: float = 0.5):
    """
    model_path      : path to your .pt model file
    input_path      : path to a single image OR a folder of images
    output_dir      : where to save visualised results
    exist_threshold : minimum sigmoid score to consider a lane active
    """
    os.makedirs(output_dir, exist_ok=True)
    model = load_model(model_path)

    # Collect image paths
    p = Path(input_path)
    if p.is_dir():
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_paths = [f for f in p.iterdir() if f.suffix.lower() in exts]
    else:
        image_paths = [p]

    if not image_paths:
        print("No images found.")
        return

    print(f"Running inference on {len(image_paths)} image(s)…\n")

    for img_path in image_paths:
        print(f"  Processing: {img_path.name}")

        orig, resized, tensor = preprocess(str(img_path))
        orig_h, orig_w = orig.shape[:2]

        with torch.no_grad():
            lane_out, exist_out = model(tensor)

        lane_points, exist_prob = decode_predictions(lane_out, exist_out, exist_threshold)

        # Draw on original resolution image
        vis = draw_lanes(orig, lane_points, exist_prob, orig_h, orig_w)
        vis = add_legend(vis, lane_points, exist_prob)

        # Print which lanes were detected
        detected = [i for i, pts in enumerate(lane_points) if pts]
        print(f"    Detected lanes: {detected}")
        print(f"    Exist scores:   {[f'{p:.3f}' for p in exist_prob]}")

        out_path = Path(output_dir) / f"pred_{img_path.name}"
        cv2.imwrite(str(out_path), vis)
        print(f"    Saved → {out_path}\n")

    print(f"Done. All results saved to: {output_dir}/")



# model_path = start_training_process()

run_inference(
    # model_path      = model_path,
    model_path      = r"C:\Users\jites\Desktop\Personal_projects\Pytorch_code\models\model_20260318_233236_85.pt",
    input_path      = r"C:\Users\jites\Desktop\Personal_projects\Pytorch_code\dummy_dataset\test_image\test_0000.jpg",
    output_dir      = CONFIG['model_folder'],
    exist_threshold = 0.5,
)



# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="LaneNet inference visualiser")
#     parser.add_argument("--model",     required=True,          help="Path to .pt model weights")
#     parser.add_argument("--input",     required=True,          help="Image file or folder")
#     parser.add_argument("--output",    default="inference_output", help="Output folder (default: inference_output)")
#     parser.add_argument("--threshold", type=float, default=0.5,   help="Lane existence threshold (default: 0.5)")
#     args = parser.parse_args()

#     run_inference(
#         model_path      = args.model,
#         input_path      = args.input,
#         output_dir      = args.output,
#         exist_threshold = args.threshold,
#     )