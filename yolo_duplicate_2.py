import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
import cv2
import torch.nn.functional as F
from torch.optim import Adam
from torch.amp.grad_scaler import GradScaler
from torch.amp import autocast
import random
import numpy as np


torch.cuda.empty_cache()



YOLO_IMG_WIDTH = 320
YOLO_IMG_HEIGHT = 320
BATCH_SIZE = 3
EPOCHS = 25
MODEL_FOLDER_PATH = Path(__file__).parent
IMAGE_TRAIN_FOLDER_PATH = r"C:\Users\jites\Downloads\archive\yolo_data_sindhi_label\images\training"
LABEL_TRAIN_FOLDER_PATH = r"C:\Users\jites\Downloads\archive\yolo_data_sindhi_label\labels\training"
IMAGE_VAL_FOLDER_PATH = r"C:\Users\jites\Downloads\archive\yolo_data_sindhi_label\images\val"
LABEL_VAL_FOLDER_PATH = r"C:\Users\jites\Downloads\archive\yolo_data_sindhi_label\labels\val"
GRID_SIZE = 20



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def horizontal_flip(img, boxes):

    if random.random() < 0.5:
        img = np.fliplr(img).copy()
        boxes[:,1] = 1 - boxes[:,1]

    return img, boxes


# Random Brightness / Contrast
def color_jitter(img):

    if random.random() < 0.5:
        alpha = 1 + random.uniform(-0.3,0.3)
        beta = random.uniform(-0.1,0.1)

        img = img * alpha + beta
        img = np.clip(img,0,1)

    return img


# Helps with size variation. object appears smaller/larger
def random_scale(img, boxes):

    scale = random.uniform(0.8,1.2)
    h,w = img.shape[:2]

    new_w = int(w*scale)
    new_h = int(h*scale)

    img = cv2.resize(img,(new_w,new_h))
    img = cv2.resize(img,(w,h))

    boxes[:,3] *= scale
    boxes[:,4] *= scale

    boxes[:,3] = boxes[:,3].clip(0,1)
    boxes[:,4] = boxes[:,4].clip(0,1)

    return img, boxes



def random_translate(img, boxes):

    if random.random() < 0.5:
        tx = random.uniform(-0.1,0.1)
        ty = random.uniform(-0.1,0.1)

        boxes[:,1] += tx
        boxes[:,2] += ty

        boxes[:,1] = boxes[:,1].clip(0,1)
        boxes[:,2] = boxes[:,2].clip(0,1)

    return img, boxes


class ImageDataset(Dataset):

    def __init__(self, img_w, img_h, img_folder, label_folder):
        self.img_w = img_w
        self.img_h = img_h
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.images = os.listdir(img_folder)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_name = self.images[index]
        img_path = os.path.join(self.img_folder, image_name)

        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_w, self.img_h))
        img = img / 255     # bring RGB values < 1

        label_path = os.path.join(self.label_folder, image_name.replace('.jpg', '.txt'))
        boxes = []
        with open(label_path, 'r') as f:
            # lines = f.readline()
            # cls_id, center_x, center_y, w, h = map(float, lines.split())
            # cls_id = 0.0 if cls_id == 5.0 else cls_id

            for line in f.readlines():
                cls_id, center_x, center_y, w, h = map(float, line.split())
                cls_id = 0.0 if cls_id == 5.0 else cls_id
                boxes.append([cls_id, center_x, center_y, w, h])

        boxes = torch.tensor(boxes, dtype=torch.float32)

        boxes = boxes.numpy()
        img, boxes = horizontal_flip(img, boxes)
        img = color_jitter(img)
        img, boxes = random_translate(img, boxes)
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # (3, 320, 320) pytorch model format
        boxes = torch.tensor(boxes, dtype=torch.float32)

        return img, boxes



def collate_fn(batch):
    images = []
    targets = []
    for img, boxes in batch:
        images.append(img)
        targets.append(boxes)

        # torch.stack concatenates a sequence of tensors along a new dimension
        # Stack along dim=0 (default)
    images = torch.stack(images, dim=0)
    return images, targets



class DetectionModel(nn.Module):

    def __init__(self, num_classes, grid_size):

        super().__init__()

        self.num_classes = num_classes
        self.grid_size = grid_size

        self.feature = nn.Sequential(
            # (B, 3, 320, 320)
            nn.Conv2d(3,32,3,padding=1),    # (B, 32, 320, 320)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                # o/p size will be (B, 32, 160, 160)

            nn.Conv2d(32,64,3,padding=1),   # (B, 64, 160, 160)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                # (B, 64, 80, 80)

            nn.Conv2d(64,128,3,padding=1),      # (B, 128, 80, 80)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # (B, 128, 40, 40)

            nn.Conv2d(128,256,3,padding=1),     # (B, 256, 40, 40)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # (B, 256, 20, 20)

        )

        self.detect = nn.Conv2d(                # (B, 256, 20, 20)
            256,
            5 + num_classes,
            kernel_size=1
        )                                       # (B, 5+C, 20, 20)

    def forward(self,x):

        x = self.feature(x)
        x = self.detect(x)
        x = x.permute(0,2,3,1)
        # stabilize bbox predictions
        x[...,0:2] = torch.sigmoid(x[...,0:2])
        x[...,2:4] = torch.sigmoid(x[...,2:4])
        return x
    


training_img_dataset = ImageDataset(img_w = YOLO_IMG_WIDTH, img_h = YOLO_IMG_HEIGHT,
                           img_folder = IMAGE_TRAIN_FOLDER_PATH, label_folder = LABEL_TRAIN_FOLDER_PATH)

training_loader = DataLoader(dataset = training_img_dataset, shuffle = True, batch_size = BATCH_SIZE,
                        collate_fn=collate_fn
)
# for i, val_data in enumerate(loader):
#     val_img, val_label = val_data

model = DetectionModel(num_classes = 5, grid_size = GRID_SIZE).to(device)
optimizer = Adam(params = model.parameters(), lr = 1e-4,  weight_decay= 1e-4)   # Weight Decay (L2 Regularization)

scaler = GradScaler()

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.5
)


def build_target(targets, num_classes, grid_size ):
    batch = len(targets)
    new_targets = torch.zeros(size = (batch, grid_size, grid_size, 5+num_classes), dtype=torch.float32)

    for b in range(batch):
        for obj in targets[b]:
            cls_id, x, y, w, h = obj

            # cell_x = int(x * grid_size)
            # cell_y = int(y * grid_size)
            cell_x = min(int(x * grid_size), grid_size - 1)     # whcih cell of the grid
            cell_y = min(int(y * grid_size), grid_size - 1)     # whcih cell of the grid

            x_cell = x * grid_size - cell_x
            y_cell = y * grid_size - cell_y

                # [x_cell, y_cell, w, h, objectness, class_probs...]
                # [x, y, w, h, obj, c1, c2, c3, c4, c5]
                # pred[b, i, j] = [x_cell, y_cell, w, h, objectness, class1, class2, ...]
            new_targets[b, cell_y, cell_x, 0:4] = torch.tensor([x_cell, y_cell, w, h], dtype=torch.float32)
                # objectness probability 100 % at the 5 cell beacuse one object is present
            new_targets[b, cell_y, cell_x, 4] = 1.0
                # add probability 1 at the class_id cell
            new_targets[b, cell_y, cell_x, 5+int(cls_id)] = 1.0

    return new_targets


# def calculate_loss(pred, actual):
#     new_target = build_target(targets = actual, num_classes = 5, grid_size = GRID_SIZE).to(device)
#     return F.mse_loss(input = pred, target = new_target)



def calculate_loss(pred, targets):

    target = build_target(
        targets = targets, num_classes = 5, grid_size = GRID_SIZE
    ).to(device)

    obj_mask = target[...,4] == 1
    noobj_mask = target[...,4] == 0

    # -----------------------
    # BOX LOSS
    # -----------------------

    pred_xy = pred[...,0:2][obj_mask]
    target_xy = target[...,0:2][obj_mask]

    pred_wh = pred[...,2:4][obj_mask]
    target_wh = target[...,2:4][obj_mask]

    box_loss = (
        F.mse_loss(pred_xy, target_xy) +
        F.mse_loss(
            torch.sqrt(pred_wh + 1e-6),
            torch.sqrt(target_wh + 1e-6)
        )
    )

    # -----------------------
    # OBJECT LOSS
    # -----------------------

    obj_loss = F.binary_cross_entropy_with_logits(
        pred[...,4][obj_mask],
        target[...,4][obj_mask]
    )

    # -----------------------
    # NO OBJECT LOSS
    # -----------------------

    noobj_loss = F.binary_cross_entropy_with_logits(
        pred[...,4][noobj_mask],
        target[...,4][noobj_mask]
    )

    # -----------------------
    # CLASS LOSS
    # -----------------------

    class_loss = F.binary_cross_entropy_with_logits(
        pred[...,5:][obj_mask],
        target[...,5:][obj_mask]
    )

    # -----------------------
    # TOTAL LOSS
    # -----------------------

    total_loss = (
        5 * box_loss +
        obj_loss +
        0.1 * noobj_loss +
        class_loss
    )

    return total_loss



def train_one_epoch():
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        img, labels = data
        img = img.to(device)
        labels = [t.to(device) for t in labels]

        # # # zero your gradients for every epochs
        # optimizer.zero_grad()

        # # make prediction
        # output = model(img)

        # # compute loss and its gradients
        # loss = calculate_loss(output, labels)
        # loss.backward()

        # # adjust lerning weights
        # optimizer.step()

        optimizer.zero_grad()  # IMPORTANT

        # Mixed Precision Forward Pass
        with autocast(device_type= 'cuda'):
            output = model(img)
            loss = calculate_loss(output, labels)

        # Scaled Backward Pass
        scaler.scale(loss).backward()

            # Detection models sometimes have exploding gradients. This improves stability.
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

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

    val_img_dataset = ImageDataset(img_w = YOLO_IMG_WIDTH, img_h = YOLO_IMG_HEIGHT,
                            img_folder = IMAGE_VAL_FOLDER_PATH, label_folder = LABEL_VAL_FOLDER_PATH)
    val_loader = DataLoader(dataset = val_img_dataset, shuffle = False, batch_size = BATCH_SIZE,
                            collate_fn=collate_fn)


    for epoch in range(EPOCHS):
        print("Epoch {}".format(epoch+1))

        # make sure gradient tracking is on
        model.train()
        avg_loss = train_one_epoch()

        running_loss = 0.

        # set model to eval mode
        model.eval()

        with torch.no_grad():
            # for i, val_data in enumerate(val_loader):
            #     val_img, val_label = val_data
            #     val_ouput = model(val_img)
            #     val_loss = calculate_loss(val_ouput, val_label)
            #     running_loss += val_loss
            for i, val_data in enumerate(val_loader):
                val_img, val_label = val_data
                val_img = val_img.to(device)
                val_label = [t.to(device) for t in val_label]

                with autocast(device_type = 'cuda'):
                    val_output = model(val_img)
                    val_loss = calculate_loss(val_output, val_label)

                running_loss += val_loss.item()

        scheduler.step()
        print("Current LR:", optimizer.param_groups[0]['lr'])

        avg_val_loss = running_loss / (i + 1)
        print("Loss train {} valid {}".format(avg_loss, avg_val_loss))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(
                MODEL_FOLDER_PATH,
                f'model_{timestamp}_{epoch_num+1}.pt'
            )
            torch.save(model.state_dict(), model_path)
        
        epoch_num += 1
    torch.save(model.state_dict(), 'last_model.pt')
    return model_path




def extract_boxes_from_prediction(pred, conf_threshold=0.9, grid_size=6):
    boxes = []
    scores = []
    classes = []

    pred = pred.squeeze(0)  # remove batch dim

    for row in range(grid_size):
        for col in range(grid_size):
            cell_pred = pred[row, col]
            conf = cell_pred[4]
            if conf > conf_threshold:
                x_cell, y_cell, w, h = cell_pred[0:4]

                # convert to global coordinates
                x = (col + x_cell) / grid_size
                y = (col + y_cell) / grid_size

                # convert to corner format
                x1 = x - w / 2
                x2 = x + w / 2
                y1 = y - h / 2
                y2 = y + h / 2

                class_scores = cell_pred[5:]
                    # Predicted class 
                cls = torch.argmax(class_scores)
                boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
                scores.append(conf.item())
                classes.append(cls.item())
        
    return boxes, scores, classes



def yolo_coords_to_pixel_coords(boxes):
    pixel_boxes = []

    for x1, y1, x2, y2 in boxes:
        pixel_boxes.append(
            [
                int(x1 * YOLO_IMG_WIDTH),
                int(y1 * YOLO_IMG_HEIGHT),
                int(x2 * YOLO_IMG_WIDTH),
                int(y2 * YOLO_IMG_HEIGHT)
            ]
        )
    return pixel_boxes



def draw_boxes(img, boxes, classes, scores):
    for box, cls, conf in zip(boxes, classes, scores):
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{cls} - {conf}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img



def get_prediction(model_path):
    model = DetectionModel(num_classes = 5, grid_size = GRID_SIZE).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()

    img = cv2.imread(r"C:\Users\jites\Downloads\archive\yolo_data_sindhi_label\images\val\pexels-vlad-chetan-3121979.jpg")
    img = cv2.resize(img, (YOLO_IMG_WIDTH, YOLO_IMG_HEIGHT))
    img_scale = img / 255.0
    img_tens = torch.tensor(img_scale, dtype= torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tens)

    boxes, scores, classes = extract_boxes_from_prediction(output)
    boxes = yolo_coords_to_pixel_coords(boxes)
    output_img = draw_boxes(img, boxes, classes, scores)
    
    cv2.imshow('draw bbox', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# pt = start_training_process()
# get_prediction(pt)

get_prediction(r"C:\Users\jites\Desktop\Personal_projects\last_model")

