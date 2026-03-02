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


torch.cuda.empty_cache()



YOLO_IMG_WIDTH = 320
YOLO_IMG_HEIGHT = 320
BATCH_SIZE = 3
EPOCHS = 4
MODEL_FOLDER_PATH = Path(__file__).parent
IMAGE_TRAIN_FOLDER_PATH = r"C:\Users\jites\Downloads\archive\yolo_data_sindhi_label\images\training"
LABEL_TRAIN_FOLDER_PATH = r"C:\Users\jites\Downloads\archive\yolo_data_sindhi_label\labels\training"
IMAGE_VAL_FOLDER_PATH = r"C:\Users\jites\Downloads\archive\yolo_data_sindhi_label\images\val"
LABEL_VAL_FOLDER_PATH = r"C:\Users\jites\Downloads\archive\yolo_data_sindhi_label\labels\val"
GRID_SIZE = 6



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # (3, 320, 320) pytorch model format

        label_path = os.path.join(self.label_folder, image_name.replace('.jpg', '.txt'))
        boxes = []
        with open(label_path, 'r') as f:
            lines = f.readline()
            cls_id, center_x, center_y, w, h = map(float, lines.split())
            cls_id = 0.0 if cls_id == 5.0 else cls_id
            boxes.append([cls_id, center_x, center_y, w, h])

        boxes = torch.tensor(boxes, dtype=torch.float32)

        return img, boxes
    


class DetectionModel(nn.Module):

    def __init__(self, num_classes, grid_size):     # grid_size means the image will devided into 8 * 8 grid
        super().__init__()

        self.num_classes = num_classes
        self.grid_size = grid_size

        self.feature =nn.Sequential(
            # (B, 3, 320, 320)
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 1, padding = 1, dtype=torch.float32),
            # ((W - K + 2p)/S) + 1
            # ((320 - 3 + 2*1)/1) + 1 = 320
            # (B, 16, 320, 320)
            nn.ReLU(),  # Size will not change
            nn.MaxPool2d(kernel_size = 2),  # o/p size will be (B, 16, 160, 160)
            
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, dtype=torch.float32),
            # ((W - K + 2p)/S) + 1
            # ((160 - 3 + 2*1)/1) + 1 = 160
            # (B, 32, 160, 160)
            nn.ReLU(),  # Size will not change
            nn.MaxPool2d(kernel_size = 2),  # o/p size will be (B, 32, 80, 80)
            
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, dtype=torch.float32),
            # ((W - K + 2p)/S) + 1
            # ((80 - 3 + 2*1)/1) + 1 = 80
            # (B, 64, 80, 80)
            nn.ReLU(),  # Size will not change
            nn.MaxPool2d(kernel_size = 2),  # o/p size will be (B, 64, 40, 40)
            
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, dtype=torch.float32),
            # ((W - K + 2p)/S) + 1
            # ((40 - 3 + 2*1)/1) + 1 = 40
            # (B, 128, 40, 40)
            nn.ReLU(),  # Size will not change
            nn.MaxPool2d(kernel_size = 2),  # o/p size will be (B, 128, 20, 20)

                # forch output 
            # nn.AdaptiveAvgPool2d(output_size = (self.grid_size, self.grid_size))
        )

        self.detection = nn.Sequential(
            nn.Flatten(),   # o/p size (B, 128*20*20)
            nn.Linear(in_features = 128*20*20, out_features = 4056, dtype=torch.float32),  # o/p size (B, 4056)
            nn.ReLU(),
            nn.Linear(in_features = 4056, out_features = self.grid_size*self.grid_size*(5+self.num_classes),
                      dtype=torch.float32),
                # 5 + self.num_classes => Yolo model label data has 5 values (cls_id, center_x, center_y, w, h)
                #                         And the and the number of classes because the model will predict every class
        )

    def forward(self, x):
            # input (B, 3, 320, 320)
        x = self.feature(x)     # output (B, 128, 20, 20)
        x = self.detection(x)   # output (B, grid * grid * (5 + num_classes))
        x = x.view(-1, self.grid_size, self.grid_size, 5+self.num_classes)    # output (B, grid, grid, 5+self.num_classes)
        return x
    


img_dataset = ImageDataset(img_w = YOLO_IMG_WIDTH, img_h = YOLO_IMG_HEIGHT,
                           img_folder = IMAGE_TRAIN_FOLDER_PATH, label_folder = LABEL_TRAIN_FOLDER_PATH)

loader = DataLoader(dataset = img_dataset, shuffle = True, batch_size = BATCH_SIZE)
# for i, val_data in enumerate(loader):
#     val_img, val_label = val_data

model = DetectionModel(num_classes = 5, grid_size = GRID_SIZE).to(device)
optimizer = Adam(params = model.parameters(), lr = 0.001)

scaler = GradScaler()

def build_target(targets, num_classes, grid_size ):
    batch = len(targets)
    new_targets = torch.zeros(size = (batch, grid_size, grid_size, 5+num_classes), dtype=torch.float32)

    for b in range(batch):
        for obj in targets[b]:
            cls_id, x, y, w, h = obj

            cell_x = int(x * grid_size)
            cell_y = int(y * grid_size)

            x_cell = x * grid_size - cell_x
            y_cell = y * grid_size - cell_y

            new_targets[b, cell_y, cell_x, 0:4] = torch.tensor([x_cell, y_cell, w, h], dtype=torch.float32)
            new_targets[b, cell_y, cell_x, 4] = 1.0
            new_targets[b, cell_y, cell_x, 5+int(cls_id)] = 1.0

    return new_targets


def calculate_loss(pred, actual):
    new_target = build_target(targets = actual, num_classes = 5, grid_size = GRID_SIZE).to(device)
    return F.mse_loss(input = pred, target = new_target)


def train_one_epoch():
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(loader):
        img, labels = data
        img = img.to(device)
        labels = labels.to(device)

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
    val_loader = DataLoader(dataset = val_img_dataset, shuffle = True, batch_size = BATCH_SIZE)


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

                with autocast(device_type = 'cuda'):
                    val_output = model(val_img)
                    val_loss = calculate_loss(val_output, val_label)

                running_loss += val_loss.item()


        avg_val_loss = running_loss / (i + 1)
        print("Loss train {} valid {}".format(avg_loss, avg_val_loss))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(
                MODEL_FOLDER_PATH,
                f'model_{timestamp}_{epoch_num+1}'
            )
            torch.save(model.state_dict(), model_path)
        
        epoch_num += 1
    return "Training completed"


# print(start_training_process())


def extract_boxes_from_prediction(pred, conf_threshold=0.1, grid_size=6):
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
    print(boxes)
    print(classes)
    print(scores)
    for box, cls, conf in zip(boxes, classes, scores):
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{cls} - {conf}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img



def get_prediction():
    model = DetectionModel(num_classes = 5, grid_size = GRID_SIZE).to(device)
    checkpoint = torch.load(r"C:\Users\jites\Desktop\Personal_projects\Pytorch_code\model_20260302_112701_3")
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


# get_prediction()

