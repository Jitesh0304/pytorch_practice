import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.model_selection import train_test_split
# PyTorch TensorBoard support
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from pathlib import Path



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
INPUT_DIM = 10
OUTPUT_DIM = 3
DATA_SIZE = 1000
EPOCHS = 10
MODEL_PATH = os.path.join(Path(__file__).parent, "ann_model.pth")


x = torch.randint(low = 1, high = 50, size = (DATA_SIZE, INPUT_DIM), dtype = torch.float32, requires_grad = False,
                  device = device)
y = torch.randint(low = 1, high = 10, size = (DATA_SIZE, OUTPUT_DIM), dtype = torch.float32, requires_grad = False,
                  device = device)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)



class DummyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.to(device)
        self.y = y.to(device)

    def __len__(self):
        return len(x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    


class SimpleANN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleANN, self).__init__()

        self.layer1 = nn.Linear(in_features= input_dim, out_features= 20,
                                device= device)
        self.drp = nn.Dropout(p = 0.1)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(in_features= 20, out_features= output_dim, device= device)
        
    
    def forward(self, x):
        x = self.drp(self.layer1(x))
        x = self.relu(x)
        x = self.layer2(x)
        return x


dataset_obj = DummyDataset(x, y)
data_loader = DataLoader(dataset= dataset_obj, batch_size= BATCH_SIZE, shuffle= True)

model = SimpleANN(input_dim= INPUT_DIM, output_dim= OUTPUT_DIM).to(device)
optimizer = Adam(params= model.parameters(), lr= 0.002)

def calculate_loss(prediction, actual):
    return F.mse_loss(input= prediction, target= actual)


def start_basic_training_process():
    for i in range(EPOCHS):
        for data, label in data_loader:
            pred = model(data)

            loss = calculate_loss(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch: {i}, Loss: {loss.item()}")

    print("     ------  EVALUATION  ------     ")
    print(model.eval())

    torch.save(model.state_dict(), MODEL_PATH)
    return "Basic training Completed"


def predict_result(input_data, model_path):
    model = SimpleANN(input_dim= INPUT_DIM, output_dim= OUTPUT_DIM).to(device)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)

    input_data.to(device)

    with torch.no_grad():
        pred = model(input_data)
    print(pred)

    return pred

# predict_result(input_data= torch.randint(low = 1, high = 50, size = (1, INPUT_DIM), dtype = torch.float32, requires_grad = False,
#                   device = device), model_path = MODEL_PATH)




def train_one_epoch(epoch_index):   # tb_writer
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(data_loader) instead of
    # iter(data_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(data_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = calculate_loss(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(data_loader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss



def start_model_training_process():
        # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0


    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number)    # writer


        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(data_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = calculate_loss(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
        # writer.add_scalars('Training vs. Validation Loss',
        #                 { 'Training' : avg_loss, 'Validation' : avg_vloss },
        #                 epoch_number + 1)
        # writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = os.path.join(Path(__file__).parent, 'ann_model_{}_{}'.format(timestamp, epoch_number))
            torch.save(model.state_dict(), model_path)

        epoch_number += 1




