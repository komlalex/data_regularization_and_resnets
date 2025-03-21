import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split 

from torchvision.transforms import ToTensor
from torchvision.utils import make_grid 
from torchvision.datasets import CIFAR10

import matplotlib.pyplot as plt 
import numpy as np  

datasets = CIFAR10(root="data/", 
                   download=True, 
                   transform=ToTensor())
test_ds = CIFAR10(root="data/", 
                  download=True,
                  train=False, 
                  transform=ToTensor())

class_names = datasets.classes

def show_image(img, label): 
    plt.figure(figsize=(10, 15)) 
    plt.imshow(img.permute(1, 2, 0)) 
    plt.axis(False) 
    plt.title(f"Label: {class_names[label]}")   

show_image(*datasets[0])
show_image(*datasets[10])

"""Create training and validation sets""" 
VAL_SIZE = 5_000 
TRAIN_SIZE = len(datasets) - VAL_SIZE 

train_ds, val_ds = random_split(datasets, [TRAIN_SIZE, VAL_SIZE])   

"""Create data loaders""" 
BATCH_SIZE = 128
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE*2, pin_memory=True) 
test_dl = DataLoader(test_ds, batch_size= BATCH_SIZE*2, pin_memory=True) 

"""View a batch using make_grid""" 
def show_batch(batch): 
    images, _ = batch 
    plt.figure(figsize=(10, 15))
    plt.imshow(make_grid(images, nrow=16).permute(1, 2, 0)) 
    plt.axis(False) 
    plt.title("Sample Batch") 

for batch in train_dl: 
    show_batch(batch)  
    break 

"""Create Device Agnostic code""" 
def get_default_device(): 
    """Get GPU if available, else CPU"""
    if torch.cuda.is_available(): 
        return torch.device("cuda") 
    else: 
        return torch.device("cpu")

device = get_default_device() 

def to_device(data, device): 
    """Sends data to appropriate device""" 
    if isinstance(data, (list, tuple)): 
        return [to_device(x, device) for x in data] 
    return data.to(device) 

class DeviceDataLoader(): 
    """Wraps a data loader to send it to GPU if available, else CPU"""
    def __init__(self, dl, device):
        self.dl = dl 
        self.device = device 
    
    def __iter__(self): 
        for batch in self.dl: 
            yield to_device(batch, device) 

    def __len__(self): 
        return len(self.dl) 
    
"""Now, let's wrap our data loaders""" 
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
test_dl = DeviceDataLoader(test_dl, device) 

"""Define an accuracy function to compute the model's accruacy""" 
def accuracy(outputs, y_true): 
    y_preds = torch.argmax(outputs, dim=1) 
    return torch.tensor(torch.sum(y_true==y_preds).item() / len(y_true))


"""Model"""
class ImageClassificationBase(nn.Module): 

    def training_step(self, batch): 
        images, labels = batch 
        outputs = self(images) 
        loss = F.cross_entropy(outputs, labels) 
        return loss  
    
    def validation_step(self, batch): 
        images, labels = batch 
        outputs = self(images) 
        loss = F.cross_entropy(outputs, labels) 
        acc = accuracy(outputs, labels)
        return {"val_loss": loss.detach(), "val_acc": acc} 
    
    def validation_epoch_end(self, result): 
        batch_losses = [x["val_loss"] for x in result]
        epoch_loss = torch.stack(batch_losses).mean().item() 
        batch_accs = [x["val_acc"] for x in result] 
        epoch_acc = torch.stack(batch_accs).mean().item() 
        return {"val_loss": epoch_loss, "val_acc": epoch_acc} 
    
    def epoch_end(self, epoch, result): 
        print(f"\33[32m Epoch: {epoch+1} | train_loss: {result["train_loss"]:.4f} | val_loss: {result["val_loss"]:.4f} | val_acc: {result["val_acc"]:.4f}") 
    
class Cfar10CnnModel(ImageClassificationBase): 
    def __init__(self):
        super().__init__() 
        self.network = nn.Sequential(  

            # Convolutional layers
            # Input: 3 x 32 x 32
            nn.Conv2d(3, 32 ,kernel_size=3, stride=1, padding=1), # Output 32 x 32 x 32
            nn.ReLU(), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Output 64 x 32 x 32  
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), # Output 64 x 16 x 16 

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # Output: 128 x 32 x 32
            nn.ReLU(), 
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # OUtput: 128 x 32 x 32 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), # Output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # Output: 256 x 8 x 8
            nn.ReLU(), 
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # Output: 256 x 8 x 8
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), # Output: 256 x 4 x 4 

            # Classification layers
            nn.Flatten(), # Return: 256 * 4 * 4 = 4096 
            nn.Linear(256*4*4, 1024), # Output: 1024
            nn.ReLU(), 
            nn.Linear(1024, 512), # Output: 1024
            nn.ReLU(),
            nn.Linear(512, 10) 
        )
    def forward(self, x) -> torch.Tensor: 
        return self.network(x) 

"""TRAIN THE MODEL""" 
def evaluate(model:nn.Module, val_dl: DataLoader): 
    model.eval() 
    with torch.inference_mode():
        outputs= [model.validation_step(batch) for batch in val_dl] 
        return model.validation_epoch_end(outputs)  

def fit(num_epochs: int, lr: float, model:nn.Module, train_dl: DataLoader, val_dl: DataLoader, opt_func=torch.optim.Adam): 
    optimizer = opt_func(model.parameters(), lr) 
    history = [] 

    for epoch in range(num_epochs):
        # Training phase 
        model.train()
        train_losses = []
        for batch in train_dl: 
            loss = model.training_step(batch) 
            train_losses.append(loss)
            loss.backward() 
            optimizer.step() 
            optimizer.zero_grad()  

        result = evaluate(model, val_dl) 
        result["train_loss"] = torch.stack(train_losses).mean().item()  
        model.epoch_end(epoch, result)
        history.append(result)  
    return history 

model = Cfar10CnnModel() 
to_device(model, device) 

"""Train model""" 
opt_func = torch.optim.Adam 
lr = 0.005
fit(10, lr, model, train_dl, val_dl, opt_func) 




