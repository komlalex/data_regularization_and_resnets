import sys  

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split 

import torchvision.transforms as tf
from torchvision.utils import make_grid 
from torchvision.datasets import CIFAR10

import matplotlib.pyplot as plt 
import numpy as np  

"""In addition to ToTensor transform, we'll also apply some other transforms to the images.
There are a few important changes we'll make while creating PyTorch datasets for training and validation: 

1. Use test set for validation: Instead of setting aside a fraction (e.g. 10%) of 
the data for validation, we'll simply use the test set as our validation set. This gives us 
s little more data to train with. In general, once you picked the best model architecture & hyperparameters using 
a fixed validation set, it is a good idea to retrain the same model on the entire datset just 
to give it a small final boost in performance. 

2. Channel-wise data normalization: We will normalize the image tensors by sibtracting 
the mean and dividing by the standard deviation across each channel. As a result, the mean of the data across 
each channel is 10, and the standard deviation is 1. Normalizing the data prevents the values from any one 
channel from disproportionately affecting the loss and gradients while training, simply by having a 
higher or wider range of values than others. 

3. Randomized data augmentation: We will apply randomly chosen transformations while loading images 
from the training dataset. Specifically, we will pad each image by 4 pixels , and then take a random crop of 
size 32 x 32 pixels, then flip the image horizontally with 50% probability. Since the transformation will be applied randomly and 
dynamicaly each time a particular image is loaded, the model sees slightly different images in each epoch of training, which 
allows it to generalize.""" 

# Data transforms (normalization & augmentation)
STATS = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
train_tfms = tf.Compose([tf.RandomCrop(32, padding=4, padding_mode="reflect"), 
                         tf.RandomHorizontalFlip(), 
                         #tf.RandomRotation(), 
                         #tf.RandomResizedCrop(256, scale=(0.5, 0.9), ratio=(1, 1)), 
                         #tf.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), 
                         tf.ToTensor(), 
                         tf.Normalize(*STATS, inplace=True)
                         ]) 

val_tfms = tf.Compose([tf.ToTensor(), 
                       tf.Normalize(*STATS)
                       ]) 

train_ds = CIFAR10(root="data/", 
                   download=True, 
                   transform=train_tfms)
val_ds = CIFAR10(root="data/", 
                  download=True,
                  train=False, 
                  transform=val_tfms)

class_names = train_ds.classes


"""Next, we create data loaders for retrieving images in batches.""" 
BATCH_SIZE = 128
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE*2, pin_memory=True) 

"""Let's take a look at some samples images from the training dataloader. To display the images, we'll need to 
denormalize the pixel values to bring them back into the range (0, 1). 
""" 

def denormalize(images, means, stds): 
    means = torch.tensor(means).reshape(1, 3, 1, 1) 
    stds  = torch.tensor(stds).reshape(1, 3, 1, 1)  
    return images * stds + means   

def show_example(img: torch.Tensor, label): 
    denorm_image = denormalize(img, *STATS)
    plt.figure(figsize=(10, 15)) 
    plt.imshow(denorm_image.permute(1, 2, 0)) 
    plt.axis(False) 
    plt.title(f"Label: {class_names[label]}")   

#show_example(*train_ds[0])
#show_example(*train_ds[10])


"""View a batch using make_grid""" 
def show_batch(batch): 
    images, _ = batch 
    denorm_images = denormalize(images, *STATS)
    plt.figure(figsize=(10, 15))
    plt.imshow(make_grid(denorm_images, nrow=16).permute(1, 2, 0).clamp(0, 1)) 
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
        """Yield a batch of data after moving it to device"""
        for batch in self.dl: 
            yield to_device(batch, device) 

    def __len__(self): 
        """Number of batches"""
        return len(self.dl)  

"""Get the default device"""  
device = get_default_device() 
    
"""Now, let's wrap our data loaders""" 
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)


"""Define an accuracy function to compute the model's accruacy""" 
def accuracy(outputs, y_true): 
    y_preds = torch.argmax(outputs, dim=1) 
    return torch.tensor(torch.sum(y_true==y_preds).item() / len(y_true))


"""Model with Residual Bloacks and  Batch Normalization 
One of the key changes to our CNN model this time is the addition of residual
 block, which adds the original input back to the output feature obtained by passing the 
 input through one or more convolutional layers""" 

"""Here's a very simple Residual block."""
class SimpleResidualBlock(nn.Module): 
    def __init__(self):
        super().__init__() 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU() 
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU() 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        out = self.conv1(x) 
        out = self.relu1(out) 
        out = self.conv2(out) 
        out = self.relu2(out) 
        return out + x # ReLU can be applied before or after adding the input 

simple_resnet = SimpleResidualBlock()
to_device(simple_resnet, device) 

for images, labels in train_dl:
    out = simple_resnet(images) 
    #print(out.shape) 
    break 


"""This seemingly small change produces a drastic change in the perfromance 
of the model. Also, we'll add a batch normalization layer, which normalizes the output of 
the previous layer. We'll use the ResNet9 architecture."""


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
        print(f"\33[33m Epoch: {epoch+1} | lr: {result["lrs"][-1]:.4f} : train_loss: {result["train_loss"]:.4f} | val_loss: {result["val_loss"]:.4f} | val_acc: {result["val_acc"]:.4f}") 



def conv_block(in_channels, out_channels, pool = False): 
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)] 
    if pool: layers.append(nn.MaxPool2d(2)) 

    return nn.Sequential(*layers) 


class ResNet9(ImageClassificationBase): 
    def __init__(self, in_channels, num_classes):
        super().__init__() 
        self.conv1 = conv_block(in_channels, 64) # 64 x 32 x 32
        self.conv2 = conv_block(64, 128, pool=True) # 128 x 16 x 16
        self.res1 = nn.Sequential(conv_block(128, 128), 
                                  conv_block(128, 128)) # 128 x 16 x 16

        self.conv3 = conv_block(128, 256, pool=True) # 256 x 8 x 8
        self.conv4 = conv_block(256, 512, pool=True) # 512 x 4 x 4
        self.res2 = nn.Sequential(conv_block(512, 512), 
                                  conv_block(512, 512)) # 512 x 4 x 4

        self.classifier = nn.Sequential(nn.MaxPool2d(4), # 512 x 1 x 1
                                        nn.Flatten(), # 512
                                        nn.Dropout(0.2), 
                                        nn.Linear(512, num_classes)) # 10

    def forward(self, xb: torch.Tensor) -> torch.Tensor: 
       out = self.conv1(xb) 
       out = self.conv2(out) 
       out = self.res1(out) + out 
       out = self.conv3(out) 
       out = self.conv4(out) 
       out = self.res2(out) + out 
       out = self.classifier(out) 
       return out


model = ResNet9(3, len(class_names)) 
to_device(model, device)  

for images, label in train_dl: 
    outputs = model(images) 
    #print(outputs.shape) 
    break

"""Train the Model
Before we train the model, we're going to make a bunch of small but important improvements to our 
fit function: 

* Learning rate scheduling: Instead of using a fixed learning rate, we'll use a 
learning rate scheduler, which will change the learning rate after every batch of training. There
many strategies for varying the learning rate, and the one we'll use is called the 
"One Cycle Learning Rate Policy", which involves starting with a low learning rate, gradually increasing it 
batch-by-batch to a high learning rate for about 30% of epochs, then gradually decreasing it to a very low value 
for the remaining epochs. 

*Weight decay: We'll use a weight decay, which is yet another regularization
technique which prevents the weights from becoming too large by adding additional 
terms to the loss funtion. 

*Gradient clipping: apart from the layer weights and outputs, it is also useful to 
limit the values of gradients to a small range to prevent undesirable changes in the paramters
due to too large gradient values. This simple yet effective technique is called gradient 
clipping.  

""" 
"""TRAIN THE MODEL""" 
def evaluate(model:nn.Module, val_dl: DataLoader): 
    model.eval() 
    with torch.inference_mode():
        outputs= [model.validation_step(batch) for batch in val_dl] 
        return model.validation_epoch_end(outputs)    
    

def get_lr(optimizer: torch.optim.Optimizer): 
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    
def fit_one_cycle(epochs, max_lr, model: nn.Module, train_dl, val_dl,
                  weight_decay=0, grad_clip = None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache() 
    history = [] 

    # Set up custom optimizer with weight decay 
    optimizer = opt_func(model.parameters(), lr=max_lr, weight_decay=weight_decay)

    # Set up one-cycle learning rate schedule 
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl)) 
    
    for epoch in range(epochs): 
        # Training Phase 
        model.train() 
        train_losses = [] 
        lrs = [] 
        for batch in train_dl: 
            loss = model.training_step(batch) 
            train_losses.append(loss) 
            loss.backward() 

            # Gradient clipping 
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip) 

            optimizer.step() 
            optimizer.zero_grad()
            
            # Record & update learning rate
            lrs.append(get_lr(optimizer)) 
            sched.step() 
        
        # Validation Phase 
        result = evaluate(model, val_dl)
        result["train_loss"] = torch.stack(train_losses).mean().item() 
        result["lrs"] = lrs 
        model.epoch_end(epoch, result) 
    history.append(result)


"""Let's train the model now"""
epochs = 8 
max_lr = 0.01
grad_clip = 0.1 
weight_decay = 1e-4
opt_func = torch.optim.Adam 


history = fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, 
                        grad_clip=grad_clip, 
                        weight_decay= weight_decay,
                        opt_func=opt_func) 

"""Plot losses and accuracies""" 

def plot_losses_and_accs(history):
    train_losses = [x["train_loss"] for x in history] 
    val_losses = [x["val_loss"] for x in history]  
    val_accs = [x["val_acc"] for x in history]

    plt.figure(figsize=(10, 15))  
    plt.plot(val_accs, "-x") 
    plt.xlabel("Epoch") 
    plt.ylabel("Accuracy") 
    plt.title("Accuracy Vs No. epochs")  

    plt.figure(figsize=(10, 15))  
    plt.plot(train_losses, "-bx") 
    plt.plot(val_losses, "-rx") 
    plt.xlabel("Epoch") 
    plt.ylabel("Loss") 
    plt.title("Loss Vs No. epochs")  


plot_losses_and_accs(history)

"""Predict and view image"""
def predict_image(image:torch.Tensor, model:nn.Module): 
    image = to_device(image, device)
    outputs = model(image.unsqueeze(0))  
    preds = torch.argmax(outputs, dim=1)
    return preds[0].item()

def show_prediction(image: torch.Tensor, label, model:nn.Module): 
    plt.figure(figsize=(10, 10))
    plt.imshow(image.permute(1, 2, 0)) 
    plt.title(f"Label: {class_names[label]} | Prediction: {class_names[predict_image(image, model)]}")
    plt.axis(False)

show_prediction(*val_ds[0] , model) 
show_prediction(*val_ds[108] , model)  
show_prediction(*val_ds[3900] , model) 


