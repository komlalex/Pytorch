"""PyTorch Computer Vision 

1. torchvision - base domain library for PyTorch computer vision 
2. torchvision.datasets - get datasets and data loading for computer vision 
3. torchvision.model - get pretrained computer vision models that you can leverage for your own problems
4. torchvision.transforms - functions for manipulating your vision data (images) to be suitable for use with an ML model
5. torch.utils.data.Dataset - Base dataset class for PyTorch 
6. torch.utils.data.DataLoader - Creates a Python iterable over a dataset
"""  
# Import PyTorch
import torch
from torch import nn 
from torch.utils.data import DataLoader


# Import torchvision 
import torchvision 
from torchvision import datasets 
from torchvision import transforms 
from torchvision.transforms import ToTensor 


# Import matplotlib for visualization 
import matplotlib.pyplot as plt  

# Import timeit 
from timeit import default_timer as timer

# Import tqdm for progress bar
from tqdm.auto import tqdm 

# Import accuracy metric from helper_functions.py 
from helper_functions import accuracy_fn

# Set device agnostic code 
device = "cuda" if torch.cuda.is_available() else "cpu"

"""
GETTING A DATASET 

The dataset we'll be using is the FashionMNIST from torchvision.datasets
"""

# Setup training data 

train_data = datasets.FashionMNIST(
    root="data", #where to download data to, 
    train= True, # do we want the training dataset?
    download= True, # do we want to download yes/no
    transform= ToTensor(), # how do we want to transform the data
    target_transform= None # how do we want to transform the lables/targets?
) 

test_data = datasets.FashionMNIST(
    root="data", 
    train = False, 
    download=True,
    transform= ToTensor(), 
    target_transform=None
)

class_names = train_data.classes
# Setup the batch size parameter 

BATCH_SIZE = 32

# Turn dataset into iterables 
train_dataloader = DataLoader(
    dataset= train_data, 
    batch_size= BATCH_SIZE, 
    shuffle= True
)

test_dataloader = DataLoader(
    dataset=test_data, 
    batch_size= BATCH_SIZE,
    shuffle= False 
)


def print_train_time(start: float,
                     end: float,
                     device: torch.device=None): 
    """Prints difference between start and end time""" 
    total_time = end - start
    print(f"Train time on {device}: {total_time: .3f} seconds") 
    return total_time


"""
Functionizing training and testing loops 
*training = train_step()
*testing loop = tst_step()
"""

def train_step(model: nn.Module, 
               data_loader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module, 
                optimizer: torch.optim.Optimizer, 
                accuracy_fn, 
                device: torch.device = device): 
    
    """Performs a training with model trying to learn on data_loader"""
    
    train_loss, train_acc = 0, 0

    # Put model in training mode
    model.train() 

    # Add a loop to loop through the training bacthes
    for x, y in data_loader:
        # Put data on target device
        x, y = x.to(device), y.to(device)

        # Forward pass 
        y_pred = model(x)  

        # Calculate the loss and acc (per batch) 
        loss = loss_fn(y_pred, y) 
        train_loss += loss # Accumulate the loss 

        train_acc += accuracy_fn(y_true= y, y_pred=y_pred.argmax(dim=1))
        # Optimizer zero grad 
        optimizer.zero_grad() 

        # Loss backward 
        loss.backward() 

        # Optimizer step 
        optimizer.step() 

    # Divide total train loss and acc by length of dataload 
    train_loss /= len(data_loader) 
    train_acc /= len(data_loader) 
    print(f"Train loss: {train_loss: .5f} | Train acc: {train_acc: .2f}%")

def test_step(model: nn.Module, 
              data_loader: torch.utils.data.DataLoader, 
              loss_fn: nn.Module, 
              accuracy_fn, 
              device: torch.device = device): 
    """Performs a testing loop step on model going over data_loader"""
    test_loss, test_acc = 0, 0

    model.eval() 
    with torch.inference_mode(): 
        for x, y in data_loader: 
            x, y = x.to(device), y.to(device) 
            
            test_pred = model(x) 

            test_loss += loss_fn(test_pred, y) 
            test_acc += accuracy_fn(y, test_pred.argmax(dim=1))  
        
        test_loss /= len(data_loader) 
        test_acc /= len(data_loader)

        print(f"Test loss: {test_loss: .5f} | Test acc: {test_acc: .2f}%")

def eval_model(model: torch.nn.Module, 
                data_loader: torch.utils.data.DataLoader, 
                loss_fn: torch.nn.Module, 
                accuracy_fn, 
                device: torch.device = device): 
    """Returns a dictionary containing the results of model prediction on data loader"""
    loss, acc = 0, 0 
    model.eval() 

    with torch.inference_mode(): 
        for x, y in tqdm(data_loader): 
            x, y = x.to(device), y.to(device)
            # Make predictions 
            y_pred = model(x) 

            # Accumulate loss and acc per batch 
            loss += loss_fn(y_pred, y) 
            acc += accuracy_fn(y, y_pred.argmax(dim=1))
        # Scale the loss and acc to find the average per batch 

        loss /= len(data_loader) 
        acc /= len(data_loader) 
    
    return {"model_name": model.__class__.__name__, 
            "model_loss": loss.item(), 
            "model_acc": acc}

"""
BUILDING A CONVOLUTIONAL NEURAL NETWORK 

CNN's are also known as ConvNets.

CNN's are known for their capabilities to find patterns in visual data
""" 

# Create a convolutional neural network 
class FashionMNISTV2(nn.Module):
    """Model architecture that replicates the TinyVGG model from CNN explainer website"""
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__() 

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size= (3, 3), 
                      stride=1,
                      padding=1), 
            nn.ReLU(), 
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units, 
                      kernel_size=(3, 3), 
                      stride=1,
                      padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=(2, 2))
        ) 

        self.con_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units, 
                      kernel_size=(3, 3), 
                      stride=1, 
                      padding=1), 
            nn.ReLU(), 
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units, 
                      kernel_size=(3, 3), 
                      stride=1, 
                      padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features=hidden_units*0, # There's a trick to calculating this ..
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.conv_block_1(x) 
        print(x.shape)
        x = self.con_block_2(x) 
        print(x.shape) 
        x = self.classifier(x) 
        return x
    
torch.manual_seed(42) 
model_2 = FashionMNISTV2(input_shape= 1, 
                         hidden_units=10,
                         output_shape=len(class_names)
                         ).to(device) 

# Steping through nn.Conv2d 
torch.manual_seed(42)

# Create a match of images
images = torch.randn(size=(32, 3, 64, 64))
test_image = images[0]

print(f"Image batch shape: {images.shape}")
print(f"Single image shape: {test_image.shape}") 
#print(f"Test image:\n {test_image}") 

# Create a single Con2d layer 
conv_layer = nn.Conv2d(
    in_channels=3, 
    out_channels=64, 
    kernel_size= (3, 3), 
    padding=1

)

# Pass the data through the convolutional layer 
conv_output = conv_layer(test_image) 

print(conv_output.shape)

