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

# Check versions 
#print(torch.__version__) 
#print(torchvision.__version__)

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

#print(len(train_data), len(test_data))

# See the first training example 
image, label = train_data[0] 
#print(label)
#print(train_data.classes) 
class_to_idx = train_data.class_to_idx 
class_names = train_data.classes
#print(class_to_idx) 

# Check shape of our image 
#print(f"Image shape {image.shape}") 

# Visualizing our data 
image, label , train_data[0] 
#print(f"Image shape {image.shape}") 

plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.imshow(image.squeeze())
plt.title(label) 

plt.subplot(1, 2, 2)
plt.imshow(image.squeeze(), cmap="gray") 
plt.title(class_names[label])
#plt.show() 

# Plot more images 
torch.manual_seed(42) 
fig = plt.figure(figsize=(9, 9)) 
rows, cols = 4, 4 

for i in range(1, rows*cols + 1): 
    rand_idx  = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[rand_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label]) 
    plt.axis(False)

#plt.show() 

"""
Do you think these times of clothing (images) could be modelled with pure linear
lines? Or do you think we'lll ned non-linearity?
"""

# Prepare DataLoader 
"""
Right now, our data is in the form of PyTorch Datasets 

DataLoader turns our dataset into a Python iterable. 

More specifically, we want to turn our data into batches (or min-batches)

Wjy would we do this? 
1. It is more computationally efficient, as in, your computing hardware may not be 
able to look (store in memory) all 60000 images in one hit. So we break it down to 
32 iamges at a time (batch size 32). 
2. It gives our neural network more chances to upgrade its gradients per epoch
"""

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


# Let's check what we've created 
#ataloader, test_dataloader) 

#print(f"DataLoaders: {train_dataloader, test_dataloader}")
#print(f"Length of train_dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
#print(f"Length of test_dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")

# Check out what's in the training dataloader 
train_features_batch, train_labels_batch = next(iter(train_dataloader)) 

#print(train_features_batch.shape, train_labels_batch.shape)
# Show sample 
torch.manual_seed(42)  
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx] 
plt.imshow(img.squeeze(), cmap="gray")
plt.title(class_names[label]) 
plt.axis(False) 
#plt.show() 

"""
Model 0: Baseline model
When starting to buils a series of machine learning experiments, it's best practice to start 
with a baseline model.  
A baseline model is a simple model you'll try to improve upon in subsequent models/ experiments. 
In other words: start simply and add complexity when necessary. 
""" 

# Create a flatten layer 
flatten_model = nn.Flatten() 

# Get a single sample 
x = train_features_batch[0]
print(x.shape) 

# Flatten the sample 
output = flatten_model(x)
print(output.shape) 

class FashionMNISTModelV0(nn.Module): 
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__() 
        self.layer_stack = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.layer_stack(x) 
    
torch.manual_seed(42) 

# Setup model with input parameters 
model_0 = FashionMNISTModelV0(input_shape=784, # 28 * 28
                              hidden_units= 10,
                              output_shape= len(class_names) # One for every class
                              ).to("cpu")
dummy_x = torch.rand([1, 1, 28, 28])

print(model_0(dummy_x))