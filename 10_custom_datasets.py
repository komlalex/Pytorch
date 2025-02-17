import torch 
from torch import nn 
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os 
from pathlib import Path
import random 
from PIL import Image
from typing import Dict, Tuple, List
# Setup device agmostic code 
device = "cuda" if torch.cuda.is_available() else "cpu"  

# Set diectory path 
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# Becoming one with the data (data preparation and exploration) 

def walk_through_dir(dir_path): 
    """Walks through dir_path returning its contents.""" 
    for dirpath, dirnames, filenames in os.walk(dir_path): 
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}. ")

#walk_through_dir(image_path) 

# Setup train and testing paths 
train_dir = image_path / "train" 
test_dir = image_path / "test" 

"""
VISUALIZING AN IMAGE

Let's write some code to: 
1. Get all of the image paths 
2. Pick a random image path using Python's random.choice() 
3. Get the image class names using `pathlib.Path.parent.stem`
4. Since we're working with images, let's open the image with Python's PIL
5. We'll then show the image and print metadata
"""

# Set seed 
#random.seed(42) 

# Get all the image paths 
image_path_list = list(image_path.glob("*/*/*.jpg")) 

# Pick a random image path 
random_image_path = random.choice(image_path_list) 

# Get the image class from the path name (the image class is the name of the directory where the image is stored)
image_class = random_image_path.parent.stem 

# Open image 
img = Image.open(random_image_path) 

# Print metadata 
print(f"Random image path: {random_image_path}") 
print(f"Image class: {image_class}") 
print(f"Image height: {img.height}")
print(f"Image width: {img.width}")
#Image._show(img)

"""VISUALIZE THE IMAGE WITH MATPLOTLIB"""
# Turn the image into array 
img_as_array = np.asanyarray(img)
plt.figure(figsize=(10, 7))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape}")
plt.axis(False)
#plt.show()



"""
TRANSFORMING DATA

Before we can use our image data with PyTorch: 
1. Turn your data into tensors (in our case numerical representation of our images)
2. Turn it into a `torch.util.Dataset` and subsquently `torch.utils.DataLoader`, we'll call 
these Dataset and DataLoader
"""

# Transforming data with torchvision.tranforms 
data_transforms = transforms.Compose([
    # Resize images to 64x64
    transforms.Resize(size=(64, 64)), 
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5),
    # Turn the image into a tensor
    transforms.ToTensor()
])

#print(data_transforms(img).shape)  

"""Transforming data with `torchvision.transforms` 

Transforms help you get your images ready to be used with a model/perform data augmentation
""" 

def plot_transformed_images(image_paths: list, transforms, n=3, seed = None): 
    """
    Selects random images from a path of images and loads/transforms them, and plots 
    the original vs the transformed version.
    """
    if seed: 
        random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n) 

    for image_path in random_image_paths: 
        with Image.open(image_path) as f: 
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(f) 
            ax[0].set_title(f"Original\nSize: {f.size}")
            ax[0].axis(False) 

            # Transform and plot target 
            transformed_image = transforms(f).permute(1, 2, 0) # note we will need to change the shape (C, W, W) -> (W, C, H)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed\nShape: {transformed_image.shape}")
            ax[1].axis("off")  

            fig.suptitle(f"Class {image_path.parent.stem}", fontsize=16) 


#plot_transformed_images(image_paths=image_path_list ,transforms=data_transforms, n=3, seed=None) 

#lt.show() 

"""
Option 1: Loading image data using ImageFolder

We can load image classification data using `torchvision.datasets.ImageFolder`
"""

# Use ImageFolder to create dataset
train_data = datasets.ImageFolder(
    root=train_dir,
    transform=data_transforms, # Transform for data
    target_transform=None, # target transform
)

test_data = datasets.ImageFolder(
    root = test_dir, 
    transform=data_transforms, 
)

# Get class names as a list 
class_names = train_data.classes 
# Get class names as dict 
class_dict = train_data.class_to_idx

# Check the lengths of our datasets 
#print(len(train_data), len(test_data)) 


# Index on the train_data dataset to get a single image and label 
img, label = train_data[0] 
#print(f"Image tensor:\n {img}")
#print(f"Image shape: {img.shape}")
#print(f"Image datatype: {img.dtype}")
#print(f"Image label: {label}")
#print(f"Label datatype: {type(label)}") 

# Rearrange the order dimensions 
img_permute = img.permute(1, 2, 0)

# Print out the different shapes 
print(f"Original shape: {img.shape} -> [color_channels, height, width]")
print(f"Image permute: {img_permute.shape} -> [height, width, color_channels]")

# Plot the image
plt.figure(figsize=(10, 7))
plt.imshow(img_permute)
plt.title(class_names[label], fontsize=16) 
plt.axis(False)
#plt.show()

"""
Turn loaded images into `DataLoader`

A DataLoader is going to help us turn our Dataset into iterables and we can customize
the batch_size images at a time 
"""
BATCH_SIZE = 1
train_dataloader = DataLoader(
    dataset=train_data,  # os.cpu_count()
    batch_size=BATCH_SIZE, 
    shuffle=True,
    #num_workers= 1
)

test_dataloader = DataLoader(
    dataset=test_data, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    #num_workers= 1
)


img, label = next(iter(train_dataloader)) 

# Batch size will now be 1, you can change the batch size if you like 
#print(f"Image shape: {img.shape}")
#print(f"Label shape {label.shape}") 


"""
Option 2: Loading Image Data with a custom `Dataset`
1. Want to be able to load images from file
2. Want to be able to get class names from the Dataset
3. Want to be able to get classes as a dictionary from the Dataset 

Pros: 
1. Can create a `Dataset` out of almost anything
2. Not limited to PyTorch pre-built Dataset functions 

Cons: 
1. Even though you could create Dataset out of almost anything, it doesn't mean it will work...
2. Using a custom `Dataset` often results in us writing more code, which could be prone to erros 
or performance issues.  

All custom datasets often subclass Dataset
"""
# Instance of torchvision.datasets.ImageFolder() 
print(train_data.classes, train_data.class_to_idx)
















