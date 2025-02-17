import torch 
from torch import nn 
import matplotlib.pyplot as plt
import numpy as np
import os 
from pathlib import Path
import random 
from PIL import Image
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

walk_through_dir(image_path) 

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
plt.show()

print(img_as_array)