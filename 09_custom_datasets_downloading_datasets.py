"""
We've used some datatsets with PyTorch before. 

But how do you get your own data into PyTorch? 

One of the ways to do so is via custom datasets 

# Domain libaries

Depending on what you're working on, vision, test, audio, recommendation, you'll want 
to look into each of the PyTorch domain libraries for existing data laoding 
functions and customizable data laoding functions.
"""

import torch 
from torch import nn 

# Setup device agnostic code 
device = "cuda" if torch.cuda.is_available() else "cpu" 

"""
GET DATA
Our dataset is a subset of Food101 dataset. 
Food101 starts with 101 different classes of food and 1000 images per class (750, training, 250 testing) 

Our dataset starts with 3 classes of food and only 10% of images (~75 training, 25 testing)

Why do this? 
When starting out ML projects, it's important to try things on a small scale and increase the scale when 
necessary. 

The whole point is to speed up how fast you can experiment. 

"""

import requests
import zipfile 
from pathlib import Path 

# Setup path to data folder 
data_path = Path("data/") 
image_path = data_path / "pizza_steak_sushi" 

# If the image folder doesn't exist, download it and prepare it ....
if image_path.is_dir(): 
    print(f"{image_path} directory already exists... skipping download") 
else: 
    print(f"{image_path} does not exist, creating one...") 
    image_path.mkdir(parents=True, exist_ok=True) 


# Download pizza, stake and sushi data 
with open(data_path/"pizza_steak_sushi.zip", "wb") as f: 
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print("Downloading pizza, steak, sushi data ...") 
    f.write(request.content)

# Unzip pizza, steak, sushi data 
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref: 
    print("Unzipping pizza, steak and shushi data") 
    zip_ref.extractall(image_path) 