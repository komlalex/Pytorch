import torch 
from torch import nn 
import os 
from pathlib import Path

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