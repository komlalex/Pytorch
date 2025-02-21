import torch 
from torch import nn 
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchinfo import summary
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
#print(train_data.classes, train_data.class_to_idx)  

"""
Creating a helper function to get class names: 
1. Get the classnames using os.scandir() to traverse a target directory (ideally the directory is in standard image classification format)
2. Raise an eror if the class names aren't found (if this happens, there might be something wrong with the directory structure)
3. Turn the class names into a dict and a list and return them 

"""

# Setup path for target directory 
target_directory = train_dir
#print(f"Target dir: {target_directory}") 

# Get the class classnames from the target directory 

def find_classes(directory: str) -> Tuple[ List[str], Dict[str, int]]:
    """Find the class folder names in the target directory"""
    # 1. Get the class names by scanning the target directory 
    classes = sorted([entry.name for entry in os.scandir(directory) if entry.is_dir()]) 

    # Raise an error if classnames could not be found
    if not classes: 
        raise FileNotFoundError(f"Couldn't find any classses in {directory} ... please check file structure")
    
    # Create a dictionary of index labels (computers prefer numbers rather than strings as labels)
    class_to_idx = {classname: i for i, classname in enumerate(classes)}
    return classes, class_to_idx

class_names, class_to_idx = find_classes(target_directory)
print(class_names, class_to_idx)

"""
Create a  custom Dataset to replicate ImageFolder 

To create our own custom dataset, we want to: 
1. Subclass torch.utils.data.Dataset
2. Init our subclass with a target directory (the directory we'd like to get data from) as well as 
transform if we'd like to transform our data. 
3. Create several attributes: 
* paths - paths of our images
* transform - transform we'd like to use 
* classes - a list of target classes
* class_to_idx - a dict of the target classes mapped integer labels 
4. Create a function to `load_images()`, this function will open an image 
5. Overwrite the `__len__()` method to return the length of our dataset 
6. Overwrite `__getitem__()` method to return a give sample when passed an index
""" 

# Write a custom dataset class 
from torch.utils.data import Dataset 

# Subclass torch.utils.data.Dataset 
class ImageFolderCustom(Dataset): 
    # Initialize our custom dataset 
    def __init__(self,
                  targ_dir: str,
                  transforms=None):
        super().__init__()
    
        # Create class attibutes 
        # Get all of the image paths 
        self.paths = list(Path(targ_dir).glob("*/*.jpg")) 

        # Setup transforms 
        self.transform = transforms

        # Create classes and class to idx 
        self.classes, self.class_to_idx = find_classes(targ_dir) 

    # Create a function to load images 
    def load_image(self, index: int) -> Image.Image: 
        "Opens an image via a path and opens it"
        image_path = self.paths[index]
        return Image.open(image_path) 
    
    # Overwrite __len__() method 
    def __len__(self): 
        """Returns the total number of samples."""
        return len(self.paths) 
    
    # Overwrite __getitem__()
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Returns one sample of data, data and label (x, y)"""
        img  = self.load_image(index)
        class_name = self.paths[index].parent.name # Expects path in format: data_folder/class_name/image.jpg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary 
        if self.transform: 
            return self.transform(img), class_idx # Return data, label (x, y) 
        
        else: 
            return img, class_idx # return untransformed image and label 

# Create a transform 
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.ToTensor()
])

# Test out ImageFolderCustom 
train_data_custom = ImageFolderCustom(targ_dir=train_dir, 
                                      transforms=train_transforms) 

test_data_custom =  ImageFolderCustom(targ_dir=test_dir, 
                                      transforms=test_transforms) 

# Check for equality betwen ImageFolder Dataset and ImageFolderCustom Dataset 
print(train_data_custom.classes == train_data.classes)
print(train_data_custom.class_to_idx == train_data.class_to_idx) 

"""
Create a function to display random images 

1. Take in a Dataset and a number of parameters such as classnames and how 
many images to visualize. 
2. To prevent display getting out of hand, let's cap the number of images to see at 10 
3. Set a random seed for reproducibility
4. Get a list the random sample indices from the target dataset
5. Setup a matplotlib plot
6. Loop through the random sample images and plot them with matplotlib
7. Make sure the dimensions of our images line up with matplotlib (HWC)
""" 

# Create a function to take in a dataset 
def display_random_images(dataset: Dataset, 
                          classes: List[str], 
                          n: int = 10, 
                          display_shape: bool = True, 
                          seed: int = None): 
    # Adjust display if n is too high 
    if n > 10: 
        n = 10 
        display_shape = False 
        print("For display purposes, n shouldn't be larger than 10, setting it 10 and removing shape display")

    if seed: 
        random.seed(seed) 

    # Get random samples 
    random_samples_idx = random.sample(range(len(dataset)), k=n) 

    # Setup plot 
    plt.figure(figsize=(16, 8))

    # Loop through indices and plot them with matplotlib 
    for i, targ_sample in enumerate(random_samples_idx): 
        targ_image, targ_label = dataset[targ_sample]   

        # Adjust tensor dimensions for plotting 
        targ_image_adjust = targ_image.permute(1, 2, 0)  

        # Plot adjusted samples 
        plt.subplot(1, n, i + 1) 
        plt.imshow(targ_image_adjust) 
        plt.axis(False)
        if classes: 
            title = f"Class {classes[targ_label]}"
            if display_shape: 
                title = title + f"\nShape: {targ_image_adjust.shape}"
        plt.title(title)

# Display random images from the ImageFolder ceated Dataset
display_random_images(dataset=train_data, 
                      n = 5, 
                      classes=class_names, 
                      seed=42) 

# Display random images from the ImageFolderCustom creared Dataset
display_random_images(dataset=train_data_custom, 
                      n = 20,
                      classes=class_names, 
                      seed=42)
#plt.show() 

# Turn custom loaded images into DataLoader
BATCH_SIZE = 1
train_dataloader_custom = DataLoader(
    dataset=train_data_custom, 
    batch_size=BATCH_SIZE, 
    drop_last=False, 
    shuffle=True
)

test_dataloader_custom = DataLoader(
    dataset=test_data_custom, 
    batch_size = BATCH_SIZE, 
    shuffle=False 
) 

# Get image and label from custom dataloader
img_custom, label_custom  = next(iter(train_dataloader_custom)) 
#print(img_custom.shape, label_custom.shape)

"""
Other forms of transforms (Data Augmentation)
Data augmentation is the process of adding diversity to your data. 

In the case of image data, this might mean applying various image transformations to the training images

This practice hopefully results in a model that's more generalizeable to unseen data. 
Let's take a look at one particular type of data augmentation used to train PyTorch vision
models to state of the art levels ...
"""

# Let's look at TrivialAugment 

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=25), 
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor()
]) 

# Get all the iamge paths 
image_path_list = list(image_path.glob("*/*/*.jpg"))

# Plot random transformed images 
plot_transformed_images(
    image_paths=image_path_list, 
    transforms=train_transform, 
    n = 3, 
    seed = None
)

"""
Model 0: TinyVGG without data augmentation

Let's replicate the TinyVGG from the CNN explainer website 
"""

# Let's create a simple tranaform 
simple_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_data_simple = datasets.ImageFolder(
    root=train_dir, 
    transform=simple_transform,
    target_transform=None
)

test_data_simple = datasets.ImageFolder(
    root= test_dir, 
    transform=simple_transform, 
    target_transform=None
)

# Setup batch size and number of workers 
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

train_dataloader_simple = DataLoader(
    dataset=train_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True
)

test_dataloader_simple = DataLoader(
    dataset=test_data, 
    batch_size=BATCH_SIZE, 
    shuffle=False
)  

# Create TinyVGG model class 

class TinyVGG(nn.Module): 
    def __init__(self, input_shape: int, 
                 hidden_units: int, 
                 output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3,
                      stride=1, 
                      ),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units, 
                      kernel_size=3, 
                      stride=1, 
                      ),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, 
                         stride=2) # default stride value is the same as the kernel_size
        ) 

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units, 
                      kernel_size=3,
                      stride=1, 
                      ),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units, 
                      kernel_size=3, 
                      stride=1, 
                      ),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, 
                         stride=2) # default stride value is the same as the kernel_size
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features=hidden_units * 13 * 13,
                      out_features=output_shape, 
                      )
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.conv_block_1(x) 
        print(x.shape) 
        x = self.conv_block_2(x) 
        print(x.shape)
        x = self.classifier(x) 
        print(x.shape)
        #return self.classifier(self.conv_block_2(self.conv_block_1(x)))
        return x 
    
torch.manual_seed(42) 

model_0 = TinyVGG(input_shape=3, 
                  hidden_units=10, 
                  output_shape=len(class_names)
                  ).to(device) 


# Try a forward pass on a single image 
image_batch, label_batch = next(iter(train_dataloader_simple)) 

print(model_0(image_batch.to(device)))

print(summary(model_0, input_size=(32, 3, 64, 64)))






    










