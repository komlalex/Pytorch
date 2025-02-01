""" 
NEURAL NETWORK CLASSIFICATION WITH PYTORCH 

Classification is a problem of predicting whether something is one thing or another (there
can multiple things as options)

"""

# Make classification data and get it ready 

import torch
from torch import nn
import sklearn
from sklearn.datasets import make_circles 
from sklearn.model_selection import train_test_split
import pandas as pd 
import matplotlib.pyplot as plt

# Make 1000 samples 
n_samples = 1000

# Create circles 
x, y = make_circles(n_samples, 
                    noise=0.03,
                    random_state=42) 


#print(f"First 5 samples of x: \n {x[0:5]}")
#print(f"First 5 samples of y: \n {y[0:5]}")

# Make a DataFrame of circles data 

circles = pd.DataFrame({"x1": x[:, 0], 
                        "x2": x[:, 1],
                        "label": y}) 

#print(circles.head(10))

# VISUALIZE 
plt.scatter(x= x[:, 0],
            y=x[:, 1], 
            c=y,
            cmap=plt.cm.RdYlBu)
#plt.show()

"""
The dataset we're working with is often referred to as a toy dataset.
That's it is small enough to experiment but still sizeable enough to practice the fundamentals
"""

# Check input and output shapes 

#print(x.shape, y.shape)
#print(x.ndim, y.ndim) 

# View the first example of features and labels 
x_sample = x[0]
y_sample = y[0] 

#print(f"Values for one sample of x: {x_sample} and the same for y: {y_sample}") 
#print(f"Shapes for one sameple of x: {x_sample.shape} and the same for y: {y_sample.shape}")

# Convert our data into tensors and create train and test splits 
x = torch.from_numpy(x).type(torch.float32) 
y = torch.from_numpy(y).type(torch.float32)

torch.manual_seed(42)
# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x,
                                                     y, 
                                                     test_size=0.2,
                                                    random_state=42) # 20% for test, 80% for train


# Building a model 

"""
Let's build a model to classify our blue and red dots.
To do so, we want to: 
1. Setup device agnostic code
2. Construct a model (by subclassing nn.Module)
3. Define a loss function and optimizer 
4. Create a training and testing loop
"""

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"  

"""
CREATE OUR MODEL: 
1. Subclass nn.Module
2. Create 2 nn.Linear() layers that are capable of handling the shapes of our data 
3. Define a forward() method that outlines the forward pass (or forward computation)
4. Instantiate an instance of our model class and send it to our target `device`.
"""

class CircleModelV0(nn.Module): 
    def __init__(self):
        super().__init__() 
        self.layer_1 = nn.Linear(in_features=2, out_features=5) # Takes in 2 featurs and upscales to 5 features
        self.layer_2 = nn.Linear(in_features=5, out_features=1) # Takes in 5 features from previous layer and outputs a single features

    def forward(self, x: torch.Tensor) ->   torch.Tensor: 
        return self.layer_2(self.layer_1(x)) 
    

# Instantiate an instance of our model class and send it to the target device 
model_0 = CircleModelV0().to(device) 


print(model_0.state_dict())