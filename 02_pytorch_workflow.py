# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 08:34:46 2025

@author: Alexander Komla
"""

"""PYTORCH WORKFLOW""" 

import torch 
from torch import nn # Contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt


""" 1. DATA (PREPARING AND LOADING) 

Data can be almost anything... in machine learning: 
* Excel spreadsheet
* Images of any kind 
* Videos 
* Audio like songs or podcasts
* DNA
* Text

Machine learning is a game of two parts: 
1. Get data into a numerical representation
2. Build a model to learn patterns in that numerical representation 

To showcase this, let's create some "known" data using the linear regression formula.

We'll use a linear regression formula to make a straight line with known parameters
"""

# Create "known" paramters 

weight = 0.7 
bias = 0.3  

# Create data 
start = 0 
end= 1 
step = 0.02 
X = torch.arange(start, end, step).unsqueeze_(dim=1) 
Y = weight * X + bias 

# Splitting data into training and testing sets 

train_split = int(0.8 * len(X)) 
X_train, Y_train = X[:train_split], Y[:train_split]  
X_test, Y_test = X[train_split:], Y[train_split:] 

# VISUALIZE 

def plot_predictions(train_data=X_train,
                     train_labels = Y_train, 
                     test_data = X_test, 
                     test_labels = Y_test,
                     predictions =None): 
    """Plots training data, test, data and compares predictions""" 
    plt.figure(figsize=(10, 7))
    
    # Plots training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    
    # plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data") 
    
    # Are there predictions? 
    
    if predictions is not None: 
        # Plot the predictions if they exist
        plt.scatter(test_data, predictions, c="r", s= 4, label="Predictions") 
    
    # Show the legend 
    plt.legend(prop={"size": 14}) 
    plt.show()

plot_predictions()  


"""BUILD OUR FIRST PYTORCH MODEL

What our model does: 
1. Start with random values
2. Look at the training data and adjust the random values to better represent the  ideal values 

This is done through two main algorithms: 
    1. Gradient descent 
    2. Backpropagation 
""" 
# Create a linear regression model class

class LinearRegressionModel(nn.Module): # Almost everything in PyTorch inherits nn.Module
    def __init__(self): 
        super().__init__() 
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32)) 
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))
    
    # Forward method to define the computation in the model 
    def forward(self, x: torch.Tensor): 
        return self.weights * x + self.bias # The linear regression formula 

"""
PYTORCH MODEL BUILDING ESSENTIALS 

torch.nn - contains all of the building blocks for computational graphs (a neural network can be considered a computational graph)
torch.nn.Parameter - what parameters should our model try and learn, often a PyTorch layer from torch.nn will set these for us
torch.nn.Module - The base class for all neural network modules, if you subclass it, you should override foraward() 
torch.optim - this is where the optimizers in PyTorch live. They will help gradient descent. 
def forward() - All nn.Module subclasses require you to overwrite forward. 

"""

"""
# Checking the content of our PyTorch model
We can check our model parameters or what's inside our model using `.parameters()` 

"""

# Create a random seed 
torch.manual_seed(42)  

# Create an instance of the model (this is a subclass of nn.Module) 
model_0 = LinearRegressionModel() 

# Check out the parameters 

#print(list(model_0.parameters())) 

# List named parameters 
#print(model_0.state_dict()) 
#print(weight, bias) 

"""
Making prediction using `torch.inference_mode()` 

To check our model's predictive power, let's see how well it predicts `Y_test` based on `X_test` 

When we pass data to our model, it's going to run it through the forward() method 
"""

# Make predictions with model 
with torch.inference_mode(): 
     y_preds = model_0(X_test) 

plot_predictions(predictions=y_preds)
print(y_preds)