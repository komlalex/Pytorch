# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 08:34:46 2025

@author: Alexander Komla
"""

"""PYTORCH WORKFLOW""" 

import torch 
from torch import nn # Contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt 
import numpy as np
from pathlib import Path


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

#plot_predictions()  


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
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
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
# Using inference mode make our predictions run faster. It is fairly new.
# We can also do something similar with torch.no_grad(), however, torch.inference_mode() is more preferable
with torch.inference_mode(): 
     y_preds = model_0(X_test) 

#plot_predictions(predictions=y_preds)
# print(y_preds) 

""" 
TRAIN MODEL 

The whole idea of training is for a model to movev from some unoknown parameters to known parameters  

In other words from a poor representation of the data to a better representation of the data

One way to measure how poor or how wrong your model's predictions are is to use a loss function

Note: Loss functions may also be called cost functions or criterion in different areas. For our case, are going to refer to it as as loss function.

THINGS WE NEED TO TRAIN
1. Loss function: A function measure how wrong your model prediction is to the idea outputs. Lower is better 
2. Optimizer: Takes into account the loss of a model and adjusts the model's parameters (e.g.weight and bias) to improve the loss function 

Typicallly for PyTorch, we need: 
    1. Training loop
    2. Testing loop

"""

# Setup a loss functiom 
loss_fn = nn.L1Loss()

# Setup an optimizer (stochastic gradient descent)
optimizer = torch.optim.SGD(params  = model_0.parameters(), 
                            lr = 0.01) # learning rate = possibly the most import hyperparameter you can set

"""
Building aa training loop 

A couple of things we need in a training loop: 
0. Loop through the data 
1. Forward pass (This involves data moving through our forward() functions) - also called forward propagation
2. Calculate the loss (compare forward pass predictions to ground truth labels)
3. Optimizer zero grad
4. Loss backward - move backwards through the network to calculate the gradient of of the parameters of our model with respect to the loss (back propagation)
5. Optimizer step - use the optimizer to adjust our model's parameters to try and improve the loss (gradient descent)
"""

"""An epoch is one loop through the data"""
epochs = 300

epoch_count = [] 
train_loss_values = [] 
test_loss_values = []


# 0. Loop through the data
for epoch in range(epochs): 
    epoch_count.append(epoch)
    # Set the model to training mode
    model_0.train() # Train mode sets all parameters that require gradients to require gradients 
    
    # 1. Forward pass
    y_pred = model_0(X_train)
    
    # 2. Calculate the loss (how different the model's predictions are to the true values)
    loss = loss_fn(y_pred, Y_train)
    train_loss_values.append(loss) 
    
    # 3. Optimizer zeor grad 
    optimizer.zero_grad() 
    
    
    # 4. Perform back propagation
    loss.backward()
    
    # 5. Step the optimizer (perform gradient descent) 
    optimizer.step()
    #model_0.eval()

    # Testing 
    model_0.eval()  # This turns off different settings not needed for evaluation/testing (dropout, batch norm)
    with torch.inference_mode(): # turns off gradient tracking and a couple more things behind the scenes
        # Do the forward pass
        test_pred = model_0(X_test)  
    
        # calculate loss 
        test_loss = loss_fn(test_pred, Y_test)
        test_loss_values.append(test_loss)
    # Print out what's happening
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
    
    
# Plot the loss curves 
plt.plot(epoch_count, np.array(torch.tensor(train_loss_values).numpy()), label="Train Loss")
plt.plot(epoch_count, np.array(torch.tensor(test_loss_values).numpy()), label="Test Loss") 
plt.title("Training and test loss curves") 
plt.ylabel("Loss") 
plt.xlabel("Epochs") 
plt.legend()
plt.show()






with torch.inference_mode(): 
    y_preds = model_0(X_test)
    
#plot_predictions(predictions=y_preds)

"""
SAVING OUR PYTORCH MODEL
 There are three main methods you should know about for saving and loading models in PyTorch
 1. torch.save() -- Allows you to save a PyTorch object in Python's pickle format
 2. torch.load() -- allows you to load a saved PyTorch object 
 3. torch.nn.Module.load_state_dict() - allows you to load a model's saved state dictionary 
"""
# Create model directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Create model save path 
MODEL_NAME = "01_pytorch_workflow_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME  

# SAVE THE MODEL STATE DICT 
torch.save(model_0.state_dict(), MODEL_SAVE_PATH)


"""
LOADING OUR PYTORCH MODEL

Since we saved our model's state_dict() instead of the entire
model, we'll create a new instance of our model class and load 
the saved state_dict() into that.
"""
# Create new instance
loaded_model = LinearRegressionModel() 

# Load the saved state_dict() of model_0 
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH, weights_only=True)) 

# Make some predictions 
loaded_model.eval() 

with torch.inference_mode(): 
    loaded_model_preds = loaded_model(X_test) 

# Compare loaded model's predictions to original model's predictions 
print(y_preds == loaded_model_preds)









































