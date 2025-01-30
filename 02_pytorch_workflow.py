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
        plt.scatter(test_data, predictions, c="r", s= 4, labels="Predictions") 
    
    # Show the legend 
    plt.legend(prop={"size": 14}) 
    plt.show()

plot_predictions()