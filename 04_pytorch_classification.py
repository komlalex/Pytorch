""" 
NEURAL NETWORK CLASSIFICATION WITH PYTORCH 

Classification is a problem of predicting whether something is one thing or another (there
can multiple things as options)

"""

# Make classification data and get it ready 

import torch
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


print(len(y_train), len(y_test))