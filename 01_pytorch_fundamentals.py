# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 07:03:03 2025

@author: Alexnader Komla
"""
import torch
import pandas as pd 
import numpy 
import matplotlib.pyplot as plt 

"""INTRODUCTION TO TENSORS""" 

# SCALARS

# Scalars
scalar = torch.tensor(7) 

# print(scalar)
# print(scalar.ndim)  # Get number of dimensions (0)
# print(scalar.item()) # Get tensor back as int  

# VECTORS 
vector = torch.tensor([7, 7])  
#print(vector.ndim) # 
#print(vector.shape) # 2  

# MATRIX 
MATRIX = torch.tensor([ [7, 8], 
                       [9, 10] ]) 

#print(MATRIX.ndim) # 2
#print(MATRIX.shape) # [2, 2]
#print(MATRIX[0]) # [7, 8] 

# TENSOR 
TENSOR = torch.tensor([ 
                        [[1, 2, 3], [4, 5, 6]],
                       
                       [[7, 8, 9], [10, 11, 12]]
                       ]) 
#print(TENSOR.ndim) #3 
#print(TENSOR.shape) #[2, 2, 3]
#print(TENSOR[0][1][2]) # 6   

# RANDOM TENSORS 
"""Random tensors are important because the way neural networks learn is that they start 
tensors full of random numbers and adjust those random numbers to better represent the data

Start with random numbers --> look at data --> update random numbers --> look at data --> update random numbers
"""

# Create random tensor of size (3, 4) 

random_tensor = torch.rand(3, 4) 
#print(random_tensor) 


# Create a random tensor with similar shape to an image 
random_image_size_tensor = torch.rand(size=(224, 224, 3)) # Height, width, color channels (R, G, B) 
#print(random_image_size_tensor.ndim) 
#print(random_image_size_tensor.size())

# Random tensor of all zeros 
zeros = torch.zeros(size=(3, 4))
print(zeros)

# Random tensor of one 
ones = torch.ones(size=(2, 4), dtype=torch.int32)
print(ones)
print(ones.dtype)