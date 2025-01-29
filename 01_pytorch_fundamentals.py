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
#vector = torch.tensor([7, 7])  
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
print(TENSOR[0][1][2])
