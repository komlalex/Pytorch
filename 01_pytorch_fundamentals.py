# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 07:03:03 2025

@author: Alexander Komla
"""
import torch

import pandas as pd 
import numpy as np
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
#print(zeros)

# Random tensor of one 
ones = torch.ones(size=(2, 4))
#print(ones)
#print(ones.dtype) 

# Creating a range of tensors and tensors-like 
one_to_ten = torch.arange(0, 11)  
#print(one_to_ten)  

# Creating tensors like 

ten_zeros = torch.zeros_like(input=one_to_ten) 

#print(ten_zeros) 

# TENSOR DATATYPES 
# Float 32 tensor 
"""
Tensor datatypes is one of the 3 big errors you'll run into with PyTorch & deep learning
1. Tensors not right datatype
2. Tensors not right shape 
3. Tensors not on right device 
"""
float_32_tensor = torch.tensor([3.0, 6, 9.0], 
                               dtype=torch.float32, # What datatype is the tensor (e.g. float32)
                               device="cuda", #What device is your tensor on 
                               requires_grad=False)  #Whether or not to track gradients with this tensor's operations
float_16_tensor = float_32_tensor.type(torch.float32)
#print(float_16_tensor.dtype)

result = float_16_tensor * float_32_tensor
#print(result)

# Getting information from tensors 
"""
datatype = tensor.dtype
shape = tensor.shape 
evice  = tensor.device
"""
some_tensor = torch.rand(3, 4) 

#print(f"Datatype of tensor: {some_tensor.dtype}")
#print(f"Shape of tensor: {some_tensor.shape}") # or some_tensor.size()
#print(f"Device of tensor: {some_tensor.device}") 


### Manipulating tensors 
"""
Tensor operations include: 
Addition 
Subtraction 
Multiplication 
Division 
Matrix multiplication (Element-wise)
""" 
tensor = torch.tensor([1, 2, 3]) 
# Add 10
add = ten_zeros + 10 

# Multiply by 10 
mul = tensor * 10 

# Subtract 10
sub = tensor - 10

# Divide by 10 
div = tensor / 10 
 
# Try out PyTorch built-in functions
"""
torch.mul()
torch.add()
torch.div()
torch.sub()
""" 

# Matrix multiplication(element-wise) 
mul = tensor * tensor 


# Matrix multiplication (dot product)
matmul = torch.matmul(tensor, tensor) # or tensor @ tensor
#print(matmul) 

"""
There are two rules tha that perfornming matrix multiplication needs to satisfy
1. The inner dimensions must match: 
(3, 2) @ (3, 2) won't work 
(2, 3) @ (3, 2) will work 
(3, 2) @ (2, 3) will work 

2. The resulting matrix has the shape of the outer dimensions 
(2, 3) * (3, 2) -> (2, 2) 
(3, 2) * (2, 3) -> (3, 3) 
"""
#print(torch.matmul(torch.rand(3, 10), torch.rand(10, 3)))  

# SHAPE ERRORS 
tensor_A = torch.tensor([[1, 2],
                         [3, 3],
                         [5, 6]]) 

tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]]) 

"""
To fix our tensor shape issues, we can manipulate the shape of one of our tensors using transpose
"""
tensor_B = tensor_B.T 
output = torch.matmul(tensor_A, tensor_B) # torch.mm is the same as torch.matmul
#print(output) 
#print(output.shape) 

"""
FINDING min, max, mean, sum, etc (TENSOR AGGREGATION) 
"""
x = torch.arange(1, 100, 10, dtype=torch.float32) 


# Find min 
x.max()
torch.max(x)

# Find average 
mean = torch.mean(x) 

# Find the sum 
#print(torch.sum(x)) 


"""
Fidning the positional min and max
"""
#print(x.argmin())
#print(x.argmax()) 


"""
RESHAPING, VIEWING, STACKING, SQEEZING, UNSQEEZING, AND PERMUTING TENSORS
1. Reshaping - reshaping an input tensor to a defined shaped
2. View - Return a view of an input tensor of a certain shape but keep the same memory
as the original tensor 
3. Stacking - Combining multiple vectors on top of each other (vstack, hstack, etc) 
4. Squeeze - remove all '1' dimensions from a tensor 
5. Unsqueeze - Adds a one dimension to a target tensor
6. Permute - Return a view of the input with dimensions permuted (swapped) in a certain way.
"""

x = torch.arange(1.0, 10.0) 
#print(x, x.shape)

# Add an extra dimension 
x_reshaped = x.reshape(1, 9) 
#print(x_reshaped) 

# Change the view 
z = x.view(3, 3) 
#print(z, z.shape) 

"""
Changing z changes x (because a view of a tensor shares the same memory as the original input) 

""" 
z[:, 0] = 5 

#print(x) # The first element of x also changes 

# Stack tensors on top of each other 
x_stacked = torch.stack([x, x, x, x], dim=1)    
#print(x_stacked)      

# Sqeeze  - removes all single dimensionsssss from a tensor 
#print(x_reshaped)
x_squeezed =  torch.squeeze(x_reshaped) 

#print(x_squeezed)   
#print(x_squeezed.shape) 

# Unsqueeze - adds a single dimension to a target tensor at a specific dim (dimension) 
x_unsqueezed = x_squeezed.unsqueeze(dim=0) 
#print(x_unsqueezed.shape) 

# permute - rearranges the dimensions of the target tensor in a specified order 

x_original = torch.rand(size=(224, 224, 3)) 

# Permute the original tensor to rearrange the axis(or dim) order 

x_permuted = x_original.permute(2, 0, 1)  # Color channels, height, view

#print(x_permuted.shape) 

"""
INDEXING - SELECTING DATA FROM TENSORS
 
"""

x = torch.arange(1, 10).reshape(1, 3, 3)
#print(x[0]) 
#print(x[0][0])
# print(x[0][2][2])  

# You can also use the ":" to select "all" of a target dimension 
#print(x[:, 0]) 

# Get all values of the 0th and 1st dimensions but only index 1 of the 2nd dimension
#print(x[:, :, 1]) 

# Get all values of the 0th dimension but only 1 index of the 1st and second dimensions 
#print(x[:, 1, 1]) 
 
# Get index 0 of 0th and 1st dimension and all values of the 2nd dimension 
#print(x[0, 0, :]) 

# Index on x to return 9 
#print(x[0][2][2])

# Index on x to return 3, 6, 9
#print(x[0, :, 2]) 

"""
PyTorch Tensors and Numpy 

NumPy is a popular scientific Python computing library.
Because of this, PyTorch has functionality to interact with it 

Data in Numpy -> PyTorch tensor "torch.from_numpy(ndarray)"
Pytorch tensor -> Numpy array "torch.Tensor.numpy()"
"""

array = np.arange(1.0, 8.0) 
tensor = torch.from_numpy(array) # Warning: when converting from Numpy to PyTorch, PyTorch reflects Numpy's datatype of float64, unless specified otherwise

# Change the value of array, what will this do to "tensor"? (not affected)
array = array +  1


# Tensor to Numpy array 
tensor = torch.ones(7) 
numpy_tensor = torch.Tensor.numpy(tensor) 


"""
Reproducibility (trying to take random out of random) 

In short, how a neural network works is: 
    
Start with random  -> tensor operations  -> update random numbers to try and make them better representations of the data -> again -> again ...

To reduce the randomness in neural networks and PyTorch comes the concept of "random seed" 

Essentially what random seed does is "flavour" the randomness

""" 
random_tensorA = torch.rand(3, 4) 
random_tensorB = torch.rand(3, 4) 

# print(tensor_A) 
# print(tensor_B) 
# print(random_tensorA == random_tensorB)
 
# let's make some random but reproducible tensors
# Set the random seed 
RANDOM_SEED = 42 

torch.manual_seed(RANDOM_SEED)   
random_tensorC = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4) 

#print(random_tensorC) 
#print(random_tensor_D) 
#print(random_tensorC == random_tensor_D) 

"""
Running tensors and PyTorch objects on the GPU (and making faster computations) 
 

GPUs = faster computations on numbers, thanks to CUDA + NVIDIA hardware + PyTorch working behind the scenes to make everything hunky dory (good)

1. Easiest - use Google Colab for free GPU 
2. Use your own GPU     
3. Use cloud computing - GCP, AWS, Azure, these services allowwww you to rent computers on the cloud and access them
""" 

# Check for GPU access with PyTorch 

#print(torch.cuda.is_available()) 

# Setup device agnostic code 

device = "cuda" if torch.cuda.is_available() else "cpu" 

#print(device) 

# Count the number of devices 

# print(torch.cuda.device_count()) 

# PUTTING TENSORS AND MODELS ON THE GPU 
tensor = torch.tensor([1, 2, 3]) 

# Tensor not on GPU 
#print(tensor, tensor.device) #cpu

# Move tensor to GPU (if available) 
tensor_on_gpu = tensor.to(device) 

#print(tensor_on_gpu) 

### Moving tensors back to the CPU  
# If tensor is not on GPU, can't transform it to Numpy 
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy() 

print(tensor_back_on_cpu)
