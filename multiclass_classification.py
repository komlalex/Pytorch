"""
PUTTING IT ALL TOGETHER WITH A MULTI-CLASS CLASSIFICATION
1. Binary classification = one thing or another (cat vs dog, spam or not spam, fraud or not fraud)
2. Multi-class clasfication = more thaan one thing or another (cat vs dog vs chicken)
"""

import torch 
from torch import nn 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs 
from sklearn.model_selection import train_test_split 

# Set hyperparameters for data creation 
NUM_CLASSES = 4 
NUM_FEATURES = 2
RANDOM_SEED = 42 

# Create multi-class data 
x_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                             centers= NUM_CLASSES,
                              cluster_std= 1.5, #  Give the clusters a little shake up 
                              random_state= RANDOM_SEED)
# Turn data into datas 
x_blob = torch.from_numpy(x_blob).type(torch.float32) 
y_blob = torch.from_numpy(y_blob).type(torch.float32) 

# Create train and test split 
x_train, x_test, y_train, y_test = train_test_split(x_blob,
                                                    y_blob,
                                                test_size=0.2)  


print(x_blob[:5])
print(y_blob[:5])

# Plot data 
plt.figure(figsize=(10, 7)) 
plt.scatter(x_blob[:, 0], x_blob[:,  1], c=y_blob, cmap =plt.cm.RdYlBu) 
plt.show()