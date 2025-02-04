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

#print(torch.unique(y_train))

# Create train and test split 
x_train, x_test, y_train, y_test = train_test_split(x_blob,
                                                    y_blob,
                                                test_size=0.2)  



# Plot data 
plt.figure(figsize=(10, 7)) 
#plt.scatter(x_blob[:, 0], x_blob[:,  1], c=y_blob, cmap =plt.cm.RdYlBu) 
#plt.show() 

# Building our multi-class classification model in PyTorch 
device = "cuda" if torch.cuda.is_available() else "cpu" 

class BlobModel(nn.Module): 
    def __init__(self, input_features, output_features, hidden_units=8):
        """Initializes multi-class classification model
        Args: 
        input_features (int): Number of input features to the model 
        out_features (int): Number of output features (number of output classes)
        hidden_units (int): Number of hidden units betwen layers , default 8

        Returns (torch.Tensor):
        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.linear_layer_stack(x) 
    
# Instantiate our mode  
model_0 = BlobModel(input_features=2, 
                    output_features=4,
                    hidden_units=8)  

print(model_0)


