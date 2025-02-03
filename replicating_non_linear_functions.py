import matplotlib.pyplot as plt 
import torch 
from torch import nn 

"""
REPLICATING NON-LINEAR ACTIVATION FUNCTIONS
Neural networks, rather than us telling the model what to learn, 
we give it the tools to discover patterns in data and it tries to figures out the patterns on its own. 

And these tools are linear & non-linear functions
"""  
# Create a tensor 
A = torch.arange(-10, 10, 1, dtype=torch.float32) 

# Visualize our tensor 
#plt.plot(A) 
#plt.show()

#plt.plot(torch.relu(A))
#plt.show() 

def relu(x: torch.Tensor) -> torch.Tensor: # both input and output must be tensors
    return torch.maximum(torch.tensor(0), x)  

print(relu(A)) 
print(torch.relu(A)) 

# Visualize 
plt.plot(relu(A))
plt.show()
