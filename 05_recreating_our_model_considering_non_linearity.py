"""
THE MISSING PIECE: NON-LINEARITY

What patterns could you draw if you were given an infinite amount of straight and non-straight lines? 

Or in machine learning terms, an infinite (but it is realy finite) amount of linear 
and non-linear functions?  
"""

# Make and plot data 
import matplotlib.pylab as plt 
from sklearn.datasets import make_circles 
from sklearn.model_selection import train_test_split 
import torch 

n_samples = 1000 

x, y = make_circles(n_samples, 
                    noise=0.03,
                    random_state=42) 

plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.RdYlBu) 
plt.show()

# Convert data to tensors and then to train and test plits 

# Turn data into tensors 
x = torch.from_numpy(x).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32) 

# Split into train and test sets 
x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42) 

print(x_train[:5], y_train[:5])
