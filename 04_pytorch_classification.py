""" 
NEURAL NETWORK CLASSIFICATION WITH PYTORCH 

Classification is a problem of predicting whether something is one thing or another (there
can multiple things as options)

"""

# Make classification data and get it ready 

import sklearn
from sklearn.datasets import make_circles 
import pandas as pd 
import matplotlib.pyplot as plt

# Make 1000 samples 
n_samples = 1000

# Create circles 
x, y = make_circles(n_samples, 
                    noise=0.03,
                    random_state=42) 


print(f"First 5 samples of x: \n {x[0:5]}")
print(f"First 5 samples of y: \n {y[0:5]}")

# Make a DataFrame of circles data 

circles = pd.DataFrame({"x1": x[:, 0], 
                        "x2": x[:, 1],
                        "label": y}) 

#print(circles.head(10))

# VISUALIZE 
plt.scatter(x= x[:, 0],
            y=x[:, 1], 
            c=y,
            cmap=plt.cm.RdBu)
plt.show()

"""
The dataset we're working with is often referred to as a toy dataset.
That's it is small enough to experiment but still sizeable enough to practice the fundamentals
"""
