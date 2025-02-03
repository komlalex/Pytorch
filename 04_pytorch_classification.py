""" 
NEURAL NETWORK CLASSIFICATION WITH PYTORCH 

Classification is a problem of predicting whether something is one thing or another (there
can multiple things as options)

"""

# Make classification data and get it ready 

import torch
from torch import nn
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


# Building a model 

"""
Let's build a model to classify our blue and red dots.
To do so, we want to: 
1. Setup device agnostic code
2. Construct a model (by subclassing nn.Module)
3. Define a loss function and optimizer 
4. Create a training and testing loop
"""

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"  

"""
CREATE OUR MODEL: 
1. Subclass nn.Module
2. Create 2 nn.Linear() layers that are capable of handling the shapes of our data 
3. Define a forward() method that outlines the forward pass (or forward computation)
4. Instantiate an instance of our model class and send it to our target `device`.
"""

class CircleModelV0(nn.Module): 
    def __init__(self):
        super().__init__() 
        self.layer_1 = nn.Linear(in_features=2, out_features=5) # Takes in 2 featurs and upscales to 5 features
        self.layer_2 = nn.Linear(in_features=5, out_features=1) # Takes in 5 features from previous layer and outputs a single features

    def forward(self, x: torch.Tensor) ->   torch.Tensor: 
        return self.layer_2(self.layer_1(x)) 
    

# Instantiate an instance of our model class and send it to the target device 
model_0 = CircleModelV0().to(device) 



# Let's replicate the model using nn.Sequential  

"""
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device) 

"""

"""
Set up loss function and optimizer 

Which loss function or optimizer should you use? 

Again... this is problem specific. 
For example you might want to MAE or MSE (mean absolute error or mean square error).
For classification, you might want binary cross entropy or categorical cross entropy (cross entropy)

And for optimizers, two of the most common are SGD and Adam.  

* For loss functioi, we're going to use `torch.nn.BCEWithLogitsLoss()`
""" 

# Setup the loss function 
#loss_fn = nn.BCELoss # Requiress inputs to have gone through the sigmoid activation function prior to BCElOSS
loss_fn = nn.BCEWithLogitsLoss() # sigmoid activation function  

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr = 0.1)

# Calculate accuracy - out of a 100 examples, what percentage does our model get right 

def accuracy_fn(y_true, y_pred): 
    corrent = torch.eq(y_true, y_pred).sum().item() 
    acc = (corrent/len(y_pred)) * 100 
    return acc 

"""
GOING FROM RAW LOGITS -> PREDICTION PROBABILITIES => PREDICTION LABELS
Our model outputs are going to be raw logits.
We can convert those logits into prediction probabilities by passing them to some 
kind of activation function (eg. sigmoid for binary crossentropy and softmax for multiclass classification) 

Then we can convert our model's prediction probabilitiess to prediction labels by either rounding them or taking the argmax(). 
"""
# View the first 5 outputs of the forward pass on the test data
model_0.eval()
with torch.inference_mode(): 
    y_logits = model_0(x_test.to(device))[:5]
    print(y_logits)


"""
# For our prediction probability values, we need to perform a range-style rouding on them: 
# y_pred_probs >= 05 
#print(torch.round(y_pred_probs)) y = 1 (class 1) 
#y_pred_probs < 0.5 y = 0 (class 0) 
"""
# Convert logits to probabilities 
y_probs = torch.sigmoid(y_logits)

# Find the predicted labels 
y_preds = torch.round(y_probs) 

# In full (logits -> pred_probs -> labels)
y_pred_labels = torch.round(torch.sigmoid(model_0(x_train.to(device))[:5]))

#print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze())) 

#print(torch.eq(y_preds.squeeze().cpu(), y_test[:5].squeeze())) 


#BUILDING A TRAINING AND TEST LOOP
torch.cuda.manual_seed(42)
torch.cuda.manual_seed(42)


epochs = 100
x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)

for epoch in range(epochs): 
    # Training
    model_0.train() 

    # Forward pass
    y_logits = model_0(x_train).squeeze() # This los function expects raw logits as input
    y_preds = torch.round(torch.sigmoid(y_logits))

    # Calculate loss accuracy
    loss = loss_fn(y_logits, y_train) 

    acc = accuracy_fn(y_true=y_train, y_pred=y_preds) 


    # Optimizer zero grad
    optimizer.zero_grad() 

    # loss backward 
    loss.backward()  

    # Optimizer step 
    optimizer.step()

    #Testing 
    model_0.eval() 
    with torch.inference_mode() :
        test_logits = model_0(x_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        # Calculate test loss 
        test_loss = loss_fn(test_logits, y_test ) 

        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    # Print out what's happening 
    if epoch % 10 == 0: 
  
        print(f"Epoch: {epoch} : Train Loss: {loss: .5f} | Train Acc: {acc: .2f}% | Test Loss: {test_loss: .5f} | Test Acc: {test_acc: .2f}%")


"""
MAKE PREDICTIONS AND EVALUATE OUR MODEL  

From the mtetrics it looks like our model isn't learning anything. 

To inspect it, let's makes some predictions and make them visual.
To do some, we're going to import a function called plot_decision_boundry()
"""

import requests 

from pathlib import Path

# Download heper functions from  Learn PyTorch repo (if it's not already downloaded) 

if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download")
else: 
    print("Downloading helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py") 

    with open("helper_functions.py", "wb") as f: 
        f.write(request.content) 

from helper_functions import plot_predictions, plot_decision_boundary  

"""
Plot decision boundary
"""
plt.figure(figsize=(12, 6)) 
plt.subplot(1, 2, 1)
plt.title("Train") 
plot_decision_boundary(model_0, x_train, y_train) 

plt.subplot(1, 2, 2) 
plt.title("Test") 
plot_decision_boundary(model_0, x_test, y_test) 

#plt.show() 

"""
IMPROVING A MODEL (FROM A MODEL'S PERSPECTIVE)
* Add more layers
* Add more hidden units - g0 from 5 hidden units to 10 hidden units
* Fit for longer 
* Chaning the activation function
* Change the learning rate
* Change the loss function 
These options are all from our model perspective because they deal diectly with the model, rather than the data.
And because these options are all values we (as machine learning engineers and data scientists) change, they are referred to as hyperparameters.  

Let's try and improve our model by: 
1. Adding more hidden units: 5 -> 10
2. Increasing the number of layers: 2 -> 3
3. Increase the number of epochs: 100 -> 1000 

"""

class CircleModelV1(nn.Module): 
    def __init__(self):
        super().__init__() 
        self.layer_1 = nn.Linear(in_features=2, out_features=10) 
        self.layer_2 = nn.Linear(in_features=10, out_features=10) 
        self.layer_3 = nn.Linear(in_features=10, out_features=1) 

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        #z = self.layer_1(x) 
        #z = self.layer_2(z) 
        #z = self.layer_3(z) 
        return self.layer_3(self.layer_2(self.layer_1(x))) # This way of writing operations leverages speed-ups where posible
model_1 = CircleModelV1().to(device) 

# Create a loss function 
loss_fn = nn.BCEWithLogitsLoss()
# Create an optimizer 
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)
# Write a training and evaluation loop for model_1  

torch.manual_seed(42) 
torch.cuda.manual_seed(42)
epochs = 1000 

# Put data on device 
x_train, y_train = x_train.to(device), y_train.to(device) 
x_test, y_test = x_test.to(device), y_test.to(device)


for epoch in range(epochs): 
    model_1.train() 

    # Forward pass
    y_logits = model_1(x_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) 

    # Calcuate accuracy/loss 
    loss = loss_fn(y_logits, y_train) 
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)  

    # Optimizer zero grad 
    optimizer.zero_grad() 

    # Loss backward 
    loss.backward() 

    #Optimizer step 
    optimizer.step()

    # TESTING
    model_1.eval() 

    with torch.inference_mode(): 

        #Forward pass
        test_logits = model_1(x_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))  

        # Calculate accuracy/loss 
        test_loss = loss_fn(test_logits, y_test)  
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred) 

    # Print out what's happening
    if epoch % 100 == 0: 
        print(f"Epoch: {epoch} | Loss: {loss: .5f} | Acc: {acc:.2f}% | Test loss: {test_loss: .5f} | Test acc: {test_acc:.2f}%")

"""
Plot decision boundary
"""
plt.figure(figsize=(12, 6)) 
plt.subplot(1, 2, 1)
plt.title("Train") 
plot_decision_boundary(model_1, x_train, y_train) 

plt.subplot(1, 2, 2) 
plt.title("Test") 
plot_decision_boundary(model_1, x_test, y_test) 
# plt.show()

"""
PREPARING DATA TO SEE IF OUR MODEL CAN FIT A STRAIGHT LINE
One way to troubleshoot a larger problem is to test pit a smaller problem. 

""" 
# Create some data 
weight= 0.7 
bias = 0.3 
start, end = 0, 1 
step = 0.01

x_regression = torch.arange(start, end, step).unsqueeze(dim=1)  
y_regression = weight * x_regression + bias 

train_split = int(0.8 * len(x_regression))  

x_train_reg, y_train_reg = x_regression[:train_split], y_regression[:train_split] 
x_test_reg, y_test_reg  = x_regression[train_split:], y_regression[train_split:]

print(len(x_train_reg), len(y_train_reg)) 
print(len(x_test_reg), len(y_test_reg)) 

# Plot predictions 
from helper_functions import plot_predictions 

plot_predictions(x_train_reg, y_train_reg, x_test_reg, y_test_reg)



"""
Adjusting model_1 to fit a straight line
"""

# Same architecture as model_1 using sequential 

model_2 = nn.Sequential(
    nn.Linear(in_features=1, out_features=10), 
    nn.Linear(in_features=10, out_features=10), 
    nn.Linear(in_features=10, out_features=1) 
).to(device)

print(model_2)

# Loss optimizer 
loss_fn = nn.L1Loss() 
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.01) 

torch.manual_seed(42)
torch.cuda.manual_seed(42) 
# Train our model 
epochs = 1000
x_train_reg, y_train_reg = x_train_reg.to(device), y_train_reg.to(device)
x_test_reg, y_test_reg = x_test_reg.to(device), y_test_reg.to(device)


for epoch in range(epochs): 
    model_2.train() 
    y_pred = model_2(x_train_reg)  
  
    loss = loss_fn(y_pred, y_train_reg) 

    optimizer.zero_grad() 

    loss.backward() 

    optimizer.step() 

    # Test 
    model_2.eval() 
    with torch.inference_mode(): 
        test_pred = model_2(x_test_reg) 
        test_loss = loss_fn(test_pred, y_test_reg) 
        

    if epoch % 100 == 0: 
        print(f"Epoch {epoch} | Loss: {loss: .5f} | Test loss: {test_loss: .5f} ")

# Make predictions 
model_2.eval() 

with torch.inference_mode()  : 
    y_pred = model_2(x_test_reg)

#Plot predictions 
plot_predictions(x_train_reg.cpu(),
                  y_train_reg.cpu(), 
                  x_test_reg.cpu(), 
                  y_test_reg.cpu(), 
                  predictions=test_pred.cpu()) 
plt.show()



















