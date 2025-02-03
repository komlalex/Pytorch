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
from torch import nn
from helper_functions import accuracy_fn, plot_decision_boundary

device = "cuda" if torch.cuda.is_available() else "cpu"

n_samples = 1000 

x, y = make_circles(n_samples, 
                    noise=0.03,
                    random_state=42) 

plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.RdYlBu) 
#plt.show()

# Convert data to tensors and then to train and test plits 

# Turn data into tensors 
x = torch.from_numpy(x).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32) 

# Split into train and test sets 
x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42) 

"""
Building our model with non-linearity 
1. Linear = straight line 
2. Non-linear = non-straight lines 

Artificial neural networks are a large comnbination of linear and non-linear functions which are 
potentially patterns in data
"""
class CircleModel(nn.Module): 
    def __init__(self):
        super().__init__() 
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU() # relu is a non-linear activation function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
    
model_0 = CircleModel().to(device)
# Setup loss and optimizer 

loss_fn = nn.BCEWithLogitsLoss() 
optimizer = torch.optim.SGD(params=model_0.parameters(), 
                            lr=0.1) 

# Training a model with non-linearity 

# Random seeds
torch.manual_seed(42) 
torch.cuda.manual_seed(42) 

# Put all data on target device 
x_train , y_train = x_train.to(device), y_train.to(device) 
x_test, y_test = x_test.to(device), y_test.to(device) 


# Loop through data 
epochs = 3000 

for epoch in range(epochs): 
    model_0.train() 

    y_logits = model_0(x_train).squeeze() 
    y_pred = torch.round(torch.sigmoid(y_logits)) 

    # Calculate the loss
    loss = loss_fn(y_logits, y_train) 
    acc = accuracy_fn(y_train, y_pred) 

    optimizer.zero_grad() 

    loss.backward() 

    optimizer.step() 

    # Testing 
    model_0.eval() 

    with torch.inference_mode(): 
        test_logits  = model_0(x_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits)) 

        test_loss = loss_fn(test_logits, y_test) 
        test_acc = accuracy_fn(y_test, test_pred)  

    if epoch % 100 == 0: 
        print(f"Epoch: {epoch} | Loss: {loss: .5f} | Acc: {acc: .2f}% | Test loss: {test_loss: .5f} | Test acc: {test_acc: .2f}%")

"""
Evaluating our trained model 
"""
model_0.eval() 

with torch.inference_mode(): 
    y_pred = torch.round(torch.sigmoid(model_0(x_test))).squeeze() 

    print(torch.eq(y_pred, y_test)) 

# Plot decision boundary 

plt.figure(figsize=(12, 6)) 
plt.subplot(1, 2, 1) 
plt.title("train") 
plot_decision_boundary(model_0, x_train, y_train) 
plt.subplot(1, 2, 2) 
plt.title("Test") 
plot_decision_boundary(model_0, x_test, y_test) 
#plt.show() 

"""
REPLICATING NON-LINEAR ACTIVATION FUNCTIONS
Neural networks, rather than us telling the model what to learn, 
we give it the tools to discover patterns in data and it tries to figures out the patterns on its own. 

And these tools are linear & non-linear functions
"""  
# Create a tensor 
A = torch.arange(-10, 10, 1, dtype=torch.float32) 

plt.plot(A)
plt.show()