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
from helper_functions import accuracy_fn, plot_decision_boundary

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
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor) 

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
            #nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            #nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.linear_layer_stack(x) 
    
# Instantiate our mode  
model_0 = BlobModel(input_features=2, 
                    output_features=4,
                    hidden_units=8).to(device)

# Setup loss function and optimizer 
loss_fn = nn.CrossEntropyLoss() 

optimizer = torch.optim.SGD(params=model_0.parameters(), 
                            lr=0.1) 

"""
In order to train and evalaute our model, we need to convert our model's 
outputs(logits) to prediction probabilities and then to prediction labels
logits -> probabilities (torch.softmax()) -> labels (torch.argmax())
""" 
# Getting prediction probabilities for a multi-clas PyTorch model 
# Let's get the raw outputs of our model
model_0.eval()
with torch.inference_mode(): 
    y_logits = model_0(x_test.to(device))


# Convert our model's logit outputs to prediction probabilities 
y_pred_probs  = torch.softmax(y_logits, dim=1)
#print(torch.sum(y_pred_probs[0]))

# Convert our model's prediction probabilities to labels 
y_preds = torch.argmax(y_pred_probs, dim=1)

#print(y_preds)
#print(y_test) 

"""
Creating our training and testing loops for our model
"""
torch.manual_seed(42) 
torch.cuda.manual_seed(42)
x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)  

epochs = 1000 

for epoch in range(epochs): 
    model_0.train() 
    
    y_logits = model_0(x_train) 
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)
    #y_preds = torch.argmax(y_probs, dim=1)
    acc = accuracy_fn(y_train, y_preds)
    loss = loss_fn(y_logits, y_train) 

    optimizer.zero_grad() 

    loss.backward() 

    optimizer.step()

    # Test 
    model_0.eval() 
    with torch.inference_mode(): 
        test_logits = model_0(x_test)
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1) 

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_preds)

    if epoch % 100 == 0: 
        print(f"Epoch {epoch} | Loss: {loss: .5f} | Acc: {acc:.2f}% : Test Loss: {test_loss: .5f} | Test Acc {test_acc: .2f}%")

"""Making predictions"""
model_0.eval() 
with torch.inference_mode(): 
    y_logits = model_0(x_test) 
    y_probs = torch.softmax(y_logits, dim=1)
    y_preds = torch.argmax(y_probs, dim=1) 

    print(torch.eq(y_preds, y_test))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, x_train, y_train)

plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, x_test, y_test)
plt.plot()
plt.show()