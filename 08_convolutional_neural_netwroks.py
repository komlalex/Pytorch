"""PyTorch Computer Vision 

1. torchvision - base domain library for PyTorch computer vision 
2. torchvision.datasets - get datasets and data loading for computer vision 
3. torchvision.model - get pretrained computer vision models that you can leverage for your own problems
4. torchvision.transforms - functions for manipulating your vision data (images) to be suitable for use with an ML model
5. torch.utils.data.Dataset - Base dataset class for PyTorch 
6. torch.utils.data.DataLoader - Creates a Python iterable over a dataset
"""  
# Import PyTorch
import torch
from torch import nn 
from torch.utils.data import DataLoader


# Import torchvision 
import torchvision 
from torchvision import datasets 
from torchvision import transforms 
from torchvision.transforms import ToTensor 


# Import matplotlib for visualization 
import matplotlib.pyplot as plt  

# Import timeit 
from timeit import default_timer as timer

# Import tqdm for progress bar
from tqdm.auto import tqdm 

# Import accuracy metric from helper_functions.py 
from helper_functions import accuracy_fn

# Set device agnostic code 
device = "cuda" if torch.cuda.is_available() else "cpu"

"""
GETTING A DATASET 

The dataset we'll be using is the FashionMNIST from torchvision.datasets
"""

# Setup training data 

train_data = datasets.FashionMNIST(
    root="data", #where to download data to, 
    train= True, # do we want the training dataset?
    download= True, # do we want to download yes/no
    transform= ToTensor(), # how do we want to transform the data
    target_transform= None # how do we want to transform the lables/targets?
) 

test_data = datasets.FashionMNIST(
    root="data", 
    train = False, 
    download=True,
    transform= ToTensor(), 
    target_transform=None
)

class_names = train_data.classes
# Setup the batch size parameter 

BATCH_SIZE = 32

# Turn dataset into iterables 
train_dataloader = DataLoader(
    dataset= train_data, 
    batch_size= BATCH_SIZE, 
    shuffle= True
)

test_dataloader = DataLoader(
    dataset=test_data, 
    batch_size= BATCH_SIZE,
    shuffle= False 
)


def print_train_time(start: float,
                     end: float,
                     device: torch.device=None): 
    """Prints difference between start and end time""" 
    total_time = end - start
    print(f"Train time on {device}: {total_time: .3f} seconds") 
    return total_time


"""
Functionizing training and testing loops 
*training = train_step()
*testing loop = tst_step()
"""

def train_step(model: nn.Module, 
               data_loader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module, 
                optimizer: torch.optim.Optimizer, 
                accuracy_fn, 
                device: torch.device = device): 
    
    """Performs a training with model trying to learn on data_loader"""
    
    train_loss, train_acc = 0, 0

    # Put model in training mode
    model.train() 

    # Add a loop to loop through the training bacthes
    for x, y in data_loader:
        # Put data on target device
        x, y = x.to(device), y.to(device)

        # Forward pass 
        y_pred = model(x)  

        # Calculate the loss and acc (per batch) 
        loss = loss_fn(y_pred, y) 
        train_loss += loss # Accumulate the loss 

        train_acc += accuracy_fn(y_true= y, y_pred=y_pred.argmax(dim=1))
        # Optimizer zero grad 
        optimizer.zero_grad() 

        # Loss backward 
        loss.backward() 

        # Optimizer step 
        optimizer.step() 

    # Divide total train loss and acc by length of dataload 
    train_loss /= len(data_loader) 
    train_acc /= len(data_loader) 
    print(f"Train loss: {train_loss: .5f} | Train acc: {train_acc: .2f}%")

def test_step(model: nn.Module, 
              data_loader: torch.utils.data.DataLoader, 
              loss_fn: nn.Module, 
              accuracy_fn, 
              device: torch.device = device): 
    """Performs a testing loop step on model going over data_loader"""
    test_loss, test_acc = 0, 0

    model.eval() 
    with torch.inference_mode(): 
        for x, y in data_loader: 
            x, y = x.to(device), y.to(device) 
            
            test_pred = model(x) 

            test_loss += loss_fn(test_pred, y) 
            test_acc += accuracy_fn(y, test_pred.argmax(dim=1))  
        
        test_loss /= len(data_loader) 
        test_acc /= len(data_loader)

        print(f"Test loss: {test_loss: .5f} | Test acc: {test_acc: .2f}%")

def eval_model(model: torch.nn.Module, 
                data_loader: torch.utils.data.DataLoader, 
                loss_fn: torch.nn.Module, 
                accuracy_fn, 
                device: torch.device = device): 
    """Returns a dictionary containing the results of model prediction on data loader"""
    loss, acc = 0, 0 
    model.eval() 

    with torch.inference_mode(): 
        for x, y in tqdm(data_loader): 
            x, y = x.to(device), y.to(device)
            # Make predictions 
            y_pred = model(x) 

            # Accumulate loss and acc per batch 
            loss += loss_fn(y_pred, y) 
            acc += accuracy_fn(y, y_pred.argmax(dim=1))
        # Scale the loss and acc to find the average per batch 

        loss /= len(data_loader) 
        acc /= len(data_loader) 
    
    return {"model_name": model.__class__.__name__, 
            "model_loss": loss.item(), 
            "model_acc": acc}

"""
BUILDING A CONVOLUTIONAL NEURAL NETWORK 

CNN's are also known as ConvNets.

CNN's are known for their capabilities to find patterns in visual data
""" 

# Create a convolutional neural network 
class FashionMNISTV2(nn.Module):
    """Model architecture that replicates the TinyVGG model from CNN explainer website"""
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__() 

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size= (3, 3), 
                      stride=1,
                      padding=1), 
            nn.ReLU(), 
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units, 
                      kernel_size=(3, 3), 
                      stride=1,
                      padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=(2, 2))
        ) 

        self.con_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units, 
                      kernel_size=(3, 3), 
                      stride=1, 
                      padding=1), 
            nn.ReLU(), 
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units, 
                      kernel_size=(3, 3), 
                      stride=1, 
                      padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features= hidden_units * 7 * 7, # There's a trick to calculating this ..
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.conv_block_1(x) 
        #print(f"Output shape of conv_block_1: {x.shape}")
        x = self.con_block_2(x) 
        #print(f"Outoput shape of conv_block_2: {x.shape}") 
        x = self.classifier(x) 
        #print(f"Output shape of classifier: {x.shape}")
        return x
    
torch.manual_seed(42) 
model_2 = FashionMNISTV2(input_shape= 1, 
                         hidden_units=10,
                         output_shape=len(class_names)
                         ).to(device) 

# Steping through nn.Conv2d 
torch.manual_seed(42)

# Create a match of images
images = torch.randn(size=(32, 3, 64, 64))
test_image = images[0]

print(f"Image batch shape: {images.shape}")
print(f"Single image shape: {test_image.shape}") 
#print(f"Test image:\n {test_image}") 

# Create a single Con2d layer 
conv_layer = nn.Conv2d(
    in_channels=3, 
    out_channels=10, 
    kernel_size= (3, 3), 
    stride= 1,
    padding=1

)

# Pass the data through the convolutional layer 
conv_output = conv_layer(test_image) 

print(conv_output.shape)


""" Breaking own the nn.MaxPool2d layers step-by-step """
# Print original image without unsqueezed dimension
print(f"Test iamges original shape: {test_image.shape}")
print(f"Test images with unsqeezed dimension: {test_image.unsqueeze(0).shape}")


# Create a sample nn.MaxPool2d layer 
max_pool_layer = nn.MaxPool2d(kernel_size=4) 

# Pass data through just the con_layer 
test_image_through_conv = conv_layer(test_image) 
print(f"Shape after going through conv_layer(): {test_image_through_conv.shape}")


# Pass data through the max pool layer 
test_image_through_conv_and_max_pool = max_pool_layer(test_image_through_conv)
print(f"Shape after going through conv and max pool {test_image_through_conv_and_max_pool.shape}")


torch.manual_seed(42) 

# Create a random tensor with a smaller number of dimensions to our image
random_tensor = torch.randn(1, 1, 2, 2) 

# Create a max pool layer 
max_pool_layer = nn.MaxPool2d(kernel_size=2) 

# Pass the radnom tensor through the max pool layer 
max_pool_tensor = max_pool_layer(random_tensor)
#print(f"Random tensor shape: {random_tensor.shape}")
#print(f"Max pool tensor shape: {max_pool_tensor.shape}") 

image, label = train_data[0] 
plt.imshow(image.squeeze(), cmap="gray") 
plt.title(label)
#plt.show() 

rand_image_tensor = torch.randn(size=(1, 28, 28)).to(device)
# Pass image through modle 
model_2(rand_image_tensor.unsqueeze(0)) 




# Set up an optimizer and loss function
loss_fn = nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1) 


# Training and testing model_2 using our training and testing functions

torch.manual_seed(42) 
torch.cuda.manual_seed(42) 

start_time = timer() 

# Train and test model 
epochs = 3 
for epoch in tqdm(range(epochs)): 
    print(f"Epoch {epoch}...") 
    train_step(model=model_2, 
                data_loader=train_dataloader, 
                loss_fn=loss_fn, 
                optimizer=optimizer, 
                accuracy_fn=accuracy_fn, 
                device=device) 
    test_step(model=model_2, 
              data_loader=test_dataloader, 
              loss_fn=loss_fn, 
              accuracy_fn=accuracy_fn, 
              device=device)
end_time = timer() 

print_train_time(start=start_time, end=end_time, device=device) 

# Get model 
model_2_results = eval_model(model=model_2, 
                             data_loader=test_dataloader, 
                             loss_fn=loss_fn, 
                             accuracy_fn=accuracy_fn, 
                             device=device) 
print(model_2_results)  

# Make and evalaute random predictions with the best model 

def make_predictions(model: nn.Module, 
                      data: list, 
                      device: torch.device= device):
    pred_probs = [] 
    model.to(device)
    model.eval() 

    with torch.inference_mode(): 

        for sample in data:
            # Prepare the sample (add a batch dimension and pass to target device) 
            sample = torch.unsqueeze(sample, dim=0).to(device) 

            # Forward pass (model outputs raw logits) 
            pred_logits = model(sample)  

            # Get prediction probability (logits -> prodection probability) 

            pred_prob = torch.softmax(pred_logits.squeeze(), dim=0)

            # Get pred_prob off GPU for further calculations 
            pred_probs.append(pred_prob.cpu()) 
        
        #   Stack the pred_probs to turn list into a tensor
        return torch.stack(pred_probs) 
    
import random 
random.seed(42)
test_samples = [] 
test_labels = [] 

for sample, label in random.sample(list(test_data), k=9): 
    test_samples.append(sample)
    test_labels.append(label) 

# View the first sample shape 
plt.imshow(test_samples[0].squeeze(), cmap="gray") 
plt.title(class_names[test_labels[0]]) 
#plt.show()  


pred_probs = make_predictions(model=model_2, 
                              data = test_samples, 
                              device=device)

# View first two prediction probabilities
print(pred_probs[:2]) 

# Convert prediction probabilities to labels
pred_classes = pred_probs.argmax(dim=1) 


# Plot predictions 

plt.figure(figsize=(9, 9)) 
nrows = 3 
ncols = 3 

for i, sample in enumerate(test_samples):
    plt.subplot(nrows, ncols, i + 1)  

    # Plot the target image 
    plt.imshow(sample.squeeze(), cmap="gray") 

    # Find the prediction (in text form, e.g. "Sandals") 
    pred_label = class_names[pred_classes[i]] 

    # Get the truth label (in text form)  
    truth_label = class_names[test_labels[i]] 

    # Create a title for plot 
    title_text = f"Pred: {pred_label} | Truth: {truth_label}" 

    # Check for equality between pred and truth and change color of title text
    colour = "g" if pred_label == truth_label else "r" 
    plt.title(title_text, fontsize=10, c=colour) 
    plt.axis(False)
#plt.show()  

"""
Making a confusion matrix for further predictions 

A confusion matrix is a fantastic way of evaluating your classification matrix visually

1. Make predictions with our trained model 
2. Make a confusion matrix 'torchmetrics.ConfusionMatrix`
3. Plot the confusion matrix using `mlxtend.plot_confusion_matrix()`
"""
from tqdm.auto import tqdm 
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# Make predictions with trained model 
y_preds = [] 

model_2.eval() 

with torch.inference_mode(): 
    for x, y in tqdm(test_dataloader, desc="Making predictions..."): 
        x, y = x.to(device), y.to(device) 

        # Forward pass 
        y_logits = model_2(x) 

        # Turn predictions from logits -> prediction probabilities -> labels 
        y_pred = torch.softmax(y_logits.squeeze(), dim=0).argmax(dim=1) 

        # Put predictions on cpu for evaluation 
        y_preds.append(y_pred.cpu()) 

#  Concatenate list of predictions into a single tensor 
y_preds_tensor = torch.cat(y_preds)  

# Setup confusion instance and compare predictions to target 
confmat = ConfusionMatrix(num_classes=len(class_names), 
                          task="multiclass") 

confmat_tensor = confmat(preds=y_preds_tensor, 
                         target = test_data.targets) 

print(confmat_tensor) 

# Plot the confusion matrix 
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),  # Matplotlib likes working with numpy
    class_names=class_names, 
    figsize=(10, 7)
)
plt.show() 

# Saving and loading the trained model 
from pathlib import Path
MODEL_PATH = Path("models" )
MODEL_PATH.mkdir(parents=True, 
                 exist_ok=True) 

# Create model save 
MODEL_NAME = "cnn_model_2.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME   

# Model the model state dict 
torch.save(obj=model_2.state_dict(), 
           f=MODEL_SAVE_PATH) 

