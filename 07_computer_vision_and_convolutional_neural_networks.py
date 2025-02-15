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

# Import accuracy metric from helper_functions.py 
from helper_functions import accuracy_fn
# Check versions 
#print(torch.__version__) 
#print(torchvision.__version__)

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

#print(len(train_data), len(test_data))

# See the first training example 
image, label = train_data[0] 
#print(label)
#print(train_data.classes) 
class_to_idx = train_data.class_to_idx 
class_names = train_data.classes
#print(class_to_idx) 

# Check shape of our image 
#print(f"Image shape {image.shape}") 

# Visualizing our data 
image, label = train_data[0] 
#print(f"Image shape {image.shape}") 

plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.imshow(image.squeeze())
plt.title(label) 

plt.subplot(1, 2, 2)
plt.imshow(image.squeeze(), cmap="gray") 
plt.title(class_names[label])
#plt.show() 

# Plot more images 
torch.manual_seed(42) 
fig = plt.figure(figsize=(9, 9)) 
rows, cols = 4, 4 

for i in range(1, rows*cols + 1): 
    rand_idx  = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[rand_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label]) 
    plt.axis(False)

#plt.show() 

"""
Do you think these times of clothing (images) could be modelled with pure linear
lines? Or do you think we'lll ned non-linearity?
"""

# Prepare DataLoader 
"""
Right now, our data is in the form of PyTorch Datasets 

DataLoader turns our dataset into a Python iterable. 

More specifically, we want to turn our data into batches (or min-batches)

Wjy would we do this? 
1. It is more computationally efficient, as in, your computing hardware may not be 
able to look (store in memory) all 60000 images in one hit. So we break it down to 
32 iamges at a time (batch size 32). 
2. It gives our neural network more chances to upgrade its gradients per epoch
"""

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


# Let's check what we've created 
#ataloader, test_dataloader) 

#print(f"DataLoaders: {train_dataloader, test_dataloader}")
#print(f"Length of train_dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
#print(f"Length of test_dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")

# Check out what's in the training dataloader 
train_features_batch, train_labels_batch = next(iter(train_dataloader)) 

#print(train_features_batch.shape, train_labels_batch.shape)
# Show sample 
torch.manual_seed(42)  
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx] 
plt.imshow(img.squeeze(), cmap="gray")
plt.title(class_names[label]) 
plt.axis(False) 
#plt.show() 

"""
Model 0: Baseline model
When starting to buils a series of machine learning experiments, it's best practice to start 
with a baseline model.  
A baseline model is a simple model you'll try to improve upon in subsequent models/ experiments. 
In other words: start simply and add complexity when necessary. 
""" 

# Create a flatten layer 
flatten_model = nn.Flatten() 

# Get a single sample 
x = train_features_batch[0]
print(x.shape) 

# Flatten the sample 
output = flatten_model(x)
print(output.shape) 

class FashionMNISTModelV0(nn.Module): 
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__() 
        self.layer_stack = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.layer_stack(x) 
    
torch.manual_seed(42) 

# Setup model with input parameters 
model_0 = FashionMNISTModelV0(input_shape=784, # 28 * 28
                              hidden_units= 10,
                              output_shape= len(class_names) # One for every class
                              ).to("cpu")

"""
Setup loss, optimizer aand evaluation metrics 
Loss function - since we're working with multi-class multi-class data, our loss function will be 
nn.CrossEntropyLoss() 

Optimizer - our optimizer torch.optim.SGD( (stochastic gradient descent))
Evaluation metrics - since we're working on a classification problem, let's use accuracy as our evaluation metric
"""

# Setup loss function and optimizer 
loss_fn = nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1) 

"""
Creating a function to time our experimental. 

Two of the main things you'll often want to track are: 
1. Model's performance (loss and accuracy values etc)
2. How fast it runs 
"""
def print_train_time(start: float,
                     end: float,
                     device: torch.device=None): 
    """Prints difference between start and end time""" 
    total_time = end - start
    print(f"Train time on {device}: {total_time: .3f} seconds") 
    return total_time


""""
Creating a training loop and training the model on batches of data 
1. Loop through training epochs
2. Loop through training batches, performing training steps, calculate the train loss "per batch"
3. Looop through testing batches, performing testing steps, calcualate test loss "per batch"
4. Print out what's happening 
5. Time it all (for fun)
"""

# Import tqdm for progress bar
from tqdm.auto import tqdm 

# Set the seed and start the timer 
torch.manual_seed(42)
start_time = timer() 

# Set the number of epochs 
epochs = 3 


# Create training and test loop
for epoch in tqdm(range(epochs)): 
    print(f"Epoch: {epoch} \n")

    #Training 
    train_loss = 0

    # Add a loop to loop through the training bacthes
    for batch, (x, y) in enumerate(train_dataloader):
        model_0.train() 

        # Forward pass 
        y_pred = model_0(x)  

        # Calculate the loss (per batch) 
        loss = loss_fn(y_pred, y) 
        train_loss += loss # Accumulate the loss 

        # Optimizer zero grad 
        optimizer.zero_grad() 

        # Loss backward 
        loss.backward() 

        # Optimizer step 
        optimizer.step() 

        # print out what's happening 

        if batch % 400 == 0: 
            print(f"Looked at {batch * len(x)} / {len(train_dataloader.dataset)} samples")
    
    # Divide total train loss by the length of the train loader 
    train_loss /= len(train_dataloader)

    # Testing 

    test_loss, test_acc = 0, 0 
    model_0.eval() 
    with torch.inference_mode(): 
        for x_test, y_test in test_dataloader: 

            # Forward pass 
            test_pred = model_0(x_test) 

            # Calculate loss (accumulatively)
            test_loss += loss_fn(test_pred, y_test)

            # Calculate accuracy 
            test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))
    
    # Calculate the test loss average per batch 
        test_loss /= len(test_dataloader)
    # Calculate the test acc average per batch 
        test_acc /= len(test_dataloader)  

    # Print out what is happening 
    print(f"\nTrain Loss: {train_loss: .4f} | Test Loss: {test_loss: .4f} | Test acc: {test_acc: .4f}")


end_time = timer()  
total_train_time_model_0 = print_train_time(start_time, end_time, device=str(next(model_0.parameters()).device))


# Make predicitions and get Model 0 results 
torch.manual_seed(42) 

def eval_model(model: torch.nn.Module, 
                data_loader: torch.utils.data.DataLoader, 
                loss_fn: torch.nn.Module, 
                accuracy_fn): 
    """Returns a dictionary containing the results of model prediction on data loader"""
    loss, acc = 0, 0 
    model.eval() 

    with torch.inference_mode(): 
        for x, y in tqdm(data_loader): 
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

# Calculate model 0 results on test dataset 
model_0_results = eval_model(model=model_0, 
data_loader=test_dataloader, 
loss_fn=loss_fn, 
accuracy_fn=accuracy_fn) 

print(model_0_results)


# Setup device agnostic code 
device = "cuda" if torch.cuda.is_available() else "cpu" 



"""
BUIDLING A BETTER MODEL WITH NON-LINEARITY 

We learned about the power of non-linearity in notebook 02 

"""
class FashionMNISTModelV1(nn.Module): 
    def __init__(self, 
                input_shape: int, 
                hidden_units: int, 
                output_shape: int):
        super().__init__()

        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units), 
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.layer_stack(x) 


# Create an instance of model_1 
torch.manual_seed(42)
model_1 = FashionMNISTModelV1(
    input_shape= 784, # This is the output of the flatten layer after our 28 * 28 imgae goes in 
    hidden_units= 10 ,
    output_shape= len(class_names)
).to(device)

loss_fn = nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(params=model_1.parameters(), 
                            lr=0.1)


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
# Train model_1 
torch.manual_seed(42)

# Set start time
start_time_on_gpu = timer() 

# Set epochs
epochs = 3

# Create an optimizing loop using train_loss() and test_loss()
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n------") 

    # Train step
    train_step(model=model_1, 
               data_loader=train_dataloader, 
               loss_fn=loss_fn, 
               optimizer=optimizer, 
               accuracy_fn=accuracy_fn, 
            )  
    # Test Step 
    test_step(model=model_1, 
              data_loader=test_dataloader, 
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn, 
              device=device) 
    
end_time_on_gpu = timer() 
total_train_time_model_1 = print_train_time(start=start_time_on_gpu, end=end_time_on_gpu, device=device)

"""
Somtimes, depending on your data/hardware you might find your model trains 
faster on CPU than GPU. 

Why is this? 
1. It could be that the overhead for copying data/'model to and from the gpu outweighs the compute 
benefit offered by the GPU. 
2. The hardware you're using has a better CPU in terms of compute capability than 
the CPU.
"""

# Get model_1 results dictionary 

model_1_results = eval_model(model=model_1, 
                             data_loader=test_dataloader, 
                             loss_fn=loss_fn, 
                             accuracy_fn=accuracy_fn) 

print(model_0_results) 
print(model_1_results) 

# Create a convolutional neural network 
class FashionMNISTModelV2(nn.Module):
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
model_2 = FashionMNISTModelV2(input_shape= 1, 
                         hidden_units=10,
                         output_shape=len(class_names)
                         ).to(device) 

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

total_train_time_model_2  = print_train_time(start=start_time, end=end_time, device=device) 

# Get model 
model_2_results = eval_model(model=model_2, 
                             data_loader=test_dataloader, 
                             loss_fn=loss_fn, 
                             accuracy_fn=accuracy_fn, 
                             device=device) 
print(model_2_results) 

# Compare model results and training time 
import pandas as pd 
compare_results = pd.DataFrame([model_0_results, 
                                model_1_results, 
                                model_2_results]) 
compare_results["training_time"] = [total_train_time_model_0, 
                                    total_train_time_model_1, 
                                    total_train_time_model_2]
print(compare_results)


# Visualize our model 
compare_results.set_index("model_name")["model_acc"].plot(kind="barh") 
plt.xlabel("accuracy (%)") 
plt.ylabel("model") 
plt.show()
 