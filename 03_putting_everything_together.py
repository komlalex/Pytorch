import torch 
from torch import nn 
import matplotlib.pyplot as plt 
import numpy as np 
from pathlib import Path 

# Device agnostic code 
device= "cuda" if torch.cuda.is_available() else "cpu"
x = torch.arange(start=0, end= 1, step=0.02).unsqueeze(dim=1)

weight, bias = 0.7, 0.3 

y = x * weight + bias 

split_ratio = int(0.8 * len(x)) 

x_train, y_train = x[: split_ratio], y[: split_ratio] 
x_test, y_test = x[split_ratio:], y[split_ratio:]  


def plot_predictions(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, predictions=None):

    plt.figure(figsize=(10, 7))
    plt.scatter(x_train, y_train, c="blue", s=4, label="Training data")
    plt.scatter(x_test, y_test, c="g", s=4, label="Testing data") 

    if predictions is not None: 
        plt.scatter(x_test, predictions, s=4, c="r", label="Predictions") 

    plt.xlabel("X Values")
    plt.ylabel("Y Values")
    plt.legend(prop={"size": 14})
    plt.show()   


class LinearRegressionModel(nn.Module): 
    def __init__(self):
        super().__init__()

        # Use nn.Linear() for creating the model parameters | also called linear transform, probing layer, dense layer, fully-connected layer 
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.linear_layer(x)  
    
torch.manual_seed(42)
my_model = LinearRegressionModel() 
my_model.to(device)


torch.manual_seed(42)
epochs = 200


# Put data on the right device 

x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)
loss_func = nn.L1Loss() 
optimizer = torch.optim.SGD(params=my_model.parameters(), lr=0.01)

for epoch in range(epochs): 
   
    my_model.train() 

    y_preds = my_model(x_train) 

    loss = loss_func(y_preds, y_train) 
    

    optimizer.zero_grad() 

    loss.backward() 

    optimizer.step()   

    # Testing 
    my_model.eval() 

    with torch.inference_mode(): 
        test_pred = my_model(x_test) 
        
        test_loss = loss_func(test_pred, y_test)

    if epoch % 10 == 0: 
        print(f"Epoch: {epoch}, Loss: {loss} Test loss: {test_loss}") 

my_model.eval()
with torch.inference_mode(): 
    y_preds = my_model(x_test) 

    y_preds = y_preds.cpu()
    plot_predictions(predictions=y_preds)




print(my_model.state_dict())
#test_loss_values = np.array(torch.tensor(test_loss_values).numpy())
#plt.plot(epoch_values, test_loss_values) 
#plt.show()