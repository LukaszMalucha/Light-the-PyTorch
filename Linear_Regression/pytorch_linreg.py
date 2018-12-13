# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn as nn
import torch.tensor as tensor
from torch.autograd import Variable

import matplotlib.pyplot as plt




x_values = [i for i in range(11)]


## Convert to np array

x_train = np.array(x_values, dtype=np.float32)
x_train.shape


## Covert to 2d array

x_train = x_train.reshape(-1, 1)
x_train.shape


## Fit algorithm

y_values = [2*i + 1 for i in x_values]

## Convert to numpy array

y_train = np.array(y_values, dtype=np.float32)



# IMPORTANT: 2D required
y_train = y_train.reshape(-1, 1)

##### Linear regression class

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()   ## inherit nn.Module 
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        out = self.linear(x)
        return out
    
input_dim = 1
output_dim = 1

model = LinearRegressionModel(input_dim, output_dim)


##### Loss Class  - mean square error

criterion = nn.MSELoss()

##### Optimizer

learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 100



for epoch in range(epochs):
    epoch += 1
    # Convert numpy array to torch Variable
    inputs = torch.from_numpy(x_train).requires_grad_()
    labels = torch.from_numpy(y_train)

    # Clear gradients w.r.t. parameters
    optimizer.zero_grad() 
    
    # Forward to get output
    outputs = model(inputs)
    
    # Calculate Loss
    loss = criterion(outputs, labels)
    
    # Getting gradients w.r.t. parameters
    loss.backward()
    
    # Updating parameters
    optimizer.step()
    
    print('epoch {}, loss {}'.format(epoch, loss.item()))


### Compare data

predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()


# Clear figure
plt.clf()

# Get predictions
predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()

# Plot true data
plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)

# Plot predictions
plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)

# Legend and plot
plt.legend(loc='best')
plt.show()


##### SAVE MODEL


torch.save(model.state_dict(), 'linreg_model.pkl')



##### LOAD MODEL


model.load_state_dict(torch.load('linreg_model.pkl'))




   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    