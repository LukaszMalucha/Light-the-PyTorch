# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn as nn
import torch.tensor as tensor
from torch.autograd import Variable

import matplotlib.pyplot as plt




'''
INPUTS & LABELS
''' 


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




'''
STEP 1: CREATE MODEL CLASS
'''
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  
    
    def forward(self, x):
        out = self.linear(x)
        return out

'''
STEP 2: INSTANTIATE MODEL CLASS
'''
input_dim = 1
output_dim = 1

model = LinearRegressionModel(input_dim, output_dim)


############ GPU #########
if torch.cuda.is_available():
    model.cuda()
#########################
'''
STEP 3: INSTANTIATE LOSS CLASS
'''

criterion = nn.MSELoss()

'''
STEP 4: INSTANTIATE OPTIMIZER CLASS
'''

learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'''
STEP 5: TRAIN THE MODEL
'''
epochs = 100
for epoch in range(epochs):
    epoch += 1
    # Convert numpy array to torch Variable
    
    ############ GPU #########
    inputs = Variable(torch.from_numpy(x_train).cuda())
    labels = Variable(torch.from_numpy(y_train).cuda())
    #########################
    
    
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