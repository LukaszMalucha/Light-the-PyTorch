# -*- coding: utf-8 -*-


######################################################################## Imports

import torch
import torch.nn as nn
import torchvision.transforms as transforms         ## transforming images
import torchvision.datasets as dsets                ## pytorch datasets
from torch.autograd import Variable


####################################################### Prepare Training Dataset

train_dataset = dsets.MNIST(root='./data',
                            train=True,                       ## training dataset
                            transform=transforms.ToTensor(),  ## read dataset elements to tensors
                            download=True)                    ## if you don't have it already 
                            
## type - tuple                            
type(train_dataset[0])   
                         
## Input Matrix - image(1) of 28x28 pixels
train_dataset[0][0].size()

## Label - digit 0-9
train_dataset[1][1]



############################################################### Display an Image

import matplotlib.pyplot as plt
import numpy as np

show_img = train_dataset[0][0].numpy().reshape(28,28) ## drop first dimension for matplotlib

plt.imshow(show_img, cmap='gray')


######################################################### Preparing Test Dataset

test_dataset = dsets.MNIST(root='./data',
                            train=False,                       
                            transform=transforms.ToTensor())  



################################################################# Specify Epochs

batch_size = 100

n_iters = 3000

num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)



######################################################## Create Iterable Objects

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)        ## different sequence every epoch
                                           
                                           
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)        ## one single forward pass                                          

## Iterability check

import collections
isinstance(train_loader, collections.Iterable)      ## True




######################################################### Building Log Reg Model

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()         ## inherit from nn module
        self.linear = nn.Linear(input_dim, output_dim)
        
    ## prediction    
    def forward(self, x):
        out = self.linear(x)
        return out



### Dimensions

input_dim = 28*28
output_dim = 10

## initiate model
model = LogisticRegressionModel(input_dim, output_dim)

##GPU
model.cuda()


## loss function
criterion = nn.CrossEntropyLoss()


## learning_rate
learning_rate = 0.001

## optimizer - update parameters on every iteration
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



##################################################################### Parameters

print(model.parameters)
#<bound method Module.parameters of LogisticRegressionModel(
#  (linear): Linear(in_features=784, out_features=10, bias=True)
#)>

## Coefficient - a in y=ax + b
print(list(model.parameters())[0].size())
#torch.Size([10, 784]) 

## Intercept/Bias - b in y=ax + b
print(list(model.parameters())[1].size())
#torch.Size([10]) 



#################################################################### Train Model

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):           ## for each tuple (im, labels) in training data loader
        
        ## Load images as variables
        images = Variable(images.view(-1, 28*28).cuda())
        labels = Variable(labels.cuda())
        
        # Clear gradients from previous iteration
        optimizer.zero_grad()
        
        # Passing image to model to get output
        outputs = model(images)
        
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        
        # Get the gradients after loss function
        loss.backward()
        
        # Updating parameters with gradients
        optimizer.step()
        
        iter += 1
        
        ## LOGISTIC REGRESSION 
        
        if iter % 500 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:

                images = Variable(images.view(-1, 28*28).cuda())
                
                # Forward pass only to get logits/output
                outputs = model(images)
                
                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)
                
                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                correct += (predicted.cpu() == labels.cpu()).sum()
            
            accuracy = 100 * correct / total
            
            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))

























