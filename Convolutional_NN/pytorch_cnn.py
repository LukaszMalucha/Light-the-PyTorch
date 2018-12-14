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
                            
                            
######################################################### Preparing Test Dataset

test_dataset = dsets.MNIST(root='./data',
                            train=False,                       
                            transform=transforms.ToTensor())  
            
######################################################## Create Iterable Objects

batch_size = 100
n_iters = 6000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)        ## different sequence every epoch
                                                                                      
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)        ## one single forward pass    
                                          
   
########################### Building Convolutional Neural Network Neural Network  

class CNNModel(nn.Module):
    
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Convolution 1        ## grayscale  ## feature maps  ## filter size
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        
        # Max pooling 1  
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        
        # Max pooling 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected layer
        self.fc1 = nn.Linear(32 * 4 * 4, 10)
        
    def forward(self, x):
        
        out = self.cnn1(x)
        out = self.relu1(out)
        
        out = self.maxpool1(out)
        
        out = self.cnn2(out)
        out = self.relu2(out)
        
        out = self.maxpool2(out)
        
        ## resize from (100,32,7,7) to (100, 32*7*7)
        out = out.view(out.size(0), -1)  ## -1 means reshape to reamining values
        
        out = self.fc1(out)
        
        return out
        
model = CNNModel()    


##GPU
if torch.cuda.is_available():
    model.cuda()    
            
            
criterion = nn.CrossEntropyLoss()            
            
learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
              
            
            
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):           ## for each tuple (im, labels) in training data loader
        
        ## Load images as variables
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)
            
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
                
                if torch.cuda.is_available():
                    images = Variable(images.cuda())
                else:
                    images = Variable(images)
                
                # Forward pass only to get logits/output
                outputs = model(images)
                
                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)
                
                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / total
            
            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))    
    
    
    
##### SAVE MODEL


torch.save(model.state_dict(), 'cnn_model.pkl')



##### LOAD MODEL


model.load_state_dict(torch.load('cnn_model.pkl'))                
            
            
            
            
            
            
            
            
            
            
            
            
            
        
    
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                          
                                                                                    