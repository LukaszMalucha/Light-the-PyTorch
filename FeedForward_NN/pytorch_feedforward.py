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
                                          
                                          
                                          
                                          
                                          
########################################### Building Feed Forward Neural Network                                 
                                          
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedforwardNeuralNetModel, self).__init__()
        ## linear function - connecting to hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        ## non-linear function 
        self.relu1 = nn.ReLU()
        
### LAYER 2       
        ## linear function - from hidden layer to hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        ## non-linear function 
        self.relu2 = nn.ReLU()
            
        ## linear function (readout) - from hidden layer to output
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)        
        
        
        
        
        
    def forward(self, x):
        ## Linear function
        out = self.fc1(x)
        ## Non-linearity
        out = self.relu1(out)
        ## Linear function (readout)
        out = self.fc2(out)
        ## Non-linearity
        out = self.relu2(out)     
        ## Linear function (readout)
        out = self.fc3(out)        
        return out
    
    
input_dim = 28*28
hidden_dim = 100
output_dim = 10

model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    
##GPU
if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss()

learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
  
    
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):           ## for each tuple (im, labels) in training data loader
        
        ## Load images as variables
        if torch.cuda.is_available():
            images = Variable(images.view(-1, 28*28).cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images.view(-1, 28*28))
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
                    images = Variable(images.view(-1, 28*28).cuda())
                else:
                    images = Variable(images.view(-1, 28*28))
                
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


torch.save(model.state_dict(), 'feedforward_model.pkl')



##### LOAD MODEL


model.load_state_dict(torch.load('feedforward_model.pkl'))    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    














                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          