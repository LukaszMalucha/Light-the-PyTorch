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
                                           
                                           
 ############################### Building Recurrent Neural Network Neural Network 
                                           

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        ## Hidden dimensions
        self.hidden_dim = hidden_dim
        
        ## Hidden layers
        self.layer_dim = layer_dim
        
        
        ## Building RNN (batch_first - means that tensor will have a shape of (batch_dim, seq_dim, input_dim))
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True) ## no activation function
        
        
        ## Redout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
## Each forward consist of 28 steps, but we want to get only last step            
    def forward(self, x):
        # Initialize hidden state with zeros (layer_dim, batch_size, hidden_dim)
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:    
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        
        ## Initialize cell state
        if torch.cuda.is_available():
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:    
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        
        # 28 time steps
        out, (hn, cn) = self.lstm(x, (h0,c0))
        
        ## get final prediction of hiddent states
        out = self.fc(out[:, -1, :])   ## transition form 100,28,100 size into 100,100
        
        return out
    
    
input_dim = 28
hidden_dim = 100
layer_dim = 2
output_dim = 10

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    
##GPU
if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss()        
            
learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)                                           
                                 
          
##  6 3x coefficient & bias                                             
for i in range (len(list(model.parameters()))):
    print(list(model.parameters())[i].size())                                           

                                         
                                           
                                           
                                           
                                           
seq_dim = 28                                           
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):           ## for each tuple (im, labels) in training data loader
        
        ## Load images as variables
        if torch.cuda.is_available():
            images = Variable(images.view(-1, seq_dim, input_dim).cuda())       ## resize image
            labels = Variable(labels.cuda())
        else:
            images = Variable(images.view(-1, seq_dim, input_dim))
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
                    images = Variable(images.view(-1, seq_dim, input_dim).cuda())
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


torch.save(model.state_dict(), 'lstm_model.pkl')



##### LOAD MODEL


model.load_state_dict(torch.load('lstm_model.pkl'))                                                                  
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           