# -*- coding: utf-8 -*-


######################################################################## Imports

import torch                                                            ## main library
import torch.nn as nn                                                   ## neaural network
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim                                             ## optimizer
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler   ## moving data to torch
from torch.autograd import Variable                                     ## data loading
from collections import OrderedDict


import pandas as pd  ## reading dataset
import numpy as np   ## arrays


### Reading Dataset

dataset = pd.read_csv('Churn_Modelling.csv')

################################################################## PREPROCESSING

### Separating features from labels
X = dataset.iloc[:,3:13].values   ## skip customer id, row number
y = dataset.iloc[:,13].values 


### Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])  ## Countries

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])  ## Gender


### One Hot Encoding
onehotencoder = OneHotEncoder(categorical_features = [1])  ## only countries as gender is eithet 1 or 0
X = onehotencoder.fit_transform(X).toarray()


### Avoid Dummy Variable Trap
X = X[:, 1:]


### Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


### Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

######################################################## Create Iterable Objects

batch_size = 64
n_iters = 4000
num_epochs = n_iters / (len(dataset) / batch_size)
num_epochs = int(num_epochs)



### Torch from numpy

X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train).view(-1,1)

X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test).view(-1,1)



## train, test sets into torch datasets
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test,y_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)        ## different sequence every epoch
                                                                                      
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)        ## one single forward pass    




############################################# Building Artificial Neural Network


class ANNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):
        super(ANNModel, self).__init__()                        ## inherit from module
        
        ## LAYERS(in,out)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_dim_1,hidden_dim_1)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(hidden_dim_1,hidden_dim_1)
        self.relu3 = nn.ReLU()
        
        self.fc4 = nn.Linear(hidden_dim_1,hidden_dim_1)
        self.relu4 = nn.ReLU()
        
        self.fc5 = nn.Linear(hidden_dim_1,hidden_dim_2)
        self.relu5 = nn.ReLU()
        
        self.fc6 = nn.Linear(hidden_dim_2,hidden_dim_2)
        self.relu6 = nn.ReLU()
        
        self.output = nn.Linear(hidden_dim_2, output_dim)
        
        self.dropout = nn.Dropout(0.1)
        
     
    def forward(self,x):
        ## Linear function
        out = self.fc1(x)
        ## Non-linearity        
        out = self.relu1(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        
        out = self.dropout(out)
        
        out = self.fc3(out)
        out = self.relu3(out)
        
        out = self.fc4(out)
        out = self.relu4(out)
        
        out = self.fc5(out)
        out = self.relu5(out)
        
        out = self.fc6(out)
        out = self.relu6(out)
        
        ## Output
        out = self.output(out)
        
        return nn.Sigmoid()(out)                                  
        
        
## Neural Network Parameters

input_dim = 11          # dataset columns
hidden_dim_1 = 128       # first hidden layer
hidden_dim_2 = 64       # second hidden layer
output_dim = 1          # Exited or not        
        
        
model = ANNModel(input_dim, hidden_dim_1, hidden_dim_2, output_dim)        


##GPU
if torch.cuda.is_available():
    model.cuda()

criterion = nn.BCELoss()

learning_rate = 0.01

optimizer = optim.SGD(model.parameters(), lr=learning_rate)  


#################################################################### TRAIN MODEL  

      
iter = 0
for epoch in range(num_epochs):
    


    
    for data, labels in train_loader:           ## for each tuple (im, labels) in training data loader
        
        ## Load data as variables
        if torch.cuda.is_available():
            data = Variable(data).float().cuda()
            labels = Variable(labels).type(torch.FloatTensor).cuda()
        else:
            data = Variable(data).float()
            labels = Variable(labels).type(torch.FloatTensor)
            
        # Clear gradients from previous iteration
        optimizer.zero_grad()
        
        # Passing data to model to get output
        outputs = model(data)

        
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        
        # Get the gradients after loss function
        loss.backward()
        
        # Updating parameters with gradients
        optimizer.step()

        
        iter += 1
        
        # Test Set
        
        if iter % 500 == 0:
            
            val_loss = 0.0    
        
            correct = 0
            total = 0    
                    
            for data, labels in test_loader:
                
                ## Check for graphic card support
                if torch.cuda.is_available():
                    data = Variable(data).float().cuda()
                    labels = Variable(labels).float().cuda()
                    
                else:
                    data = Variable(data).float()
                    labels = Variable(labels).float()
                    
                # forward pass: compute predicted outputs by passing inputs to the model    
                outputs = model(data)    
                
                
                predicted = (torch.round(outputs.data[0]))
                
                total += labels.size(0)
                             
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()
            
            
            accuracy = 100 * correct/ float(total)
        
           
            print('Iteration: {}. Loss: {:.4f}. Accuracy: {:.2f}%'.format(iter, loss.item(), accuracy))  
                  




 
#    
###### SAVE MODEL
#
#
#torch.save(model.state_dict(), 'feedforward_model.pkl')
#
#
#
###### LOAD MODEL
#
#
#model.load_state_dict(torch.load('feedforward_model.pkl'))            
#        
#        
        
        
        
        
        
        
        
        

















