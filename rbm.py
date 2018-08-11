# Restricted Boltzmann Machines

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn               ## nn
import torch.nn.parallel            ## parallel computation
import torch.optim as optim         ## optmizer
import torch.utils.data             ## tools
from torch.autograd import Variable ## stochastic gradient descent

# Importing the dataset - https://grouplens.org/datasets/movielens/
###                   path             ; separator ; header       ; for correct import ; for special chars in titles          
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')

## convert to numpy array for pytorch
training_set = np.array(training_set, dtype = 'int')
test_set = np.array(test_set, dtype = 'int')



# Creating matrixes of users and movies
# total number of users/movies by choosing maxiumu value for the column in tran/test set
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Matrix with users as rows and movies as columns(list of lists for Torch) 943x1682
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):             ## for each user 0 - 943
        id_movies = data[:,1][data[:,0] == id_users]    ## movies ratings for the user
        id_ratings = data[:,2][data[:,0] == id_users]   ## ratings for the user
        ratings = np.zeros(nb_movies)                   ## initialize list of zeros 
        ratings[id_movies - 1] = id_ratings             ## index of the movies -1 to match pyhon indicies 
        new_data.append(list(ratings))                  ## add to new list
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network

class RBM():
    def __init__(self, nv, nh):                       ## nv - visible nodes  , nh - hidden nodes
        self.W = torch.randn(nh, nv)                  ## Weights - Torch tensor, initialized with random number
        self.a = torch.randn(1, nh)                   ## bias for hidden nodes - 2d vector of nh elements
        self.b = torch.randn(1, nv)                   ## bias for visible nodes - 2d vector of nv elements
        
    def sample_h(self, x):                            ## SIGMOID ACTIVATION FUNCTION (x - visible neurons for probability count) 
        wx = torch.mm(x, self.W.t())                  ## Weights times visible neurons. '.t' - transpose
        activation = wx + self.a.expand_as(wx)        ## linear function of neurons - weights + bias applied to each line of wx
        p_h_given_v = torch.sigmoid(activation)       ## probabiliy of hidden node activation with given visible node value
        return p_h_given_v, torch.bernoulli(p_h_given_v)  ## return probability, yes/no aactivation for each hidden node

    def sample_v(self, y):                                                     ## y - given hidden node 
        wy = torch.mm(y, self.W)                                               ## no transpose 
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)                                ## reversed 
        return p_v_given_h, torch.bernoulli(p_v_given_h)                       ## return probability that visible node = '1'

    def train(self, v0, vk, ph0, phk):            ## input vector of ratings v0, visible nodes after k-iterations vk, first iteration probability vector ph0, probability of hidden nodes phk 
        self.W += (torch.mm(v0.t(),ph0) - torch.mm(vk.t(),phk)).t()                ## update weights 
        self.b += torch.sum((v0 - vk), 0)                                      ## update bias  
        self.a += torch.sum((ph0 - phk), 0)                                    ## update bias 


## Initialize variables
        
nv = len(training_set[0])     
nh = 100                 ## try different numbers     
batch_size = 50         ## fast training, if done online = 1          
rbm = RBM(nv, nh) 

# Training the RBM
nb_epoch = 10                                                     ## choose number of epochs
for epoch in range(1, nb_epoch + 1):                         
    train_loss = 0                                                ## init train loss
    s = 0.                                                        ## counter (float)
    for id_user in range(0, nb_users - batch_size, batch_size):   ## for users in a batch..., stop index, step size
        vk = training_set[id_user:id_user+batch_size]             ## init vk - input batch of observations 
        v0 = training_set[id_user:id_user+batch_size]             ## init v0 - batch of original ratings for comparison
        ph0,_ = rbm.sample_h(v0)                                  ## initial probabilities of hidden node activation | ,_ we just want first element
        for k in range(10):                                       ## trips from visible nodes to hidden nodes for updates 
            _,hk = rbm.sample_h(vk)                               ## _, = second element
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))