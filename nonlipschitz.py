#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sympy
import numpy as np
import random

needed_prime_length = 33
non_lipschitz = 0
max_len=0
alpha=0.01
while max_len!=needed_prime_length:
    b=random.getrandbits(needed_prime_length)
    print(b)
    p = sympy.nextprime(b)
    bin_p = bin(p)[2:]
    max_len = len(bin_p)

below=0
while(below==0):
    a = random.getrandbits(max_len)
    if(a<=p-1):
        below=1 
print(p)
print(a)
print(max_len)

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"     
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

# In[2]:


import random
from tqdm import tqdm
X = []
Y = []
x = 0
for i in range(100000):
    below=0
    while(below==0):
        x = random.getrandbits(max_len)
        if(x<=p-1):
            below=1 
    b=bin(x)[2:]
    result = "0"*(max_len-len(b))+b
    X.append([int(el) for el in result])
    noise = round(np.random.normal(0, alpha*p, 1)[0])%p
    b = bin((a*x+noise)%p)[2:]
    result = "0"*(max_len-len(b))+b
    Y.append([int(result[1])])
#    Y.append([int(result[max_len-1])]) #[int(el) for el in result[::-1]]


# In[3]:


X = np.array(X, dtype='float32')
Y = np.array(Y, dtype='float32')
print(X.shape)
print(Y.shape)


# In[4]:


import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

class BaseModel1(nn.Module):
    def __init__(self, ):
        super(BaseModel1, self).__init__()
        self.linear1 = nn.Linear(max_len, 1000) 
        self.linear2 = nn.Linear(1000, 1000)
        self.linear3 = nn.Linear(1000, 1)
        
    def forward(self, inputs):
        h1 = torch.tanh(self.linear1(inputs))
        h2 = torch.tanh(self.linear2(h1))
        h3 = self.linear3(h2)
        if non_lipschitz==1:
            logits = h3.sign()*h3.abs().pow(0.5)
        else:
            logits = h3
        return logits
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# In[5]:


batch_size = 100
X_train = torch.from_numpy(X_train) # transform to torch tensor
y_train = torch.from_numpy(y_train)

X_test = torch.from_numpy(X_test) # transform to torch tensor
y_test = torch.from_numpy(y_test)


train_dataset = TensorDataset(X_train,y_train) # create your datset
train_loader = DataLoader(train_dataset, batch_size = batch_size)

test_dataset = TensorDataset(X_test,y_test) # create your datset
test_loader = DataLoader(test_dataset, batch_size = batch_size)


# In[6]:


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, inputs, targets):        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = 1-2*targets.view(-1)
        myloss=torch.sum(F.relu(1-inputs*targets))
        return myloss


# In[7]:


EPOCHS = 1000
model = BaseModel1()
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

data = []

criterion = MyLoss()

for i in range(EPOCHS):
    train_correct = 0
    total_train_loss = 0
    total_test_loss = 0
    test_correct = 0
    
    for (x,y) in train_loader:
        ####
        model.zero_grad()
        optimizer.zero_grad()
        ####
        x = x.cuda()
        y = y.cuda()    
        logits = model(x)

        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0, error_if_nonfinite=False)
        optimizer.step()
        total_train_loss += loss.item()
        train_correct += (logits.view(-1)*(1-2*y.view(-1)) > 0).sum().item()

    with torch.no_grad():
        for (x,y) in test_loader:
            x = x.cuda()
            y = y.cuda()    

            logits = model(x)
            loss = criterion(logits, y)
            total_test_loss += loss.item()
            test_correct += (logits.view(-1)*(1-2*y.view(-1)) > 0).sum().item()

    
    
    train_acc = train_correct/len(X_train)
    train_loss = total_train_loss/len(X_train)
    test_loss = total_test_loss/len(X_test)
    test_acc = test_correct/len(X_test)
    data.append([train_acc, train_loss, test_loss, test_acc])
    if i % 10==0:
        print(i,"train_acc:", round(train_acc, 6), 'train_loss:',
              round(train_loss, 7), 'val_loss:', round(test_loss, 6), 
              "val_acc:", round(test_acc, 6))
    if (test_acc>=1) and (train_acc>=1):
        break
    # if val_acc>=best_val_acc:
    #     best_val_acc = val_acc
    #     best_state_dict = copy.deepcopy(model.state_dict())


# In[ ]:




