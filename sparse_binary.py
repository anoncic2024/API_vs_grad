#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
from math import comb

q=3329
n=554
N=1000

def ax_bx (n, q, a, b):
    res = 0.0
    for i in range(n):
        res = res + a[i]*b[i]*(q-1)*(2*q-1)/(6*q**2)
        for j in range(n):
            if i!=j:
                res = res + a[i]*b[j]*(q-1)**2/(4*q**2)
    return res - (q-1)**2/(4*q**2)

def general(n, q):
    a = np.random.randint(0, q, size=(N,n))
    b = np.random.randint(0, q, size=(N,n))
    res = 0.0
    for i in range(N):
        axbx = ax_bx(n,q,a[i,:],b[i,:])
        res = res+axbx**2
    return -2*np.log(q)+0.5*np.log(res/N)



def binary(n, q):
    a = np.random.randint(0, 2, size=(N,n))
    b = np.random.randint(0, 2, size=(N,n))
    res = 0.0
    for i in range(N):
        axbx = ax_bx(n,q,a[i,:],b[i,:])
        res = res+axbx**2
    return -2*np.log(q)+0.5*np.log(res/N)

def sparse(n, q):
    rg = range(n)
    ss = int(n*0.2)+1
    
    a = np.zeros((N,n))
    for_a = [random.sample(rg, ss) for _ in range(N)]
    i=0
    for h in for_a:
        for j in h:
            a[i,j]=1
        i=i+1
    b = np.zeros((N,n))
    for_b = [random.sample(rg, ss) for _ in range(N)]
    i=0
    for h in for_b:
        for j in h:
            b[i,j]=1
        i=i+1
    res = 0.0
    for i in range(N):
        axbx = ax_bx(n,q,a[i,:],b[i,:])
        res = res+axbx**2
    return -2*np.log(q)+0.5*np.log(res/N)

def ternary(n, q):
    a = np.random.randint(-1, 2, size=(N,n))
    b = np.random.randint(-1, 2, size=(N,n))
    res = 0.0
    for i in range(N):
        axbx = ax_bx(n,q,a[i,:],b[i,:])
        res = res+axbx**2
    return -2*np.log(q)+0.5*np.log(res/N)

def sparse_ternary(n, q):
    rg = range(n)
    ss = int(n*0.2)+1
    
    a = np.zeros((N,n))
    for_a = [random.sample(rg, ss) for _ in range(N)]
    i=0
    for h in for_a:
        for j in h:
            rrr = np.random.randint(0, 2)
            if rrr==1:
                a[i,j]=1
            else:
                a[i,j]=q-1
        i=i+1
    b = np.zeros((N,n))
    for_b = [random.sample(rg, ss) for _ in range(N)]
    i=0
    for h in for_b:
        for j in h:
            rrr = np.random.randint(0, 2)
            if rrr==1:
                b[i,j]=1
            else:
                b[i,j]=q-1
        i=i+1
    res = 0.0
    for i in range(N):
        axbx = ax_bx(n,q,a[i,:],b[i,:])
        res = res+axbx**2
    return -2*np.log(q)+0.5*np.log(res/N)


# In[ ]:


for n in range(50,1000):
    a = general(n, 3329)
    b = binary(n, 3329)
    c = sparse(n, 3329)
    d = ternary(n, 3329)
    e = sparse_ternary(n, 3329)
    print("general {} {}\n".format(n,a))
    print("binary {} {}\n".format(n,b))
    print("sparse {} {}\n".format(n,c))
    print("ternary {} {}\n".format(n,d))
    print("sparse ternary {} {}\n".format(n,e))


# In[ ]:




