import numpy as np
import scipy as sc
import scipy.linalg as scl
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt



def pprint(num):
    print("%10.3e" % num)

## GramSmith
def GramSmith(_x,thresh=0.001):
    rows,cols = _x.shape
    _e = _x.copy()
    _e[:,0] = _x[:,0]/np.linalg.norm(_x[:,0])
    for i in range(1,cols):
        _e[:,i] = _x[:,i]
        for j in range(0,i):
            _e[:,i] -= np.dot(_x[:,i],_e[:,j])*_e[:,j]
        l = np.linalg.norm(_e[:,i])
        if (thresh < l):
            _e[:,i] /= l
        else:
            _e[:,i] = np.zeros(_e[:,i].shape,dtype=_e.dtype)
    return _e

def bLS(_b,_u):
    rows,cols = _u.shape
    _result = np.zeros(_b.shape,dtype=np.double)
    for k in range(0,cols):
        _result += np.dot(_b,_u[:,k])*_u[:,k]
    return _result

def _dk(x):
    result = np.zeros(x.size)
    for i in range(1,x.size):
        result[i] = x[i]-x[i-1]
    return result


## Bisect
def bisect(f,x0,y0,precision=10**(-8)):
    i = 0
    x = np.array([x0],dtype=np.float64)
    y = np.array([y0],dtype=np.float64)
    while np.abs(x[i]-y[i])>=precision:
        x = np.append(x,(x[i]+y[i])/2)
#        print("x: ",x[i]," y: ",y[i]," i: ",i," x-y: ",x[i]-y[i])
        if f(x[i+1])*f(y[i])<0:
            y = np.append(y,y[i])
        else:
            y = np.append(y,x[i])
        i+=1
    return np.arange(i+1)+1 ,x,_dk(x)

## Secant

def secant(f,x0,x1,precision=10**(-8)):
    i=1
    x = np.array([x0,x1],dtype=np.float64)
    y = f(x)
    while np.abs(y[i])>precision:
        x = np.append(x, x[i]-( (x[i]-x[i-1])/(y[i]-y[i-1]) )*y[i] )
        i+=1
        y = np.append(y,f(x[i]))
    return np.arange(i+1)+1,x,_dk(x)

## False position

def false_position(f,x0,y0,precision=10**(-8)):
    i = 0
    x = np.array([x0],dtype=np.float64)
    y = np.array([y0],dtype=np.float64)
    while np.abs(x[i]-y[i])>=precision:
        x = np.append(x,x[i]-( (x[i]-y[i])/(f(x[i])-f(y[i])) )*f(x[i]) )
#        print("x: ",x[i]," y: ",y[i]," i: ",i," x-y: ",x[i]-y[i])
        if f(x[i+1])*f(y[i])<0:
            y = np.append(y,y[i])
        else:
            y = np.append(y,x[i])
        i+=1
    return np.arange(i+1)+1 ,x,_dk(x)


## Ridder
def ridder(f,x0,y0,precision=10**(-8)):
    i = 0
    x = np.array([x0],dtype=np.float64)
    y = np.array([y0],dtype=np.float64)
    while np.abs(x[i]-y[i])>=precision:
        z = (x[i]+y[i])/2
        x = np.append(x,z + (z-x[i])*(np.sign(f(x[i])-f(y[i]))*f(z))/np.sqrt(f(z)**2-f(x[i])*f(y[i])) )
#        print("x: ",x[i]," y: ",y[i]," i: ",i," x-y: ",x[i]-y[i])
        if f(x[i+1])*f(z)<0:
            y = np.append(y,z)
        elif f(x[i+1])*f(y[i])<0:
            y = np.append(y,y[i])
        else:
            y = np.append(y,x[i])
        i+=1
    return np.arange(i+1)+1 ,x,_dk(x)


## Newton method
def newton(f,f_prime,x0,precision=10**(-8)):
    i=0
    x = np.array([x0],dtype=np.float64)
    y = f(x)
    while np.abs(y[i])>precision:
        x = np.append(x, x[i] - ( 1/f_prime(x[i]) ) * y[i] )
        i+=1
        y = np.append(y,f(x[i]))
    return np.arange(i+1)+1,x,_dk(x)

