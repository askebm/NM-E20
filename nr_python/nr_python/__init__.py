import numpy as np
import scipy as sc
import scipy.linalg as scl
import sympy as sp
import pandas as pd


## GramSmith
def GramSmith(x,thresh=0.001):
    rows,cols = x.shape
    e = x.copy();
    e[:,0] = x[:,0]/np.linalg.norm(x[:,0])
    for i in range(1,cols):
        e[:,i] = x[:,i]
        for j in range(0,i):
            e[:,i] -= (x[:,i].transpose()*(e[:,j]))[0,0]*e[:,j]
        l = np.linalg.norm(e[:,i])
        if (thresh < l):
            e[:,i] /= np.linalg.norm(e[:,i])
        else:
            e[:,i] = np.zeros(e[:,i].shape,dtype=e.dtype)
    return e

def bLS(b,u):
    rows,cols = u.shape
    result = zeros(b.shape,dtype=float)
    for k in range(0,cols):
        result += (b.transpose()*u[:,k])[0,0]*u[:,k]
    return result

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

