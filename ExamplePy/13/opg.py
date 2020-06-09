from nr_python import *


N=4
inx = lambda j,k: (N-1)*(k-1)+j
A = np.zeros(((N-1)**2,(N-1)**2))
phi = np.zeros((N-1)**2)
a0,a1,b0,b1 = 0,0,0,0
h = 1 / N
y = 0
f = lambda x,y: 1 + x + y

for j in np.arange(N-1)+1:
    for k in np.arange(N-1)+1:
        i = inx(j,k) - 1
        A[i,i] = 4+(h**2)*y
        cnt = 4
        phi[i] = h**2*f(j*h,k*h)
        if 1 <= j-1:
            A[i,inx(j-1,k)-1] = -1
        else:
            phi[i] += b0
        if 1 <= k-1:
            A[i,inx(j,k-1)-1] = -1
        else:
            phi[i] += a0
        if k+1 <= N-1:
            A[i,inx(j,k+1)-1] = -1
        else:
            phi[i] += b1
        if j+1 <= N-1:
            A[i,inx(j+1,k)-1] = -1
        else:
            phi[i] += a1

