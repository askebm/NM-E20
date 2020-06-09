from nr_python import *

## Define matrices
A = np.array([[0.114483, -0.604119, 0.282498, 0.161592],
    [ 0.571331, -0.135017,-0.952496, -0.459998],
    [ -0.881116, 0.423059,0.816948, 0.160139],
    [ 0.986944, 0.336941, 0.185398, 1.15178],
    [ -0.734637, -0.177438, 0.96215, 0.219571],
    [ -0.611998, 0.922288, 0.848041, 0.595876]],dtype=np.double)
b = np.array([ -0.253529, -0.557641, 0.704294, 0.184275, 0.389425, 0.992613],dtype=np.double)


### Part i
## SVD
u,w,vt = np.linalg.svd(A,full_matrices=False)
v=vt.T
condition = w[0]/w[-1]
### Part ii
## ---- Taken from earlier assignment ------
def apply_thresh(t=10**(-8)):
    _w = np.array([0 if i <= t else i for i in w])
    wi = np.array([0 if i <= 0 else 1/i for i in _w])
    x = v @ np.diag(wi) @ u.T @ b
    re = np.linalg.norm( A @ x -b ) / np.linalg.norm(b)
    N = v[:,0].size
    e = np.zeros(N)
    for j in range(N):
        s = 0
        for i in range(N):
            s += 0 if _w[i] == 0 else v[j,i]/_w[i]
        e[j] = np.sqrt(s**2 )
    return x,re,e
## ---------------------------------------
x,re,e = apply_thresh(t=0)

### Part iii
y = v[:,-1]
