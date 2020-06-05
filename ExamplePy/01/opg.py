from nr_python import *

## Declare variables
A=np.array([[1,2,3],[2,-4,6],[3,-9,-3]])
b=np.array([5,18,6],ndmin=2).T

x = scl.lu_solve(scl.lu_factor(A),b)
