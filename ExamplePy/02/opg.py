from nr_python import *

DIR= '/home/aske/Semester/NUM/Eksamen/ExamplePy/02'
## Import data
Filip = pd.read_table(DIR + '/FilipData.dat',sep='\s+',names=['x','y'])
Pontius = pd.read_table(DIR + '/PontiusData.dat',sep='\s+',names=['x','y'])

## Design Matrix
FA = np.array([ [ Filip['x'][i]**j for j in range(3) ] for i in range(Filip['x'].size) ])
PA = np.array([ [ Pontius['x'][i]**j for j in range(11) ] for i in range(Pontius['x'].size) ])

## Normal Equations
FAA = FA.T @ FA
PAA = PA.T @ PA

Fb = FA.T @ np.array(Filip['y'])
Pb = PA.T @ np.array(Pontius['y'])

## Solve LU
FxLU = scl.lu_solve(scl.lu_factor(FAA),Fb)
PxLU = scl.lu_solve(scl.lu_factor(PAA),Pb)

## Solve Cholesky
FxCho = scl.cho_solve(scl.cho_factor(FAA),Fb)
PxCho = scl.cho_solve(scl.cho_factor(PAA),Pb)
