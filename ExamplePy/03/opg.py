from nr_python import *

DIR= '/home/aske/Semester/NUM/Eksamen/ExamplePy/02'
## Import data
Filip = pd.read_table(DIR + '/FilipData.dat',sep='\s+',names=['x','y'])
Pontius = pd.read_table(DIR + '/PontiusData.dat',sep='\s+',names=['x','y'])

## Design Matrix
FA = np.array([ [ Filip['x'][i]**j for j in range(3) ] for i in range(Filip['x'].size) ])
PA = np.array([ [ Pontius['x'][i]**j for j in range(11) ] for i in range(Pontius['x'].size) ])

## SVD
FU, FW, FVt = np.linalg.svd(FA,full_matrices=False)
PU, PW, PVt = np.linalg.svd(PA,full_matrices=False)
FV = FVt.T
PV = PVt.T

FC = FW[0]/FW[-1]
PC = PW[0]/PW[-1]

thresh = 0
FW_inv = np.array([ 0 if i <= thresh else 1/i for i in FW])
PW_inv = np.array([ 0 if i <= thresh else 1/i for i in PW])

Fy = np.array(Filip['y'])
Py = np.array(Pontius['y'])

Fx = FV @ np.diag( FW_inv ) @ FU.T @ Fy
Px = PV @ np.diag( PW_inv ) @ PU.T @ Py

Fre = np.linalg.norm( FA @ Fx - Fy) / np.linalg.norm(Fy) 
Pre = np.linalg.norm( PA @ Px - Py) / np.linalg.norm(Py) 
