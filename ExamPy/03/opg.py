from nr_python import *

# Define knowns
f = lambda x: np.cos(x**2)*np.exp(-x**3)
limits = np.array([1,4],dtype=np.double)

### Part i
## ------ Taken from lecture 8 assignment ---------
def trapezoidal(f,interval,imax=10,err=10**(-8)):
    subdiv = 2
    _alpha_k = subdiv**2
    n_interval = 1
    x0 = interval[0]
    x1 = interval[1]
    dist = np.float64(x1-x0)
    cols = [r'$A_i$',r'$A_{i-1}-A_i$',r'Richardson $\alpha^k$',
            r'Richardson error',r'Number of f computations ']
    df = pd.DataFrame(
            columns=cols,
            dtype=np.float64)
    _sum = np.float64(f(x0)+f(x1))/2.
    for i in np.arange(imax):
        df = df.append(pd.Series(dtype=np.float64),ignore_index=True)
        step = dist/n_interval
        for fi in np.arange(x0+step,x1,step=step*2):
            _sum += f(fi)
        df[cols[0]][i] = _sum*step
        df[cols[1]][i] = (df[cols[0]][i-1]-df[cols[0]][i]) if 1<i else np.nan
        df[cols[2]][i] = (df[cols[1]][i-1]/df[cols[1]][i]) if 2<i else np.nan
        df[cols[3]][i] = (df[cols[1]][i]/(_alpha_k-1)) if 2<i else np.nan
        df[cols[4]][i] = n_interval+1
        n_interval *= subdiv
        if np.abs(df[cols[3]][i]) < err:
            break
    df.index.name='i'
    return df
## ----------------------------------
table = trapezoidal(f,limits)

### Part ii
f = lambda x: (1/np.sqrt(x)) * np.cos(x**2) * np.exp(-x**3)
limits = np.array([0,1],dtype=np.double)
# Rectangle.  See latex

### Part iii

## -------------- Taken from lecture 9 assignment ---------
def DErule(f,interval):
    a = interval[0]
    b = interval[1]
    c = 1
    _x = lambda t: (b+a)/2 + (b-a) * np.tanh( c * np.sinh(t) ) / 2
    sech = lambda x: 1/np.cosh(x)
    _dxdt = lambda t: (1/2)*(b-a) * ( sech( c*np.sinh(t) )**2 ) * c * np.cosh(t)
    _d = lambda t: (b-a) * np.exp( -2*np.sinh(t) )/( 1+np.exp(-2*np.sinh(t)) )
    F = lambda t: f(a+_d(-t))*_dxdt(t) if t<0 else f(b-_d(t))*_dxdt(t)
    return F
## ----------------------------------

table3 = trapezoidal(DErule(f,limits),[-4.3,4.3]).iloc[:,[0,1,-1]]
