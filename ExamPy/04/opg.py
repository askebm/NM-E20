from nr_python import *
import scipy.optimize as sco
import sys

## Define knowns
rTarget = 0.7
rSSI = 3.25
y0 = np.array([0.0045])

r1 = lambda t: rTarget + (rSSI - rTarget) * np.exp(-t/2)
f1 = lambda t,y: np.array([ - y[0] + r1(t)*y[0]*(1-y[0])])

start = 0.0
end = 8.0

### Part i

## ----------------- Taken from lecture 10 assignment ---
def midpointStep(initial,h,_f,end):
    y = np.array(initial,ndmin=2).astype(np.float64)
    i = np.int(0)
    while i*h < end:
        xn = h*i
        k1 = h*_f(xn,y[i])
        k2 = h*_f(xn+h/2,y[i]+k1/2)
        y = np.vstack((y,y[i]+k2)).astype(np.float64)
        i += 1
    return y.flatten()
## -----------------------------------------------

def midpointStepTable(initial,_f,_end,N=8.,maxiter=50):
    cols = ['$f_i$','$f_i-f_{i-1}$',r'richardson $\alpha^k$',
        r'richardson error',r'number of f computations ']
    df = pd.DataFrame(columns=cols)
    df.index.name='i'
    subdiv = 2.
    h = _end/N
    _alpha_k = subdiv**2
    for i in range(maxiter):
        df = df.append(pd.Series(dtype=np.float64),ignore_index=True)
        y = midpointStep(initial,h,_f,_end)
        df[cols[0]][i] = y[-1]
        df[cols[1]][i] = df[cols[0]][i-1]-df[cols[0]][i] if 1<i else np.nan
        df[cols[2]][i] = df[cols[1]][i-1]/df[cols[1]][i] if 2<i else np.nan
        df[cols[3]][i] = df[cols[1]][i]/(_alpha_k-1) if 2<i else np.nan
        df[cols[4]][i] = df[cols[4]][i-1] + y.size -1 if 0<i else y.size -1
        
        error = np.linalg.norm(df[cols[3]][i])
        h /= subdiv
        if error < 10**(-6):
            break
    return df
## 
table1 = midpointStepTable(y0,f1,end)

### Part ii

rReal = 2.0
fIn = 0.002
r2 = lambda t: rTarget + (rReal - rTarget) * np.exp(-t)
f2 = lambda t,y: np.array([ - y[0] + r2(t)*y[0]*(1-y[0]) + fIn*np.exp(-t)])

## ----------------- Taken from lecture 10 assignment ---
def trapzoidalStep(initial,h,_f,end):
    N = np.size(initial)
    y = np.array(initial,ndmin=2).astype(np.float64)
    i = np.int(0)
    while h*i <= end:
        y_guess = y[i] + h*_f(h*i,y[i])
        f_newton = lambda _y: _y - y[i] - (h/2) * (_f(h*i,y[i])+_f(h*i+h,_y))
        y_root = sco.newton(f_newton,y_guess)
        y = np.vstack((y,y_root)).astype(np.float64)
        i += 1
    return y.flatten()
## --------------------------------------------------
y1 = trapzoidalStep(y0,0.2,f2,end)[-1]
y2 = trapzoidalStep(y0,0.1,f2,end)[-1]

richard_error = (y2-y1)/(4-1)

### Part iii
p1N = table1.iloc[-1,-1] - table1.iloc[-2,-1]
p1h = end/p1N
p1y = 60000*midpointStep(y0,p1h,f1,end)
p1t = np.linspace(start,end,p1y.size)

p2y = 60000*trapzoidalStep(y0,0.1,f2,end)
p2t = np.linspace(start,end,p2y.size)


## --- Take from  https://matplotlib.org/3.1.1/gallery/userdemo/pgf_fonts.html#sphx-glr-gallery-userdemo-pgf-fonts-py
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": []                    # use latex default serif font
})
## ----------------------------------------------
fig, ax = plt.subplots()
ax.plot(p1t,p1y,label=r'Solution i')
ax.plot(p2t,p2y,label=r'Solution ii')
ax.set_xlabel('Time')
ax.set_ylabel('People hospitalized')
legend = ax.legend()


