from nr_python import *
import scipy.optimize as sco

## Define knowns
a1 = np.double(3.157578)
a2 = np.double(0.574858)
q1 = np.double(0.875492)
q2 = np.double(0.936386)
x1 = np.double(2.34174 )
x2 = np.double(2.90639 )

f1 = lambda _dq1,_dq2: a1* np.cos(q1+_dq1)+a2*np.cos(q2+_dq2) -x1
f2 = lambda _dq1,_dq2: a1* np.sin(q1+_dq1)+a2*np.cos(q2+_dq2) -x2
f = lambda _dq: np.array([f1(_dq[0],_dq[1]),f2(_dq[0],_dq[1])],dtype=np.double)
f_prime = lambda _dq: np.array([[-a1*np.sin(_dq[0] + q1), -a2*np.sin(_dq[1] + q2)],
    [a1*np.cos(_dq[0] + q1), -a2*np.sin(_dq[1] + q2)]])

### Part i
f_0 = f([0,0])

### Part ii
## --- Partly inspired from lecture 07 ----
# Does not contains divide by zero protection in backtrack more part of the code
# nor does it contain  code that ensures that the new y will be between y2*0.5 and y1*0.1
def newton(_F,_fprime,_x0,precision=10**(-8),alpha=10**(-4)):
    _f = lambda _dq: (1/2) * _F(_dq).dot(_F(_dq))
    _df = lambda _dq: _F(_dq)@_fprime(_dq)
    _xNew = _x0
    _g1 = _f(_x0)
    _data = np.array([_g1,np.nan,np.nan,np.nan],dtype=np.double,ndmin=2)
    _k = 0
    while True:
        _k += 1
        _xOld = _xNew
        _g0 = _g1
        _J = _fprime(_xOld)
        _Ji = np.linalg.inv(_J)
        _step = -_Ji@_F(_xOld)
        _xNew = _xOld + _step
        _g1 = _f(_xNew)
        _g0p = -_g0
        _c1 = _g1
        _c2 = _g0 + alpha * (_df(_xOld).dot(_xNew-_xOld))
        if _c1 > _c2:
            #Backtrack
            _y = -_g0p/(2*(_g1-_g0-_g0p))
            _y2 = 1
            while _f(_xOld + _y*_step) > _f(_xOld) + alpha * _df(_xOld)@(_y*_step):
                #Backtrack more
                _gy = _f(_xOld + _y*_step)
                _gy2 = _f(_xOld + _y2*_step)
                _ab = (1/(_y-_y2))* np.array([[1/(_y**2),-1/(_y2**2)],
                    [-_y2/(_y**2) , _y/(_y2**2)]]) @ np.array(
                        [_gy - _g0p*_y - _g0,_gy2-_g0p*_y2-_g0])
                _a = _ab[0]
                _b = _ab[1]
                _y2 = _y
                _y = ( -_b + np.sqrt(_b**2 - 3*_a*_g0p))/(3*_a)
            _xNew = _xOld + _step*_y
        _cEnd = _f(_xNew)
        _entry1 = _cEnd
        _entry2 = _entry1 - _data[_k-1,0]
        _entry3 = _entry2/(_data[_k-1,1]**2)
        _entry4 = 2*(np.linalg.norm(_entry2)**2)
        _data = np.vstack((_data,[_entry1,_entry2,_entry3,_entry4]))
        if _cEnd <= precision:
            _cols=[r'$f_k$',r'$f_k-f_{k-1}$',r'$C=\frac{f_k-f_{k-1}}{(f_{k-1}-f_{k-2})^2}$',
                    r'$ |\epsilon _k | = C \cdot |f_k-f_{k-1}|^2$' ]
            _daf = pd.DataFrame(_data,columns=_cols)
            _daf.index.name='k'
            return _xNew,_daf
## ---------------------------------------
root,table = newton(f,f_prime,np.array([0,0]))
    
