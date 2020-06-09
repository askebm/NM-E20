import sympy as sp

a1 = sp.Symbol('a1')
a2 =  sp.Symbol('a2')
q1 =  sp.Symbol('q1')
q2 =  sp.Symbol('q2')
x1 =  sp.Symbol('x1')
x2 =  sp.Symbol('x2')
dq1 =  sp.Symbol('dq1')
dq2 =  sp.Symbol('dq2')

f1 =  a1* sp.cos(q1+dq1)+a2*sp.cos(q2+dq2) -x1
f2 =  a1* sp.sin(q1+dq1)+a2*sp.cos(q2+dq2) -x2
f = lambda _dq: sp.Matrix([f1(_dq[0],_dq[1]),f2(_dq[0],_dq[1])],dtype=np.double)

F= sp.Matrix([f1,f2])
J = F.jacobian([dq1,dq2])
