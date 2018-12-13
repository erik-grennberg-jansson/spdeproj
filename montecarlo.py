import numpy as np
from dolfin import *
import matplotlib.pyplot as plt
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}
hs=(1/2,1/4,1/8,1/16,1/32,1/64)
n=6
dt=10**(-3)
real_h=10**(-2)
print(hs)
print(real_h)
N=50
seednums=np.random.randint(n*N**4,size=n*N)
result=np.zeros((6,1))
set_log_level(40)
T=0.1
M = 5 # M^2 terms in the K-L expansion
Q_eigval = lambda i,j: 250/(float(i)*float(j))**3; # eigenvalues of Q

class QWienerProcess(UserExpression): # Q-Wiener process as an expression
    def __init__(self, randoms, **kwargs):
        # Invoke the superclass constructor to properly set up the UserExpression
        super(QWienerProcess, self).__init__(**kwargs)
        self.randoms = randoms # random numbers to be updated each time step
    def eval(self, value, x): # evaluate K-L expansion
        v = 0
        for i in range(0,M):
            for j in range(0,M):
                v += 2*sqrt(dt)*sin((i+1)*np.pi*x[0])*sin((j+1)*np.pi*x[1])\
                     *self.randoms[i,j]*sqrt(Q_eigval(i+1,j+1))
                value[0]=v
    def value_shape(self):
        return ()
def boundary(x,on_boundary):
    return on_boundary



for i in range(0,6):
    this_sum=0
    mesh1=UnitSquareMesh(int(1/hs[i]),int(1/hs[i]))
    mesh2=UnitSquareMesh(int(1/real_h),int(1/real_h))
    P1=FiniteElement("Lagrange",mesh1.ufl_cell(),degree=1)
    P2=FiniteElement("Lagrange",mesh2.ufl_cell(),degree=1)
    V1=FunctionSpace(mesh,P1)
    V2=FunctionSpace(mesh,P2)
    num_steps=int(T/dt)
    u_n=interpolate(Expression('20',degree=0),V1)
    ue_n=interpolate(Expression('20',degree=0),V2)
    bc1=DirichletBC(V1,Constant(0),boundary)
    bc2=DirichletBC(V2,Constant(0),boundary)
    u1=TrialFunction(V1)
    v1=TestFunction(V1)
    u2=TrialFunction(V2)
    v2=TestFunction(V2)
       
    for j in range(0,N):
        print("Repetition number %.2d:"%j)
        print(seednums[i+j])
        np.random.seed(seednum)
        s=np.random.normal(0,1,(M,M))
        dW=QWienerProcess(degree=2,randoms=s)
        F1=u1*v1*dx+dt*dot(grad(u1),grad(v1))*dx-(u_n+dt*dW)*v1*dx
        F2=u2*v2*dx+dt*dot(grad(u2),grad(v2))*dx-(ue_n0dt*dW)*v2*dx
        a1,L1=lhs(F),rhs(F)
        a2,L2=lhs(F),rhs(F)
        u1=Function(V1)
        u2=Function(V2)
        t=0
        for n in range(num_steps):
            t+=dt
            dW.randoms =  np.random.normal(0, 1, (M,M))
            solve(a==L,u1,bc)
            solve(a==L,u2,bc)
            u_n.assign(u1)
            ue_n.assign(u2)
        

        this_error=errornorm(ue_n,u_n)
        this_sum+=this_error
        print(this_sum)
    result[i]=this_sum/N
    
print(result)
print(seednums)
print(errornorm(solve_spde(1/16,0.01,1),solve_spde(1/8,0.01,2)))

