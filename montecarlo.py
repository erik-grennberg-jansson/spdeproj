import numpy as np
from dolfin import *
import matplotlib.pyploy as plt




hs=(1/2,1/4,1/8,1/16,1/32,1/64)
n=6
dt=10**(-3)
real_h=0.5*10**(-2)
print(hs)
print(real_h)
N=10
seednums=np.random.randint(n*N)
result=np.zeros((6,1))

T=0.1
q=1

mesh=UnitSquareMesh(nx,ny)
P1=FiniteElement("Lagrange",mesh.ufl_cell(),degree=1)
V=FunctionSpace(mesh,P1)

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


def solve_spde(h,dt,seednum):
    #set here seed by seednum
    s=np.random.normal(0,1,(M,M))
    dW=QWienerProcess(degree= 




for i in range(0,6):
    this_sum=0
    for j in range(0,N):
        this_error=0.5 #change here
        this_sum+=this_error
    result[i]=this_sum/N
    
print(result)

