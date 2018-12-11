from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

T=0.1
dt=10**(-3)
num_steps=int(T/dt)
nx=ny=16
q=10.0
mesh=UnitSquareMesh(nx,ny)
P1=FiniteElement("Lagrange",mesh.ufl_cell(),degree=1)
V=FunctionSpace(mesh,P1)

print(dt)
# K-L expansion

M = 5 # M^2 terms in the K-L expansion
Q_eigval = lambda i,j: 1/(float(i)*float(j))**0; # eigenvalues of Q

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

s = np.random.normal(0, 1, (M,M))
dW = QWienerProcess(degree=2,randoms=s)
dx=Measure('dx',domain=mesh)

def boundary(x,on_boundary):
    return on_boundary
bc=DirichletBC(V,Constant(0),boundary)

u_0=Expression('exp(-x[0]*x[0]-x[1]*x[1])',degree=2)

u_n=interpolate(u_0,V)

u=TrialFunction(V)
v=TestFunction(V)
f=Constant(0)

F=u*v*dx+dt*dot(grad(u),grad(v))*dx-(u_n+dt*f+dt*dW*q)*v*dx
a,L=lhs(F),rhs(F)

vtkfile=File('stoch_gaussian/solution.pvd')

u=Function(V)
t=0
for n in range(num_steps):
    t+=dt
    dW.randoms =  np.random.normal(0, 1, (M,M))
    solve(a==L,u,bc)
    print('t=%.2f:'%t)
    vtkfile<<(u,t)
    u_n.assign(u)



