#d𝑋(𝑡) + 𝐴𝑋(𝑡) d𝑡 + 𝑓(𝑋(𝑡)) d𝑡 = d𝑊(𝑡), 𝑡 ∈ (0, 𝑇 ]; 𝑋(0) = 𝑋0
#𝑓(𝑢)(𝑠) = 𝑃 (𝑢(𝑠))
#𝑃 (𝑟) = 𝑟3 − 𝛽2𝑟

from __future__ import print_function
from fenics import *
import numpy as np
from dolfin import * 
import random
import matplotlib.pyplot as plt


np.random.seed(1)
M = 5 # M^2 terms in the K-L expansion
Q_eigval = lambda i,j: 250/(float(i)*float(j))**3; # eigenvalues of Q

class QWienerProcess(UserExpression): # Q-Wiener process as an expression
	def __init__(self, randoms, **kwargs):
                super(QWienerProcess,self).__init__(**kwargs)
                self.randoms=randoms
	def eval(self, value, x): # evaluate K-L expansion
		v = 0
		for i in range(0,M):
			for j in range(0,M):
				v += 2*sqrt(dt)*sin((i+1)*np.pi*x[0])*sin((j+1)*np.pi*x[1])*self.randoms[i,j]*sqrt(Q_eigval(i+1,j+1))
				value[0]=v

        def value_shape(self):
                return()
        
g = 1  # Amplitude for noise.  g=0 no noise.  
s = np.random.normal(0, 1, (M,M))
dW = QWienerProcess(degree=2,randoms=s)

T = 10.0            # final time
num_steps = 100     # number of time steps
dt = T / num_steps 
lmbda  = 1.0e-02  
#dt     = 5.0e-06  
theta  = 1.0 
beta=0.2

nx=ny=20
mesh=UnitSquareMesh(nx,ny)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
ME = FunctionSpace(mesh, P1)
X=TrialFunction(ME)
Y=TrialFunction(ME)
Z=TrialFunction(ME)
v=TestFunction(ME)
q=TestFunction(ME)
w=TestFunction(ME)
def boundary(x, on_boundary):
    return on_boundary
X_D=Expression('1+t+x[0]*x[0]+pow(x[1],x[0])+pow(t,x[1]*x[0])',degree=2,t=0)
bc = DirichletBC(ME, X_D, boundary)

class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = 0.63 + 0.02*(0.5 - random.random())
        values[1] = 0.0
    def value_shape(self):
        return (2,)
temp=InitialConditions()
X_n=project(temp,ME)
#X_n=interpolate(temp, ME)
Y_n=interpolate(X_D,ME)
Z_n=interpolate(X_D,ME)



def f(u):
	return u**3-u*beta**2‘

g=Expression('x[0]*x[1]*x[0]-beta',beta=beta,degree=2)
aY=Y*v*dx+0.5*dt*dot(grad(Y),grad(v))*dx
LY=X_n*v*dx

"""
aZ=Z*q*dx+dt*(Z*2)*q*dx
LZ=Y_n*q*dx
"""
aX=X*w*dx+0.5*dt*dot(grad(X),grad(w))*dx
LX=Z_n*w*dx+g*w*dx
#+0*w*dx

X = Function(ME)
Y=  Function(ME)
Z= Function(ME)
FZ=Z*q*dx+dt*f(Z)*q*dx-Y_n*q*dx


t=0

for i in range(num_steps):
	t=+dt
	X_D.t = t
	dW.randoms =  np.random.normal(0, 1, (M,M))

	solve(aY==LY, Y,bc)
	Y_n.assign(Y)
#	solve(aZ==LZ,Z,bc)
	solve(FZ==0,Z,bc)
	Z_n.assign(Z)
	solve(aX==LX,X,bc)
	X_n.assign(X)

plot(X_n)
plt.savefig('allencahn1.png', bbox_inches='tight')
