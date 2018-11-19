"""
FEniCS tutorial demo program: Heat equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.
  u'= Laplace(u) + f  in the unit square
  u = u_D             on the boundary
  u = u_0             at t = 0
  u = 1 + x^2 + alpha*y^2 + \beta*t
  f = beta - 2 - 2*alpha
"""

from __future__ import print_function
from fenics import *
import numpy as np
from dolfin import * 
import random
import matplotlib.pyplot as plt

parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
np.random.seed()
T = 10.0            # final time
num_steps = 200     # number of time steps
dt = T / num_steps # time step size
alpha = 3          # parameter alpha
beta = 1.2         # parameter beta
q=1.0
# Create mesh and define function space
nx = ny = 8
mesh = UnitSquareMesh(nx, ny)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh,P1)

# Define boundary condition
#u_D =Constant(0);
u_D=Expression('1+x[0]*x[0]+alpha*x[1]*x[1]+beta*t',degree=2,alpha=alpha,beta=beta,t=0)
# Expression('1 + cos(x[0]*x[0]) + alpha*sin(x[1]) + beta*(t)',
#                 degree=2, alpha=alpha, beta=beta, t=0)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define initial value
u_n = interpolate(u_D, V)
#u_n = project(u_D, V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(beta - 2 - 2*alpha)
#f = Constant(0)


##

# K-L expansion
M = 5 # M^2 terms in the K-L expansion
Q_eigval = lambda i,j: 250/(float(i)*float(j))**3; # eigenvalues of Q
class QWienerProcess(UserExpression): # Q-Wiener process as an expression
  def __init__(self, randoms, degree):
    self.randoms = randoms # random numbers to be updated each time step
    self.degree = degree

  def eval(self, value, x): # evaluate K-L expansion
    v = 0
    for i in range(0,M):
      for j in range(0,M):
        v += 2*sqrt(dt)*cos((i+1)*np.pi*x[0])*cos((j+1)*np.pi*x[1])*self.randoms[i,j]*sqrt(Q_eigval(i+1,j+1))
        value[0]=v

g = 1  # Amplitude for noise.  g=0 no noise.  
s = np.random.normal(0, 1, (M,M))
dW = QWienerProcess(degree=2,randoms=s)

#F = 
a= u*v*dx + dt*dot(grad(u), grad(v))*dx
L=(u_n + dt*q*dW+dt*f)*v*dx
#L=dW

#a, L = lhs(F), rhs(F)


# Time-stepping
u = Function(V)
t = 0
for n in range(num_steps):

    # Update current time, fix solver? at the time it is very slow
    t += dt
    #u_D.t = t
    dW.randoms =  np.random.normal(0, 1, (M,M))
    # Compute solution
    solve(a == L, u, bc)

    # Plot solution

    # Compute error at vertices
    ##u_e = interpolate(u_D, V)
    #error = np.abs(u_e.vector().get_local()- u.vector().get_local()).max()
    #print('t = %.2f: error = %.3g' % (t, error))

    # Update previous solution
    u_n.assign(u)
	

# Hold plot
#cast u_n to a vector 
plot(u_n)

#ax = plt.subplot(111)
#im = ax.imshow(u_n)
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("rigth", size="5%", pad = 0.05)

#plt.colorbar(im, cax = cax)
plt.savefig('brusGrovMconstq=1.png', bbox_inches='tight')
