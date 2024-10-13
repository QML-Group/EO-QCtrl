import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit.library import UGate, UnitaryGate
from qutip import *
import math

parts = 30
a1 = np.pi/2 /parts
a2 = 0
a3 = 0
U3 = UnitaryGate(UGate(a1,a2,a3))._params[0]
U3dg = UnitaryGate(UGate(-a1,-a2,-a3))._params[0]
U = Qobj(U3)
Udg = Qobj(U3dg)

Uz = Qobj(UnitaryGate(UGate(0,0,np.pi/8))._params[0])
Uinit = Qobj(UnitaryGate(UGate(np.pi/6,0,np.pi/8))._params[0])

fig = plt.figure()
b = Bloch(fig=fig)

pts_1 = 35      # How many applications of U. Target state gets determined by this.
# pts_2 = 86
pts_2 = int((2*np.pi - a1*pts_1)/a1) # U determines space between arrows. pts_1 determines how many arrows. This formula finds how many points are required for Udg to reach the point the other way round with same space-between-arrows.

b.vector_color = ["b"]*(pts_1+1) + ["g"]*(pts_1+1) + ["r"]*(pts_2+1) + ['#555555']*3
b.vector_alpha = list(np.linspace(0.1,0.9,pts_1+1)) + list(np.linspace(0.1,0.9,pts_1+1)) + list(np.linspace(0.1,0.5,pts_2+1))
b.vector_width = 1

b.point_color = ['#000000' ]*2
b.point_marker = ['o','^']
b.point_alpha = [1]*2
b.point_size = [35]*2

psi = Uinit*Qobj(np.array([1,0], dtype=complex).T)
b.add_states(Uz*psi,kind='point')
for i in range(0,pts_1+1):
    b.add_states(Uz*psi)
    psi = U*psi
b.add_states(Uz*(Udg*psi),kind='point')

ang = list(np.linspace(0,2*np.pi,pts_1+1))
psi = Uinit*Qobj(np.array([1,0], dtype=complex).T)
for i in range(0,pts_1+1):
    rz = Qobj(UnitaryGate(UGate(0,0,math.sin(ang[i])*np.pi/8))._params[0])
    b.add_states(rz*Uz*psi)
    psi = U*psi

psi = Uinit*Qobj(np.array([1,0], dtype=complex).T)
for i in range(0,pts_2+1):
    b.add_states(Uz*psi)
    psi = Udg*psi

vec = [[1,0,0],[0,1,0],[0,0,1]]
b.add_vectors(vec)

fig.set_size_inches(5,5.5)
b.render()
plt.show()