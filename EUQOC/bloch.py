import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit.library import UGate, UnitaryGate
from qutip import *

parts = 50
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

pts_1 = 25
pts_2 = 176

b.vector_color = ["b"]*pts_1 + ["r"]*pts_2 + ['#555555']*3
b.vector_alpha = list(np.linspace(0.1,0.9,pts_1)) + list(np.linspace(0.1,0.5,pts_2))
b.vector_width = 1

b.point_color = ['#000000' ]*2
b.point_marker = ['o','^']
b.point_alpha = [1]*2
b.point_size = [35]*2

psi = Uinit*Qobj(np.array([1,0], dtype=complex).T)
b.add_states(Uz*psi,kind='point')
for i in range(0,pts_1):
    b.add_states(Uz*psi)
    psi = U*psi
b.add_states(Uz*psi,kind='point')

psi = Uinit*Qobj(np.array([1,0], dtype=complex).T)
for i in range(0,pts_2):
    b.add_states(Uz*psi)
    psi = Udg*psi

vec = [[1,0,0],[0,1,0],[0,0,1]]
b.add_vectors(vec)

fig.set_size_inches(5,5.5)
b.render()
plt.show()