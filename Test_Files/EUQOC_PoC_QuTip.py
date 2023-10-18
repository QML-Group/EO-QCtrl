import matplotlib.pyplot as plt
import time
import numpy as np
from qutip import *
from qutip.control import *
from qutip.qip.operations import cnot
from qutip.qip.operations import cphase
from qutip.control.grape import plot_grape_control_fields, _overlap, grape_unitary_adaptive, cy_grape_unitary
from scipy.interpolate import interp1d
from qutip.ui.progressbar import TextProgressBar

# Define total time 

T = 2 * np.pi # Total evolution time
times = np.linspace(0, T, 500) # Total time array in 500 timesteps

# Define target unitary and Hamiltonian

U_target = cnot()

R = 10

H_ops = [tensor(sigmax(), identity(2)),
         tensor(sigmay(), identity(2)),
         tensor(sigmaz(), identity(2)),
         tensor(identity(2), sigmax()),
         tensor(identity(2), sigmay()),
         tensor(identity(2), sigmaz()),
         tensor(sigmax(), sigmax()) +
         tensor(sigmay(), sigmay()) +
         tensor(sigmaz(), sigmaz())]

H_labels = [r'$u_{1x}$', r'$u_{1y}$', r'$u_{1z}$',
            r'$u_{2x}$', r'$u_{2y}$', r'$u_{2z}$',
            r'$u_{xx}$',
            r'$u_{yy}$',
            r'$u_{zz}$'
            ]

H0 = 0 * np.pi * (tensor(sigmax(), identity(2))) + tensor(identity(2), sigmax())

c_ops = []

# Implement GRAPE algorithm

u0 = np.array([np.random.rand(len(times)) * 2 * np.pi * 0.05 for _ in range(len(H_ops))])

u0 = [np.convolve(np.ones(10)/10, u0[idx, :], mode = 'same') for idx in range(len(H_ops))]

u_limits = None 

alpha = None

result = cy_grape_unitary(U_target, H0, H_ops, R, times, u_limits=u_limits, u_start = u0 , eps=2*np.pi*1, alpha=alpha, phase_sensitive=False, progress_bar=TextProgressBar())

# Plot Control Fields 

plot_grape_control_fields(times, result.u / (2 * np.pi), H_labels, uniform_axes=True)
plt.show()

# Calculate Fidelity 

print(_overlap(U_target, result.U_f).real, abs(_overlap(U_target, result.U_f)) ** 2)

test_fidelity = process_fidelity(U_target, result.U_f)
print("Process Fidelity is:", test_fidelity)

# Calculate and Plot Energetic Cost
# Investigate what is the final R (best Fidelity)
norm = np.linalg.norm(H0) + result.u[0,0]*np.linalg.norm(H_ops[0]) + result.u[0,1]*np.linalg.norm(H_ops[1]) + result.u[0,2]*np.linalg.norm(H_ops[2]) + result.u[0,3]*np.linalg.norm(H_ops[3]) + result.u[0,4]*np.linalg.norm(H_ops[4]) + result.u[0,5]*np.linalg.norm(H_ops[5]) + result.u[0,6]*np.linalg.norm(H_ops[6])

energetic_cost = 0
list = []
timestep = max(times)/len(times)

for i in range(len(times)):
    energetic_cost += norm[i]
    list.append(energetic_cost*timestep)


plt.plot(times, list)
plt.xlabel('Time t')
plt.ylabel('Energetic Cost (a.u.)')
plt.title('Energetic Cost over Time')
plt.show()

print(result.u[-1])
print(result.U_f)