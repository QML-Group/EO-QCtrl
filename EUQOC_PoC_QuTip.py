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
times = np.linspace(0, T, 10) # Total time array in 500 timesteps

# Define target unitary and Hamiltonian

U_target = cnot()

R = 500

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

result = cy_grape_unitary(U_target, H0, H_ops, R, times, u_start=u0, u_limits=u_limits, eps=2*np.pi*1, alpha=alpha, phase_sensitive=False, progress_bar=TextProgressBar())

# Plot Control Fields 

plot_grape_control_fields(times, result.u / (2 * np.pi), H_labels, uniform_axes=True)
plt.show()

# Calculate Fidelity 

print(_overlap(U_target, result.U_f).real, abs(_overlap(U_target, result.U_f)) ** 2)

