import numpy as np
import matplotlib.pyplot as plt
from functions import *
from input import *
import random as rd

""" 

This main file uses 'input.py' and 'functions.py' to simulate optimal control problems 
using NumPy in Python 

-----------------

Work in Progress

"""

# Run Optimization Code

T = 2 * np.pi
steps = 100
TimeSpace = np.linspace(0, T, steps)

ErrorRate, ControlPulses, FinalUnitary = Run_Optimizer(U_Target_CNOT, H_Static_1, H_Control_5, T, steps, 'Nelder-Mead')

print("Fidelity is",  1 - ErrorRate)
print("Optimal Control Pulses are:", ControlPulses)
print("Final Unitary is:", FinalUnitary)

Control_Matrix = np.zeros((len(H_Control_5), steps))

for i in range(len(H_Control_5)):
    for j in range(steps):
        Control_Matrix[i, j] = ControlPulses[j * len(H_Control_5) + i]

fig, ax = plt.subplots(len(H_Control_5))

for i in range(len(H_Control_5)):
    ax[i].plot(TimeSpace, Control_Matrix[i], label = f'{H_Labels_5[i]}')
    ax[i].set(xlabel = 'Time', ylabel = f'{H_Labels_5[i]}')

plt.subplot_tool()
plt.show()
