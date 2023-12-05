import numpy as np
import matplotlib.pyplot as plt
import functions 
import input
import random as rd
import time

""" 

This main file uses 'input.py' and 'functions.py' to simulate optimal control problems 
using NumPy in Python 

-----------------

Work in Progress

"""

# Run GRAPE Optimization

T = 2 * np.pi
steps = 500
GRAPE_Iterations = 100

Control_Pulses, Final_Unitary, Gradient_List, Fidelity = functions.Calculate_Optimal_Control_Pulses(input.U_Target_CNOT, input.H_Static_1, input.H_Control_4, input.H_Labels_4,
                                                                                                  GRAPE_Iterations, steps, T, w_f = 1, w_e = 0,
                                                                                                  Plot_Control_Field = True, Plot_Tomography = False, Plot_du = True)

print(f"Fidelity method 1 is {Fidelity}")


# Run Nelder Mead Optimization

"""
start_time = time.time()

T = 2 * np.pi
steps = 50
TimeSpace = np.linspace(0, T, steps)

ErrorRate, ControlPulses, FinalUnitary = Run_Optimizer(U_Target_CNOT, H_Static_1, H_Control_5, T, steps, 'Nelder-Mead', Weight_F = 0.8 , Weight_EC = 0.2)

print("Total time is:", time.time() - start_time, "seconds")

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
"""

