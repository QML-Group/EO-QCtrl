import numpy as np
import matplotlib.pyplot as plt
from functions import * 
from input import * 


""" 

This main file uses 'input.py' and 'functions.py' to simulate optimal control problems 
using NumPy in Python 

-----------------

Work in Progress

"""

# Run GRAPE Optimization

Control_Pulses, Final_Unitary, Gradient_List, Fidelity = Run_GRAPE_Simulation(U_Target_CNOT, H_Static_Ising, H_Control_4, H_Labels_4,
                                                                                                  GRAPE_Iterations, Timesteps, T, w_f = 1, w_e = 0,
                                                                                                  eps_f = 2 * np.pi, eps_e = 2 * np.pi,
                                                                                                  Plot_Control_Field = True, Plot_Tomography = False, Plot_du = True)

print(f"Fidelity method 1 is {Fidelity}, Final Unitary is {Final_Unitary}")