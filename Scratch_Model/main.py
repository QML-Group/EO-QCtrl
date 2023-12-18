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

Control_Pulses, Final_Unitary, Gradient_List, Fidelity, Energy = Run_GRAPE_Simulation(U_Target_Identity, H_Static_Ising, H_Control_5, H_Labels_5,
                                                                                        GRAPE_Iterations, Timesteps, T, w_f = 1, w_e = 0,
                                                                                        eps_f = eps_f, eps_e = eps_e, Return_Normalized = True,
                                                                                        Plot_Control_Field = True, Plot_Tomography = True, Plot_du = False, Plot_Cost_Function = True)

print(f"Fidelity method is {Fidelity}, Energy is {Energy}")