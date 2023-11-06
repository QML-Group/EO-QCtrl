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
steps = 50

ErrorRate, ControlPulses, FinalUnitary = Run_Optimizer(U_Target_CNOT, H_Static_1, H_Control_5, T, steps)

print("Fidelity is",  1 - ErrorRate)
print("Optimal Control Pulses are:", ControlPulses)
print("Final Unitary is:", FinalUnitary)





