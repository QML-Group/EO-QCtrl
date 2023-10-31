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
time = np.linspace(0, T, 20)
Cost, Opt_Params = Run_Optimizer(U_Target_CNOT, H_Static_1, H_Control_4, 1, T, 20)
print('Fidelity is', Cost)
plt.plot(time, Opt_Params[0])
plt.plot(time, Opt_Params[1])
plt.plot(time, Opt_Params[2])
plt.show()

