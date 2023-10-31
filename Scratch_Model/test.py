from input import *
from functions import *

"""
This is a test file to experiment with certain functions and features

-----------------

Work in Progress

"""

NumberOfTimesteps = 500
time = np.linspace(0, T, NumberOfTimesteps)
N = len(time)
K = len(H_Control_4)
u_start = np.zeros((K, N))

Unitary = Calculate_Unitary(H_Static_1, H_Control_4, u_start, NumberOfTimesteps, T)
print(Unitary)


