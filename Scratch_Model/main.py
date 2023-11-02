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
#time = np.linspace(0, T, 20)
#Cost, Opt_Params = Run_Optimizer(U_Target_CNOT, H_Static_1, H_Control_2, 1, T, 20)
#print("Fidelity is", 1-Cost)
#print(Opt_Params)


u = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
             [0.92431986,  1.34042781,  1.39771818,  1.07451561,  0.49360382, -0.13070387, -0.57642126, -0.69735338, -0.47786727, -0.03671082,  0.42137133,  0.68589618,  0.62316304,  0.22917477, -0.36866862, -0.95946176, -1.32823794, -1.3409839,-0.99965316, -0.43906062,  0.12988338,  0.50206993,  0.55527877,  0.29572316, -0.14621808, -0.56786282 ,-0.77326215, -0.65047193, -0.21617879,  0.39056099, 0.96036492,  1.29224169,  1.27292314,  0.92294126, 0.3859671,  -0.13076319, -0.43665856, -0.42959017, -0.13391574,  0.31032098,  0.70225191,  0.85848553, 0.68597693 , 0.21936481, -0.39232251, -0.9414146 , -1.2396323 , -1.19222074, -0.83607751 ,-0.32256118 ,0.14449679 , 0.3875739 ,  0.32266669 ,-0.00928695, -0.45902696 ,-0.82679535 ,-0.94036039 ,-0.72401903 ,-0.2291548 ,  0.3856759 ,0.91406904,  1.17930359,  1.10373176 ,0.73963789, 0.24604763, -0.17565578, -0.35948717, -0.23807197,  0.1318612 ,  0.59152506,  0.94103084,  1.01768375, 0.76169199 , 0.24041438, -0.37789049 ,-0.88699788, -1.12011136, -1.01516123, -0.63913427 ,-0.15928299 ,0.22384792 , 0.35376022 , 0.17797794 ,-0.23168869, -0.70628691, -1.04407019 ,-1.08974861 ,-0.79767918, -0.2504193 ,  0.3735707 ,0.86660477,  1.06959543 , 0.93415399 , 0.54127226 , 0.06730326 ,-0.28593349 ,-0.36887382 ,-0.14188902 , 0.3088994 ,  0.3088994],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]])


U_F = Calculate_Unitary(H_Static_1, H_Control_4, u, 100, T)

print(U_F)
print(U_Target_CNOT)

print("Fidelity is", Calculate_Fidelity(U_Target_CNOT, U_F))

# test = Calculate_Cost_Function(u, 1, 0, U_Target_CNOT, H_Static_1, H_Control_4, 100, T)
# print(test)



