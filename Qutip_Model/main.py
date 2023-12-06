import matplotlib.pyplot as plt
import numpy as np
from qutip import *
from qutip.control import *
from input import *
from functions import *

""" 

This main file uses 'input.py' and 'functions.py' to simulate optimal control problems 
using built in functions from QuTip 


"""

U_Final, EC, F, du_list = CalculateOptimalFieldEnergeticCost(U_target_CNOT, H_Static_Ising, 
                                            H_Control_4, Iterations_GRAPE, 
                                            Timesteps, T, H_Labels_4, 
                                            weight_ec = 0, weight_fidelity = 1,
                                            Plot_Control_Field = False, Plot_Tomography = False, Plot_du = False)

print(f"Final Unitary is: {U_Final},
      Energetic Cost is: {EC},
      Fidelity is: {F}")

