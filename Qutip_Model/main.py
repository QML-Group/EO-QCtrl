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

EC, F = CalculateOptimalFieldEnergeticCost(U_target_CNOT, H_Static_2, # Run algorithm with set of initial parameters
                                           H_Control_5, Iterations_GRAPE, 
                                           Timesteps, H_Labels_5, 
                                           weight_ec = 0.8, weight_fidelity= 1.0,
                                           Plot_Control_Field = True, Plot_Tomography = False) 

Output = f"""  

**** PROGRAM OUTPUT & RESULTS ****

----------

    Random Quantum Unitary: 

    {U_target_rand}

    ----------

    Optimal Fidelity: {F}

    ----------

    Energetic Cost: {EC}

    ----------

    Number of GRAPE Iterations: {Iterations_GRAPE}

    ----------

    Number of Timesteps: {Timesteps}
"""

print(Output) #Print Output
