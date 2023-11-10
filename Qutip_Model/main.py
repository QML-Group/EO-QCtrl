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

#### Run Optimziation ####

EC, F = CalculateOptimalFieldEnergeticCost(U_target_CNOT, H_Static_2, # Run algorithm with set of initial parameters
                                           H_Control_5, Iterations_GRAPE, 
                                           Timesteps, H_Labels_5, 
                                           weight_ec = 0.2, weight_fidelity= 0.8,
                                           Plot_Control_Field = True, Plot_Tomography = True) 

#### Plot Pareto Front ####

Weights = np.arange(1, step = 0.1)
EnergeticCost = []
Fidelity = []

for i in Weights:
    EnergeticCost_i_j, Fidelity_i_j = CalculateOptimalFieldEnergeticCost(U_target_CNOT, H_Static_2,
                                                                            H_Control_5, Iterations_GRAPE,
                                                                            Timesteps, H_Labels_5,
                                                                            weight_ec = i, weight_fidelity = 1-i,
                                                                            Plot_Control_Field = False, Plot_Tomography = False)
    EnergeticCost.append(EnergeticCost_i_j)
    Fidelity.append(Fidelity_i_j)

EnergeticCost_Normalized = EnergeticCost / max(EnergeticCost)
Error_Rate = np.ones(len(Fidelity)) - Fidelity

plt.plot(EnergeticCost_Normalized, Error_Rate, ls = '-', color = 'green', marker = 'd') 
plt.xlabel('Energetic Cost')
plt.ylabel('Infidelity')
plt.title('Energetic Cost versus Infidelity')
plt.grid()
plt.show()

#### Plot Fidelity & EC as function of Weights ####

#plt.plot(Weights, Error_Rate, label = 'Infidelity')
#plt.plot(Weights, EnergeticCost_Normalized, label = 'Normalized Energetic Cost')
#plt.xlabel('Weight Energetic Cost')
#plt.ylabel('Infidelity & Normalized Energetic Cost')
#plt.title('Infidelity and Normalized Energetic Cost as function of Energetic Cost Weight \n CNOT Gate, Timesteps = 500, GRAPE Iterations = 100')
#plt.grid()
#plt.legend()
#plt.show()



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
