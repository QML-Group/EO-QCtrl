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
#%%
#### Run Optimziation ####
RandomUnitary = rand_unitary(4)

EC, F = CalculateOptimalFieldEnergeticCost(RandomUnitary, H_Static_2, # Run algorithm with set of initial parameters
                                           H_Control_4, Iterations_GRAPE, 
                                           Timesteps, H_Labels_4, 
                                           weight_ec = 0.5, weight_fidelity= 0.5,
                                           Plot_Control_Field = True, Plot_Tomography = True, Plot_du = True) 
print(f"EC is {EC}, Fidelity is {F}")






#%%
#### Plot Pareto Front ####
"""

RandomUnitary = rand_unitary(4)
Weights = np.arange(1.1, step = 0.1)


EnergeticCost = []
Fidelity = []

for i in Weights:
    EnergeticCost_i_j, Fidelity_i_j = CalculateOptimalFieldEnergeticCost(RandomUnitary, H_Static_2,
                                                                            H_Control_4, Iterations_GRAPE,
                                                                            Timesteps, H_Labels_4,
                                                                            weight_ec = i, weight_fidelity = 1-i, Use_Rand_u0 = False, 
                                                                            Plot_Control_Field = False, Plot_Tomography = False, Plot_du = True)
    EnergeticCost.append(EnergeticCost_i_j)
    Fidelity.append(Fidelity_i_j)
    print("Weight EC is", i ,"Energetic Cost is:", EnergeticCost_i_j, " Fidelity is:", Fidelity_i_j)

EnergeticCost_Normalized = EnergeticCost / max(EnergeticCost)
Error_Rate = np.ones(len(Fidelity)) - Fidelity
print(EnergeticCost_Normalized)
print(Error_Rate)

plt.plot(EnergeticCost_Normalized, Error_Rate, ls = '-', color = 'green', marker = 'd', label = 'Pareto Front') 
plt.xlabel('Energetic Cost')
plt.ylabel('Infidelity')
plt.title('Energetic Cost versus Infidelity')
plt.legend()
plt.grid()
plt.show()
"""

#rand_fidelities = []
#rand_energies = [] 
#num_points = 30

#for i in range(num_points):
#    rand_weight_ec = np.random.uniform(0, 1)
#    rand_energy, rand_fidelity = CalculateOptimalFieldEnergeticCost(RandomUnitary, H_Static_2,
#                                                                    H_Control_5, 5,
#                                                                    Timesteps, H_Labels_5,
#                                                                    weight_ec = rand_weight_ec, weight_fidelity = 1 - rand_weight_ec, Use_Rand_u0 = True, 
#                                                                    Plot_Control_Field = False, Plot_Tomography = False)
#    rand_fidelities.append(rand_fidelity)
#    rand_energies.append(rand_energy)

#print(rand_energies)
#print(rand_fidelities)

#rand_energies_normalized = rand_energies / max(EnergeticCost)
#rand_errorrate = np.ones(len(rand_fidelities)) - rand_fidelities

#print(rand_energies_normalized, rand_errorrate)

#plt.plot(EnergeticCost_Normalized, Error_Rate, ls = '-', color = 'green', marker = 'd', label = 'Pareto Front') 
#plt.scatter(rand_energies_normalized, rand_errorrate, marker = 'd', label = 'Random Control Parameters')
#plt.xlabel('Energetic Cost')
#plt.ylabel('Infidelity')
#plt.title('Energetic Cost versus Infidelity')
#plt.legend()
#plt.grid()
#plt.show()

#%%
#### Plot Fidelity & EC as function of Weights ####

#plt.plot(Weights, Error_Rate, label = 'Infidelity')
#plt.plot(Weights, EnergeticCost_Normalized, label = 'Normalized Energetic Cost')
#plt.xlabel('Weight Energetic Cost')
#plt.ylabel('Infidelity & Normalized Energetic Cost')
#plt.title('Infidelity and Normalized Energetic Cost as function of Energetic Cost Weight \n CNOT Gate, Timesteps = 500, GRAPE Iterations = 100')
#plt.grid()
#plt.legend()
#plt.show()



#Output = f"""  
#
#**** PROGRAM OUTPUT & RESULTS ****
#
#----------
#
#    Random Quantum Unitary: 
#
#    {U_target_rand}
#
#    ----------
#
#    Optimal Fidelity: {F}
#
#    ----------
#
#    Energetic Cost: {EC}
#
#    ----------
#
#    Number of GRAPE Iterations: {Iterations_GRAPE}
#
#    ----------
#
#    Number of Timesteps: {Timesteps}
#"""
#
#print(Output) #Print Output
