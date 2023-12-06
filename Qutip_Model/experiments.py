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

#### Plot Gradient with Varying Weights ####

"""
RandomUnitary = rand_unitary(4)
Weights = [0, 0.2, 0.5, 0.8]
average_du_list = np.zeros((len(Weights), Iterations_GRAPE-1))

for index, value in enumerate(Weights):
    EC, F, du_list = CalculateOptimalFieldEnergeticCost(RandomUnitary, H_Static_2, # Run algorithm with set of initial parameters
                                            H_Control_4, Iterations_GRAPE, 
                                            Timesteps, H_Labels_4, 
                                            weight_ec = value, weight_fidelity = 1 - value,
                                            Plot_Control_Field = False, Plot_Tomography = False, Plot_du = False) 
    print(f"EC is {EC}, Fidelity is {F}")

    for i in range(Iterations_GRAPE-1):
        average_du_list[index, i] = np.average(du_list[i])
    
iteration_space = np.linspace(1, Iterations_GRAPE - 1, Iterations_GRAPE - 1)

for index, value in enumerate(Weights):
    plt.plot(iteration_space, average_du_list[index], label = f"$w_E$ = {np.round(value, 1)}, $w_F$ = {np.round(1-value, 1)}")
plt.axhline(y = 0, color = 'black', linestyle = '-')
plt.xlabel("GRAPE Iteration Number")
plt.ylabel("Average Maximum Gradient per Control Line")
plt.title("Averaged Maximum Gradient per Control Line versus GRAPE Iteration Number \n $\epsilon_F$ = 2 $\pi$, $\epsilon_E$ = $20 \pi$")
plt.legend()
plt.grid()
plt.show()
"""

#%%

#### Plot Pareto Front ####

"""
RandomUnitary = rand_unitary(4)
Weights = np.arange(1.1, step = 0.1)
du_var = 0

EnergeticCost = []
Fidelity = []

for i in Weights:
    EnergeticCost_i_j, Fidelity_i_j, du_var = CalculateOptimalFieldEnergeticCost(RandomUnitary, H_Static_2,
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

#%%

#### Plot Pareto Front with Random Points ####

"""
rand_fidelities = []
rand_energies = [] 
num_points = 30

for i in range(num_points):
   rand_weight_ec = np.random.uniform(0, 1)
   rand_energy, rand_fidelity = CalculateOptimalFieldEnergeticCost(RandomUnitary, H_Static_2,
                                                                   H_Control_5, 5,
                                                                   Timesteps, H_Labels_5,
                                                                   weight_ec = rand_weight_ec, weight_fidelity = 1 - rand_weight_ec, Use_Rand_u0 = True, 
                                                                   Plot_Control_Field = False, Plot_Tomography = False)
   rand_fidelities.append(rand_fidelity)
   rand_energies.append(rand_energy)

print(rand_energies)
print(rand_fidelities)

rand_energies_normalized = rand_energies / max(EnergeticCost)
rand_errorrate = np.ones(len(rand_fidelities)) - rand_fidelities

print(rand_energies_normalized, rand_errorrate)

plt.plot(EnergeticCost_Normalized, Error_Rate, ls = '-', color = 'green', marker = 'd', label = 'Pareto Front') 
plt.scatter(rand_energies_normalized, rand_errorrate, marker = 'd', label = 'Random Control Parameters')
plt.xlabel('Energetic Cost')
plt.ylabel('Infidelity')
plt.title('Energetic Cost versus Infidelity')
plt.legend()
plt.grid()
plt.show()
"""

#%%
#### Plot Fidelity & EC as function of Weights ####

"""
plt.plot(Weights, Error_Rate, label = 'Infidelity')
plt.plot(Weights, EnergeticCost_Normalized, label = 'Normalized Energetic Cost')
plt.xlabel('Weight Energetic Cost')
plt.ylabel('Infidelity & Normalized Energetic Cost')
plt.title('Infidelity and Normalized Energetic Cost as function of Energetic Cost Weight \n CNOT Gate, Timesteps = 500, GRAPE Iterations = 100')
plt.grid()
plt.legend()
plt.show()
"""
