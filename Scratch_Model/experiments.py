import numpy as np
import matplotlib.pyplot as plt
from functions import * 
from input import * 
from matplotlib import cm
from matplotlib.ticker import LinearLocator

"""
This file is intended for experiments using "input.py" and "functions.py"

"""

# Pareto Front Plots

weights = np.arange(0, 1, 0.1)

eps_e_optimal = 100

eps_f_optimal = 0.1

Target_Unitary = Generate_Rand_Unitary(4)

EnergyList = list()

InfidelityList = list()

for index, value in enumerate(weights):

    print(f"{index/len(weights) * 100} \%")

    Control_Pulses, Final_Unitary, Gradient_List, Fidelity, Energy = Run_GRAPE_Simulation(Target_Unitary, H_Static_Ising, 
                                                                                          H_Control_4, H_Labels_4,
                                                                                          GRAPE_Iterations, Timesteps,
                                                                                          T, 1 - value, value, eps_f_optimal, eps_e_optimal,
                                                                                          Return_Normalized = False)
    EnergyList.append(Energy)
    InfidelityList.append(1- Fidelity)

MaxEnergyList = EnergyList / max(EnergyList)

plt.plot(MaxEnergyList, InfidelityList, ls = '-', color = 'red', marker = 'd', label = 'Pareto Front')
plt.xlabel("Normalized Energetic Cost by Max Energy")
plt.ylabel("Infidelity")
plt.title(f"Pareto Front Using $\epsilon_e$ = {eps_e_optimal}, $\epsilon_f$ = {eps_f_optimal}")
plt.legend()
plt.grid()
plt.show()







# Cost Function versus eps_e and eps_f experiment
"""

Target_Unitary = Generate_Rand_Unitary(4)

eps_e = [0.1, 1, 10, 100, 1000] 
eps_f = [0.001, 0.01, 0.1, 1, 10] 

xv, yv = np.meshgrid(eps_e, eps_f)

EnergyArray = np.zeros((len(eps_e), len(eps_f)))
FidelityArray = np.zeros((len(eps_e), len(eps_f)))

for index_eps_e, value_eps_e in enumerate(eps_e):

    for index_eps_f, value_eps_f in enumerate(eps_f):

        Control_Pulses, Final_Unitary, Gradient_List, FidelityArray[index_eps_e, index_eps_f], EnergyArray[index_eps_e, index_eps_f] = Run_GRAPE_Simulation(Target_Unitary, H_Static_Ising, H_Control_4,
                                                                                                                                                            H_Labels_4, GRAPE_Iterations, Timesteps, T, w_f = 0.5, w_e = 0.5,
                                                                                                                                                            eps_f = value_eps_f, eps_e = value_eps_e)
        
InfidelityArray = np.ones((len(eps_e), len(eps_f))) - FidelityArray

CostFunctionArray = 0.5 * InfidelityArray + 0.5 * EnergyArray


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

surface_cost_fn = ax.plot_surface(xv, yv, CostFunctionArray, cmap = cm.coolwarm, linewidth = 0, antialiased = False, label = "Cost Function")

ax.set_xlabel("$\epsilon_e$")
ax.set_ylabel("$\epsilon_f$")
ax.set_zlabel("Cost Function")
fig.colorbar(surface_cost_fn, shrink=0.5, aspect=5)


plt.show()
"""