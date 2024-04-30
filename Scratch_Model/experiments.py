import numpy as np
import matplotlib.pyplot as plt
from functions import * 
from input import * 
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MultipleLocator

"""
This file is intended for experiments using "input.py" and "functions.py"

"""

# Plotting Control fields with varying weight sets


eps_e = 100
eps_f = 1
weights = [0, 0.2, 0.5, 0.8]
Target_Unitary = Generate_Rand_Unitary(2)
Control_Fields = np.zeros((len(h_c_1_qubit), Timesteps, len(weights)))
Timespace = np.linspace(0, T, Timesteps)

for weight_index, weight_value in enumerate(weights):

    Control_Fields[:, :, weight_index], Final_Unitary, Gradient_List, Fidelity, Energy = Run_GRAPE_Simulation(Target_Unitary, h_d_1_qubit, h_c_1_qubit, h_l_1_qubit,
                                                                                                        GRAPE_Iterations, Timesteps, T,
                                                                                                        1 - weight_value, weight_value,
                                                                                                        eps_f, eps_e, Return_Normalized = False)
    print(Fidelity)
    
fig, ax = plt.subplots(len(h_c_1_qubit), sharex=True)
xticks = [0, np.pi, 2 * np.pi]
colors = ['#03080c','#214868', '#5b97ca', '#9fc2e0']

for i in range(len(h_c_1_qubit)):
    for index_weight, value_weight in enumerate(weights):
        ax[i].plot(Timespace, Control_Fields[i, :, index_weight], label = f"$w_f$ = {round(1 - value_weight, 1)}, $w_e$ = {round(value_weight, 1)}", color = colors[index_weight])
        ax[i].set(xlabel = "Time", ylabel = f"{h_l_1_qubit[i]}")
plt.xticks(xticks, ['0', '$\pi$', '2 $\pi$'])

plt.legend()
plt.subplot_tool()
plt.tight_layout()
plt.show()


# Pareto Front Plots with varying eps_e/eps_f parameter sets

"""

eps_e_range = [100, 250, 500]
weights = np.arange(0, 1, 0.1)
eps_f_optimal = 1
Target_Unitary = Generate_Rand_Unitary(4)
EnergyArray = np.zeros((len(eps_e_range), len(weights)))
FidelityArray = np.zeros((len(eps_e_range), len(weights)))

for eps_index, eps_value in enumerate(eps_e_range):

    print(f"Progress: {((eps_index+1)/len(eps_e_range)) * 100} %")

    for weight_index, weight_value in enumerate(weights):

        Control_Pulses, Final_Unitary, Gradient_List, FidelityArray[eps_index, weight_index], EnergyArray[eps_index, weight_index] = Run_GRAPE_Simulation(Target_Unitary, H_Static_Ising,
                                                                                                                                                          H_Control_4, H_Labels_4,
                                                                                                                                                          GRAPE_Iterations, Timesteps,
                                                                                                                                                          T, 1 - weight_value, weight_value,
                                                                                                                                                          eps_f_optimal, eps_value, Return_Normalized = False)

InfidelityArray = np.ones((len(eps_e_range), len(weights))) - FidelityArray
NormalizedEnergyArray = EnergyArray / np.amax(EnergyArray)

for eps_index, eps_value in enumerate(eps_e_range):
    plt.plot(NormalizedEnergyArray[eps_index], InfidelityArray[eps_index], label = f"$\epsilon_f$ = 1, $\epsilon_e$ = {eps_value}", marker = 'd')

plt.xlabel("Normalized Energetic Cost")
plt.ylabel("Infidelity (1-F)")
plt.title(f"Pareto Front for different $\epsilon_e$")
plt.legend()
plt.grid()
plt.show()
"""
# Pareto Front Plots

"""

weights = np.arange(0, 1, 0.1)

number_of_simulations = 5

eps_e_optimal = 100

eps_f_optimal = 1

EnergyList = np.zeros((number_of_simulations, len(weights)))
FidelityList = np.zeros((number_of_simulations, len(weights)))

for sim in range(number_of_simulations):

    Target_Unitary = Generate_Rand_Unitary(4)

    for index, value in enumerate(weights):

        print(f"{index/len(weights) * 100} %")

        Control_Pulses, Final_Unitary, Gradient_List, FidelityList[sim, index], EnergyList[sim, index] = Run_GRAPE_Simulation(Target_Unitary, H_Static_Ising, 
                                                                                                                                H_Control_4, H_Labels_4,
                                                                                                                                GRAPE_Iterations, Timesteps,
                                                                                                                                T, 1 - value, value, eps_f_optimal, eps_e_optimal,
                                                                                                                                Return_Normalized = False)

InfidelityList = np.ones((number_of_simulations, len(weights))) - FidelityList
MaxEnergyList = EnergyList / np.amax(EnergyList)
InfidelityErrorBars = list()    
EnergyErrorBars = list()

for i in range(len(weights)):

    InfidelityErrorBar = 0.5 * (max(InfidelityList[:, i]) - min(InfidelityList[:, i]))
    EnergyErrorBar = 0.5 * (max(EnergyList[:, i]) - min(EnergyList[:, i]))
    InfidelityErrorBars.append(InfidelityErrorBar)
    EnergyErrorBars.append(EnergyErrorBar)

print(InfidelityErrorBars)
print(EnergyErrorBars)

plt.errorbar(MaxEnergyList[0], InfidelityList[0], xerr = EnergyErrorBars, yerr = InfidelityErrorBars, ls = '-', color = 'green', marker = 'd', label = 'Pareto Front')
plt.xlabel("Normalized Energetic Cost")
plt.ylabel("Infidelity (1-F)")
plt.title(f"Pareto Front using $\epsilon_e$ = {eps_e_optimal}, $\epsilon_f$ = {eps_f_optimal}")
plt.legend()
plt.grid()
plt.show()
"""
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

