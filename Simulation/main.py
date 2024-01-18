import numpy as np 
from qutip import basis, fidelity, identity, sigmax, sigmaz, tensor, destroy
from qutip_qip.operations import expand_operator, toffoli, snot
import functions as fc
from qutip.metrics import fidelity
from simulator import QuantumEnvironment
import tensorflow as tf 
from scipy.optimize import minimize 


# Input Parameters

h_d = np.pi * (tensor(sigmaz(), identity(2)) + tensor(identity(2), sigmaz())) + (1/2) * np.pi * tensor(sigmaz(), sigmaz()) # Define Drift Hamiltonian used in "Processor"
h_c = [tensor(identity(2), sigmax())]
h_l = [r'$u_{1x}$', r'$u_{2x}$', r'$u_{xx}$'] 
target_unitary = fc.cnot()
number_qubits = 2
gate_duration = 2 * np.pi
t1 = 100 * gate_duration
t2 = 100 * gate_duration
number_of_timesteps = 10
number_of_grape_iterations = 1000
initial_state = basis(4, 2)
weight_fidelity = 1
weight_energy = 0
epsilon_f = 1
epsilon_e = 100

# Test Quantum Environment Class

Environment = QuantumEnvironment(number_qubits, h_d, h_c, h_l, t1, t2, initial_state, target_unitary, number_of_timesteps, gate_duration, number_of_grape_iterations) # Create instance of Quantum Environment

pulses = Environment.run_grape_optimization(weight_fidelity, weight_energy, epsilon_f, epsilon_e) # Calculate pulses by EO-GRAPE algorithm

energy = Environment.calculate_energetic_cost(pulses) # Calculate energetic cost of pulses

result = Environment.run_pulses(pulses, plot_pulses = True) # Run the EO-GRAPE pulses on the environment

reward = Environment.calculate_fidelity_reward(result, plot_result = True) # Calculate the Fidelity reward of this set of pulses

#Environment.plot_grape_pulses(pulses) # Plot the EO-GRAPE generated pulses

#Environment.plot_tomography() # Plot Tomography of the final unitary and target unitary 

#Environment.plot_du() # Plot gradient versus GRAPE iterations

#Environment.plot_cost_function() # Plot cost function versus GRAPE iterations

print(f"Fidelity reward is: {reward}") # Print energetic cost and fidelity reward of this set of pulses