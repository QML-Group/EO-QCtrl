import numpy as np 
from qutip import basis, fidelity, identity, sigmax, sigmaz, tensor, destroy
from qutip_qip.operations import expand_operator, toffoli, snot
import functions as fc
from qutip.metrics import fidelity
from simulator import QuantumEnvironment

# Input Parameters

H_Drift_Qutip = np.pi * (tensor(sigmaz(), identity(2)) + tensor(identity(2), sigmaz())) + (1/2) * np.pi * tensor(sigmaz(), sigmaz()) # Define Drift Hamiltonian used in "Processor"
H_Drift_Scratch = np.pi * (fc.tensor(fc.sigmaz(), fc.identity(2)) + fc.tensor(fc.identity(2), fc.sigmaz())) + (1/2) * np.pi * fc.tensor(fc.sigmaz(), fc.sigmaz()) # Define Drift Hamiltonian used for optimization
H_Control_Qutip = [tensor(sigmax(), identity(2)), # Define Control Hamiltonian used in "Processor"
                   tensor(identity(2), sigmax()),
                   tensor(sigmax(), sigmax())]
H_Control_Scratch = [fc.tensor(fc.sigmax(), fc.identity(2)), # Define Control Hamiltonian used for optimization
                   fc.tensor(fc.identity(2), fc.sigmax()),
                   fc.tensor(fc.sigmax(), fc.sigmax())]
H_Labels = [r'$u_{1x}$', # Labels for H_Control_4 (optional for plotting)
              r'$u_{2x}$', 
              r'$u_{xx}$'] 
TargetUnitary = fc.cnot()
N_q = 2
T = 2 * np.pi
T1 = 100 * T
T2 = 100 * T
Nt = 500
Ng = 500
time = np.linspace(0, T, Nt)
Initial_State = basis(4, 2)
weight_fidelity = 1
weight_energy = 0
epsilon_f = 1
epsilon_e = 100

# Test Quantum Environment Class

Environment = QuantumEnvironment(N_q, H_Drift_Qutip, H_Control_Qutip, H_Labels, T1, T2, Initial_State, TargetUnitary, Nt, T, Ng) # Create instance of Quantum Environment

pulses = Environment.run_grape_optimization(weight_fidelity, weight_energy, epsilon_f, epsilon_e) # Calculate pulses by EO-GRAPE algorithm

energy = Environment.calculate_energetic_cost(pulses) # Calculate energetic cost of pulses

result = Environment.run_pulses(pulses, plot_pulses = True) # Run the EO-GRAPE pulses on the environment

reward = Environment.calculate_fidelity_reward(result, plot_result = True) # Calculate the Fidelity reward of this set of pulses

Environment.plot_grape_pulses(pulses) # Plot the EO-GRAPE generated pulses

Environment.plot_tomography() # Plot Tomography of the final unitary and target unitary 

Environment.plot_du() # Plot gradient versus GRAPE iterations

Environment.plot_cost_function() # Plot cost function versus GRAPE iterations

print(f"Energetic cost is: {energy}, Fidelity reward is: {reward}") # Print energetic cost and fidelity reward of this set of pulses