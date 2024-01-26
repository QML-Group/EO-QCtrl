import numpy as np 
from qutip import basis, fidelity, identity, sigmax, sigmaz, tensor, destroy
from qutip_qip.operations import expand_operator, toffoli, snot
import functions as fc
from qutip.metrics import fidelity
from simulator import QuantumEnvironment
from QuantumRLagent import QuantumRLAgent
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
number_of_timesteps = 100
number_of_grape_iterations = 500
initial_state = basis(4, 2)
weight_fidelity = 1
weight_energy = 0
epsilon_f = 1
epsilon_e = 100
n_cycles = 1
num_episodes = 500

# Test Quantum RL Agent Class

TrainingEnvironment = QuantumEnvironment(number_qubits, h_d, h_c, h_l, t1, t2, initial_state, target_unitary, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles)

EvaluationEnvironment = QuantumEnvironment(number_qubits, h_d, h_c, h_l, t1, t2, initial_state, target_unitary, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles)

RLAgent = QuantumRLAgent(TrainingEnvironment, EvaluationEnvironment, num_episodes)

RLAgent.run_training()

RLAgent.plot_fidelity_per_iteration()

RLAgent.plot_final_pulse()

_, fidelity = EvaluationEnvironment.calculate_fidelity_reward(RLAgent.pulse_2d, plot_result = True)

print(f"Final Pulse Fidelity is: {fidelity}")