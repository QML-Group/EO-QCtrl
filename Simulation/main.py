import numpy as np 
from qutip import basis, fidelity, identity, sigmax, sigmaz, tensor, destroy
import functions as fc
from qutip.metrics import fidelity
from simulator import QuantumEnvironment
from qrla import QuantumRLAgent
from input import *
import matplotlib.pyplot as plt

   
# Initialize Environments
TrainingEnvironment = QuantumEnvironment(number_qubits, h_d, h_c_3, h_l_3, t1, t2, initial_state, target_unitary_cnot, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles)
EvaluationEnvironment = QuantumEnvironment(number_qubits, h_d, h_c_3, h_l_3, t1, t2, initial_state, target_unitary_cnot, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles)
RLAgent = QuantumRLAgent(TrainingEnvironment, EvaluationEnvironment, num_episodes, fc_layer_params = (50, 30, 10))

# Run Training
RLAgent.run_training()

# Retrieve highest fidelity pulse
max_fid_pulse = RLAgent.return_highest_fidelity_pulse()

# Run pulse on Evaluation Environment to cross-check
_, fidelity_rl = EvaluationEnvironment.calculate_fidelity_reward(max_fid_pulse, plot_result = False)

# Print Results

print(f"    RESULTS 
      -------------------
      < T1: {t1} | T2: {t2} >
      < Target Unitary: {label_cnot} >
      < Number of training episodes: {num_episodes} | Number of GRAPE Iterations: {number_of_grape_iterations} | Number of Timesteps: {number_of_timesteps} >
      < RL Fidelity is: {fidelity_rl} >")

# Plot Results 

RLAgent.plot_fidelity_reward_per_iteration()