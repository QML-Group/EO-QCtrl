import numpy as np 
from qutip import basis, fidelity, identity, sigmax, sigmaz, tensor, destroy
import functions as fc
from qutip.metrics import fidelity
from environments import QuantumEnvironment
from environments import GRAPEApproximation
from qrla import QuantumRLAgent
from qrla import GRAPEQRLAgent
from input import *
import matplotlib.pyplot as plt

   
# Initialize Environments
TrainingEnvironment = QuantumEnvironment(number_qubits, h_d, h_c, h_l, t1, t2, target_unitary_cnot, 0.2, 0.8, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles)
EvaluationEnvironment = QuantumEnvironment(number_qubits, h_d, h_c, h_l, t1, t2, target_unitary_cnot, 0.2, 0.8, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles)

TrainingEnvironmentGRAPE = GRAPEApproximation(number_qubits, h_d, h_c, h_l, target_unitary_cnot, w_f = 1.0, w_e = 0, timesteps = number_of_timesteps, grape_iterations = number_of_grape_iterations)
EvaluationEnvironmentGRAPE = GRAPEApproximation(number_qubits, h_d, h_c, h_l, target_unitary_cnot, w_f = 1.0, w_e = 0, timesteps = number_of_timesteps, grape_iterations = number_of_grape_iterations)

ApproximationAgent = GRAPEQRLAgent(TrainingEnvironmentGRAPE, EvaluationEnvironmentGRAPE, num_iterations_Approx, fc_layer_params = (100, 100, 100), replay_buffer_capacity = 100)

# Run GRAPE Approximation Training Phase and save policy
ApproximationAgent.run_training()
ApproximationAgent.save_weights('Test_Policy_Approx')

# Initialize RLAgent Environment including loaded policy
RLAgent = QuantumRLAgent(TrainingEnvironment, EvaluationEnvironment, num_iterations_RL, w_f = 0.2, w_e = 0.8, fc_layer_params = (200, 100, 50, 30, 10), replay_buffer_capacity = 10, policy = None, rand_initial_state = False)

# Run Trainingq
RLAgent.run_training()
RLAgent.save_weights('Test_Policy_RL')

# Plot the Reward per iteration of the Approximation Agent 
ApproximationAgent.plot_reward_per_iteration()

# Plot Best Pulse Generated by the Approximation agent 
ApproximationAgent.plot_best_pulse()

# Plot the Pulse Generated by the QRLAgent 
RLAgent.plot_final_pulse()

# PLot the Fidelity per iteration of the QRLAgent
RLAgent.plot_fidelity_energy_reward_per_iteration()