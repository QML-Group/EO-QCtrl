import numpy as np 
from qutip import basis, fidelity, identity, sigmax, sigmaz, tensor, destroy
import functions as fc
from qutip.metrics import fidelity
from simulator import QuantumEnvironment
from simulator import GRAPEApproximation
from qrla import QuantumRLAgent
from qrla import GRAPEQRLAgent
from input import *
import matplotlib.pyplot as plt

   
# Initialize Environments
TrainingEnvironment = QuantumEnvironment(number_qubits, h_d, h_c, h_l, t1, t2, target_unitary_cnot, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles)
EvaluationEnvironment = QuantumEnvironment(number_qubits, h_d, h_c, h_l, t1, t2, target_unitary_cnot, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles)

TrainingEnvironmentGRAPE = GRAPEApproximation(number_qubits, h_d, h_c, h_l, target_unitary_cnot, timesteps = number_of_timesteps, grape_iterations = number_of_grape_iterations)
EvaluationEnvironmentGRAPE = GRAPEApproximation(number_qubits, h_d, h_c, h_l, target_unitary_cnot, timesteps = number_of_timesteps, grape_iterations = number_of_grape_iterations)

ApproximationAgent = GRAPEQRLAgent(TrainingEnvironmentGRAPE, EvaluationEnvironmentGRAPE, num_iterations_Approx, fc_layer_params = (100, 100, 100), replay_buffer_capacity = 100)

# Run GRAPE Approximation Training Phase and save policy

ApproximationAgent.run_training()
ApproximationAgent.save_weights('Test_Policy_Approx')

# Initialize RLAgent Environment including loaded policy

RLAgent = QuantumRLAgent(TrainingEnvironment, EvaluationEnvironment, num_iterations_RL, fc_layer_params = (200, 100, 50, 30, 10), replay_buffer_capacity = 10, policy = 'Test_Policy_Approx')

# Run Training

RLAgent.run_training()

# Retrieve highest fidelity pulse
final_pulse = RLAgent.get_final_pulse()

# Run pulse on Evaluation Environment to cross-check
_, fidelity_rl = EvaluationEnvironment.calculate_fidelity_reward(final_pulse, plot_result = False)

# Print Results

result = f"""

RESULTS
-------

RL Fidelity is: {fidelity_rl}

T1 : {t1} | T2: {t2}

Target Unitary: {label_cnot}

Number of training episodes: {num_iterations_RL}

Number of timesteps: {number_of_timesteps}


"""

print(result)

RLAgent.plot_fidelity_reward_per_iteration()

TrainingEnvironment.plot_rl_pulses(final_pulse)

ApproximationAgent.plot_final_pulse()

ApproximationAgent.plot_reward_per_iteration()