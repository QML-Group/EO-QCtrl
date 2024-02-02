import numpy as np 
from qutip import basis, fidelity, identity, sigmax, sigmaz, tensor, destroy
import functions as fc
from qutip.metrics import fidelity
from simulator import QuantumEnvironment
from qrla import QuantumRLAgent
from input import *
import matplotlib.pyplot as plt

   
# Initialize Environments
TrainingEnvironment = QuantumEnvironment(number_qubits, h_d, h_c_3, h_l_3, t1, t2, target_unitary_random, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles)
EvaluationEnvironment = QuantumEnvironment(number_qubits, h_d, h_c_3, h_l_3, t1, t2, target_unitary_random, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles)
RLAgent = QuantumRLAgent(TrainingEnvironment, EvaluationEnvironment, num_iterations, fc_layer_params = (200, 100, 50, 30, 10), replay_buffer_capacity = 10)

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

Number of training episodes: {num_iterations}

Number of timesteps: {number_of_timesteps}


"""

print(result)

RLAgent.plot_fidelity_reward_per_iteration()