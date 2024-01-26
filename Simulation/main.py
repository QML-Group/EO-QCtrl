import numpy as np 
from qutip import basis, fidelity, identity, sigmax, sigmaz, tensor, destroy
import functions as fc
from qutip.metrics import fidelity
from simulator import QuantumEnvironment
from QuantumRLagent import QuantumRLAgent
from input import *

# Test Quantum RL Agent Class

TrainingEnvironment = QuantumEnvironment(number_qubits, h_d, h_c, h_l, t1, t2, initial_state, target_unitary, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles)

EvaluationEnvironment = QuantumEnvironment(number_qubits, h_d, h_c, h_l, t1, t2, initial_state, target_unitary, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles)

RLAgent = QuantumRLAgent(TrainingEnvironment, EvaluationEnvironment, num_episodes)

RLAgent.run_training()

RLAgent.plot_fidelity_per_iteration()

RLAgent.plot_final_pulse()

_, fidelity = EvaluationEnvironment.calculate_fidelity_reward(RLAgent.pulse_2d, plot_result = True)

print(f"Final Pulse Fidelity is: {fidelity}")