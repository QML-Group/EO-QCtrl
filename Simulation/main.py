import numpy as np 
from qutip import basis, fidelity, identity, sigmax, sigmaz, tensor, destroy
import functions as fc
from qutip.metrics import fidelity
from simulator import QuantumEnvironment
from QuantumRLagent import QuantumRLAgent
from input import *

# Test Quantum RL Agent Class

TrainingEnvironment = QuantumEnvironment(number_qubits, h_d, h_c_3, h_l_3, t1, t2, initial_state, target_unitary, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles)

EvaluationEnvironment = QuantumEnvironment(number_qubits, h_d, h_c_3, h_l_3, t1, t2, initial_state, target_unitary, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles)

RLAgent = QuantumRLAgent(TrainingEnvironment, EvaluationEnvironment, num_episodes)

RLAgent.run_training()

RLAgent.plot_fidelity_per_iteration()

pulses = RLAgent.return_final_pulse()

_, fidelity = EvaluationEnvironment.calculate_fidelity_reward(pulses, plot_result = True)

EvaluationEnvironment.plot_rl_pulses(pulses)

print(f"Final Pulse Fidelity is: {fidelity}")