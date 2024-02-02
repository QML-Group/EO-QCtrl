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

TrainingEnvironment = GRAPEApproximation(number_qubits, h_d, h_c_3, h_l_3, target_unitary_cnot, timesteps = 100)
EvaluationEnvironment = GRAPEApproximation(number_qubits, h_d, h_c_3, h_l_3, target_unitary_cnot, timesteps = 100)

RLAgent = GRAPEQRLAgent(TrainingEnvironment, EvaluationEnvironment, num_iterations, fc_layer_params = (200, 200, 200))

RLAgent.run_training()

TrainingEnvironment.plot_grape_pulses(TrainingEnvironment.target_pulse)

RLAgent.plot_final_pulse()
RLAgent.plot_reward_per_iteration()