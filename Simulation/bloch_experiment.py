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
from qutip import rand_ket

# Define input parameters
drift_hamiltonian = h_d_identity
n_q = number_qubits
control_hamiltonian = h_c_1_qubit
hamiltonian_label = h_l_1_qubit
u_target = fc.rx_gate(np.pi/2)
initial_state = basis(2,0)
gate_duration = 2 * np.pi
number_of_timesteps = 500
t1 = 1e5 * gate_duration
t2 = 1e5 * gate_duration
custom_pulse = np.zeros((1, number_of_timesteps))

# Create Quantum Environment

environment = QuantumEnvironment(n_q, drift_hamiltonian, control_hamiltonian, hamiltonian_label, t1, t2, u_target, w_f = 1, w_e = 0, timesteps = number_of_timesteps, pulse_duration = gate_duration, grape_iterations = 200, n_steps = 1, sweep_noise = False)

environment.initial_state = initial_state

grape_pulses = environment.run_grape_optimization(w_f = 1, w_e = 0, eps_f = 1, eps_e = 100)

_, f_grape= environment.calculate_fidelity_reward(grape_pulses, plot_result = False)

environment.plot_grape_pulses(grape_pulses)

print(f_grape)

environment.plot_bloch_sphere_trajectory()