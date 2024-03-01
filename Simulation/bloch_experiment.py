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
u_target = fc.sigmax()
initial_state = basis(2,0)
gate_duration = 2*np.pi
number_of_timesteps = 2
pulse_amplitude = -1/(2*np.pi)
pulse_list = pulse_amplitude * np.ones(shape = (number_of_timesteps))
print(pulse_list.shape)

t1 = 1e3 * gate_duration
t2 = 1e3 * gate_duration

# Create Quantum Environment

environment = QuantumEnvironment(n_q, drift_hamiltonian, control_hamiltonian, hamiltonian_label, t1, t2, u_target, w_f = 1, w_e = 0, timesteps = number_of_timesteps, pulse_duration = gate_duration, grape_iterations = 100, n_steps = 1, sweep_noise = True)

#grape_pulse = environment.run_grape_optimization(w_f = 1, w_e = 0, eps_f = 1, eps_e = 100)

environment.environment.pulses[0].coeff = pulse_list

#environment.plot_grape_pulses(pulse_list)

result = environment.environment.run_state(init_state = initial_state)

dm_sim = result.states[-1]

print(dm_sim)