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
drift_hamiltonian = h_d_1_qubit
n_q = number_qubits
control_hamiltonian = h_c_1_qubit
hamiltonian_label = h_l_1_qubit
u_target = fc.rx_gate(np.pi/2)
initial_state = basis(2,0)
gate_duration = 2 * np.pi
number_of_timesteps = 500
t1 = 100 * gate_duration
t2 = 100 * gate_duration

# Simple Bloch Sphere Plot 
def bloch_sphere_grape():
    environment = QuantumEnvironment(n_q, drift_hamiltonian, control_hamiltonian, hamiltonian_label, t1, t2, u_target, w_f = 1, w_e = 0, timesteps = number_of_timesteps, pulse_duration = gate_duration, grape_iterations = 200, n_steps = 1, sweep_noise = False)
    environment.initial_state = initial_state
    grape_pulses = environment.run_grape_optimization(w_f = 1, w_e = 0, eps_f = 1, eps_e = 100)
    _, f_grape= environment.calculate_fidelity_reward(grape_pulses, plot_result = False)
    environment.plot_grape_pulses(grape_pulses)
    print(f_grape)
    environment.plot_bloch_sphere_trajectory()
    total_arc_length = environment.get_total_arc_length()
    print(total_arc_length)

def bloch_sphere_rl():

    TrainingEnvironment = QuantumEnvironment(n_q, drift_hamiltonian, control_hamiltonian, hamiltonian_label, t1, t2, u_target, w_f = 1, w_e = 0, timesteps = number_of_timesteps, pulse_duration = gate_duration, grape_iterations = 200, n_steps = 1, sweep_noise = False)
    TrainingEnvironment.initial_state = initial_state
    EvaluationEnvironment = QuantumEnvironment(n_q, drift_hamiltonian, control_hamiltonian, hamiltonian_label, t1, t2, u_target, w_f = 1, w_e = 0, timesteps = number_of_timesteps, pulse_duration = gate_duration, grape_iterations = 200, n_steps = 1, sweep_noise = False)
    EvaluationEnvironment.initial_state = initial_state
    RLAgent = QuantumRLAgent(TrainingEnvironment, EvaluationEnvironment, num_iterations_RL, w_f = 1, w_e = 0, fc_layer_params = (200, 100, 50, 30, 10), replay_buffer_capacity = 10, policy = None, rand_initial_state = False)
    RLAgent.initial_state = initial_state
    RLAgent.run_training()

    BestPulse = RLAgent.get_highest_fidelity_pulse()

    _, f_rl = EvaluationEnvironment.calculate_fidelity_reward(BestPulse, plot_result = False)

    print(f_rl)

    EvaluationEnvironment.plot_bloch_sphere_trajectory()


bloch_sphere_grape()