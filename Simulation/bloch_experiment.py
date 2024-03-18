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
number_of_timesteps = 200
t1 = 1000 * gate_duration
t2 = 1000 * gate_duration

# Simple Bloch Sphere Plot 
def bloch_sphere_grape():
    environment = QuantumEnvironment(n_q, drift_hamiltonian, control_hamiltonian, hamiltonian_label, t1, t2, u_target, w_f = 1, w_e = 0, timesteps = number_of_timesteps, pulse_duration = gate_duration, grape_iterations = 200, n_steps = 1, sweep_noise = False)
    environment.initial_state = initial_state
    grape_pulses = environment.run_grape_optimization(w_f = 1, w_e = 0, eps_f = 1, eps_e = 100)
    _, f_grape= environment.calculate_fidelity_reward(grape_pulses, plot_result = False)

    environment.plot_grape_pulses(grape_pulses)

    print("Fidelity is:", f_grape)

    environment.plot_bloch_sphere_trajectory()

    total_arc_length = environment.get_total_arc_length()

    print("Total Arc Length is:", total_arc_length)

def bloch_sphere_rl():

    TrainingEnvironment = QuantumEnvironment(n_q, drift_hamiltonian, control_hamiltonian, hamiltonian_label, t1, t2, u_target, w_f = 0.5, w_e = 0.5, timesteps = number_of_timesteps, pulse_duration = gate_duration, grape_iterations = 200, n_steps = 1, sweep_noise = False)
    TrainingEnvironment.initial_state = initial_state
    EvaluationEnvironment = QuantumEnvironment(n_q, drift_hamiltonian, control_hamiltonian, hamiltonian_label, t1, t2, u_target, w_f = 0.5, w_e = 0.5, timesteps = number_of_timesteps, pulse_duration = gate_duration, grape_iterations = 200, n_steps = 1, sweep_noise = False)
    EvaluationEnvironment.initial_state = initial_state
    RLAgent = QuantumRLAgent(TrainingEnvironment, EvaluationEnvironment, num_iterations_RL, w_f = 0.5, w_e = 0.5, fc_layer_params = (200, 100, 50, 30, 10), replay_buffer_capacity = 10, policy = None, rand_initial_state = False)
    RLAgent.initial_state = initial_state
    RLAgent.run_training()

    BestPulse = RLAgent.get_highest_fidelity_pulse()

    _, f_rl = EvaluationEnvironment.calculate_fidelity_reward(BestPulse, plot_result = False)

    print("Fidelity is", f_rl)

    EvaluationEnvironment.plot_bloch_sphere_trajectory()

    arc_length = EvaluationEnvironment.get_total_arc_length()

    print("Total Arc Length is:" , arc_length)

def correlation_experiment_grape():

    weights = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    arc_length_list = []
    energetic_cost_list = []

    for i in range(len(weights)):

        environment = QuantumEnvironment(n_q, drift_hamiltonian, control_hamiltonian, hamiltonian_label, t1, t2, u_target, w_f = weights[i], w_e = 1 - weights[i], timesteps = number_of_timesteps, pulse_duration = gate_duration, grape_iterations = 200, n_steps = 1, sweep_noise = False)
        environment.initial_state = initial_state
        grape_pulses = environment.run_grape_optimization(w_f = weights[i], w_e = 1 - weights[i], eps_f = 1, eps_e = 100)
        _, f_grape= environment.calculate_fidelity_reward(grape_pulses, plot_result = False)
        total_arc_length = environment.get_total_arc_length()
        energetic_cost = environment.calculate_energetic_cost(grape_pulses, return_normalized = False)
        arc_length_list.append(total_arc_length)
        energetic_cost_list.append(energetic_cost)

    plt.plot(energetic_cost_list, arc_length_list, marker = 'd', color = '#214868')
    plt.xlabel("Energetic Cost (a.u.)")
    plt.ylabel("Bloch Sphere Arc Length (a.u.)")
    plt.grid()
    plt.tight_layout()
    plt.show()

def correlation_experiment_rl():

    weights = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    arc_length_list = []
    energetic_cost_list = []

    for i in range(len(weights)):

        TrainingEnvironment = QuantumEnvironment(n_q, drift_hamiltonian, control_hamiltonian, hamiltonian_label, t1, t2, u_target, w_f = weights[i], w_e = 1 - weights[i], timesteps = number_of_timesteps, pulse_duration = gate_duration, grape_iterations = 200, n_steps = 1, sweep_noise = False)
        TrainingEnvironment.initial_state = initial_state
        EvaluationEnvironment = QuantumEnvironment(n_q, drift_hamiltonian, control_hamiltonian, hamiltonian_label, t1, t2, u_target, w_f = weights[i], w_e = 1- weights[i], timesteps = number_of_timesteps, pulse_duration = gate_duration, grape_iterations = 200, n_steps = 1, sweep_noise = False)
        EvaluationEnvironment.initial_state = initial_state
        RLAgent = QuantumRLAgent(TrainingEnvironment, EvaluationEnvironment, num_iterations_RL, w_f = weights[i], w_e = 1 - weights[i], fc_layer_params = (200, 100, 50, 30, 10), replay_buffer_capacity = 10, policy = None, rand_initial_state = False)
        RLAgent.initial_state = initial_state
        RLAgent.run_training()

        BestPulse = RLAgent.get_final_pulse()

        _, f_rl = EvaluationEnvironment.calculate_fidelity_reward(BestPulse, plot_result = False)
        total_arc_length = EvaluationEnvironment.get_total_arc_length()
        energetic_cost = EvaluationEnvironment.calculate_energetic_cost(BestPulse, return_normalized = False)
        arc_length_list.append(total_arc_length)
        energetic_cost_list.append(energetic_cost)

    np.save("Arc_Length_2.npy", arc_length_list)
    np.save("EC_RL_2.npy", energetic_cost_list)
    plt.grid()
    plt.scatter(energetic_cost_list, arc_length_list, marker = 'd', color = '#214868')
    plt.xlabel("Energetic Cost (a.u.)")
    plt.ylabel("Bloch Sphere Arc Length (a.u.)")
    plt.grid()
    plt.tight_layout()
    plt.show()


arc_length_1 = np.load("Arc_Length.npy")
arc_length_2 = np.load("Arc_Length_2.npy")
ec_1 = np.load("EC_RL.npy")
ec_2 = np.load("EC_RL_2.npy")
combined_arc_length = np.hstack((arc_length_1, arc_length_2))
combined_ec = np.hstack((ec_1, ec_2))
fit = np.polyfit(combined_ec, combined_arc_length, 1)
p = np.poly1d(fit)
plt.scatter(combined_ec, combined_arc_length, marker = 'd', color = '#214868')
plt.plot(combined_ec, p(combined_ec), color = '#5b97ca')
plt.xlabel("Energetic Cost (a.u.)")
plt.ylabel("Bloch Sphere Arc Length (a.u.)")
plt.grid()
plt.tight_layout()
plt.show()