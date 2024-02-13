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

# Define Noise Sweep
noise = np.linspace(start = 50, stop = 1, num = num_iterations_RL) * gate_duration

# Plot Standard GRAPE Pulses Fidelity & Energy as function of Noise
def grape_simulation():
    GRAPEEnvironment = QuantumEnvironment(number_qubits, h_d, h_c, h_l, t1, t2, target_unitary_cnot, 0.5, 0.5, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles)
    GRAPEEnvironment.initial_state = basis(4, 2)

    pulses = GRAPEEnvironment.run_grape_optimization(0.5, 0.5, epsilon_f, epsilon_e)

    fidelity_list = []
    energy_list = []
    reward_list = []

    for index, value in enumerate(noise):
        print(index)
        GRAPEEnvironment = QuantumEnvironment(number_qubits, h_d, h_c, h_l, value, value, target_unitary_cnot, 0.5, 0.5, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles)
        GRAPEEnvironment.initial_state = basis(4, 2)
        _, fid = GRAPEEnvironment.calculate_fidelity_reward(pulses)
        energy = GRAPEEnvironment.calculate_energetic_cost(pulses)
        reward = 0.5 * fid + 0.5 * (1 - energy)
        fidelity_list.append(fid)
        energy_list.append(energy)
        reward_list.append(reward)

    plt.plot(np.flip(noise), fidelity_list, label = "Fidelity")
    plt.plot(np.flip(noise), energy_list, label = "Energy")
    plt.plot(np.flip(noise), reward_list, label = "Reward")
    plt.xlabel("Noise Gain")
    plt.legend()
    plt.title("GRAPE Pulse Fidelity, Energy & Reward as function of Processor Decoherence time")
    plt.tight_layout()
    plt.show()

# Plot RL Agent Pulses Fidelity & Energy as function of Noise WITHOUT GRAPE Head Start
def rl_simulation():
    TrainingEnvironment = QuantumEnvironment(number_qubits, h_d, h_c, h_l, t1, t2, target_unitary_cnot, 1, 0, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles, sweep_noise = True)
    EvaluationEnvironment = QuantumEnvironment(number_qubits, h_d, h_c, h_l, t1, t2, target_unitary_cnot, 1, 0, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles, sweep_noise = True)
    
    TrainingEnvironmentGRAPE = GRAPEApproximation(number_qubits, h_d, h_c, h_l, target_unitary_cnot, w_f = 1, w_e = 0, timesteps = number_of_timesteps, grape_iterations = number_of_grape_iterations)
    EvaluationEnvironmentGRAPE = GRAPEApproximation(number_qubits, h_d, h_c, h_l, target_unitary_cnot, w_f = 1, w_e = 0, timesteps = number_of_timesteps, grape_iterations = number_of_grape_iterations)
    
    ApproximationAgent = GRAPEQRLAgent(TrainingEnvironmentGRAPE, EvaluationEnvironmentGRAPE, num_iterations_Approx, fc_layer_params = (100, 100, 100), replay_buffer_capacity = 100)
    ApproximationAgent.run_training()
    ApproximationAgent.save_weights('Test_Policy_Approx')

    RLAgent = QuantumRLAgent(TrainingEnvironment, EvaluationEnvironment, num_iterations_RL, w_f = 1, w_e = 0, fc_layer_params = (200, 100, 50, 30, 10), replay_buffer_capacity = 10, policy = 'Test_Policy_Approx', rand_initial_state = True, sweep_noise = True)
    RLAgent.run_training()
    RLAgent.save_weights('Test_Policy_RL_Sweep_Noise')
    RLAgent.plot_fidelity_energy_reward_per_iteration()

rl_simulation()