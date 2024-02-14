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

# Define Noise Sweep
noise = np.linspace(start = 100, stop = 0.01, num = 10000) * gate_duration

def moving_average(a, n = 10):

        ret = np.cumsum(a)
        ret[n:] = ret[n:] - ret[:-n]

        return ret[n - 1:] / n

# Plot Standard GRAPE Pulses Fidelity & Energy as function of Noise
def grape_simulation():
    GRAPEEnvironment = QuantumEnvironment(number_qubits, h_d, h_c_3, h_l_3, t1, t2, target_unitary_cnot, 0.8, 0.2, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles)
    GRAPEEnvironment.initial_state = rand_ket(4)

    pulses = GRAPEEnvironment.run_grape_optimization(0.7, 0.3, epsilon_f, epsilon_e)

    fidelity_list = []
    energy_list = []

    for index, value in enumerate(noise):
        if (index % 100 == 0):
            GRAPEEnvironment = QuantumEnvironment(number_qubits, h_d, h_c_3, h_l_3, value, value, target_unitary_cnot, 0.8, 0.2, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles)
        new_state = rand_ket(4)
        GRAPEEnvironment.initial_state = new_state
        _, fid = GRAPEEnvironment.calculate_fidelity_reward(pulses)
        print(index)
        energy = GRAPEEnvironment.calculate_energetic_cost(pulses)
        fidelity_list.append(fid)
        energy_list.append(energy)
        ma_fid_list = moving_average(fidelity_list)
        ma_energy_list = moving_average(energy_list)

    ma_iteration_space = np.arange(len(ma_fid_list))

    np.save('GRAPE_Fidelity_Noise_RAW', fidelity_list)
    np.save('GRAPE_Energy_List_RAW', energy_list)
    np.save('GRAPE_Fidelity_Noise_MA', ma_fid_list)
    np.save('GRAPE_Energy_List_MA', ma_energy_list)
    plt.plot(ma_iteration_space, fidelity_list[:len(ma_fid_list)], label = "Fidelity raw", color = '#FFCCCB')
    plt.plot(ma_iteration_space, energy_list[:len(ma_fid_list)], label = "Energy raw", color = '#ECFFDC')
    plt.plot(ma_iteration_space, ma_fid_list, label = "Fidelity MA", color = '#F70D1A')
    plt.plot(ma_iteration_space, ma_energy_list, label = "Energy MA", color = '#7CFC00')
    plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], ['100 T', '90 T', '80 T', '70 T', '60 T', '50 T', '40 T', '30 T', '20 T', '10 T', '0T'])
    plt.xlabel("Decoherence time")
    plt.legend()
    plt.title("GRAPE Fidelity, Energy vs. Decoherence Time")
    plt.tight_layout()
    plt.show()

# Plot RL Agent Pulses Fidelity & Energy as function of Noise WITHOUT GRAPE Head Start
def rl_simulation():
    TrainingEnvironment = QuantumEnvironment(number_qubits, h_d, h_c, h_l, t1, t2, target_unitary_cnot, 0.7, 0.3, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles, sweep_noise = True)
    EvaluationEnvironment = QuantumEnvironment(number_qubits, h_d, h_c, h_l, t1, t2, target_unitary_cnot, 0.7, 0.3, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles, sweep_noise = True)
    
    TrainingEnvironmentGRAPE = GRAPEApproximation(number_qubits, h_d, h_c, h_l, target_unitary_cnot, w_f = 0.7, w_e = 0.3, timesteps = number_of_timesteps, grape_iterations = number_of_grape_iterations)
    EvaluationEnvironmentGRAPE = GRAPEApproximation(number_qubits, h_d, h_c, h_l, target_unitary_cnot, w_f = 0.7, w_e = 0.3, timesteps = number_of_timesteps, grape_iterations = number_of_grape_iterations)
    
    ApproximationAgent = GRAPEQRLAgent(TrainingEnvironmentGRAPE, EvaluationEnvironmentGRAPE, num_iterations_Approx, fc_layer_params = (100, 100, 100), replay_buffer_capacity = 100)
    ApproximationAgent.run_training()
    ApproximationAgent.save_weights('Test_Policy_Approx')

    RLAgent = QuantumRLAgent(TrainingEnvironment, EvaluationEnvironment, num_iterations_RL, w_f = 0.7, w_e = 0.3, fc_layer_params = (200, 100, 50, 30, 10), replay_buffer_capacity = 10, policy = None, rand_initial_state = True, sweep_noise = True)
    RLAgent.run_training()
    RLAgent.save_weights('Test_Policy_RL_Sweep_Noise')
    RLAgent.plot_fidelity_energy_reward_per_iteration()

rl_simulation()