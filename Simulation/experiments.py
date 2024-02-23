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
noise_low = np.linspace(start = 200, stop = 10, num = 10000) * gate_duration
noise_high = np.linspace(start = 10, stop = 0.01, num = 10000) * gate_duration

nn_size_1 = (200, 100, 50, 30, 10)
nn_size_2 = (400, 200, 100, 50, 30, 10)
nn_size_3 = (600, 400, 200, 100, 50, 30, 10)

def moving_average(a, n = 100):

        ret = np.cumsum(a)
        ret[n:] = ret[n:] - ret[:-n]

        return ret[n - 1:] / n

# Plot Standard GRAPE Pulses Fidelity & Energy as function of Noise
def grape_simulation(noise_level = "Low"):

    if noise_level == "Low":

        noise = noise_low

    else:

        noise = noise_high

    GRAPEEnvironment = QuantumEnvironment(1, h_d_1_qubit, h_c_1_qubit, h_l_1_qubit, t1, t2, target_unitary_hadamard_1_qubit, 1, 0, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles)
    GRAPEEnvironment.initial_state = rand_ket(2)

    pulses = GRAPEEnvironment.run_grape_optimization(1, 0, epsilon_f, epsilon_e)
    GRAPEEnvironment.plot_grape_pulses(pulses)
    fidelity_list = []
    energy_list = []

    for index, value in enumerate(noise):
        print(index)
        if (index % 10 == 0):
            GRAPEEnvironment = QuantumEnvironment(1, h_d_1_qubit, h_c_1_qubit, h_l_1_qubit, value, value, target_unitary_hadamard_1_qubit, 1, 0, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles)
        new_state = rand_ket(2)
        GRAPEEnvironment.initial_state = new_state
        _, fid = GRAPEEnvironment.calculate_fidelity_reward(pulses)
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
    plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], ['200 T', '180 T', '160 T', '140 T', '120 T', '100 T', '80 T', '60 T', '40 T', '20 T', '0T'])
    plt.xlabel("Decoherence time")
    plt.legend()
    plt.title("GRAPE Fidelity, Energy vs. Decoherence Time")
    plt.tight_layout()
    plt.show()

# Plot RL Agent Pulses Fidelity & Energy as function of Noise WITHOUT GRAPE Head Start
def rl_simulation(nn_size, warm_start = False, noise_level = "Low"):    

    if warm_start == True:
        policy = 'Test_Policy_Approx'

    else: 
        policy = None

    TrainingEnvironment = QuantumEnvironment(1, h_d_1_qubit, h_c_1_qubit, h_l_1_qubit, t1, t2, target_unitary_hadamard_1_qubit, 1, 0, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles, sweep_noise = True)
    EvaluationEnvironment = QuantumEnvironment(1, h_d_1_qubit, h_c_1_qubit, h_l_1_qubit, t1, t2, target_unitary_hadamard_1_qubit, 1, 0, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles, sweep_noise = True)

    if warm_start == True:
        TrainingEnvironmentGRAPE = GRAPEApproximation(1, h_d_1_qubit, h_c_1_qubit, h_l_1_qubit, target_unitary_hadamard_1_qubit, w_f = 1, w_e = 0, timesteps = number_of_timesteps, grape_iterations = number_of_grape_iterations)
        EvaluationEnvironmentGRAPE = GRAPEApproximation(1, h_d_1_qubit, h_c_1_qubit, h_l_1_qubit, target_unitary_hadamard_1_qubit, w_f = 1, w_e = 0, timesteps = number_of_timesteps, grape_iterations = number_of_grape_iterations)
        
        ApproximationAgent = GRAPEQRLAgent(TrainingEnvironmentGRAPE, EvaluationEnvironmentGRAPE, num_iterations_Approx, fc_layer_params = (100, 100, 100), replay_buffer_capacity = 100)
        ApproximationAgent.run_training()
        ApproximationAgent.save_weights('Test_Policy_Approx')

    RLAgent = QuantumRLAgent(TrainingEnvironment, EvaluationEnvironment, num_iterations_RL, w_f = 1, w_e = 0, fc_layer_params = nn_size, replay_buffer_capacity = 10, policy = policy, rand_initial_state = True, sweep_noise = True, noise_level = noise_level)
    RLAgent.run_training()
    RLAgent.save_weights('Test_Policy_RL_Sweep_Noise')
    RLAgent.plot_fidelity_energy_reward_per_iteration()

def plot_combined_result(plot_energy = False, noise_level = "Low"):

    GRAPE_RAW_E = np.load('GRAPE_Energy_List_RAW.npy')
    GRAPE_MA_E = np.load('GRAPE_Energy_List_MA.npy')
    GRAPE_RAW_F = np.load('GRAPE_Fidelity_Noise_RAW.npy')
    GRAPE_MA_F = np.load('GRAPE_Fidelity_Noise_MA.npy')
    RL_RAW_E = np.load('RL_Energy_Noise_RAW.npy')
    RL_MA_E = np.load('RL_Energy_List_MA.npy')
    RL_RAW_F = np.load('RL_Fidelity_Noise_RAW.npy')
    RL_MA_F = np.load('RL_Fidelity_List_MA.npy')

    iteration_space_grape = np.arange(len(GRAPE_MA_F))
    iteration_space_rl = np.arange(len(RL_MA_F))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twiny()

    if noise_level == "Low":

        ax2.set_xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], ['200 T', '180 T', '160 T', '140 T', '120 T', '100 T', '80 T', '60 T', '40 T', '20 T', '0T'])

    elif noise_level == "High":

        ax2.set_xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], ['10 T', '9 T', '8 T', '7 T', '6 T', '5 T', '4 T', '3 T', '2 T', '1 T', '0T'])

    ax2.set_xlabel("Decoherence Time (T1, T2)")
    ax2.axhline(y = 0, color = "grey")

    ax1.plot(iteration_space_grape, GRAPE_RAW_F[:len(iteration_space_grape)], color = "blue", alpha = 0.2)
    ax1.plot(iteration_space_rl, RL_RAW_F[:len(iteration_space_rl)], color = "red", alpha = 0.2)
    ax1.plot(iteration_space_grape, GRAPE_MA_F, label = r"$F_{GRAPE}$", color = "blue")
    ax1.plot(iteration_space_rl, RL_MA_F, label = r"$F_{QRLA}$", color = "red")

    if plot_energy == "True":

        ax1.plot(iteration_space_grape, GRAPE_RAW_E[:len(iteration_space_grape)], color = "purple", alpha = 0.2)
        ax1.plot(iteration_space_grape, GRAPE_MA_E, label = r"$E_{GRAPE}$", color = "purple")
        ax1.plot(iteration_space_rl, RL_RAW_E[:len(iteration_space_rl)], color = "green", alpha = 0.2)
        ax1.plot(iteration_space_rl, RL_MA_E, label = r"$E_{QRLA}$", color = "green")

    #ax1.set_ylim(0.0, 1.0)
    ax1.set_xlabel("Episode number")
    ax1.legend()
    fig.tight_layout()
    plt.show()

def plot_rl_only_combined(plot_energy = False, noise_level = "Low"):

    WS_RAW_E = np.load('WS_RAW_E.npy')
    WS_MA_E = np.load('WS_MA_E.npy')
    WS_RAW_F = np.load('WS_RAW_F.npy')
    WS_MA_F = np.load('WS_MA_F.npy')
    WOWS_RAW_E = np.load('WOWS_RAW_E.npy')
    WOWS_MA_E = np.load('WOWS_MA_E.npy')
    WOWS_RAW_F = np.load('WOWS_RAW_F.npy')
    WOWS_MA_F = np.load('WOWS_MA_F.npy')

    iteration_space_rl = np.arange(len(WS_MA_E))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twiny()

    if noise_level == "Low":

        ax2.set_xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9500], ['200 T', '180 T', '160 T', '140 T', '120 T', '100 T', '80 T', '60 T', '40 T', '20 T', '0T'])

    elif noise_level == "High":

        ax2.set_xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], ['10 T', '9 T', '8 T', '7 T', '6 T', '5 T', '4 T', '3 T', '2 T', '1 T', '0 T'])

    ax2.set_xlabel("Decoherence Time (T1, T2)")
    ax2.axhline(y = 0, color = "grey")

    ax1.plot(iteration_space_rl, WS_RAW_F[:len(iteration_space_rl)], color = "blue", alpha = 0.2)
    ax1.plot(iteration_space_rl, WOWS_RAW_F[:len(iteration_space_rl)], color = "red", alpha = 0.2)
    ax1.plot(iteration_space_rl, WS_MA_F, label = r"$F_{WS}$", color = "blue")
    ax1.plot(iteration_space_rl, WOWS_MA_F, label = r"$F_{WOWS}$", color = "red")

    if plot_energy == True:

        ax1.plot(iteration_space_rl, WS_RAW_E[:len(iteration_space_rl)], color = "purple", alpha = 0.2)
        ax1.plot(iteration_space_rl, WS_MA_E, label = r"$E_{WS}$", color = "purple")
        ax1.plot(iteration_space_rl, WOWS_RAW_E[:len(iteration_space_rl)], color = "green", alpha = 0.2)
        ax1.plot(iteration_space_rl, WOWS_MA_E, label = r"$E_{WOWS}$", color = "green")

    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlabel("Episode number")
    ax1.legend(loc = "center left")
    fig.tight_layout()
    plt.show()


#grape_simulation(noise_level = "High")

rl_simulation(nn_size = nn_size_3, warm_start = False, noise_level = "High")  

plot_combined_result(plot_energy = False, noise_level = "High") 