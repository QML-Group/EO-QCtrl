import numpy as np 
from qutip import basis, fidelity, identity, sigmax, sigmaz, tensor, destroy
import functions as fc
from qutip.metrics import fidelity
from simulator import QuantumEnvironment
from QuantumRLagent import QuantumRLAgent
from input import *
import matplotlib.pyplot as plt

t1_t2 = [0.1 * gate_duration, 0.5 * gate_duration,  1 * gate_duration, 5 * gate_duration, 10 * gate_duration, 50 * gate_duration, 100 * gate_duration, 500 * gate_duration, 1000 * gate_duration, 5000 * gate_duration]

rl_errorrate = []

grape_errorrate = []

for index, t1_t2_value in enumerate(t1_t2):
        
    # Initialize Environments
    TrainingEnvironment = QuantumEnvironment(number_qubits, h_d, h_c_3, h_l_3, t1_t2_value, t1_t2_value, initial_state, target_unitary_cnot, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles)
    EvaluationEnvironment = QuantumEnvironment(number_qubits, h_d, h_c_3, h_l_3, t1_t2_value, t1_t2_value, initial_state, target_unitary_cnot, number_of_timesteps, gate_duration, number_of_grape_iterations, n_cycles)
    RLAgent = QuantumRLAgent(TrainingEnvironment, EvaluationEnvironment, num_episodes, fc_layer_params = (50, 30, 10))

    # Run Training
    RLAgent.run_training()

    # Retrieve highest fidelity pulse
    max_fid_pulse = RLAgent.return_highest_fidelity_pulse()

    # Run pulse on Evaluation Environment to cross-check
    _, fidelity_rl = EvaluationEnvironment.calculate_fidelity_reward(max_fid_pulse, plot_result = False)

    # Run Fidelity Only EO-GRAPE Algorithm
    if index == 0:
        grape_pulse = EvaluationEnvironment.run_grape_optimization(w_f = 1.0, w_e = 0.0, eps_f = 1, eps_e = 1000)

    # Calculate Fidelity of EO-GRAPE Pulse
    _, fidelity_grape = EvaluationEnvironment.calculate_fidelity_reward(grape_pulse, plot_result = False)

    rl_errorrate.append(1 - fidelity_rl)
    grape_errorrate.append(1 - fidelity_grape)

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.axhline(y = 0.01, color = "grey", ls = "dashed", label = "$\epsilon = 0.01$")
ax.plot(t1_t2, rl_errorrate, label = "QRLA", marker = "d", color = "#03080c")
ax.plot(t1_t2, grape_errorrate, label = "EO-GRAPE", marker = "d", color = "#5b97ca")
ax.set_ylim(0, 1.1)
ax.set_xlabel("Processor Decoherence time ($\mu s$)")
ax.set_ylabel("CNOT pulse error rate ($1-F$)")
ax.grid()
ax.legend(loc = "upper right")
plt.show()



"""
# Plot and Print Results
EvaluationEnvironment.plot_rl_pulses(max_fid_pulse)
EvaluationEnvironment.plot_grape_pulses(grape_pulse)
print(f"T1: {t1} | T2: {t2}")
print(f"Target Unitary: {label_cnot}")
print(f"Number of training episodes: {num_episodes} | Number of GRAPE Iterations: {number_of_grape_iterations} | Number of Timesteps: {number_of_timesteps}")
print(f"RL Fidelity is: {fidelity_rl} | GRAPE Fidelity is: {fidelity_grape}")
"""
