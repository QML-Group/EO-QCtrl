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


weights = [0, 0.2, 0.5, 0.8]
target_unitary = fc.yaqq_unitary_1()
drift_hamiltonian = h_d_1_qubit
n_q = number_qubits
control_hamiltonian = h_c_yaqq
hamiltonian_label = h_l_yaqq
initial_state_yaqq = basis(2,0)
control_pulses = np.zeros((len(control_hamiltonian), number_of_timesteps, len(weights)))
timespace = np.linspace(0, gate_duration, number_of_timesteps)

for weight_index, weight_value in enumerate(weights):

    environment = QuantumEnvironment(n_q, drift_hamiltonian, control_hamiltonian, hamiltonian_label, t1, t2, target_unitary, w_f = 1 - weight_value, w_e = weight_value, timesteps = number_of_timesteps, pulse_duration = gate_duration, grape_iterations = 200, n_steps = 1, sweep_noise = False)
    environment.initial_state = initial_state_yaqq
    control_pulses[:, :, weight_index] = environment.run_grape_optimization(w_f = 1 - weight_value, w_e = weight_value, eps_f = 1, eps_e = 100)
    _, fidelity_pulse = environment.calculate_fidelity_reward(control_pulses[:, :, weight_index], plot_result = False)
    print(fidelity_pulse)

fig, ax = plt.subplots(len(control_hamiltonian), sharex = True)
xticks = [0, np.pi, 2 * np.pi]
colors = ['#03080c','#214868', '#5b97ca', '#9fc2e0']

for i in range(len(control_hamiltonian)):
    for index_weight, value_weight in enumerate(weights):
        ax[i].plot(timespace, control_pulses[i, :, index_weight], label = f"$w_f$ = {round(1 - value_weight, 1)}, $w_e$ = {round(value_weight, 1)}", color = colors[index_weight])
        ax[i].set(xlabel = "Time", ylabel = f"{hamiltonian_label[i]}")

plt.xticks(xticks, ['0', '$\pi$', '2 $\pi$'])
plt.legend()
plt.subplot_tool()
plt.tight_layout()
plt.show()