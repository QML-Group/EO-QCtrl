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
drift_hamiltonian = h_d_1_qubit
control_hamiltonian = h_c_yaqq
hamiltonian_label = h_l_yaqq
timespace = np.linspace(0, gate_duration, number_of_timesteps)

fig, ax = plt.subplots(len(control_hamiltonian), sharex = True)
xticks = [0, np.pi, 2 * np.pi]
colors = ['#03080c','#214868', '#5b97ca', '#9fc2e0']

control_pulses = np.load("p_1_opt_500_a.npy")
#control_pulses = np.load("p_1_opt_500_b.npy")

for i in range(len(control_hamiltonian)):
    for index_weight, value_weight in enumerate(weights):
        ax[i].plot(timespace, control_pulses[i, :, index_weight], label = f"$w_f$ = {round(1 - value_weight, 1)}, $w_e$ = {round(value_weight, 1)}", color = colors[index_weight])
        ax[i].set(xlabel = "Time", ylabel = f"{hamiltonian_label[i]}")

plt.xticks(xticks, ['0', '$\pi$', '2 $\pi$'])
plt.legend()
plt.subplot_tool()
plt.tight_layout()
plt.show()