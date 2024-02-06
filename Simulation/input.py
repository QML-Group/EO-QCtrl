import numpy as np 
import functions as fc
from qutip import basis, fidelity, identity, sigmax, sigmaz, tensor, destroy, rand_ket

h_d = np.pi * (tensor(sigmaz(), identity(2)) + tensor(identity(2), sigmaz())) + (1/2) * np.pi * tensor(sigmaz(), sigmaz()) # Define Drift Hamiltonian used in "Processor"
h_c = [tensor(identity(2), sigmax())]
h_c_3 = [tensor(sigmax(), identity(2)), tensor(identity(2), sigmax()), tensor(sigmax(), sigmax())]
h_l = [r'$u_{2x}$'] 
h_l_3 = [r'$u_{1x}$', r'$u_{2x}$', r'$u_{xx}$']
target_unitary_cnot, label_cnot = fc.cnot(), "CNOT"
target_unitary_random, label_random = fc.Generate_Rand_Unitary(4), "Random Unitary"
number_qubits = 2
gate_duration = 2 * np.pi
t1 = 10 * gate_duration
t2 = 10 * gate_duration
number_of_timesteps = 10
number_of_grape_iterations = 500
initial_state = basis(4, 2)
initial_state_random = rand_ket(4)
weight_fidelity = 1
weight_energy = 0
epsilon_f = 1
epsilon_e = 100
n_cycles = 1
num_iterations_RL = 1000
num_iterations_Approx = 2000
