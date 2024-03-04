import numpy as np 
import functions as fc
from qutip import basis, fidelity, identity, sigmax, sigmaz, sigmay, tensor, destroy, rand_ket

h_d = np.pi * (tensor(sigmaz(), identity(2)) + tensor(identity(2), sigmaz())) + (1/2) * np.pi * tensor(sigmaz(), sigmaz()) # Define Drift Hamiltonian used in "Processor"
h_d_1_qubit = np.pi * sigmaz()
h_d_identity = identity(2)
h_c = [tensor(identity(2), sigmax())]
h_c_hadamard = [tensor(sigmax(), identity(2))]
h_c_1_qubit = [sigmax(), sigmay()]
h_l_1_qubit = [r'$u_{x}$', r'$u_{z}$']
h_l_hadamard = [r'$u_{1x}$'] 
h_c_3 = [tensor(sigmax(), identity(2)), tensor(identity(2), sigmax()), tensor(sigmax(), sigmax())]
h_l = [r'$u_{2x}$'] 
h_l_3 = [r'$u_{1x}$', r'$u_{2x}$', r'$u_{xx}$']
target_unitary_cnot, label_cnot = fc.cnot(),     "CNOT"
target_unitary_random, label_random = fc.Generate_Rand_Unitary(4), "Random Unitary"
target_unitary_hadamard = fc.tensor(fc.hadamard(), fc.identity(2))
target_unitary_t_gate = fc.tensor(fc.t_gate(), fc.identity(2))
target_unitary_hadamard_1_qubit = fc.hadamard()
target_unitary_t_gate_1_qubit = fc.t_gate()
number_qubits = 1
gate_duration = 2 * np.pi
t1 = 100 * gate_duration
t2 = 100 * gate_duration
number_of_timesteps = 100
number_of_grape_iterations = 500
initial_state = basis(4, 2)
initial_state_random = rand_ket(4)
weight_fidelity = 1
weight_energy = 0
epsilon_f = 1
epsilon_e = 100
n_cycles = 1
num_iterations_RL = 10000
num_iterations_Approx = 2000

