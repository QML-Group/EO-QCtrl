import numpy as np 
from qutip import basis, fidelity, identity, sigmax, sigmaz, tensor, destroy
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import expand_operator, toffoli, snot
import matplotlib.pyplot as plt 
from qutip.qip.device import Processor
import functions as fc
import qutip.visualization as vz
from qutip.qip.noise import RandomNoise, DecoherenceNoise, RelaxationNoise

Run_Analytical = False # Define to run analytical or not 

a = destroy(2)

Hadamard = snot()

N_q = 2 # Define number of qubits

w1 = 5e9 # Hz

w2 = 5e9 # Hz

g = 500e6 # Hz

hbar = 1.054e-34 # Js/rad

GaussianNoise = RandomNoise(dt = 0.001, rand_gen = np.random.normal, loc = 0.2, scale = 0.1)

RelaxNoise = RelaxationNoise()

DecoNoise = DecoherenceNoise(c_ops = [a.dag() * a, Hadamard * a.dag() * a * Hadamard])

Iterations = 500 # Number of GRAPE Iterations

Timesteps = 500 # Number of Timesteps

T = 2 * np.pi # Total pulse duration

T2 = 100 # us

timespace = np.linspace(0, T, Timesteps) # Define time space based on total time and number of timesteps

U_Target = fc.cnot() # Define target unitary 

H_Drift_Qutip = np.pi * (tensor(sigmaz(), identity(2)) + tensor(identity(2), sigmaz())) + (1/2) * np.pi * tensor(sigmaz(), sigmaz()) # Define Drift Hamiltonian used in "Processor"

H_Drift_Scratch = np.pi * (fc.tensor(fc.sigmaz(), fc.identity(2)) + fc.tensor(fc.identity(2), fc.sigmaz())) + (1/2) * np.pi * fc.tensor(fc.sigmaz(), fc.sigmaz()) # Define Drift Hamiltonian used for optimization

H_Control_Qutip = [tensor(sigmax(), identity(2)), # Define Control Hamiltonian used in "Processor"
                   tensor(identity(2), sigmax()),
                   tensor(sigmax(), sigmax())]

H_Control_Scratch = [fc.tensor(fc.sigmax(), fc.identity(2)), # Define Control Hamiltonian used for optimization
                   fc.tensor(fc.identity(2), fc.sigmax()),
                   fc.tensor(fc.sigmax(), fc.sigmax())]

simulator = Processor(N = N_q) # Initialize the simulator using Qutip Processor Class

simulator.add_drift(H_Drift_Qutip, targets = [0, 1]) # Add the Drift Hamiltonian to the Processor

for operators in H_Control_Qutip:
    simulator.add_control(operators, targets = [0, 1]) # Add the Control Hamiltonian to the Processor 

pulses, final_unitary, du_array, cost_fn_array, infidelity_array, energy_array = fc.RunGrapeOptimization(U_Target, H_Drift_Scratch, H_Control_Scratch, Iterations, timespace, # Calculate the optimized pulses
                                 w_f = 1, w_e = 0, Return_Normalized = False, eps_f = 1, eps_e = 100)

for i in range(len(H_Control_Qutip)):
    simulator.pulses[i].coeff = pulses[-1, i] # Pass the pulse amplitudes to the Processor

new_timespace = np.append(timespace, timespace[-1]) # Change timespace for format used in qutip Processor 

simulator.set_all_tlist(new_timespace) # Pass timesteps for the pulses to Processor 

#simulator.add_noise(GaussianNoise)

if Run_Analytical == True: 

    test = simulator.run_analytically() # Run the Simulation on the Processor 

    Unitary_Total = np.eye(4,4) # Initialize Final Unitary 

    for x in test:
        Unitary_Total = x * Unitary_Total # Calculate final Unitary by multiplying all propagators 

    Fidelity = fc.Calculate_Fidelity(U_Target, Unitary_Total) # Calculate Fidelity between the Target and Final Unitary 

    # Print statements 
    print("Target Unitary is: ", U_Target)
    print("Final Unitary is: ", Unitary_Total)
    print("Fidelity is: ", Fidelity)

if Run_Analytical == False:

    Initial_State = basis(4, 2) 

    Initial_Density_Matrix  = Initial_State * Initial_State.dag()

    result = simulator.run_state(init_state = Initial_State)

    densitymatrix  = result.states[-1] * result.states[-1].dag()

    vz.hinton(densitymatrix, xlabels = [r'$\vert 00\rangle$', r'$\vert 01\rangle$', r'$\vert 10\rangle$', r'$\vert 11\rangle$'], 
              ylabels = [r'$\vert 00\rangle$', r'$\vert 01\rangle$', r'$\vert 10\rangle$', r'$\vert 11\rangle$'])
    
    vz.plot_wigner_fock_distribution(densitymatrix)

    print(densitymatrix)

    simulator.plot_pulses()

    plt.show()
