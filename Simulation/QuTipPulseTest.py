import numpy as np 
from qutip import basis, fidelity, identity, sigmax, sigmaz, tensor
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import expand_operator, toffoli
import matplotlib.pyplot as plt 
from qutip.qip.device import Processor
import functions as fc


N_q = 2 # Define number of qubits

Iterations = 500 # Number of GRAPE Iterations

Timesteps = 500 # Number of Timesteps

T = 2 * np.pi # Total pulse duration

timespace = np.linspace(0, T, Timesteps) # Define time space based on total time and number of timesteps

U_Target = fc.Generate_Rand_Unitary(4) # Define target unitary 

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

test = simulator.run_analytically() # Run the Simulation on the Processor 

Unitary_Total = np.eye(4,4) # Initialize Final Unitary 

for x in test:
    Unitary_Total = x * Unitary_Total # Calculate final Unitary by multiplying all propagators 

Fidelity = fc.Calculate_Fidelity(U_Target, Unitary_Total) # Calculate Fidelity between the Target and Final Unitary 

# Print statements 
print("Target Unitary is: ", U_Target)
print("Final Unitary is: ", Unitary_Total)
print("Fidelity is: ", Fidelity)

# Optional : Plot Control Pulses

#simulator.plot_pulses()
#plt.show()