import numpy as np 
from qutip import basis, fidelity, identity, sigmax, sigmaz, tensor
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import expand_operator, toffoli
import matplotlib.pyplot as plt 
from qutip.qip.device import Processor
import functions as fc


N_q = 2

Iterations = 500

Timesteps = 500

T = 2 * np.pi

timespace = np.linspace(0, T, Timesteps)

U_Target = fc.Generate_Rand_Unitary(4)
H_Drift_Qutip = np.pi * (tensor(sigmaz(), identity(2)) + tensor(identity(2), sigmaz())) + (1/2) * np.pi * tensor(sigmaz(), sigmaz())

H_Drift_Scratch = np.pi * (fc.tensor(fc.sigmaz(), fc.identity(2)) + fc.tensor(fc.identity(2), fc.sigmaz())) + (1/2) * np.pi * fc.tensor(fc.sigmaz(), fc.sigmaz())

H_Control_Qutip = [tensor(sigmax(), identity(2)),
                   tensor(identity(2), sigmax()),
                   tensor(sigmax(), sigmax())]

H_Control_Scratch = [fc.tensor(fc.sigmax(), fc.identity(2)),
                   fc.tensor(fc.identity(2), fc.sigmax()),
                   fc.tensor(fc.sigmax(), fc.sigmax())]

simulator = Processor(N = N_q)

simulator.add_drift(H_Drift_Qutip, targets = [0, 1])

for operators in H_Control_Qutip:
    simulator.add_control(operators, targets = [0, 1])

pulses, final_unitary, du_array, cost_fn_array, infidelity_array, energy_array = fc.RunGrapeOptimization(U_Target, H_Drift_Scratch, H_Control_Scratch, Iterations, timespace,
                                 w_f = 1, w_e = 0, Return_Normalized = False, eps_f = 1, eps_e = 100)

for i in range(len(H_Control_Qutip)):
    simulator.pulses[i].coeff = pulses[-1, i]

new_timespace = np.append(timespace, timespace[-1])
simulator.set_all_tlist(new_timespace)
simulator.plot_pulses()
plt.show()

test = simulator.run_state(init_state = tensor(identity(2), identity(2)), analytical = True)

Unitary_Total = np.eye(4,4)

for x in test:
    Unitary_Total = x * Unitary_Total

Fidelity = fc.Calculate_Fidelity(U_Target, Unitary_Total)

print("Target Unitary is: ", U_Target)
print("Final Unitary is: ", Unitary_Total)
print("Fidelity is: ", Fidelity)