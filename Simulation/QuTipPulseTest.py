import numpy as np 
from qutip import basis, fidelity, identity, sigmax, sigmaz, tensor
from qutip_qip.circuit import QubitCircuit
from qutip_qip.device import OptPulseProcessor
from qutip_qip.operations import expand_operator, toffoli
import qutip_qip
import matplotlib.pyplot as plt 

num_qubits = 2
# Drift Hamiltonian
H_d = 1 * np.pi * (tensor(sigmaz(), identity(2)) + tensor(identity(2), sigmaz())) + (1/2) * np.pi * tensor(sigmaz(), sigmaz())
# The (single) control Hamiltonian
H_c = [tensor(sigmax(), identity(2)),
       tensor(identity(2), sigmax()),
       tensor(sigmax(), sigmax())]
              
processor = OptPulseProcessor(num_qubits, drift=H_d)

for operators in H_c:
    processor.add_control(operators, targets = [0,1])

qc = QubitCircuit(num_qubits)
qc.add_gate("CNOT", targets = 1, controls = 0)

# This method calls optimize_pulse_unitary
tlist, coeffs = processor.load_circuit(
    qc, min_grad=1e-20, init_pulse_type="RND", num_tslots=500,
    evo_time=2*np.pi, verbose=True
)
processor.plot_pulses(
    title="Control pulse for the Hadamard gate", use_control_latex=False
)

rho0 = basis([2, 2], [0, 0])
plus = (basis(2, 0) + basis(2, 1)).unit()
minus = (basis(2, 0) - basis(2, 1)).unit()
result = processor.run_state(init_state=rho0)
#print("Fidelity:", fidelity(result.states[-1], minus))

# add noise
processor.t1 = 40.0
result = processor.run_state(init_state=rho0)
processor.plot_pulses()
plt.show()
#print("Fidelity with qubit relaxation:", fidelity(result.states[-1], minus))
