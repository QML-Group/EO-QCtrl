import numpy as np 
from qutip import basis, fidelity, identity, sigmax, sigmaz, tensor, destroy
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import expand_operator, toffoli, snot
import matplotlib.pyplot as plt 
from qutip.qip.device import Processor
import functions as fc
import qutip.visualization as vz
from qutip.qip.noise import RelaxationNoise
from qutip.metrics import fidelity
from qutip import Qobj
import simulator as sim 

H_Drift_Numpy = np.pi * (fc.tensor(fc.sigmaz(), fc.identity(2)) + fc.tensor(fc.identity(2), fc.sigmaz())) + (1/2) * np.pi * fc.tensor(fc.sigmaz(), fc.sigmaz())

H_Drift_Qutip = np.pi * (tensor(sigmaz(), identity(2)) + tensor(identity(2), sigmaz())) + (1/2) * np.pi * tensor(sigmaz(), sigmaz()) # Define Drift Hamiltonian used in "Processor"

H_C = [fc.tensor(fc.sigmax(), fc.identity(2)), 
       fc.tensor(fc.identity(2), fc.sigmax()),
       fc.tensor(fc.sigmax(), fc.sigmax())]

H_C_Qutip = [tensor(sigmax(), identity(2)), # Define Control Hamiltonian used in "Processor"
            tensor(identity(2), sigmax()),
            tensor(sigmax(), sigmax())]

H_L = [r'$u_{1x}$', 
       r'$u_{2x}$', 
       r'$u_{xx}$'] 

TargetUnitary = fc.cnot()

InitState = basis(4, 2)

GRAPEIterations = 500

N_t = 500

T1 = 2 * np.pi * 100 

T2 = 2 * np.pi * 100

weight_fidelity = 1

weight_energy  = 0

eps_fidelity = 1 

eps_energy = 100


SV_Sim, DM_Sim, SV_Th, DM_Th, F_Target_Sim, F_Th_Sim = sim.RunGRAPESimulation(TargetUnitary, 
                                                                         InitState, 
                                                                         H_Drift_Numpy, 
                                                                         H_Drift_Qutip, 
                                                                         H_C, 
                                                                         H_C_Qutip, 
                                                                         H_L, 
                                                                         GRAPEIterations, N_t, 
                                                                         T1, T2, 
                                                                         weight_fidelity, weight_energy, 
                                                                         eps_fidelity, eps_energy, 
                                                                         Plot_Control_Field = True,
                                                                         Plot_Wigner_Function = True)


print(f"Fidelity between Target and Simulated Density Matrix is: {F_Target_Sim * 100} %")

