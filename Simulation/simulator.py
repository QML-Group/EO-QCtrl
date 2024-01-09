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

def RunFullSimulation(U_Target, InitialState, H_Drift, H_Drift_Qutip, H_Control, H_Control_Qutip, H_Labels, Iterations, Timesteps, T_1, T_2, w_f, w_e, eps_f, eps_e, Plot_Control_Field = False):


    """
    Run a QuTip Processor using EO-GRAPE optimized pulses 

    Parameters
    ----------

    U_Target : Target Unitary Evolution Operator

    InitialState : Initial State Vector 

    H_Drift : The Drift Hamiltonian describing the Quantum System

    H_Drift_Qutip : The Drift Hamiltonian in Qobj 

    H_Control : List of Control Operators used to control the Quantum System 

    H_Control_Qutip : List of Control Operators in Qobj class 

    H_Labels : Labels of Control Operators in H_Control

    Iterations : Number of EO-GRAPE Iterations

    Timesteps : Number of Timesteps for discretization 

    T_1 : T1 time of Processor in microseconds 
    
    T_2 : T2 time of Processor in microseconds 

    w_f : Weight assigned to Fidelity in EO-GRAPE Algorithm

    w_e : Weight assigmned to Energetic Cost in EO-GRAPE Algorithm

    eps_f : Distance to move along the gradient when updating controls for Fidelity

    eps_e : Distance to move along the gradient when updating controls for Energy

    Plot_Control_Field : True / False

    
    Returns 
    ----------

    SV_Simulated : State vector output of Simulator after solving Master Equation

    DM_Simulated : Density matrix output of Simulator after solving Master Equation

    SV_Theoretical : Theoretical State Vector by applying Matrix Multiplication Unitary 

    DM_Theoretical : Density Matrix by applying Matrix Multiplication Unitary 

    F_Target_Simulated : Fidelity between Density Matrix output of Simulator and Target Density Matrix 

    F_Theory_Simulated : Fidelity between theoretical Density Matrix and Density Matrix output of Simulator 

    """

    N_q = 2

    T = 2 * np.pi

    InitialStateNumpy = np.array(InitialState.full())

    Timespace = np.linspace(0, T, Timesteps)

    SimulatorTimespace = np.append(Timespace, Timespace[-1])

    Noise = RelaxationNoise(t1 = T_1, t2 = T_2, targets = [0, 1])

    simulator = Processor(N = N_q)

    simulator.add_drift(H_Drift_Qutip, targets = [0, 1])

    for operators in H_Control_Qutip:
        simulator.add_control(operators, targets = [0, 1])

    pulses, final_unitary, du_array, cost_fn_array, infidelity_array, energy_array = fc.RunGrapeOptimization(U_Target, H_Drift, H_Control, Iterations, Timespace, 
                                                                                                             w_f, w_e, Return_Normalized = False, eps_f = eps_f, eps_e = eps_e)
    
    for i in range(len(H_Control_Qutip)):
        simulator.pulses[i].coeff = pulses[-1, i]

    simulator.set_all_tlist(SimulatorTimespace)

    simulator.add_noise(Noise)
    
    Init = basis(4, 2)

    result = simulator.run_state(init_state = Init)

    SV_Simulated = result.states[-1]
    DM_Simulated = SV_Simulated * SV_Simulated.dag()
    SV_Theoretical = final_unitary @ InitialStateNumpy
    DM_Theoretical = SV_Theoretical @ SV_Theoretical.conj().T
    SV_Target = U_Target @ InitialStateNumpy
    DM_Target = SV_Target @ SV_Target.conj().T

    F_Target_Simulated = fidelity(DM_Simulated, Qobj(DM_Target))
    F_Theory_Simulated = fidelity(DM_Simulated, Qobj(DM_Theoretical))

    if Plot_Control_Field == True: # Plot Control Fields 

        fig, ax = plt.subplots(len(H_Control))

        for i in range(len(H_Control)):

            ax[i].plot(Timespace, pulses[-1, i, :], label = f"{H_Labels[i]}")
            ax[i].set(xlabel = "Time", ylabel = f"{H_Labels[i]}")
        
        plt.subplot_tool()
        plt.show()

    return SV_Simulated, DM_Simulated, SV_Theoretical, DM_Theoretical, F_Target_Simulated, F_Theory_Simulated

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

SV_Sim, DM_Sim, SV_Th, DM_Th, F_Target_Sim, F_Th_Sim = RunFullSimulation(TargetUnitary, InitState, H_Drift_Numpy, H_Drift_Qutip, H_C, H_C_Qutip, H_L, 500, 500, 100 * 2 * np.pi, 100 * 2 * np.pi, 1, 0, 1, 100, Plot_Control_Field = True)


print(f"Fidelity between Target and Simulated Density Matrix is: {F_Target_Sim * 100} %")

