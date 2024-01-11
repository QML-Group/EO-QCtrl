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

def CreateEnvironment(N_q, H_Drift, H_Control, Pulses, T_1, T_2, Timesteps):

    T = 2 * np.pi
    Timespace = np.linspace(0, T, Timesteps)
    SimulatorTimespace = np.append(Timespace, Timespace[-1])

    targets = []
    for i in range(N_q):
        targets.append(i)
    
    Noise = RelaxationNoise(t1 = T_1, t2 = T_2)

    Environment = Processor(N = N_q)

    Environment.add_drift(H_Drift, targets = targets)

    for operator in H_Control:
        Environment.add_control(operator, targets = targets)

    for i in range(len(H_Control)):
        Environment.pulses[i].coeff = Pulses[i]
    
    Environment.set_all_tlist(SimulatorTimespace)

    Environment.add_noise(noise = Noise)

    return Environment

def RunPulses(Environment, InitialState, Pulses):

    for i in range(len(Pulses[0, :])):
        Environment.pulses[i].coeff = Pulses[i]

    result = Environment.run_state(init_state = InitialState)

    return result

def CalculateFidelityReward(Result, InitialState, U_Target):

    SV_Sim = Result.states[-1]
    DM_Sim = SV_Sim * SV_Sim.dag()

    Qutip_U_Target = Qobj(U_Target)
    SV_Target = Qutip_U_Target * InitialState
    DM_Target = SV_Target * SV_Target.dag()

    r_f = fidelity(DM_Sim, DM_Target)

    return r_f

def RunGRAPESimulation(U_Target, InitialState, H_Drift, H_Drift_Qutip, H_Control, H_Control_Qutip, H_Labels, Iterations, Timesteps, T_1, T_2, w_f, w_e, eps_f, eps_e, Plot_Control_Field = False, Plot_Wigner_Function = False):


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

    if Plot_Wigner_Function == True: # Plot Density Matrix and Wigner Function if True 

        vz.hinton(DM_Simulated)
        vz.plot_wigner_fock_distribution(DM_Simulated)
        plt.show()

    return SV_Simulated, DM_Simulated, SV_Theoretical, DM_Theoretical, F_Target_Simulated, F_Theory_Simulated