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

def CreateEnvironment(N_q, H_Drift, H_Control, T_1, T_2, Timesteps):

    """
    Create instance of a Qutip Processor as environment with custom 
    Drift and Control Hamiltonians, T1, and T2 times

    Parameters
    ----------

    N_q : Number of qubits (int)

    H_Drift: Drift Hamiltonian (Qobj)

    H_Control : List of Control Hamiltonians (list(Qobj))

    T_1 : Relaxation time of the Processor (int)

    T_2 : Decoherence time of the Processor (int)

    Timesteps : Number of timesteps to discretize time (int)

    
    Returns
    ----------

    Environment : Instance of Qutip Processor with all parameters defined in input (qutip.qip.device.processor.Processor object)

    """

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

    Environment.set_all_tlist(SimulatorTimespace)

    Environment.add_noise(noise = Noise)

    return Environment

def RunPulses(Environment, InitialState, Pulses):

    """
    Send pulses to a specific qutip processor instance and simulate result 

    Parameters
    ----------

    Environment : Instance of Qutip Processor (qutip.qip.device.processor.Processor object)

    InitialState : Initial state of the quantum state (Qobj)

    Pulses : (K x I_G) Array containing amplitudes of operators in Control Hamiltonian (np.array)


    Returns 
    ----------

    result : Result instance of Environment using specified Initial State and Pulse set (np.array (T x N_q^2))

    """

    for i in range(len(Pulses[:, 0])):
        Environment.pulses[i].coeff = Pulses[i]

    result = Environment.run_state(init_state = InitialState)

    Environment.plot_pulses()

    return result

def CalculateFidelityReward(Result, InitialState, U_Target):

    """
    Calculates Fidelity Reward for a specific Qutip result 

    Parameters
    ----------

    Result : Result instance of Environment using specified Initial State and Pulse set (np.array (T x N_q^2))

    InitialState : InitialState : Initial state of the quantum state (Qobj)

    U_Target : Target Unitary Evolution Operator

    Returns
    ----------

    r_f : Fidelity Reward for a specific Qutip result, Intital State and Target Unitary (float)

    """

    SV_Sim = Result.states[-1]
    DM_Sim = SV_Sim * SV_Sim.dag()

    Qutip_U_Target = Qobj(U_Target)
    SV_Target = Qutip_U_Target * InitialState
    DM_Target = SV_Target * SV_Target.dag()

    r_f = fidelity(DM_Sim, DM_Target)

    return r_f