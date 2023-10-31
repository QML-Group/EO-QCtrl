import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.stats import unitary_group

"""

This File Contains Functions that are used in the "input.py" and "main.py" files to calculate and simulate
Quantum Optimal Control problems in python

-----------------

Work in Progress

"""

def tensor(a, b):
    """
    Returns tensor product between two matrices
    """

    return np.kron(a, b)

def identity(N):
    """
    Returns Identity Matrix with Dimension 'N'
    """

    return np.identity(N)

def sigmax():
    """
    Returns Pauli-x Matrix 
    """

    return np.array([[0,1],
                     [1,0]])

def sigmay():
    """
    Returns Pauli-y Matrix 
    """

    return np.array([[0, -1j],
                     [1j, 0]])

def sigmaz():
    """
    Returns Pauli-z Matrix 
    """

    return np.array([[1, 0],
                     [0, 1]])

def cnot():
    """
    Returns CNOT Unitary Gate 
    """

    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])

def rand_unitary(N):
    """
    Returns N-Dimenstional Random Unitary 
    """

    x = unitary_group.rvs(N)
    y = np.dot(x, x.conj().T)

    return y

def Calculate_Unitary(H_Static, H_Control, Control_Pulses, Timesteps, Total_Time):

    """
    Calculates Unitary based on Static Hamiltonian, Control Hamiltonian, and control parameters

    Parameters
    ----------

    H_Static : Static/Drift Hamiltonian Term

    H_Control : Control Hamiltonian containing operators that can be tuned in the Hamiltonian via the control fields 

    Control_Pulses : The Control Parameters for each term in "H_Control"

    Timesteps : Number of timesteps 'N for time discretization

    Total_Time : Total Unitary Gate Time

    Returns 
    ----------

    Unitary_Total : Unitary Gate based on input parameters 

    """
    time = np.linspace(0, Total_Time, Timesteps)
    dt = Total_Time/Timesteps
    H_Total = 0
    U_Total = []

    for i in range(len(time)):
        for j in range(len(H_Control)):
            H_Total = H_Static
            H_Total += Control_Pulses[j, i] * H_Control[j]
        U = expm(-1*j*H_Total*dt)
        U_Total.append(U)
    
    Unitary_Total = 1
    for x in U_Total:
        Unitary_Total = Unitary_Total * x

    return Unitary_Total

def Calculate_Fidelity(U_Target, U):

    """
    Calculate Fidelity Between Target Unitary and other Unitary U

    Parameters 
    ----------

    U_Target : Target Unitary Gate 

    U: Unitary Gate to Calculate Fidelity of 

    Returns 
    ----------

    F: Fidelity between U_Target and U

    """

    F = abs(np.trace(U_Target.conj().T @ U)/np.trace(U_Target.conj().T @ U_Target))**2

    return F

def CalculateEnergeticCost(Control_Pulses, H_Static, H_Control, Timesteps, Total_Time):

    """
    Calculate Energetic Cost of certain Unitary 

    Parameters
    ----------

    Control_Pulses : The Control Parameters for each term in "H_Control"

    H_Static : Static/Drift Hamiltonian Term 

    H_Control : Control Hamiltonoian containing operators that can be tuned in the Hamiltonian via the control fields 

    Timesteps : Number of timesteps 'N' for time discretization

    Total_Time : Total time of unitary gate

    Returns 
    ----------

    EC : Energetic Cost of the Control Pulses based on the static and drift Hamiltonian 
    """

    EC = 0

    stepsize = Total_Time / Timesteps

    for i in range(Timesteps):
        for j in range(len(H_Control)):
            EC += np.abs(Control_Pulses[j, i] * np.linalg.norm(H_Control[j])) * stepsize
        EC += np.linalg.norm(H_Static) * stepsize 
    
    return EC

def Calculate_Cost_Function(Control_Pulses, Weight_Fidelity, Weight_EC, U_Target, H_Static, H_Control, Timesteps, Total_Time):

    """
    Calculate Cost Function of certain control pulses 


    Parameters 
    ----------

    Control_Pulses : The Control Parameters for each term in "H_Control"

    Weight_Fidelity: Weight given to "Fidelity" in Cost Function

    Weight_EC: Weight given to "Energetic Cost" in Cost Function

    U_Target : Target Unitary 

    H_Static : Static/Drift Hamiltonian Term 

    H_Control : Control Hamiltonoian containing operators that can be tuned in the Hamiltonian via the control fields 

    Timesteps : Number of timesteps 'N' for time discretization

    Total_Time : Total Unitary Gate Time

    Returns 
    ----------

    C: Value of the cost-function based on the control pulses provided as input

    """

    U_Final = Calculate_Unitary(H_Static, H_Control, Control_Pulses, Timesteps, Total_Time) # Calculate Final Unitary 

    Energetic_Cost = CalculateEnergeticCost(Control_Pulses, H_Static, H_Control, Timesteps, Total_Time) # Calculate Energetic Cost of Unitary

    EC_Normalized = Energetic_Cost/100

    Fidelity = Calculate_Fidelity(U_Target, U_Final) # Calculate Fidelity 

    Cost_Function = (Weight_Fidelity * Fidelity) + (Weight_EC * EC_Normalized)

    return Cost_Function

def Run_Optimizer(U_Target, H_Static, H_Control, Iterations, Total_Time, Timesteps, U_Start = None):

    """
    This Function Implements an Optimization algorithmn using NumPy and SciPy in Python
    Calculates control pulses for the Hamitonian operators in H_Control so that the unitary U_Target is realized.

    Parameters 
    ----------

    U_Target : Target Unitary Evolution Operator

    H_Static : Static/Drift Hamiltonian Term

    H_Control : Control Hamiltonian containing operators that can be tuned in the Hamiltonian via the control fields

    Iterations : Number of GRAPE iterations 

    Total_Time : Total Unitary Gate Time

    Timesteps : Number of timesteps 'N' for time discretization

    U_Start : Optional array with initial control pulse values
    

    Returns 
    ----------
        
    TBD

    """

    times = np.linspace(0, Total_Time, Timesteps) # Define Total Time Space 

    N = len(times)
    K = len(H_Control)

    u = np.zeros((K, N))

    u_limits = [-1, +1]

    u_start = None

    pass
