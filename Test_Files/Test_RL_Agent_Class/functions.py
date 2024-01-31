import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import expm
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import scipy.sparse as sp
from alive_progress import alive_bar
from scipy.stats import unitary_group
from qutip import to_super, qpt, qpt_plot_combined, Qobj
from qutip import rand_unitary


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
                     [0, -1]])

def cnot():
    """
    Returns CNOT Unitary Gate 
    """

    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])

def Generate_Rand_Unitary(N):
    """
    Returns N-Dimenstional Random Unitary 
    """

    x = rand_unitary(N)
    y = x.full()
    nparray = np.array(y)

    return nparray

def overlap(A, B):
    return np.trace(A.conj().T @ B) / A.shape[0]

def Calculate_Unitary_Scipy(H_Static, H_Control, Control_Pulses, Timesteps, Total_Time):

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

    time = np.linspace(0, Total_Time, Timesteps+1)
    
    H_Total = 0
    U_Total = []

    for i in range(Timesteps-1):
        dt = time[i+1] - time[i]
        H_Total = H_Static
        for j in range(len(H_Control)):
                H_Total += Control_Pulses[i*len(H_Control) + j] * H_Control[j] # (H_1(t = 0), H_2(t=0), H_1(t=1), ...)
        U = expm(-1j*H_Total*dt)
        U_Total.append(U)
    
    Unitary_Total = np.eye(4,4)
    for x in U_Total:
        Unitary_Total = x @ Unitary_Total

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

def CalculateEnergeticCost(Control_Pulses, H_Static, H_Control, Timesteps, Total_Time, Return_Normalized = False):

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

    H_T_Norm = []
    stepsize = Total_Time/Timesteps
    
    for i in range(Timesteps-1):
        H_T = 0
        
        for j in range(len(H_Control)):
            H_T += Control_Pulses[j, i] * H_Control[j] 
       
        #H_T += H_Static  # Optionally include Static Hamiltonian

        H_T_Norm.append(np.linalg.norm(H_T))

    EC = np.sum(H_T_Norm) * stepsize

    EC_Normalized = EC / (Total_Time * np.linalg.norm(np.sum(H_Control)))

    if Return_Normalized == True:

        Value = EC_Normalized

    elif Return_Normalized == False:

        Value = EC

    return Value

def convert_qutip_to_numpy(operator):

    data = operator.full()
    array = np.array(data)

    return array 

def convert_qutip_list_to_numpy(operator_list):

    new_list = []

    for operator in operator_list:
        new_list.append(convert_qutip_to_numpy(operator))

    return new_list