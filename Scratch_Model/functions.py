import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.stats import unitary_group
from scipy.optimize import minimize
from scipy.optimize import Bounds
import random as rd


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
    time = np.linspace(0, Total_Time, Timesteps+1)
    
    H_Total = 0
    U_Total = []

    for i in range(Timesteps-1):
        dt = time[i+1] - time[i]
        H_Total = H_Static
        for j in range(len(H_Control)):
            H_Total += Control_Pulses[i*len(H_Control) + j] * H_Control[j]
            #H_Total += Control_Pulses[j, i] * H_Control[j]
        U = expm(-1j*H_Total*dt)
        U_Total.append(U)
    
    Unitary_Total = np.eye(4,4)
    for x in U_Total:
        Unitary_Total = x @ Unitary_Total

    print("Final Unitary is", Unitary_Total)

    return Unitary_Total

def Calculate_Unitary_2(H_Static, H_Control, Control_Pulses, Timesteps, Total_Time):


    time = np.linspace(0, Total_Time, Timesteps)
    H_Total = 0
    U = np.eye(4,4)
    dt = time[1] - time[0]

    for i in range(Timesteps-1):
        H_Total += H_Static
        for j in range(len(H_Control)):
            H_Total += Control_Pulses[i * len(H_Control) + j] * H_Control[j]
            #H_Total += Control_Pulses[j, i] * H_Control[j]
    

    U = expm(-1j * dt * H_Total)
    
    return U


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
    time = np.linspace(0, Total_Time, Timesteps+1)

    for i in range(Timesteps-1):
        dt = time[i+1] - time[i]
        for j in range(len(H_Control)):
            EC += np.abs(Control_Pulses[i*len(H_Control) + j] * np.linalg.norm(H_Control[j])) * dt
            #EC += np.abs(Control_Pulses[j, i] * np.linalg.norm(H_Control[j])) * dt
        EC += np.linalg.norm(H_Static) * dt
    
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

    print(U_Final)

    Energetic_Cost = CalculateEnergeticCost(Control_Pulses, H_Static, H_Control, Timesteps, Total_Time) # Calculate Energetic Cost of Unitary

    EC_Normalized = Energetic_Cost/(Total_Time*1) # Include Bound of norm of Hamiltonian

    Fidelity = Calculate_Fidelity(U_Target, U_Final) # Calculate Fidelity 

    Cost_Function = (Weight_Fidelity * (1 - Fidelity)) + (Weight_EC * EC_Normalized)

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

    Weight_Fidelity = 1
    Weight_EC = 0

    def Calculate_Cost_Function(Control_Pulses):

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

        EC_Normalized = Energetic_Cost/(Total_Time*1) # Include Bound of norm of Hamiltonian

        Fidelity = Calculate_Fidelity(U_Target, U_Final) # Calculate Fidelity 

        Cost_Function = Weight_Fidelity * (1- Fidelity) + Weight_EC * EC_Normalized

        return Cost_Function
    

    times = np.linspace(0, Total_Time, Timesteps+1) # Define Total Time Space 

    N = len(times)
    K = len(H_Control)
    u = np.zeros((K, N))
 
  
    result = minimize(Calculate_Cost_Function, u, method = 'Nelder-Mead')
    
    return result['fun'], result['x'] 


