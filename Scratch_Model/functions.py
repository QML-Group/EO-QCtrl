import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.stats import unitary_group
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import basinhopping
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
                H_Total += Control_Pulses[i*len(H_Control) + j] * H_Control[j] # (H_1(t = 0), H_2(t=0), H_1(t=1), ...)
        U = expm(-1j*H_Total*dt)
        U_Total.append(U)
    
    Unitary_Total = np.eye(4,4)
    for x in U_Total:
        Unitary_Total = x @ Unitary_Total

    return Unitary_Total

def Calculate_Unitary_Manual(H_Static, H_Control, Control_Pulses, Timesteps, Total_Time):

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
            H_Total += Control_Pulses[j, i] * H_Control[j]
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
    Error_Rate = 1- F
    
    return Error_Rate 

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

    H_T = 0
    time = np.linspace(0, Total_Time, Timesteps+1)

    NormalizeValue = Total_Time * (np.linalg.norm(H_Static + np.sum(H_Control)))
    
    for i in range(Timesteps-1):
        dt = time[i+1] - time[i]
        for j in range(len(H_Control)):
            H_T += np.abs(Control_Pulses[i*len(H_Control) + j] * H_Control[j]) * dt
            #EC += np.abs(Control_Pulses[j, i] * np.linalg.norm(H_Control[j])) * dt
        H_T += H_Static * dt
        EC = np.linalg.norm(H_T)
    
    EC_Normalized = EC / NormalizeValue
    
    return EC_Normalized

def Run_Optimizer(U_Target, H_Static, H_Control, Total_Time, Timesteps, Optimization_Method, Weight_F, Weight_EC):

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

        U_Final = Calculate_Unitary(H_Static, H_Control, Control_Pulses, Timesteps, Total_Time) # Calculate Final Unitary g

        Error = Calculate_Fidelity(U_Target, U_Final) # Calculate Fidelity 

        Energetic_Cost = CalculateEnergeticCost(Control_Pulses, H_Static, H_Control, Timesteps, Total_Time)

        Cost_Function = Weight_EC * Energetic_Cost + Weight_F * Error 

        return Cost_Function
    

    times = np.linspace(0, Total_Time, Timesteps+1) # Define Total Time Space 

    N = len(times)
    K = len(H_Control)
    u = np.zeros((K * N))

    result = minimize(Calculate_Cost_Function, u, method = Optimization_Method)
    #result = basinhopping(Calculate_Cost_Function, u)

    Final_Unitary = Calculate_Unitary(H_Static, H_Control, result['x'], Timesteps, Total_Time)

    Energetic_Cost = CalculateEnergeticCost(result['x'], H_Static, H_Control, Timesteps, Total_Time)
    print("Energetic Cost is:", Energetic_Cost)

    return result['fun'], result['x'], Final_Unitary 

def overlap(A, B):
    return np.trace(A.conj().T @ B) / A.shape[0]

def grape_iteration(U_Target, u, r, J, M, U_b_list, U_f_list, H_Control, H_Static, dt, eps_f, eps_e, w_f, w_e):

    """
    Perform one iteration of the GRAPE algorithm 
    and update control pulse parameters.

    Parameters 
    ----------

    U_Target : Target Unitary Evolution Operator

    u : The generated control pulses with shape (iterations, controls, time)

    r : The number of this specific GRAPE iteration

    J : The number of controls in Control Hamiltonian

    M : Number of time steps

    U_b_list : Backward propagators of each time (length M)

    U_f_list : Forward propagators of each time (length M)

    H_Control: Control operators (length J)

    H_Static : Static / Drift Hamiltonian operators

    dt : Timestep size

    eps_f : Distance to move along the gradient when updating controls for Fidelity

    eps_e : Distance to move along the gradient when updating controls for Energy

    w_f : Weight assigned to Fidelity part of the Cost Function

    w_e : Weight assigned to Energy part of the Cost Function
 
    Result 
    ----------

    The updated parameters u[r + 1, :, : ]
    """

    du_list = np.zeros((J, M)) # Initialize array for storing gradient
    max_du_list = np.zeros((J)) # Initialize array for storing maximum of gradient

    for m in range(M - 1): # Loop over all time steps 

        P = U_b_list[m] @ U_Target # Backward propagator storing matmul with target unitary 

        for j in range(J): # Loop over all control operators in H_Control 

            Q = 1j * dt * H_Control[j] @ U_f_list[m] # Forward propagator storing (i * dt * ...)

            du_f = -2 * w_f * overlap(P, Q) * overlap(U_f_list[m], P) # Calculate gradient Fidelity

            denom = H_Static.conj().T @ H_Static + u[r, j, m] * (H_Static.conj().T @ H_Control[j] + H_Control[j].conj().T @ H_Static) # Calculate denominator part gradient Energy

            du_e = 0 # Initialize Energy gradient variable 

            for k in range(J): # Second loop over all control operators in H_Control 

                du_e += -1 * dt * w_e * (np.trace(H_Static.conj().T @ H_Control[j] + H_Control[j].conj().T @ H_Static) + np.trace(H_Control[j].conj().T @ H_Control[k] * (u[r, j, m] + u[r, k, m]))) # Calculate numerator Gradient Energy

                denom += u[r, j, m] * u[r, k, m] * H_Control[j].conj().T @ H_Control[k] # Update denominator Energy Gradient

            du_e /= 2 * np.trace(denom) ** (1/2) # Combine numerator and denominator and take trace + square root (^(1/2))

            du_t = du_f + du_e # Store total gradient (addition of Fidelity and Energy Gradient)

            du_list[j, m] = du_t.real # Update list with all gradients

            max_du_list[j] = np.max(du_list[j]) # Calculate maximum and store in max gradient array

            u[r + 1, j, m] = u[r, j, m] + eps_f * du_f.real + eps_e * du_e.real # Update parameters in next GRAPE iteration using Fidelity and Energy Gradient

    for j in range(J):
        u[r + 1, j, M - 1] = u[r + 1, j, M - 2] 

    return max_du_list



def RunGrapeOptimization():

    grape_iteration()

    pass

