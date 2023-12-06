import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import expm
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import scipy.sparse as sp
from alive_progress import alive_bar
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

    H_T_Norm = []
    stepsize = Total_Time/Timesteps
    
    for i in range(Timesteps-1):
        H_T = 0
        
        for j in range(len(H_Control)):
            H_T += Control_Pulses[j, i] * H_Control[j] 
       
        #H_T += H_Static  # Optionally include Static Hamiltonian

        H_T_Norm.append(np.linalg.norm(H_T))

    EC = np.sum(H_T_Norm) * stepsize
    return EC

def Run_Scipy_Optimizer(U_Target, H_Static, H_Control, Total_Time, Timesteps, Optimization_Method, Weight_F, Weight_EC):

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

        U_Final = Calculate_Unitary_Scipy(H_Static, H_Control, Control_Pulses, Timesteps, Total_Time) # Calculate Final Unitary g

        Error = Calculate_Fidelity(U_Target, U_Final) # Calculate Fidelity 

        Energetic_Cost = CalculateEnergeticCost(Control_Pulses, H_Static, H_Control, Timesteps, Total_Time)

        Cost_Function = Weight_EC * Energetic_Cost + Weight_F * Error 

        return Cost_Function
    

    times = np.linspace(0, Total_Time, Timesteps+1) # Define Total Time Space 

    N = len(times)
    K = len(H_Control)
    u = np.zeros((K * N))

    result = minimize(Calculate_Cost_Function, u, method = Optimization_Method)

    Final_Unitary = Calculate_Unitary_Scipy(H_Static, H_Control, result['x'], Timesteps, Total_Time)

    Energetic_Cost = CalculateEnergeticCost(result['x'], H_Static, H_Control, Timesteps, Total_Time)

    return result['fun'], result['x'], Final_Unitary 

def Grape_Iteration(U_Target, u, r, J, M, U_b_list, U_f_list, H_Control, H_Static, dt, eps_f, eps_e, w_f, w_e):

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
            
            du_e = 0

            for k in range(J): # Second loop over all control operators in H_Control 

                du_e += -1 * dt * w_e * (np.trace(H_Static.conj().T @ H_Control[j] + H_Control[j].conj().T @ H_Static) + np.trace(H_Control[j].conj().T @ H_Control[k] * (u[r, j, m] + u[r, k, m]))) # Calculate numerator Gradient Energy

                denom += u[r, j, m] * u[r, k, m] * H_Control[j].conj().T @ H_Control[k] # Update denominator Energy Gradient

            du_e /= (2 * np.trace(denom) ** (1/2)) # Combine numerator and denominator and take trace + square root (^(1/2))

            du_t = du_f + du_e # Store total gradient (addition of Fidelity and Energy Gradient)

            du_list[j, m] = du_t.real # Update list with all gradients

            max_du_list[j] = np.max(du_list[j]) # Calculate maximum and store in max gradient array

            u[r + 1, j, m] = u[r, j, m] + eps_f * du_f.real + eps_e * du_e.real # Update parameters in next GRAPE iteration using Fidelity and Energy Gradient

    for j in range(J):
        u[r + 1, j, M - 1] = u[r + 1, j, M - 2] 

    return max_du_list

def RunGrapeOptimization(U_Target, H_Static, H_Control, R, times, w_f, w_e, eps_f = None, eps_e = None):

    """
    Run R iterations of GRAPE algorithm to find optimal pulses given any Static and Control Hamiltonian

    Parameters 
    ----------

    U_Target : Target Unitary Evolution Operator

    H_Control: Control operators (length J)

    H_Static : Static / Drift Hamiltonian operators

    R: Number of GRAPE iterations

    times : Time array (0, T, M steps)

    w_f : Weight assigned to Fidelity part of the Cost Function

    w_e : Weight assigned to Energy part of the Cost Function

    eps_f : Distance to move along the gradient when updating controls for Fidelity

    eps_e : Distance to move along the gradient when updating controls for Energy
    
    Result 
    ----------

    u : Optimized control pulses with dimension (iterations, controls, timesteps)

    U_f_list[-1] : Final unitary based on last GRAPE iteration

    du_max_per_iteration : Array containing the max gradient of each control for all GRAPE iterations

    """

    if eps_f is None:
        eps_f = 0.1 * (2 * np.pi) / (times[-1]) # Set eps value

    if eps_e is None:
        eps_e = 0.1 * (2 * np.pi) / (times[-1]) # Set eps value

    M = len(times) # Grab total time steps
    J = len(H_Control) # Grab number of control parameters 

    u = np.zeros((R, J, M)) # Initialize control parameter matrix to all zeros

    du_max_per_iteration = np.zeros((R - 1, J)) # Initialize gradient matrix 

    with alive_bar(R - 1) as bar: # Start Progress Bar

        for r in range(R - 1): # Iterate over all GRAPE Iterations

            bar() # Call progress bar
            dt = times[1] - times[0] # Grab timestep assuming all timesteps to be equal 

            def _H_idx(idx): # Define function to calculate total Hamiltonian
                return H_Static + sum([u[r, j, idx] * H_Control[j] for j in range(J)])

            U_list = [expm(-1j * _H_idx(idx) * dt) for idx in range(M - 1)] # Calculate Unitary from Total Hamiltonian

            U_f_list = [] # Initialize forward propagator matrix
            U_b_list = [] # Initialize backward propagator matrix 

            U_f = sp.eye(*(U_Target.shape)) # Initialize start value
            U_b = sp.eye(*(U_Target.shape)) # Initialize start value

            for n in range(M - 1): # Loop over all timesteps to calculate forward and backward propagators 

                U_f = U_list[n] @ U_f
                U_f_list.append(U_f)

                U_b_list.insert(0, U_b)
                U_b = U_list[M - 2 - n].T.conj() @ U_b

            du_max_per_iteration[r] = Grape_Iteration(U_Target = U_Target, u = u, r = r, J = J, M = M, U_b_list = U_b_list, U_f_list = U_f_list, H_Control = H_Control, H_Static = H_Static,
                                                    dt = dt, eps_f = eps_f, eps_e = eps_e, w_f = w_f, w_e = w_e)
            
    return u, U_f_list[-1], du_max_per_iteration

def Run_GRAPE_Simulation(U_Target, H_Static, H_Control, H_Labels, R, Timesteps, T, w_f, w_e, eps_f, eps_e, Plot_Control_Field = False, Plot_Tomography = False, Plot_du = False):

    """
    Runs GRAPE algorithm and returns the control pulses, final unitary, Fidelity, and Energetic Cost for the Hamiltonian operators in H_Control
    so that the unitary U_target is realized 

    Parameters 
    ----------

    U_Target : Target Unitary Evolution Operator

    H_Control: Control operators (length J)

    H_Static : Static / Drift Hamiltonian operators

    H_Labels : Labels according to the Control operators in H_Control 

    R: Number of GRAPE iterations

    Timesteps : Number of timesteps 

    T : Total time 

    w_f : Weight assigned to Fidelity part of the Cost Function

    w_e : Weight assigned to Energy part of the Cost Function

    Plot_Control_Field : if True : Plot Control Fields 

    Plot_Tomography : if True : Plot Tomography of Target Unitary and Final Unitary Gate after optimization

    Plot_du : if True : plot max gradient over all timesteps per control operator as function of GRAPE iterations
    
    Result 
    ----------

    u : Optimized control pulses with dimension (iterations, controls, timesteps)

    U_f_list[-1] : Final unitary based on last GRAPE iteration

    du_max_per_iteration : Array containing the max gradient of each control for all GRAPE iterations

    F : Fidelity (normalized overlap squared) between Final Unitary and Target Unitary

    EC_Norm : Normalized Energetic Cost of optimized control pulses 

    """

    time = np.linspace(0, T, Timesteps) # Define total time space
    
    Control_Fields, U_Final, du_list = RunGrapeOptimization(U_Target = U_Target, H_Static = H_Static, H_Control = H_Control, R = R, times = time, 
                                                w_f = w_f, w_e = w_e, eps_f = eps_f, eps_e = eps_e) # Run GRAPE Optimization
    
    Fidelity = Calculate_Fidelity(U_Target, U_Final) # Calculate Fidelity

    EC = CalculateEnergeticCost(Control_Fields[-1], H_Static, H_Control, Timesteps, T) # Calculate and store Energetic Cost
    
    if Plot_Control_Field == True: # Plot Control Fields 

        fig, ax = plt.subplots(len(H_Control))

        for i in range(len(H_Control)):

            ax[i].plot(time, Control_Fields[-1, i, :], label = f"{H_Labels[i]}")
            ax[i].set(xlabel = "Time", ylabel = f"{H_Labels[i]}")
        
        plt.subplot_tool()
        plt.show()

    if Plot_Tomography == True: # Plot Tomography of Unitary 

        print("Feature to be added")

    if Plot_du == True: # Plot Gradient 

        iteration_space = np.linspace(1, R - 1, R - 1)

        for i in range(len(H_Control)):
            plt.plot(iteration_space, du_list[:, i], label = f"{H_Labels[i]}")
        plt.axhline(y = 0, color = "black", linestyle = "-")
        plt.xlabel("GRAPE Iteration Number")
        plt.ylabel("Maximum Gradient over Time")
        plt.title("Maximum Gradient over Time vs. GRAPE Iteration Number")
        plt.legend()
        plt.grid()
        plt.show()

    return Control_Fields, U_Final, du_list, Fidelity, EC

