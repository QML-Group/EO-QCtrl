import matplotlib.pyplot as plt
import time
import numpy as np
from qutip import *
from qutip.control import *
from qutip.qip.operations import cnot
from qutip.qip.operations import cphase
from qutip.control.grape import plot_grape_control_fields, _overlap, grape_unitary_adaptive, cy_grape_unitary
from scipy.interpolate import interp1d
from qutip.ui.progressbar import TextProgressBar

# Input Variables

T = 2 * np.pi # Total gate time

Iterations_1 = 10 # Total number of GRAPE iterations

H_Static_1 = 0 * np.pi * (tensor(sigmax(), identity(2))) + tensor(identity(2), sigmax()) # Static Drift Hamiltonian

U_target_1 = cnot() # CNOT Gate 

H_Control_1 =  [tensor(sigmax(), identity(2)), 
         tensor(sigmay(), identity(2)),
         tensor(sigmaz(), identity(2)),
         tensor(identity(2), sigmax()),
         tensor(identity(2), sigmay()),
         tensor(identity(2), sigmaz()),
         tensor(sigmax(), sigmax()) +
         tensor(sigmay(), sigmay()) +
         tensor(sigmaz(), sigmaz())] # General 2 qubit Hamiltonian with X, Y, and Z interaction terms 

H_Labels_1 = [r'$u_{1x}$', r'$u_{1y}$', r'$u_{1z}$',
            r'$u_{2x}$', r'$u_{2y}$', r'$u_{2z}$',
            r'$u_{xx}$',
            r'$u_{yy}$',
            r'$u_{zz}$'
            ] # Labels for H_Control_1 (optional for plotting)

def CalculateOptimalFieldEnergeticCost(U_Target, H_Static, H_Control, Iterations, Timesteps):

    """
    Calculate Optimal Control Fields and Energetic Cost for the Hamiltonian operators in H_Control so that the unitary U_target is realized

    Parameters 

    ----------

    U_Target : Target unitary evolution operator

    H_Static : Static/Drift term of the Hamiltonian

    H_Control : Control Hamiltonian containing operators that can be tuned in the Hamiltonian via the control fields

    Iterations : Number of GRAPE iterations 

    Timesteps : Number of timesteps 'N' for time discretization 

    Returns 

    ----------

    Energetic_Cost : Energetic Cost of the gate using the optimized control fields 

    Fidelity : Fidelity / overlap between the GRAPE optimized unitary after time 'T' and the target unitary 'U_target' 

    """

    time = np.linspace(0, T, Timesteps) # Define total time space

    eps = 2 * np.pi * 1 # Termination value

    u0 = np.array([np.random.rand(len(time)) * 2 * np.pi * 0.05 for _ in range(len(H_Control))]) # Initialize starting control field 

    u0 = [np.convolve(np.ones(10)/10, u0[idx, :], mode = 'same') for idx in range(len(H_Control))] # Initialize starting control field 

    result = cy_grape_unitary(U = U_Target, H0 = H_Static, H_ops = H_Control, R = Iterations, u_start = u0 , times = time, eps = eps, phase_sensitive=False, progress_bar=TextProgressBar()) # Run GRAPE Algorithm
    
    Control_Fields = result.u # Store Control Fields 

    Final_Control_Fields = result.U_f # Store Final Control Fields 

    Fidelity = process_fidelity(U_Target, Final_Control_Fields) # Calculate Fidelity Between U_Target and U_f

    print(f"Process Fidelity is: {Fidelity}")

    pass 


