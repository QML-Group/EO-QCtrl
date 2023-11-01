import matplotlib.pyplot as plt
import numpy as np
from qutip import *
from qutip.control import *
from qutip.control.grape import plot_grape_control_fields, _overlap, grape_unitary_adaptive, cy_grape_unitary
from qutip.ui.progressbar import TextProgressBar
from input import *

""" 

This file contains functions for simulating optimal control problems 
using the built-in QuTip optimal control suite and functions


"""

def CalculateOptimalFieldEnergeticCost(U_Target, H_Static, H_Control, Iterations, Timesteps,  H_Labels, Plot_Control_Field = False, Plot_Tomography = False):

    """
    Calculate Optimal Control Fields and Energetic Cost for the Hamiltonian operators in H_Control so that the unitary U_target is realized

    Parameters 
    ----------

        U_Target : Target unitary evolution operator

        H_Static : Static/Drift term of the Hamiltonian

        H_Control : Control Hamiltonian containing operators that can be tuned in the Hamiltonian via the control fields

        Iterations : Number of GRAPE iterations 

        Timesteps : Number of timesteps 'N' for time discretization 

        H_Labels : Labels for plotting control fields and process tomography

        Plot_Control_Field : if True : Plot Control Fields 

        Plot_Tomography : if True : Plot Tomography of Target Unitary and Final Unitary Gate after optimization

    Returns 
    ----------

        Energetic_Cost : Energetic Cost of the gate using the optimized control fields 

        Fidelity : Fidelity / overlap between the GRAPE optimized unitary after time 'T' and the target unitary 'U_target' 

    """

    time = np.linspace(0, T, Timesteps) # Define total time space

    eps = 2 * np.pi * 1 # Termination value

    u0 = np.array([np.random.rand(len(time)) * 2 * np.pi * 0.05 for _ in range(len(H_Control))]) # Initialize starting control field

    u0 = [np.convolve(np.ones(10)/10, u0[idx, :], mode = 'same') for idx in range(len(H_Control))] # Initialize starting control field

    result = cy_grape_unitary(U = U_Target, H0 = H_Static, H_ops = H_Control, # Run GRAPE Algorithm
                              R = Iterations, u_start = None, times = time, 
                              eps = eps, phase_sensitive=False, progress_bar=TextProgressBar()) 
    
    Control_Fields = result.u # Store Control Fields 

    Final_Control_Fields = result.U_f # Store Final Control Fields 

    print(Control_Fields[Iterations-1])

    if Plot_Control_Field == True: # Plot Control Fields if set to 'True'

        plot_grape_control_fields(time, Control_Fields / (2 * np.pi), H_Labels, uniform_axes=True)
        plt.show()

    if Plot_Tomography == True: # Plot Process Tomography of Target and Final Untiarty if set to 'True'

        op_basis = [[qeye(2), sigmax(), sigmay(), sigmaz()]] * 2
        op_label = [["I", "X", "Y", "Z"]] * 2      
        U_i_s = to_super(U_Target)
        U_f_s = to_super(Final_Control_Fields)
        chi_1 = qpt(U_i_s, op_basis)
        chi_2 = qpt(U_f_s, op_basis)


        fig_1 = plt.figure(figsize = (6,6))
        fig_1 = qpt_plot_combined(chi_1, op_label, fig=fig_1, threshold=0.001, title = 'Target Unitary Gate ')

        fig_2 = plt.figure(figsize = (6, 6))
        fig_2 = qpt_plot_combined(chi_2, op_label, fig = fig_2, threshold = 0.001, title = 'Final Unitary after Optimization')

        plt.show()
       
    Fidelity = abs(_overlap(U_Target, result.U_f)) ** 2 # Compute Fidelity (absolute overlap squared)

    stepsize = max(time)/len(time) # Define stepsize 
    
    Energetic_Cost = 0 # Initialize Energetic Cost Variable to 0 
   
    Energetic_Cost_List = [] # Initialiaze Energetic Cost List to empty
    
    # Calculate Energetic Cost of Entire Unitary (loop over every timestep and H_Control Terms)

    for i in range(len(time)):
        for j in range(len(H_Control)):
            Energetic_Cost += np.abs(Control_Fields[Iterations-1, j, i] * np.linalg.norm(H_Control[j])) * stepsize
        Energetic_Cost += np.linalg.norm(H_Static) * stepsize
        Energetic_Cost_List.append(Energetic_Cost) 

    return Energetic_Cost, Fidelity