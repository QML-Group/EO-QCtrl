import matplotlib.pyplot as plt
import numpy as np
from qutip import *
#from qutip.control import *
#from qutip.control.grape import plot_grape_control_fields, _overlap, grape_unitary_adaptive, cy_grape_unitary
#from qutip.control.grape import _overlap
from Grape_Test import *
from qutip.ui.progressbar import TextProgressBar
from input import *

""" 

This file contains functions for simulating optimal control problems 
using the built-in QuTip optimal control suite and functions


"""

def _overlap(A, B):
    return (A.dag() * B).tr() / A.shape[0]

def CalculateOptimalFieldEnergeticCost(U_Target, H_Static, H_Control, Iterations, Timesteps,  H_Labels, weight_ec, weight_fidelity, Use_Rand_u0 = False, Plot_Control_Field = False, Plot_Tomography = False, Plot_du = False):

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

    eps_f = 2 * np.pi * 1 # GRAPE step size

    eps_e =  2 * np.pi * 1 # GRAPE step size

    if Use_Rand_u0 == True:

        #u0 = np.array([np.random.rand(len(time)) * 2 * np.pi * 0.05 for _ in range(len(H_Control))]) # Initialize starting control field
        #u0 = [np.convolve(np.ones(10)/10, u0[idx, :], mode = 'same') for idx in range(len(H_Control))] # Initialize starting control field
        u0 = np.random.uniform(-1, 1, size = (len(H_Control), len(time)))

        result, du_list_per_iteration = cy_grape_unitary(U = U_Target, H0 = H_Static, H_ops = H_Control, # Run GRAPE Algorithm
                                R = Iterations, u_start = u0, times = time, 
                                eps_f = eps_f, eps_e = eps_e, weight_ec = weight_ec, weight_fidelity = weight_fidelity, phase_sensitive=False, progress_bar=TextProgressBar()) 
    
    if Use_Rand_u0 == False: 
        result, du_list_per_iteration = cy_grape_unitary(U = U_Target, H0 = H_Static, H_ops = H_Control, # Run GRAPE Algorithm
                                R = Iterations, u_start = None, times = time, 
                                eps_f = eps_f, eps_e = eps_e, weight_ec = weight_ec, weight_fidelity = weight_fidelity, phase_sensitive=False, progress_bar=TextProgressBar())

    Control_Fields = result.u # Store Control Fields 

    Final_Control_Fields = result.U_f # Store Final Control Fields 

    #print(Control_Fields[Iterations-1])

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

    if Plot_du == True:
        iteration_space = np.linspace(1, Iterations - 1, Iterations - 1)
        
        for i in range(len(H_Control)):
            plt.plot(iteration_space, du_list_per_iteration[:, i], label = f"{H_Labels[i]}")
        plt.axhline(y = 0, color = 'black', linestyle = '-')
        plt.xlabel("GRAPE Iteration Number")
        plt.ylabel("Maximum Gradient over Time")
        plt.title("Maximum Gradient Over Time vs. GRAPE Iteration Number")
        plt.legend()
        plt.grid()
        plt.show()
        
    Fidelity = abs(_overlap(U_Target, result.U_f)) ** 2 # Compute Fidelity (absolute overlap squared)
    
    stepsize = max(time)/len(time) # Define stepsize 
    
    Energetic_Cost = 0 # Initialize Energetic Cost Variable to 0 
   
    Energetic_Cost_List = [] # Initialiaze Energetic Cost List to empty

    # Calculate Energetic Cost of Entire Unitary (loop over every timestep and H_Control Terms)

    for i in range(len(time)):

        Total_Hamiltonian = 0

        for j in range(len(H_Control)):

            Total_Hamiltonian += Control_Fields[-1, j, i] * H_Control[j]

            #Energetic_Cost += np.abs(Control_Fields[Iterations-1, j, i] * np.linalg.norm(H_Control[j])) * stepsize

        #Total_Hamiltonian += H_Static

        Energetic_Cost_List.append(np.linalg.norm(Total_Hamiltonian))

    Energetic_Cost = np.sum(Energetic_Cost_List) * stepsize

        #Energetic_Cost += np.linalg.norm(H_Static) * stepsize
        #Energetic_Cost_List.append(Energetic_Cost) 

    return result.U_f, Energetic_Cost, Fidelity, du_list_per_iteration
