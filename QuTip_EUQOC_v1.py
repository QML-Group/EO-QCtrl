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


""" INPUT VARIABLES """


T = 2 * np.pi # Total gate time

Iterations_1 = 100 # Total number of GRAPE iterations

Timesteps = 500 # Total number of timesteps to discretize the time space

H_Static_1 = 0 * np.pi * (tensor(sigmax(), identity(2)) + tensor(identity(2), sigmax())) # Static Drift Hamiltonian 1

H_Static_2 = 0 * np.pi * (tensor(sigmaz(), identity(2)) + tensor(identity(2), sigmaz())) # Static Drift Hamiltonian 2

U_target_CNOT = cnot() # CNOT Gate 

U_target_rand = rand_unitary(4) # Generate Random Unitary 

H_Control_1 =  [tensor(sigmax(), identity(2)), 
         tensor(sigmay(), identity(2)),
         tensor(sigmaz(), identity(2)),
         tensor(identity(2), sigmax()),
         tensor(identity(2), sigmay()),
         tensor(identity(2), sigmaz()),
         tensor(sigmax(), sigmax()) +
         tensor(sigmay(), sigmay()) +
         tensor(sigmaz(), sigmaz())] # General 2 qubit Hamiltonian with X, Y, and Z interaction terms 

H_Control_2 = [tensor(sigmaz(), identity(2)),
               tensor(identity(2), sigmax()),
               tensor(identity(2), sigmay()),
               tensor(identity(2), sigmaz()),
               tensor(sigmaz(), sigmax())
               ] # Control Hamiltonian 2: no sigma_y terms 

H_Control_3 = [tensor(sigmax(), sigmax()), tensor(sigmay(), sigmay()), tensor(sigmaz(), sigmaz())]

H_Control_4 = [tensor(sigmaz(), identity(2)), 
               tensor(identity(2), sigmaz()),
               tensor(sigmax(), identity(2)),
               tensor(identity(2), sigmax()), 
               tensor(sigmax(), sigmax()) + 
               tensor(sigmay(), sigmay()) + 
               tensor(sigmaz(), sigmaz())]

H_Labels_1 = [r'$u_{1x}$', r'$u_{1y}$', r'$u_{1z}$',
            r'$u_{2x}$', r'$u_{2y}$', r'$u_{2z}$',
            r'$u_{xx}$',
            r'$u_{yy}$',
            r'$u_{zz}$'
            ] # Labels for H_Control_1 (optional for plotting)

H_Labels_2 = [r'$u_{1z}$', r'$u_{2x}$', r'$u_{2y}$', r'$u_{2z}$', r'$u_{zx}$']

H_Labels_3 = [r'$u_{xx}$', r'$u_{yy}$', r'$u_{zz}$']

H_Labels_4 = [r'$u_{1z}$', r'$u_{2z}$', r'$u_{1x}$', r'$u_{2x}$', r'$u_{xx}$']


""" FUNCTIONS """


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

    result = cy_grape_unitary(U = U_Target, H0 = H_Static, H_ops = H_Control, 
                              R = Iterations, u_start = u0 , times = time, 
                              eps = eps, phase_sensitive=False, progress_bar=TextProgressBar()) # Run GRAPE Algorithm
    
    Control_Fields = result.u # Store Control Fields 

    Final_Control_Fields = result.U_f # Store Final Control Fields 

    if Plot_Control_Field == True: # Plot Control Fields if set to 'True'

        plot_grape_control_fields(time, Control_Fields / (2 * np.pi), H_Labels, uniform_axes=True)
        plt.show()

    if Plot_Tomography == True: # Plot Process Tomography of Target and Final Untiarty if set to 'True'

        op_basis = [[qeye(2), sigmax(), sigmay(), sigmaz()]] * 2
        op_label = [["i", "x", "y", "z"]] * 2      
        U_i_s = to_super(U_Target)
        U_f_s = to_super(Final_Control_Fields)
        chi_1 = qpt(U_i_s, op_basis)
        chi_2 = qpt(U_f_s, op_basis)


        fig_1 = plt.figure(figsize = (6,6))
        fig_1 = qpt_plot_combined(chi_1, op_label, fig=fig_1, threshold=0.001, title = 'Target Unitary Gate ')

        fig_2 = plt.figure(figsize = (6, 6))
        fig_2 = qpt_plot_combined(chi_2, op_label, fig = fig_2, threshold = 0.001, title = 'Final Unitary after Optimization')

        plt.show()
       
    Fidelity = abs(_overlap(U_Target, result.U_f)) ** 2

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


""" TESTING AND CALCULATIONS """


EC, F = CalculateOptimalFieldEnergeticCost(U_target_CNOT, H_Static_1, H_Control_1, Iterations_1, Timesteps, H_Labels_1, Plot_Control_Field = True, Plot_Tomography = True)

Output = f"""

**** PROGRAM OUTPUT & RESULTS ****

----------

    Random Quantum Unitary: 

    {U_target_rand}

    ----------

    Optimal Fidelity: {F}

    ----------

    Energetic Cost: {EC}

    ----------

    Number of GRAPE Iterations: {Iterations_1}

    ----------

    Number of Timesteps: {Timesteps}
"""

print(Output)

#%%
#GRAPE_Iterations = np.arange(10, 110, 10)
#Timestep_Iterations = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300, 400, 500]
#print(Timestep_Iterations)
#EC_List = []
#F_List = []
#
#for i in Timestep_Iterations:
#    EC_test, F_test = CalculateOptimalFieldEnergeticCost(U_target_rand, H_Static_1, H_Control_1, Iterations_1, i)
#    EC_List.append(EC_test)
#   F_List.append(F_test)
#    print(EC_List)
#   print(F_List)
#
#fig, ax1 = plt.subplots()
#
#color = 'tab:red'
#ax1.set_xlabel('Number of Timestep Iterations')
#ax1.set_ylabel('Process Fidelity', color=color)
#lns1 = ax1.plot(Timestep_Iterations, F_List, color=color, label='Process Fidelity', linestyle = '-', marker = 'd')
#ax1.tick_params(axis='y', labelcolor=color)
#ax2 = ax1.twinx()  
#
#color = 'tab:blue'
#ax2.set_ylabel('Energetic Cost (a.u.)', color=color) 
#lns2 = ax2.plot(Timestep_Iterations, EC_List, color=color, label='Energetic Cost', linestyle = '-', marker = 'd')
#ax2.tick_params(axis='y', labelcolor=color)
#
#fig.tight_layout() 
#lns = lns1+lns2
#labs = [l.get_label() for l in lns]
#ax1.legend(lns, labs, loc=0)
#plt.title('Process Fidelity and Energetic Cost of Random Unitary versus GRAPE Iterations')
#plt.show()