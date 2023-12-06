import matplotlib.pyplot as plt
import numpy as np
from qutip import *
from qutip.ui.progressbar import TextProgressBar
from scipy.interpolate import interp1d
import scipy.sparse as sp
from qutip.qobj import Qobj
from qutip.ui.progressbar import BaseProgressBar
from qutip.control.cy_grape import cy_overlap
from qutip.qip.operations import gate_sequence_product


""" 

This file contains functions for simulating optimal control problems 
using the built-in QuTip optimal control suite and functions


"""

class GRAPEResult:
    """
    Class for representing the result of a GRAPE simulation.

    Attributes
    ----------
    u : array
        GRAPE control pulse matrix.

    H_t : time-dependent Hamiltonian
        The time-dependent Hamiltonian that realize the GRAPE pulse sequence.

    U_f : Qobj
        The final unitary transformation that is realized by the evolution
        of the system with the GRAPE generated pulse sequences.
    """
    def __init__(self, u=None, H_t=None, U_f=None):

        self.u = u
        self.H_t = H_t
        self.U_f = U_f

def cy_grape_inner(U, u, r, J, M, U_b_list, U_f_list, H_ops, dt, eps_f, eps_e, weight_ec, weight_fidelity, H_Static, H_Control, alpha, beta, phase_sensitive, use_u_limits, u_min, u_max):
    
    """
    Perform one iteration of GRAPE control pulse
    updates.

    Parameters
    ----------
    U : :class:`qutip.data.Data`
        The target unitary.

    u : np.ndarray
        The generated control pulses. It's shape
        is (iterations, controls, times), i.e.
        (R, J, M). The result of this iteration
        is stored in u[r, :, :].

    r : int
        The number of this GRAPE iteration.

    J : int
        The number of controls in the Hamiltonian.

    M : int
        The number of times.

    U_b_list : list of :class:`qutip.data.Data`
        The backward propagators for each time.
        The list has length M.

    U_f_list : list of :class:`qutip.data.Data`
        The forward propagators for each time.
        The list has length M.

    H_ops : list of :class:`qutip.data.Data`
        The control operators from the Hamiltonian.
        The list has length J.

    dt : float
        The time step.

    eps : float
        The distance to move along the gradient when updating
        the controls.

    Result
    ------
    The results are stored in u[r + 1, : , :].
    """

    du_list = np.zeros((J, M))
    avg_du_list = np.zeros((J))
    max_du_list = np.zeros((J))
    
    for m in range(M - 1): # Loop over all time steps 
        P = U_b_list[m] @ U # Backward propagator storing mat multiplication with target unitary 
        
        for j in range(J): # Loop over all control fields 
            Q = 1j * dt * H_ops[j] @ U_f_list[m] # Forward propagator storing (i * dt * )
            
            du_f = -2 * weight_fidelity * cy_overlap(P, Q) * cy_overlap(U_f_list[m], P) # Calculate Gradient Fidelity
            
            #du += -(np.sqrt(7)/3) * weight_ec * dt * (u[r, j, m] / ((3*np.pi/2 + u[r, 0, m]**2 + u[r, 1, m]**2 + u[r, 2, m]**2)))  # Calculate Gradient Energetic Cost

            #du += -2 * weight_ec * dt * u[r, j, m] * (H_Control[j].dag() * H_Control[j]).tr()
            
            denom = H_Static.dag() * H_Static + u[r, j, m] * (H_Static.dag() * H_Control[j] + H_Control[j].dag() * H_Static)

            du_e = 0

            for k in range(J):
                
                du_e += -1 * dt * weight_ec * ((H_Static.dag() * H_Control[j] + H_Control[j].dag() * H_Static).tr() + (H_Control[j].dag() * H_Control[k] * (u[r, j, m] + u[r, k, m])).tr())
                
                denom += u[r, j, m] * u[r, k, m] * H_Control[j].dag() * H_Control[k]

            du_e /= 2 * denom.tr()**(1/2)

            du_t = du_f + du_e

            du_list[j, m] = du_t.real

            avg_du_list[j] = np.average(du_list[j])

            max_du_list[j] = np.max(du_list[j])
            
            u[r + 1, j, m] = u[r, j, m] + eps_f * du_f.real + eps_e * du_e.real # Update control pulses according to gradient (gradient * distance to move along gradient)

    for j in range(J):
        u[r + 1, j, M - 1] = u[r + 1, j, M - 2]

    return max_du_list

def cy_grape_unitary(U, H0, H_ops, R, times, weight_ec, weight_fidelity, eps_f=None, eps_e = None, u_start=None,
                     u_limits=None, interp_kind='linear', use_interp=False,
                     alpha=None, beta=None, phase_sensitive=True,
                     progress_bar=BaseProgressBar()):
   
    #if eps_f is None:
    #    eps_f = 0.1 * (2 * np.pi) / (times[-1]) # Set eps value

    #if eps_e is None:
    #    eps_e = 0.1 * (2 * np.pi) / (times[-1]) # Set eps value

    M = len(times) # Store number of time steps 
    J = len(H_ops) # Store number of control lines 

    u = np.zeros((R, J, M)) # Create initial control field array (GRAPE Iterations, Control Lines, Time Steps)

    H_ops_data = [H_op.data for H_op in H_ops] # Store Control Hamiltonian Terms 

    use_u_limits = 0 # Set u_limits to 0
    u_min = u_max = alpha_val = beta_val = 0.0 # Set all optional values to 0

    du_max_per_iteration = np.zeros((R - 1, J))

    eps_e_range = np.linspace(2 * np.pi, 0.1, R - 1)
    eps_f_range = np.linspace(2 * np.pi, 0.1, R - 1)

    progress_bar.start(R) # Start progress bar
    for r in range(R - 1): # Start GRAPE Iterations Loop
        progress_bar.update(r) # Update progress bar for each GRAPE Iteration

        dt = times[1] - times[0] # Calculate timestep (assuming all are equal)

        def _H_idx(idx): # Define nested function that calculates the total Hamiltonian in a certain timestep
            return H0 + sum([u[r, j, idx] * H_ops[j] for j in range(J)])

        U_list = [(-1j * _H_idx(idx) * dt).expm().data
                    for idx in range(M-1)] # Calculate total Unitary --> Use function above and loop through all timesteps 

        U_f_list = [] # Initialize forward propagator array
        U_b_list = [] # Initialize backward propagator array 

        U_f = 1 # Initialize first forward propagator
        U_b = sp.eye(*(U.shape)) # Initialize backward propagator 
        for n in range(M - 1): # Loop over all timesteps to calculate forward and backward propagator arrays 

            U_f = U_list[n] * U_f # Forward propagator array 
            U_f_list.append(U_f)

            U_b_list.insert(0, U_b)
            U_b = U_list[M - 2 - n].T.conj().tocsr() * U_b # Backward propagator 

        du_max_per_iteration[r] = cy_grape_inner(U.data, u, r, J, M, U_b_list, U_f_list, H_ops_data, # Calculate Gradient based on cy_grape_inner function --> update control parameters --> do R times
                       dt, eps_f, eps_e, weight_ec, weight_fidelity, H0, H_ops, alpha_val, beta_val, phase_sensitive,
                       use_u_limits, u_min, u_max)
        
    H_td_func = [H0] + [[H_ops[j], u[-1, j, :]] for j in range(J)] # Store total Hamiltonian over time 

    progress_bar.finished() # End progress bar 

    return GRAPEResult(u=u, U_f=Qobj(U_f_list[-1], dims=U.dims), # Return result 
                       H_t=H_td_func), du_max_per_iteration

def plot_grape_control_fields(times, u, labels, uniform_axes=False):


    """
    Plot a series of plots showing the GRAPE control fields given in the
    given control pulse matrix u.

    Parameters
    ----------
    times : array
        Time coordinate array.

    u : array
        Control pulse matrix.

    labels : list
        List of labels for each control pulse sequence in the control pulse
        matrix.

    uniform_axes : bool
        Whether or not to plot all pulse sequences using the same y-axis scale.
    """
    import matplotlib.pyplot as plt

    R, J, M = u.shape

    fig, axes = plt.subplots(J, 1, figsize=(8, 2 * J), squeeze=False)

    y_max = abs(u).max()

    for r in range(R):
        for j in range(J):

            if r == R - 1:
                lw, lc, alpha = 2.0, 'k', 1.0

                axes[j, 0].set_ylabel(labels[j], fontsize=18)
                axes[j, 0].set_xlabel(r'$t$', fontsize=18)
                axes[j, 0].set_xlim(0, times[-1])

            else:
                lw, lc, alpha = 0.5, 'b', 0.25

            axes[j, 0].step(times, u[r, j, :], lw=lw, color=lc, alpha=alpha)

            if uniform_axes:
                axes[j, 0].set_ylim(-y_max, y_max)

    fig.tight_layout()

    return fig, axes

def _overlap(A, B):
    return (A.dag() * B).tr() / A.shape[0]

def CalculateOptimalFieldEnergeticCost(U_Target, H_Static, H_Control, Iterations, Timesteps, T, H_Labels, weight_ec, weight_fidelity, Use_Rand_u0 = False, Plot_Control_Field = False, Plot_Tomography = False, Plot_du = False):

    """
    Calculate Optimal Control Fields and Energetic Cost for the Hamiltonian operators in H_Control so that the unitary U_target is realized

    Parameters 
    ----------

        U_Target : Target unitary evolution operator

        H_Static : Static/Drift term of the Hamiltonian

        H_Control : Control Hamiltonian containing operators that can be tuned in the Hamiltonian via the control fields

        Iterations : Number of GRAPE iterations 

        Timesteps : Number of timesteps 'N' for time discretization 

        T : Total Time 

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

    for i in range(len(time)):

        Total_Hamiltonian = 0

        for j in range(len(H_Control)):

            Total_Hamiltonian += Control_Fields[-1, j, i] * H_Control[j]

        Energetic_Cost_List.append(np.linalg.norm(Total_Hamiltonian))

    Energetic_Cost = np.sum(Energetic_Cost_List) * stepsize

    return result.U_f, Energetic_Cost, Fidelity, du_list_per_iteration
