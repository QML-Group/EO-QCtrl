import numpy as np 
from qutip import basis, fidelity, identity, sigmax, sigmaz, sigmay, tensor, destroy, to_super, qpt, qpt_plot_combined
from qutip_qip.operations import expand_operator, toffoli, snot
import matplotlib.pyplot as plt 
from qutip.qip.device import Processor
import functions as fc
import qutip.visualization as vz
from qutip.qip.noise import RelaxationNoise
from qutip.metrics import fidelity
from qutip import Qobj
from alive_progress import alive_bar
from scipy.sparse.linalg import expm

class QuantumEnvironment:
   
    def __init__(self, n_q, h_drift, h_control, labels, t_1, t_2, initial_state, u_target, timesteps = 500, pulse_duration = 2 * np.pi, grape_iterations = 500):

        """
        QuantumEnvironment Class
        ------------------------

        Create instance of a Quantum Processor with customizable Drift and Control Hamiltonian, Relaxation and Decoherence times for Pulse Level control 
        
        Contains EO-GRAPE Algorithm (Energy Optimized Gradient Ascent Pulse Engineering)

        Parameters
        ----------
        n_q : int
            Number of qubits.
        h_drift : Qobj
            Drift Hamiltonian.
        h_control : list of Qobj
            List of Control Hamiltonians.
        t_1 : int
            Relaxation time of the Processor.
        t_2 : int
            Decoherence time of the Processor.
        initial_state : Qobj, optional
            Initial state of the quantum state.
        u_target : array, optional
            Target Unitary Evolution Operator.
        timesteps : int, optional
            Number of timesteps to discretize time. Default is 500.
        pulse_duration : float, optional 
            Total pulse duration time. Default is 2pi
        grape_iterations : int, optional
            Number of GRAPE iterations. Default is 500
        """

        self.n_q = n_q
        self.h_drift = h_drift
        self.h_control = h_control
        self.labels = labels 
        self.t_1 = t_1
        self.t_2 = t_2
        self.initial_state = initial_state
        self.u_target = u_target
        self.timesteps = timesteps
        self.pulse_duration = pulse_duration
        self.grape_iterations = grape_iterations
        self.h_drift_numpy = fc.convert_qutip_to_numpy(h_drift)
        self.h_control_numpy = fc.convert_qutip_list_to_numpy(h_control)


        self.create_environment()

    def create_environment(self):

        """
        Create instance of a Qutip Processor as environment with custom 
        Drift and Control Hamiltonians, T1, and T2 times
        """

        timespace = np.linspace(0, self.pulse_duration, self.timesteps)
        simulatortimespace = np.append(timespace, timespace[-1])

        targets = list(range(self.n_q))
        noise = RelaxationNoise(t1 = self.t_1, t2 = self.t_2)

        self.environment = Processor(N = self.n_q)
        self.environment.add_drift(self.h_drift, targets = targets)

        for operator in self.h_control:
            self.environment.add_control(operator, targets = targets)

        self.environment.set_all_tlist(simulatortimespace)
        self.environment.add_noise(noise = noise)

    def run_pulses(self, pulses, plot_pulses = False):

        """
        Send pulses to a specific qutip processor instance and simulate result 

        Parameters
        ----------
        pulses : array
            (K x I_G) Array containing amplitudes of operators in Control Hamiltonian.

        Returns
        -------
        result : Result instance of Environment using specified Initial State and Pulse set.
        """

        for i in range(len(pulses[:, 0])):
            self.environment.pulses[i].coeff = pulses[i]

        result = self.environment.run_state(init_state = self.initial_state)
        
        if plot_pulses == True:
            self.environment.plot_pulses()
            plt.show()

        return result.states[-1]
    
    def calculate_fidelity_reward(self, result, plot_result = False):

        """
        Calculates Fidelity Reward for a specific Qutip result.

        Parameters
        ----------
        Result : Result instance of Environment using specified Initial State and Pulse set.

        Returns
        -------
        r_f : Fidelity Reward for a specific Qutip result, Initial State, and Target Unitary.
        """

        sv_sim = result
        dm_sim = sv_sim * sv_sim.dag()

        qutip_u_target = Qobj(self.u_target)
        sv_target = qutip_u_target * self.initial_state
        dm_target = sv_target * sv_target.dag()

        r_f = fidelity(dm_sim, dm_target)

        if plot_result == True:

            vz.hinton(dm_sim, xlabels = [r'$\vert 00\rangle$', r'$\vert 01\rangle$', r'$\vert 10\rangle$', r'$\vert 11\rangle$'], 
              ylabels = [r'$\vert 00\rangle$', r'$\vert 01\rangle$', r'$\vert 10\rangle$', r'$\vert 11\rangle$'])
            plt.show()

        return r_f
    
    def calculate_energetic_cost(self, pulses, return_normalized = False):

        """
        Calculate Energetic Cost of certain set of Pulses

        Parameters
        ----------
        pulses : array
            (K x I_G) Array containing amplitudes of operators in Control Hamiltonian.

        Returns 
        ---------

        return_value : float
            Energetic cost of the quantum operation 
        """

        h_t_norm = []
        stepsize = self.pulse_duration/self.timesteps

        for i in range(self.timesteps - 1):
            h_t = 0
            for j in range(len(self.h_control)):
                h_t += pulses[j, i] * self.h_control_numpy[j]

            h_t_norm.append(np.linalg.norm(h_t))

        energetic_cost = np.sum(h_t_norm) * stepsize
        energetic_cost_normalized = energetic_cost / (self.pulse_duration * np.linalg.norm(np.sum(self.h_control_numpy)))

        if return_normalized == True:

            return_value = energetic_cost_normalized

        elif return_normalized == False:

            return_value = energetic_cost

        return return_value
    
    def grape_iteration(self, u, r, J, u_b_list, u_f_list, dt, eps_f, eps_e, w_f, w_e):

        """
        Perform one iteration of the GRAPE algorithm and update control pulse parameters

        Parameters
        ----------

        u : The generated control pulses with shape (iterations, controls, time)

        r : The number of this specific GRAPE iteration

        J : The number of controls in Control Hamiltonian

        u_b_list : Backward propagators of each time (length M)

        u_f_list : Forward propagators of each time (length M)

        dt : Timestep size

        eps_f : Distance to move along the gradient when updating controls for Fidelity

        eps_e : Distance to move along the gradient when updating controls for Energy

        w_f : Weight assigned to Fidelity part of the Cost Function

        w_e : Weight assigned to Energy part of the Cost Function

        Returns 
        --------

        u[r + 1, :, :] : The updated parameters 
        """

        du_list = np.zeros((J, self.timesteps))
        max_du_list = np.zeros((J))

        for m in range(self.timesteps - 1):

            P = u_b_list[m] @ self.u_target

            for j in range(J):

                Q = 1j * dt * self.h_control_numpy[j] @ u_f_list[m]

                du_f = -2 * w_f * fc.overlap(P, Q) * fc.overlap(u_f_list[m], P)

                denom = self.h_drift_numpy.conj().T @ self.h_drift_numpy + u[r, j, m] * (self.h_drift_numpy.conj().T @ self.h_control_numpy[j] + self.h_control_numpy[j].conj().T @ self.h_drift_numpy)

                du_e = 0 

                for k in range(J):

                    du_e += -1 * dt * w_e * (np.trace(self.h_drift_numpy.conj().T @ self.h_control_numpy[j] + self.h_control_numpy[j].conj().T @ self.h_drift_numpy) + np.trace(self.h_control_numpy[j].conj().T @ self.h_control_numpy[k] * (u[r, j, m] + u[r, k, m])))

                    denom += u[r, j, m] * u[r, k, m] * self.h_control_numpy[j].conj().T @ self.h_control_numpy[k]

                du_e /= (2 * np.trace(denom) ** (1/2))

                du_e_norm = du_e / (self.pulse_duration * (np.linalg.norm(self.h_drift_numpy) + np.linalg.norm(np.sum(self.h_control_numpy))))

                du_t = du_f + du_e_norm 

                du_list[j, m] = du_t.real

                max_du_list[j] = np.max(du_list[j])

                u[r + 1, j, m] = u[r, j, m] + eps_f * du_f.real + eps_e * du_e_norm.real

        for j in range(J):
            u[r + 1, j, self.timesteps - 1] = u[r + 1, j, self.timesteps - 2] 

        return max_du_list
    
    def run_grape_optimization(self, w_f, w_e, eps_f, eps_e):

        """
        Runs GRAPE algorithm and returns the control pulses, final unitary, Fidelity, and Energetic Cost for the Hamiltonian operators in H_Control
        so that the unitary U_target is realized 

        Parameters 
        ----------

        w_f : float
            Weight assigned to Fidelity part of the Cost Function

        w_e : float
            Weight assigned to Energetic Cost part of the Cost Functions

        eps_f : int
            Learning rate for fidelity

        eps_e : int
            Learning rate for energy

        Returns
        --------

        u : Optimized control pulses with dimension (iterations, controls, timesteps)

        u_f_list[-1] : Final unitary based on last GRAPE iteration

        du_max_per_iteration : Array containing the max gradient of each control for all GRAPE iterations

        cost_function : Array containing the value of the cost function for all GRAPE iterations

        infidelity_array : Array containing the infidelity for all GRAPE iterations

        energy_array : Array containing the energetic cost for all GRAPE iterations
        
        """

        self.w_f = w_f

        self.w_e = w_e 

        times = np.linspace(0, self.pulse_duration, self.timesteps)

        if eps_f is None:
            eps_f = 0.1 * (self.pulse_duration) /(times[-1])

        if eps_e is None:
            eps_e = 0.1 * (self.pulse_duration) / (times[-1])

        M = len(times)
        J = len(self.h_control_numpy)

        u = np.zeros((self.grape_iterations, J, M))

        self.du_max_per_iteration = np.zeros((self.grape_iterations - 1, J))

        self.cost_function_array = []
        self.infidelity_array = []
        self.energy_array = []

        with alive_bar(self.grape_iterations - 1) as bar:
            
            for r in range(self.grape_iterations - 1):

                bar()
                dt = times[1] - times[0]

                def _H_idx(idx):
                    return self.h_drift_numpy + sum([u[r, j, idx] * self.h_control_numpy[j] for j in range(J)])
                
                u_list = [expm(-1j * _H_idx(idx) * dt) for idx in range(M - 1)]

                u_f_list = []
                u_b_list = []

                u_f = np.eye(*(self.u_target.shape))
                u_b = np.eye(*(self.u_target.shape))

                for n in range(M - 1):

                    u_f = u_list[n] @ u_f
                    u_f_list.append(u_f)
                    
                    u_b_list.insert(0, u_b)
                    u_b = u_list[M - 2 - n].T.conj() @ u_b

                self.du_max_per_iteration[r] = self.grape_iteration(u, r, J, u_b_list, u_f_list, dt, eps_f, eps_e, w_f, w_e)

                cost_function = w_f * (1 - fc.Calculate_Fidelity(self.u_target, u_f_list[-1])) + w_e * self.calculate_energetic_cost(u[r])
                self.cost_function_array.append(cost_function)
                self.infidelity_array.append(1 - fc.Calculate_Fidelity(self.u_target, u_f_list[-1]))
                self.energy_array.append(self.calculate_energetic_cost(u[r]))
                self.final_unitary = u_f_list[-1]

        return u[-1]
    
    def plot_grape_pulses(self, pulses):

        """
        Plot the pulses generated by the EO-GRAPE Algorithm

        Parameters 
        ----------

        pulses : Pulses generated by the EO-GRAPE algorithm

        labels : Labels for the operators in h_control
        """

        time = np.linspace(0, self.pulse_duration, self.timesteps)

        fig, ax = plt.subplots(len(self.h_control_numpy))

        for i in range(len(self.h_control_numpy)):
            
            ax[i].plot(time, pulses[i, :], label = f"{self.labels[i]}")
            ax[i].set(xlabel = "Time", ylabel = f"{self.labels[i]}")

        plt.subplot_tool()
        plt.show()

    def plot_tomography(self):

        """
        Plot the tomography of the target unitary and the unitary realized by the EO-GRAPE algorithm

        Parameters
        ----------

        final_unitary : The final unitary obtained by the EO-GRAPE algorithm
        """

        op_basis = [[Qobj(identity(2)), Qobj(sigmax()), Qobj(sigmay()), Qobj(sigmaz())]] * 2
        op_label = [["I", "X", "Y", "Z"]] * 2   

        u_i_s = to_super(Qobj(self.u_target))
        u_f_s = to_super(Qobj(self.final_unitary))
        chi_1 = qpt(u_i_s, op_basis)
        chi_2 = qpt(u_f_s, op_basis)

        fig_1 = plt.figure(figsize = (6,6))
        fig_1 = qpt_plot_combined(chi_1, op_label, fig=fig_1, threshold=0.001, title = 'Target Unitary Gate ')

        fig_2 = plt.figure(figsize = (6, 6))
        fig_2 = qpt_plot_combined(chi_2, op_label, fig = fig_2, threshold = 0.001, title = 'Final Unitary after Optimization')

        plt.show()

    def plot_du(self):
        
        """
        Plot the max gradient over all timesteps per control operator as function of GRAPE iterations

        Parameters
        ----------

        du_list : The list containing gradient values from the EO-GRAPE algorithm

        labels : The labels associated to the operators in h_control 
        """

        iteration_space = np.linspace(1, self.grape_iterations - 1, self.grape_iterations - 1)

        for i in range(len(self.h_control_numpy)):
            plt.plot(iteration_space, self.du_max_per_iteration[:, i], label = f"{self.labels[i]}")
        plt.axhline(y = 0, color = "black", linestyle = "-")
        plt.xlabel("GRAPE Iteration Number")
        plt.ylabel("Maximum Gradient over Time")
        plt.title("Maximum Gradient over Time vs. GRAPE Iteration Number")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_cost_function(self):

        """
        Plot the value of the cost function as function of GRAPE iterations

        Parameters
        ----------

        cost_fn : The array containing cost function values obtained by the EO-GRAPE algorithm

        infidelity : The array containing the infidelity values obtained by the EO-GRAPE algorithm

        energy : The array containing the energetic cost values obtained by the EO-GRAPE algorithm

        w_f : The weight assigned to the fidelity part of the cost function

        w_e : The weight assigned to the energetic cost part of the cost function
        """

        iteration_space = np.linspace(1, self.grape_iterations - 1, self.grape_iterations - 1)

        plt.plot(iteration_space, self.cost_function_array, label = f"Cost Function, $w_f$ = {self.w_f}, $w_e$ = {self.w_e}")
        plt.plot(iteration_space, self.infidelity_array, label = f"Infidelity (1-F)")
        plt.plot(iteration_space, self.energy_array, label = f"Normalized Energetic Cost")
        plt.axhline(y = 0, color = "black", linestyle = "-")
        plt.xlabel("GRAPE iteration number")
        plt.ylabel("Cost Function")
        plt.title("Cost Function, Infidelity, and Energetic Cost vs. GRAPE Iteration Number")
        plt.legend()
        plt.grid()
        plt.show()


