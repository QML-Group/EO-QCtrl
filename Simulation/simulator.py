import numpy as np 
from qutip import basis, fidelity, identity, sigmax, sigmaz, tensor, destroy
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import expand_operator, toffoli, snot
import matplotlib.pyplot as plt 
from qutip.qip.device import Processor
import functions as fc
import qutip.visualization as vz
from qutip.qip.noise import RelaxationNoise
from qutip.metrics import fidelity
from qutip import Qobj
import tensorflow as tf 
from tensorflow.python import keras
from keras import optimizers 

class QuantumEnvironment:

    def __init__(self, n_q, h_drift, h_control, t_1, t_2, initial_state, u_target, timesteps = 500, pulse_duration = 2 * np.pi, grape_iterations = 500):

        """
        Initialize QuantumEnvironment instance.

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
        self.t_1 = t_1
        self.t_2 = t_2
        self.initial_state = initial_state
        self.u_target = u_target
        self.timesteps = timesteps
        self.pulse_duration = pulse_duration
        self.grape_iterations = grape_iterations

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
        Pulses : array
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

        return result
    
    def calculate_fidelity_reward(self, result):

        """
        Calculates Fidelity Reward for a specific Qutip result.

        Parameters
        ----------
        Result : Result instance of Environment using specified Initial State and Pulse set.

        Returns
        -------
        r_f : Fidelity Reward for a specific Qutip result, Initial State, and Target Unitary.
        """

        sv_sim = result.states[-1]
        dm_sim = sv_sim * sv_sim.dag()

        qutip_u_target = Qobj(self.u_target)
        sv_target = qutip_u_target * self.initial_state
        dm_target = sv_target * sv_target.dag()

        r_f = fidelity(dm_sim, dm_target)

        return r_f