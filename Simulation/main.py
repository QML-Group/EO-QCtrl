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
import simulator as sim 
from keras import optimizers 

# Input Parameters

H_Drift_Qutip = np.pi * (tensor(sigmaz(), identity(2)) + tensor(identity(2), sigmaz())) + (1/2) * np.pi * tensor(sigmaz(), sigmaz()) # Define Drift Hamiltonian used in "Processor"

H_Drift_Scratch = np.pi * (fc.tensor(fc.sigmaz(), fc.identity(2)) + fc.tensor(fc.identity(2), fc.sigmaz())) + (1/2) * np.pi * fc.tensor(fc.sigmaz(), fc.sigmaz()) # Define Drift Hamiltonian used for optimization

H_Control_Qutip = [tensor(sigmax(), identity(2)), # Define Control Hamiltonian used in "Processor"
                   tensor(identity(2), sigmax()),
                   tensor(sigmax(), sigmax())]

H_Control_Scratch = [fc.tensor(fc.sigmax(), fc.identity(2)), # Define Control Hamiltonian used for optimization
                   fc.tensor(fc.identity(2), fc.sigmax()),
                   fc.tensor(fc.sigmax(), fc.sigmax())]

TargetUnitary = fc.cnot()

N_q = 2

T = 2 * np.pi

T1 = 100 * T

T2 = 100 * T

Nt = 500

Ng = 500

time = np.linspace(0, T, Nt)

Initial_State = basis(4, 2)

# Run RL Agent and Training

env = sim.QuantumEnvironmentWrapper(N_q, H_Drift_Qutip, H_Control_Qutip, T1, T2, Nt, Initial_State, TargetUnitary)

ppo_model = sim.PPOModel(env.action_space_size)

ppo_optimizer = optimizers.Adam(learning_rate = 0.001)

sim.train_agent(env, ppo_model, ppo_optimizer, num_episodes = 100)
