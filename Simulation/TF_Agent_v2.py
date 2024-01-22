import numpy as np 
from qutip import basis, fidelity, identity, sigmax, sigmaz, tensor, destroy
from qutip_qip.operations import expand_operator, toffoli, snot
from tf_agents.typing.types import NestedArraySpec
import functions as fc
from qutip.metrics import fidelity
from simulator import QuantumEnvironment
import tensorflow as tf 
from scipy.optimize import minimize 
from keras import layers, models 
import keras
from collections import deque
import random
import matplotlib.pyplot as plt 
from alive_progress import alive_bar
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
import tensorflow as tf
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.trajectories import trajectory
from tensorflow.python.framework import tensor_spec as tsp

# Initial parameters 
h_d = np.pi * (tensor(sigmaz(), identity(2)) + tensor(identity(2), sigmaz())) + (1/2) * np.pi * tensor(sigmaz(), sigmaz()) # Define Drift Hamiltonian used in "Processor"
h_c = [tensor(identity(2), sigmax())]
h_l = [r'$u_{1x}$', r'$u_{2x}$', r'$u_{xx}$'] 
target_unitary = fc.cnot()
number_qubits = 2
gate_duration = 2 * np.pi
t1 = 100 * gate_duration
t2 = 100 * gate_duration
number_of_timesteps = 5
number_of_grape_iterations = 500
max_train_steps = 1000
initial_state = basis(4, 2)
initial_dm = initial_state * initial_state.dag()
numpy_initial_state = fc.convert_qutip_to_numpy(initial_state)
numpy_initial_dm = fc.convert_qutip_to_numpy(initial_dm)
weight_fidelity = 1
weight_energy = 0
epsilon_f = 1
epsilon_e = 100
state_size = (2*number_qubits)**2
state_shape = (number_qubits**2, number_qubits**2)
action_size = len(h_c) * number_of_timesteps
action_shape = (len(h_c), number_of_timesteps)
