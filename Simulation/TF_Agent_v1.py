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



# Intial Values
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

# Create Quantum Environment Wrapper

class QuantumEnvironmentWrapper(py_environment.PyEnvironment):

    def __init__(self, quantum_environment, max_steps_per_episode):
        self._quantum_env = quantum_environment
        self._action_spec = tf.TensorSpec(1, dtype = tf.float32, name = 'action')
        self._observation_spec = tf.TensorSpec(1, dtype = tf.float32, name = 'observation')
        self._time_step_spec = ts.time_step_spec(self._observation_spec)
        self._current_time_step = None
        self._max_steps_per_episode = max_steps_per_episode
        self._current_step = None

    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def time_step_spec(self):
        return self._time_step_spec
    
    def _reset(self):
        state = self._quantum_env.reset()
        self._current_time_step = ts.restart(state)
        self._current_step = 0
        return self._current_time_step
    
    def _step(self, action):
        action_2d = np.reshape(action, (len(h_c), number_of_timesteps))
        next_state = self._quantum_env.run_pulses(action_2d)
        next_state_flat = fc.convert_qutip_to_numpy(next_state).flatten()
        reward = self._quantum_env.calculate_fidelity_reward(next_state)
        self.current_time_step = ts.transition(next_state_flat, reward)
        self._current_step += 1

        if self._current_step >= self._max_steps_per_episode:
            self._current_time_step = ts.termination(next_state_flat, reward)

        return self._current_time_step
    
# Create Quantum Environment Instance
    
quantum_env = QuantumEnvironment(number_qubits, h_d, h_c, h_l, t1, t2, initial_state, target_unitary, number_of_timesteps, gate_duration)

# Wrap Quantum Environment with a PyEnvironment

wrapped_env = QuantumEnvironmentWrapper(quantum_env, max_train_steps)

# Convert the PyEnvironment to a TFEnvironment

tf_env = tf_py_environment.TFPyEnvironment(wrapped_env)

# Define the Q-Network

fc_layer_params = (100,)
print(tf_env.action_spec())
q_net = q_network.QNetwork(tf_env.observation_spec(), tf_env.action_spec(), fc_layer_params = fc_layer_params)

# Define the DQN Agent

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = 1e-3)
train_step_counter = tf.Variable(0)
agent = dqn_agent.DqnAgent(tf_env.time_step_spec(), 
                           tf_env.action_spec(), 
                           q_network = q_net, 
                           optimizer = optimizer, 
                           td_errors_loss_fn = common.element_wise_squared_loss,
                           train_step_counter = train_step_counter)

agent.initialize()

# Create replay buffer
replay_buffer_capacity = 10000
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec = agent.collect_data_spec,
                                                               batch_size = tf_env.batch_size,
                                                               max_length = replay_buffer_capacity)

# Define collect data function
def collect_data(env, policy, buffer):
    time_step = env._current_time_step()
    action_step = policy.action(time_step)
    next_time_step = env.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    buffer.add_batch(traj)

# Train the agent
num_iterations = 10000

for _ in range(num_iterations):
    # Collect data using agent's policy
    collect_data(tf_env, agent.collect_policy, replay_buffer)

    # Sample a batch of data from the replay buffer
    experience, unused_info = replay_buffer.get_next()

    # Train the agent on the sampled batch
    train_loss = agent.train(experience).loss

    # Print the training loss for monitoring
    if _ % 100 == 0:
        print('Iteration: {}, Loss: {}'.format(_, train_loss.numpy()))

# Save the trained agent if needed
#tf.saved_model.save(agent.policy, '')
    
