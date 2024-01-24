import numpy as np 
from qutip import basis, fidelity, identity, sigmax, sigmaz, tensor, destroy
from qutip_qip.operations import expand_operator, toffoli, snot
from tf_agents.typing.types import NestedArraySpec
import functions as fc
from qutip.metrics import fidelity
from simulator import QuantumEnvironment, run_training
import tensorflow as tf 
from scipy.optimize import minimize 
from keras import layers, models 
import keras
from collections import deque
import random
import matplotlib.pyplot as plt
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
tf.random.set_seed(12357111317)

# Initial parameters 
h_d = np.pi * (tensor(sigmaz(), identity(2)) + tensor(identity(2), sigmaz())) + (1/2) * np.pi * tensor(sigmaz(), sigmaz()) # Define Drift Hamiltonian used in "Processor"
h_c = [tensor(identity(2), sigmax())]
h_l = [r'$u_{1x}$', r'$u_{2x}$', r'$u_{xx}$'] 
target_unitary = fc.cnot()
number_qubits = 2
gate_duration = 2 * np.pi
t1 = 100 * gate_duration
t2 = 100 * gate_duration
number_of_timesteps = 100
number_of_grape_iterations = 500
max_train_steps = 1 # Inner "for loop"
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
time = np.linspace(0, gate_duration, number_of_timesteps)




# Hyperparameters

fc_layer_params = (100, 100, 100)
learning_rate = 1e-3
num_iterations = 800 # Number of episodes "outer for loop in training loop"
collect_episodes_per_iteration = 1
eval_interval = 1
replay_buffer_capacity = 100

env_train_py = QuantumEnvironment(number_qubits, h_d, h_c, h_l, t1, t2, initial_state, target_unitary, number_of_timesteps, gate_duration, number_of_grape_iterations, max_train_steps)
env_eval_py = QuantumEnvironment(number_qubits, h_d, h_c, h_l, t1, t2, initial_state, target_unitary, number_of_timesteps, gate_duration, number_of_grape_iterations, max_train_steps)

train_env = tf_py_environment.TFPyEnvironment(env_train_py)
eval_env = tf_py_environment.TFPyEnvironment(env_eval_py)

# Agent

actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params = fc_layer_params,
)

optimizer = keras.optimizers.Adam(learning_rate = learning_rate)

train_step_counter = tf.compat.v2.Variable(0)

tf_agent = reinforce_agent.ReinforceAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_network = actor_net,
    optimizer = optimizer,
    normalize_returns = True,
    train_step_counter = train_step_counter,
)

tf_agent.initialize()

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

# Replay buffer

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec = tf_agent.collect_data_spec,
    batch_size = train_env.batch_size,
    max_length = replay_buffer_capacity,
)

eval_replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec = tf_agent.collect_data_spec,
    batch_size = eval_env.batch_size,
    max_length = max_train_steps + 1,
)

# Data Collection

avg_return = tf_metrics.AverageReturnMetric()

eval_observers = [avg_return, eval_replay_buffer.add_batch]
eval_driver = dynamic_episode_driver.DynamicEpisodeDriver(
    eval_env,
    eval_policy,
    eval_observers,
    num_episodes = 1,
)

train_observers = [replay_buffer.add_batch]
train_driver = dynamic_episode_driver.DynamicEpisodeDriver(
    train_env,
    collect_policy,
    train_observers,
    num_episodes = collect_episodes_per_iteration,
)

tf_agent.train = common.function(tf_agent.train)

# Training the agent 

return_list = []
episode_list = []
iteration_list = []

return_list_, episode_list_, iteration_list_ = run_training(
    tf_agent,
    train_driver,
    replay_buffer,
    eval_driver,
    eval_replay_buffer,
    avg_return,
    num_iterations = num_iterations,
    eval_interval = eval_interval,
    save_episodes = True,
    clear_buffer = False,
)

return_list += return_list_
episode_list += episode_list_
iteration_list += iteration_list_

# Plotting the results 


avg_eval_reward_per_episode = []
avg_train_reward_per_episode = []
iteration_space = np.linspace(1, max_train_steps*num_iterations, max_train_steps*num_iterations)

for i in range(num_iterations):
    sum_eval = np.sum(env_eval_py.reward_list[max_train_steps*i : max_train_steps + max_train_steps*i])
    sum_train =np.sum(env_train_py.reward_list[max_train_steps*i : max_train_steps + max_train_steps*i])
    avg_eval_reward_per_episode.append(sum_eval/max_train_steps)
    avg_train_reward_per_episode.append(sum_train/max_train_steps)



fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(iteration_list, avg_eval_reward_per_episode, label = "Average Fidelity per episode Eval Env", color = "blue")
#ax1.plot(iteration_list, avg_train_reward_per_episode, label = "Average Fidelity per episode Train Env", color = "green")
ax2.plot(iteration_list, return_list, label = "Average Return per episode", color = "orange")
ax1.set_ylabel("Average Fidelity per Episode")
ax2.set_ylabel("Average Return per Episode")
ax1.legend(loc = "upper left")
ax1.grid()
ax2.legend(loc = "upper right")
plt.show()

plt.plot(iteration_space, env_eval_py.reward_list, label = "Fidelity Eval Env")
#plt.plot(iteration_space, env_train_py.reward_list, label = "Fidelity Train Env")
plt.xlabel("Iteration number (# episodes x # cycles)")
plt.ylabel("Fidelity per iteration")
plt.grid()
plt.legend()
plt.show()

final_val = episode_list[-1]
final_pulse = final_val.action.numpy()[0, 0, :]

#plt.plot(time, final_pulse)
plt.step(time, final_pulse)
plt.axhline(y = 0, color = "black", ls = "-")
plt.xlabel("Time")
plt.ylabel(r"$\sigma_{xx}$")
plt.grid()
plt.show()
