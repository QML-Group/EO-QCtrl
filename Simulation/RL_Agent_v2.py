import numpy as np 
from qutip import basis, fidelity, identity, sigmax, sigmaz, tensor, destroy
from qutip_qip.operations import expand_operator, toffoli, snot
import functions as fc
from qutip.metrics import fidelity
from simulator import QuantumEnvironment
import tensorflow as tf 
from scipy.optimize import minimize 
import keras 
from collections import deque
import random

# Input Parameters

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
initial_state = basis(4, 2)
weight_fidelity = 1
weight_energy = 0
epsilon_f = 1
epsilon_e = 100

# Define parameters

state_size = (1, number_qubits**2)
action_size = (len(h_c), number_of_timesteps)
num_episodes = 1000
batch_size = 32
training_timesteps = 100
quantum_env = QuantumEnvironment(number_qubits, h_d, h_c, h_l, t1, t2, initial_state, target_unitary, number_of_timesteps, gate_duration, number_of_grape_iterations)

# Define Q-Network Class

class QNetwork(keras.Model):

    def __init__(self, state_size, action_size):

        super(QNetwork, self).__init__()

        # Define neural network layers

        self.dense1 = keras.layers.Dense(64, activation = 'tanh', input_shape = state_size)
        self.dense2 = keras.layers.Dense(32, activation = 'tanh')
        self.output_layer = keras.layers.Dense(action_size[1], activation= 'tanh')

    def call(self, state):
        # Define the forward pass of the neural network
        x = self.dense1(state)
        x = self.dense2(x)
        return self.output_layer(x)
    
# Define the Q-learning agent class
    
class QAgent:

    def __init__(self, state_size, action_size, learning_rate = 0.001, gamma = 0.95, epsilon = 1.0, epsilon_decay = 0.995, epsilon_min = 0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Intialize Q-Network models

        self.model = QNetwork(state_size, action_size)
        self.target_model = QNetwork(state_size, action_size)

        # Define optimizer

        self.optimizer = keras.optimizers.Adam(learning_rate)

        # Initialize replay memory

        self.memory = deque(maxlen = 2000)

    def act(self, state):
        # Epsilon-greedy exploration strategy

        if np.random.rand() <= self.epsilon:
            return np.random.uniform(-1, 1, size = self.action_size)
        
        return self.model.predict(np.array([state]))[0]
    
    def remember(self, state, action, reward, next_state):
        # Add an experience tuple to the replay memory
        self.memory.append((state, action, reward, next_state))

    def replay(self, batch_size):
        # Experience replay to update Q-Network
        if len(self.memory) < batch_size:
            return 
        
        minibatch = np.array(random.sample(self.memory, batch_size))
        states = np.vstack(minibatch[:, 0])
        actions = np.vstack(minibatch[:, 1])
        rewards = minibatch[:, 2]
        next_states = np.vstack(minibatch[:, 3])

        # Compute Q-values for the minibatch
        targets = rewards + self.gamma * np.max(self.target_model.predict(next_states), axis = 1)
        targets_f = self.model.predict(states)
        targets_f[range(batch_size), actions] = targets

        # Update Q-network using the minibatch
        self.model.fit(states, targets_f, epochs = 1, verbose  = 0)

    def target_train(self):
        # Update target Q-Network with the current Q-network weights
        self.target_model.set_weights(self.model.get_weights())

    def decay_epsilon(self):
        # Decay epsilon for epsilon-greedy exploration
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Run RL Agent 

agent = QAgent(state_size, action_size)

for episode in range(num_episodes):
    state = np.random.uniform(-1, 1, size = state_size)
    total_reward  = 0

    for timestep in range(training_timesteps):
        # Agent chooses an action based on the current state
        action = agent.act(state)
        print(timestep)
        print(np.shape(action))

        # Environment provides the next state and the reward for the chosen action
        next_state = quantum_env.run_pulses(action)
        print(next_state)
        reward = quantum_env.calculate_fidelity_reward(next_state)

        # Store the experience tuple in the agent's memory
        agent.remember(state, action, reward, next_state)
        state = next_state
        total_reward += reward

        # Experience replay and Q-network update
        agent.replay(batch_size)
        # Update the target Q-network
        agent.target_train()
        # Decay epsilon for exploration-exploitation trade-off
        agent.decay_epsilon()

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# Save the weights after training
    
agent.model.save_weights('q_learning_weights.h5')


