import numpy as np 
from qutip import basis, fidelity, identity, sigmax, sigmaz, tensor, destroy
from qutip_qip.operations import expand_operator, toffoli, snot
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
initial_state = basis(4, 2)
initial_dm = initial_state * initial_state.dag()
numpy_initial_state = fc.convert_qutip_to_numpy(initial_state)
numpy_initial_dm = fc.convert_qutip_to_numpy(initial_dm)
weight_fidelity = 1
weight_energy = 0
epsilon_f = 1
epsilon_e = 100
state_size = (2*number_qubits)**2
action_size = len(h_c) * number_of_timesteps

# Define Q Learning Agent Class
class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size # State size for input
        self.action_size = action_size # Action size for output
        self.gamma = 0.99 # Discount factor
        self.epsilon = 1.0 # Exploration rate
        self.epsilon_min = 0.01 # Minimum exploration rate
        self.epsilon_decay = 0.995 # Exploration decay rate 
        self.learning_rate = 0.001 # Learning rate of the Agent
        self.memory = [] # Initialize memory list 
        self.model = self._build_model() # Build Q-network

    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(units = 24, input_shape = (self.state_size,), activation = 'relu'))
        model.add(layers.Dense(units = 24, activation = 'relu'))
        model.add(layers.Dense(self.action_size, activation = 'tanh'))
        model.compile(loss = 'mse', optimizer = keras.optimizers.Adam(learning_rate = self.learning_rate))
        model.get_metrics_result()
        return model 
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(low = -1, high = 1, size = self.action_size)
        act_values = self.model.predict(np.array(state))[0]
        return np.squeeze(act_values)
    
    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state in minibatch:
            target = reward + self.gamma * np.amax(self.model.predict(next_state))
            self.history = self.model.fit(np.array(state), np.array(target).flatten(), epochs = 1, verbose = 0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 

    def save_weights(self, filename = 'q_learning_weights.h5'):
        self.model.save_weights(filename)

    def load_weights(self, filename = 'q_learning_weights.h5'):
        self.model.load_weights(filename)

# Create Quantum Environment Instance    
env = QuantumEnvironment(number_qubits, h_d, h_c, h_l, t1, t2, initial_state, target_unitary, number_of_timesteps, gate_duration, number_of_grape_iterations)

# Create Q-Learning Agent
agent = QLearningAgent(state_size, action_size)

# Training parameters
episodes = 1000
batch_size = 32
training_timesteps = 100
episode_array = np.linspace(1, episodes, episodes)
total_reward_array = []
loss_array = []

# Q-Learning training loop
with alive_bar(episodes) as bar:
    for episode in range(episodes):
        bar()
        state = numpy_initial_dm.flatten()
        state = np.reshape(state, (1, state_size))
        total_reward = 0
        
        for timestep in range(training_timesteps):
            action = agent.act(state)
            action_2d = np.reshape(action, (len(h_c), number_of_timesteps))
            next_state = env.run_pulses(action_2d)
            reward = env.calculate_fidelity_reward(next_state)
            next_state = fc.convert_qutip_to_numpy(next_state).flatten()
            next_state = np.reshape(next_state, (1, state_size))
            agent.remember(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        agent.replay(batch_size)
        print(f"Episode: {episode + 1}, Total Average Reward: {total_reward/training_timesteps}")
        total_reward_array.append(total_reward/training_timesteps)
        metrics = agent.history.history['loss']
        loss_array.append(metrics)
        
# Save the trained weights
agent.save_weights('q_learning_weights.h5')

# Plot total reward versus episodes 

plt.plot(episode_array, total_reward_array, label = 'Total Average Reward')
plt.plot(episode_array, loss_array, label = 'MSE loss')
plt.xlabel("Training episode number (#)")
plt.ylabel("Total Average Reward")
plt.title("Total average reward versus training episode number")
plt.legend()
plt.grid()
plt.show()