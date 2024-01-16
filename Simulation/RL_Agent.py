import numpy as np 
from qutip import basis, fidelity, identity, sigmax, sigmaz, tensor, destroy
from qutip_qip.operations import expand_operator, toffoli, snot
import functions as fc
from qutip.metrics import fidelity
from simulator import QuantumEnvironment
import tensorflow as tf 
from scipy.optimize import minimize 
import keras 

def clip(A):
    return np.clip(A, -1, +1)

# Input Parameters

h_d = np.pi * (tensor(sigmaz(), identity(2)) + tensor(identity(2), sigmaz())) + (1/2) * np.pi * tensor(sigmaz(), sigmaz()) # Define Drift Hamiltonian used in "Processor"
h_c = [tensor(identity(2), sigmax())]
h_l = [r'$u_{1x}$', r'$u_{2x}$', r'$u_{xx}$'] 
target_unitary = fc.cnot()
number_qubits = 2
gate_duration = 2 * np.pi
t1 = 100 * gate_duration
t2 = 100 * gate_duration
number_of_timesteps = 500
number_of_grape_iterations = 500
initial_state = basis(4, 2)
weight_fidelity = 1
weight_energy = 0
epsilon_f = 1
epsilon_e = 100

# Initiate Quantum Environment  

Environment = QuantumEnvironment(number_qubits, h_d, h_c, h_l, t1, t2, initial_state, target_unitary, number_of_timesteps, gate_duration, number_of_grape_iterations) # Create instance of Quantum Environment

# Define Q-Learning parameters

learning_rate = 0.001
discount_factor = 0.99
exploration_rate = 1.0 
exploration_decay = 0.995
min_exploration_rate = 0.01
num_episodes = 1000

# Define NN for the Q-function approximation 

model = keras.Sequential([keras.layers.Dense(number_of_timesteps, activation='tanh', input_shape=(len(h_c), number_of_timesteps)), keras.layers.Dense(number_of_timesteps)])

model.compile(optimizer = keras.optimizers.Adam(learning_rate), loss = 'mse')

print(model.summary())
state = np.random.random((len(h_c), number_of_timesteps))
predict = model.predict(np.array([state]))
print(predict)
print(np.shape(predict))
model.fit(np.array([state]), np.array([target]), epochs=1, verbose = 0)


"""

# Q-learning training loop
for episode in range(num_episodes):
    state = np.random.random((len(h_c), number_of_timesteps))  # Initial random state
    total_reward = 0
    done = False
    print(f"Episode {episode}")

    while not done:
        # Choose action using epsilon-greedy policy with increased initial exploration
        if np.random.rand() < exploration_rate:  # Start with higher exploration
            # Randomly sample a pulse sequence
            pulses = np.random.random((len(h_c), number_of_timesteps))
        else:
            # Use the trained model to predict control pulses
            pulses = clip(model.predict(np.array([state]))[0])
            
        # Take action and observe new state and reward
        result = Environment.run_pulses(pulses) # Should be the agent's model 
        reward = Environment.calculate_fidelity_reward(result)

        # Update Q-value using Bellman equation
        next_state = np.random.random((len(h_c), number_of_timesteps))  # This should come from the actual experiment 

        target = reward + discount_factor * np.max(clip(model.predict(np.array([next_state]))[0]))
        model.fit(np.array([state]), np.array([target]), epochs=1, verbose = 0)

        total_reward += reward
        state = next_state

        # Check termination condition (fidelity larger than 0.95)
        if reward > 0.1:
            done = True

        # Print additional information for debugging
        print(f"Exploration Rate: {exploration_rate}, Reward: {reward}")

    # Decay exploration rate
    exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")
"""
