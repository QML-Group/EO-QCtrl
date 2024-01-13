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

class PPOModel(keras.Model):

    def __init__(self, action_space_size):
        super(PPOModel, self).__init__()
        self.dense1 = keras.layers.Dense(64, activation = 'relu')
        self.dense2 = keras.layers.Dense(action_space_size, activation = 'linear')

    def call(self, state):
        x = self.dense1(state)
        return self.dense2(x)
    
class QuantumEnvironmentWrapper:
    def __init__(self, N_q, H_Drift, H_Control, T_1, T_2, Timesteps, Initial_State, U_Target):
        self.environment = self.create_environment(N_q, H_Drift, H_Control, T_1, T_2, Timesteps)
        self.initial_state = Initial_State
        self.state = None
        self.action_space_size = len(H_Control) * Timesteps
        self.state_space_size = len(Initial_State.full())
        self.targetunitary = U_Target

    def create_environment(self, N_q, H_Drift, H_Control, T_1, T_2, Timesteps):

        """
        Create instance of a Qutip Processor as environment with custom 
        Drift and Control Hamiltonians, T1, and T2 times

        Parameters
        ----------

        N_q : Number of qubits (int)

        H_Drift: Drift Hamiltonian (Qobj)

        H_Control : List of Control Hamiltonians (list(Qobj))

        T_1 : Relaxation time of the Processor (int)

        T_2 : Decoherence time of the Processor (int)

        Timesteps : Number of timesteps to discretize time (int)

        
        Returns
        ----------

        Environment : Instance of Qutip Processor with all parameters defined in input (qutip.qip.device.processor.Processor object)
        """

        T = 2 * np.pi # Total gate duration
        
        Timespace = np.linspace(0, T, Timesteps) # Discretized Time Space

        SimulatorTimespace = np.append(Timespace, Timespace[-1]) # Customize Time Space for Qutip Processor 

        targets = [] # Create target list based on number of Qubits 
        for i in range(N_q):
            targets.append(i)
        
        Noise = RelaxationNoise(t1 = T_1, t2 = T_2) # Define Noise based on given T1 and T2 time

        Environment = Processor(N = N_q) # Initialize Environment using number of Qubits 

        Environment.add_drift(H_Drift, targets = targets) # Specify custom Drift Hamiltonian for the Processor 

        for operator in H_Control:
            Environment.add_control(operator, targets = targets) # Specify all operators in H_Control as Control Hamiltonians 

        Environment.set_all_tlist(SimulatorTimespace) # Set time space of the processor 

        Environment.add_noise(noise = Noise) # Add Relaxation Noise to processor 

        return Environment
    
    def generate_initial_state(self):

        """ Generate Initial State """

        return self.initial_state
    
    def reset(self):

        """ Define reset functionality """

        self.state = self.generate_initial_state()
        
        return np.array(self.state).flatten()
    
    def step(self, action):

        """ Define next step functionality """

        next_state, result  = self.run_pulses(pulses = action)
        self.state = next_state
        return np.array(next_state).flatten(), self.calculate_fidelity_reward(result), False, result
    
    def calculate_fidelity_reward(self, result):
        """
        Calculates Fidelity Reward for a specific Qutip result 

        Parameters
        ----------

        Result : Result instance of Environment using specified Initial State and Pulse set (np.array (T x N_q^2))

        InitialState : InitialState : Initial state of the quantum state (Qobj)

        U_Target : Target Unitary Evolution Operator

        Returns
        ----------

        r_f : Fidelity Reward for a specific Qutip result, Intital State and Target Unitary (float)
        """

        SV_Sim = result.states[-1] # Define final State Vector (final time step)

        DM_Sim = SV_Sim * SV_Sim.dag() # Calculate simulated density matrix from state vector (a * a.dag())

        Qutip_U_Target = Qobj(self.targetunitary) # Convert Target Unitary to a Qobj Object

        SV_Target = Qutip_U_Target * self.initial_state # Calculate Target State Vector based on Initial State and Target Unitary 

        DM_Target = SV_Target * SV_Target.dag() # Calculate target density matrix from state vector (a * a.dag())

        r_f = fidelity(DM_Sim, DM_Target) # Calculate fidelity between simulated and target density matrix as reward

        return r_f
    
    def run_pulses(self, pulses):

        """ Run Pulses on Processor """
        print(pulses)
        for i in range(len(pulses[:, 0])):
            self.environment.pulses[i].coeff = pulses[i] # Pass all amplitudes in Pulses array to the Processor 

        result = self.environment.run_state(init_state = self.initial_state, pulses = pulses)
        self.state = result.states[-1]

        return result.states[-1], result 
  
def ppo_training(model, optimizer, states, actions, rewards, next_states, dones, discount_factor = 0.99, epsilon = 0.2, epochs = 10):
    
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    dones = np.array(dones)

    # Compute advantages using Generalized Advantage Estimation (GAE)
    advantages = compute_advantages(rewards, model.predict(next_states), dones, discount_factor)

    # Normalize advantages
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    # Train the policy for multiple epochs
    for epoch in range(epochs):
        # Shuffle data
        indices = np.arange(len(states))
        np.random.shuffle(indices)

        # Iterate over mini-batches
        batch_size = 32
        for start in range(0, len(states), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            # Compute loss and update policy
            compute_policy_loss(model, optimizer, states[batch_indices], actions[batch_indices], advantages[batch_indices], epsilon)

# Helper function to compute advantages using Generalized Advantage Estimation (GAE)
def compute_advantages(model, states, rewards, next_state_values, dones, discount_factor):
    td_targets = rewards + discount_factor * next_state_values * (1 - dones)
    advantages = td_targets - model.predict(states)
    return advantages

# Helper function to compute policy loss and update policy
def compute_policy_loss(model, optimizer, states, actions, advantages, epsilon):
    action_probs = model(states)
    chosen_action_probs = tf.reduce_sum(action_probs * tf.one_hot(actions, model.output_shape[1]), axis=1)

    old_action_probs = tf.placeholder(tf.float32, shape=(None,))

    ratio = chosen_action_probs / old_action_probs
    clipped_ratio = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)

    surrogate_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

    optimizer.minimize(surrogate_loss, model.trainable_variables)

def train_agent(environment, model, optimizer, num_episodes = 1000, max_timestep_per_episode = 100):

    state = environment.reset()
    action_space_size = environment.action_space_size
    state_space_size = environment.state_space_size

    model = PPOModel(action_space_size)
    optimizer = optimizers.Adam(learning_rate = 0.001)

    for episode in range(num_episodes):
        states, actions, rewards, next_states = [], [], [], []
        timestep = 0

        while timestep < max_timestep_per_episode:

            def softmax(x):
                exp_x = np.exp(x - np.max(x))  # Avoid numerical instability by subtracting the max
                return exp_x / exp_x.sum(axis=0)
            
            state_array = np.array(state).flatten()
            action_probs = softmax(model.predict(state_array.reshape(1, -1))[0])
            #action_probs = action_probs / np.sum(action_probs)
            action = np.random.choice(range(action_space_size), p = action_probs)

            next_state, result = environment.step(action)

            states.append(state_array)
            actions.append(action)
            rewards.append(result)
            next_states.append(np.array(next_state).flatten())

            state = next_state
            timestep += 1

    ppo_training(model, optimizer, states, actions, rewards, next_states)

    print(f"Episode: {episode + 1}, Total Reward: {sum(rewards)}")

def create_environment(N_q, H_Drift, H_Control, T_1, T_2, Timesteps):

    """
    Create instance of a Qutip Processor as environment with custom 
    Drift and Control Hamiltonians, T1, and T2 times

    Parameters
    ----------

    N_q : Number of qubits (int)

    H_Drift: Drift Hamiltonian (Qobj)

    H_Control : List of Control Hamiltonians (list(Qobj))

    T_1 : Relaxation time of the Processor (int)

    T_2 : Decoherence time of the Processor (int)

    Timesteps : Number of timesteps to discretize time (int)

    
    Returns
    ----------

    Environment : Instance of Qutip Processor with all parameters defined in input (qutip.qip.device.processor.Processor object)
    """

    T = 2 * np.pi # Total gate duration
    
    Timespace = np.linspace(0, T, Timesteps) # Discretized Time Space

    SimulatorTimespace = np.append(Timespace, Timespace[-1]) # Customize Time Space for Qutip Processor 

    targets = [] # Create target list based on number of Qubits 
    for i in range(N_q):
        targets.append(i)
    
    Noise = RelaxationNoise(t1 = T_1, t2 = T_2) # Define Noise based on given T1 and T2 time

    Environment = Processor(N = N_q) # Initialize Environment using number of Qubits 

    Environment.add_drift(H_Drift, targets = targets) # Specify custom Drift Hamiltonian for the Processor 

    for operator in H_Control:
        Environment.add_control(operator, targets = targets) # Specify all operators in H_Control as Control Hamiltonians 

    Environment.set_all_tlist(SimulatorTimespace) # Set time space of the processor 

    Environment.add_noise(noise = Noise) # Add Relaxation Noise to processor 

    return Environment

def run_pulses(Environment, InitialState, Pulses):

    """
    Send pulses to a specific qutip processor instance and simulate result 

    Parameters
    ----------

    Environment : Instance of Qutip Processor (qutip.qip.device.processor.Processor object)

    InitialState : Initial state of the quantum state (Qobj)

    Pulses : (K x I_G) Array containing amplitudes of operators in Control Hamiltonian (np.array)


    Returns 
    ----------

    result : Result instance of Environment using specified Initial State and Pulse set (np.array (T x N_q^2))
    """

    for i in range(len(Pulses[:, 0])):
        Environment.pulses[i].coeff = Pulses[i] # Pass all amplitudes in Pulses array to the Processor 

    result = Environment.run_state(init_state = InitialState) # Simulate result with specified Environment, Pulses and Initial State

    Environment.plot_pulses() # Plot Pulses to check expected result

    return result

def calculate_fidelity_reward(Result, InitialState, U_Target):

    """
    Calculates Fidelity Reward for a specific Qutip result 

    Parameters
    ----------

    Result : Result instance of Environment using specified Initial State and Pulse set (np.array (T x N_q^2))

    InitialState : InitialState : Initial state of the quantum state (Qobj)

    U_Target : Target Unitary Evolution Operator

    Returns
    ----------

    r_f : Fidelity Reward for a specific Qutip result, Intital State and Target Unitary (float)
    """

    SV_Sim = Result.states[-1] # Define final State Vector (final time step)

    DM_Sim = SV_Sim * SV_Sim.dag() # Calculate simulated density matrix from state vector (a * a.dag())

    Qutip_U_Target = Qobj(U_Target) # Convert Target Unitary to a Qobj Object

    SV_Target = Qutip_U_Target * InitialState # Calculate Target State Vector based on Initial State and Target Unitary 

    DM_Target = SV_Target * SV_Target.dag() # Calculate target density matrix from state vector (a * a.dag())

    r_f = fidelity(DM_Sim, DM_Target) # Calculate fidelity between simulated and target density matrix as reward

    return r_f

