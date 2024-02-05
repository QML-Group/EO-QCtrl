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
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tqdm.auto import trange

class QuantumRLAgent:

    def __init__(self, TrainEnvironment, EvaluationEnvironment, num_iterations, num_cycles = 1, fc_layer_params = (100, 100, 100), learning_rate = 1e-3, collect_episodes_per_iteration = 1, eval_interval = 1, replay_buffer_capacity = 10):
        
        """
        QuantumRLAgent Class
        ---------------------

        Create instance of a reinforcement learning agent interacting with a QuantumEnvironment Instance

        Parameters
        ----------

        TrainEnvironment : class
            Quantum Environment Instance

        EvaluationEnvironment : class
            Quantum Environment Instance

        num_iterations : int
            Number of training loop iterations
        """

        self.env_train_py = TrainEnvironment
        self.env_eval_py = EvaluationEnvironment 
        self.num_cycles = num_cycles
        self.env_train_py.n_steps = self.num_cycles
        self.env_eval_py.n_steps = self.num_cycles
        self.num_iterations = num_iterations
        self.fc_layer_params = fc_layer_params
        self.learning_rate = learning_rate
        self.collect_episodes_per_iteration = collect_episodes_per_iteration
        self.eval_interval = eval_interval
        self.replay_buffer_capacity = replay_buffer_capacity

        self.create_network_agent()

    def create_network_agent(self):

        """
        Create Neural Network and Agent Instance based on Quantum Environment Class
        """

        self.train_env = tf_py_environment.TFPyEnvironment(self.env_train_py)
        self.eval_env = tf_py_environment.TFPyEnvironment(self.env_eval_py)

        self.actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            fc_layer_params = self.fc_layer_params
        )

        self.optimizer = keras.optimizers.Adam(learning_rate = self.learning_rate)
        
        self.train_step_counter = tf.compat.v2.Variable(0)

        self.tf_agent = reinforce_agent.ReinforceAgent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            actor_network = self.actor_net,
            optimizer = self.optimizer,
            normalize_returns = True,
            train_step_counter = self.train_step_counter
        )

        self.tf_agent.initialize()

        self.eval_policy = self.tf_agent.policy
        self.collect_policy = self.tf_agent.collect_policy

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec = self.tf_agent.collect_data_spec,
            batch_size = self.train_env.batch_size,
            max_length = self.replay_buffer_capacity
        )

        self.eval_replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec = self.tf_agent.collect_data_spec,
            batch_size = self.eval_env.batch_size,
            max_length = self.num_cycles + 1 
        )

        #self.avg_return = tf_metrics.AverageReturnMetric()

        self.avg_return = tf_metrics.MaxReturnMetric()

        self.eval_observers = [self.avg_return, self.eval_replay_buffer.add_batch]
        self.eval_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.eval_env,
            self.eval_policy,
            self.eval_observers,
            num_episodes = 1
        )
        
        self.train_observers = [self.replay_buffer.add_batch]
        self.train_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.train_env,
            self.collect_policy,
            self.train_observers,
            num_episodes = self.collect_episodes_per_iteration
        )

        self.tf_agent.train = common.function(self.tf_agent.train)

    def run_training(self, save_episodes = True, clear_buffer = False):

        """
        Starts training on Quantum RL Agent

        Parameters
        ----------

        save_episodes : bool : True
            Saves episodes if set to True

        clear_buffer : bool : False
            Clears buffer each episode if set to True
        """

        self.return_list = []
        self.episode_list = []
        self.iteration_list = []
 

        with trange(self.num_iterations, dynamic_ncols = False) as t:

            # Probably here need to write another for loop, where for each loop we generate a new initial state
            # new_initial_state = qutip.random_objects.rand_ket(N = 4)
            # self.env_train_py.initial_state = new_initial_state
            # self.env_eval_py.initial_state = new_initial_state

            for i in t:

                t.set_description(f"Episode {i}")

                if clear_buffer:
                    self.replay_buffer.clear()

                final_time_step, policy_state = self.train_driver.run()
                experience = self.replay_buffer.gather_all()
                train_loss = self.tf_agent.train(experience)

                if i % self.eval_interval == 0 or i == self.num_iterations - 1:

                    self.avg_return.reset()
                    final_time_step, policy_state = self.eval_driver.run()

                    self.iteration_list.append(self.tf_agent.train_step_counter.numpy())
                    self.return_list.append(self.avg_return.result().numpy())

                    t.set_postfix({"Return" : self.return_list[-1]})

                    if save_episodes:
                        self.episode_list.append(self.eval_replay_buffer.gather_all())

    def plot_fidelity_return_per_episode(self):

        """
        Plots the average fidelity and return per episode versus number of episode
        """

        avg_eval_reward_per_episode = []

        for i in range(self.num_iterations):
            sum_eval = np.sum(self.env_eval_py.fidelity_list[self.num_cycles * i : self.num_cycles + self.num_cycles * i])
            avg_eval_reward_per_episode.append(sum_eval/self.num_cycles)

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax2.axhline(y = 0, color = "grey")
        ax1.plot(self.iteration_list, avg_eval_reward_per_episode, label = "Fidelity", marker = "d", color = "#03080c", markevery = 20)
        ax2.plot(self.iteration_list, self.return_list, label = "Return", marker = "d", color = "#5b97ca", markevery = 20)
        ax1.set_ylabel("Fidelity")
        ax2.set_ylabel("Return")
        ax1.set_xlabel("Episode number")
        ax1.legend(loc = (0.7, 0.45))
        ax2.legend(loc = (0.7, 0.55))
        fig.tight_layout()
        #ax1.grid()
        plt.show()

    def plot_fidelity_reward_per_iteration(self):

        """
        Plots fidelity per iteration
        """

        self.iteration_space = np.linspace(1, self.num_cycles * self.num_iterations, self.num_cycles * self.num_iterations)
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax2.axhline(y = 0, color = "grey")
        ax1.plot(self.iteration_space, self.env_eval_py.fidelity_list, label = "Fidelity", marker = "d", color = "#03080c", markevery = 20)
        ax2.plot(self.iteration_space, self.env_eval_py.reward_list, ls = "dashed", label = "Reward", marker = "d", color = "#5b97ca", markevery = 20)
        ax1.set_xlabel("Episode number")
        ax1.set_ylabel("Fidelity")
        ax2.set_ylabel("Reward")
        ax1.legend(loc = (0.7, 0.45))
        ax2.legend(loc = (0.7, 0.55))
        fig.tight_layout()
        plt.show()

    def get_final_pulse(self):

        """
        Returns final action of agent
        """

        self.final_val = self.episode_list[-1]
        self.final_pulse = self.final_val.action.numpy()[0, 0, :]
        self.pulse_2d = np.reshape(self.final_pulse, (len(self.env_train_py.h_control), self.env_train_py.timesteps))

        return self.pulse_2d
    
    def get_highest_fidelity_pulse(self):

        """
        Returns the pulse with the highest fidelity
        """
    
        self.max_index = self.env_eval_py.fidelity_list.index(max(self.env_eval_py.fidelity_list))
        self.max_val = self.episode_list[self.max_index]
        self.max_pulse = self.max_val.action.numpy()[0, 0, :]
        self.max_pulse_2d = np.reshape(self.max_pulse, (len(self.env_train_py.h_control), self.env_train_py.timesteps))

        return self.max_pulse_2d

    def plot_final_pulse(self):

        """
        Plots the final action generated by the RL agent 
        """

        self.timespace = np.linspace(0, self.env_eval_py.pulse_duration, self.env_eval_py.timesteps)

        self.final_val = self.episode_list[-1]
        self.final_pulse = self.final_val.action.numpy()[0, 0, :]

        self.pulse_2d = np.reshape(self.final_pulse, (len(self.env_train_py.h_control), self.env_train_py.timesteps))

        for i in range(len(self.env_train_py.h_control)):
            plt.step(self.timespace, self.pulse_2d[i])

        plt.axhline(y = 0, color = "grey", ls = "dashed")
        plt.xlabel("Time (a.u.)")
        plt.ylabel(r"$\sigma_{xx}$")
        plt.grid()
        plt.show()

    def show_summary(self):

        """
        Prints summary of the Actor Network (including number of parameters)
        """

        self.actor_net.summary()

