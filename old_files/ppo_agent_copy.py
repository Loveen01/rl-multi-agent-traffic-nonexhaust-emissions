import numpy as np

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from utils import util 

from actor_critic import PolicyNetwork, ValueNetwork

class PPO:
    def __init__(self, multi_agent_par_env, config):
        self.multi_agent_env = multi_agent_par_env
        
        self.batch_size = config.no_episodes * config.rollout_length # this is the size of the flattened vector for one agent, before dividing into mini-batches  
        
        assert(self.batch_size > config.minibatch_size), "Batch size (no_episodes * self.config.rollout_length) must be greater than minibatch size"
        self.config = config 

        self.no_minibatches = self.batch_size // self.config.minibatch_size # predicting size, may be useful some day

        # configure a neural network for each agent 
        self.agents_neuralnetwork = [self._init_agent(self.config.state_dim, self.config.action_dim, \
                                                      self.config.no_hidden_layers, self.config.lr) \
                                     for _ in range(multi_agent_par_env.max_num_agents)]

        self.AGENT_IDS = multi_agent_par_env.possible_agents # list 
        self.NUM_AGENTS = multi_agent_par_env.max_num_agents # int 

        # counters 
        self.total_steps = 0
        self.training_iter_count = 0 # Tracks the number of training iteration (a training iteration is optimisation of a generated batch of episodes)
        self.episode_count = 0
        
        self.reset_batch_stats()

        if config.play_only:
            self.load_models()

        # store all trajectory data 
        self.ep_obs = np.zeros((self.NUM_AGENTS, self.config.no_episodes, self.config.rollout_length, self.config.state_dim))
        self.ep_actions = np.zeros((self.NUM_AGENTS, self.config.no_episodes, self.config.rollout_length))
        self.ep_rewards = np.zeros((self.NUM_AGENTS, self.config.no_episodes, self.config.rollout_length))

        self.ep_returns = np.zeros((self.NUM_AGENTS, self.config.no_episodes, self.config.rollout_length))
        self.ep_pred_values = np.zeros((self.NUM_AGENTS, self.config.no_episodes, self.config.rollout_length))
        self.ep_advantages = np.zeros((self.NUM_AGENTS, self.config.no_episodes, self.config.rollout_length)) # returns - pred_values

        self.ep_action_probs = np.zeros((self.NUM_AGENTS, self.config.no_episodes, self.config.rollout_length))


    def load_models(self):
        for i in range(self.NUM_AGENTS):
            self.agents_neuralnetwork[i]["policy"].load_state_dict(torch.load(self.config.saved_checkpoint)) 
            self.agents_neuralnetwork[i]["policy"].to('cuda:0')

    def _init_agent(self, state_dim, action_dim, no_hidden_layers, lr):
        policy_net = PolicyNetwork(state_dim, action_dim, no_hidden_layers)
        value_net = ValueNetwork(state_dim, no_hidden_layers) 
        policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        value_optimizer = optim.Adam(value_net.parameters(), lr=lr)
        return {"policy": policy_net, "value": value_net, "policy_opt": policy_optimizer, "value_opt": value_optimizer}

    def get_policy_network(self, agent_i:int):
        return self.agents_neuralnetwork[agent_i]['policy']

    def get_value_network(self, agent_i:int):
        return self.agents_neuralnetwork[agent_i]['value']
    
    def get_policy_optimiser(self, agent_i:int):
        return self.agents_neuralnetwork[agent_i]['policy_opt']

    def get_value_optimiser(self, agent_i:int):
        return self.agents_neuralnetwork[agent_i]['value_opt']
    
    def sample_policy_action(self, agent_i:int, agent_observations:torch.Tensor) -> torch.Tensor :
        policy_network = self.get_policy_network(agent_i)
        return policy_network(agent_observations)
    
    def generate_value(self, agent_i, agent_observations:torch.Tensor) -> np.ndarray: 
        value_network = self.get_value_network(agent_i)
        pred_value = value_network(agent_observations) 
        return pred_value.detach().numpy()

    def generate_single_trajectory(self, observations:dict) -> tuple:
        '''Expecting observations for all agents from multi-agent parallel environment setup.
        Ensure to pass in the current observations of all agents in env'''

        for t in range(self.config.rollout_length): # we step the environment simultaneously for all traffic signals

            all_actions = {agent_id:None for agent_id in self.AGENT_IDS} # initialise agent actions for every step

            # print('Generating trajectory - at timestep: ', t) # logging purposes

            for agent_i, agent_id in enumerate(self.AGENT_IDS):
                # print("inner loop agent_id: ", id)

                agent_obs = util.get_agent_observation_as_tensor(observations, agent_id)

                action_probs = self.sample_policy_action(agent_i, agent_obs) # Each agent will sample from its own policy. This is a list 
                action = action_probs.argmax()
                
                # append data to all-agent actions dict
                all_actions[agent_id]= int(action) # update this, as next it will go in the step() func

                # append data to buffers 
                self.ep_obs[agent_i][self.episode_count][t] = agent_obs.numpy()
                self.ep_actions[agent_i][self.episode_count][t] = action
                self.ep_action_probs[agent_i][self.episode_count][t] = action_probs.max().detach().numpy()
            
            # step the environment
            next_observations, rewards, terminations, truncations, infos = self.multi_agent_env.step(all_actions) # takes in a dictionary of all agents + their corresponding actions

            # store info to buffer after env stepped, for every agent 
            for agent_i, agent_id in enumerate(self.AGENT_IDS):
                self.ep_rewards[agent_i][self.episode_count][t] = rewards[agent_id]

            observations = next_observations
        
        # next steps: this function should append to trajectory buffer, and that buffer once full should append to the main episodic buffer after capacity is reached 

        # return observation_trajectories, action_trajectories, pred_prob_trajectories, reward_trajectories

    def generate_episodes(self, observations) -> tuple:
        '''Generate multiple episodes returning obs, actions, rewards, advantages tensors over all the episodes'''

        for ep_i in range(self.config.no_episodes): 
            # print(f"Generating trajectory {ep_i})
            self.generate_single_trajectory(observations) 
            self.episode_count += 1

            for agent_i, agent_id in enumerate(self.AGENT_IDS): 
                # todo: 

                reversed_returns =  np.zeros((self.config.rollout_length))
                reversed_pred_values = np.zeros((self.config.rollout_length))
                reversed_advantages = np.zeros((self.config.rollout_length)) # returns - pred_values(V(s))

                running_returns = 0

                for t in reversed(range(self.config.rollout_length)): 
                    # calculate returns from rewards 
                    reward_t = self.ep_rewards[agent_i][ep_i][t]
                    running_returns += reward_t # we have not implemented a 1 step return, currently it consists of entire return. 
                    reversed_returns[t] = running_returns 

                    # calculate predicted values from value network at time t 
                    agent_observations_t = self.ep_obs[agent_i][ep_i][t]
                    agent_pred_value = self.generate_value(agent_i, agent_observations_t) # simple forward pass in network to calculate value of state. 

                    # actual return - predicted value 
                    advantage = running_returns - agent_pred_value

                    # 
                    reversed_advantages[t] = advantage
                    reversed_pred_values[t] = agent_pred_value

                # reverse all arrays after rollout ended 
                advantages = reversed_advantages[::-1]
                returns = reversed_returns[::-1] 
                pred_values = reversed_pred_values[::-1]

                # append all arrays to episodic arrays
                self.ep_advantages[agent_i][ep_i] = advantages 
                self.ep_returns[agent_i][ep_i] = returns
                self.ep_pred_values[agent_i][ep_i] = pred_values

        # return (ep_obs, ep_actions, ep_pred_probs, ep_rewards, ep_returns, ep_pred_values, ep_advantages)

    def __compute_policy_loss(self, old_log_probs, new_log_probs, advantages):
        '''Takes sequence of log_probs and advantages in tensors, calculates J(0) which is the prob ratios * advantages'''

        # Calculate the ratio of new and old probabilities
        ratios = torch.exp(new_log_probs - old_log_probs)

        # Calculate policy loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.config.ppo_ratio_clip, 1 + self.config.ppo_ratio_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()  # Negative because we perform gradient ascent

        return policy_loss
    
    def __compute_value_loss(self, actual_returns, predicted_values) -> torch.Tensor:

        # Mean squared error loss between predicted and actual returns
        value_loss = F.mse_loss(predicted_values, actual_returns)

        return value_loss
    
    def update_agents_neuralnetwork_old(self):
        '''This function does a deep copy of the current agent policies, returning dictionaries'''

        self.agents_neuralnetwork_old = [{"policy": deepcopy(self.agents_neuralnetwork[i]['policy']), "value": deepcopy(self.agents_neuralnetwork[i]['value']), \
                                         "policy_opt": deepcopy(self.agents_neuralnetwork[i]['policy_opt']), \
                                            "value_opt": deepcopy(self.agents_neuralnetwork[i]['value_opt'])}\
                                         for i in range(self.NUM_AGENTS)]


    def mini_batch_update_agent_network(self, agent_enumer, observations_batch, pred_probs_batch, returns_batch, pred_values_batch, advantages_batch):
        '''Expects data in tensor format, returns in tensor format'''
    
        agent_i, agent_id = agent_enumer # unwrap tuple(int,str)

        old_pred_probs_batch = torch.zeros((len(observations_batch)), dtype=float)
        
        # agent_neuralnetwork_old = self.agents_neuralnetwork_old[agent_i]
        agent_neuralnetwork_old = self.agents_neuralnetwork[agent_i]

        # calc predicted probability from the old network - do it for [obs1, obs2, obs3, ...]
        for i, observations in enumerate(observations_batch):
            # print(i, observations) 
            policy_network = self.get_policy_network(agent_i)
            old_pred_probs_batch[i] = policy_network(observations).max()
        
        policy_loss = self.__compute_policy_loss(old_pred_probs_batch, pred_probs_batch, advantages_batch)  
        
        # Backpropagate policy loss
        policy_optimiser = self.get_policy_optimiser(agent_i)
        
        policy_optimiser.zero_grad()
        policy_loss.backward()
        policy_optimiser.step()

        # Calculate value loss
        value_loss = self.__compute_value_loss(returns_batch, pred_values_batch)

        # Backpropagate value loss
        value_optimiser = self.get_value_optimiser(agent_i)
        
        value_optimiser.zero_grad()
        value_loss.backward()
        value_optimiser.step()

        self.agent_update_count[agent_i] += 1

        return policy_loss, value_loss 

    def reset_batch_stats(self):
        self.episode_count = 0
        self.agent_update_count = [0] * 4 # update steps for each agent, safely assuming every mini-batch processed is an update. 
        self.sum_returns = [0] * 4
        self.sum_advantages = [0] * 4
        self.sum_policy_loss = [0] * 4
        self.sum_critic_loss = [0] * 4
        self.sum_entropy = [0] * 4

    def reset_buffer(self):
        # store all trajectory data 
        self.ep_obs = np.zeros((self.NUM_AGENTS, self.config.no_episodes, self.config.rollout_length, self.config.state_dim))
        self.ep_actions = np.zeros((self.NUM_AGENTS, self.config.no_episodes, self.config.rollout_length))
        self.ep_rewards = np.zeros((self.NUM_AGENTS, self.config.no_episodes, self.config.rollout_length))

        self.ep_returns = np.zeros((self.NUM_AGENTS, self.config.no_episodes, self.config.rollout_length))
        self.ep_pred_values = np.zeros((self.NUM_AGENTS, self.config.no_episodes, self.config.rollout_length))
        self.ep_advantages = np.zeros((self.NUM_AGENTS, self.config.no_episodes, self.config.rollout_length)) # returns - pred_values

        self.ep_action_probs = np.zeros((self.NUM_AGENTS, self.config.no_episodes, self.config.rollout_length))


    def train(self):
        '''Perform a training epochs using same data with different shuffles of minibatches. 
            Calling this function will generate a new trajectories for each agent, 
            and update the network using minibatches in multiple epochs'''  
        
        print(f"\n ---------------| Starting training iteration {self.training_iter_count} |---------------\n")
                
        self.reset_batch_stats() # resets the batch statistics again for each training iteration 
        self.reset_buffer() # reset buffers

        observations, truncations = self.multi_agent_env.reset() # reset env for every training iteration 

        self.generate_episodes(observations)

        print(f"\n ---------------|  Buffer full, Finished generating {self.config.no_episodes} episodes |---------------\n")

        for ep in range(self.config.optimisation_epochs):   
            print("\nStarting optimisation_epoch no ", ep)

            sampler = self.generate_sampler()
            self.update_agents_networks(sampler)

        # assuming no_updates_total is same for all agents, so simply slice the counts for the first
        assert (self.agent_update_count[0]==self.agent_update_count[1]), "No of agent updates are not equal"
        self.total_steps += self.agent_update_count[0]
        
        print(f"Agent update count {self.agent_update_count}")
        print(f"Total updates so far: {self.total_steps}")
        
        self.training_iter_count+=1
        # # this depends on what is maximum as configured by user? is it the batch steps per agent 
        # if self.total_steps % config.max_steps == 0:
        #     print("Reached maximum steps in total, max_steps = total batch updates over epochs + training iterations")
        #     self.validate(False)
        
        return


    def generate_sampler(self):
        # episodic_obs (no_agents, no_epi, time_steps, dim)
        subset_sampler = SubsetRandomSampler(range(self.batch_size)) # random assort integers from 1 - 84, put in list [3, 4, 9, 84, ...]
        # divide this list into batches of size minibatch_size [1, 3, ..], [54, 76, 2..]
        sampler = BatchSampler(subset_sampler, self.config.minibatch_size, True) 
        return sampler 


    def get_agent_data_and_flatten(self, agent_i):

        agent_obs = self.ep_obs[agent_i].reshape(-1,84)

        agent_returns = self.ep_returns[agent_i].reshape(-1) # (no_epi * time_steps) 1dim
        agent_pred_values = self.ep_pred_values[agent_i].reshape(-1)

        agent_pred_probs = self.ep_action_probs[agent_i].reshape(-1) # (no_epi, time_steps, dim) -> (no_epi*timesteps, dim)
        agent_advantages = self.ep_advantages[agent_i].reshape(-1)

        return (agent_obs, agent_returns, agent_pred_values, agent_pred_probs, agent_advantages)


    def update_agents_networks(self, sampler):
        '''Takes in the episodes data, iterates through data, flattens it, then processes mini-batches and perform updates on them '''

        for agent_i, agent_id in enumerate(self.AGENT_IDS): # update network for every agent

            agent_obs, agent_returns, agent_pred_values, agent_pred_probs, agent_advantages = self.get_agent_data_and_flatten(agent_i)

            for k, indices in enumerate(sampler):

                agent_obs_batch = torch.tensor(agent_obs[indices]) # shape = (len(indices), 84)

                agent_pred_probs_batch = torch.tensor(agent_pred_probs[indices], requires_grad=True)
                agent_advantages_batch = torch.tensor(agent_advantages[indices], requires_grad=True)

                agent_returns_batch = torch.tensor(agent_returns[indices], requires_grad=True)
                agent_pred_values_batch = torch.tensor(agent_pred_values[indices], requires_grad=True)

                policy_loss, value_loss = self.mini_batch_update_agent_network((agent_i, agent_id), agent_obs_batch, agent_pred_probs_batch, \
                                                                agent_returns_batch, agent_pred_values_batch, agent_advantages_batch,)
                
                self.log_stats((agent_i, agent_id), policy_loss, value_loss, agent_returns_batch, agent_advantages_batch)
            
        print(f"Finished updating networks for all agents")

        return 

    def log_stats(self, agent_enumer, policy_loss, value_loss, returns, advantages):
        # track statistics
        agent_i, agent_id = agent_enumer

        self.sum_returns[agent_i] += returns.mean()
        self.sum_advantages[agent_i] += advantages.mean()
    
        logger = self.config.logger

        step_idx = self.agent_update_count[agent_i] + self.total_steps
    
        logger.add_scalar(f"policy_loss/agent_{agent_id}", policy_loss.item(), step_idx)
        logger.add_scalar(f"value_loss/agent_{agent_id}", value_loss.item(), step_idx)
        
        logger.add_scalar(f"average_returns/agent_{agent_id}", self.sum_returns[agent_i] / self.agent_update_count[agent_i], step_idx)
        logger.add_scalar(f"average_advantages/agent_{agent_id}", self.sum_advantages[agent_i] / self.agent_update_count[agent_i], step_idx)

        return 


