import numpy as np

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from actor_critic import PolicyNetwork, ValueNetwork






def convert_arr_to_tensor(obs:np.ndarray):
    assert (type(obs) is np.ndarray)
    return torch.from_numpy(obs)

def get_agent_observation_as_tensor(all_agents_obs:dict, agent_id:str):
    '''Takes in entire observations in dict, returning observations for particular agent in tensor'''
    if agent_id not in all_agents_obs.keys():
        raise ValueError(f"Agent {agent_id} not present in observation dict")

    agent_obs = all_agents_obs[agent_id]
    return convert_arr_to_tensor(agent_obs)

class PPO:
    def __init__(self, multi_agent_par_env, config):
        self.multi_agent_env = multi_agent_par_env
        self.config = config 

        # configure a neural network for each agent 
        self.agents_neuralnetwork = [self._init_agent(self.config.state_dim, self.config.action_dim, \
                                                      self.config.no_hidden_layers, self.config.lr) \
                                     for _ in range(multi_agent_par_env.max_num_agents)]

        self.agent_ids = multi_agent_par_env.possible_agents # list 
        self.num_agents = multi_agent_par_env.max_num_agents # int 

        self.batch_size = config.no_episodes * config.rollout_length # this is the size of the flattened vector for one agent, before dividing into mini-batches  
        assert(self.batch_size > config.minibatch_size), "Batch size (no_episodes * self.config.rollout_length) must be greater than minibatch size"
        
        self.total_steps = 0
        self.training_iter_count = 0 # Tracks the number of training iteration (a training iteration is optimisation of a generated batch of episodes)

        self.no_minibatches = self.batch_size // config.minibatch_size # predicting size, may be useful some day

        if config.play_only:
            self.load_models()

    def load_models(self):
        for i in range(self.num_agents):
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
    
    def sample_policy_action(self, observations) -> torch.Tensor :
        policy_network = self.get_policy_network()
        return policy_network(observations)
    
    def sample_policy_actions(self, observations:dict) -> dict:
        '''Takes in observations of all agents, returning the action vectors for use in environment'''
        return {id: self.sample_policy_action(observations[id]) for i, id in enumerate(self.agent_ids)}

    def generate_value(self, agent_i, agent_observations:torch.Tensor) -> np.ndarray: 
        value_network = self.get_value_network(agent_i)
        pred_value = value_network(agent_observations) 
        return pred_value.detach().numpy()

    def generate_single_trajectory(self, observations:dict) -> tuple:
        '''Expecting observations for all agents from multi-agent parallel environment setup.
        Ensure to pass in the current observations of all agents in env'''

        observation_trajectories = np.zeros((self.num_agents, self.config.rollout_length, self.config.state_dim))
        action_trajectories = np.zeros((self.num_agents, self.config.rollout_length))
        reward_trajectories = np.zeros((self.num_agents, self.config.rollout_length))
        pred_prob_trajectories = np.zeros((self.num_agents, self.config.rollout_length))

        for t in range(self.config.rollout_length): # we step the environment simultaneously for all traffic signals

            agents_actions = {agent_id:None for agent_id in self.agent_ids} # initialise agent actions for every step

            # print('Generating trajectory - at timestep: ', t) # logging purposes

            for i, id in enumerate(self.agent_ids):
                # get immediate action from policy network
                # print("inner loop agent_id: ", id)

                agent_obs = get_agent_observation_as_tensor(observations, agent_id=id)

                pred_probs = self.sample_policy_action(i, agent_obs) # Each agent will sample from its own policy
                action = pred_probs.argmax()
                
                # append data to all-agent actions buffer 
                agents_actions[id]= int(action) # update this, as next it will go in the step() func

                # append data to buffers 
                observation_trajectories[i][t] = agent_obs.numpy()
                action_trajectories[i][t] = action
                pred_prob_trajectories[i][t] = pred_probs.max().detach().numpy()
            
            # step the environment
            observations, rewards, terminations, truncations, infos = self.multi_agent_env.step(agents_actions) # takes in a dictionary of all agents + their corresponding actions

            # store rewards recieved, for every agent 
            for i, id in enumerate(self.agent_ids):
                reward_trajectories[i][t] = rewards[id]

        return observation_trajectories, action_trajectories, pred_prob_trajectories, reward_trajectories

    def generate_episodes(self, observations) -> tuple:
        '''Generate multiple episodes returning obs, actions, rewards, advantages tensors over all the episodes'''
        no_episodes = self.config.no_episodes

        ep_obs = np.zeros((self.num_agents, self.config.no_episodes, self.config.rollout_length, self.config.state_dim))
        ep_actions = np.zeros((self.num_agents, self.config.no_episodes, self.config.rollout_length))
        ep_rewards = np.zeros((self.num_agents, self.config.no_episodes, self.config.rollout_length))

        ep_returns = np.zeros((self.num_agents, self.config.no_episodes, self.config.rollout_length))
        ep_pred_values = np.zeros((self.num_agents, self.config.no_episodes, self.config.rollout_length))
        ep_advantages = np.zeros((self.num_agents, self.config.no_episodes, self.config.rollout_length)) # returns - pred_values

        ep_pred_probs = np.zeros((self.num_agents, self.config.no_episodes, self.config.rollout_length))

        for ep_i in range(self.config.no_episodes): 

            # print(f"Generating trajectory {ep_i})
            observations_trajec, actions_trajec, pred_prob_trajec, rewards_trajec = self.generate_single_trajectory(observations)

            for agent_i, agent_id in enumerate(self.agent_ids):

                reversed_returns =  np.zeros((self.config.rollout_length))
                reversed_pred_values = np.zeros((self.config.rollout_length))

                reversed_advantages = np.zeros((self.config.rollout_length)) # returns - pred_values(V(s))

                running_returns = 0

                for t in reversed(range(self.config.rollout_length)): # t is reversed here 
                    
                    # calculate returns from rewards 
                    rewards = rewards_trajec[agent_i][t]
                    running_returns += rewards # we have not implemented a 1 step return, currently it consists of entire return. 
                    reversed_returns[t] = running_returns 

                    # calculate predicted values from value network at time t 
                    agent_observations_t = observations_trajec[agent_i][t]
                    agent_pred_value = self.generate_value(agent_i, agent_observations_t) # simple forward pass in network to calculate value of state. 

                    # calculate advantage at time t 
                    advantage = running_returns - agent_pred_value

                    reversed_advantages[t] = advantage
                    reversed_pred_values[t] = agent_pred_value

                # reverse all arrays
                advantages = reversed_advantages[::-1]
                returns = reversed_returns[::-1] 
                pred_values = reversed_pred_values[::-1]

                # append [t] arrays to episodic arrays
                ep_advantages[agent_i][ep_i] = advantages 
                ep_returns[agent_i][ep_i] = returns
                ep_pred_values[agent_i][ep_i] = pred_values

                # append other data to large episode tensor
                ep_obs[agent_i][ep_i] = observations_trajec[agent_i]
                ep_actions[agent_i][ep_i] = actions_trajec[agent_i]
                ep_rewards[agent_i][ep_i] = rewards_trajec[agent_i]
                ep_pred_probs[agent_i][ep_i] = pred_prob_trajec[agent_i]

        return (ep_obs, ep_actions, ep_pred_probs, ep_rewards, ep_returns, ep_pred_values, ep_advantages)

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
                                         for i in range(self.num_agents)]


    def mini_batch_update_agent_network(self, agent_enumer, observations_batch, pred_probs_batch, returns_batch, pred_values_batch, advantages_batch):
        '''Expects data in tensor format, returns in tensor format'''
    
        agent_i, agent_id = agent_enumer # unwrap tuple(int,str)

        old_pred_probs_batch = torch.zeros((len(observations_batch)), dtype=float)
        
        # agent_neuralnetwork_old = self.agents_neuralnetwork_old[agent_i]
        agent_neuralnetwork_old = self.agents_neuralnetwork[agent_i]

        # calc predicted probability from the old network - do it for [obs1, obs2, obs3, ...]
        for i, observations in enumerate(observations_batch):
            # print(i, observations)
            old_pred_probs_batch[i] = agent_neuralnetwork_old['policy'](observations).max()
        
        policy_loss = self.__compute_policy_loss(old_pred_probs_batch, pred_probs_batch, advantages_batch)  
        
        # Backpropagate policy loss
        policy_optimiser = self.agents_neuralnetwork[agent_i]['policy_opt']
        
        policy_optimiser.zero_grad()
        policy_loss.backward()
        policy_optimiser.step()

        # Calculate value loss
        value_loss = self.__compute_value_loss(returns_batch, pred_values_batch)

        # Backpropagate value loss
        value_optimiser = self.agents_neuralnetwork[agent_i]['value_opt']
        
        value_optimiser.zero_grad()
        value_loss.backward()
        value_optimiser.step()

        self.agent_update_count[agent_i] += 1

        return policy_loss, value_loss 

    def reset_batch_stats(self):
        self.agent_update_count = [0] * 4 # update steps for each agent, safely assuming every mini-batch processed is an update. 
        self.sum_returns = [0] * 4
        self.sum_advantages = [0] * 4
        self.sum_policy_loss = [0] * 4
        self.sum_critic_loss = [0] * 4
        self.sum_entropy = [0] * 4

    def train(self):
        '''Perform a training epochs using same data with different shuffles of minibatches. 
            Calling this function will generate a new trajectories for each agent, 
            and update the network using minibatches in multiple epochs'''  
        
        print(f"\n ---------------| Starting training iteration {self.training_iter_count} |---------------\n")
        
        observations, truncations = self.multi_agent_env.reset() # reset env for every training iteration 

        config = self.config
        
        self.reset_batch_stats() # resets the batch statistics again for each training iteration 

        episodic_data = self.generate_episodes(observations)
        print(f"\n ---------------|  Finished generating {config.no_episodes} episodes |---------------\n")

        for ep in range(config.optimisation_epochs):
            
            print("\nStarting optimisation_epoch no ", ep)

            sampler = self.generate_sampler()
            episodic_data = self.update_agents_networks(sampler, episodic_data)
        

        # assuming no_updates_total is same for all agents, so simply slice the counts for the first
        assert (self.agent_update_count[0]==self.agent_update_count[1]), "No of agent updates are not equal"
        self.total_steps += self.agent_update_count[0]
        
        print(self.agent_update_count)
        print(self.total_steps)
        
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


    def flatten_data(self, episodic_data, agent_i):
        ep_obs, ep_actions, ep_pred_probs, ep_rewards, ep_returns, ep_pred_values, ep_advantages = episodic_data

        agent_obs = ep_obs[agent_i].reshape(-1,84)

        agent_returns = ep_returns[agent_i].reshape(-1) # (no_epi * time_steps) 1dim
        agent_pred_values = ep_pred_values[agent_i].reshape(-1)

        agent_pred_probs = ep_pred_probs[agent_i].reshape(-1) # (no_epi, time_steps, dim) -> (no_epi*timesteps, dim)
        agent_advantages = ep_advantages[agent_i].reshape(-1)

        return (agent_obs, agent_returns, agent_pred_values, agent_pred_probs, agent_advantages)


    def update_agents_networks(self, sampler, episodic_data):
        '''Takes in the episodes data, iterates through data, flattens it, then processes mini-batches and perform updates on them '''

        for agent_i, agent_id in enumerate(self.agent_ids): # update network for every agent

            agent_obs, agent_returns, agent_pred_values, agent_pred_probs, agent_advantages = self.flatten_data(episodic_data, agent_i)

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

        return episodic_data

    def log_stats(self, agent_ids, policy_loss, value_loss, returns, advantages):
        # track statistics
        agent_i, agent_id = agent_ids

        self.sum_returns[agent_i] += returns.mean()
        self.sum_advantages[agent_i] += advantages.mean()
    
        logger = self.config.logger

        step_idx = self.agent_update_count[agent_i] + self.total_steps
    
        logger.add_scalar(f"policy_loss/agent_{agent_id}", policy_loss.item(), step_idx)
        logger.add_scalar(f"value_loss/agent_{agent_id}", value_loss.item(), step_idx)
        
        logger.add_scalar(f"average_returns/agent_{agent_id}", self.sum_returns[agent_i] / self.agent_update_count[agent_i], step_idx)
        logger.add_scalar(f"average_advantages/agent_{agent_id}", self.sum_advantages[agent_i] / self.agent_update_count[agent_i], step_idx)

        return 


