import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import utils
from actor_critic import Policy, Critic

import time

class PPO_Trainer:
    '''Takes env and configs, networks, and runs the policy - TODO: SEPARAT TRAINING LOGIC AWAY FROM THIS CLASS'''
    def __init__(self, multi_agent_par_env, config, buffer, episode_buffer, logger):
        self.multi_agent_env = multi_agent_par_env

        self.batch_size = config.no_episodes * config.rollout_length # this is the size of the flattened vector for one agent, before dividing into mini-batches  
        assert(self.batch_size > config.minibatch_size), "Batch size (no_episodes * self.config.rollout_length) must be greater than minibatch size"
        
        self.config = config 

        # configure a neural network for each agent [{} {} {} {}]
        self.agents_neuralnetwork = [self._init_agent(self.config.state_dim, self.config.action_dim, \
                                                      self.config.no_hidden_layers, self.config.lr) \
                                     for _ in range(multi_agent_par_env.max_num_agents)]
        
        self.old_policies = []
        self._clone_policies()

        self.agent_ids = multi_agent_par_env.possible_agents # list 
        self.num_agents = multi_agent_par_env.max_num_agents # int 

        self.buffer = buffer
        self.episode_buffer = episode_buffer

        self.logger = logger 

        self.update_count_total = 0 # since start of training 
        self.update_count_per_train_iter = [0] * 4 

        self.no_minibatches = self.batch_size // config.minibatch_size  # predicting size, may be useful some day

    def load_models(self):
        for i in range(self.num_agents):
            self.agents_neuralnetwork[i]["policy"].load_state_dict(torch.load(self.config.saved_checkpoint)) 
            self.agents_neuralnetwork[i]["policy"].to('cuda:0')

    def _init_agent(self, state_dim, action_dim, no_hidden_layers, lr):
        policy = Policy(state_dim, action_dim, no_hidden_layers)
        critic = Critic(state_dim, no_hidden_layers) 
        return {
            "policy": policy, 
            "critic": critic, 
            "policy_opt": optim.Adam(policy.parameters(), lr=lr), 
            "critic_opt": optim.Adam(critic.parameters(), lr=lr)}

    def run(self):
        '''runs 1 training iteration for now. collects trajectories and trains
        Calling this function will generate a new trajectories for each agent, 
            and update the network using minibatches in multiple epochs'''
        start = time.time()

        for iter_i in range(self.config.no_training_iterations):
            print(f"\n -----| Starting training iteration {iter_i} |-----\n")
            self.reset_batch_stats() 
            self.episode_buffer.reset()

            self.__prep_rollout() 
            observations, truncations = self.multi_agent_env.reset()    

            self._generate_and_store_episodes(observations) 

            print(f"\n -----|  Buffer full, Finished generating {self.config.no_episodes} episodes |-----\n")

            train_info = self._train() 
            
            self._log_end_train_info(train_info, self.update_count_total) 

        self.logger.close()

        end = time.time() 

        print(f"taken {end - start} to run")

    def _train(self):
        '''Perform a training epochs using same data with different shuffles of minibatches''' 
        
        self._prep_training() 

        self.reset_batch_stats() 

        train_info = [{'critic_loss': 0, 
                       'policy_loss': 0} for i in range(self.num_agents)] # scalars for every training iter

        for ep in range(self.config.optimisation_epochs):   
            print("\nStarting optimisation_epoch no ", ep)

            # generate new indices at every epoch
            indices_batch = utils.generate_sampler(self.batch_size, self.config.minibatch_size) 
            
            for agent_i, agent_id in enumerate(self.agent_ids): # update network for every agent
                agent_data = self.episode_buffer.flatten_data(agent_i) # agent_obs, agent_pred_values, agent_returns, agent_pred_probs, agent_advantages
                
                for indices in indices_batch:
                    obs_minibatch, pred_probs_minibatch, returns_minibatch, \
                        pred_values_minibatch, advantages_minibatch = self.batch_data(agent_data, indices) 

                    policy_loss, critic_loss, ratios = self._minibatch_ppo_agent_update(agent_i, obs_minibatch, pred_probs_minibatch, \
                                                                    returns_minibatch, pred_values_minibatch, advantages_minibatch,)
                                        
                    agent_step_idx = self.update_count_per_train_iter[agent_i] + self.update_count_total
                    
                    self.logger.add_scalar(f"policy_loss/agent_{agent_id}", policy_loss.item(), agent_step_idx)
                    self.logger.add_scalar(f"critic_loss/agent_{agent_id}", critic_loss.item(), agent_step_idx)
                    self.logger.add_scalar(f"ratios/agent_{agent_id}", ratios.mean(), agent_step_idx)
                    
                    train_info[agent_i]['policy_loss'] += policy_loss.item()
                    train_info[agent_i]['critic_loss'] += critic_loss.item()

        
        self._clone_policies() 

        if self.config.optimisation_epochs * self.no_minibatches != self.update_count_per_train_iter[0]:   # assuming the self.batch_size // self.config.minibatch_size is constant 
            print(f"self.config.optimisation_epochs: ({self.config.optimisation_epochs}) * self.no_minibatches: ({self.no_minibatches}) \
                  is not equal to self.update_count_per_train_iter[0]: ({self.update_count_per_train_iter[0]})")
        
        for agent_i in range(self.num_agents): 
            train_info_agent = train_info[agent_i]
            for k in train_info_agent.keys():
                train_info_agent[k] /= self.update_count_per_train_iter[0]  # average

        self.update_count_total += self.update_count_per_train_iter[0] # after training, update the total update count

        return train_info


    def _log_end_train_info(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        title = "train_iter_end_summary"
        for agent_i, agent_id in enumerate(self.agent_ids):
            self.logger.add_scalar(f'{title}_policy_loss/agent_{agent_id}', train_infos[agent_i]['policy_loss'], total_num_steps)
            self.logger.add_scalar(f'{title}_critic_loss/agent_{agent_id}', train_infos[agent_i]['critic_loss'], total_num_steps)

    def _prep_training(self):
        for agent_i in range(self.num_agents):
            self.agents_neuralnetwork[agent_i]['policy'].train()
            self.agents_neuralnetwork[agent_i]['critic'].train()
    
    def __prep_rollout(self):
        for agent_i in range(self.num_agents):
            self.agents_neuralnetwork[agent_i]['policy'].eval()
            self.agents_neuralnetwork[agent_i]['critic'].eval()

    def _generate_and_store_episodes(self, observations):
        '''Generate multiple episodes here, saving obs, act, rew, pred_values,  '''

        for ep_i in range(self.config.no_episodes): 

            # generate episode and compute pred_values 
            self._generate_episode(observations)
            self._evaluate_value_estimates()

            # compute advantages 
            self.buffer.evaluate_advantages()
            
            # get current buffer 
            buffer = self.buffer.get_state()

            # store buffer in larger buffer 
            self.episode_buffer.store_buffer(buffer, ep_i) # should we allow the same class to do this function, or shall we modularise it? 

            self.buffer.reset()

    def _generate_episode(self, observations:dict) -> tuple:
        '''Generates single episode of (s,a,r ....), updating {obs, actions, action_probs, rewards} keys on single_trajectory_self.buffer
        :param observations: dictionary of all observations. 
        '''
        for t in range(self.config.rollout_length): # we step the environment simultaneously for all traffic signals

            all_actions = {agent_id:None for agent_id in self.agent_ids} # initialise agent actions for every step

            for agent_i, agent_id in enumerate(self.agent_ids):

                agent_obs = utils.get_agent_observation_as_tensor(observations, agent_id)

                action_probs = self._sample_policy_action(agent_i, agent_obs) # Each agent will sample from its own policy
                action = action_probs.argmax()
                
                # append data to all-agent actions self.buffer 
                all_actions[agent_id]= int(action) # update this, as next it will go in the step() func

                # append data to buffers 
                self.buffer.add_data('obs', agent_i, agent_obs.numpy(), t) # new design 
                self.buffer.add_data('actions', agent_i, action, t) # new design 
                self.buffer.add_data('action_probs', agent_i, action_probs.max().detach().numpy(), t) # new design 
            
            # step the environment
            next_observations, rewards, terminations, truncations, infos = self.multi_agent_env.step(all_actions) # takes in a dictionary of all agents + their corresponding actions

            # store info to buffer after env stepped, for every agent 
            for agent_i, agent_id in enumerate(self.agent_ids):
                self.buffer.add_data('rewards', agent_i, rewards[agent_id], t) # new design 

            observations = next_observations

    
    def batch_data(self, agent_data, indices):
        '''batches agent data according to indices, returning mini-batch of data'''
        agent_obs, agent_pred_values, agent_returns, agent_pred_probs, agent_advantages = agent_data

        agent_obs_batch = torch.tensor(agent_obs[indices])                                      # shape = (len(indices), 84)
        agent_pred_probs_batch = torch.tensor(agent_pred_probs[indices], requires_grad=True)
        agent_advantages_batch = torch.tensor(agent_advantages[indices], requires_grad=True)
        agent_returns_batch = torch.tensor(agent_returns[indices], requires_grad=True)
        agent_pred_values_batch = torch.tensor(agent_pred_values[indices], requires_grad=True)

        return agent_obs_batch, agent_pred_probs_batch, agent_returns_batch, agent_pred_values_batch, agent_advantages_batch


    def _minibatch_ppo_agent_update(self, agent_i, obs_minibatch, pred_probs_minibatch, returns_minibatch, pred_values_minibatch, advantages_minibatch):
        '''Expects mini-batches in tensor format, returns in tensor format'''
    
        # initialise old_probs vector
        old_pred_probs_minibatch = torch.zeros(len(obs_minibatch), dtype=float)

        old_policy = self.old_policies[agent_i]
        for i, observations in enumerate(obs_minibatch):
            old_pred_probs_minibatch[i] = old_policy(observations).max()
        
        policy_loss, ratios = self._compute_policy_loss(old_pred_probs_minibatch, pred_probs_minibatch, advantages_minibatch)  
        
        # Backpropagate policy loss
        policy_optimiser = self._get_policy_optimiser(agent_i)
        
        policy_optimiser.zero_grad()
        policy_loss.backward()
        policy_optimiser.step()

        # Calculate value loss
        critic_loss = self._compute_critic_loss(returns_minibatch, pred_values_minibatch)

        # Backpropagate value loss
        value_optimiser = self._get_critic_optimiser(agent_i)
        
        value_optimiser.zero_grad()
        critic_loss.backward()
        value_optimiser.step()

        self.update_count_per_train_iter[agent_i] += 1

        return policy_loss, critic_loss, ratios 
    
    def _evaluate_value_estimates(self):
        for agent_i in range(self.num_agents): 
            self._evaluate_agent_value_estimates(agent_i)
    
    def _evaluate_agent_value_estimates(self, agent_i):
        ''''forward pass on all value_networks saving pred_values to buffer state, assuming observation key is filled up'''   
        for t in range(self.config.rollout_length):
            agent_observations_t = self.buffer.get_data('obs', agent_i, t)
            agent_pred_value = self._generate_value(agent_i, agent_observations_t) # simple forward pass in network to calculate value of state. 
            self.buffer.add_data('pred_values', agent_i, agent_pred_value, t)

    def _compute_policy_loss(self, old_log_probs, new_log_probs, advantages):
        '''Takes sequence of log_probs and advantages in tensors, calculates J(0) which is the prob ratios * advantages'''

        # Calculate the ratio of new and old probabilities
        ratios = torch.exp(new_log_probs - old_log_probs)

        # Calculate policy loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.config.ppo_ratio_clip, 1 + self.config.ppo_ratio_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()  # Negative because we perform gradient ascent

        return policy_loss, ratios
    
    def _compute_critic_loss(self, actual_returns, predicted_values) -> torch.Tensor:

        # Mean squared error loss between predicted and actual returns
        critic_loss = F.mse_loss(predicted_values, actual_returns)

        return critic_loss
  
    def _sample_policy_action(self, agent_i:int, agent_observations:torch.Tensor) -> torch.Tensor :
        policy = self._get_policy(agent_i)
        return policy(agent_observations)

    def _get_policy(self, agent_i:int):
        return self.agents_neuralnetwork[agent_i]['policy']

    def _get_critic(self, agent_i:int):
        return self.agents_neuralnetwork[agent_i]['critic']
    
    def _get_policy_optimiser(self, agent_i:int):
        return self.agents_neuralnetwork[agent_i]['policy_opt']

    def _get_critic_optimiser(self, agent_i:int):
        return self.agents_neuralnetwork[agent_i]['critic_opt']
    
    def _get_policy_old(self, agent_i):
        self.old_policies[agent_i]['policy']

    def _sample_action_old_network(self, agent_i, agent_observations:torch.Tensor) -> torch.Tensor :
        policy = self._get_policy_old(agent_i)
        return policy(agent_observations)
    
    def _clone_policies(self):
        for agent_neural_network_dict in self.agents_neuralnetwork:
            self.old_policies.append(deepcopy(agent_neural_network_dict['policy']))
    
    def _generate_value(self, agent_i, agent_observations:torch.Tensor) -> np.ndarray: 
        critic = self._get_critic(agent_i)
        pred_value = critic(agent_observations) 
        return pred_value.detach().numpy()

    def reset_batch_stats(self):
        self.update_count_per_train_iter = [0] * self.num_agents # update steps for each agent, safely assuming every mini-batch processed is an update. 
        # self.sum_returns = [0] * self.num_agents
        # self.sum_advantages = [0] * self.num_agents
        # self.sum_policy_loss = [0] * self.num_agents
        # self.sum_critic_loss = [0] * self.num_agents
        # self.sum_entropy = [0] * self.num_agents

    def initialise_batch_stats(self):
        self.reset_batch_stats()

    def _display_training_stats(self):
        print(f"Agent update count {self.agent_update_count[0]}")
        print(f"Total updates so far: {self.total_steps}")
    