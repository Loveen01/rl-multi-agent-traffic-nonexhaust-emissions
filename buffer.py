import numpy as np 
from copy import copy

class Buffer:
    '''stateful class to manage individual trajectory buffer, with (keys, agent_i, t, data_dim)'''

    def __init__(self, 
                 NUM_AGENTS, 
                 rollout_length, 
                 state_dim, 
                 discount, 
                 gae_lamda
                 ):
        
        self.NUM_AGENTS = NUM_AGENTS
        self.rollout_length = rollout_length
        self.state_dim = state_dim
        self.discount = discount 
        self.gae_lamda = gae_lamda

        self.buffer = self.initialise_buffer()    
        
    def initialise_buffer(self):
        
        self.buffer = {
            'obs': np.zeros((self.NUM_AGENTS, self.rollout_length, self.state_dim)),
            'actions': np.zeros((self.NUM_AGENTS, self.rollout_length)),
            'rewards': np.zeros((self.NUM_AGENTS, self.rollout_length)),
            'returns': np.zeros((self.NUM_AGENTS, self.rollout_length)),
            'pred_values': np.zeros((self.NUM_AGENTS, self.rollout_length)),
            'advantages': np.zeros((self.NUM_AGENTS, self.rollout_length)),  # returns - pred_values
            'action_probs': np.zeros((self.NUM_AGENTS, self.rollout_length)),
            }
        
        return self.buffer


    def reset(self):
        
        self.buffer = {
            'obs': np.zeros((self.NUM_AGENTS, self.rollout_length, self.state_dim)),
            'actions': np.zeros((self.NUM_AGENTS, self.rollout_length)),
            'rewards': np.zeros((self.NUM_AGENTS, self.rollout_length)),
            'returns': np.zeros((self.NUM_AGENTS, self.rollout_length)),
            'pred_values': np.zeros((self.NUM_AGENTS, self.rollout_length)),
            'advantages': np.zeros((self.NUM_AGENTS, self.rollout_length)),  # returns - pred_values
            'action_probs': np.zeros((self.NUM_AGENTS, self.rollout_length)),
            }
    

    def add_data(self, key:str, agent_i, data, t):
        '''stores data in one '''
        if key not in self.buffer:
            raise KeyError(f"{key} not found in self.buffer")
        
        elif t > self.rollout_length-1:
            raise KeyError(f"Appending data to timestep {t}, buffer already full")
        
        else:
            self.buffer[key][agent_i][t] = data 
    
    def get_data(self, key:str, agent_i, t):
        if key not in self.buffer:
            raise KeyError(f"{key} not found in self.buffer")
        else: 
            return self.buffer[key][agent_i][t]


    def evaluate_gae_advantages(self):
        ''''calculates advantages from buffer, assuming rewards and pred_values is saved to buffer'''
        for agent_i in range(self.NUM_AGENTS): 
            self._evaluate_agent_gae_advantage(agent_i)


    def _evaluate_agent_gae_advantage(self, agent_i):

        for t in reversed(range(self.rollout_length)): 
                
            reward = self.buffer['rewards'][agent_i][t]         # get data from buffer
            pred_value = self.buffer['pred_values'][agent_i][t]

            # calculate TD error: discounted reward - baseline 
            if t < self.rollout_length - 1: 
                pred_value_next = self.buffer['pred_values'][agent_i][t + 1] 
                td_error = reward + (self.discount * pred_value_next) - pred_value

                advantage_next = self.buffer['advantages'][agent_i][t + 1]
                advantage = td_error + self.discount * self.gae_lamda * advantage_next
                
            else:  # handle case when t==self.rollout_length-1
                td_error = reward - pred_value
                advantage = td_error
                
            # append result
            self.buffer['advantages'][agent_i][t] = advantage # advantages = td_error + lamda*gamma*advantage of next step


    def evaluate_advantages(self):
        ''''calculates advantages from buffer, assuming rewards and pred_values is saved to buffer'''
        for agent_i in range(self.NUM_AGENTS): 
            self._evaluate_agent_advantage(agent_i)


    def _evaluate_agent_advantage(self, agent_i):
        
        for t in reversed(range(self.rollout_length)): 
        
            reward = self.buffer['rewards'][agent_i][t]         
            pred_value = self.buffer['pred_values'][agent_i][t]

            advantage =  reward - pred_value
                
            self.buffer['advantages'][agent_i][t] = advantage

        self._normalise_data('advantages', agent_i)


    def _normalise_agent_data(self, key, agent_i):        
        
        data_copy = self.buffer[key][agent_i].copy()

        # data_copy[buffer.active_masks[:-1] == 0.0] = np.nan

        mean_data = np.nanmean(data_copy)
        std_data = np.nanstd(data_copy)
        
        self.buffer[key][agent_i] = (data_copy - mean_data) / (std_data + 1e-5)
    

    def get_state(self):
        return self.buffer


class EpisodeBuffer:
    '''initialises an episode buffer of dimensions (self.NUM_AGENTS, self.NUM_EPISODES, self.ROLLOUT_LENGTH, self.state_dim)
      similar to Buffer, but instead has another dimension, num_episodes'''

    def __init__(self, NUM_AGENTS, NUM_EPISODES, ROLLOUT_LENGTH, state_dim):
        self.NUM_AGENTS = NUM_AGENTS
        self.NUM_EPISODES = NUM_EPISODES
        self.ROLLOUT_LENGTH = ROLLOUT_LENGTH

        self.state_dim = state_dim
 
        self.all_episodes_buffer  = self._initialise_episode_buffer()

    def reset(self): 
        self.all_episodes_buffer = {
        'obs': np.zeros((self.NUM_AGENTS, self.NUM_EPISODES, self.ROLLOUT_LENGTH, self.state_dim)),
        'actions': np.zeros((self.NUM_AGENTS, self.NUM_EPISODES, self.ROLLOUT_LENGTH)),
        'rewards': np.zeros((self.NUM_AGENTS, self.NUM_EPISODES, self.ROLLOUT_LENGTH)),
        'returns': np.zeros((self.NUM_AGENTS, self.NUM_EPISODES, self.ROLLOUT_LENGTH)),  # note that this is the bootstrapped returns 
        'pred_values': np.zeros((self.NUM_AGENTS, self.NUM_EPISODES, self.ROLLOUT_LENGTH)),
        'advantages': np.zeros((self.NUM_AGENTS, self.NUM_EPISODES, self.ROLLOUT_LENGTH)),  # returns - pred_values
        'action_probs': np.zeros((self.NUM_AGENTS, self.NUM_EPISODES, self.ROLLOUT_LENGTH)),
        }

    def _initialise_episode_buffer(self):
        self.all_episodes_buffer = {
        'obs': np.zeros((self.NUM_AGENTS, self.NUM_EPISODES, self.ROLLOUT_LENGTH, self.state_dim)),
        'actions': np.zeros((self.NUM_AGENTS, self.NUM_EPISODES, self.ROLLOUT_LENGTH)),
        'rewards': np.zeros((self.NUM_AGENTS, self.NUM_EPISODES, self.ROLLOUT_LENGTH)),
        'returns': np.zeros((self.NUM_AGENTS, self.NUM_EPISODES, self.ROLLOUT_LENGTH)),  # note that this is the bootstrapped returns 
        'pred_values': np.zeros((self.NUM_AGENTS, self.NUM_EPISODES, self.ROLLOUT_LENGTH)),
        'advantages': np.zeros((self.NUM_AGENTS, self.NUM_EPISODES, self.ROLLOUT_LENGTH)),  # returns - pred_values
        'action_probs': np.zeros((self.NUM_AGENTS, self.NUM_EPISODES, self.ROLLOUT_LENGTH)),
        }
        return self.all_episodes_buffer 


    def flatten_data(self, agent_i): 
        '''Slides data for agent, then flattens it across (self.NUM_EPISODES, self.ROLLOUT_LENGTH)'''
        
        agent_obs = self.all_episodes_buffer['obs'][agent_i].reshape(-1,self.state_dim)

        agent_returns = self.all_episodes_buffer['returns'][agent_i].reshape(-1) # (no_epi * time_steps) 1dim
        agent_pred_values = self.all_episodes_buffer['pred_values'][agent_i].reshape(-1)

        agent_pred_probs = self.all_episodes_buffer['action_probs'][agent_i].reshape(-1) # (no_epi, time_steps, dim) -> (no_epi*timesteps, dim)
        agent_advantages = self.all_episodes_buffer['advantages'][agent_i].reshape(-1)

        return (agent_obs, agent_pred_values, agent_returns, agent_pred_probs, agent_advantages)


    def store_buffer(self, buffer: dict, episode_idx):
        '''Store the trajectory data into the episodic data for all agents'''
        for key in buffer.keys():
            if key not in self.all_episodes_buffer.keys():
                raise KeyError(f"Key '{key}' in buffer not found in episodic buffer")
            for agent_i in range(self.NUM_AGENTS):
                self.all_episodes_buffer[key][agent_i][episode_idx] = buffer[key][agent_i]

    def get_state(self):
        return self.all_episodes_buffer
        