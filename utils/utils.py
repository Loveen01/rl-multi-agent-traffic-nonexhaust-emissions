import torch 
import numpy as np 

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler



def convert_arr_to_tensor(obs:np.ndarray):
    assert (type(obs) is np.ndarray)
    return torch.from_numpy(obs)

def get_agent_observation_as_tensor(all_agents_obs:dict, agent_id:str):
    '''Takes in entire observations in dict, returning observations for particular agent in tensor'''
    if agent_id not in all_agents_obs.keys():
        raise ValueError(f"Agent {agent_id} not present in observation dict")

    agent_obs = all_agents_obs[agent_id]
    return convert_arr_to_tensor(agent_obs)

def generate_sampler(batch_size, minibatch_size ):
    # episodic_obs (no_agents, no_epi, time_steps, dim)
    subset_sampler = SubsetRandomSampler(range(batch_size)) # random assort integers from 1 - 84, put in list [3, 4, 9, 84, ...]
    # divide this list into batches of size minibatch_size [1, 3, ..], [54, 76, 2..]
    sampler = BatchSampler(subset_sampler, minibatch_size, True) 
    return sampler 