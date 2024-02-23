import torch 
import numpy as np 

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler



def convert_arr_to_tensor(obs:np.ndarray):
    assert (type(obs) is np.ndarray)
    return torch.from_numpy(obs)

def get_agent_observation_as_tensor(all_agents_obs:dict, agent_id:str):
    '''Takes in entire observations in dict, returning observations for particular agent in tensor'''
    if agent_id not in all_agents_obs.keys():
        raise ValueError(f"Agent {agent_id} not present in observation dict {all_agents_obs}")

    agent_obs = all_agents_obs[agent_id]
    return convert_arr_to_tensor(agent_obs)

def generate_sampler(batch_size, minibatch_size ):
    # episodic_obs (no_agents, no_epi, time_steps, dim)
    subset_sampler = SubsetRandomSampler(range(batch_size)) # random assort integers from 1 - 84, put in list [3, 4, 9, 84, ...]
    # divide this list into batches of size minibatch_size [1, 3, ..], [54, 76, 2..]
    sampler = BatchSampler(subset_sampler, minibatch_size, True) 
    return sampler 

# def config_to_text(config):
#     text = f"Config Settings:\n\n"
#     text += f"DEVICE: {config.device}\n"
#     text += f"state_dim: {config.state_dim}\n"
#     text += f"action_dim: {config.action_dim}\n"
#     text += f"lr: {config.lr}\n"
#     text += f"no_hidden_layers: {config.no_hidden_layers}\n"
#     text += f"no_episodes: {config.no_episodes}\n"
#     text += f"rollout_length: {config.rollout_length}\n"
#     text += f"no_training_iterations: {config.no_training_iterations}\n"
#     text += f"optimisation_epochs: {config.optimisation_epochs}\n"
#     text += f"minibatch_size: {config.minibatch_size}\n"
#     text += f"discount: {config.discount}\n"
#     text += f"ppo_ratio_clip: {config.ppo_ratio_clip}\n"
#     text += f"gae_lamda: {config.gae_lamda}\n"
#     # text += f"logger_dir: {config.logger_dir}\n"
#     text += f"play_only: {config.play_only}\n"
#     text += f"saved_checkpoint: {config.saved_checkpoint}\n"
#     return text

def config_to_text(config):
    text = f"Config Settings:\n\n"
    for attr, value in vars(config).items():
        text += f"{attr}: {value}\n"
    return text

def save_configs(config, config_txt_file_path):
    text = f"Config Settings:\n\n"
    for attr, value in vars(config).items():
        text += f"{attr}: {value}\n\n"
    
    with open(config_txt_file_path, 'w') as text_file:
        text_file.write(text)
    
    print(f"Config text file saved to {config_txt_file_path}")
