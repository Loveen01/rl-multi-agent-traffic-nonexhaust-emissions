import torch 
import numpy as np 

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import json 

import os

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

def extract_folder_from_path(path:str, index:int) -> str:
    '''extracts a folder from a path, given an index'''

    # Split the path into components
    path_parts = path.split(os.sep)

    assert(index < len(path_parts)), f"Index out of range: attempted to access index {index}, but there are only {len(path_parts)} parts."
    # Extract the third folder (remember that indexing starts at 0)
    extracted_folder = path_parts[index]

    return extracted_folder

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if callable(obj):
            return f"{obj.__module__}.{obj.__name__}"
        if isinstance(obj, type):  # For classes
            return f"{obj.__module__}.{obj.__name__}"
        return json.JSONEncoder.default(self, obj)
    
def extract_folder_from_paths(list_of_paths, index) -> list[str]:
    '''extracts the dir_names from from paths here, given an index of position of dir_name on the path'''
    extracted_folder_list = []
    for path in list_of_paths:
        extracted_folder = extract_folder_from_path(path, index)
        extracted_folder_list.append(extracted_folder)
    return extracted_folder_list

