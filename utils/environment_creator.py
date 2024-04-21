import os 
import pathlib

import sumo_rl

from ray.tune import register_env
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv

from pettingzoo.utils.conversions import aec_to_parallel_wrapper
from pettingzoo.utils import wrappers

import sys
sys.path.append('/Users/loveen/Desktop/Masters project/rl-multi-agent-traffic-nonexhaust-emissions')
from environment.envs import SumoEnvironmentPZCountAllRewards

def init_env_params_2x2(reward_function, observation_function, num_seconds, seed=None, fixed_ts=False, render_mode=None):
        
        env_folder = "data/2x2grid"

        env_params = {"net_file": os.path.abspath(os.path.join(env_folder, "2x2.net.xml")),
                      "route_file": os.path.abspath(os.path.join(env_folder, "2x2.rou.xml")),
                      "reward_fn": reward_function,
                      "observation_class": observation_function,
                      "num_seconds": num_seconds,
                      "add_per_agent_info": True,
                      "add_system_info": True,
                      "sumo_seed": seed,
                      "single_agent": False,
                      "render_mode": render_mode, 
                #     "fixed_ts":fixed_ts
                      }
        
        return env_params

def par_pz_env_2x2_creator(env_params, single_eval_mode=False, multiple_eval_mode=False, csv_path=None, tb_log_dir=None):
        ''''creates a parallel petting zoo environment, by creating a AEC PZ env and then 
        wrapping it to parallel'''
        
        # my own subclass inheriting from SumoEnvironmentPZ (a class that implements PettingZoo API)
        marl_aec_pz_env = SumoEnvironmentPZCountAllRewards(single_eval_mode=single_eval_mode,
                                                           multiple_eval_mode=multiple_eval_mode, 
                                                           csv_path=csv_path, 
                                                           tb_log_dir=tb_log_dir, 
                                                           **env_params)

        # do some processing with petting zoo lib
        marl_aec_pz_env_asserted = wrappers.AssertOutOfBoundsWrapper(marl_aec_pz_env)
        marl_aec_pz_env_order_enfor = wrappers.OrderEnforcingWrapper(marl_aec_pz_env_asserted)
        
        # still with pettingzoo, convert to parallel wrapper 
        par_env_pz = aec_to_parallel_wrapper(marl_aec_pz_env_order_enfor)
        
        return par_env_pz