import os 
import pathlib

import sumo_rl
from sumo_rl.environment.env import env, parallel_env, SumoEnvironment

import ray

from ray import air, tune

from ray.tune import register_env
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv
from ray.rllib.algorithms import ppo 
from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from pettingzoo.utils.conversions import aec_to_parallel_wrapper
from pettingzoo.utils import wrappers

import sys
sys.path.append('/Users/loveen/Desktop/Masters project/rl-multi-agent-traffic-nonexhaust-emissions')
from environment.envs import RealMultiAgentSumoEnv
from environment.observation import Grid2x2ObservationFunction, EntireObservationFunction
from environment.reward_functions import combined_reward

from datetime import datetime


def par_env_2x2_creator(seed, eval=False, csv_path=None, tb_log_dir=None):
        
        env_folder = "data/2x2grid"

        env_params = {"net_file": os.path.abspath(os.path.join(env_folder, "2x2.net.xml")),
                "route_file": os.path.abspath(os.path.join(env_folder, "2x2.rou.xml")),
                "reward_fn": combined_reward,
                "observation_class": EntireObservationFunction, 
                # "out_csv_name": "outputs/2x2grid/ppo", 
                "num_seconds": 100000000,
                "add_per_agent_info": True,
                "add_system_info": True,
                "sumo_seed": seed,
                "single_agent": False}
        
        # my own subclass inheriting from SumoEnvironmentPZ (a class that implements PettingZoo API)
        marl_aec_pz_env = RealMultiAgentSumoEnv(**env_params, eval=eval, csv_path=csv_path, tb_log_dir=tb_log_dir) 

        # do some processing with petting zoo lib
        marl_aec_pz_env_asserted = wrappers.AssertOutOfBoundsWrapper(marl_aec_pz_env)
        marl_aec_pz_env_order_enfor = wrappers.OrderEnforcingWrapper(marl_aec_pz_env_asserted)
        
        # still with pettingzoo, convert to parallel wrapper 
        marl_par_env = aec_to_parallel_wrapper(marl_aec_pz_env_order_enfor)
        
        return marl_par_env


def par_env_2x2_creator_v2(seed, eval, csv_path=None, tb_log_dir=None):
        
        env_folder = "data/2x2grid"

        env_params = {"net_file": os.path.abspath(os.path.join(env_folder, "2x2.net.xml")),
                "route_file": os.path.abspath(os.path.join(env_folder, "2x2.rou.xml")),
                "reward_fn": combined_reward,
                "observation_class": EntireObservationFunction, 
                "out_csv_name": "outputs/2x2grid/ppo", 
                "num_seconds": 100000000,
                "add_per_agent_info": True,
                "add_system_info": True,
                "sumo_seed": seed,
                "single_agent": False}
        
        return _par_env_creator(**env_params)
        

def _par_env_creator(**env_params):
        # my own subclass inheriting from SumoEnvironmentPZ (a class that implements PettingZoo API)
        marl_aec_pz_env = RealMultiAgentSumoEnv(**env_params, eval=eval, csv_path=None, tb_log_dir=None) 

        # do some processing with petting zoo lib
        marl_aec_pz_env_asserted = wrappers.AssertOutOfBoundsWrapper(marl_aec_pz_env)
        marl_aec_pz_env_order_enfor = wrappers.OrderEnforcingWrapper(marl_aec_pz_env_asserted)
        
        # still with pettingzoo, convert to parallel wrapper 
        marl_par_env = aec_to_parallel_wrapper(marl_aec_pz_env_order_enfor)
        
        return marl_par_env